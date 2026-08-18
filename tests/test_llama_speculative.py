import builtins

import pytest

from llama_cpp.llama_speculative import (
    LlamaMTPDecoding,
    LlamaNGramMapDecoding,
    SpecConfig,
    SpeculativeType,
    speculative_output_limits,
)


def test_ngram_map_lifecycle_and_acceptance_feedback():
    decoder = LlamaNGramMapDecoding(
        ngram_size=3,
        num_pred_tokens=3,
        spec_type=SpeculativeType.NGRAM_MAP_K,
    )
    history = [1, 2, 3, 7, 8, 1, 2, 3]
    decoder.begin(history)

    draft = decoder.draft(history, n_past=len(history), id_last=3, n_max=2)
    assert draft.tolist() == [7, 8]

    decoder.accept(1)
    draft = decoder.draft(history, n_past=len(history), id_last=3, n_max=3)
    assert draft.tolist() == [7]


def test_spec_config_requires_draft_model_for_external_architectures():
    for spec_type in (
        SpeculativeType.DRAFT_EAGLE3,
        SpeculativeType.DRAFT_DFLASH,
        SpeculativeType.DRAFT_DSPARK,
    ):
        with pytest.raises(ValueError, match="draft_model_path"):
            SpecConfig(spec_type=spec_type).validate()


def test_mtp_allows_target_internal_heads():
    SpecConfig(spec_type=SpeculativeType.DRAFT_MTP).validate()


def test_speculative_output_limits_match_llama_cpp():
    assert speculative_output_limits(32, 1, 3) == (4, 4)
    assert speculative_output_limits(32, 4, 3) == (16, 4)
    assert speculative_output_limits(3, 4, 8) == (3, 3)


def test_spec_config_selects_algorithm_specific_draft_limit():
    assert SpecConfig(
        spec_type=SpeculativeType.DRAFT_MTP,
        draft_n_max=5,
    ).max_draft_tokens() == 5
    assert SpecConfig(
        spec_type=SpeculativeType.NGRAM_MAP_K,
        ngram_size_m=48,
    ).max_draft_tokens() == 48
    assert SpecConfig(
        spec_type=SpeculativeType.NGRAM_MOD,
        ngram_mod_n_max=64,
    ).max_draft_tokens() == 64


def test_spec_config_validates_draft_runtime_arguments():
    with pytest.raises(ValueError, match="draft_n_threads"):
        SpecConfig(
            spec_type=SpeculativeType.DRAFT_MTP,
            draft_n_threads=0,
        ).validate()
    with pytest.raises(ValueError, match="draft_n_cpu_moe"):
        SpecConfig(
            spec_type=SpeculativeType.DRAFT_MTP,
            draft_n_cpu_moe=-1,
        ).validate()


def test_spec_config_parses_arg_cpp_gpu_layer_spellings():
    assert SpecConfig(draft_n_gpu_layers="auto").resolved_draft_n_gpu_layers() == -1
    assert SpecConfig(draft_n_gpu_layers="all").resolved_draft_n_gpu_layers() == -2
    assert SpecConfig(draft_n_gpu_layers=12).resolved_draft_n_gpu_layers() == 12


class _FakeVocabModel:
    def __init__(self, tokens, *, vocab_type=1, add_bos=True, add_eos=False):
        self.tokens = tokens
        self._vocab_type = vocab_type
        self._add_bos = add_bos
        self._add_eos = add_eos

    def vocab_type(self):
        return self._vocab_type

    def get_add_bos(self):
        return self._add_bos

    def get_add_eos(self):
        return self._add_eos

    def token_bos(self):
        return 1

    def token_eos(self):
        return 2

    def n_vocab(self):
        return len(self.tokens)

    def token_get_text(self, token):
        return self.tokens[token]


def test_mtp_vocab_compatibility_rejects_token_mismatch():
    target = _FakeVocabModel([str(i) for i in range(8)])
    draft = _FakeVocabModel([str(i) for i in range(8)])
    draft.tokens[6] = "different"

    with pytest.raises(ValueError, match="token 6"):
        LlamaMTPDecoding._validate_vocab_compatibility(target, draft)


def test_mtp_close_does_not_import_during_interpreter_shutdown():
    class _NativeAPI:
        def __init__(self):
            self.detached = False

        def llama_set_sampler(self, context, seq_id, sampler):
            assert context == "draft-context"
            assert seq_id == 0
            assert sampler is None
            self.detached = True

    class _Closable:
        def __init__(self, *, context=None):
            self.ctx = context
            self.closed = False

        def close(self):
            self.closed = True

    engine = object.__new__(LlamaMTPDecoding)
    engine._closed = False
    engine._llama_cpp_lib = _NativeAPI()
    engine._backend_sampler = _Closable()
    engine._backend_sampling = True
    engine.batch = _Closable()
    engine.draft_context = _Closable(context="draft-context")
    engine._owns_model = False
    engine.draft_model = object()

    backend_sampler = engine._backend_sampler
    batch = engine.batch
    draft_context = engine.draft_context
    original_import = builtins.__import__

    def fail_import(*args, **kwargs):
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    builtins.__import__ = fail_import
    try:
        engine.close()
    finally:
        builtins.__import__ = original_import

    assert engine._llama_cpp_lib.detached
    assert backend_sampler.closed
    assert batch.closed
    assert draft_context.closed
    assert engine._backend_sampler is None
    assert engine.draft_context is None


def test_mtp_runtime_configuration_reports_requested_and_resolved_values(capsys):
    class _Context:
        def __init__(self, n_batch):
            self._n_batch = n_batch

        def n_batch(self):
            return self._n_batch

    class _TargetParams:
        n_rs_seq = 4
        n_outputs_max = 5
        n_outputs_max_per_seq = 5

    engine = object.__new__(LlamaMTPDecoding)
    engine.config = SpecConfig(
        spec_type=SpeculativeType.DRAFT_MTP,
        draft_n_min=1,
        draft_n_max=4,
        draft_p_min=0.2,
        draft_p_split=0.15,
        draft_backend_sampling=True,
        draft_n_threads=6,
        draft_n_threads_batch=8,
    )
    engine._owns_model = False
    engine._backend_sampling = True
    engine.n_mtp_layers = 2
    engine.is_mem_shared = True
    engine.chain_heads = False
    engine.target_context = _Context(512)
    engine.draft_context = _Context(512)

    engine._print_runtime_configuration(_TargetParams())
    output = capsys.readouterr().err

    assert "draft-mtp" in output
    assert "draft_n_min=1, draft_n_max=4" in output
    assert "draft_p_min=0.2" in output
    assert "backend_sampling=requested:True/active:True" in output
    assert "mtp_heads=2" in output
    assert "n_rs_seq=4" in output
    assert "outputs=5/5" in output
