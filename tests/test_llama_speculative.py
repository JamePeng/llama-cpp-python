import builtins

import numpy as np
import pytest

import llama_cpp
from llama_cpp import llama_cpp as llama_cpp_lib

from llama_cpp.llama_speculative import (
    LlamaMTPDecoding,
    LlamaNGramMapDecoding,
    LlamaSpecEngine,
    SpecConfig,
    SpeculativeType,
    speculative_output_limits,
)


def test_spec_engine_is_the_public_base_class():
    assert llama_cpp.LlamaSpecEngine is LlamaSpecEngine
    assert issubclass(LlamaNGramMapDecoding, LlamaSpecEngine)
    assert issubclass(LlamaMTPDecoding, LlamaSpecEngine)


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


class _FakeCheckpointContext:
    def __init__(self, *, position=7, state_size=8):
        self.position = position
        self.state_size = state_size
        self.removals = []
        self.get_flags = []
        self.set_flags = []

    def memory_seq_pos_max(self, seq_id):
        assert seq_id == 0
        return self.position

    def memory_seq_rm(self, seq_id, p0, p1):
        self.removals.append((seq_id, p0, p1))
        return True

    def get_state_seq_size_ext(self, seq_id, flags):
        assert seq_id == 0
        self.get_flags.append(flags)
        return self.state_size

    def get_state_seq_data_ext(self, buffer, size, seq_id, flags):
        assert size == self.state_size
        assert seq_id == 0
        assert flags == self.get_flags[-1]
        return size

    def set_state_seq_data_ext(self, buffer, size, seq_id, flags):
        assert buffer is not None
        assert size == self.state_size
        assert seq_id == 0
        self.set_flags.append(flags)
        return size


def _checkpoint_test_engine(*, native):
    engine = object.__new__(LlamaMTPDecoding)
    engine.draft_context = _FakeCheckpointContext()
    engine.n_embd = 2
    engine.pending_h = np.asarray([1.0, 2.0], dtype=np.float32)
    engine.verify_h = np.empty((0, 2), dtype=np.float32)
    engine.verify_tokens = []
    engine.verify_positions = []
    engine._use_native_draft_rollback = native
    engine._pending_verification_checkpoint = None
    engine.reset_checkpoint_stats()
    return engine


def test_mtp_native_checkpoint_is_reused_for_verification():
    engine = _checkpoint_test_engine(native=True)
    checkpoint = engine.checkpoint()
    engine.pending_h.fill(0.0)
    engine.restore(checkpoint)
    engine._pending_verification_checkpoint = checkpoint

    reused = engine.take_verification_checkpoint()
    stats = engine.checkpoint_stats()

    assert reused is checkpoint
    assert engine.draft_context.removals == [(0, 8, -1)]
    np.testing.assert_array_equal(engine.pending_h, [1.0, 2.0])
    assert stats["captures"] == 1
    assert stats["restores"] == 1
    assert stats["verification_reuses"] == 1
    assert stats["native_captures"] == 1
    assert stats["device_captures"] == 0


def test_mtp_checkpoint_fallback_keeps_state_on_device():
    engine = _checkpoint_test_engine(native=False)
    checkpoint = engine.checkpoint()
    engine.restore(checkpoint)
    stats = engine.checkpoint_stats()

    expected_flags = (
        llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
        | llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
    )
    assert checkpoint["mode"] == "on-device"
    assert engine.draft_context.get_flags == [expected_flags]
    assert engine.draft_context.set_flags == [expected_flags]
    assert stats["device_captures"] == 1
    assert stats["device_restores"] == 1
    assert stats["buffer_bytes"] == engine.draft_context.state_size


def test_mtp_native_verification_rollback_removes_only_rejected_suffix():
    engine = _checkpoint_test_engine(native=True)
    engine.is_mem_shared = False
    engine.verify_tokens = [10, 11, 12]
    engine.verify_positions = [8, 9, 10]
    engine.verify_h = np.asarray(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
    )

    engine.rollback_verified(engine.checkpoint(), n_accepted=1)

    assert engine.draft_context.removals == [(0, 10, -1)]
    np.testing.assert_array_equal(engine.pending_h, [2.0, 2.0])
    stats = engine.checkpoint_stats()
    assert stats["restores"] == 0
    assert stats["native_verification_rollbacks"] == 1


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

        def n_rs_seq(self):
            return 4

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
    engine._use_native_draft_rollback = True
    engine.target_context = _Context(512)
    engine.draft_context = _Context(512)

    engine._print_runtime_configuration(_TargetParams())
    output = capsys.readouterr().err

    assert "draft-mtp" in output
    assert "draft_n_min=1, draft_n_max=4" in output
    assert "draft_p_min=0.2" in output
    assert "backend_sampling=requested:True/active:True" in output
    assert "mtp_heads=2" in output
    assert "target_n_rs_seq=4, draft_n_rs_seq=4" in output
    assert "draft_checkpoint=native-rs" in output
    assert "outputs=5/5" in output
