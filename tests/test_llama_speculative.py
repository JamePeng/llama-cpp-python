import builtins
import ctypes

import numpy as np
import pytest

import llama_cpp
from llama_cpp import llama_cpp as llama_cpp_lib
from llama_cpp import _internals as internals

from llama_cpp.llama_speculative import (
    LlamaDFlashDecoding,
    LlamaMTPDecoding,
    LlamaNGramMapDecoding,
    LlamaSpecEngine,
    SpecConfig,
    SpeculativeType,
    create_native_spec_engine,
    speculative_output_limits,
)


def test_spec_engine_is_the_public_base_class():
    assert llama_cpp.LlamaSpecEngine is LlamaSpecEngine
    assert issubclass(LlamaNGramMapDecoding, LlamaSpecEngine)
    assert issubclass(LlamaMTPDecoding, LlamaSpecEngine)
    assert issubclass(LlamaDFlashDecoding, LlamaSpecEngine)
    assert LlamaMTPDecoding.__bases__ == LlamaDFlashDecoding.__bases__
    assert LlamaMTPDecoding._copy_rows is LlamaDFlashDecoding._copy_rows
    assert LlamaMTPDecoding._candidate is LlamaDFlashDecoding._candidate
    assert "_copy_rows" not in LlamaDFlashDecoding.__dict__
    assert "_candidate" not in LlamaDFlashDecoding.__dict__


def test_llama_context_wraps_backend_sampling_and_perf(monkeypatch):
    context = object.__new__(internals.LlamaContext)
    context.ctx = "native-context"
    context._sampler_refs = {}
    sampler = type("Sampler", (), {"sampler": "native-sampler"})()
    calls = []

    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_set_sampler",
        lambda ctx, seq_id, ptr: calls.append(("set", ctx, seq_id, ptr)) or True,
    )
    assert context.set_sampler(2, sampler)
    assert context._sampler_refs[2] is sampler
    assert context.set_sampler(2, None)
    assert 2 not in context._sampler_refs

    pointer_results = {
        "llama_get_sampled_probs_ith": object(),
        "llama_get_sampled_logits_ith": object(),
        "llama_get_sampled_candidates_ith": object(),
    }
    scalar_results = {
        "llama_get_sampled_token_ith": 17,
        "llama_get_sampled_probs_count_ith": 3,
        "llama_get_sampled_logits_count_ith": 4,
        "llama_get_sampled_candidates_count_ith": 5,
    }
    for name, result in {**pointer_results, **scalar_results}.items():
        monkeypatch.setattr(
            internals.llama_cpp,
            name,
            lambda ctx, index, result=result, name=name: (
                calls.append((name, ctx, index)) or result
            ),
        )

    assert context.get_sampled_token_ith(-1) == 17
    assert context.get_sampled_probs_count_ith(-1) == 3
    assert context.get_sampled_logits_count_ith(-1) == 4
    assert context.get_sampled_candidates_count_ith(-1) == 5
    assert (
        context.get_sampled_probs_ith(-1)
        is pointer_results["llama_get_sampled_probs_ith"]
    )
    assert (
        context.get_sampled_logits_ith(-1)
        is pointer_results["llama_get_sampled_logits_ith"]
    )
    assert (
        context.get_sampled_candidates_ith(-1)
        is pointer_results["llama_get_sampled_candidates_ith"]
    )

    perf = llama_cpp_lib.llama_perf_context_data()
    perf.n_eval = 9
    monkeypatch.setattr(
        internals.llama_cpp, "llama_perf_context", lambda ctx: perf
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_perf_context_print",
        lambda ctx: calls.append(("perf-print", ctx)),
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_perf_context_reset",
        lambda ctx: calls.append(("perf-reset", ctx)),
    )

    assert context.perf_context().n_eval == 9
    context.print_timings()
    context.reset_timings()
    assert ("perf-print", "native-context") in calls
    assert ("perf-reset", "native-context") in calls
    context.ctx = None


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


def test_ngram_map_k4v_uses_vendor_continuation_limit_by_default():
    decoder = LlamaNGramMapDecoding(
        ngram_size=3,
        num_pred_tokens=8,
        spec_type=SpeculativeType.NGRAM_MAP_K4V,
    )

    assert decoder.max_entries_per_key == 4


def test_spec_config_requires_draft_model_for_external_architectures():
    for spec_type in (
        SpeculativeType.DRAFT_EAGLE3,
        SpeculativeType.DRAFT_DFLASH,
        SpeculativeType.DRAFT_DSPARK,
    ):
        with pytest.raises(ValueError, match="draft_model_path"):
            SpecConfig(spec_type=spec_type).validate()


@pytest.mark.parametrize(
    "spec_type",
    [SpeculativeType.DRAFT_DFLASH, SpeculativeType.DRAFT_DSPARK],
)
def test_native_factory_routes_dflash_family_to_shared_engine(
    monkeypatch, spec_type
):
    created = []

    def fake_engine(config, **kwargs):
        created.append((config, kwargs))
        return "dflash-engine"

    monkeypatch.setattr(
        "llama_cpp.llama_speculative.LlamaDFlashDecoding", fake_engine
    )
    config = SpecConfig(
        spec_type=spec_type,
        draft_model_path="draft.gguf",
    )
    result = create_native_spec_engine(
        config,
        target_model="target-model",
        target_context="target-context",
        model_params="model-params",
        context_params="context-params",
        verbose=False,
    )

    assert result == "dflash-engine"
    assert created[0][0] is config
    assert created[0][1]["target_model"] == "target-model"


def test_mtp_allows_target_internal_heads():
    SpecConfig(spec_type=SpeculativeType.DRAFT_MTP).validate()


def test_speculative_output_limits_match_llama_cpp():
    assert speculative_output_limits(32, 1, 3) == (4, 4)
    assert speculative_output_limits(32, 4, 3) == (16, 4)
    assert speculative_output_limits(3, 4, 8) == (3, 3)


def test_draft_context_does_not_inherit_target_embedding_mode():
    target = llama_cpp_lib.llama_context_default_params()
    target.embeddings = True
    target.pooling_type = llama_cpp_lib.LLAMA_POOLING_TYPE_MEAN

    draft = LlamaMTPDecoding._copy_draft_context_params(
        target, llama_cpp_lib.LLAMA_POOLING_TYPE_UNSPECIFIED
    )

    assert target.embeddings is True
    assert target.pooling_type == llama_cpp_lib.LLAMA_POOLING_TYPE_MEAN
    assert draft.embeddings is False
    assert draft.pooling_type == llama_cpp_lib.LLAMA_POOLING_TYPE_UNSPECIFIED


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


def test_speculative_vocab_compatibility_rejects_token_mismatch():
    target = _FakeVocabModel([str(i) for i in range(8)])
    draft = _FakeVocabModel([str(i) for i in range(8)])
    draft.tokens[6] = "different"

    with pytest.raises(ValueError, match="token 6"):
        LlamaMTPDecoding._validate_speculative_vocab_compatibility(
            target, draft
        )


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
    class _Closable:
        def __init__(self, *, context=None):
            self.ctx = context
            self.closed = False
            self.sampler_changes = []

        def set_sampler(self, seq_id, sampler):
            assert self.ctx == "draft-context"
            self.sampler_changes.append((seq_id, sampler))
            return True

        def close(self):
            self.closed = True

    engine = object.__new__(LlamaMTPDecoding)
    engine._closed = False
    engine._llama_cpp_lib = object()
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

    assert draft_context.sampler_changes == [(0, None)]
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


class _FakeEvalBatch:
    def __init__(self):
        self.batch = self
        self.n_tokens = 0


class _FakeEvalContext:
    def __init__(self, statuses, events):
        self.statuses = list(statuses)
        self.events = events
        self.decode_sizes = []

    def decode(self, batch):
        self.decode_sizes.append(batch.batch.n_tokens)
        self.events.append("target-decode")
        return self.statuses.pop(0)

    def synchronize(self):
        self.events.append("target-sync")


class _FakeSpecEngine:
    def __init__(self, events):
        self.events = events

    def process(self, batch, seq_id=0):
        assert batch is not None
        assert seq_id == 0
        self.events.append("spec-process")


def _eval_test_llama(*, statuses, verifying, speculative=True):
    events = []
    llm = object.__new__(llama_cpp.Llama)
    llm._batch = _FakeEvalBatch()
    llm._ctx = _FakeEvalContext(statuses, events)
    llm._speculative_verifying = verifying
    llm.speculative = _FakeSpecEngine(events) if speculative else None
    llm._active_speculative_phase_stats = {
        "target_decode_seconds": 0.0,
        "target_sync_seconds": 0.0,
        "process_calls": 0,
        "process_seconds": 0.0,
    }
    llm.n_tokens = 0
    llm.verbose = False
    return llm, events


def test_speculative_target_sync_precedes_process_and_has_separate_timing():
    llm, events = _eval_test_llama(statuses=[0], verifying=True)

    assert llm._decode_eval_batch([1, 2, 3], 3) == 3
    llm._process_speculative_batch()

    assert events == ["target-decode", "target-sync", "spec-process"]
    stats = llm._active_speculative_phase_stats
    assert stats["target_decode_seconds"] >= 0.0
    assert stats["target_sync_seconds"] >= 0.0
    assert stats["process_seconds"] >= 0.0
    assert stats["process_calls"] == 1


def test_speculative_verification_batch_is_not_dynamically_split():
    llm, _ = _eval_test_llama(statuses=[1, 0], verifying=True)

    with pytest.raises(RuntimeError, match="verification batch cannot be split"):
        llm._decode_eval_batch([1, 2, 3, 4], 4)

    assert llm._ctx.decode_sizes == [4]


def test_ordinary_eval_batch_can_still_retry_at_a_smaller_size():
    llm, _ = _eval_test_llama(
        statuses=[1, 0], verifying=False, speculative=False
    )

    assert llm._decode_eval_batch([1, 2, 3, 4], 4) == 2
    assert llm._ctx.decode_sizes == [4, 2]


def test_speculative_verification_batch_must_fit_n_batch():
    llm = object.__new__(llama_cpp.Llama)
    llm._speculative_verifying = True
    llm.n_batch = 3

    with pytest.raises(RuntimeError, match="verification batch exceeds n_batch"):
        llm.eval([1, 2, 3, 4], copy_logits=False)


def test_llama_memory_removal_failure_is_fatal():
    class _Context:
        def memory_seq_rm(self, seq_id, p0, p1):
            return False

    llm = object.__new__(llama_cpp.Llama)
    llm._ctx = _Context()

    with pytest.raises(RuntimeError, match="test rollback"):
        llm._memory_seq_rm_or_raise(0, 12, -1, "test rollback")


def test_interrupted_hybrid_speculation_uses_native_rollback():
    class _Context:
        def __init__(self):
            self.removals = []

        def memory_seq_rm(self, seq_id, p0, p1):
            self.removals.append((seq_id, p0, p1))
            return True

    class _Speculative:
        def __init__(self):
            self.rollbacks = []

        def rollback_verified(self, checkpoint, n_accepted, seq_id=0):
            self.rollbacks.append((checkpoint, n_accepted, seq_id))

    llm = object.__new__(llama_cpp.Llama)
    llm._ctx = _Context()
    llm.speculative = _Speculative()
    llm.is_hybrid = True
    llm.n_tokens = 9
    llm._last_eval_output_start = 6
    llm._last_eval_output_count = 3

    mode = llm._recover_interrupted_speculation(
        verification_start=6,
        evaluated_tokens=[10, 11, 12],
        delivered_accepted=1,
        speculative_checkpoint="draft-checkpoint",
        use_native_rollback=True,
        active_loras=None,
        control_vector=None,
    )

    assert mode == "native"
    assert llm._ctx.removals == [(0, 8, -1)]
    assert llm.speculative.rollbacks == [("draft-checkpoint", 1, 0)]
    assert llm.n_tokens == 8
    assert llm._last_eval_output_start == 6
    assert llm._last_eval_output_count == 2


def test_interrupted_hybrid_speculation_restores_checkpoint_and_replays():
    class _Checkpoint:
        pos = 6

    class _Cache:
        def __init__(self):
            self.checkpoint = _Checkpoint()
            self.lookups = []
            self.restores = []

        def find_best_checkpoint(self, tokens, seq_id=0):
            self.lookups.append((tokens, seq_id))
            return self.checkpoint

        def restore_checkpoint(self, checkpoint, seq_id=0):
            self.restores.append((checkpoint, seq_id))
            return True

    class _Speculative:
        def __init__(self):
            self.restores = []

        def restore(self, checkpoint, seq_id=0):
            self.restores.append((checkpoint, seq_id))

    llm = object.__new__(llama_cpp.Llama)
    llm.speculative = _Speculative()
    llm._hybrid_cache_mgr = _Cache()
    llm.is_hybrid = True
    llm.input_ids = np.asarray([1, 2, 3, 4, 5, 6, 10, 11, 12], dtype=np.intc)
    llm.n_tokens = 9
    replayed = []

    def eval_tokens(tokens, **kwargs):
        replayed.append((list(tokens), kwargs))
        llm.n_tokens += len(tokens)

    llm.eval = eval_tokens
    active_loras = [{"name": "test", "scale": 0.5}]
    control_vector = {"data": [1.0]}

    mode = llm._recover_interrupted_speculation(
        verification_start=6,
        evaluated_tokens=[10, 11, 12],
        delivered_accepted=1,
        speculative_checkpoint="draft-checkpoint",
        use_native_rollback=False,
        active_loras=active_loras,
        control_vector=control_vector,
    )

    assert mode == "checkpoint"
    assert llm._hybrid_cache_mgr.lookups == [([1, 2, 3, 4, 5, 6], 0)]
    assert llm._hybrid_cache_mgr.restores == [
        (llm._hybrid_cache_mgr.checkpoint, 0)
    ]
    assert llm.speculative.restores == [("draft-checkpoint", 0)]
    assert replayed == [
        (
            [10, 11],
            {
                "active_loras": active_loras,
                "control_vector": control_vector,
                "copy_logits": False,
            },
        )
    ]
    assert llm.n_tokens == 8


def test_interrupted_non_hybrid_speculation_truncates_both_contexts():
    class _Context:
        def __init__(self):
            self.removals = []

        def memory_seq_rm(self, seq_id, p0, p1):
            self.removals.append((seq_id, p0, p1))
            return True

    class _Speculative:
        def __init__(self):
            self.truncations = []

        def truncate(self, position, seq_id=0):
            self.truncations.append((position, seq_id))

    llm = object.__new__(llama_cpp.Llama)
    llm._ctx = _Context()
    llm.speculative = _Speculative()
    llm.is_hybrid = False
    llm.n_tokens = 9
    llm._last_eval_output_start = 6
    llm._last_eval_output_count = 3

    mode = llm._recover_interrupted_speculation(
        verification_start=6,
        evaluated_tokens=[10, 11, 12],
        delivered_accepted=0,
        speculative_checkpoint=None,
        use_native_rollback=False,
        active_loras=None,
        control_vector=None,
    )

    assert mode == "truncate"
    assert llm._ctx.removals == [(0, 7, -1)]
    assert llm.speculative.truncations == [(7, 0)]
    assert llm.n_tokens == 7
    assert llm._last_eval_output_count == 1


def test_mtp_truncate_fails_when_native_memory_removal_fails():
    class _Model:
        def is_recurrent(self):
            return False

        def is_hybrid(self):
            return False

    class _Context:
        def memory_seq_rm(self, seq_id, p0, p1):
            return False

    engine = object.__new__(LlamaMTPDecoding)
    engine.draft_model = _Model()
    engine.draft_context = _Context()

    with pytest.raises(RuntimeError, match="draft-context truncation failed"):
        engine.truncate(12)


class _FakeDFlashBatch:
    def __init__(self):
        self.tokens = []
        self.positions = []
        self.embeddings = None
        self.logits = []
        self.mrope_positions = None

    def reset(self):
        self.tokens = []
        self.positions = []
        self.embeddings = None
        self.logits = []
        self.mrope_positions = None

    def add_sequence(
        self, token_array, pos_array, seq_ids, logits_array
    ):
        assert seq_ids == [0]
        assert len(token_array) == len(pos_array) == len(logits_array)
        self.tokens = list(token_array)
        self.positions = list(pos_array)
        self.logits = list(logits_array)

    def add_embeddings(
        self, embeddings, *, pos_array, seq_ids, logits_array=None
    ):
        assert seq_ids == [0]
        self.embeddings = np.asarray(embeddings, dtype=np.float32).copy()
        self.positions = list(pos_array)
        self.logits = list(logits_array or [])

    def add_embeddings_mrope(
        self, embeddings, *, pos_array, seq_ids, logits_array=None
    ):
        self.add_embeddings(
            embeddings,
            pos_array=pos_array[0],
            seq_ids=seq_ids,
            logits_array=logits_array,
        )
        self.mrope_positions = [list(plane) for plane in pos_array]


class _FakeDFlashDraftContext:
    def __init__(self, *, fused=None, confidence=None):
        self.fused = fused
        self.confidence = confidence
        self.decoded = []
        self.encoded = []
        self.synchronized = False
        self.removals = []

    def n_ubatch(self):
        return 8

    def encode(self, batch):
        self.encoded.append(batch.embeddings.copy())

    def decode(self, batch):
        self.decoded.append(
            (list(batch.tokens), list(batch.positions), batch.embeddings)
        )
        return 0

    def get_embeddings_nextn(self):
        values = self.confidence if self.confidence is not None else self.fused
        return values.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

    def synchronize(self):
        self.synchronized = True

    def memory_seq_rm(self, seq_id, p0, p1):
        self.removals.append((seq_id, p0, p1))
        return True


def _draft_test_dflash_engine(*, dspark, sample_from_anchor, p_min=0.0):
    engine = object.__new__(LlamaDFlashDecoding)
    engine.config = SpecConfig(
        spec_type=(
            SpeculativeType.DRAFT_DSPARK
            if dspark
            else SpeculativeType.DRAFT_DFLASH
        ),
        draft_model_path="draft.gguf",
        draft_n_max=3,
        draft_p_min=p_min,
    )
    engine.is_dspark = dspark
    engine.is_dflash2 = False
    engine.selector_top_k = 0
    engine.sample_from_anchor = sample_from_anchor
    engine.draft_limit = 3
    engine.mask_token_id = 99
    engine.n_embd_dec = 2
    engine.noise_batch = _FakeDFlashBatch()
    engine._pending_verification_checkpoint = None
    engine._candidate = lambda output_index: (100 + output_index, 0.9)
    engine.checkpoint = lambda seq_id=0: {"position": 5}
    engine.restore = lambda checkpoint, seq_id=0: None
    return engine


def test_dflash_builds_anchor_plus_mask_block_and_reads_mask_outputs():
    engine = _draft_test_dflash_engine(
        dspark=False, sample_from_anchor=True, p_min=0.1
    )
    engine.draft_context = _FakeDFlashDraftContext()

    result = engine.draft([], n_past=7, id_last=42, n_max=3)

    assert engine.noise_batch.tokens == [42, 99, 99, 99]
    assert engine.noise_batch.positions == [7, 8, 9, 10]
    assert engine.noise_batch.logits == [True, True, True, True]
    assert result.tolist() == [101, 102, 103]
    assert engine._pending_verification_checkpoint == {"position": 5}


def test_dspark_anchor_layout_uses_confidence_to_truncate_block():
    engine = _draft_test_dflash_engine(
        dspark=True, sample_from_anchor=True, p_min=0.5
    )
    confidence = np.asarray(
        [[0.9, 0.9], [0.4, 0.4], [0.8, 0.8]], dtype=np.float32
    )
    engine.draft_context = _FakeDFlashDraftContext(confidence=confidence)

    result = engine.draft([], n_past=7, id_last=42, n_max=3)

    assert engine.noise_batch.tokens == [42, 99, 99]
    assert result.tolist() == [100]


def test_dflash2_walks_selector_lattice_without_requesting_logits():
    engine = _draft_test_dflash_engine(
        dspark=False, sample_from_anchor=True, p_min=0.0
    )
    engine.is_dflash2 = True
    engine.selector_top_k = 2
    engine.n_embd_dec = 6
    lattice = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [101, 102, 0, 2, 0, 2],
            [201, 202, 0, 0, 3, 1],
            [301, 302, 0, 4, 0, 0],
        ],
        dtype=np.float32,
    )
    engine.draft_context = _FakeDFlashDraftContext(confidence=lattice)

    result = engine.draft([], n_past=7, id_last=42, n_max=3)

    assert engine.noise_batch.tokens == [42, 99, 99, 99]
    assert engine.noise_batch.logits == [False, False, False, False]
    assert result.tolist() == [102, 201, 302]
    assert engine._pending_verification_checkpoint == {"position": 5}


def test_dflash2_selector_probability_truncates_before_low_confidence_token():
    engine = _draft_test_dflash_engine(
        dspark=False, sample_from_anchor=True, p_min=0.8
    )
    engine.is_dflash2 = True
    engine.selector_top_k = 2
    engine.n_embd_dec = 6
    lattice = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [101, 102, 0, 3, 0, 3],
            [201, 202, 0, 0, 0, 1],
            [301, 302, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    engine.draft_context = _FakeDFlashDraftContext(confidence=lattice)

    result = engine.draft([], n_past=7, id_last=42, n_max=3)

    assert result.tolist() == [102]


@pytest.mark.parametrize(
    ("is_dflash2", "expected_masked", "expected_backend_calls"),
    [(False, True, 1), (True, False, 0)],
)
def test_dflash2_configures_unmasked_nextn_without_backend_sampling(
    is_dflash2, expected_masked, expected_backend_calls
):
    engine = object.__new__(LlamaDFlashDecoding)
    engine.is_dflash2 = is_dflash2
    engine.target_layer_ids = [2, 5]
    backend_calls = []
    engine._enable_backend_sampling = lambda internals: backend_calls.append(
        internals
    )

    class _TargetContext:
        def __init__(self):
            self.layers = []

        def set_embeddings_layer_inp(self, layer_id, enabled):
            self.layers.append((layer_id, enabled))

    class _DraftContext:
        def __init__(self):
            self.nextn = []
            self.causal = []

        def set_embeddings_nextn(self, enabled, masked):
            self.nextn.append((enabled, masked))

        def set_causal_attn(self, enabled):
            self.causal.append(enabled)

    engine.target_context = _TargetContext()
    engine.draft_context = _DraftContext()
    internals = object()

    engine._configure_draft_execution(internals)

    assert len(backend_calls) == expected_backend_calls
    assert engine.target_context.layers == [(2, True), (5, True)]
    assert engine.draft_context.nextn == [(True, expected_masked)]
    assert engine.draft_context.causal == [False]


def test_dflash_process_interleaves_target_layers_and_injects_fused_rows():
    engine = object.__new__(LlamaDFlashDecoding)
    engine.target_layer_ids = [1, 3]
    engine.n_embd_tgt = 2
    engine.n_embd_enc = 4
    engine.n_embd_dec = 3
    layer_1 = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    layer_3 = np.asarray([[10, 20], [30, 40]], dtype=np.float32)

    class _TargetContext:
        def get_embeddings_layer_inp(self, layer_id):
            values = {1: layer_1, 3: layer_3}[layer_id]
            return values.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )

    fused = np.asarray([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
    engine.target_context = _TargetContext()
    engine.draft_context = _FakeDFlashDraftContext(fused=fused)
    engine.encoder_batch = _FakeDFlashBatch()
    engine.inject_batch = _FakeDFlashBatch()
    engine.verify_positions = []
    engine.is_mrope = False

    class _TargetBatch:
        n_tokens = 2
        token = [11, 12]
        embd = None
        pos = [4, 5]
        n_seq_id = [1, 1]
        seq_id = [[0], [0]]

    engine.process(_TargetBatch())

    np.testing.assert_array_equal(
        engine.draft_context.encoded[0].reshape(2, 4),
        [[1, 2, 10, 20], [3, 4, 30, 40]],
    )
    np.testing.assert_array_equal(
        engine.draft_context.decoded[0][2].reshape(2, 3), fused
    )
    assert engine.draft_context.decoded[0][1] == [4, 5]
    assert engine.draft_context.synchronized
    assert engine.verify_positions == [4, 5]
    assert engine.encoder_batch.logits == [True, True]
    assert engine.inject_batch.logits == []


def test_dflash_mrope_process_uses_target_positions_for_encoder_and_injection():
    engine = object.__new__(LlamaDFlashDecoding)
    engine.target_layer_ids = [1]
    engine.n_embd_tgt = 2
    engine.n_embd_enc = 2
    engine.n_embd_dec = 2
    engine.is_mrope = True
    target_rows = np.asarray([[1, 2], [3, 4]], dtype=np.float32)

    class _TargetContext:
        def get_embeddings_layer_inp(self, layer_id):
            assert layer_id == 1
            return target_rows.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )

    fused = np.asarray([[5, 6], [7, 8]], dtype=np.float32)
    engine.target_context = _TargetContext()
    engine.draft_context = _FakeDFlashDraftContext(fused=fused)
    engine.encoder_batch = _FakeDFlashBatch()
    engine.inject_batch = _FakeDFlashBatch()
    engine.verify_positions = []

    class _TargetBatch:
        n_tokens = 2
        token = [11, 12]
        embd = None
        pos = [4, 5]
        n_seq_id = [1, 1]
        seq_id = [[0], [0]]

    engine.process(_TargetBatch())

    expected = [[4, 5], [4, 5], [4, 5], [0, 0]]
    assert engine.encoder_batch.mrope_positions == expected
    assert engine.inject_batch.mrope_positions == expected
    assert engine.verify_positions == [4, 5]


def test_dflash_rejects_multimodal_embedding_batches_until_positions_are_mapped():
    engine = object.__new__(LlamaDFlashDecoding)

    class _EmbeddingBatch:
        n_tokens = 1
        token = None
        embd = [0.0]

    with pytest.raises(NotImplementedError, match="text-only"):
        engine.process(_EmbeddingBatch())


def test_dflash_accepts_final_target_layer_input_tap_for_nemotron():
    LlamaDFlashDecoding._validate_target_layer_ids([0, 17, 52], 52)

    with pytest.raises(ValueError, match=r"53 not in \[0, 52\]"):
        LlamaDFlashDecoding._validate_target_layer_ids([53], 52)


def test_dflash_native_verification_rollback_checks_memory_removal():
    engine = object.__new__(LlamaDFlashDecoding)
    engine._use_native_draft_rollback = True
    engine.verify_positions = [8, 9, 10]
    engine._checkpoint_stats = {"native_verification_rollbacks": 0}

    class _Context:
        def memory_seq_rm(self, seq_id, p0, p1):
            return False

    engine.draft_context = _Context()
    with pytest.raises(RuntimeError, match="verification rollback failed"):
        engine.rollback_verified({"position": 7}, n_accepted=1)


def test_dflash_native_checkpoint_uses_suffix_removal_without_state_buffer():
    engine = object.__new__(LlamaDFlashDecoding)
    engine.draft_context = _FakeCheckpointContext(position=7, state_size=8)
    engine._use_native_draft_rollback = True
    engine.n_embd_dec = 2
    engine.verify_positions = [8, 9]
    engine.verify_fused = np.ones((2, 2), dtype=np.float32)
    engine._active_verification_checkpoint = None
    engine.reset_checkpoint_stats()

    checkpoint = engine.checkpoint()
    engine.restore(checkpoint)
    stats = engine.checkpoint_stats()

    assert checkpoint["mode"] == "native"
    assert checkpoint["buffer"] is None
    assert engine.draft_context.get_flags == []
    assert engine.draft_context.set_flags == []
    assert engine.draft_context.removals == [(0, 8, -1)]
    assert stats["native_captures"] == 1
    assert stats["native_restores"] == 1
    assert stats["device_captures"] == 0
    assert stats["device_restores"] == 0


def test_dflash_can_follow_target_rollback_with_device_draft_checkpoints():
    engine = object.__new__(LlamaDFlashDecoding)
    engine._use_native_draft_rollback = False

    assert engine.can_follow_target_native_rollback()


def test_dflash_device_checkpoint_restore_reclaims_noise_suffix():
    engine = object.__new__(LlamaDFlashDecoding)
    engine.draft_context = _FakeCheckpointContext(position=7, state_size=8)
    engine.n_embd_dec = 2
    engine.verify_positions = [8, 9]
    engine.verify_fused = np.ones((2, 2), dtype=np.float32)
    engine._active_verification_checkpoint = {"position": 7}
    engine.reset_checkpoint_stats()
    checkpoint = {
        "position": 7,
        "mode": "on-device",
        "buffer": (ctypes.c_uint8 * 8)(),
        "size": 8,
        "flags": llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
        | llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE,
    }

    engine.restore(checkpoint)

    assert engine.draft_context.removals == [(0, 8, -1)]
    assert engine.checkpoint_stats()["device_restores"] == 1


def test_dflash_device_checkpoint_rollback_replays_accepted_fused_prefix():
    engine = object.__new__(LlamaDFlashDecoding)
    engine._use_native_draft_rollback = False
    engine.verify_positions = [8, 9, 10]
    engine.verify_fused = np.asarray(
        [[1, 2], [3, 4], [5, 6]], dtype=np.float32
    )
    engine.n_embd_dec = 2
    engine.inject_batch = _FakeDFlashBatch()
    engine.draft_context = _FakeDFlashDraftContext()
    restored = []

    def restore(checkpoint, seq_id=0):
        restored.append(checkpoint)
        engine.verify_positions.clear()
        engine.verify_fused = np.empty((0, 2), dtype=np.float32)

    engine.restore = restore
    engine.rollback_verified({"position": 7}, n_accepted=1)

    assert restored == [{"position": 7}]
    np.testing.assert_array_equal(
        engine.draft_context.decoded[0][2].reshape(2, 2), [[1, 2], [3, 4]]
    )
    assert engine.draft_context.decoded[0][1] == [8, 9]
    assert engine.draft_context.synchronized


def test_dflash_checkpoint_truncation_replays_only_the_accepted_prefix():
    engine = object.__new__(LlamaDFlashDecoding)
    engine._use_native_draft_rollback = False
    engine._active_verification_checkpoint = {"position": 7}
    engine.verify_positions = [8, 9, 10]
    engine.verify_fused = np.asarray(
        [[1, 2], [3, 4], [5, 6]], dtype=np.float32
    )
    engine.n_embd_dec = 2
    engine.inject_batch = _FakeDFlashBatch()
    engine.draft_context = _FakeDFlashDraftContext()
    restored = []
    engine.restore = lambda checkpoint, seq_id=0: restored.append(checkpoint)

    engine.truncate(10)

    assert restored == [{"position": 7}]
    np.testing.assert_array_equal(
        engine.draft_context.decoded[0][2].reshape(2, 2), [[1, 2], [3, 4]]
    )
    assert engine.draft_context.decoded[0][1] == [8, 9]
    assert engine.draft_context.synchronized
