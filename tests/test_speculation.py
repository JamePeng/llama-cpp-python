import os
from contextlib import closing, contextmanager
import builtins
import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import llama_cpp
from llama_cpp import llama_cpp as llama_cpp_lib
from llama_cpp import _internals as internals

from llama_cpp.llama_speculative import (
    LlamaDFlashDecoding,
    LlamaMTPDecoding,
    LlamaNGramMapDecoding,
    SpecConfig,
    SpeculativeType,
    create_native_spec_engine,
    speculative_output_limits,
)


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

    results = {
        "llama_get_sampled_probs_ith": object(),
        "llama_get_sampled_logits_ith": object(),
        "llama_get_sampled_candidates_ith": object(),
        "llama_get_sampled_token_ith": 17,
        "llama_get_sampled_probs_count_ith": 3,
        "llama_get_sampled_logits_count_ith": 4,
        "llama_get_sampled_candidates_count_ith": 5,
    }
    for name, result in results.items():
        monkeypatch.setattr(internals.llama_cpp, name,
                            lambda ctx, index, result=result, name=name: calls.append((name, ctx, index)) or result)
        assert getattr(context, name.removeprefix("llama_"))(-1) == result
        assert calls[-1] == (name, "native-context", -1)

    perf = llama_cpp_lib.llama_perf_context_data()
    perf.n_eval = 9
    monkeypatch.setattr(internals.llama_cpp, "llama_perf_context", lambda ctx: perf)
    assert context.perf_context().n_eval == 9
    for action, method in (("print", "print_timings"), ("reset", "reset_timings")):
        monkeypatch.setattr(internals.llama_cpp, f"llama_perf_context_{action}",
                            lambda ctx, action=action: calls.append((f"perf-{action}", ctx)))
        getattr(context, method)()
        assert calls[-1] == (f"perf-{action}", "native-context")
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


def test_speculative_output_limits_match_llama_cpp():
    assert speculative_output_limits(32, 1, 3) == (4, 4)
    assert speculative_output_limits(32, 4, 3) == (16, 4)
    assert speculative_output_limits(3, 4, 8) == (3, 3)


def test_draft_context_does_not_inherit_target_embedding_mode_or_moe_cache():
    target = llama_cpp_lib.llama_context_default_params()
    target.embeddings = True
    target.pooling_type = llama_cpp_lib.LLAMA_POOLING_TYPE_MEAN
    target.moe_cache_size = 64 * 1024 * 1024

    draft = LlamaMTPDecoding._copy_draft_context_params(
        target, llama_cpp_lib.LLAMA_POOLING_TYPE_UNSPECIFIED
    )

    assert target.embeddings is True
    assert target.pooling_type == llama_cpp_lib.LLAMA_POOLING_TYPE_MEAN
    assert target.moe_cache_size == 64 * 1024 * 1024
    assert draft.embeddings is False
    assert draft.pooling_type == llama_cpp_lib.LLAMA_POOLING_TYPE_UNSPECIFIED
    assert draft.moe_cache_size == 0


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


def _checkpoint_test_engine(*, native, engine_class=LlamaMTPDecoding):
    engine = object.__new__(engine_class)
    engine.is_mem_shared = False
    engine.draft_context = _FakeCheckpointContext()
    engine.n_embd = 2
    engine.pending_h = np.asarray([1.0, 2.0], dtype=np.float32)
    engine.verify_h = np.empty((0, 2), dtype=np.float32)
    engine.verify_tokens = []
    engine.verify_positions = []
    engine.verify_embeddings = None
    engine.verify_media_positions = None
    engine.embedding_batch = None
    engine._llama_cpp_lib = llama_cpp_lib
    engine.n_embd_enc = engine.n_embd_dec = 2
    engine.verify_features = np.ones((2, 2), dtype=np.float32)
    engine._active_verification_checkpoint = None
    engine._use_native_draft_rollback = native
    engine._pending_verification_checkpoint = None
    engine.reset_checkpoint_stats()
    return engine


class _MTPCatchupBatch:
    def __init__(self):
        self.rows = []

    def reset(self):
        self.rows.clear()

    def add_token_embedding(self, token, hidden, position, seq_ids, output):
        self.rows.append((token, np.array(hidden, copy=True), position))

    def add_embeddings(self, features, *, positions, states):
        self.rows.extend((row.copy(), hidden.copy(), position)
                         for row, hidden, position in zip(features, states, positions))


@contextmanager
def _target_batch(positions, *, tokens=None, embeddings=None, outputs=None, seq_ids=(0,)):
    """Own a legacy target batch; accept scalar or entry-major media positions."""
    count = len(positions)
    width = 0 if embeddings is None else embeddings.shape[1]
    coordinates = np.asarray(positions)
    flags = [False] * count if outputs is None else outputs
    with closing(internals.LlamaBatch(n_tokens=count, embd=width, n_seq_max=max(seq_ids) + 1)) as batch:
        if embeddings is None:
            batch.add_sequence(tokens, coordinates[:, 0] if coordinates.ndim == 2 else positions, seq_ids, flags)
        elif coordinates.ndim == 2:
            batch.add_embeddings_mrope(embeddings.reshape(-1), pos_array=coordinates.T, seq_ids=seq_ids, logits_array=flags)
        else:
            batch.add_embeddings(embeddings.reshape(-1), pos_array=positions, seq_ids=seq_ids, logits_array=flags)
        yield batch


@pytest.fixture
def mtp_process_engine(monkeypatch):
    engine = _checkpoint_test_engine(native=True)
    engine.is_mem_shared = False
    engine.chain_heads = False
    engine.n_mtp_layers = 1
    engine._llama_cpp_lib = llama_cpp_lib
    engine.batch = _MTPCatchupBatch()
    rows = np.array([[10, 11], [20, 21]], dtype=np.float32)
    engine.target_context = SimpleNamespace(get_embeddings_nextn=lambda: rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    engine.target_model = SimpleNamespace(n_embd_inp=lambda: 3, rope_type=lambda: llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_NONE)
    decoded, offsets, allocations = [], [], []
    engine.draft_context.decode = lambda batch: decoded.append(list(batch.rows)) or 0
    engine.draft_context.set_nextn_layer_offset = offsets.append

    def allocate(*, context):
        assert context is engine.draft_context
        allocations.append(True)
        return _MTPCatchupBatch()

    monkeypatch.setattr(internals, "LlamaBatchExt", allocate)
    return engine, rows, decoded, offsets, allocations


@pytest.mark.parametrize("rope", [llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_NONE, llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_MROPE, llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_IMROPE])
def test_mtp_process_media_positions_and_cross_batch_hidden_shift(mtp_process_engine, rope):
    engine, target_h, decoded, _, allocations = mtp_process_engine
    engine.target_model.rope_type = lambda: rope
    features = np.arange(6, dtype=np.float32).reshape(2, 3)
    positions = [[8, 9], [2, 3], [4, 5], [0, 0]]
    if rope != llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_NONE:
        positions[0] = [8, 8]  # image patches may share their cache position
    expected_positions = positions[0] if rope == llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_NONE else np.array(positions).T.tolist()
    with _target_batch(expected_positions, embeddings=features) as source:
        engine.process(source.batch)
        assert [row[2] for row in decoded[-1]] == expected_positions
        np.testing.assert_array_equal([row[0] for row in decoded[-1]], features)
        np.testing.assert_array_equal([row[1] for row in decoded[-1]], [[1, 2], target_h[0]])
        np.testing.assert_array_equal(engine.pending_h, target_h[-1])
        # Reuse the builder and carry the last target row across media batches.
        from llama_cpp.mtmd_cpp import mtmd_helper_embd_batch
        engine.process_media(mtmd_helper_embd_batch(
            2, source.batch.embd, 3, source.batch.pos,
            1 if rope == llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_NONE else 4, 0,
        ))
        np.testing.assert_array_equal(decoded[-1][0][1], target_h[-1])
        assert allocations == [True]
        np.ctypeslib.as_array(source.batch.embd, shape=(6,))[:] = -1
    np.testing.assert_array_equal(engine.verify_embeddings, features)
    assert engine.verify_positions == positions[0]
    with _target_batch([10, 11], tokens=[1, 2]) as source:
        engine.process(source.batch)
    assert [row[0] for row in decoded[-1]] == [1, 2]
    np.testing.assert_array_equal(decoded[-1][0][1], target_h[-1])
    assert engine.verify_embeddings is None and engine.verify_media_positions is None


@pytest.mark.parametrize("mode", ["shared", "chain", "failure"])
def test_mtp_media_catchup_modes_preserve_committed_hidden_state(mtp_process_engine, mode):
    engine, target_h, decoded, offsets, allocations = mtp_process_engine
    engine.is_mem_shared = mode == "shared"
    engine.chain_heads = mode in {"chain", "failure"}
    engine.n_mtp_layers = 2 if engine.chain_heads else 1
    if mode == "failure":
        engine.draft_context.decode = lambda batch: 1
    with _target_batch([8, 9], embeddings=np.arange(6, dtype=np.float32).reshape(2, 3)) as source:
        if mode == "failure":
            with pytest.raises(RuntimeError, match="catch-up decode failed"):
                engine.process(source.batch)
            np.testing.assert_array_equal(engine.pending_h, [1, 2])
            assert engine.verify_positions == [] and offsets == [0, 0]
        else:
            engine.process(source.batch)
            np.testing.assert_array_equal(engine.pending_h, target_h[-1])
            assert len(decoded) == (0 if mode == "shared" else 2)
            assert allocations == ([] if mode == "shared" else [True])
            if mode == "chain":
                assert offsets == [0, 1, 0]
                assert engine.draft_context.removals == [(0, 8, -1)] * 2


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
def test_mtp_media_rollback_restores_accepted_rows_and_coordinates(mtp_process_engine, native, pinned):
    engine, target_h, decoded, _, _ = mtp_process_engine
    engine._use_native_draft_rollback = native
    checkpoint = engine.checkpoint()
    features = np.arange(6, dtype=np.float32).reshape(2, 3)
    positions = [8, 8 if pinned else 9]
    coordinates = [[positions[0], 2, 4, 0], [positions[1], 3, 5, 0]]
    engine._process_target_rows([], positions, target_h.copy(), embeddings=features, media_positions=coordinates)
    engine.rollback_verified(checkpoint, n_accepted=0)
    np.testing.assert_array_equal(engine.pending_h, target_h[0])
    assert engine.draft_context.removals == [(0, 9 if native and not pinned else 8, -1)]
    if native and not pinned:
        assert len(decoded) == 1
    else:
        assert engine.checkpoint_stats()["native_restores" if native else "device_restores"] == 1
        assert len(decoded) == 2 and len(decoded[-1]) == 1
        np.testing.assert_array_equal(decoded[-1][0][0], features[0])
        np.testing.assert_array_equal(decoded[-1][0][1], [1, 2])
        assert decoded[-1][0][2] == coordinates[0]


@pytest.mark.parametrize("invalid", ["position", "sequence", "membership", "media-width", "media-axes", "media-sequence"])
def test_mtp_process_rejects_invalid_metadata_before_reading_hidden_rows(mtp_process_engine, invalid):
    engine, _, decoded, _, allocations = mtp_process_engine
    with _target_batch([8, 9], embeddings=np.arange(6, dtype=np.float32).reshape(2, 3),
                       seq_ids=(0, 1) if invalid == "membership" else (0,)) as source:
        batch = SimpleNamespace(n_tokens=source.batch.n_tokens, token=source.batch.token, embd=source.batch.embd,
                                pos=source.batch.pos, n_seq_id=source.batch.n_seq_id, seq_id=source.batch.seq_id)
        if invalid == "position":
            batch.pos = None
        elif invalid == "sequence":
            batch.seq_id[0][0] = 1
        engine.target_context.get_embeddings_nextn = lambda: pytest.fail("Invalid metadata must not read target states")
        if invalid.startswith("media-"):
            from llama_cpp.mtmd_cpp import mtmd_helper_embd_batch
            view = mtmd_helper_embd_batch(2, batch.embd,
                4 if invalid == "media-width" else 3, batch.pos,
                4 if invalid == "media-axes" else 1,
                1 if invalid == "media-sequence" else 0)
            with pytest.raises(NotImplementedError if invalid == "media-sequence" else ValueError):
                engine.process_media(view)
        else:
            with pytest.raises(ValueError if invalid == "position" else NotImplementedError):
                engine.process(batch)
    assert decoded == allocations == []
    np.testing.assert_array_equal(engine.pending_h, [1, 2])


@pytest.mark.parametrize("engine_class,native,shared", [
    pytest.param(LlamaMTPDecoding, True, False, id="mtp-native"),
    pytest.param(LlamaMTPDecoding, False, False, id="mtp-device"),
    pytest.param(LlamaMTPDecoding, False, True, id="mtp-shared-device"),
    pytest.param(LlamaDFlashDecoding, True, False, id="dflash-native"),
    pytest.param(LlamaDFlashDecoding, False, False, id="dflash-device"),
])
def test_model_draft_checkpoint_restores_suffix_and_reuses_capture(engine_class, native, shared):
    engine = _checkpoint_test_engine(native=native, engine_class=engine_class)
    engine.is_mem_shared = shared
    checkpoint = engine.checkpoint()
    engine.pending_h.fill(0.0)
    engine._active_verification_checkpoint = checkpoint
    engine.restore(checkpoint)
    engine._pending_verification_checkpoint = checkpoint
    assert engine.take_verification_checkpoint() is checkpoint
    stats = engine.checkpoint_stats()
    assert stats["captures"] == stats["restores"] == stats["verification_reuses"] == 1
    assert checkpoint["mode"] == ("native" if native else "on-device")
    assert engine.draft_context.removals == ([] if shared else [(0, 8, -1)])
    for mode in ("native", "device"):
        expected = int(native == (mode == "native"))
        assert stats[f"{mode}_captures"] == stats[f"{mode}_restores"] == expected
    flags = llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
    assert engine.draft_context.get_flags == engine.draft_context.set_flags == ([] if native else [flags])
    assert stats["buffer_bytes"] == (0 if native else engine.draft_context.state_size)
    assert (checkpoint["buffer"] is None) == native
    if engine_class is LlamaMTPDecoding:
        np.testing.assert_array_equal(engine.pending_h, [1, 2])
    else:
        assert engine.verify_features.shape == (0, 2)
        assert engine._active_verification_checkpoint is checkpoint


@pytest.mark.parametrize("engine_class", [LlamaMTPDecoding, LlamaDFlashDecoding])
@pytest.mark.parametrize("invalidate", ["capture", "clear", "restore_failure"])
def test_draft_checkpoint_rejects_invalidated_objects(engine_class, invalidate):
    engine = _checkpoint_test_engine(native=True, engine_class=engine_class)
    engine._closed = False
    engine._backend_sampler = None
    checkpoint = engine.checkpoint()
    if invalidate == "capture":
        engine.checkpoint()
    elif invalidate == "clear":
        engine.draft_context.memory_clear = lambda data: None
        engine.clear()
    else:
        engine.draft_context.memory_seq_rm = lambda *args: False
        with pytest.raises(RuntimeError, match="rollback failed"):
            engine.restore(checkpoint)
    with pytest.raises(RuntimeError, match="stale"):
        engine.restore(checkpoint)


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
    engine.embedding_batch = _Closable()
    engine.draft_context = _Closable(context="draft-context")
    engine._owns_model = False
    engine.draft_model = object()

    backend_sampler = engine._backend_sampler
    batch = engine.batch
    embedding_batch = engine.embedding_batch
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
    assert embedding_batch.closed
    assert draft_context.closed
    assert engine._backend_sampler is None
    assert engine.draft_context is None


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


@pytest.mark.parametrize("verifying", [False, True], ids=["ordinary-retry", "atomic-verification"])
def test_eval_batch_retry_respects_verification_boundary(verifying):
    llm, _ = _eval_test_llama(statuses=[1, 0], verifying=verifying, speculative=verifying)
    if verifying:
        with pytest.raises(RuntimeError, match="verification batch cannot be split"):
            llm._decode_eval_batch([1, 2, 3, 4], 4)
    else:
        assert llm._decode_eval_batch([1, 2, 3, 4], 4) == 2
    assert llm._ctx.decode_sizes == ([4] if verifying else [4, 2])


def test_speculative_verification_batch_must_fit_n_batch():
    llm = object.__new__(llama_cpp.Llama)
    llm._speculative_verifying = True
    llm.n_batch = 3

    with pytest.raises(RuntimeError, match="verification batch exceeds n_batch"):
        llm.eval([1, 2, 3, 4], copy_logits=False)


@pytest.mark.parametrize("mode", ["native", "checkpoint", "truncate"])
def test_interrupted_speculation_restores_committed_boundary(mode):
    llm = object.__new__(llama_cpp.Llama)
    llm._ctx = _FakeCheckpointContext()
    llm.speculative = SimpleNamespace(rollback_verified=Mock(), restore=Mock(), truncate=Mock())
    checkpoint = SimpleNamespace(pos=6)
    llm._hybrid_cache_mgr = SimpleNamespace(find_best_checkpoint=Mock(return_value=checkpoint),
                                          restore_checkpoint=Mock(return_value=True))
    llm.is_hybrid = mode != "truncate"
    llm.input_ids = np.array([1, 2, 3, 4, 5, 6, 10, 11, 12], dtype=np.intc)
    llm.n_tokens = 9
    llm._last_eval_output_start, llm._last_eval_output_count = 6, 3
    replayed = []

    def replay(tokens, **kwargs):
        replayed.append((list(tokens), kwargs))
        llm.n_tokens += len(tokens)

    llm.eval = replay
    active_loras = [{"name": "test", "scale": 0.5}]
    control_vector = {"data": [1.0]}
    accepted = 0 if mode == "truncate" else 1
    draft_checkpoint = None if mode == "truncate" else "draft-checkpoint"
    assert llm._recover_interrupted_speculation(
        verification_start=6, evaluated_tokens=[10, 11, 12], delivered_accepted=accepted,
        speculative_checkpoint=draft_checkpoint, use_native_rollback=mode == "native",
        active_loras=active_loras, control_vector=control_vector,
    ) == mode
    assert llm.n_tokens == 7 + accepted
    assert llm._last_eval_output_start == 6
    for operation, expected_mode in (("rollback_verified", "native"), ("restore", "checkpoint"), ("truncate", "truncate")):
        assert getattr(llm.speculative, operation).call_count == int(mode == expected_mode)
    assert llm._hybrid_cache_mgr.find_best_checkpoint.call_count == int(mode == "checkpoint")
    assert llm._hybrid_cache_mgr.restore_checkpoint.call_count == int(mode == "checkpoint")
    if mode == "checkpoint":
        llm._hybrid_cache_mgr.find_best_checkpoint.assert_called_once_with([1, 2, 3, 4, 5, 6], 0)
        llm._hybrid_cache_mgr.restore_checkpoint.assert_called_once_with(checkpoint, seq_id=0)
        llm.speculative.restore.assert_called_once_with(draft_checkpoint, seq_id=0)
        assert replayed == [([10, 11], dict(active_loras=active_loras, control_vector=control_vector, copy_logits=False))]
        assert llm._ctx.removals == []
    else:
        assert llm._ctx.removals == [(0, 7 + accepted, -1)]
        assert llm._last_eval_output_count == 1 + accepted and replayed == []
        if mode == "native":
            llm.speculative.rollback_verified.assert_called_once_with(draft_checkpoint, accepted, seq_id=0)
        else:
            llm.speculative.truncate.assert_called_once_with(7, seq_id=0)


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
        self.synchronized = False
        self.removals = []

    def n_ubatch(self):
        return 8

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
    engine.draft_min = engine.config.draft_n_min
    engine.mask_token_id = 99
    engine.n_embd_dec = 2
    engine.noise_batch = _FakeDFlashBatch()
    engine._pending_verification_checkpoint = None
    engine._candidate = lambda output_index: (100 + output_index, 0.9)
    engine.checkpoint = lambda seq_id=0: {"position": 5}
    engine.restore = lambda checkpoint, seq_id=0: None
    return engine


@pytest.mark.parametrize("dflash2", [False, True], ids=["dflash", "dflash2"])
@pytest.mark.parametrize("n_max,n_min,trained_limit,cutoff,expected_count", [
    pytest.param(3, 0, 3, False, 3, id="full-block"),
    pytest.param(1, 3, 3, False, 1, id="cap-after-minimum"),
    pytest.param(1, 3, 3, True, 0, id="confidence-before-minimum"),
    pytest.param(1, 0, 3, True, 1, id="cap-filtered-prefix"),
    pytest.param(3, 3, 1, False, 1, id="trained-clamp"),
    pytest.param(8, 0, 3, False, 3, id="cap-above-configured"),
    pytest.param(0, 0, 3, False, 0, id="disabled-call"),
])
def test_dflash_block_length_filtering_precedes_runtime_cap(dflash2, n_max, n_min, trained_limit, cutoff, expected_count):
    engine = _draft_test_dflash_engine(dspark=False, sample_from_anchor=True, p_min=0.8)
    engine.is_dflash2 = dflash2
    engine.config.draft_n_min = n_min
    engine.draft_limit, engine.draft_min = trained_limit, min(n_min, trained_limit)
    engine.checkpoint = Mock(return_value={"position": 5})
    engine.restore = Mock()
    if dflash2:
        engine.selector_top_k, engine.n_embd_dec = 2, 6
        lattice = np.array([[0] * 6, [101, 102, 0, 3, 0, 3],
                            [201, 202, 3, 0, 3, 0], [301, 302, 0, 3, 0, 3]], dtype=np.float32)
        if cutoff:
            lattice[2, 2:] = 0
        engine.draft_context = _FakeDFlashDraftContext(confidence=lattice)
        proposals = [102, 201, 302]
    else:
        engine.draft_context = _FakeDFlashDraftContext()
        engine._candidate = lambda i: (100 + i, 0.4 if cutoff and i == 2 else 0.9)
        proposals = [101, 102, 103]
    result = engine.draft([], n_past=7, id_last=42, n_max=n_max)
    assert result.tolist() == proposals[:expected_count]
    if n_max == 0:
        assert engine.draft_context.decoded == []
        engine.checkpoint.assert_not_called()
        engine.restore.assert_not_called()
    else:
        assert engine.noise_batch.tokens == [42] + [99] * trained_limit
        assert engine.noise_batch.positions == list(range(7, 8 + trained_limit))
        assert engine.noise_batch.logits == [not dflash2] * (trained_limit + 1)
        engine.checkpoint.assert_called_once_with(0)
        engine.restore.assert_called_once_with({"position": 5}, 0)
    assert engine._pending_verification_checkpoint == ({"position": 5} if expected_count else None)


@pytest.mark.parametrize("n_max", [1, 3])
def test_dspark_anchor_layout_uses_confidence_to_truncate_block(n_max):
    engine = _draft_test_dflash_engine(
        dspark=True, sample_from_anchor=True, p_min=0.5
    )
    confidence = np.asarray(
        [[0.9, 0.9], [0.4, 0.4], [0.8, 0.8]], dtype=np.float32
    )
    engine.draft_context = _FakeDFlashDraftContext(confidence=confidence)

    result = engine.draft([], n_past=7, id_last=42, n_max=n_max)

    assert engine.noise_batch.tokens == [42, 99, 99]
    assert result.tolist() == [100]


@pytest.mark.parametrize("p_min,selector_rows,expected", [
    pytest.param(0.0, [[101, 102, 0, 2, 0, 2], [201, 202, 0, 0, 3, 1], [301, 302, 0, 4, 0, 0]],
                 [102, 201, 302], id="walk-lattice"),
    pytest.param(0.8, [[101, 102, 0, 3, 0, 3], [201, 202, 0, 0, 0, 1], [301, 302, 0, 0, 0, 0]],
                 [102], id="stop-low-confidence"),
])
def test_dflash2_selector_lattice_and_probability_cutoff(p_min, selector_rows, expected):
    engine = _draft_test_dflash_engine(dspark=False, sample_from_anchor=True, p_min=p_min)
    engine.is_dflash2 = True
    engine.selector_top_k, engine.n_embd_dec = 2, 6
    lattice = np.array([[0] * 6] + selector_rows, dtype=np.float32)
    engine.draft_context = _FakeDFlashDraftContext(confidence=lattice)
    assert engine.draft([], n_past=7, id_last=42, n_max=3).tolist() == expected
    assert engine.noise_batch.tokens == [42, 99, 99, 99]
    assert engine.noise_batch.logits == [False] * 4
    assert engine._pending_verification_checkpoint == {"position": 5}


@pytest.mark.parametrize(
    ("is_dflash2", "expected_masked", "expected_backend_calls"),
    [(False, True, 1), (True, False, 0)],
)
def test_dflash2_configures_unmasked_nextn_without_backend_sampling(
    is_dflash2, expected_masked, expected_backend_calls
):
    engine = object.__new__(LlamaDFlashDecoding)
    engine.is_dflash2 = is_dflash2
    engine.causal_attn = True
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
    assert engine.draft_context.causal == [True]


def test_dspark_probability_filter_requires_confidence_head():
    metadata = {"dflash.has_confidence_head": "false"}

    with pytest.raises(ValueError, match="no confidence head"):
        LlamaDFlashDecoding._validate_dspark_confidence_head(metadata, 0.1)

    LlamaDFlashDecoding._validate_dspark_confidence_head(metadata, 0.0)


@pytest.mark.parametrize("kind,positions", [
    pytest.param("layers", [4, 5], id="interleaved-layers"),
    pytest.param("mrope", [4, 5], id="mrope-positions"),
    pytest.param("embedding", [7], id="single-image-row"),
    pytest.param("embedding", [7, 8], id="adjacent-image-rows"),
    pytest.param("embedding", [7, 20], id="noncontiguous-image-rows"),
    pytest.param("embedding", [7, 7, 7], id="pinned-image-skip"),
])
def test_dflash_process_preserves_feature_layout_and_positions(kind, positions):
    engine = object.__new__(LlamaDFlashDecoding)
    engine.target_layer_ids = [1, 3] if kind == "layers" else [1]
    engine.n_embd_tgt = engine.n_embd_dec = 2
    engine.n_embd_enc = 2 * len(engine.target_layer_ids)
    engine.is_mrope = kind == "mrope"
    layers = {1: np.array([[1, 2], [3, 4]], dtype=np.float32),
              3: np.array([[10, 20], [30, 40]], dtype=np.float32)}
    pinned = positions == [7, 7, 7]

    def features(layer_id):
        if pinned:
            pytest.fail("Pinned images must skip feature extraction")
        assert layer_id in engine.target_layer_ids
        return layers[layer_id].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    engine.target_context = SimpleNamespace(get_embeddings_layer_inp=features)
    engine.draft_context = _FakeDFlashDraftContext()
    engine.inject_batch = _FakeDFlashBatch()
    engine.verify_positions = [3]
    engine.verify_features = np.array([[1, 2]], dtype=np.float32)
    batch = SimpleNamespace(n_tokens=len(positions), token=None if kind == "embedding" else [11, 12],
                            embd=[0.0] if kind == "embedding" else None, pos=positions,
                            n_seq_id=[1] * len(positions), seq_id=[[0]] * len(positions))
    engine.process(batch)
    if pinned:
        assert engine.draft_context.decoded == [] and engine.verify_positions == [3]
        np.testing.assert_array_equal(engine.verify_features, [[1, 2]])
        return
    expected = [[1, 2, 10, 20], [3, 4, 30, 40]] if kind == "layers" else layers[1][:len(positions)]
    np.testing.assert_array_equal(engine.draft_context.decoded[0][2].reshape(len(positions), engine.n_embd_enc), expected)
    np.testing.assert_array_equal(engine.verify_features, expected)
    assert engine.verify_positions == engine.draft_context.decoded[0][1] == positions
    assert engine.inject_batch.logits == []
    if kind == "mrope":
        assert engine.inject_batch.mrope_positions == [positions] * 3 + [[0] * len(positions)]


def test_dflash_native_position_is_independent_of_history_length():
    engine = _draft_test_dflash_engine(dspark=False, sample_from_anchor=True)
    engine.draft_context = _FakeDFlashDraftContext()
    result = engine.draft_at_position([1] * 100, pos0=7, id_last=42, n_max=3)
    assert result.tolist() == [101, 102, 103]
    assert engine.noise_batch.positions == [7, 8, 9, 10]
    with pytest.raises(ValueError, match="non-negative"):
        engine.draft_at_position([], pos0=-1, id_last=42, n_max=3)


@pytest.mark.parametrize("native_max,cursor", [(-1, 0), (6, 7), (99, 100)])
def test_speculative_start_position_matches_verification(native_max, cursor):
    from types import SimpleNamespace

    llm = object.__new__(llama_cpp.Llama)
    llm._ctx = SimpleNamespace(memory_seq_pos_max=lambda seq_id: native_max)
    assert llm._speculative_start_position(cursor) == cursor
    with pytest.raises(NotImplementedError, match="position ledger"):
        llm._speculative_start_position(cursor + 10)


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


@pytest.mark.parametrize("operation", ["rollback", "truncate"])
def test_dflash_checkpoint_recovery_replays_only_accepted_prefix(operation):
    engine = _checkpoint_test_engine(native=False, engine_class=LlamaDFlashDecoding)
    checkpoint = {"position": 7}
    engine._active_verification_checkpoint = checkpoint
    engine.verify_positions = [8, 9, 10]
    engine.verify_features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    engine.inject_batch = _FakeDFlashBatch()
    engine.draft_context = _FakeDFlashDraftContext()
    restored = []

    def restore(saved, seq_id=0):
        restored.append(saved)
        engine.verify_positions.clear()
        engine.verify_features = np.empty((0, 2), dtype=np.float32)

    engine.restore = restore
    if operation == "rollback":
        engine.rollback_verified(checkpoint, n_accepted=1)
    else:
        engine.truncate(10)
    assert restored == [checkpoint]
    np.testing.assert_array_equal(engine.draft_context.decoded[0][2].reshape(2, 2), [[1, 2], [3, 4]])
    assert engine.draft_context.decoded[0][1] == [8, 9]


@pytest.mark.parametrize("kind", [SpeculativeType.NGRAM_MAP_K, SpeculativeType.NGRAM_MAP_K4V])
def test_ngram_never_proposes_media_ids(kind):
    engine = LlamaNGramMapDecoding(ngram_size=2, num_pred_tokens=2, spec_type=kind, min_hits=1)
    try:
        history = [1, 2, 3, -100, 1, 2]
        engine.begin(history)
        draft = engine.draft(history, n_past=5, id_last=2, n_max=2)
        assert draft.tolist() == [3]
    finally:
        engine.close()


@pytest.fixture(scope="module", params=["DFLASH", "DFLASH2"], ids=["dflash", "dflash2"])
def dflash_model(request):
    target = os.environ.get(f"LLAMA_TEST_{request.param}_MODEL")
    draft = os.environ.get(f"LLAMA_TEST_{request.param}_DRAFT")
    if not target or not draft:
        pytest.skip(f"Set LLAMA_TEST_{request.param}_MODEL and LLAMA_TEST_{request.param}_DRAFT")
    assert os.path.isfile(target) and os.path.isfile(draft)
    with closing(llama_cpp.Llama(
        model_path=target, n_ctx=256, n_batch=64, n_ubatch=64,
        n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0")),
        verbose=False, ctx_checkpoints=4,
        speculative=SpecConfig(spec_type=SpeculativeType.DRAFT_DFLASH,
                               draft_model_path=draft, draft_n_max=3, draft_n_min=3,
                               draft_n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0"))),
    )) as llm:
        assert llm.speculative.is_dflash2 == (request.param == "DFLASH2")
        yield llm


def test_real_dflash_block_limits_verification_and_interrupted_retry(dflash_model):
    llm = dflash_model
    engine = llm.speculative
    prompt = "Count from 1 to 20: 1,"
    tokens = llm.tokenize(prompt.encode())
    llm.reset()
    try:
        llm.eval(tokens[:-1], copy_logits=False)
        history = tokens[:-1]
        engine.begin(history)
        pos0 = llm._ctx.memory_seq_pos_max(0) + 1
        full = engine.draft(history, n_past=pos0, id_last=tokens[-1], n_max=3)
        assert len(full) == 3 and engine.noise_batch.n_tokens() == 4
        capped = engine.draft(history, n_past=pos0, id_last=tokens[-1], n_max=1)
        np.testing.assert_array_equal(capped, full[:1])
        assert engine.noise_batch.n_tokens() == 4
        assert engine.draft_context.memory_seq_pos_max(0) == pos0 - 1
        assert engine._pending_verification_checkpoint is not None

        def complete():
            return llm.create_completion(prompt, max_tokens=12, temperature=0, seed=42)["choices"][0]["text"]

        reference = complete()
        stats = llm.last_speculative_stats
        assert stats["drafted"] > 0 and stats["verified"] > 0
        assert stats["checkpoint_restores"] > 0
        stream = llm.create_completion(prompt, max_tokens=12, temperature=0, seed=42, stream=True)
        try:
            next(stream)
            next(stream)
        finally:
            stream.close()
        assert complete() == reference
        llm.reset()
        assert llm._ctx.memory_seq_pos_max(0) == engine.draft_context.memory_seq_pos_max(0) == -1
    finally:
        llm.reset()


@pytest.fixture(scope="module", params=[False, True], ids=["native", "device-fallback"])
def shared_mtp_model(request):
    path = os.environ.get("LLAMA_TEST_HYBRID_MODEL")
    if not path:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("LLAMA_TEST_HYBRID_MODEL is required in Actions")
        pytest.skip("Set LLAMA_TEST_HYBRID_MODEL to run real-model tests")
    assert os.path.isfile(path), path
    with closing(llama_cpp.Llama(
        model_path=path, n_ctx=256, n_batch=64, n_ubatch=64,
        n_gpu_layers=0, verbose=False, ctx_checkpoints=4,
        speculative=SpecConfig(spec_type=SpeculativeType.DRAFT_MTP, draft_n_max=3),
    )) as llm:
        assert llm.is_hybrid and llm._model.n_layer_nextn() > 0
        if request.param:
            llm._ctx.n_rs_seq = lambda: 0
            llm.speculative._use_native_draft_rollback = False
        yield llm, request.param


@pytest.fixture
def mtp_model(shared_mtp_model):
    llm, _ = shared_mtp_model
    llm._ctx.set_causal_attn(True)
    llm.reset()
    try:
        yield shared_mtp_model
    finally:
        llm._ctx.set_causal_attn(True)
        llm.reset()


def test_real_mtp_verification_and_interrupted_retry(mtp_model):
    llm, fallback = mtp_model
    def complete():
        return llm.create_completion("Count from 1 to 20: 1,", max_tokens=16,
                                     temperature=0, seed=42)["choices"][0]["text"]
    reference = complete()
    stats = llm.last_speculative_stats
    assert stats["drafted"] > 0 and stats["verified"] > 0
    assert stats["checkpoint_restores"] > 0
    if fallback:
        assert stats["checkpoint_device_captures"] > 0
    else:
        assert stats["checkpoint_native_captures"] > 0
    llm.reset()
    stream = llm.create_completion("Count from 1 to 20: 1,", stream=True,
                                   max_tokens=16, temperature=0, seed=42)
    try:
        next(stream)
        next(stream)
    finally:
        stream.close()
    assert complete() == reference
    llm.reset()
    assert llm._ctx.memory_seq_pos_max(0) == -1
    assert llm.speculative.draft_context.memory_seq_pos_max(0) == -1


def test_real_mtp_media_then_text_matches_native_catchup(mtp_model):
    llm, _ = mtp_model
    engine = llm.speculative
    checkpoint = engine.checkpoint()
    rng = np.random.default_rng(42)
    features = rng.normal(0, 0.1, (2, llm._model.n_embd_inp())).astype(np.float32)
    mrope = llm._model.rope_type() in (
        llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_MROPE,
        llama_cpp_lib.llama_rope_type.LLAMA_ROPE_TYPE_IMROPE,
    )
    coordinates = [[0, 1], [2, 3], [4, 5], [0, 0]]
    token = llm.tokenize(b"Hello", add_bos=True)[0]

    def continuation_output(hidden):
        with closing(internals.LlamaBatch(n_tokens=1, embd=engine.n_embd, n_seq_max=1, mixed=True)) as text:
            text.add_token_embedding(token, hidden, 2, [0], True)
            assert engine.draft_context.decode(text) == 0
            return (engine._copy_rows(engine.draft_context.get_embeddings_nextn_ith(0), 1, engine.n_embd),
                    engine._candidate(0))

    media_positions = np.array(coordinates).T.tolist() if mrope else coordinates[0]
    with _target_batch(media_positions, embeddings=features) as media:
        assert llm._ctx.decode(media) == 0
        target_h = engine._copy_rows(llm._ctx.get_embeddings_nextn(), 2, engine.n_embd)
        engine.process(media.batch)
    np.testing.assert_array_equal(engine.pending_h, target_h[-1])
    assert engine.draft_context.memory_seq_pos_max(0) == 1
    # Compare hidden outputs and candidates after identical catch-up inputs.
    engine.restore(checkpoint)
    native = llama_cpp_lib.llama_batch_ext_init(engine.draft_context.ctx)
    assert native
    try:
        hidden = np.vstack([engine.pending_h, target_h[0]])
        for i in range(2):
            embd = llama_cpp_lib.llama_embd(features[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, features.shape[1])
            state = llama_cpp_lib.llama_embd(hidden[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, engine.n_embd)
            assert llama_cpp_lib.llama_batch_ext_add_embd(native, 0, embd) == i
            assert llama_cpp_lib.llama_batch_ext_set_embd_state(native, i, state)
            pos = (llama_cpp_lib.llama_pos * 4)(*([axis[i] for axis in coordinates] if mrope else [i]))
            assert llama_cpp_lib.llama_batch_ext_set_pos(native, i, pos)
        assert llama_cpp_lib.llama_process(engine.draft_context.ctx, llama_cpp_lib.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, native) == 0
    finally:
        llama_cpp_lib.llama_batch_ext_free(native)
    expected_h, expected_candidate = continuation_output(target_h[-1])
    engine.restore(checkpoint)
    engine._process_target_rows([], [0, 1], target_h, embeddings=features,
                               media_positions=np.array(coordinates).T.tolist() if mrope else [0, 1])
    actual_h, actual_candidate = continuation_output(engine.pending_h)
    np.testing.assert_allclose(actual_h, expected_h, rtol=1e-5, atol=1e-5)
    assert actual_candidate[0] == expected_candidate[0]
    assert actual_candidate[1] == pytest.approx(expected_candidate[1], rel=1e-5, abs=1e-5)
    engine.restore(checkpoint)
    engine._process_target_rows([], [0, 1], target_h, embeddings=features,
                               media_positions=np.array(coordinates).T.tolist() if mrope else [0, 1])
    with _target_batch([2], tokens=[token], outputs=[True]) as text:
        assert llm._ctx.decode(text) == 0
        text_h = engine._copy_rows(llm._ctx.get_embeddings_nextn(), 1, engine.n_embd)
        engine.process(text.batch)
    assert engine.draft_context.memory_seq_pos_max(0) == 2
    assert engine.verify_embeddings is None
    np.testing.assert_array_equal(engine.pending_h, text_h[-1])



def test_real_mtp_image_prefill_process_and_draft(mtp_model):
    import io
    from PIL import Image
    from llama_cpp.llama_multimodal import MTMDBaseHandler

    mmproj = os.environ.get("LLAMA_TEST_MMPROJ")
    if not mmproj:
        pytest.skip("Set LLAMA_TEST_MMPROJ to run real image catch-up")
    llm, _ = mtp_model
    engine = llm.speculative
    image = io.BytesIO()
    Image.new("RGB", (112, 112), "red").save(image, format="PNG")
    with MTMDBaseHandler(mmproj_path=mmproj, image_min_tokens=16, image_max_tokens=32, verbose=False) as handler:
        with handler.tokenize_parts(llm, ["Look at this image: ", image.getvalue(), "\nDescribe the picture."]) as request:
            chunks = request.encode()
        media_rows = 0
        for chunk in chunks:
            count = len(chunk.tokens) if chunk.embeddings is None else len(chunk.embeddings)
            if chunk.use_non_causal:
                assert count <= llm._ctx.n_ubatch()
            llm._ctx.set_causal_attn(not chunk.use_non_causal)
            for start in range(0, count, llm.n_batch):
                end = min(start + llm.n_batch, count)
                positions = chunk.positions[start:end]
                outputs = [False] * (end - start - 1) + [True]
                features = None if chunk.embeddings is None else chunk.embeddings[start:end]
                with _target_batch(positions, tokens=chunk.tokens[start:end], embeddings=features, outputs=outputs) as batch:
                    assert llm._ctx.decode(batch) == 0
                    target_h = engine._copy_rows(llm._ctx.get_embeddings_nextn(), end - start, engine.n_embd)
                    engine.process(batch.batch)
                np.testing.assert_array_equal(engine.pending_h, target_h[-1])
                assert engine.draft_context.memory_seq_pos_max(0) == llm._ctx.memory_seq_pos_max(0)
                if chunk.embeddings is not None:
                    media_rows += end - start
                    np.testing.assert_array_equal(engine.verify_media_positions, positions)
        assert media_rows > 0
    llm._ctx.set_causal_attn(True)
    pos0 = llm._ctx.memory_seq_pos_max(0) + 1
    logits = engine._copy_rows(llm._ctx.get_logits_ith(-1), 1, llm.n_vocab())
    anchor = int(np.argmax(logits[0]))
    pending = engine.pending_h.copy()
    drafts = engine.draft([], n_past=pos0, id_last=anchor, n_max=3)
    assert 0 < len(drafts) <= 3 and np.all((drafts >= 0) & (drafts < llm.n_vocab()))
    np.testing.assert_array_equal(engine.pending_h, pending)
    assert engine.draft_context.memory_seq_pos_max(0) == pos0 - 1
    checkpoint = engine.take_verification_checkpoint()
    tokens = [anchor] + drafts.tolist()
    with _target_batch(range(pos0, pos0 + len(tokens)), tokens=tokens, outputs=[True] * len(tokens)) as batch:
        assert llm._ctx.decode(batch) == 0
        engine.process(batch.batch)
    assert llm._ctx.memory_seq_rm(0, pos0 + 1, -1)
    engine.rollback_verified(checkpoint, n_accepted=0)
    assert engine.draft_context.memory_seq_pos_max(0) == llm._ctx.memory_seq_pos_max(0) == pos0
