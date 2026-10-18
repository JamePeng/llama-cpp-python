"""Full snapshots, partial checkpoints, and their shared memory lifecycle."""

import ctypes
import os
from contextlib import closing
import gc
import pickle
import weakref
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from llama_cpp import Llama
from llama_cpp.llama import AbortCriteria, LlamaState, StoppingCriteriaList
from llama_cpp import llama_cpp as lib
from llama_cpp._internals import LlamaBatch, LlamaContext, LlamaSampler, LlamaSamplingContext
from llama_cpp.llama_cache import HybridCheckpoint, HybridCheckpointCache, LlamaDiskCache, LlamaRAMCache, LlamaTrieCache
from llama_cpp.llama_embedding import LlamaEmbedding


@pytest.mark.parametrize("verbose", [False, True])
def test_generation_keyboard_interrupt_discards_partial_logits_state(model, monkeypatch, capsys, verbose):
    llm, _ = model
    llm.verbose = verbose
    llm._model = object()
    llm._abort_event = threading.Event()
    llm._active_speculative_phase_stats = None
    llm._ctx.get_logits_ith = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr("llama_cpp.llama.LlamaSamplingContext", Mock())

    def interrupted_eval(tokens, **kwargs):
        llm.n_tokens += len(tokens)
        llm._ctx.get_logits_ith(-1)

    llm.eval = interrupted_eval
    assert list(llm.generate([1], reset=False)) == []
    assert llm._abort_event.is_set()
    assert llm.n_tokens == 0 and llm._last_eval_output_count == 0
    assert llm._restored_logits is None and llm._prefilled_prompt is None
    llm._ctx.memory_clear.assert_called_once_with(True)
    captured = capsys.readouterr()
    assert captured.out == ""
    if verbose:
        assert "KeyboardInterrupt" in captured.err
        assert "finish_reason=abort" in captured.err
    else:
        assert captured.err == ""


def _state_model(monkeypatch):
    llm = object.__new__(Llama)
    llm.verbose = False
    llm._n_ctx, llm._n_vocab = 8, 3
    llm._logits_all = False
    llm.n_tokens = 2
    llm.input_ids = np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=np.intc)
    llm.scores = np.array([[99, 0, 0]], dtype=np.single)  # stale Python copy
    llm._seed = 42
    llm.is_hybrid = False
    llm.speculative = None
    llm._sampling_ctx = None
    llm._last_eval_output_start = 0
    llm._last_eval_output_count = 2
    llm._restored_logits = None
    llm._state_compatibility = lambda: {"model": "same"}
    native_logits = np.array([0, 4, 1], dtype=np.single)
    llm._ctx = SimpleNamespace(
        ctx=object(), memory_clear=Mock(),
        set_state_data=lambda buf, size: lib.llama_state_set_data(llm._ctx.ctx, buf, size),
        get_logits_ith=lambda idx: native_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    monkeypatch.setattr(lib, "llama_state_get_size", lambda ctx: 2)

    def save(ctx, buf, size):
        buf[:] = b"ok"
        return 2

    monkeypatch.setattr(lib, "llama_state_get_data", save)
    monkeypatch.setattr(lib, "llama_state_set_data", lambda ctx, buf, size: size)
    return llm, native_logits


@pytest.fixture
def model(monkeypatch):
    return _state_model(monkeypatch)


@pytest.fixture
def generation_model(model, monkeypatch):
    llm, native_logits = model
    llm._model = object()
    llm._abort_event = threading.Event()
    llm._active_speculative_phase_stats = None
    llm.draft_model = None
    llm.eval_calls = []

    def evaluate(tokens, **kwargs):
        llm.eval_calls.append((list(tokens), kwargs))
        start = llm.n_tokens
        llm.input_ids[start:start + len(tokens)] = tokens
        llm.n_tokens += len(tokens)
        llm._last_eval_output_start = start
        llm._last_eval_output_count = len(tokens)
        llm._restored_logits = None
        if kwargs.get("copy_logits", True):
            llm.scores[0] = native_logits

    llm.eval = evaluate
    monkeypatch.setattr("llama_cpp.llama.LlamaSamplingContext", lambda *args: SimpleNamespace(
        sample=lambda *args, **kwargs: 1, accept=lambda *args: None, close=lambda: None,
    ))
    return llm, native_logits


@pytest.mark.parametrize("checkpoints", [0, 4])
@pytest.mark.parametrize("output", ["native", "owned", "missing", "wrong_range"])
def test_hybrid_full_match_requires_valid_output(generation_model, checkpoints, output):
    llm, _ = generation_model
    llm.is_hybrid = True
    llm._hybrid_cache_mgr = SimpleNamespace(
        max_checkpoints=checkpoints, find_best_checkpoint=Mock(return_value=None),
        save_checkpoint=Mock(return_value=False),
    )
    if output in ("owned", "missing"):
        state = llm.save_state()
        if output == "missing":
            state.last_logits = None
        llm.load_state(state)
    elif output == "wrong_range":
        llm._last_eval_output_start = llm.n_tokens
    with closing(llm.generate([1, 2])) as stream:
        assert next(stream) == 1
    reuse = output == "owned" or (output == "native" and checkpoints == 0)
    assert bool(llm.eval_calls) is not reuse
    if not reuse:
        assert [token for tokens, _ in llm.eval_calls for token in tokens] == [1, 2]
    assert llm.n_tokens == 2
    if checkpoints == 0:
        llm._hybrid_cache_mgr.save_checkpoint.assert_not_called()


@pytest.mark.parametrize("prompt,expected", [
    ([1, 2, 0], [[0]]), ([1, 0], [[1, 0]]), ([2, 1], [[2, 1]]), ([1], [[1]]),
])
def test_disabled_hybrid_checkpoints_append_or_rebuild(generation_model, prompt, expected):
    llm, _ = generation_model
    llm.is_hybrid = True
    cache = llm._hybrid_cache_mgr = HybridCheckpointCache(llm._ctx.ctx, max_checkpoints=0)
    cache.find_best_checkpoint = Mock(side_effect=AssertionError("Disabled"))
    llm.longest_token_prefix = Mock(wraps=Llama.longest_token_prefix)
    with closing(llm.generate(prompt)) as stream:
        assert next(stream) == 1
    assert [tokens for tokens, _ in llm.eval_calls] == expected
    assert llm._input_ids.tolist() == prompt
    assert cache.cache_size == 0
    llm.longest_token_prefix.assert_called_once()
    cache.find_best_checkpoint.assert_not_called()


@pytest.mark.parametrize("checkpoints,native_slots,requested,expected", [
    (0, 1, 8, 1), (0, 3, 8, 3), (0, 3, 1, 1), (4, 1, 8, 7),
])
def test_zero_checkpoint_draft_limit_respects_native_rollback_capacity(
    model, checkpoints, native_slots, requested, expected
):
    llm, _ = model
    llm.is_hybrid, llm.n_batch = True, 8
    llm.speculative = object()
    llm._hybrid_cache_mgr = SimpleNamespace(max_checkpoints=checkpoints)
    llm._ctx.n_rs_seq = lambda: native_slots
    assert llm._limit_speculative_draft_n_max(requested) == expected


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("logits_all", [False, True])
@pytest.mark.parametrize("hook", ["processor", "criteria"])
def test_disabled_hybrid_fast_path_refreshes_scores_for_hooks(
    generation_model, monkeypatch, restored, logits_all, hook
):
    llm, native_logits = generation_model
    llm._logits_all = logits_all
    if logits_all:
        llm.scores = np.full((llm._n_ctx, llm._n_vocab), 99, dtype=np.single)
    llm.is_hybrid = True
    llm._hybrid_cache_mgr = HybridCheckpointCache(llm._ctx.ctx, max_checkpoints=0)
    if restored:
        llm.load_state(llm.save_state())
    llm.scores.fill(99)
    seen = {"processor": [], "criteria": []}

    def processor(ids, scores):
        assert ids.tolist() == [1, 2]
        seen["processor"].append(scores.copy())
        return np.array([0, 0, 10], dtype=np.single)

    def stop(ids, scores):
        assert ids.tolist() == [1, 2]
        seen["criteria"].append(scores.copy())
        return False

    monkeypatch.setattr("llama_cpp.llama.CustomSampler", lambda callback: callback)

    def sampling_context(params, model):
        def sample(*args, **kwargs):
            if not params.custom_samplers:
                return 1
            data = [SimpleNamespace(id=i, logit=0.0) for i in range(3)]
            params.custom_samplers[0](SimpleNamespace(size=3, data=data))
            return max(data, key=lambda item: item.logit).id
        return SimpleNamespace(sample=sample, accept=lambda *args: None, close=lambda: None)

    monkeypatch.setattr("llama_cpp.llama.LlamaSamplingContext", sampling_context)
    hooks = ({"logits_processor": processor} if hook == "processor" else
             {"stopping_criteria": StoppingCriteriaList([stop])})
    with closing(llm.generate([1, 2], **hooks)) as stream:
        assert next(stream) == (2 if hook == "processor" else 1)
    assert llm.eval_calls == []
    np.testing.assert_array_equal(seen[hook], [native_logits])
    assert seen["criteria" if hook == "processor" else "processor"] == []


@pytest.mark.parametrize("needs_scores", [False, True])
def test_abort_only_criteria_do_not_copy_logits(generation_model, needs_scores):
    llm, _ = generation_model
    llm.n_tokens = 0
    criteria = StoppingCriteriaList([AbortCriteria(llm._abort_event)])
    if needs_scores:
        criteria.append(lambda ids, scores: False)
    with closing(llm.generate([1, 2], stopping_criteria=criteria)) as stream:
        assert next(stream) == 1
    assert llm.eval_calls[0][1]["copy_logits"] is needs_scores


@pytest.mark.parametrize("speculative", [False, True])
def test_final_hybrid_checkpoint_is_saved_only_for_reusable_prompt_history(
    generation_model, speculative
):
    llm, _ = generation_model
    llm.n_tokens, llm.n_batch = 0, 8
    llm.is_hybrid = True
    llm._hybrid_cache_mgr = SimpleNamespace(max_checkpoints=4, save_checkpoint=Mock())
    if speculative:
        llm._ctx.n_rs_seq = lambda: 3
        llm._ctx.memory_seq_pos_max = lambda seq: llm.n_tokens - 1
        llm.speculative = Mock()
        llm.speculative.draft_at_position.return_value = np.empty(0, dtype=np.intc)
        llm.speculative.checkpoint_stats.return_value = {}
    with closing(llm.generate([1])) as stream:
        assert next(stream) == 1
    assert llm._hybrid_cache_mgr.save_checkpoint.call_count == int(not speculative)


def test_disabled_periodic_checkpoints_do_not_prepare_histories(model):
    llm, _ = model

    class NoHistoryCopy(np.ndarray):
        def tolist(self):
            raise AssertionError("Disabled checkpoints must not materialize history")

    llm.n_tokens = 0
    llm.input_ids = np.zeros(128, dtype=np.intc).view(NoHistoryCopy)
    llm._n_ctx, llm._n_vocab, llm.n_batch = 128, 256, 8
    llm.is_hybrid = True
    llm.checkpoint_interval = 16
    llm._speculative_verifying = False
    llm._hybrid_cache_mgr = SimpleNamespace(max_checkpoints=0, save_checkpoint=Mock())
    llm._batch = SimpleNamespace(reset=lambda: None, add_sequence=lambda **kwargs: None)
    llm._ctx.clear_loras = llm._ctx.clear_cvec = lambda: None
    llm._decode_eval_batch = lambda tokens, size: size
    llm._process_speculative_batch = lambda: None
    llm.eval(list(range(64)), copy_logits=False)
    assert llm.n_tokens == 64
    llm._hybrid_cache_mgr.save_checkpoint.assert_not_called()


@pytest.mark.parametrize("native_slots,engine_support,allowed", [
    (0, False, False), (16, False, False), (0, True, False), (16, True, True),
])
def test_disabled_hybrid_speculation_checks_rollback_before_mutation(
    model, monkeypatch, native_slots, engine_support, allowed
):
    llm, _ = model
    llm._model = object()
    llm.is_hybrid = True
    llm._hybrid_cache_mgr = SimpleNamespace(max_checkpoints=0)
    llm._ctx.n_rs_seq = lambda: native_slots
    llm.speculative = SimpleNamespace(
        can_follow_target_native_rollback=lambda: engine_support, clear=Mock(),
    )
    llm._prefilled_prompt = None if allowed else (1, 2)
    llm.eval = Mock()

    class SamplingBoundary(Exception):
        pass

    monkeypatch.setattr("llama_cpp.llama.LlamaSamplingContext", Mock(side_effect=SamplingBoundary))
    error, message = (SamplingBoundary, None) if allowed else (RuntimeError, "ctx_checkpoints > 0")
    with pytest.raises(error, match=message):
        next(llm.generate([1, 2]))
    llm.eval.assert_not_called()
    assert llm._ctx.memory_clear.call_count == int(allowed)
    assert llm.n_tokens == (0 if allowed else 2)
    if not allowed:
        assert llm._prefilled_prompt == (1, 2)


@pytest.mark.parametrize("cache_type", [LlamaRAMCache, LlamaTrieCache, LlamaDiskCache])
@pytest.mark.parametrize("stream", [False, True])
def test_completion_only_restores_usable_states_and_saves_actual_history(
    monkeypatch, tmp_path, cache_type, stream
):
    for mode, should_load in (
        ("equal", False), ("better", True), ("speculative", False),
        ("hybrid_branch", False), ("hybrid_append", True),
        ("hybrid_exact_no_logits", False), ("hybrid_exact_owned", True),
    ):
        # Fresh models and patch scopes keep all 42 combinations independent.
        with monkeypatch.context() as patch:
            llm, _ = _state_model(patch)
            llm._model = SimpleNamespace(
                vocab=object(), token_bos=lambda: 1, token_eos=lambda: 2,
                token_sep=lambda: -1, token_fim_pre=lambda: -1,
                token_fim_mid=lambda: -1, token_fim_suf=lambda: -1, get_add_sep=lambda: False,
            )
            llm._abort_event = threading.Event()
            llm.metadata = {}
            llm.spm_infill = False
            llm.model_path = "fake.gguf"
            llm.is_hybrid = mode.startswith("hybrid")
            cached_tokens = [1, 2] if mode == "hybrid_append" else [1, 2, 0]
            llm.n_tokens = len(cached_tokens)
            llm.input_ids[:llm.n_tokens] = cached_tokens
            llm._last_eval_output_start, llm._last_eval_output_count = 0, llm.n_tokens
            cached = llm.save_state()
            if mode == "hybrid_exact_no_logits":
                cached.last_logits = None
            cache = LlamaDiskCache(str(tmp_path / mode)) if cache_type is LlamaDiskCache else cache_type()
            llm.cache = cache
            cache[cached_tokens] = cached
            llm.n_tokens = 2 if mode == "equal" else 0
            llm.speculative = object() if mode == "speculative" else None
            prompt = [1, 2, 1] if mode in ("equal", "hybrid_branch") else [1, 2, 0]
            load = Mock(wraps=llm.load_state)
            save = Mock(wraps=llm.save_state)
            patch.setattr(llm, "load_state", load)
            patch.setattr(llm, "save_state", save)
            reads = []
            if cache_type is LlamaDiskCache:
                original_read = type(cache.cache).__getitem__

                def read(instance, key):
                    reads.append(key)
                    return original_read(instance, key)

                patch.setattr(type(cache.cache), "__getitem__", read)

            def generate(tokens, **kwargs):
                llm.n_tokens = len(tokens)
                llm.input_ids[:llm.n_tokens] = tokens
                llm._last_eval_output_start, llm._last_eval_output_count = llm.n_tokens - 1, 1
                yield 1  # This sampled token has not been decoded into native state.

            llm.generate = generate
            llm.detokenize = lambda tokens, **kwargs: b"x" * len(tokens)
            patch.setattr(lib, "llama_token_is_eog", lambda *args: False)
            try:
                list(llm._create_completion(prompt=prompt, max_tokens=1, stream=stream))
                assert load.call_count == int(should_load), mode
                assert save.call_count == int(mode != "speculative"), mode
                if mode in ("equal", "speculative"):
                    assert reads == [], mode
                if mode != "speculative":
                    assert cache._find_longest_prefix_key(tuple(prompt)) == tuple(prompt), mode
                    assert cache[prompt].n_tokens == len(prompt), mode
            finally:
                if cache_type is LlamaDiskCache:
                    cache.cache.close()


def test_state_owns_native_logits_not_stale_scores_and_survives_pickle(model):
    llm, logits = model
    state = pickle.loads(pickle.dumps(llm.save_state()))
    logits[:] = [8, 0, 0]
    llm.input_ids[:] = 0
    llm.load_state(state)
    assert llm.n_tokens == 2
    assert len(llm.input_ids) == llm._n_ctx
    np.testing.assert_array_equal(llm.input_ids[:2], [1, 2])
    sampler = SimpleNamespace(sample=lambda ctx, **kw: int(kw["logits"].argmax()))
    assert llm._sample_output(sampler, 1) == 1
    np.testing.assert_array_equal(llm.scores, [[0, 4, 1]])
    llm._restored_logits[1] = 0
    assert state.last_logits[1] == 4
    assert state.input_ids.shape == (2,)
    refs = [weakref.ref(a) for a in (state.input_ids, state.scores, state.last_logits)]
    del state
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("request_kind", ["new", "continue", "media_prefill"])
def test_generation_request_boundary_preserves_prefill_and_clears_once(
    model, monkeypatch, speculative, request_kind
):
    llm, _ = model
    llm._model = object()
    llm.speculative = Mock() if speculative else None
    tokens = [2, 1]  # No prefix match against the existing [1, 2].
    if request_kind == "media_prefill":
        tokens = [1, -9]
        llm.input_ids[:2] = tokens
        llm._prefilled_prompt = tuple(tokens)
        llm._restored_logits = np.array([0, 4, 1], dtype=np.single)

    class SamplingBoundary(Exception):
        pass

    # Stop at sampler construction: request cleanup must be complete before
    # sampling starts, and must not discard an already decoded media prompt.
    monkeypatch.setattr(
        "llama_cpp.llama.LlamaSamplingContext", Mock(side_effect=SamplingBoundary)
    )
    with pytest.raises(SamplingBoundary):
        next(llm.generate(tokens, reset=request_kind != "continue"))
    expected_clears = int(request_kind == "new")
    assert llm._ctx.memory_clear.call_count == expected_clears
    assert llm.n_tokens == (0 if expected_clears else 2)
    if speculative:
        assert llm.speculative.clear.call_count == expected_clears
    if request_kind == "media_prefill":
        np.testing.assert_array_equal(llm._restored_logits, [0, 4, 1])
        assert llm._last_eval_output_count == 2
        assert llm._prefilled_prompt is None  # Handoff is consumed exactly once.


@pytest.mark.parametrize("cls", [Llama, LlamaEmbedding])
@pytest.mark.parametrize("failure", [None, "decode", "output"])
def test_embedding_request_cleans_generation_state_and_preserves_result_order(model, monkeypatch, cls, failure):
    llm, _ = model
    llm.__class__ = cls
    llm.context_params = SimpleNamespace(embeddings=True, n_seq_max=2, no_perf=True)
    llm.n_batch = 8
    llm.pooling_type = lambda: lib.LLAMA_POOLING_TYPE_MEAN
    llm.n_embd_out = lambda: 2
    llm._ctx.get_causal_attn = lambda: True
    llm._model = SimpleNamespace(model=object())
    llm._batch = SimpleNamespace(reset=Mock(), n_tokens=lambda: 0, add_sequence=Mock())
    llm._ctx.decode = Mock(return_value=1 if failure == "decode" else 0)
    output = Mock(return_value=[3.0, 4.0])
    if failure == "output":
        output.side_effect = RuntimeError("output unavailable")
    monkeypatch.setattr(lib, "llama_get_embeddings_seq", output)
    if failure:
        with pytest.raises(RuntimeError):
            llm.embed([[1], [], [2]], normalize=2, return_count=True)
        if failure == "decode":
            output.assert_not_called()
    else:
        values, count = llm.embed([[1], [], [2]], normalize=2, return_count=True)
        assert count == 2 and values[1] == []
        np.testing.assert_allclose([values[0], values[2]], [[0.6, 0.8]] * 2)
    assert llm.n_tokens == 0 and llm._last_eval_output_count == 0
    assert llm._restored_logits is None and llm._prefilled_prompt is None
    assert llm._ctx.memory_clear.call_count >= 2
    # An exception must not poison the next embedding request on either API.
    llm._ctx.decode.return_value = 0
    output.side_effect = None
    values, count = llm.embed([[2], [], [1]], normalize=2, return_count=True)
    assert count == 2 and values[1] == []
    np.testing.assert_allclose([values[0], values[2]], [[0.6, 0.8]] * 2)
    assert llm.n_tokens == 0 and llm._restored_logits is None


@pytest.mark.parametrize("entry", ["batch", "generate"])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("checkpoints", [0, 4])
def test_fatal_decode_cleanup_allows_retry(model, monkeypatch, entry, speculative, checkpoints):
    llm, _ = model
    llm.is_hybrid = True
    llm._hybrid_cache_mgr = SimpleNamespace(max_checkpoints=checkpoints, save_checkpoint=Mock())
    llm._model = object()
    llm._abort_event = threading.Event()
    llm._active_speculative_phase_stats = None
    llm._batch = SimpleNamespace(batch=SimpleNamespace(n_tokens=0))
    llm.speculative = Mock() if speculative else None
    if speculative:
        llm.speculative.checkpoint_stats.return_value = {}
    llm._ctx.synchronize = Mock()
    llm._ctx.n_rs_seq = lambda: 16
    llm._ctx.decode = Mock(side_effect=RuntimeError("partial native decode"))
    monkeypatch.setattr("llama_cpp.llama.LlamaSamplingContext", Mock())

    def decode(tokens, **kwargs):
        return llm._decode_eval_batch(tokens, len(tokens))

    llm.eval = decode
    with pytest.raises(RuntimeError, match="partial native decode"):
        if entry == "batch":
            llm.eval([1])
        else:
            next(llm.generate([1, 2], reset=False))
    assert llm.n_tokens == 0 and llm._last_eval_output_count == 0
    assert llm._restored_logits is None and llm._prefilled_prompt is None
    assert llm._ctx.memory_clear.called
    llm._hybrid_cache_mgr.save_checkpoint.assert_not_called()
    if speculative:
        assert llm.speculative.clear.called
    llm._ctx.decode.side_effect = None
    llm._ctx.decode.return_value = 0
    assert llm.eval([2]) == 1


@pytest.mark.parametrize("damage", ["size", "identity", "tokens", "logits"])
def test_invalid_state_rejected_before_native_mutation(model, monkeypatch, damage):
    llm, _ = model
    state = llm.save_state()
    if damage == "size":
        state.llama_state_size += 1
    elif damage == "identity":
        state.compatibility = {"model": "different"}
    elif damage == "tokens":
        state.n_tokens = 99
    else:
        state.last_logits = np.zeros(1)
    setter = Mock()
    monkeypatch.setattr(lib, "llama_state_set_data", setter)
    with pytest.raises(ValueError):
        llm.load_state(state)
    setter.assert_not_called()
    assert llm.n_tokens == 2


def test_native_load_failure_clears_both_sides(model, monkeypatch):
    llm, _ = model
    state = llm.save_state()
    monkeypatch.setattr(lib, "llama_state_set_data", lambda *args: 0)
    with pytest.raises(RuntimeError, match="Failed to set"):
        llm.load_state(state)
    assert llm.n_tokens == 0 and llm._last_eval_output_count == 0
    llm._ctx.memory_clear.assert_called_once_with(True)


@pytest.mark.parametrize("kind", ["legacy", "memory_only", "empty"])
def test_state_without_output_cannot_sample(model, kind):
    llm, _ = model
    if kind == "empty":
        llm.reset()
    elif kind == "memory_only":
        llm._last_eval_output_count = 0
    state = llm.save_state()
    if kind == "legacy":
        state = LlamaState(state.input_ids, state.scores, 2, b"ok", 2, 42)
        del state.last_logits  # old pickle predates the new fields
        del state.compatibility
    else:
        assert state.last_logits is None
    state = pickle.loads(pickle.dumps(state))
    llm.load_state(state)
    assert llm._last_eval_output_count == 0
    with pytest.raises(RuntimeError, match="unavailable"):
        llm._sample_output(Mock(), llm.n_tokens - 1)


@pytest.mark.parametrize("alias_live_scores", [False, True])
def test_logits_all_reuses_buffer_and_restores_history(model, alias_live_scores):
    llm, _ = model
    llm._logits_all = True
    llm.scores = np.full((8, 3), 99, dtype=np.single)
    llm.scores[0] = [1, 2, 3]
    state = llm.save_state()
    assert state.scores.shape == (2, 3)
    allocation = llm.scores
    if alias_live_scores:
        state.scores = allocation
    llm.load_state(state)
    assert llm.scores is allocation
    np.testing.assert_array_equal(llm.scores[:2], [[1, 2, 3], [0, 4, 1]])
    assert not llm.scores[2:].any()


def test_load_borrows_immutable_native_bytes_with_embedded_nul(model, monkeypatch):
    llm, _ = model
    state = llm.save_state()
    state.llama_state = b"o\x00"
    expected_address = ctypes.cast(ctypes.c_char_p(state.llama_state), ctypes.c_void_p).value

    def restore(ctx, pointer, size):
        assert ctypes.cast(pointer, ctypes.c_void_p).value == expected_address
        assert ctypes.string_at(pointer, size) == state.llama_state
        return size

    monkeypatch.setattr(lib, "llama_state_set_data", restore)
    llm.load_state(state)


@pytest.mark.parametrize("cache_type", [LlamaRAMCache, LlamaTrieCache])
def test_cache_accounts_for_arrays_and_releases_evicted_snapshot(model, cache_type):
    llm, _ = model
    state = llm.save_state()
    ref = weakref.ref(state)
    cache = cache_type(capacity_bytes=state.nbytes)
    cache[[1, 2]] = state
    payload_size = state.nbytes
    assert cache.cache_size == payload_size > state.llama_state_size
    del state
    cache[[3, 4]] = llm.save_state()
    gc.collect()
    assert ref() is None
    assert cache.cache_size == payload_size
    assert [1, 2] not in cache and [3, 4] in cache


@pytest.mark.parametrize("shared_cache", [False, True])
def test_model_close_releases_owned_memory_and_detaches_cache(model, shared_cache):
    llm, _ = model
    state = llm.save_state()
    llm.load_state(state)
    output = weakref.ref(llm._restored_logits)
    snapshot = weakref.ref(state)
    cache = LlamaRAMCache()
    cache[[1, 2]] = state
    llm.set_cache(cache)
    del state
    if not shared_cache:
        del cache
    llm._prefilled_prompt = (1, 2)
    llm.close()
    llm.close()
    gc.collect()
    assert output() is None
    assert llm._prefilled_prompt is None and llm.cache is None
    if shared_cache:
        assert snapshot() is not None and cache[[1, 2]] is snapshot()
    else:
        assert snapshot() is None


def test_close_attempts_all_resources_when_one_fails(model):
    llm, _ = model
    events = []

    def fail():
        events.append("sampler")
        raise RuntimeError("close failed")

    llm._sampling_ctx = SimpleNamespace(close=fail)
    llm.speculative = SimpleNamespace(close=lambda: events.append("draft"))
    llm.chat_handler = SimpleNamespace(close=lambda: events.append("media"))
    llm._stack = SimpleNamespace(close=lambda: events.append("context-model"))
    llm._restored_logits = np.zeros(3)
    output = weakref.ref(llm._restored_logits)
    with pytest.raises(RuntimeError, match="close failed"):
        llm.close()
    assert events == ["sampler", "draft", "media", "context-model"]
    assert output() is None
    assert llm._stack is None and llm.speculative is None
    llm.close()


def test_save_uses_committed_output_row_after_verification_truncation(model):
    llm, _ = model
    rows = np.array([[0, 8, 0], [9, 0, 0]], dtype=np.single)
    llm.n_tokens = 1
    llm._last_eval_output_count = 1
    llm._ctx.get_logits_ith = lambda idx: rows[idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    assert llm.save_state().last_logits.argmax() == 1


def test_speculative_state_requires_new_request(model):
    llm, _ = model
    state = llm.save_state()
    llm.speculative = Mock()
    llm.load_state(state)
    with pytest.raises(RuntimeError, match="draft state"):
        next(llm.generate([], reset=False))
    llm._speculative_verifying = True
    with pytest.raises(RuntimeError, match="verification"):
        llm.save_state()


def test_prefill_handoff_owns_final_sparse_output(model):
    llm, logits = model
    llm.input_ids[0] = -100
    llm._mark_prefilled_prompt()
    logits[:] = 0
    assert llm._prefilled_prompt == (-100, 2)
    assert llm._last_eval_output_start == 1
    assert llm._restored_logits.argmax() == 1
    llm.reset()
    assert llm._prefilled_prompt is None and llm._restored_logits is None


def test_owned_logits_bypass_backend_sampled_token_and_pointer():
    # Use a real native greedy sampler, without allocating a model/context.
    lib.llama_backend_init()
    chain = LlamaSampler()
    chain.add_greedy()
    ctx = SimpleNamespace(get_sampled_token_ith=Mock(side_effect=AssertionError),
                          get_logits_ith=Mock(side_effect=AssertionError))
    sampler = SimpleNamespace(n_vocab=3, grammar_sampler=None, _cur_p=None,
                              params=SimpleNamespace(logit_bias=[]), sampler_chain=chain)
    try:
        assert LlamaSamplingContext.sample(sampler, ctx, logits=np.array([0, 1, 9], dtype=np.single)) == 2
    finally:
        chain.close()
    ctx.get_sampled_token_ith.assert_not_called()
    ctx.get_logits_ith.assert_not_called()


# Partial checkpoints and native memory mutations.

@pytest.fixture
def tracked_cache(monkeypatch):
    native = {}
    for name in (
        "get_memory", "memory_clear", "memory_seq_rm", "memory_seq_cp",
        "memory_seq_keep", "memory_seq_add", "memory_seq_div", "free", "decode",
        "state_set_data", "state_load_file", "state_seq_set_data",
        "state_seq_load_file", "state_seq_set_data_ext", "state_seq_get_data_ext",
    ):
        native[name] = Mock(return_value=1)
        monkeypatch.setattr(lib, "llama_" + name, native[name])
    context = LlamaContext.__new__(LlamaContext)
    context.ctx = object()
    cache = HybridCheckpointCache(context.ctx)
    cache.checkpoints = [
        HybridCheckpoint(3, b"x", "early", 1, 0, 0, 2),
        HybridCheckpoint(9, b"xx", "late", 2, 0, 0, 8),
        HybridCheckpoint(3, b"xxx", "other", 3, 1, 0, 2),
    ]
    cache._current_size = 6
    context._register_checkpoint_cache(cache)
    yield context, cache, native
    context.close()


@pytest.mark.parametrize("operation", ["reset", "load"])
def test_llama_invalidates_registered_checkpoints_once(model, tracked_cache, operation):
    llm, _ = model
    state = llm.save_state()
    context, cache, native = tracked_cache
    old_checkpoints = list(cache.checkpoints)
    llm._ctx = context
    llm.is_hybrid = True
    llm._hybrid_cache_mgr = cache
    llm.speculative = Mock()
    cache.clear = Mock(wraps=cache.clear)
    cache._invalidate_memory = Mock(wraps=cache._invalidate_memory)
    native["state_set_data"].return_value = state.llama_state_size
    if operation == "reset":
        llm.reset()
    else:
        llm.load_state(state)
        assert llm.n_tokens == state.n_tokens
        np.testing.assert_array_equal(llm._restored_logits, state.last_logits)
    assert cache.cache_size == 0 and not cache.checkpoints
    for checkpoint in old_checkpoints:
        assert not cache.restore_checkpoint(checkpoint, checkpoint.seq_id)
    native["state_seq_set_data_ext"].assert_not_called()
    cache._invalidate_memory.assert_called_once()
    cache.clear.assert_not_called()  # Native wrapper owns invalidation.
    llm.speculative.clear.assert_called_once()


@pytest.mark.parametrize("method,args,kept", [
    ("memory_clear", (True,), []),
    ("memory_seq_rm", (0, 5, -1), [0, 2]),
    ("memory_seq_rm", (0, 1, 2), [2]),
    ("memory_seq_rm", (-1, 0, -1), []),
    ("memory_seq_rm", (0, 2, 2), [0, 1, 2]),
    ("memory_seq_cp", (0, 1, 0, -1), [0, 1]),
    ("memory_seq_keep", (0,), [0, 1]),
    ("memory_seq_add", (0, 0, -1, -2), [2]),
    ("memory_seq_add", (0, 0, -1, 0), [0, 1, 2]),
    ("memory_seq_div", (0, 0, -1, 2), [2]),
    ("set_state_data", (None, 0), []),
    ("load_state_file", (b"file", None, 0, None), []),
    ("set_state_seq_data", (None, 0, 0), [2]),
    ("load_state_seq_file", (b"file", 0, None, 0, None), [2]),
    ("set_state_seq_data_ext", (None, 0, 0, 1), [2]),
])
def test_context_mutations_invalidate_only_affected_checkpoints(tracked_cache, method, args, kept):
    context, cache, native = tracked_cache
    before = list(cache.checkpoints)
    getattr(context, method)(*args)
    assert cache.checkpoints == [before[i] for i in kept]
    assert cache.cache_size == sum(before[i].size for i in kept)
    for i, checkpoint in enumerate(before):
        if i not in kept:
            assert not cache.restore_checkpoint(checkpoint, checkpoint.seq_id)
    if method != "set_state_seq_data_ext":
        native["state_seq_set_data_ext"].assert_not_called()


@pytest.mark.parametrize("operation,result,remaining", [
    ("remove", False, 3), ("remove", RuntimeError("native"), 1),
    ("restore", 0, 1), ("restore", RuntimeError("native"), 1),
    ("decode", 0, 3), ("decode", 1, 3),
    ("decode", 2, 0), ("decode", -3, 0),
    ("decode", RuntimeError("native"), 0),
])
def test_checkpoint_invalidation_respects_failure_contract(tracked_cache, operation, result, remaining):
    context, cache, native = tracked_cache
    name, call = {
        "remove": ("memory_seq_rm", lambda: context.memory_seq_rm(0, 5, -1)),
        "restore": ("state_seq_set_data", lambda: context.set_state_seq_data(None, 0, 0)),
        "decode": ("decode", lambda: context.decode(SimpleNamespace(batch=object()))),
    }[operation]
    if isinstance(result, Exception):
        native[name].side_effect = result
    else:
        native[name].return_value = result
    if isinstance(result, Exception) or (operation == "decode" and result not in (0, 1)):
        with pytest.raises(RuntimeError):
            call()
    else:
        call()
    assert len(cache.checkpoints) == remaining


@pytest.mark.parametrize("on_device", [False, True])
def test_device_capture_invalidates_old_slot_even_on_failure(tracked_cache, on_device):
    context, cache, native = tracked_cache
    cache.on_device = on_device
    native["state_seq_get_data_ext"].side_effect = RuntimeError("capture failed")
    with pytest.raises(RuntimeError):
        context.get_state_seq_data_ext(None, 0, 0, lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)
    assert len(cache.checkpoints) == (1 if on_device else 3)


def test_context_checkpoint_registration_does_not_own_cache_and_close_rejects_restore(tracked_cache):
    context, cache, native = tracked_cache
    temporary = HybridCheckpointCache(context.ctx)
    context._register_checkpoint_cache(temporary)
    reference = weakref.ref(temporary)
    del temporary
    gc.collect()
    assert reference() is None
    checkpoint = cache.checkpoints[0]
    context.close()
    context.close()
    assert not cache.restore_checkpoint(checkpoint)
    assert cache.cache_size == 0 and cache._ctx is None
    assert context.model is None and context.params is None
    native["free"].assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        context.memory_seq_rm(0, 0, -1)
    with pytest.raises(RuntimeError, match="closed"):
        context.set_state_data(None, 0)


@pytest.mark.parametrize("failure", [None, "short_read", "suffix", "exception"])
def test_checkpoint_restore_transaction_updates_all_registered_caches(tracked_cache, failure):
    context, cache, native = tracked_cache
    peer = HybridCheckpointCache(context.ctx)
    peer.checkpoints = list(cache.checkpoints)
    context._register_checkpoint_cache(peer)
    checkpoint, future, other = cache.checkpoints
    cache._get_size_ext = Mock(return_value=checkpoint.size)
    if failure == "short_read":
        native["state_seq_set_data_ext"].return_value = 0
    elif failure == "suffix":
        native["memory_seq_rm"].return_value = False
    elif failure == "exception":
        native["memory_seq_rm"].side_effect = RuntimeError("partial restore")
    if failure == "exception":
        with pytest.raises(RuntimeError):
            cache.restore_checkpoint(checkpoint)
    else:
        assert cache.restore_checkpoint(checkpoint) is (failure is None)
    expected = [checkpoint, other] if failure is None else [other]
    assert cache.checkpoints == peer.checkpoints == expected
    assert cache.cache_size == peer.cache_size == sum(cp.size for cp in expected)
    assert not cache.restore_checkpoint(future)


def test_checkpoint_restore_borrows_bytes_and_accepts_changed_live_size(tracked_cache):
    context, cache, native = tracked_cache
    cp = cache.checkpoints[0]
    cp.data, cp.size = b"a\0b", 3
    cache._get_size_ext = Mock(side_effect=AssertionError("Live export size is not a restore constraint"))
    expected_address = ctypes.cast(ctypes.c_char_p(cp.data), ctypes.c_void_p).value

    def restore(ctx, buffer, size, seq_id, flags):
        assert ctx is context.ctx and seq_id == cp.seq_id
        assert size == 3 and ctypes.string_at(buffer, size) == b"a\0b"
        assert ctypes.cast(buffer, ctypes.c_void_p).value == expected_address
        return size

    native["state_seq_set_data_ext"].side_effect = restore
    assert cache.restore_checkpoint(cp)
    cache._get_size_ext.assert_not_called()
    native["memory_seq_rm"].assert_called_once_with(native["get_memory"].return_value, 0, 3, -1)


@pytest.mark.parametrize("size", [0, 2])
def test_checkpoint_invalid_payload_size_is_rejected_before_native_restore(tracked_cache, size):
    _, cache, native = tracked_cache
    cp = cache.checkpoints[0]
    cp.size = size
    assert not cache.restore_checkpoint(cp)
    native["state_seq_set_data_ext"].assert_not_called()
    native["memory_seq_rm"].assert_not_called()


def test_duplicate_host_checkpoint_replaces_cursor_after_success(tracked_cache):
    _, cache, native = tracked_cache
    cache._get_size_ext = Mock(return_value=1)
    cache._get_memory = lambda ctx: object()
    cache._memory_seq_pos_min = lambda *args: 2
    cache._memory_seq_pos_max = lambda *args: 2
    previous = cache.checkpoints[0]
    for _ in range(2):
        assert cache.save_checkpoint(3, [1, 2, 3])
        assert len(cache.checkpoints) == 3 and cache.cache_size == 6
    assert not cache.restore_checkpoint(previous)
    retained = list(cache.checkpoints)
    native["state_seq_get_data_ext"].return_value = 0
    assert not cache.save_checkpoint(3, [1, 2, 3])
    assert cache.checkpoints == retained and cache.cache_size == 6


def test_hybrid_checkpoint_preserves_native_positions_and_removes_suffix(monkeypatch):
    state = bytes([1, 2, 3, 4])
    memory = object()
    removals = []

    monkeypatch.setattr(lib, "llama_state_seq_get_size_ext", lambda ctx, seq_id, flags: len(state))

    def get_data(ctx, buffer, size, seq_id, flags):
        buffer[:size] = state
        return size

    monkeypatch.setattr(lib, "llama_state_seq_get_data_ext", get_data)
    monkeypatch.setattr(lib, "llama_state_seq_set_data_ext", lambda ctx, buffer, size, seq_id, flags: size)
    monkeypatch.setattr(lib, "llama_get_memory", lambda ctx: memory)
    monkeypatch.setattr(lib, "llama_memory_seq_pos_min", lambda mem, seq_id: 11)
    monkeypatch.setattr(lib, "llama_memory_seq_pos_max", lambda mem, seq_id: 42)

    def memory_seq_rm(mem, seq_id, p0, p1):
        removals.append((mem, seq_id, p0, p1))
        return True

    monkeypatch.setattr(lib, "llama_memory_seq_rm", memory_seq_rm)

    cache = HybridCheckpointCache(ctx=object())
    assert cache.save_checkpoint(current_pos=17, tokens=list(range(17)), seq_id=3)

    checkpoint = cache.checkpoints[0]
    assert checkpoint.pos == 17
    assert checkpoint.pos_min == 11
    assert checkpoint.pos_max == 42

    assert cache.restore_checkpoint(checkpoint, seq_id=3)
    assert removals == [(memory, 3, 43, -1)]


def _native_greedy_reference(llm, context, batch, chunks, max_tokens):
    """Decode fixed batches in a separate context without Python prefix reuse."""
    context.memory_clear(True)
    position = 0
    prompt = []
    for chunk in chunks:
        if not chunk:
            continue
        batch.reset()
        outputs = [True] * len(chunk) if llm._logits_all else [False] * (len(chunk) - 1) + [True]
        batch.add_sequence(chunk, list(range(position, position + len(chunk))), [0], outputs)
        assert context.decode(batch) == 0
        position += len(chunk)
        prompt.extend(chunk)
    logits = np.ctypeslib.as_array(context.get_logits_ith(-1), shape=(llm._n_vocab,)).copy()
    pos_min, pos_max = context.memory_seq_pos_min(0), context.memory_seq_pos_max(0)
    generated = []
    with closing(LlamaSampler()) as sampler:
        sampler.add_greedy()
        for index in range(max_tokens):
            token = sampler.sample(context, -1)
            if lib.llama_token_is_eog(llm._model.vocab, token):
                break
            generated.append(token)
            sampler.accept(token)
            if index + 1 < max_tokens:
                batch.reset()
                batch.add_sequence([token], [position], [0], [True])
                assert context.decode(batch) == 0
                position += 1
    return SimpleNamespace(
        logits=logits, pos_min=pos_min, pos_max=pos_max,
        text=llm.detokenize(generated, prev_tokens=prompt).decode("utf-8", errors="ignore"),
    )


@pytest.fixture(scope="module", params=[False, True], ids=["host", "device"])
def hybrid_model(request):
    path = os.environ.get("LLAMA_TEST_HYBRID_MODEL")
    if not path:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("LLAMA_TEST_HYBRID_MODEL is required in Actions")
        pytest.skip("Set LLAMA_TEST_HYBRID_MODEL to run real-model tests")
    assert os.path.isfile(path), path
    with closing(Llama(model_path=path, n_ctx=256, n_batch=64, n_ubatch=64,
               n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0")),
               verbose=False, ctx_checkpoints=4,
               checkpoint_interval=32, checkpoint_on_device=request.param)) as llm:
        assert llm.is_hybrid
        yield llm


@pytest.mark.parametrize("rollback", [False, True], ids=["append", "rollback"])
def test_real_hybrid_prefix_restore_and_snapshot(hybrid_model, monkeypatch, rollback):
    llm = hybrid_model
    tokens = llm.tokenize(b"The capital of France is")
    def complete():
        return llm.create_completion(tokens, max_tokens=4, temperature=0,
                                     seed=42)["choices"][0]["text"]
    llm.reset()
    reference = complete()
    llm.reset()
    llm.eval(tokens[:-1])
    cache = llm._hybrid_cache_mgr
    assert cache.save_checkpoint(llm.n_tokens, tokens[:-1])
    checkpoint = cache.checkpoints[-1]
    if rollback:
        llm.eval(tokens[-1:])
    restore = Mock(wraps=cache.restore_checkpoint)
    monkeypatch.setattr(cache, "restore_checkpoint", restore)
    assert complete() == reference
    assert restore.called == rollback
    state = pickle.loads(pickle.dumps(llm.save_state()))
    llm.eval([tokens[-1]])
    llm.load_state(state)
    assert llm.n_tokens == state.n_tokens
    np.testing.assert_array_equal(llm._restored_logits, state.last_logits)
    assert not cache.restore_checkpoint(checkpoint)
    llm.reset()
    assert not cache.checkpoints and cache.cache_size == 0


@pytest.mark.parametrize("missing_output", [False, True])
def test_real_hybrid_zero_checkpoints_reuses_or_rebuilds_output(
    hybrid_model, monkeypatch, missing_output
):
    llm = hybrid_model
    monkeypatch.setattr(llm, "ctx_checkpoints", 0)
    monkeypatch.setattr(llm._hybrid_cache_mgr, "max_checkpoints", 0)
    capture = Mock(wraps=llm._hybrid_cache_mgr._get_data_ext)
    monkeypatch.setattr(llm._hybrid_cache_mgr, "_get_data_ext", capture)
    tokens = llm.tokenize(b"The capital of France is")
    llm.reset()
    reference = llm.create_completion(tokens, max_tokens=1, temperature=0, seed=42)
    llm.reset()
    llm.eval(tokens, copy_logits=False)
    expected = np.ctypeslib.as_array(llm._ctx.get_logits_ith(-1), shape=(llm._n_vocab,)).copy()
    if missing_output:
        state = llm.save_state()
        state.last_logits = None
        llm.load_state(state)
    llm.scores.fill(99)
    evaluate = Mock(wraps=llm.eval)
    monkeypatch.setattr(llm, "eval", evaluate)
    seen = []

    def observe(ids, scores):
        seen.append(scores.copy())
        return False

    result = llm.create_completion(
        tokens, max_tokens=1, temperature=0, seed=42,
        stopping_criteria=StoppingCriteriaList([observe]),
    )
    assert result["choices"][0]["text"] == reference["choices"][0]["text"]
    assert bool(evaluate.call_count) is missing_output
    np.testing.assert_allclose(seen[0], expected, rtol=1e-4, atol=1e-4)
    capture.assert_not_called()
    llm.reset()


@pytest.fixture(scope="module")
def hybrid_completion_cases(hybrid_model):
    llm = hybrid_model
    tokens = llm.tokenize(b"The capital of France is")
    prompts = {
        "exact": tokens,
        "append": tokens + llm.tokenize(b" and Italy is", add_bos=False),
        "branch": tokens[:-1] + llm.tokenize(b" was", add_bos=False),
    }
    suffix = prompts["append"][len(tokens):]
    batches = {
        "exact": (tokens,),
        "append": (tokens, suffix[:-1], suffix[-1:]),
        "branch": (prompts["branch"][:-1], prompts["branch"][-1:]),
    }
    references = {}
    llm.reset()
    try:
        # Match native prefill shapes: cache append retains the whole snapshot,
        # while a fresh Hybrid request evaluates its final token separately.
        with closing(LlamaContext(model=llm._model, params=llm.context_params,
                                  verbose=False)) as context:
            with closing(LlamaBatch(n_tokens=llm.n_batch, embd=0, n_seq_max=1,
                                    verbose=False)) as batch:
                for kind, chunks in batches.items():
                    references[kind] = _native_greedy_reference(llm, context, batch, chunks, 4).text
        llm.eval(tokens)
        state = llm.save_state()
        for array in (state.input_ids, state.scores, state.last_logits):
            array.flags.writeable = False
    finally:
        llm.reset()
    return tokens, prompts, batches, references, state


@pytest.mark.parametrize("cache_type", [LlamaRAMCache, LlamaTrieCache, LlamaDiskCache])
@pytest.mark.parametrize("request_kind", ["exact", "append", "branch"])
def test_real_hybrid_completion_cache_matches_cold_generation(
    hybrid_model, hybrid_completion_cases, monkeypatch, tmp_path, cache_type, request_kind
):
    llm = hybrid_model
    tokens, prompts, batches, references, state = hybrid_completion_cases
    options = dict(max_tokens=4, temperature=0, seed=42)
    cache = LlamaDiskCache(str(tmp_path)) if cache_type is LlamaDiskCache else cache_type()
    cache[tokens] = state
    llm.set_cache(cache)
    llm.reset()
    restore = Mock(wraps=llm.load_state)
    monkeypatch.setattr(llm, "load_state", restore)
    evaluated = []
    native_eval = llm.eval

    def evaluate(tokens, **kwargs):
        evaluated.append((llm.n_tokens, list(tokens)))
        return native_eval(tokens, **kwargs)

    monkeypatch.setattr(llm, "eval", evaluate)
    try:
        result = llm.create_completion(prompts[request_kind], **options)
        assert result["choices"][0]["text"] == references[request_kind]
        assert restore.call_count == int(request_kind != "branch")
        retained = len(tokens) if request_kind != "branch" else 0
        expected, position = [], 0
        for chunk in batches[request_kind]:
            if chunk and position >= retained:
                expected.append((position, chunk))
            position += len(chunk)
        assert [(pos, chunk) for pos, chunk in evaluated if pos < len(prompts[request_kind])] == expected
    finally:
        llm.set_cache(None)
        llm.reset()
        if cache_type is LlamaDiskCache:
            cache.cache.close()


@pytest.mark.parametrize("logits_all", [False, True])
def test_real_hybrid_zero_checkpoints_multiturn_matches_cold_generation(monkeypatch, logits_all):
    path = os.environ.get("LLAMA_TEST_HYBRID_MODEL")
    if not path:
        pytest.skip("Set LLAMA_TEST_HYBRID_MODEL to run real-model tests")
    with closing(Llama(
        model_path=path, n_ctx=256, n_batch=64, n_ubatch=64,
        n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0")),
        ctx_checkpoints=0, logits_all=logits_all, verbose=False,
    )) as llm:
        assert llm.is_hybrid
        cache = llm._hybrid_cache_mgr
        cache._get_size_ext = Mock(side_effect=AssertionError("Checkpoints are disabled"))
        cache._get_data_ext = Mock(side_effect=AssertionError("Checkpoints are disabled"))
        base = llm.tokenize(b"The capital of France is")
        append = base + llm.tokenize(b" and Italy is", add_bos=False)
        branch = base[:-1] + llm.tokenize(b" was", add_bos=False)
        cases = (
            ("cold", base, 0), ("exact", base, len(base)),
            ("append", append, len(base)), ("shorter", base, 0),
            ("branch", branch, 0), ("single-token", base[:1], 0),
            ("append-after-single", base, 1),
        )
        evaluated = []
        native_eval = llm.eval

        def evaluate(tokens, **kwargs):
            evaluated.append((llm.n_tokens, list(tokens)))
            return native_eval(tokens, **kwargs)

        monkeypatch.setattr(llm, "eval", evaluate)
        # Quantized backends can change greedy tokens across batch shapes.
        # Rebuild each reference in an independent context with the same split;
        # do not derive the expected split from the path under test.
        with closing(LlamaContext(model=llm._model, params=llm.context_params,
                                  verbose=False)) as reference:
            with closing(LlamaBatch(n_tokens=llm.n_batch, embd=0, n_seq_max=1,
                                    verbose=False)) as batch:
                for name, prompt, retained in cases:
                    chunks = (prompt[:retained], prompt[retained:]) if retained else (prompt,)
                    expected = _native_greedy_reference(llm, reference, batch, chunks, 1)
                    seen = []

                    def observe(ids, scores):
                        assert ids.tolist() == prompt, name
                        assert scores.shape == (llm._n_vocab,) and np.isfinite(scores).all(), name
                        native = np.ctypeslib.as_array(llm._ctx.get_logits_ith(-1), shape=(llm._n_vocab,))
                        np.testing.assert_array_equal(scores, native, err_msg=name)
                        np.testing.assert_array_equal(scores, expected.logits, err_msg=name)
                        seen.append(True)
                        return False

                    evaluated.clear()
                    result = llm.create_completion(
                        prompt, max_tokens=1, temperature=0, seed=42,
                        stopping_criteria=StoppingCriteriaList([observe]),
                    )
                    assert result["choices"][0]["text"] == expected.text, name
                    assert seen, name
                    assert evaluated == ([] if retained == len(prompt) else [(retained, prompt[retained:])]), name
                    assert llm._input_ids.tolist() == prompt, name
                    assert llm.n_tokens == len(prompt), name
                    assert llm._ctx.memory_seq_pos_min(0) == expected.pos_min, name
                    assert llm._ctx.memory_seq_pos_max(0) == expected.pos_max == len(prompt) - 1, name
        assert not cache.checkpoints and cache.cache_size == 0
        cache._get_size_ext.assert_not_called()
        cache._get_data_ext.assert_not_called()


def test_real_hybrid_trie_eviction_matches_cold_generation(hybrid_model, monkeypatch):
    llm = hybrid_model
    base = llm.tokenize(b"Continue the list of squares: 1, 4, 9, 16, 25,")
    prompts = [
        base,
        base + llm.tokenize(b" 36, 49,", add_bos=False),
        base[:-1] + llm.tokenize(b" and cubes:", add_bos=False),
        llm.tokenize(b"The capital of France is"),
        base,
    ]
    options = dict(max_tokens=4, temperature=0, seed=42)
    reference = []
    for prompt in prompts:
        llm.reset()
        reference.append(llm.create_completion(prompt, **options)["choices"][0]["text"])
    llm.reset()
    llm.eval(base)
    state = llm.save_state()
    cache = LlamaTrieCache(capacity_bytes=state.nbytes * 2)
    cache[base] = state
    prune = Mock(wraps=cache._prune)
    monkeypatch.setattr(cache, "_prune", prune)
    llm.set_cache(cache)
    try:
        for prompt, expected in zip(prompts, reference):
            llm.reset()
            assert llm.create_completion(prompt, **options)["choices"][0]["text"] == expected
            assert cache.cache_size <= cache.capacity_bytes
        assert prune.call_count > 0
    finally:
        llm.set_cache(None)
        llm.reset()


@pytest.mark.parametrize("failure", ["decode", "interrupt"])
def test_real_hybrid_failure_then_retry(hybrid_model, monkeypatch, failure):
    llm = hybrid_model
    def complete():
        return llm.create_completion("Count: 1, 2,", max_tokens=4,
                                     temperature=0, seed=42)
    llm.reset()
    reference = complete()["choices"][0]["text"]
    native_decode = llm._ctx.decode
    def fail_after_commit(batch):
        assert native_decode(batch) == 0
        assert llm._ctx.memory_seq_pos_max(0) >= 0
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise RuntimeError("injected after native commit")
    with monkeypatch.context() as patch:
        patch.setattr(llm._ctx, "decode", fail_after_commit)
        if failure == "interrupt":
            assert complete()["choices"][0]["finish_reason"] == "abort"
        else:
            with pytest.raises(RuntimeError, match="injected after native commit"):
                complete()
    assert llm.n_tokens == 0 and llm._ctx.memory_seq_pos_max(0) == -1
    assert not llm._hybrid_cache_mgr.checkpoints
    assert llm._restored_logits is None and llm._prefilled_prompt is None
    assert complete()["choices"][0]["text"] == reference
    llm.reset()
