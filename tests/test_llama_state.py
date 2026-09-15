import ctypes
import gc
import pickle
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from llama_cpp import Llama
from llama_cpp.llama import LlamaState
from llama_cpp import llama_cpp as lib
from llama_cpp._internals import LlamaSampler, LlamaSamplingContext
from llama_cpp.llama_cache import LlamaRAMCache, LlamaTrieCache
from llama_cpp.llama_speculative import LlamaNGramMapDecoding, SpeculativeType


@pytest.fixture
def model(monkeypatch):
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
        get_logits_ith=lambda idx: native_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    monkeypatch.setattr(lib, "llama_state_get_size", lambda ctx: 2)

    def save(ctx, buf, size):
        buf[:] = b"ok"
        return 2

    monkeypatch.setattr(lib, "llama_state_get_data", save)
    monkeypatch.setattr(lib, "llama_state_set_data", lambda ctx, buf, size: size)
    return llm, native_logits


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
