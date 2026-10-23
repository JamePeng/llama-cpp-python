import os
import ctypes
import json
import multiprocessing
import threading
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import closing, contextmanager

import pytest
import numpy as np

import llama_cpp
import llama_cpp._internals as internals
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def test_model_suppress_tokens_copies_native_storage(monkeypatch):
    for tokens in ([], [3, 1, 3]):
        storage = (ctypes.c_int32 * len(tokens))(*tokens)
        calls = []

        def get_suppressed(vocab, count):
            calls.append(vocab)
            ctypes.cast(count, ctypes.POINTER(ctypes.c_int32))[0] = len(tokens)
            return storage if tokens else None

        monkeypatch.setattr(internals.llama_cpp, "llama_vocab_get_suppress_tokens", get_suppressed)
        model = SimpleNamespace(vocab=123)
        result = internals.LlamaModel.suppress_tokens(model)
        if tokens:
            storage[0] = 99
        assert result == tokens
        model.vocab = None
        with pytest.raises(RuntimeError, match="vocab is None"):
            internals.LlamaModel.suppress_tokens(model)
        assert calls == [123]


@pytest.fixture
def sampling_model(monkeypatch):
    internals.llama_cpp.llama_backend_init()
    monkeypatch.setattr(internals.llama_cpp, "llama_model_get_vocab", lambda _: None)
    return SimpleNamespace(model=None, n_vocab=lambda: 32,
                           suppress_tokens=Mock(return_value=[31]))


@pytest.mark.parametrize("suppressed", [[], [31], [31, 30]])
@pytest.mark.parametrize("temp", [0.0, 1.0])
@pytest.mark.parametrize("probability_filters", [False, True], ids=["top-k", "top-p-min-p-temp"])
def test_sampling_suppression_and_bias_match_native_chain(sampling_model, suppressed, temp, probability_filters):
    lib = internals.llama_cpp
    sampling_model.suppress_tokens.return_value = suppressed
    logits = np.arange(32, dtype=np.float32) * 0.25
    saved = logits.copy()
    # Without suppression, doubling this bias incorrectly moves token 29 to first.
    biases = [lib.llama_logit_bias(29, 0.3)]
    if suppressed:
        biases.append(lib.llama_logit_bias(31, 100.0))
    original_biases = [(item.token, item.bias) for item in biases]
    samplers = [internals.CommonSamplerType.TOP_K]
    if probability_filters:
        samplers.extend([internals.CommonSamplerType.TOP_P, internals.CommonSamplerType.MIN_P,
                         internals.CommonSamplerType.TEMPERATURE])
    params = internals.LlamaSamplingParams(
        temp=temp, top_k=10, top_p=0.8, min_p=0.12, min_keep=1, seed=42,
        logit_bias=biases, samplers=samplers,
    )
    native = internals.LlamaTokenDataArray(n_vocab=32)
    native.copy_logits(logits)
    with closing(internals.LlamaSampler()) as reference:
        reference.add_logit_bias(32, biases + [lib.llama_logit_bias(token, float("-inf")) for token in suppressed])
        reference.add_top_k(10)
        if probability_filters:
            reference.add_top_p(0.8, 1)
            reference.add_min_p(0.12, 1)
            reference.add_temp(temp)
        if temp == 0:
            reference.add_greedy()
        else:
            reference.add_dist(42)
        lib.llama_sampler_apply(reference.sampler, ctypes.byref(native.candidates))
    expected = native.candidates_data[:native.candidates.size]
    token = int(expected["id"][native.candidates.selected])
    assert token not in suppressed
    context = SimpleNamespace(
        get_sampled_token_ith=lambda _: lib.LLAMA_TOKEN_NULL,
        get_logits_ith=lambda _: logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    with closing(internals.LlamaSamplingContext(params, sampling_model)) as sampler:
        for source in (None, logits):  # Context outputs and restored snapshots.
            sampler.reset()
            with closing(sampler.cp()) as clone:
                for current in (sampler, clone):
                    assert current.sample(context, logits=source) == token
                    actual = current._cur_p.candidates_data[:current._cur_p.candidates.size]
                    np.testing.assert_array_equal(actual["id"], expected["id"])
                    # Greedy selection does not populate probabilities.
                    for field in (("logit", "p") if temp else ("logit",)):
                        np.testing.assert_allclose(actual[field], expected[field], rtol=2e-6, atol=1e-8)
    sampling_model.suppress_tokens.assert_called_once_with()
    assert [(item.token, item.bias) for item in params.logit_bias] == original_biases
    np.testing.assert_array_equal(logits, saved)


@pytest.mark.parametrize("grammar_first", [False, True])
def test_sampling_suppression_survives_grammar_resampling(sampling_model, grammar_first):
    lib = internals.llama_cpp
    logits = np.arange(32, dtype=np.float32)
    params = internals.LlamaSamplingParams(temp=0.0, top_k=1,
                                          samplers=[internals.CommonSamplerType.TOP_K])
    with closing(internals.LlamaSampler()) as grammar, closing(internals.LlamaSamplingContext(params, sampling_model)) as sampler:
        # A native mask stands in for a grammar that rejects the first allowed token.
        grammar.add_logit_bias(32, [lib.llama_logit_bias(30, float("-inf"))])
        sampler.grammar_sampler = SimpleNamespace(grammar=grammar.sampler, close=lambda: None)
        assert sampler.sample(SimpleNamespace(), logits=logits, grammar_first=grammar_first) == 29
    np.testing.assert_array_equal(logits, np.arange(32, dtype=np.float32))


def test_sampling_backend_selected_token_skips_cpu_suppression(sampling_model):
    context = SimpleNamespace(get_sampled_token_ith=lambda _: 31,
                              get_logits_ith=Mock(side_effect=AssertionError("CPU fallback is not allowed")))
    params = internals.LlamaSamplingParams(samplers=[internals.CommonSamplerType.TOP_K])
    with closing(internals.LlamaSamplingContext(params, sampling_model)) as sampler:
        assert sampler.sample(context) == 31
        assert sampler._cur_p is None
    context.get_logits_ith.assert_not_called()


@pytest.mark.parametrize("suppressed", [[31], [30], [30, 31]])
def test_sampling_reasoning_budget_preserves_native_logits_and_suppression(sampling_model, suppressed):
    lib = internals.llama_cpp
    sampling_model.suppress_tokens.return_value = suppressed
    sampling_model.tokenize = lambda *args, **kwargs: [30]
    params = internals.LlamaSamplingParams(
        temp=1.0, top_k=32, seed=42, reasoning_budget=0,
        reasoning_start_in_prompt=True, logit_bias=[lib.llama_logit_bias(30, 0.5)],
        samplers=[internals.CommonSamplerType.TOP_K],
    )
    logits = np.arange(32, dtype=np.float32) * 0.25
    # common_reasoning_budget_apply masks other tokens and preserves the forced
    # token's original score; common_sampler_sample then applies the bias chain.
    expected = np.full(32, -np.inf, dtype=np.float32)
    expected[30] = logits[30] + 0.5
    expected[suppressed] = -np.inf
    candidates = internals.LlamaTokenDataArray(n_vocab=32)
    candidates.copy_logits(logits)
    with closing(internals.LlamaSamplingContext(params, sampling_model)) as sampler:
        chain = sampler.sampler_chain.sampler
        # Inspect preprocessing before top-k/dist, including the all-masked
        # conflict case for which no valid sampling distribution exists.
        for index in range(lib.llama_sampler_chain_n(chain) - 2):
            lib.llama_sampler_apply(lib.llama_sampler_chain_get(chain, index), ctypes.byref(candidates.candidates))
        np.testing.assert_array_equal(candidates._logit_view, expected)
        assert not np.isnan(candidates._logit_view).any()
        if 30 not in suppressed:
            assert sampler.sample(SimpleNamespace(), logits=logits) == 30
            probabilities = [sampler._cur_p.candidates.data[i].p for i in range(32)]
            assert np.isfinite(probabilities).all() and sum(probabilities) == pytest.approx(1.0)
    np.testing.assert_array_equal(logits, np.arange(32, dtype=np.float32) * 0.25)


def test_multimodal_embedding_requires_initialization():
    model = LlamaEmbedding.__new__(LlamaEmbedding)
    with pytest.raises(RuntimeError, match="not initialized"):
        model.embed_multimodal("hello")
    model._ctx = SimpleNamespace(ctx=123)
    try:
        with pytest.raises(ValueError, match="mmproj_path"):
            model.embed_multimodal("hello")
    finally:
        model._ctx = None


@pytest.mark.parametrize("profile,width,pooling,causal", [
    ("MULTIMODAL", 768, llama_cpp.LLAMA_POOLING_TYPE_MEAN, False),
    ("QWEN3VL", 4096, llama_cpp.LLAMA_POOLING_TYPE_LAST, True),
])
def test_real_multimodal_embedding_entrypoint_readiness(profile, width, pooling, causal):
    import io
    from PIL import Image
    from llama_cpp.llama_multimodal import MTMDEmbeddingHandler

    model_path = os.environ.get(f"LLAMA_TEST_{profile}_EMBEDDING_MODEL")
    mmproj_path = os.environ.get(f"LLAMA_TEST_{profile}_EMBEDDING_MMPROJ")
    if not model_path or not mmproj_path:
        pytest.skip(f"Set LLAMA_TEST_{profile}_EMBEDDING_MODEL/MMPROJ paths")
    image = io.BytesIO()
    Image.new("RGB", (224, 224), "red").save(image, format="PNG")
    handler = MTMDEmbeddingHandler(mmproj_path, use_gpu=False,
                                   image_max_tokens=64 if causal else 70, verbose=False)
    try:
        with closing(LlamaEmbedding(model_path, embedding_handler=handler, n_ctx=256,
                            n_batch=256, n_ubatch=256, n_threads=4, verbose=False,
                            n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0")),
                            pooling_type=pooling)) as model:
            assert handler.mtmd_ctx is None
            prefix = ("<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
                      "<|im_start|>user\n") if causal else "title: none | text: "
            suffix = "<|im_end|>\n<|im_start|>assistant\n" if causal else ""
            parts = [prefix, image.getvalue(), suffix]
            # Call directly after construction; no manual projector initialization.
            vector = model.embed_multimodal(parts=parts)
            assert len(vector) == width and np.isfinite(vector).all()
            assert model._ctx.get_causal_attn() is causal
            np.testing.assert_allclose(model.embed_multimodal(parts=parts), vector, atol=1e-5)
            if causal:
                # Match a direct native submission using the same decode shape.
                model.reset()
                with handler.tokenize_parts(model, parts) as request:
                    encoded = request.encode()
                    for chunk in encoded:
                        if chunk.tokens:
                            assert all(isinstance(pos, int) for pos in chunk.positions)
                        else:
                            assert all(len(pos) == 4 for pos in chunk.positions)
                    native = llama_cpp.llama_batch_ext_init(model._ctx.ctx)
                    assert native
                    try:
                        for chunk in encoded:
                            rows = chunk.tokens if chunk.tokens else chunk.embeddings
                            for row, pos in zip(rows, chunk.positions):
                                if chunk.tokens:
                                    idx = llama_cpp.llama_batch_ext_add_token(native, 0, row)
                                    coords = (llama_cpp.llama_pos * 1)(pos)
                                else:
                                    embd = llama_cpp.llama_embd(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, len(row))
                                    idx = llama_cpp.llama_batch_ext_add_embd(native, 0, embd)
                                    coords = (llama_cpp.llama_pos * 4)(*pos)
                                assert idx >= 0 and llama_cpp.llama_batch_ext_set_pos(native, idx, coords)
                                assert llama_cpp.llama_batch_ext_set_output_logits(native, idx, True)
                        assert llama_cpp.llama_process(model._ctx.ctx, llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, native) == 0
                        raw = np.ctypeslib.as_array(model._ctx.get_embeddings_seq(0), shape=(width,)).copy()
                        assert np.dot(vector, raw / np.linalg.norm(raw)) > 0.99999
                    finally:
                        llama_cpp.llama_batch_ext_free(native)
                model.reset()
            handler.is_support_vision = False
            with pytest.raises(ValueError, match="does not support image"):
                model.embed_multimodal(parts=["title: none | text: ", image.getvalue()])
            handler.is_support_vision = True
            model._ctx.set_embeddings(False)
            vector = model.embed_multimodal(prefix + "hello" + suffix)
            assert len(vector) == width and np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
            assert model._ctx.get_causal_attn() is causal
        assert handler.mtmd_ctx is None
        with pytest.raises(RuntimeError, match="closed"):
            model.embed_multimodal("hello")
    finally:
        handler.close()


@pytest.fixture
def runtime_context(monkeypatch):
    ctx = internals.LlamaContext.__new__(internals.LlamaContext)
    ctx.ctx = 123
    ctx.model = SimpleNamespace(
        n_vocab=lambda: 100,
        rope_type=lambda: internals.llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_MROPE,
    )
    ctx.params = object()
    ctx.verbose = False
    ctx._loras_applied = False
    ctx._cvec_applied = False
    ctx._lora_refs = ()
    monkeypatch.setattr(internals.llama_cpp, "llama_free", Mock())
    monkeypatch.setattr(internals.llama_cpp, "llama_n_batch", lambda _: 3)
    monkeypatch.setattr(internals.llama_cpp, "llama_n_seq_max", lambda _: 2)
    try:
        yield ctx
    finally:
        ctx.ctx = None
        ctx._batch_ext_refs = []
        ctx._checkpoint_caches = []


@pytest.fixture
def batch_runtime(runtime_context, monkeypatch):
    lib = internals.llama_cpp
    rows, calls, freed = [], [], []
    state = SimpleNamespace(fail=False, status=0)
    context = runtime_context
    monkeypatch.setattr(lib, "llama_batch_ext_init", lambda _: 456)
    monkeypatch.setattr(lib, "llama_batch_ext_free", lambda p: freed.append(p))
    monkeypatch.setattr(lib, "llama_batch_ext_clear", lambda _: rows.clear())

    def add(_, sid, token):
        rows.append(dict(token=token, seq_ids=[sid]))
        return len(rows) - 1

    def payload(_, idx, embd, field):
        if state.fail == field:
            return False
        rows[idx][field] = np.ctypeslib.as_array(embd.data, shape=(embd.n_rows * embd.n_embd,)).copy()
        return True

    def add_embd(p, sid, embd):
        idx = add(p, sid, None)
        return idx if payload(p, idx, embd, "embedding") else -2

    def sequence(_, idx, sid):
        rows[idx]["seq_ids"].append(sid)
        return True

    def position(_, idx, pos):
        n = 1 if rows[idx]["token"] is not None else 4
        rows[idx]["position"] = tuple(pos[i] for i in range(n))
        return True

    def output(_, idx, flag):
        rows[idx]["output"] = flag
        return True

    def process(ctx, kind, native):
        calls.append((ctx, kind, native))
        return state.status

    monkeypatch.setattr(lib, "llama_batch_ext_add_token", add)
    monkeypatch.setattr(lib, "llama_batch_ext_add_embd", add_embd)
    monkeypatch.setattr(lib, "llama_batch_ext_set_embd_token", lambda *args: payload(*args, "embedding"))
    monkeypatch.setattr(lib, "llama_batch_ext_set_embd_state", lambda *args: payload(*args, "state"))
    monkeypatch.setattr(lib, "llama_batch_ext_add_seq", sequence)
    monkeypatch.setattr(lib, "llama_batch_ext_set_pos", position)
    monkeypatch.setattr(lib, "llama_batch_ext_set_output_logits", output)
    monkeypatch.setattr(lib, "llama_process", process)
    try:
        with internals.LlamaBatchExt(context=context) as batch:
            yield SimpleNamespace(context=context, batch=batch, rows=rows, calls=calls, freed=freed, state=state)
    finally:
        context.ctx = None


def test_embedding_normalization_modes_preserve_nonzero_vectors():
    from llama_cpp._utils import normalize_embedding

    for mode in (-1, False, True, 0, 1, 2, 6):
        for values in ([0.0, 0.0], [3.0, -4.0], [1e7, -2e7]):
            case = f"mode={mode!r}, values={values}"
            with np.errstate(over="raise", invalid="raise"):
                result = np.asarray(normalize_embedding(values, mode), dtype=np.float64)
            assert np.isfinite(result).all(), case
            if mode is False or mode == -1 or not any(values):
                np.testing.assert_array_equal(result, values, err_msg=case)
            elif mode == 0:
                assert np.max(np.abs(result)) == pytest.approx(32760, rel=1e-6), case
            else:
                assert np.linalg.norm(result, ord=2 if mode is True else mode) == pytest.approx(1, rel=1e-6), case
            if any(values):
                assert result[0] / result[1] == pytest.approx(values[0] / values[1]), case


@pytest.mark.parametrize("causal,media_non_causal", [(True, False), (True, True), (False, True), (False, False)])
def test_multimodal_pooling_attention_and_failure_contract(batch_runtime, tmp_path, monkeypatch, causal, media_non_causal):
    from llama_cpp.llama_multimodal import MTMDEncodedChunk, MTMDEmbeddingHandler

    r = batch_runtime
    raw = np.array([3, 4], dtype=np.float32)
    monkeypatch.setattr(r.context, "get_causal_attn", lambda: causal)
    monkeypatch.setattr(r.context, "set_causal_attn", Mock())
    monkeypatch.setattr(r.context, "pooling_type", lambda: llama_cpp.LLAMA_POOLING_TYPE_LAST)
    monkeypatch.setattr(r.context, "n_ubatch", lambda: 3)
    monkeypatch.setattr(r.context, "set_embeddings", Mock())
    monkeypatch.setattr(r.context, "get_embeddings_seq", lambda seq: raw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    model = SimpleNamespace(_ctx=r.context, context_params=SimpleNamespace(embeddings=True),
                            n_ctx=lambda: 3, n_embd_out=lambda: 2, reset=Mock())
    chunks = (
        MTMDEncodedChunk(0, (1,), None, (0,), 1, False),
        MTMDEncodedChunk(1, (), np.ones((1, 2), dtype=np.float32), ((1, 2, 3, 0),), 1, media_non_causal),
    )
    request = SimpleNamespace(n_tokens=2, n_pos=2, encode=lambda: chunks)
    released = []

    @contextmanager
    def tokenize(*args, **kwargs):
        try:
            yield request
        finally:
            released.append(True)

    with MTMDEmbeddingHandler(str(tmp_path), verbose=False) as handler:
        monkeypatch.setattr(handler, "_init_mtmd_context", Mock())
        monkeypatch.setattr(handler, "tokenize", tokenize)
        handler.is_support_vision = True
        if causal and media_non_causal:
            with pytest.raises(NotImplementedError, match="chunk-local"):
                handler.embed(model, "prompt")
            assert not r.calls
        else:
            r.state.status = 1
            with pytest.raises(RuntimeError, match="decode failed"):
                handler.embed(model, "prompt")
            r.state.status = 0
            np.testing.assert_allclose(handler.embed(model, "prompt"), [0.6, 0.8])
        assert model.reset.call_count == len(released) * 2
        r.context.set_causal_attn.assert_not_called()
        assert r.context.get_causal_attn() is causal


class MockLoraAdapter:
    def __init__(self, address):
        self.adapter = ctypes.cast(ctypes.c_void_p(address), internals.llama_cpp.llama_adapter_lora_p_ctypes)


def test_closed_context_rejected_before_native_call(runtime_context, monkeypatch):
    runtime_context.close()
    for method, args, native in (
        ("n_ctx", (), "llama_n_ctx"),
        ("n_ctx_seq", (), "llama_n_ctx_seq"),
        ("n_batch", (), "llama_n_batch"),
        ("n_ubatch", (), "llama_n_ubatch"),
        ("n_seq_max", (), "llama_n_seq_max"),
        ("n_rs_seq", (), "llama_n_rs_seq"),
        ("pooling_type", (), "llama_pooling_type"),
        ("set_n_threads", (1, 1), "llama_set_n_threads"),
        ("n_threads", (), "llama_n_threads"),
        ("n_threads_batch", (), "llama_n_threads_batch"),
        ("set_causal_attn", (True,), "llama_set_causal_attn"),
        ("get_causal_attn", (), "llama_get_causal_attn"),
        ("clear_loras", (), "llama_set_adapters_lora"),
        ("apply_loras", ([],), "llama_set_adapters_lora"),
        ("clear_cvec", (), "llama_set_adapter_cvec"),
        ("apply_cvec", ([], 1, 1, 1), "llama_set_adapter_cvec"),
        ("graph_reserve", (1, 1, 1), "llama_graph_reserve"),
    ):
        with monkeypatch.context() as patch:
            call = Mock(name=native)
            patch.setattr(internals.llama_cpp, native, call)
            with pytest.raises(RuntimeError, match="closed"):
                getattr(runtime_context, method)(*args)
            assert call.call_count == 0, method


def test_graph_reserve_validation_failure_and_recovery(runtime_context, monkeypatch):
    context = runtime_context
    events, graph = [], object()
    reserve = Mock(side_effect=lambda *args: events.append(args) or graph)
    sync = Mock(side_effect=lambda _: events.append('sync'))
    monkeypatch.setattr(internals.llama_cpp, 'llama_synchronize', sync)
    monkeypatch.setattr(internals.llama_cpp, 'llama_graph_reserve', reserve)
    for dimensions, error in (
        ((0, 1, 1), ValueError), ((1, 0, 1), ValueError),
        ((1, 1, 0), ValueError), ((-1, 1, 1), ValueError),
        ((2**32, 1, 1), ValueError), ((1, 3, 1), ValueError),
        ((1, 1, 2), ValueError), ((2**31 - 1, 2, 1), ValueError),
        ((True, 1, 1), TypeError), ((1.5, 1, 1), TypeError),
    ):
        with pytest.raises(error):
            context.graph_reserve(*dimensions)
    reserve.assert_not_called()
    sync.assert_not_called()
    for failure, message in ((None, 'failed to reserve'),
                              (RuntimeError('missing symbol'), 'missing symbol')):
        reserve.side_effect = failure
        reserve.return_value = None
        with pytest.raises(RuntimeError, match=message):
            context.graph_reserve(1, 1, 1)
        assert context.ctx == 123
    events.clear()
    reserve.side_effect = lambda *args: events.append(args) or graph
    # Reservation synchronizes first and permits native sequence rounding.
    assert context.graph_reserve(3, 2, 4) is graph
    assert events == ['sync', (123, 3, 2, 4)]


def test_attached_lora_survives_gc_and_failed_replacement(runtime_context, monkeypatch):
    context = runtime_context
    native = Mock(return_value=0)
    monkeypatch.setattr(internals.llama_cpp, "llama_set_adapters_lora", native)
    with pytest.raises(RuntimeError, match="closed or invalid"):
        context.apply_loras([(SimpleNamespace(adapter=None), 1.0)])
    native.assert_not_called()
    original = MockLoraAdapter(1)
    reference = weakref.ref(original)
    context.apply_loras([(original, 1.0)])
    del original
    gc.collect()
    assert reference() is not None
    native.return_value = -1
    with pytest.raises(RuntimeError, match="Failed to set LoRA"):
        context.apply_loras([(MockLoraAdapter(2), 1.0)])
    with pytest.raises(RuntimeError, match="detach failed"):
        context.clear_loras()
    gc.collect()
    assert reference() is not None
    native.return_value = 0
    context.clear_loras()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("failure", ["batch", "native"])
def test_context_close_attempts_all_resources_and_frees_once(runtime_context, monkeypatch, failure):
    context = runtime_context
    events = []

    def fail():
        events.append("batch")
        if failure == "batch":
            raise RuntimeError("batch cleanup failed")

    context._batch_ext_refs = [SimpleNamespace(close=fail)]
    context._checkpoint_caches = [SimpleNamespace(close=lambda: events.append("checkpoint"))]
    context._exit_stack = SimpleNamespace(close=lambda: events.append("exit"))
    adapter = MockLoraAdapter(1)
    reference = weakref.ref(adapter)
    context._lora_refs = (adapter,)
    del adapter

    def free(pointer):
        assert pointer == 123 and reference() is not None
        events.append("native")
        if failure == "native":
            raise RuntimeError("native cleanup failed")

    monkeypatch.setattr(internals.llama_cpp, "llama_free", free)
    with pytest.raises(RuntimeError, match=f"{failure} cleanup failed"):
        context.close()
    assert events == ["batch", "checkpoint", "native", "exit"]
    assert context.ctx is None and context.model is None and context.params is None
    gc.collect()
    assert reference() is None
    context._batch_ext_refs = []
    context.close()
    assert events.count("native") == 1


def test_interleaved_range_preserves_positions_and_output_mapping(batch_runtime):
    r = batch_runtime
    r.batch.add_token(1, 0)
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    r.batch.add_embeddings(source, positions=[(1, 2, 3, 0), (2, 3, 4, 0)], seq_ids=(0, 1))
    source[:] = -1  # logical inputs own the data, even before rendering
    r.batch.add_token(2, 3, output=True)
    assert len(r.batch) == 4 > r.batch.capacity()
    assert r.context.decode(r.batch, start=1, count=3) == 0
    assert [e["token"] for e in r.rows] == [None, None, 2]
    assert r.rows[0]["position"] == (1, 2, 3, 0)
    assert r.rows[0]["seq_ids"] == [0, 1]
    np.testing.assert_array_equal(r.rows[0]["embedding"], np.arange(4))
    assert r.batch.submitted_indices == (1, 2, 3)
    assert r.batch.output_indices == (3,)
    assert r.calls == [(123, internals.llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, 456)]
    with pytest.raises(ValueError, match="exceeds"):
        r.context.decode(r.batch)


def test_batch_render_failure_clears_partial_rows_and_can_retry(batch_runtime, monkeypatch):
    r = batch_runtime
    invalidations = []
    r.context._invalidate_checkpoints = lambda: invalidations.append(True)
    for failure in ("embedding", "position", "state"):
        r.batch.reset()
        r.calls.clear()
        invalidations.clear()
        for i in range(2):
            r.batch.add_embedding([1, 2], (i, i, i, 0), state=[3, 4])
        symbol = "llama_batch_ext_set_embd_state" if failure == "state" else "llama_batch_ext_set_pos"
        with monkeypatch.context() as patch:
            if failure == "embedding":
                patch.setattr(r.state, "fail", "embedding")
            else:
                patch.setattr(internals.llama_cpp, symbol, lambda _, idx, value: idx == 0)
            message = "-2" if failure == "embedding" else f"{'state embedding' if failure == 'state' else failure} at logical entry 1"
            with pytest.raises(RuntimeError, match=message):
                r.context.decode(r.batch)
        assert r.rows == [] and r.calls == [], failure
        assert invalidations == [] and len(r.batch) == 2, failure
        assert r.batch.submitted_indices == r.batch.output_indices == (), failure
        assert r.context.decode(r.batch) == 0, failure


def test_bulk_append_validation_is_atomic_and_paired_mode_is_distinct(batch_runtime):
    r = batch_runtime
    with pytest.raises(ValueError, match="vocabulary"):
        r.batch.add_tokens([1, 100], positions=[0, 1])
    assert len(r.batch) == 0
    r.batch.add_token_embedding(1, [[1, 2], [3, 4]], 0)
    assert r.context.decode(r.batch) == 0
    np.testing.assert_array_equal(r.rows[0]["embedding"], [1, 2, 3, 4])
    r.batch.add_token(2, 1)
    with pytest.raises(ValueError, match="Paired"):
        r.context.decode(r.batch)
    assert len(r.calls) == 1


@pytest.mark.parametrize("kind", ["token", "embedding", "paired"])
def test_extended_batch_keeps_input_embedding_and_hidden_state_separate(batch_runtime, kind):
    r = batch_runtime
    hidden = np.arange(6, dtype=np.float32).reshape(2, 3)
    embedding = np.arange(4, dtype=np.float32)
    if kind == "token":
        r.batch.add_token(1, 2, seq_ids=(0, 1), output=True, state=hidden)
    elif kind == "embedding":
        r.batch.add_embedding(embedding, (2, 3, 4, 0), seq_ids=(0, 1), output=True, state=hidden)
    else:
        r.batch.add_token_embedding(1, embedding, 2, seq_ids=(0, 1), output=True, state=hidden)
    hidden[:] = -1
    embedding[:] = -1
    assert r.context.decode(r.batch) == 0
    np.testing.assert_array_equal(r.rows[0]["state"], np.arange(6))
    if kind == "token":
        assert "embedding" not in r.rows[0]
    else:
        np.testing.assert_array_equal(r.rows[0]["embedding"], np.arange(4))
    assert r.rows[0]["position"] == ((2, 3, 4, 0) if kind == "embedding" else (2,))
    assert r.rows[0]["seq_ids"] == [0, 1]
    assert r.batch.output_indices == (0,)


def test_lilicorr_draft_submits_separate_states_through_extended_batch(batch_runtime, monkeypatch):
    from llama_cpp.llama_speculative import LlamaDFlashDecoding, SpecConfig, SpeculativeType

    r = batch_runtime
    engine = object.__new__(LlamaDFlashDecoding)
    engine.config = SpecConfig(spec_type=SpeculativeType.DRAFT_DFLASH,
                               draft_model_path="lilicorr.gguf", draft_n_max=1)
    engine.is_lilicorr, engine.is_dflash2, engine.is_dspark = True, False, False
    engine.block_size, engine.draft_limit, engine.draft_min = 3, 1, 0
    engine.lilicorr_top_k, engine.n_embd_dec, engine.mask_token_id = 2, 6, 99
    anchor = np.arange(6, dtype=np.float32)
    engine._lilicorr_features = {6: anchor}
    engine.noise_batch, engine.draft_context = r.batch, r.context
    engine.checkpoint, engine.restore = Mock(return_value={"position": 6}), Mock()
    lattice = np.array([[0] * 6, [40, 41, 0, 3, 0, 3],
                        [50, 51, 0, 3, 0, 3]], dtype=np.float32)
    monkeypatch.setattr(internals.llama_cpp, "llama_get_embeddings_nextn",
                        lambda _: lattice.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    assert engine.draft([], n_past=7, id_last=42, n_max=1).tolist() == [41]
    assert [row["token"] for row in r.rows] == [42, 99, 99]
    assert [row["position"] for row in r.rows] == [(7,), (8,), (9,)]
    assert r.calls == [(123, internals.llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, 456)]
    anchor.fill(-1)
    for row in r.rows:
        assert "embedding" not in row and not row.get("output", False)
        np.testing.assert_array_equal(row["state"], np.arange(6))
    assert r.batch.output_indices == ()
    engine.restore.assert_called_once_with({"position": 6}, 0)


@pytest.mark.parametrize("kind,field", [("tokens", "state"), ("embeddings", "state"), ("embeddings", "embedding")])
def test_bulk_payload_ownership_range_and_reset(batch_runtime, kind, field):
    r = batch_runtime
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    kwargs = {"states": source} if field == "state" else {}
    if kind == "tokens":
        r.batch.add_tokens([1, 2, 3, 4], positions=range(4), **kwargs)
    else:
        r.batch.add_embeddings(source if field == "embedding" else np.ones((4, 2)),
                               positions=[(i, i, i, 0) for i in range(4)], **kwargs)
    snapshot = r.batch.entries
    owner = getattr(snapshot[0], field).base
    assert all(getattr(entry, field).base is owner and not getattr(entry, field).flags.writeable for entry in snapshot)
    reference = weakref.ref(owner)
    source[:] = -1
    assert r.context.decode(r.batch, start=1, count=2) == 0
    np.testing.assert_array_equal(r.rows[0][field], np.arange(6, 12))
    np.testing.assert_array_equal(r.rows[1][field], np.arange(12, 18))
    assert r.batch.submitted_indices == (1, 2)
    r.batch.reset()
    assert len(r.batch) == 0 and r.rows == []
    assert r.batch.output_indices == r.batch.submitted_indices == ()
    assert reference() is not None  # retained snapshots keep their payloads alive
    del snapshot, owner
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("kind", ["tokens", "embeddings"])
def test_invalid_bulk_states_do_not_append_partial_inputs(batch_runtime, kind):
    r = batch_runtime
    r.batch.add_token(3, 0)
    for states in ([1, 2], [[1, 2]], np.ones((2, 1, 2)), np.empty((2, 0))):
        with pytest.raises(ValueError, match="states"):
            if kind == "tokens":
                r.batch.add_tokens([1, 2], positions=[1, 2], states=states)
            else:
                r.batch.add_embeddings([[1, 2], [3, 4]], positions=[(1, 1, 1, 0), (2, 2, 2, 0)], states=states)
        assert len(r.batch) == 1 and r.calls == []


@pytest.mark.parametrize("invalid", [None, [1, 2, 3]])
def test_state_presence_and_width_are_validated_per_range(batch_runtime, invalid):
    r = batch_runtime
    r.batch.add_token(1, 0, state=[1, 2])
    r.batch.add_token(2, 1, state=invalid)
    with pytest.raises(ValueError, match="State embeddings|state embeddings"):
        r.context.decode(r.batch)
    assert r.rows == [] and r.calls == []
    assert r.context.decode(r.batch, count=1) == 0


@pytest.mark.parametrize("extended", [False, True], ids=["legacy", "extended"])
def test_context_decode_status_and_checkpoint_invalidation(batch_runtime, monkeypatch, extended):
    r = batch_runtime
    r.batch.add_token(1, 0, state=[1, 2] if extended else None)
    invalidations = []
    r.context._invalidate_checkpoints = lambda: invalidations.append(True)
    kind = internals.llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE
    monkeypatch.setattr(internals.llama_cpp, "llama_decode", lambda *_: r.state.status)
    decode = (lambda: r.context.process(r.batch, kind)) if extended else (lambda: r.context.decode(SimpleNamespace(batch=object())))
    for status in (0, 1, 2, -1, -2, 0):
        r.state.status = status
        invalidations.clear()
        if status in (0, 1):
            assert decode() == status
            assert invalidations == [], status
        else:
            error = internals.LlamaDecodeAbort if status == 2 else RuntimeError
            with pytest.raises(error):
                decode()
            assert invalidations == [True], status


def test_encode_context_binding_and_lifecycle(batch_runtime):
    r = batch_runtime
    r.batch.add_token(1, 0)
    kind = internals.llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_ENCODE
    assert r.context.process(r.batch, kind) == 0
    r.state.status = -1
    with pytest.raises(RuntimeError, match="encode returned"):
        r.context.encode(r.batch)
    with pytest.raises(ValueError, match="different context"):
        r.batch.render(object())
    with pytest.raises(ValueError, match="process type"):
        r.context.process(r.batch, 99)
    r.context.ctx = None
    with pytest.raises(RuntimeError, match="closed"):
        r.batch.render()
    r.batch.close()
    r.batch.close()
    assert r.freed == [456]
    with pytest.raises(RuntimeError, match="closed"):
        r.batch.add_token(1, 0)


def test_invalid_positions_sequences_and_widths_never_reach_processing(batch_runtime):
    r = batch_runtime
    for position in (0, (0, 0, 0), (0, 0, 0, 2**31)):
        with pytest.raises(ValueError, match="position"):
            r.batch.add_embedding([1, 2], position)
    for ids in ((), (-1,), (2,)):
        with pytest.raises(ValueError, match="seq_ids"):
            r.batch.add_token(1, 0, seq_ids=ids)
    r.batch.add_embedding([1, 2], (0, 0, 0, 0))
    r.batch.add_embedding([1, 2, 3], (1, 1, 1, 0))
    with pytest.raises(ValueError, match="width"):
        r.context.decode(r.batch)
    assert r.calls == []
    assert r.context.decode(r.batch, count=1) == 0
    r.batch.reset()
    assert len(r.batch) == 0 and r.rows == []
    assert r.batch.output_indices == r.batch.submitted_indices == ()
    r.batch.add_token(np.int64(3), np.int32(1))
    assert r.context.decode(r.batch) == 0


@pytest.mark.parametrize("cleanup", ["close", "gc_cycle", "close_failure"])
def test_batch_cleanup_releases_payloads_context_and_native_handle(batch_runtime, monkeypatch, cleanup):
    r = batch_runtime
    context = internals.LlamaContext.__new__(internals.LlamaContext)
    context.__dict__.update({k: v for k, v in r.context.__dict__.items() if k != "_batch_ext_refs"})
    context_ref = weakref.ref(context)
    if cleanup == "close_failure":
        def fail(pointer):
            r.freed.append(pointer)
            raise RuntimeError("cleanup failed")
        monkeypatch.setattr(internals.llama_cpp, "llama_batch_ext_free", fail)
    batch = internals.LlamaBatchExt(context=context)
    batch.add_embedding([1, 2], (0, 0, 0, 0), state=[3, 4, 5])
    payload_refs = [weakref.ref(getattr(batch.entries[0], field)) for field in ("embedding", "state")]
    batch_ref = weakref.ref(batch)
    del context
    gc.collect()
    assert context_ref() is not None
    if cleanup == "gc_cycle":
        batch.cycle = batch
    else:
        if cleanup == "close_failure":
            with pytest.raises(RuntimeError, match="cleanup failed"):
                batch.close()
        else:
            batch.close()
        assert len(batch) == 0 and batch._context is None
        batch.close()
    del batch
    gc.collect()
    assert batch_ref() is None and context_ref() is None
    assert all(reference() is None for reference in payload_refs)
    assert r.freed == [456]
    gc.collect()
    assert r.freed == [456]


def test_context_close_releases_batches_before_native_context(batch_runtime, monkeypatch):
    r = batch_runtime
    r.batch.add_embedding([1, 2], (0, 0, 0, 0))
    storage = weakref.ref(r.batch.entries[0].embedding)

    def free_context(_):
        assert r.freed == [456]
        assert storage() is None

    monkeypatch.setattr(internals.llama_cpp, "llama_free", free_context)
    r.context.close()
    assert len(r.batch) == 0
    with pytest.raises(RuntimeError, match="closed"):
        r.batch.render()
    r.batch.close()
    assert r.freed == [456]


@pytest.mark.parametrize('resource', ['context', 'batch', 'batch_finalizer',
                                    'model_vocab', 'legacy_batch', 'sampler'])
def test_runtime_init_failure_releases_only_owned_native_handles(runtime_context, monkeypatch, resource):
    free = Mock()
    owned = None
    if resource == 'context':
        monkeypatch.setattr(internals.llama_cpp, 'llama_init_from_model', lambda *_: None)
        monkeypatch.setattr(internals.llama_cpp, 'llama_free', free)
        create = lambda: internals.LlamaContext(
            model=SimpleNamespace(model=123), params=internals.llama_cpp.llama_context_default_params())
        error, message = RuntimeError, 'Failed to create'
    elif resource in ('batch', 'batch_finalizer'):
        monkeypatch.setattr(internals.llama_cpp, 'llama_batch_ext_init', lambda _: 456 if resource == 'batch_finalizer' else None)
        monkeypatch.setattr(internals.llama_cpp, 'llama_batch_ext_free', free)
        if resource == 'batch_finalizer':
            monkeypatch.setattr(internals.weakref, 'finalize', Mock(side_effect=MemoryError('finalizer allocation failed')))
            owned = 456
        create = lambda: internals.LlamaBatchExt(context=runtime_context)
        error, message = MemoryError, 'finalizer allocation' if resource == 'batch_finalizer' else 'init failed'
    elif resource == 'model_vocab':
        owned = object()
        monkeypatch.setattr(internals.os.path, 'exists', lambda _: True)
        monkeypatch.setattr(internals.llama_cpp, 'llama_model_load_from_file', lambda *_: owned)
        monkeypatch.setattr(internals.llama_cpp, 'llama_model_get_vocab', lambda _: None)
        monkeypatch.setattr(internals.llama_cpp, 'llama_model_free', free)
        create = lambda: internals.LlamaModel(path_model='model.gguf', params=object(), verbose=False)
        error, message = ValueError, 'Failed to get vocab'
    elif resource == 'legacy_batch':
        owned = SimpleNamespace(token=object(), embd=object())
        monkeypatch.setattr(internals.llama_cpp, 'llama_batch_init', lambda *_: owned)
        monkeypatch.setattr(internals.llama_cpp, 'llama_batch_free', free)
        create = lambda: internals.LlamaBatch(n_tokens=1, embd=1, n_seq_max=1, mixed=True, verbose=False)
        error, message = RuntimeError, 'expected batch.token to be NULL'
    else:
        sampler = SimpleNamespace(close=free)
        candidates = Mock()
        monkeypatch.setattr(internals, 'LlamaTokenDataArray', candidates)
        monkeypatch.setattr(internals, 'LlamaSampler', lambda: sampler)
        monkeypatch.setattr(internals.llama_cpp, 'llama_model_get_vocab', lambda _: object())
        monkeypatch.setattr(internals.LlamaSamplingContext, '_build_sampler_chain', Mock(side_effect=RuntimeError('sampler chain build failed')))
        context = internals.LlamaSamplingContext.__new__(internals.LlamaSamplingContext)
        create = lambda: context.__init__(params=internals.LlamaSamplingParams(),
            model=SimpleNamespace(model=object(), verbose=False, n_vocab=lambda: 8))
        error, message = RuntimeError, 'sampler chain build failed'
    with pytest.raises(error, match=message):
        create()
    gc.collect()
    if resource == 'sampler':
        context.close()
        context.close()
        free.assert_called_once_with()
        candidates.assert_not_called()  # Vocabulary storage is lazy.
        assert context.model is context.params is context.vocab is None
    elif owned is not None:
        free.assert_called_once_with(owned)
    else:
        free.assert_not_called()


@pytest.mark.parametrize("entry", ["create_completion", "__call__", "create_chat_completion"])
def test_public_generation_options_reach_execution(entry):
    for present, expected in ((0.0, 1.5), (0.7, 0.7)):
        llm = object.__new__(llama_cpp.Llama)
        calls = []

        def completion(**kwargs):
            calls.append(kwargs)
            yield {"choices": [{"text": "", "finish_reason": "length"}]}

        def chat(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": ""}}]}

        llm._create_completion = completion
        llm.chat_handler, llm._chat_handlers, llm.chat_format = chat, {}, None
        options = dict(present_penalty=present, presence_penalty=1.5)
        if entry == "create_chat_completion":
            llm.create_chat_completion(messages=[{"role": "user", "content": "Hello"}], **options)
        else:
            getattr(llm, entry)([1], ignore_eos=True, **options)
            assert calls[0]["ignore_eos"] is True
        assert calls[0]["present_penalty"] == expected
        assert "presence_penalty" not in calls[0]


@pytest.mark.parametrize(
    ("ignore_eos", "expected_text", "expected_finish_reason"),
    [(False, "", "stop"), (True, "<eog>", "length")],
)
def test_private_completion_respects_ignore_eos_at_eog_boundary(
    monkeypatch, ignore_eos, expected_text, expected_finish_reason
):
    llm = object.__new__(llama_cpp.Llama)
    forwarded = []

    def fake_generate(tokens, **kwargs):
        assert tokens == [1]
        forwarded.append(kwargs["ignore_eos"])
        yield 2

    llm._model = SimpleNamespace(
        vocab=object(),
        token_bos=lambda: 1,
        token_eos=lambda: 2,
        token_sep=lambda: -1,
        token_fim_pre=lambda: -1,
        token_fim_mid=lambda: -1,
        token_fim_suf=lambda: -1,
        get_add_sep=lambda: False,
    )
    llm._abort_event = threading.Event()
    llm.metadata = {}
    llm.spm_infill = False
    llm.verbose = False
    llm._n_ctx = 8
    llm._logits_all = False
    llm._seed = 0
    llm.cache = None
    llm.model_path = "fake.gguf"
    llm.input_ids = np.zeros(8, dtype=np.intc)
    llm.n_tokens = 1
    llm.scores = np.zeros((1, 4), dtype=np.float32)
    llm.generate = fake_generate
    llm.detokenize = (
        lambda tokens, prev_tokens=None, **kwargs: b"".join(
            b"<eog>" if token == 2 else b"x" for token in tokens
        )
    )
    monkeypatch.setattr(
        "llama_cpp.llama.llama_cpp_lib.llama_token_is_eog",
        lambda vocab, token: token == 2,
    )

    result = next(
        llm._create_completion(
            prompt=[1], max_tokens=1, ignore_eos=ignore_eos
        )
    )

    assert forwarded == [ignore_eos]
    assert result["choices"][0]["text"] == expected_text
    assert result["choices"][0]["finish_reason"] == expected_finish_reason


def test_context_manages_borrowed_threadpool_and_abort_callback_lifetimes(runtime_context, monkeypatch):
    context, calls = runtime_context, []
    monkeypatch.setattr(internals.llama_cpp, 'llama_synchronize', lambda ctx: calls.append(('sync', ctx)))
    monkeypatch.setattr(internals.llama_cpp, 'llama_attach_threadpool',
                        lambda ctx, pool, batch_pool: calls.append(('attach', ctx, pool, batch_pool)))
    monkeypatch.setattr(internals.llama_cpp, 'llama_detach_threadpool', lambda ctx: calls.append(('detach', ctx)))
    monkeypatch.setattr(internals.llama_cpp, 'llama_set_abort_callback',
                        lambda ctx, callback, data: calls.append((ctx, callback, data)))
    pool, data = ctypes.c_void_p(1), ctypes.c_void_p(42)
    context.attach_threadpool(pool)
    assert context._threadpool_refs == (pool, pool)
    assert calls[-2:] == [('sync', context.ctx), ('attach', context.ctx, pool, None)]
    context.set_abort_callback(lambda user_data: user_data == data.value, data)
    callback = context._abort_callback_ref
    assert callback(data) is True and context._abort_callback_data_ref is data
    assert calls[-1] == (context.ctx, callback, data)
    context.set_abort_callback(None)
    assert context._abort_callback_ref is context._abort_callback_data_ref is None
    assert not calls[-1][1]
    context.detach_threadpool()
    assert context._threadpool_refs is None
    assert calls[-2:] == [('sync', context.ctx), ('detach', context.ctx)]


def test_decode_eval_batch_resets_and_preserves_native_abort(monkeypatch):
    llm = object.__new__(llama_cpp.Llama)
    llm._batch = SimpleNamespace(batch=SimpleNamespace(n_tokens=0))
    llm._ctx = SimpleNamespace(
        decode=lambda _batch: (_ for _ in ()).throw(
            internals.LlamaDecodeAbort("native abort")
        )
    )
    llm._active_speculative_phase_stats = None
    reset_calls = []
    monkeypatch.setattr(llm, "reset", lambda: reset_calls.append(True))

    with pytest.raises(internals.LlamaDecodeAbort, match="native abort"):
        llm._decode_eval_batch([1, 2], 2)

    assert reset_calls == [True]


def test_high_level_abort_updates_python_and_native_flags():
    llm = object.__new__(llama_cpp.Llama)
    llm.verbose = False
    llm._abort_event = threading.Event()
    llm._native_abort_flag = ctypes.c_bool(False)
    data = ctypes.cast(ctypes.pointer(llm._native_abort_flag), ctypes.c_void_p)

    assert llama_cpp.llama._llama_native_abort_callback(data) is False
    llm.abort()
    assert llm._abort_event.is_set()
    assert llama_cpp.llama._llama_native_abort_callback(data) is True


@pytest.fixture(scope="module")
def vocabulary_model():
    with closing(llama_cpp.Llama(model_path=MODEL, vocab_only=True,
                                 n_gpu_layers=0, verbose=False)) as model:
        yield model


def test_model_extensions_with_vocabulary_only_model(vocabulary_model):
    model = vocabulary_model._model.model
    assert llama_cpp.llama_model_n_tokens_delay(model) == 0
    llama_cpp.llama_model_set_attn_rot(model, False)
    llama_cpp.llama_model_set_attn_rot(model, True)
    llama_cpp.llama_model_set_attn_rot(model, False)
    tokens = (llama_cpp.llama_token * 1)(0)
    output = (ctypes.c_float * 1)(123)
    assert llama_cpp.llama_model_get_tok_embd_rows(model, tokens, 1, output) == 0
    assert output[0] == 123


def test_llama_cpp_tokenization(vocabulary_model):
    """
    Test the tokenizer API (Llama.tokenize and Llama.detokenize).
    Verifies handling of BOS (Begin of Sentence), EOS (End of Sentence), and special tokens.
    """
    llama = vocabulary_model
    text = b"Hello World"

    tokens = llama.tokenize(text)
    assert tokens == [llama.token_bos(), 15043, 2787]
    assert llama.detokenize(tokens)[1:] == text

    tokens = llama.tokenize(text, add_bos=False)
    assert tokens == [15043, 2787]
    assert llama.detokenize(tokens) == text

    text = b"Hello World</s>"
    assert llama.tokenize(text) == [1, 15043, 2787, 829, 29879, 29958]
    assert llama.tokenize(text, special=True) == [1, 15043, 2787, llama.token_eos()]

    tokens = llama.tokenize(b"", add_bos=True, special=True)
    assert tokens == [llama.token_bos()]
    assert llama.detokenize(tokens) == b""


@pytest.mark.parametrize('kind', ['sequence', 'mixed', 'mrope'])
def test_legacy_batch_input_validation_and_native_layout(kind):
    with closing(internals.LlamaBatch(n_tokens=3, embd=0 if kind == 'sequence' else 2,
                                      n_seq_max=1, mixed=kind == 'mixed', verbose=False)) as batch:
        if kind == 'sequence':
            with pytest.raises(ValueError, match='n_seq_max'):
                batch.add_sequence([1], [0], [1], [True])
            assert batch.batch.n_tokens == 0
        elif kind == 'mixed':
            features = np.arange(1, 5, dtype=np.float32).reshape(2, 2)
            for i, row in enumerate(features):
                batch.add_token_embedding(10 + i, row, i, [0], i == 1)
            features[:] = -1
            np.testing.assert_array_equal(np.ctypeslib.as_array(batch.batch.embd, shape=(4,)), [1, 2, 3, 4])
            assert list(batch.batch.token[:2]) == [10, 11]
        else:
            original_pos = ctypes.cast(batch.batch.pos, ctypes.c_void_p).value
            batch.enable_mrope_positions()
            assert ctypes.cast(batch.batch.pos, ctypes.c_void_p).value != original_pos
            batch.add_embeddings_mrope([1, 2, 3, 4], pos_array=[[4, 5]] * 3 + [[0, 0]],
                                        seq_ids=[0], logits_array=[False, True])
            assert list(batch.batch.pos[:8]) == [4, 5, 4, 5, 4, 5, 0, 0]
        if kind != 'sequence':
            assert batch.batch.n_tokens == 2
            assert list(batch.batch.logits[:2]) == [0, 1]


@pytest.fixture(scope="module")
def llama_cpp_model_path():
    model_path = os.environ.get("LLAMA_TEST_TRANSFORMER_MODEL")
    if not model_path:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("LLAMA_TEST_TRANSFORMER_MODEL is required in Actions")
        pytest.skip("Set LLAMA_TEST_TRANSFORMER_MODEL to run real-model tests")
    assert os.path.isfile(model_path), model_path
    return model_path


@pytest.fixture(scope="module")
def shared_completion_model(llama_cpp_model_path):
    """Load the common completion model once for independent feature tests."""
    with closing(llama_cpp.Llama(
        llama_cpp_model_path,
        n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0")),
        verbose=False,
        n_ctx=64,
        n_batch=32,
        n_ubatch=32,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        logits_all=False,
        swa_full=True,
        kv_unified=True,
    )) as model:
        yield model


@pytest.fixture
def completion_model(shared_completion_model):
    """Give each feature test a clean context without reloading model weights."""
    shared_completion_model.reset()
    try:
        yield shared_completion_model
    finally:
        shared_completion_model.reset()


@pytest.fixture(scope="module")
def completion_logits_reference(shared_completion_model):
    model = shared_completion_model
    model.reset()
    try:
        tokens = model.tokenize(b"Hello", add_bos=True)
        model.eval(tokens)
        logits = np.ctypeslib.as_array(
            model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),)
        ).copy()
        logits.flags.writeable = False
        return tokens, logits
    finally:
        model.reset()


def test_backend_sampling_with_varying_extended_batch_outputs(completion_model):
    model = completion_model
    params = type(model.context_params).from_buffer_copy(model.context_params)
    params.n_seq_max = 2
    params.n_outputs_max = params.n_outputs_max_per_seq = 4
    params.embeddings = False
    context = internals.LlamaContext(model=model._model, params=params, verbose=False)
    sampler = internals.LlamaSampler()
    sampler.add_greedy()
    try:
        if not context.set_sampler(0, sampler):
            pytest.skip("Backend greedy sampling is unsupported by this library")
        # Attachment can succeed even when greedy falls back to CPU sampling.
        sampler_name = llama_cpp.llama_sampler_name(sampler.samplers[0])
        if sampler_name == b"-greedy":
            pytest.skip("Greedy backend sampling is unsupported by the output device")
        token = model.tokenize(b"Hello", add_bos=True)[0]
        with internals.LlamaBatchExt(context=context) as batch:
            # Keep four input entries while changing active sampling rows.
            # Sequence 1 has no sampler: padding chains must not publish tokens.
            for seq_id, outputs in ((0, 1), (0, 3), (0, 2), (0, 0), (1, 2), (0, 4)):
                context.memory_clear(True)
                sampler.reset()
                batch.reset()
                batch.add_tokens([token] * 4, positions=range(4), seq_ids=(seq_id,),
                                 outputs=[i < outputs for i in range(4)])
                assert context.decode(batch) == 0
                context.synchronize()
                assert batch.output_indices == tuple(range(outputs))
                for i in range(outputs):
                    sampled = context.get_sampled_token_ith(i)
                    if seq_id == 0:
                        assert 0 <= sampled < model.n_vocab(), (
                            f"seq_id={seq_id}, outputs={outputs}, row={i}, "
                            f"sampled={sampled}, sampler={sampler_name!r}"
                        )
                    else:
                        assert sampled == llama_cpp.LLAMA_TOKEN_NULL
    finally:
        context.close()
        sampler.close()


def test_model_token_embedding_rows_preserve_order(completion_model):
    model = completion_model._model.model
    width = llama_cpp.llama_model_n_embd(model)
    ids = completion_model.tokenize(b"Hello world", add_bos=False)
    assert len(ids) >= 2
    selected = (llama_cpp.llama_token * 3)(ids[1], ids[0], ids[1])
    output = (ctypes.c_float * (3 * width))()
    assert llama_cpp.llama_model_get_tok_embd_rows(model, selected, 3, output) == 3 * width
    rows = np.ctypeslib.as_array(output).reshape(3, width)
    assert np.isfinite(rows).all()
    for index, token in enumerate(selected):
        single = (ctypes.c_float * width)()
        assert llama_cpp.llama_model_get_tok_embd_rows(
            model, (llama_cpp.llama_token * 1)(token), 1, single
        ) == width
        np.testing.assert_array_equal(rows[index], np.ctypeslib.as_array(single))
    assert llama_cpp.llama_model_get_tok_embd_rows(model, selected, 3, None) == 0
    invalid = (llama_cpp.llama_token * 1)(-1)
    assert llama_cpp.llama_model_get_tok_embd_rows(model, invalid, 1, output) == 0


def test_context_graph_reserve_preserves_decode_results(completion_model, completion_logits_reference):
    model = completion_model
    tokens, expected = completion_logits_reference
    with internals.LlamaBatchExt(context=model._ctx) as batch:
        batch.add_tokens(tokens, positions=range(len(tokens)),
                         outputs=[False] * (len(tokens) - 1) + [True])
        assert model._ctx.decode(batch) == 0
        # decode is asynchronous; reservation must wait before resetting
        # its scheduler. Sequence memory must survive the reservation.
        position = model._ctx.memory_seq_pos_max(0)
        assert model._ctx.graph_reserve(len(tokens), 1, 1)
        assert model._ctx.memory_seq_pos_max(0) == position
        actual = np.ctypeslib.as_array(
            model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        batch.reset()
        batch.add_token(tokens[-1], len(tokens), output=True)
        assert model._ctx.decode(batch) == 0
        assert np.isfinite(np.ctypeslib.as_array(
            model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),)
        )).all()


def test_high_level_interleaved_batch_matches_native_submission(completion_model):
    model = completion_model
    ctx = model._ctx
    token = model.tokenize(b"Hello", add_bos=True)[0]
    width = model.n_embd_inp()
    embedding = np.zeros(width, dtype=np.float32)
    use_mrope = model._model.rope_type() in (
        llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_MROPE,
        llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_IMROPE,
    )
    # Compare identical submission shapes: quantized/hybrid models need
    # not produce identical logits for joint vs separate decode calls.
    native = llama_cpp.llama_batch_ext_init(ctx.ctx)
    assert native
    try:
        embd = llama_cpp.llama_embd(embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, width)
        for i in range(3):
            idx = (
                llama_cpp.llama_batch_ext_add_embd(native, 0, embd) if i == 1
                else llama_cpp.llama_batch_ext_add_token(native, 0, token)
            )
            assert idx == i
            pos = (llama_cpp.llama_pos * 4)(i, i, i, 0)
            assert llama_cpp.llama_batch_ext_set_pos(native, idx, pos)
            assert llama_cpp.llama_batch_ext_set_output_logits(native, idx, i == 2)
        assert llama_cpp.llama_process(ctx.ctx, llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, native) == 0
        expected = np.ctypeslib.as_array(ctx.get_logits_ith(-1), shape=(model.n_vocab(),)).copy()
    finally:
        llama_cpp.llama_batch_ext_free(native)
    model.reset()
    with internals.LlamaBatchExt(context=ctx) as batch:
        batch.add_token(token, 0)
        batch.add_embedding(embedding, (1, 1, 1, 0) if use_mrope else 1)
        batch.add_token(token, 2, output=True)
        assert ctx.decode(batch) == 0
        actual = np.ctypeslib.as_array(ctx.get_logits_ith(-1), shape=(model.n_vocab(),))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def shared_mtp_runtime():
    model_path = os.environ.get("LLAMA_TEST_HYBRID_MODEL")
    if not model_path:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("LLAMA_TEST_HYBRID_MODEL is required in Actions")
        pytest.skip("Set LLAMA_TEST_HYBRID_MODEL to run real MTP tests")
    assert os.path.isfile(model_path), model_path
    with closing(llama_cpp.Llama(
        model_path, load_mtp=True, n_ctx=64, n_batch=32, n_ubatch=32,
        n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0")),
        verbose=False,
    )) as model:
        if model._model.n_layer_nextn() == 0:
            pytest.fail("Configured hybrid model must have MTP heads")
        params = type(model.context_params).from_buffer_copy(model.context_params)
        params.ctx_type = llama_cpp.llama_context_type.LLAMA_CONTEXT_TYPE_MTP
        params.ctx_other = model._ctx.ctx
        params.n_outputs_max = params.n_outputs_max_per_seq = 1
        with closing(internals.LlamaContext(model=model._model, params=params, verbose=False)) as draft:
            yield model, draft


@pytest.fixture
def mtp_runtime(shared_mtp_runtime):
    model, draft = shared_mtp_runtime
    draft.memory_clear(True)
    try:
        yield model, draft
    finally:
        draft.memory_clear(True)
        model.reset()


@pytest.mark.parametrize("input_kind", ["token", "embedding"])
def test_extended_state_batch_matches_native_mtp(mtp_runtime, input_kind):
    model, draft = mtp_runtime
    width = model._model.n_embd_out()
    rng = np.random.default_rng(42)
    hidden = rng.normal(0, 0.1, width).astype(np.float32)
    token = model.tokenize(b"Hello", add_bos=True)[0]
    embedding = rng.normal(0, 0.1, model._model.n_embd_inp()).astype(np.float32)
    mrope = model._model.rope_type() in (
        llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_MROPE,
        llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_IMROPE,
    )
    position = (0, 1, 2, 0) if mrope else 0
    if input_kind == "token":
        with closing(internals.LlamaBatch(n_tokens=1, embd=width, n_seq_max=1, mixed=True)) as old:
            old.add_token_embedding(token, hidden, 0, [0], True)
            assert draft.decode(old) == 0
    else:
        native = llama_cpp.llama_batch_ext_init(draft.ctx)
        assert native
        try:
            embd = llama_cpp.llama_embd(embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, embedding.size)
            state = llama_cpp.llama_embd(hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, hidden.size)
            idx = llama_cpp.llama_batch_ext_add_embd(native, 0, embd)
            assert idx == 0
            assert llama_cpp.llama_batch_ext_set_embd_state(native, idx, state)
            pos = (llama_cpp.llama_pos * 4)(*(position if mrope else (position,)))
            assert llama_cpp.llama_batch_ext_set_pos(native, idx, pos)
            assert llama_cpp.llama_batch_ext_set_output_logits(native, idx, True)
            assert llama_cpp.llama_process(draft.ctx, llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, native) == 0
        finally:
            llama_cpp.llama_batch_ext_free(native)
    expected = np.ctypeslib.as_array(draft.get_logits_ith(-1), shape=(model.n_vocab(),)).copy()
    draft.memory_clear(True)
    with internals.LlamaBatchExt(context=draft) as batch:
        if input_kind == "token":
            batch.add_token(token, 0, output=True, state=hidden)
        else:
            batch.add_embedding(embedding, position, output=True, state=hidden)
        assert draft.decode(batch) == 0
        actual = np.ctypeslib.as_array(draft.get_logits_ith(-1), shape=(model.n_vocab(),))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_extended_batch_native_gc_and_context_cleanup(completion_model, monkeypatch):
    model = completion_model
    free = internals.llama_cpp.llama_batch_ext_free
    released = []

    def release(pointer):
        free(pointer)
        released.append(pointer)

    monkeypatch.setattr(internals.llama_cpp, "llama_batch_ext_free", release)
    data = np.zeros((2, model.n_embd_inp()), dtype=np.float32)
    mrope = model._model.rope_type() in (
        llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_MROPE,
        llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_IMROPE,
    )
    positions = [(0, 0, 0, 0), (1, 1, 1, 0)] if mrope else [0, 1]
    for _ in range(3):
        batch = internals.LlamaBatchExt(context=model._ctx)
        batch.add_embeddings(data, positions=positions)
        batch.render()
        batch.cycle = batch
        reference = weakref.ref(batch)
        del batch
        gc.collect()
        assert reference() is None
    assert len(released) == 3
    context = internals.LlamaContext(model=model._model, params=model.context_params, verbose=False)
    try:
        batch = internals.LlamaBatchExt(context=context)
        batch.add_embeddings(data, positions=positions)
        batch.render()
        storage = weakref.ref(batch.entries[0].embedding.base)
        context.close()
        assert storage() is None
        assert len(released) == 4
        batch.close()
        assert len(released) == 4
    finally:
        context.close()


def test_extended_batch_matches_legacy_decode(completion_model, completion_logits_reference, tmp_path):
    model = completion_model
    tokens, expected = completion_logits_reference
    batch = llama_cpp.llama_batch_ext_init(model._ctx.ctx)
    assert batch
    try:
        assert llama_cpp.llama_batch_ext_add_token(batch, -1, tokens[0]) == -3
        assert llama_cpp.llama_batch_ext_add_token(batch, 0, llama_cpp.LLAMA_TOKEN_NULL) == -2
        # Invalid token input leaves an empty entry in the native batch.
        llama_cpp.llama_batch_ext_clear(batch)
        for pos, token in enumerate(tokens):
            idx = llama_cpp.llama_batch_ext_add_token(batch, 0, token)
            assert idx == pos
            if pos == 0:
                assert llama_cpp.llama_batch_ext_add_seq(batch, idx, 0)
                assert not llama_cpp.llama_batch_ext_add_seq(batch, idx, model.context_params.n_seq_max)
            assert llama_cpp.llama_batch_ext_set_pos(batch, idx, ctypes.byref(llama_cpp.llama_pos(pos)))
            assert llama_cpp.llama_batch_ext_set_output_logits(batch, idx, pos == len(tokens) - 1)
        assert llama_cpp.llama_process(model._ctx.ctx, llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, batch) == 0
        actual = np.ctypeslib.as_array(model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert llama_cpp.llama_state_seq_get_size(model._ctx.ctx, 0) == llama_cpp.llama_state_seq_get_size_ext(
            model._ctx.ctx, 0, llama_cpp.LLAMA_STATE_SEQ_FLAGS_NONE
        )

        session_path = os.fsencode(tmp_path / "session.bin")
        token_array = (llama_cpp.llama_token * len(tokens))(*tokens)
        assert llama_cpp.llama_state_save_file(model._ctx.ctx, session_path, token_array, len(tokens)) is True
        sequence_path = tmp_path / "sequence.bin"
        assert llama_cpp.llama_state_seq_save_file(
            model._ctx.ctx, os.fsencode(sequence_path), 0, token_array, len(tokens)
        ) > 0
        for path, magic, version in (
            (tmp_path / "session.bin", llama_cpp.LLAMA_SESSION_MAGIC, llama_cpp.LLAMA_SESSION_VERSION),
            (sequence_path, llama_cpp.LLAMA_STATE_SEQ_MAGIC, llama_cpp.LLAMA_STATE_SEQ_VERSION),
        ):
            with path.open("rb") as saved:
                header = (ctypes.c_uint32 * 2).from_buffer_copy(saved.read(8))
            assert tuple(header) == (magic, version)
        model.reset()
        restored = (llama_cpp.llama_token * len(tokens))()
        count = ctypes.c_size_t()
        assert llama_cpp.llama_state_load_file(
            model._ctx.ctx, session_path, restored, len(tokens), ctypes.byref(count)
        ) is True
        assert count.value == len(tokens)
        assert list(restored) == tokens
        assert llama_cpp.llama_state_load_file(
            model._ctx.ctx, os.fsencode(tmp_path / "missing.bin"), restored, len(tokens), ctypes.byref(count)
        ) is False

        # Compare embedding input and the by-value llama_embd setters.
        model.reset()
        width = model.n_embd_inp()
        zeros = (ctypes.c_float * width)()
        embd = llama_cpp.llama_embd(zeros, 1, width)
        use_mrope = model._model.rope_type() in (
            llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_MROPE,
            llama_cpp.llama_rope_type.LLAMA_ROPE_TYPE_IMROPE,
        )
        legacy = internals.LlamaBatch(n_tokens=2, embd=width, n_seq_max=1)
        try:
            embedding_data = np.zeros(2 * width, dtype=np.float32)
            if use_mrope:
                legacy.add_embeddings_mrope(
                    embedding_data, pos_array=[[0, 1]] * 3 + [[0, 0]],
                    seq_ids=[0], logits_array=[False, True],
                )
            else:
                legacy.add_embeddings(
                    embedding_data, pos_array=[0, 1],
                    seq_ids=[0], logits_array=[False, True],
                )
            assert model._ctx.decode(legacy) == 0
            expected = np.ctypeslib.as_array(
                model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),)
            ).copy()
        finally:
            legacy.close()
        model.reset()
        llama_cpp.llama_batch_ext_clear(batch)
        assert llama_cpp.llama_batch_ext_add_embd(batch, 0, embd) == 0
        assert llama_cpp.llama_batch_ext_add(batch, 0) == 1
        assert llama_cpp.llama_batch_ext_set_embd_token(batch, 1, embd)
        assert not llama_cpp.llama_batch_ext_set_embd_state(batch, 1, embd)
        for pos in range(2):
            positions = (llama_cpp.llama_pos * 4)(pos, pos, pos, 0)
            assert llama_cpp.llama_batch_ext_set_pos(batch, pos, positions)
            assert llama_cpp.llama_batch_ext_set_output_embd(batch, pos, pos == 1)
        assert llama_cpp.llama_process(model._ctx.ctx, llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE, batch) == 0
        actual = np.ctypeslib.as_array(model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        llama_cpp.llama_batch_ext_clear(batch)
        for idx in range(model.n_batch):
            assert llama_cpp.llama_batch_ext_add_token(batch, 0, tokens[0]) == idx
        assert llama_cpp.llama_batch_ext_add(batch, 0) == -1
    finally:
        llama_cpp.llama_batch_ext_free(batch)


def test_native_sampler_chain_decode(completion_model):
    model = completion_model
    assert model.tokenize(b'Hello, world!', special=True) == [9707, 11, 1879, 0]
    tokens = model.tokenize(b'The quick brown fox jumps', special=True)
    with closing(internals.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)) as batch, \
            closing(internals.LlamaSampler()) as sampler:
        sampler.add_top_k(50)
        sampler.add_top_p(0.9, 1)
        sampler.add_temp(0.8)
        sampler.add_dist(1337)
        batch.add_sequence(tokens, list(range(len(tokens))), [0],
                           [False] * (len(tokens) - 1) + [True])
        assert model._ctx.decode(batch) == 0
        for pos in range(len(tokens), len(tokens) + 4):
            token = sampler.sample(model._ctx, -1)
            assert 0 <= token < model.n_vocab()
            sampler.accept(token)
            batch.reset()
            batch.add_token(token, pos, [0], True)
            assert model._ctx.decode(batch) == 0
        assert model._ctx.memory_seq_pos_max(0) == len(tokens) + 3


@pytest.mark.parametrize("stream", [False, True])
def test_native_decode_abort_finishes_and_resets_context(
    completion_model, monkeypatch, stream
):
    """A native decode abort must be a normal, reusable completion boundary."""

    class RecordingCache:
        def __init__(self):
            self.writes = []

        def __getitem__(self, _key):
            raise KeyError

        def __setitem__(self, key, value):
            self.writes.append((key, value))

    def abort_decode(_batch):
        raise internals.LlamaDecodeAbort("native abort")

    cache = RecordingCache()
    monkeypatch.setattr(completion_model._ctx, "decode", abort_decode)
    monkeypatch.setattr(completion_model, "cache", cache)

    output = completion_model.create_completion(
        "Abort this request during prompt evaluation",
        max_tokens=4,
        stream=stream,
    )
    result = list(output)[-1] if stream else output

    assert result["choices"][0]["finish_reason"] == "abort"
    assert completion_model.n_tokens == 0
    assert completion_model._ctx.memory_seq_pos_min(0) == -1
    assert completion_model._ctx.memory_seq_pos_max(0) == -1
    assert cache.writes == []


@pytest.mark.parametrize("rollback", [False, True], ids=["append", "rollback"])
def test_real_transformer_prefix_and_state(completion_model, rollback):
    model = completion_model
    assert not model.is_hybrid
    tokens = model.tokenize(b"The capital of France is")
    def complete():
        return model.create_completion(tokens, max_tokens=4, temperature=0,
                                       seed=42)["choices"][0]["text"]
    reference = complete()
    model.reset()
    model.eval(tokens[:-1])
    if rollback:
        model.eval(tokens[-1:])
    assert complete() == reference
    state = model.save_state()
    model.eval(tokens[-1:])
    model.load_state(state)
    assert model.n_tokens == state.n_tokens
    np.testing.assert_array_equal(model._restored_logits, state.last_logits)


@pytest.mark.parametrize('control', ['grammar', 'logit_bias', 'processor'])
def test_sampling_constraints_reach_generated_output(completion_model, control):
    model = completion_model
    if control == 'grammar':
        grammar = llama_cpp.LlamaGrammar.from_string(
            'root ::= "{" [ ]? "\\\"age\\\"" ":" [ ]? [0-9]+ "}"')
        result = model.create_completion('Generate a JSON with age:', max_tokens=20,
                                         grammar=grammar, temperature=0)
        parsed = json.loads(result['choices'][0]['text'])
        assert len(parsed) == 1 and isinstance(parsed['age'], int)
    else:
        token = model.tokenize(b' banana', add_bos=False)[0]
        options = {'logit_bias': {token: 100.0}}
        if control == 'processor':
            def force_token(input_ids, scores):
                assert len(input_ids) > 0
                forced = np.full_like(scores, -np.inf)
                forced[token] = 0
                return forced
            options = {'logits_processor': llama_cpp.LogitsProcessorList([force_token])}
        result = model.create_completion('I like to eat', max_tokens=1,
                                         temperature=0, **options)
        assert result['choices'][0]['text'] == model.detokenize([token]).decode()


@pytest.mark.parametrize("model_class,n_seq_max", [(llama_cpp.Llama, 2), (LlamaEmbedding, 1)])
def test_real_llama_base_embedding_api(llama_cpp_model_path, model_class, n_seq_max):
    """
    Test the maintained embedding API on the standard Llama class.

    Covers pre-tokenized batching, normalization, separator-based string
    batching, token counts, and the OpenAI-compatible response wrapper.
    """
    model = model_class(
        model_path=llama_cpp_model_path,
        n_gpu_layers=int(os.environ.get("LLAMA_TEST_N_GPU_LAYERS", "0")),
        embeddings=True,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        n_seq_max=n_seq_max,
        kv_unified=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
        verbose=False,
    )

    try:
        token_inputs = [
            model.tokenize(b"Hello"),
            model.tokenize(b"world"),
        ]
        embeddings, token_count = model.embed(
            token_inputs,
            normalize=True,
            return_count=True,
        )

        assert len(embeddings) == len(token_inputs)
        assert token_count == sum(map(len, token_inputs))
        assert len(embeddings[0]) == len(token_inputs[0])
        assert np.linalg.norm(embeddings[0][0]) == pytest.approx(1.0)

        split_embeddings = model.embed(
            "Hello\nworld",
            separator="\n",
            normalize=False,
        )
        assert len(split_embeddings) == 2

        response = model.create_embedding(
            ["Hello", "world"],
            normalize=2,
        )
        assert response["object"] == "list"
        assert len(response["data"]) == 2
        assert response["usage"]["prompt_tokens"] > 0
        assert response["usage"]["total_tokens"] == response["usage"]["prompt_tokens"]
    finally:
        model.close()
