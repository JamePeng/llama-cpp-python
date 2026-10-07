import os
import ctypes
import json
import multiprocessing
import threading
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

import llama_cpp
import llama_cpp._internals as internals
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


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

    def embedding(_, idx, embd):
        if state.fail:
            return False
        rows[idx]["embedding"] = np.ctypeslib.as_array(embd.data, shape=(embd.n_rows * embd.n_embd,)).copy()
        return True

    def add_embd(p, sid, embd):
        idx = add(p, sid, None)
        return idx if embedding(p, idx, embd) else -2

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
    monkeypatch.setattr(lib, "llama_batch_ext_set_embd_token", embedding)
    monkeypatch.setattr(lib, "llama_batch_ext_add_seq", sequence)
    monkeypatch.setattr(lib, "llama_batch_ext_set_pos", position)
    monkeypatch.setattr(lib, "llama_batch_ext_set_output_logits", output)
    monkeypatch.setattr(lib, "llama_process", process)
    try:
        with internals.LlamaBatchExt(context=context) as batch:
            yield SimpleNamespace(context=context, batch=batch, rows=rows, calls=calls, freed=freed, state=state)
    finally:
        context.ctx = None


class MockLoraAdapter:
    def __init__(self, address):
        self.adapter = ctypes.cast(ctypes.c_void_p(address), internals.llama_cpp.llama_adapter_lora_p_ctypes)


@pytest.mark.parametrize("method,args,native", [
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
])
def test_closed_context_rejected_before_native_call(runtime_context, monkeypatch, method, args, native):
    context = runtime_context
    call = Mock()
    monkeypatch.setattr(internals.llama_cpp, native, call)
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        getattr(context, method)(*args)
    call.assert_not_called()


def test_graph_reserve_synchronizes_before_scheduler_reset(runtime_context, monkeypatch):
    context = runtime_context
    calls = []
    graph = object()
    monkeypatch.setattr(internals.llama_cpp, "llama_n_seq_max", lambda _: 2)
    monkeypatch.setattr(internals.llama_cpp, "llama_synchronize", lambda _: calls.append("sync"))

    def reserve(*args):
        calls.append(args)
        return graph

    monkeypatch.setattr(internals.llama_cpp, "llama_graph_reserve", reserve)
    # The native API rounds three tokens up to four for two sequences.
    assert context.graph_reserve(3, 2, 4) is graph
    assert calls == ["sync", (123, 3, 2, 4)]


@pytest.mark.parametrize("dimensions,error", [
    ((0, 1, 1), ValueError),
    ((1, 0, 1), ValueError),
    ((1, 1, 0), ValueError),
    ((-1, 1, 1), ValueError),
    ((2**32, 1, 1), ValueError),
    ((1, 3, 1), ValueError),
    ((1, 1, 2), ValueError),
    ((2**31 - 1, 2, 1), ValueError),
    ((True, 1, 1), TypeError),
    ((1.5, 1, 1), TypeError),
])
def test_graph_reserve_invalid_dimensions_do_not_touch_scheduler(runtime_context, monkeypatch, dimensions, error):
    context = runtime_context
    reserve = Mock()
    sync = Mock()
    monkeypatch.setattr(internals.llama_cpp, "llama_n_seq_max", lambda _: 2)
    monkeypatch.setattr(internals.llama_cpp, "llama_graph_reserve", reserve)
    monkeypatch.setattr(internals.llama_cpp, "llama_synchronize", sync)
    with pytest.raises(error):
        context.graph_reserve(*dimensions)
    reserve.assert_not_called()
    sync.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_graph_reserve_reports_allocation_or_symbol_failure(runtime_context, monkeypatch, missing):
    context = runtime_context
    monkeypatch.setattr(internals.llama_cpp, "llama_n_seq_max", lambda _: 1)
    monkeypatch.setattr(internals.llama_cpp, "llama_synchronize", Mock())
    reserve = Mock(side_effect=RuntimeError("missing symbol")) if missing else Mock(return_value=None)
    monkeypatch.setattr(internals.llama_cpp, "llama_graph_reserve", reserve)
    with pytest.raises(RuntimeError, match="missing symbol" if missing else "failed to reserve"):
        context.graph_reserve(1, 1, 1)
    assert context.ctx == 123


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


@pytest.mark.parametrize("failure", ["embedding", "position"])
def test_batch_render_failure_clears_partial_rows_and_can_retry(batch_runtime, monkeypatch, failure):
    r = batch_runtime
    r.batch.add_embedding([1, 2], (0, 0, 0, 0))
    r.batch.add_embedding([3, 4], (1, 1, 1, 0))
    original = internals.llama_cpp.llama_batch_ext_set_pos
    if failure == "embedding":
        r.state.fail = True
    else:
        monkeypatch.setattr(internals.llama_cpp, "llama_batch_ext_set_pos", lambda _, idx, pos: idx == 0)
    with pytest.raises(RuntimeError, match="-2" if failure == "embedding" else "position at logical entry 1"):
        r.context.decode(r.batch)
    assert r.rows == [] and r.calls == []
    assert len(r.batch) == 2 and r.batch.submitted_indices == ()
    r.state.fail = False
    monkeypatch.setattr(internals.llama_cpp, "llama_batch_ext_set_pos", original)
    assert r.context.decode(r.batch) == 0


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


@pytest.mark.parametrize("extended", [False, True], ids=["legacy", "extended"])
@pytest.mark.parametrize("status", [0, 1, 2, -1, -2])
def test_context_decode_status_and_checkpoint_invalidation(batch_runtime, monkeypatch, status, extended):
    r = batch_runtime
    r.batch.add_token(1, 0)
    r.state.status = status
    invalidations = []
    r.context._invalidate_checkpoints = lambda: invalidations.append(True)
    kind = internals.llama_cpp.llama_process_type.LLAMA_PROCESS_TYPE_DECODE
    monkeypatch.setattr(internals.llama_cpp, "llama_decode", lambda *_: status)
    decode = (lambda: r.context.process(r.batch, kind)) if extended else (lambda: r.context.decode(SimpleNamespace(batch=object())))
    if status in (0, 1):
        assert decode() == status
        assert invalidations == []
    else:
        error = internals.LlamaDecodeAbort if status == 2 else RuntimeError
        with pytest.raises(error):
            decode()
        assert invalidations == [True]


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


def test_reset_releases_owned_embedding_storage(batch_runtime):
    r = batch_runtime
    r.batch.add_embeddings(np.ones((2, 4), dtype=np.float32), positions=[(0, 0, 0, 0), (1, 1, 1, 0)])
    snapshot = r.batch.entries
    owner = snapshot[0].embedding.base
    assert owner is snapshot[1].embedding.base
    storage = weakref.ref(owner)
    del owner
    r.batch.render()
    r.batch.reset()
    # A caller retaining entries intentionally retains their data.
    assert storage() is not None
    del snapshot
    gc.collect()
    assert storage() is None


def test_gc_frees_native_batch_once_even_in_a_cycle(batch_runtime):
    r = batch_runtime
    batch = internals.LlamaBatchExt(context=r.context)
    batch.add_embedding([1, 2], (0, 0, 0, 0))
    batch.cycle = batch
    reference = weakref.ref(batch)
    del batch
    gc.collect()
    assert reference() is None
    assert r.freed == [456]
    gc.collect()
    assert r.freed == [456]


def test_close_releases_context_keepalive(batch_runtime):
    context = internals.LlamaContext.__new__(internals.LlamaContext)
    context.__dict__.update({k: v for k, v in batch_runtime.context.__dict__.items() if k != "_batch_ext_refs"})
    reference = weakref.ref(context)
    batch = internals.LlamaBatchExt(context=context)
    del context
    gc.collect()
    assert reference() is not None
    batch.close()
    gc.collect()
    assert reference() is None


def test_close_failure_disarms_native_ownership_and_releases_python_data(batch_runtime, monkeypatch):
    calls = []

    def fail(pointer):
        calls.append(pointer)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(internals.llama_cpp, "llama_batch_ext_free", fail)
    batch = internals.LlamaBatchExt(context=batch_runtime.context)
    batch.add_embedding([1, 2], (0, 0, 0, 0))
    storage = weakref.ref(batch.entries[0].embedding)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        batch.close()
    gc.collect()
    assert storage() is None and len(batch) == 0
    assert batch._context is None
    batch.close()
    del batch
    gc.collect()
    assert calls == [456]


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


@pytest.mark.parametrize("resource", ["context", "batch", "batch_finalizer"])
def test_runtime_init_failure_releases_only_owned_native_handles(runtime_context, monkeypatch, resource):
    free = Mock()
    if resource == "context":
        monkeypatch.setattr(internals.llama_cpp, "llama_init_from_model", lambda *_: None)
        monkeypatch.setattr(internals.llama_cpp, "llama_free", free)
        create = lambda: internals.LlamaContext(
            model=SimpleNamespace(model=123), params=internals.llama_cpp.llama_context_default_params())
        error, message = RuntimeError, "Failed to create"
    else:
        monkeypatch.setattr(internals.llama_cpp, "llama_batch_ext_init", lambda _: 456 if resource == "batch_finalizer" else None)
        monkeypatch.setattr(internals.llama_cpp, "llama_batch_ext_free", free)
        if resource == "batch_finalizer":
            def fail(*args):
                raise MemoryError("finalizer allocation failed")
            monkeypatch.setattr(internals.weakref, "finalize", fail)
        create = lambda: internals.LlamaBatchExt(context=runtime_context)
        error, message = MemoryError, "finalizer allocation" if resource == "batch_finalizer" else "init failed"
    with pytest.raises(error, match=message):
        create()
    gc.collect()
    if resource == "batch_finalizer":
        free.assert_called_once_with(456)
    else:
        free.assert_not_called()


@pytest.mark.parametrize("entry", ["create_completion", "__call__", "create_chat_completion"])
@pytest.mark.parametrize("present,expected", [(0.0, 1.5), (0.7, 0.7)])
def test_public_generation_options_reach_execution(entry, present, expected):
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


def test_model_init_frees_native_model_when_vocab_lookup_fails(monkeypatch):
    native_model_handle = object()
    freed_model_handles = []

    def model_path_exists(_path):
        return True

    def load_native_model(_path, _params):
        return native_model_handle

    def fail_to_get_model_vocab(_model_handle):
        return None

    def record_model_free(model_handle):
        freed_model_handles.append(model_handle)

    monkeypatch.setattr(internals.os.path, "exists", model_path_exists)
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_load_from_file",
        load_native_model,
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_get_vocab",
        fail_to_get_model_vocab,
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_free",
        record_model_free,
    )

    with pytest.raises(ValueError, match="Failed to get vocab"):
        internals.LlamaModel(
            path_model="model.gguf",
            params=object(),
            verbose=False,
        )

    assert freed_model_handles == [native_model_handle]


def test_batch_init_frees_native_batch_when_validation_fails(monkeypatch):
    class InvalidMixedNativeBatch:
        token = object()
        embd = object()

    invalid_mixed_batch = InvalidMixedNativeBatch()
    freed_batch_handles = []

    def allocate_invalid_mixed_batch(_n_tokens, _embd, _n_seq_max):
        return invalid_mixed_batch

    def record_batch_free(batch_handle):
        freed_batch_handles.append(batch_handle)

    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_batch_init",
        allocate_invalid_mixed_batch,
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_batch_free",
        record_batch_free,
    )

    with pytest.raises(RuntimeError, match="expected batch.token to be NULL"):
        internals.LlamaBatch(
            n_tokens=1,
            embd=1,
            n_seq_max=1,
            mixed=True,
            verbose=False,
        )

    assert freed_batch_handles == [invalid_mixed_batch]


def test_context_manages_borrowed_threadpool_references(runtime_context, monkeypatch):
    context = runtime_context
    context._threadpool_refs = None
    calls = []

    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_synchronize",
        lambda ctx: calls.append(("synchronize", ctx)),
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_attach_threadpool",
        lambda ctx, pool, batch_pool: calls.append(
            ("attach", ctx, pool, batch_pool)
        ),
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_detach_threadpool",
        lambda ctx: calls.append(("detach", ctx)),
    )

    pool = ctypes.c_void_p(1)
    context.attach_threadpool(pool)
    assert context._threadpool_refs == (pool, pool)
    assert calls[-2:] == [
        ("synchronize", context.ctx),
        ("attach", context.ctx, pool, None),
    ]

    context.detach_threadpool()
    assert context._threadpool_refs is None
    assert calls[-2:] == [
        ("synchronize", context.ctx),
        ("detach", context.ctx),
    ]


def test_context_manages_native_abort_callback_lifetime(runtime_context, monkeypatch):
    context = runtime_context
    context._abort_callback_ref = None
    context._abort_callback_data_ref = None
    calls = []

    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_set_abort_callback",
        lambda ctx, callback, data: calls.append((ctx, callback, data)),
    )

    data = ctypes.c_void_p(42)
    context.set_abort_callback(lambda user_data: user_data == data.value, data)

    callback = context._abort_callback_ref
    assert callback(data) is True
    assert context._abort_callback_data_ref is data
    assert calls[-1] == (context.ctx, callback, data)

    context.set_abort_callback(None)
    assert context._abort_callback_ref is None
    assert context._abort_callback_data_ref is None
    assert bool(calls[-1][1]) is False


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


def test_sampling_context_partial_init_can_close_idempotently(monkeypatch):
    closed_resources = []

    class MinimalModelForSampling:
        model = object()
        verbose = False

        def n_vocab(self):
            return 8

    class TrackedTokenDataArray:
        def __init__(self, *, n_vocab):
            assert n_vocab == 8

        def close(self):
            closed_resources.append("token-data")

    class TrackedSamplerChain:
        def close(self):
            closed_resources.append("sampler-chain")

    def get_sampling_vocab(_model_handle):
        return object()

    def fail_sampler_chain_build(_sampling_context):
        raise RuntimeError("sampler chain build failed")

    monkeypatch.setattr(internals, "LlamaTokenDataArray", TrackedTokenDataArray)
    monkeypatch.setattr(internals, "LlamaSampler", TrackedSamplerChain)
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_get_vocab",
        get_sampling_vocab,
    )
    monkeypatch.setattr(
        internals.LlamaSamplingContext,
        "_build_sampler_chain",
        fail_sampler_chain_build,
    )

    sampling_context = internals.LlamaSamplingContext.__new__(
        internals.LlamaSamplingContext
    )
    with pytest.raises(RuntimeError, match="sampler chain build failed"):
        sampling_context.__init__(
            params=internals.LlamaSamplingParams(),
            model=MinimalModelForSampling(),
        )

    sampling_context.close()
    sampling_context.close() # Closing an already closed context must be a no-op.

    # Full-vocabulary candidate storage is lazy and was never needed because
    # sampler-chain construction failed first.
    assert closed_resources == ["sampler-chain"]
    assert sampling_context.model is None
    assert sampling_context.params is None
    assert sampling_context.vocab is None


def test_llama_cpp_tokenization():
    """
    Test the tokenizer API (Llama.tokenize and Llama.detokenize).
    Verifies handling of BOS (Begin of Sentence), EOS (End of Sentence), and special tokens.
    """
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)

    try:
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
    finally:
        llama.close()


def test_llama_batch_seq_id_error_guidance():
    """Sequence-capacity errors should explain how to fix parallel batching."""
    batch = internals.LlamaBatch(
        n_tokens=2,
        embd=0,
        n_seq_max=1,
        verbose=False,
    )
    try:
        with pytest.raises(ValueError) as exc_info:
            batch.add_sequence(
                token_array=[1],
                pos_array=[0],
                seq_ids=[1],
                logits_array=[True],
            )

        message = str(exc_info.value)
        assert "n_seq_max=1" in message
        assert "valid IDs are 0 through 0" in message
        assert "n_seq_max>=2" in message
        assert "LlamaEmbedding" in message
    finally:
        batch.close()


def test_llama_batch_mixed_embeddings_are_copied_contiguously():
    batch = internals.LlamaBatch(
        n_tokens=2,
        embd=4,
        n_seq_max=1,
        mixed=True,
        verbose=False,
    )
    try:
        first = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        second = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        batch.add_token_embedding(10, first, 0, [0], False)
        batch.add_token_embedding(11, second, 1, [0], True)

        actual = np.ctypeslib.as_array(batch.batch.embd, shape=(8,)).copy()
        np.testing.assert_array_equal(actual, np.concatenate((first, second)))
        assert batch.batch.n_tokens == 2
        assert [batch.batch.token[i] for i in range(2)] == [10, 11]
    finally:
        batch.close()


def test_llama_batch_mrope_embeddings_use_four_position_planes():
    batch = internals.LlamaBatch(
        n_tokens=3,
        embd=2,
        n_seq_max=1,
        verbose=False,
    )
    try:
        original_pos = ctypes.cast(batch.batch.pos, ctypes.c_void_p).value
        batch.enable_mrope_positions()
        expanded_pos = ctypes.cast(batch.batch.pos, ctypes.c_void_p).value
        batch.add_embeddings_mrope(
            [1.0, 2.0, 3.0, 4.0],
            pos_array=[[4, 5], [4, 5], [4, 5], [0, 0]],
            seq_ids=[0],
            logits_array=[False, True],
        )

        assert expanded_pos != original_pos
        assert [batch.batch.pos[i] for i in range(8)] == [
            4, 5, 4, 5, 4, 5, 0, 0
        ]
        assert [batch.batch.logits[i] for i in range(2)] == [0, 1]
        assert batch.batch.n_tokens == 2
    finally:
        batch.close()


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
    model = llama_cpp.Llama(
        llama_cpp_model_path,
        n_ctx=64,
        n_batch=32,
        n_ubatch=32,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        logits_all=False,
        swa_full=True,
        kv_unified=True,
    )
    try:
        yield model
    finally:
        model.close()


@pytest.fixture
def completion_model(shared_completion_model):
    """Give each feature test a clean context without reloading model weights."""
    shared_completion_model.reset()
    try:
        yield shared_completion_model
    finally:
        shared_completion_model.reset()


def test_context_graph_reserve_preserves_decode_results(completion_model):
    model = completion_model
    tokens = model.tokenize(b"Hello", add_bos=True)
    model.reset()
    try:
        model.eval(tokens)
        expected = np.ctypeslib.as_array(
            model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),)
        ).copy()
        model.reset()
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
            ).copy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            batch.reset()
            batch.add_token(tokens[-1], len(tokens), output=True)
            assert model._ctx.decode(batch) == 0
            assert np.isfinite(np.ctypeslib.as_array(
                model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),)
            )).all()
    finally:
        model.reset()


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
    model.reset()
    try:
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
    finally:
        model.reset()


def test_high_level_paired_batch_matches_legacy_mtp(llama_cpp_model_path):
    model = llama_cpp.Llama(llama_cpp_model_path, load_mtp=True, n_ctx=64, n_batch=32, n_ubatch=32, verbose=False)
    try:
        if model._model.n_layer_nextn() == 0:
            pytest.skip("Model has no MTP heads")
        params = type(model.context_params).from_buffer_copy(model.context_params)
        params.ctx_type = llama_cpp.llama_context_type.LLAMA_CONTEXT_TYPE_MTP
        params.ctx_other = model._ctx.ctx
        params.n_outputs_max = params.n_outputs_max_per_seq = 1
        draft = internals.LlamaContext(model=model._model, params=params, verbose=False)
        try:
            width = model._model.n_embd_out()
            hidden = np.zeros(width, dtype=np.float32)
            token = model.tokenize(b"Hello", add_bos=True)[0]
            old = internals.LlamaBatch(n_tokens=1, embd=width, n_seq_max=1, mixed=True)
            try:
                draft.memory_clear(True)
                old.add_token_embedding(token, hidden, 0, [0], True)
                assert draft.decode(old) == 0
                expected = np.ctypeslib.as_array(draft.get_logits_ith(-1), shape=(model.n_vocab(),)).copy()
            finally:
                old.close()
            draft.memory_clear(True)
            with internals.LlamaBatchExt(context=draft) as batch:
                batch.add_token_embedding(token, hidden, 0, output=True)
                assert draft.decode(batch) == 0
                actual = np.ctypeslib.as_array(draft.get_logits_ith(-1), shape=(model.n_vocab(),))
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        finally:
            draft.close()
    finally:
        model.close()


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


def test_extended_batch_matches_legacy_decode(completion_model, tmp_path):
    model = completion_model
    tokens = model.tokenize(b"Hello", add_bos=True)
    model.eval(tokens)
    expected = np.ctypeslib.as_array(
        model._ctx.get_logits_ith(-1), shape=(model.n_vocab(),)
    ).copy()
    model.reset()
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


def test_real_model(llama_cpp_model_path):
    """
    Test the Low-Level API (internals.*).
    This manually constructs the Model, Context, Batch, and Sampler Chain.
    """
    # 1. Setup Model Parameters
    params = llama_cpp.llama_model_default_params()
    params.check_tensors = False

    # 2. Load the Model
    model = internals.LlamaModel(path_model=llama_cpp_model_path, params=params)

    # 3. Setup Context Parameters
    cparams = llama_cpp.llama_context_default_params()
    cparams.n_ctx = 32
    cparams.n_batch = 16
    cparams.n_ubatch = 16
    cparams.n_threads = multiprocessing.cpu_count()
    cparams.n_threads_batch = multiprocessing.cpu_count()
    cparams.swa_full = True
    cparams.kv_unified = True

    # 4. Create the Context
    context = internals.LlamaContext(model=model, params=cparams)
    tokens = model.tokenize(b"Hello, world!", add_bos=True, special=True)

    assert tokens == [9707, 11, 1879, 0]

    # New prompt for generation test
    tokens = model.tokenize(b"The quick brown fox jumps", add_bos=True, special=True)

    batch = internals.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

    seed = 1337
    sampler = internals.LlamaSampler()
    sampler.add_top_k(50)
    sampler.add_top_p(0.9, 1)
    sampler.add_temp(0.8)
    sampler.add_dist(seed)

    result = list(tokens)
    n_eval = len(tokens)
    batch.reset()
    pos_array = list(range(n_eval))
    logits_array = [False] * (n_eval - 1) + [True]

    batch.add_sequence(
        token_array=tokens,
        pos_array=pos_array,
        seq_ids=[0],
        logits_array=logits_array
    )
    context.decode(batch)

    for _ in range(4):
        token_id = sampler.sample(context, -1)
        sampler.accept(token_id)
        result.append(token_id)

        batch.reset()

        batch.add_token(
            token=token_id,
            pos=n_eval,
            seq_ids=[0],
            logits=True
        )

        context.decode(batch)
        n_eval += 1

    output = result[len(tokens):]
    output_text = model.detokenize(output, special=True)
    assert b"over" in output_text or b"lazy dog" in output_text


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


def test_grammar_sampling_safety(completion_model):
    """A strict grammar must produce a complete, parseable JSON object."""
    grammar_text = r'''
        root   ::= object
        object ::= "{" space pair "}"
        pair   ::= string ":" space value
        string ::= "\"" [a-z]+ "\""
        value  ::= number
        number ::= [0-9]+
        space  ::= [ ]?
    '''

    grammar = llama_cpp.LlamaGrammar.from_string(grammar_text)
    output = completion_model.create_completion(
        "Generate a JSON with age:",
        max_tokens=20,
        grammar=grammar,
        temperature=0.1
    )

    generated_text = output["choices"][0]["text"]
    parsed = json.loads(generated_text)
    assert len(parsed) == 1
    assert isinstance(next(iter(parsed.values())), int)

def test_logit_bias(completion_model):
    """A strong positive bias must force the selected token."""
    # Target token we want to force the model to generate
    target_word = " banana"           # Note the leading space — important for most tokenizers
    # Get the token ID corresponding to " banana" (Qwen-style tokenizer expected)
    target_token = completion_model.tokenize(target_word.encode("utf-8"), add_bos=False)[0]

    # Apply very strong positive bias to make this token extremely likely
    bias = {target_token: 100.0}

    # Generate a very short continuation with temperature=0 (greedy) + strong bias
    output = completion_model.create_completion(
        "I like to eat",
        max_tokens=3,
        logit_bias=bias,
        temperature=0.0
    )

    generated_text = output["choices"][0]["text"]
    assert "banana" in generated_text, f"Expected 'banana' in output, got: '{generated_text}'"


def test_custom_logits_processor(completion_model):
    """
    Test 4: Custom Logits Processor (Pure Python Implementation).

    Verifies that we can manipulate logits in Python before sampling.
    In this test, we suppress any token containing the letter 'e'.
    """
    def no_e_processor(input_ids, scores):
        """
        Filters out tokens containing 'e'.
        """
        for token_id in range(len(scores)):
            # Decode single token → get its string representation
            token_str = completion_model.detokenize([token_id]).decode("utf-8", errors="ignore")

            # Ban tokens that contain 'e' anywhere in their decoded form
            if "e" in token_str:
                scores[token_id] = -float("inf")

        return scores

    # Generate with greedy sampling (temperature=0) + our custom processor
    output = completion_model.create_completion(
        "The alphabet starts with",
        max_tokens=10,
        logits_processor=llama_cpp.LogitsProcessorList([no_e_processor]),
        temperature=0.0
    )

    generated_text = output["choices"][0]["text"]
    assert "e" not in generated_text, \
        f"Expected no letter 'e' in output, but found one:\n  Output was: '{generated_text}'"


@pytest.mark.parametrize("model_class,n_seq_max", [(llama_cpp.Llama, 2), (LlamaEmbedding, 1)])
def test_real_llama_base_embedding_api(llama_cpp_model_path, model_class, n_seq_max):
    """
    Test the maintained embedding API on the standard Llama class.

    Covers pre-tokenized batching, normalization, separator-based string
    batching, token counts, and the OpenAI-compatible response wrapper.
    """
    model = model_class(
        model_path=llama_cpp_model_path,
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
