import ctypes
import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import struct
from contextlib import ExitStack

import pytest
from jinja2.exceptions import TemplateError


@pytest.fixture
def make_handler(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    with ExitStack() as cleanup:
        def create(name="MTMDChatHandler", **kwargs):
            options = {"mmproj_path": str(tmp_path), "verbose": False, **kwargs}
            handler = getattr(multimodal, name)(**options)
            return cleanup.enter_context(handler)
        yield create


@pytest.fixture
def chat_prefill(make_handler, monkeypatch):
    import numpy as np
    from llama_cpp import Llama
    from llama_cpp import llama_multimodal as multimodal

    handler = make_handler()
    monkeypatch.setattr(handler, "mtmd_ctx", object())
    monkeypatch.setattr(handler, "_init_mtmd_context", Mock())
    monkeypatch.setattr(handler, "_free_mtmd_resources", Mock())
    monkeypatch.setattr(handler, "_is_text_chunk", lambda kind: False)
    monkeypatch.setattr(handler, "_is_image_chunk", lambda kind: True)
    monkeypatch.setattr(multimodal, "_convert_completion_to_chat", lambda result, **kw: result)
    handler._process_mtmd_prompt = Mock(return_value=(
        [1, 2, -9, -9], [(2, 4, object(), 1, -9)], object(), []
    ))
    ctx = SimpleNamespace(
        ctx=object(), memory_seq_rm=Mock(return_value=True),
        memory_seq_add=Mock(), memory_can_shift=lambda: True,
        memory_clear=Mock(),
    )
    llm = SimpleNamespace(
        n_tokens=3, input_ids=np.array([1, 2, 3, 0, 0, 0]),
        _n_ctx=6, n_ctx=lambda: 6, n_batch=1, n_keep=0,
        _ctx=ctx, is_hybrid=False, _hybrid_cache_mgr=None,
        speculative=None, verbose=False,
        context_params=SimpleNamespace(no_perf=True),
        _prefilled_prompt=(1, 2, 3), _restored_logits=object(),
        _last_eval_output_start=2, _last_eval_output_count=1,
        longest_token_prefix=Llama.longest_token_prefix,
        create_completion=Mock(return_value="completed"),
    )
    llm.reset = Mock(side_effect=lambda: Llama.reset(llm))
    llm._memory_seq_rm_or_raise = lambda *args: Llama._memory_seq_rm_or_raise(llm, *args)
    llm._mark_prefilled_prompt = Mock()

    def evaluate(mtmd, ctx, chunk, pos, seq, batch, logits, output):
        output._obj.value = pos.value + 2
        return 0

    backend = SimpleNamespace(
        mtmd_input_chunk_get_n_tokens=lambda chunk: 2,
        mtmd_input_chunk_get_n_pos=lambda chunk: 2,
        mtmd_helper_eval_chunk_single=Mock(side_effect=evaluate),
    )
    monkeypatch.setattr(handler, "_mtmd_cpp", backend)
    yield handler, llm, backend


@pytest.fixture
def chat_tokenizer(chat_prefill):
    from llama_cpp import mtmd_cpp

    handler, llm, backend = chat_prefill
    handler.media_marker = "<__media__>"
    handler.mtmd_bos_token, handler.mtmd_eos_token = "<bos>", "<eos>"
    llm._model = SimpleNamespace(get_add_bos=lambda: True, get_add_eos=lambda: True)
    backend.mtmd_input_text = mtmd_cpp.mtmd_input_text
    backend.mtmd_bitmap_p_ctypes = mtmd_cpp.mtmd_bitmap_p_ctypes
    observed = []

    def tokenize(ctx, output, input_ptr, bitmap_array, count):
        inp = ctypes.cast(input_ptr, ctypes.POINTER(mtmd_cpp.mtmd_input_text)).contents
        observed.append((ctypes.string_at(inp.text, inp.text_len), inp.add_special,
                         inp.parse_special, count))
        return 0

    backend.mtmd_tokenize = Mock(side_effect=tokenize)
    return handler, llm, observed


@pytest.mark.parametrize("with_media", [False, True])
def test_full_chat_tokenization_keeps_special_tokens_across_turns(chat_tokenizer, with_media):
    from llama_cpp import mtmd_cpp

    handler, llm, observed = chat_tokenizer
    bitmaps = [mtmd_cpp.mtmd_bitmap_p_ctypes(1)] if with_media else []
    text = "user: " + (handler.media_marker if with_media else "hello")
    chunks = object()
    for n_tokens, prompt in ((0, text), (3, text + "\nassistant: hello\nuser: again")):
        llm.n_tokens = n_tokens
        assert handler._mtmd_tokenize(llm, prompt, bitmaps, chunks) is chunks
        assert llm.n_tokens == n_tokens
    # Both calls describe a full chat, even when the decoder already holds history.
    assert observed == [(prompt.encode(), True, True, len(bitmaps)) for prompt in
                        (text, text + "\nassistant: hello\nuser: again")]


@pytest.mark.parametrize("add_bos, add_eos", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("whitespace", ["", " \n"])
def test_auto_special_tokens_preserve_inner_turn_boundaries(chat_tokenizer, add_bos, add_eos, whitespace):
    handler, llm, observed = chat_tokenizer
    llm._model.get_add_bos = lambda: add_bos
    llm._model.get_add_eos = lambda: add_eos
    middle = "user: hello<eos>assistant: answer<eos>user: <bos>quoted"
    text = whitespace + "<bos>" + middle + "<eos>" + whitespace
    handler._mtmd_tokenize(llm, text, chunks=object())
    expected = whitespace + ("" if add_bos else "<bos>") + middle + ("" if add_eos else "<eos>") + whitespace
    assert observed == [(expected.encode(), True, True, 0)]


@pytest.mark.parametrize("handler_name", [
    "MTMDChatHandler", "GenericMTMDChatHandler", "Llava15ChatHandler", "ObsidianChatHandler",
    "MoondreamChatHandler", "Llava16ChatHandler", "NanoLlavaChatHandler", "Llama3VisionAlphaChatHandler",
    "MiniCPMv26ChatHandler", "MiniCPMv45ChatHandler", "MiniCPMV46ChatHandler", "Gemma3ChatHandler",
    "Gemma4ChatHandler", "GLM41VChatHandler", "GLM46VChatHandler", "GraniteDoclingChatHandler",
    "LFM2VLChatHandler", "LFM25VLChatHandler", "PaddleOCRChatHandler", "Qwen25VLChatHandler",
    "Qwen3ASRChatHandler", "Qwen3VLChatHandler", "Qwen35ChatHandler", "Step3VLChatHandler",
])
@pytest.mark.parametrize("generation", [False, True])
def test_builtin_templates_do_not_duplicate_auto_boundaries(chat_tokenizer, handler_name, generation):
    from llama_cpp import llama_multimodal as multimodal

    handler, llm, observed = chat_tokenizer
    cls = getattr(multimodal, handler_name)
    handler._change_chat_template(cls.CHAT_FORMAT)
    handler.extra_template_arguments.update({name: value for name, value in vars(cls).items()
                                            if name.endswith("_TOKEN") and isinstance(value, str)})
    text = handler._render_mtmd_prompt(messages=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
    ], add_generation_prompt=generation)
    handler._mtmd_tokenize(llm, text, chunks=object())
    normalized = observed[0][0].decode()
    assert not normalized.startswith("<bos>")
    assert not normalized.endswith("<eos>")
    # Removing an outer copy must leave all internal turn delimiters intact.
    assert normalized == text.removeprefix("<bos>").removesuffix("<eos>")


@pytest.mark.parametrize("failure", ["rollback", "shift", "helper", "position", "position-in-context", "interrupt"])
def test_chat_prefill_failure_resets_state_and_releases_media(chat_prefill, failure):
    handler, llm, backend = chat_prefill
    evaluate = backend.mtmd_helper_eval_chunk_single.side_effect
    if failure in ("rollback", "shift"):
        llm._ctx.memory_seq_rm.return_value = False
    if failure == "shift":
        llm.n_tokens = 6
        llm.input_ids[:] = [1, 2, 3, 4, 5, 6]
        handler._process_mtmd_prompt.return_value = (
            [1, 2, 3, 4, 5, 6, -9, -9], [(6, 8, object(), 1, -9)], object(), []
        )
    elif failure == "helper":
        backend.mtmd_helper_eval_chunk_single.side_effect = lambda *args: -1
    elif failure in ("position", "position-in-context"):
        def invalid_position(*args):
            args[-1]._obj.value = 7 if failure == "position" else 3
            return 0
        backend.mtmd_helper_eval_chunk_single.side_effect = invalid_position
    elif failure == "interrupt":
        backend.mtmd_helper_eval_chunk_single.side_effect = KeyboardInterrupt

    match = "expected 4" if failure == "position-in-context" else None
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt), match=match):
        handler(llama=llm, messages=[])
    llm.reset.assert_called_once()
    assert llm.n_tokens == 0 and llm._last_eval_output_count == 0
    assert llm._prefilled_prompt is None and llm._restored_logits is None
    llm._ctx.memory_seq_add.assert_not_called()
    llm.create_completion.assert_not_called()
    handler._free_mtmd_resources.assert_called_once()

    # Retry on the same handler/context after a partially committed request.
    llm._ctx.memory_seq_rm.return_value = True
    backend.mtmd_helper_eval_chunk_single.side_effect = evaluate
    handler._process_mtmd_prompt.return_value = (
        [-9, -9], [(0, 2, object(), 1, -9)], object(), []
    )
    assert handler(llama=llm, messages=[]) == "completed"
    assert llm.n_tokens == 2
    assert llm.create_completion.call_args.kwargs["prompt"] == [-9, -9]
    llm._mark_prefilled_prompt.assert_called_once()
    assert handler._free_mtmd_resources.call_count == 2


@pytest.mark.parametrize("video", [False, True])
def test_mtmd_media_eval_error_explains_video_context_usage(chat_prefill, video):
    handler, llm, backend = chat_prefill
    backend.mtmd_helper_eval_chunk_single.side_effect = lambda *args: 1
    messages = [{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}] if video else []

    with pytest.raises(ValueError, match="Media evaluation failed") as exc_info:
        handler(llama=llm, messages=messages)

    error = str(exc_info.value)
    assert "error code 1" in error
    assert "media_tokens=2" in error
    assert "n_ctx=6" in error
    assert ("video_fps_target" in error) is video
    assert ("chat_handler_kwargs" in error) is video


@pytest.mark.parametrize("handler_name", [
    "MiniCPMv45ChatHandler", "GLM41VChatHandler", "GLM46VChatHandler",
    "GraniteDoclingChatHandler", "LFM2VLChatHandler", "PaddleOCRChatHandler",
    "Qwen25VLChatHandler", "Qwen3ASRChatHandler", "Qwen3VLChatHandler", "Qwen35ChatHandler",
])
def test_handlers_preserve_ledger_for_base_cache_matching(make_handler, monkeypatch, handler_name):
    import numpy as np
    from llama_cpp import llama_multimodal as multimodal

    handler = make_handler(handler_name)
    llm = SimpleNamespace(input_ids=np.array([1, -9, -9, 2]))
    seen = []
    monkeypatch.setattr(multimodal.MTMDChatHandler, "__call__",
                        lambda self, **kwargs: seen.append(kwargs["llama"].input_ids.tolist()))
    handler(llama=llm, messages=[])
    assert seen == [[1, -9, -9, 2]]


@pytest.mark.parametrize("mode, embedding_rows", [
    ("plain", 2), ("hybrid", 2), ("shift", 2),
    ("partial", 2), ("hybrid-partial", 2), ("plain", 4),
])
def test_chat_prefill_success_hands_off_rebuilt_prompt(chat_prefill, mode, embedding_rows):
    handler, llm, backend = chat_prefill
    expected = [1, 2, -9, -9]
    # Embedding rows may exceed the two positions occupied by this media chunk.
    backend.mtmd_input_chunk_get_n_tokens = lambda chunk: embedding_rows
    if "partial" in mode:
        llm.input_ids[:4] = expected
        llm.n_tokens = 4
    if mode == "shift":
        llm.n_tokens = 6
        llm.input_ids[:] = [1, 2, 3, 4, 5, 6]
        handler._process_mtmd_prompt.return_value = (
            [1, 2, 3, 4, 5, 6, -9, -9], [(6, 8, object(), 1, -9)], object(), []
        )
        expected = [4, 5, 6, -9, -9]
    if mode.startswith("hybrid"):
        llm.is_hybrid = True
        llm._hybrid_cache_mgr = SimpleNamespace(
            max_checkpoints=2, clear=Mock(), save_checkpoint=Mock(),
            find_best_checkpoint=Mock(return_value=SimpleNamespace(pos=2)),
            restore_checkpoint=Mock(return_value=True),
        )
    assert handler(llama=llm, messages=[]) == "completed"
    assert llm.n_tokens == len(expected)
    assert llm.create_completion.call_args.kwargs["prompt"] == expected
    llm._mark_prefilled_prompt.assert_called_once()
    llm.reset.assert_not_called()
    handler._free_mtmd_resources.assert_called_once()
    if mode == "shift":
        llm._ctx.memory_seq_add.assert_called_once_with(0, 3, 6, -3)
    if mode.startswith("hybrid"):
        llm._hybrid_cache_mgr.clear.assert_not_called()
        llm._hybrid_cache_mgr.save_checkpoint.assert_called_once()
    if "partial" in mode:
        assert backend.mtmd_helper_eval_chunk_single.call_args.args[3].value == 2
        if mode.startswith("hybrid"):
            llm._hybrid_cache_mgr.find_best_checkpoint.assert_called_once_with([1, 2], seq_id=0)
        else:
            llm._ctx.memory_seq_rm.assert_called_once_with(0, 2, -1)


@pytest.mark.parametrize("restore", [True, False])
def test_media_checkpoint_restores_before_rewritten_assistant_suffix(chat_prefill, monkeypatch, restore):
    handler, llm, backend = chat_prefill
    llm.is_hybrid = True
    llm.n_tokens = 5
    llm.input_ids[:5] = [1, 2, -9, -9, 7]
    llm._hybrid_cache_mgr = SimpleNamespace(
        max_checkpoints=4, clear=Mock(), save_checkpoint=Mock(),
        find_best_checkpoint=Mock(return_value=None), restore_checkpoint=Mock(),
    )
    media_cache = SimpleNamespace(find_best_checkpoint=Mock(return_value=SimpleNamespace(pos=4)),
                                 restore_checkpoint=Mock(return_value=restore), save_checkpoint=Mock())
    monkeypatch.setattr(handler, "_get_media_checkpoint_cache", lambda llama: media_cache)
    monkeypatch.setattr(handler, "_is_text_chunk", lambda kind: kind == 0)
    handler._process_mtmd_prompt.return_value = (
        [1, 2, -9, -9, 8], [(0, 2, [1, 2], 0, None), (2, 4, object(), 1, -9),
                             (4, 5, [8], 0, None)], object(), [],
    )

    def tokens(chunk, count):
        count._obj.value = len(chunk)
        return chunk

    def evaluate(text):
        llm.input_ids[llm.n_tokens:llm.n_tokens + len(text)] = text
        llm.n_tokens += len(text)

    backend.mtmd_input_chunk_get_tokens_text = tokens
    llm.eval = Mock(side_effect=evaluate)
    assert handler(llama=llm, messages=[]) == "completed"
    media_cache.find_best_checkpoint.assert_called_once_with([1, 2, -9, -9], seq_id=0)
    llm._hybrid_cache_mgr.restore_checkpoint.assert_not_called()
    assert llm.create_completion.call_args.kwargs["prompt"] == [1, 2, -9, -9, 8]
    if restore:
        backend.mtmd_helper_eval_chunk_single.assert_not_called()
        llm.eval.assert_called_once_with([8])
        llm.reset.assert_not_called()
    else:
        llm.reset.assert_called_once()
        backend.mtmd_helper_eval_chunk_single.assert_called_once()
        media_cache.save_checkpoint.assert_called_once_with(4, [1, 2, -9, -9], seq_id=0)


@pytest.mark.parametrize("enabled, hybrid, qwen", [(True, True, True), (False, True, True),
                                                  (True, False, True), (True, True, False)])
def test_generic_media_checkpoint_lifecycle(tmp_path, monkeypatch, enabled, hybrid, qwen):
    from llama_cpp import llama_cache, llama_multimodal

    created = []
    def cache_factory(ctx, **kwargs):
        cache = SimpleNamespace(_context_ref=None, on_device=kwargs["on_device"], clear=Mock(), close=Mock())
        created.append((ctx, kwargs, cache))
        return cache

    monkeypatch.setattr(llama_cache, "HybridCheckpointCache", cache_factory)
    handler = llama_multimodal.GenericMTMDChatHandler(chat_format="hello", mmproj_path=str(tmp_path), verbose=False)
    handler._qwen_vision_template = qwen
    ctx = SimpleNamespace(ctx=object())
    ctx._register_checkpoint_cache = lambda cache: setattr(cache, "_context_ref", lambda: ctx)
    llm = SimpleNamespace(_ctx=ctx, is_hybrid=hybrid, speculative=None,
                          _hybrid_cache_mgr=SimpleNamespace(max_checkpoints=4 if enabled else 0, on_device=False))
    try:
        cache = handler._get_media_checkpoint_cache(llm)
        if enabled and hybrid and qwen:
            assert cache is handler._get_media_checkpoint_cache(llm)
            assert len(created) == 1
            assert created[0][1]["max_checkpoints"] == 1
            assert created[0][1]["on_device"] is True
            llm._hybrid_cache_mgr.max_checkpoints = 0
            assert handler._get_media_checkpoint_cache(llm) is None
            cache.clear.assert_called_once()
        else:
            assert cache is None and not created
    finally:
        handler.close()
    if created:
        created[0][2].close.assert_called_once()


@pytest.fixture
def media_checkpoint(make_handler, monkeypatch):
    import numpy as np
    from llama_cpp import llama_cpp as lib
    from llama_cpp._internals import LlamaContext
    from llama_cpp.llama_cache import HybridCheckpointCache

    native = {}
    for name, result in (("state_seq_get_size_ext", 4), ("state_seq_get_data_ext", 4),
                         ("state_seq_set_data_ext", 4), ("get_memory", object()),
                         ("memory_seq_pos_min", 0), ("memory_seq_pos_max", 2),
                         ("memory_seq_rm", True)):
        native[name] = Mock(return_value=result)
        monkeypatch.setattr(lib, "llama_" + name, native[name])
    ctx = SimpleNamespace(ctx=object(), _checkpoint_caches=[])
    def register(cache):
        ctx._checkpoint_caches.append(cache)
        cache._context_ref = lambda: ctx
    ctx._register_checkpoint_cache = register
    ctx._invalidate_checkpoints = lambda *a, **kw: LlamaContext._invalidate_checkpoints(ctx, *a, **kw)
    manager = HybridCheckpointCache(ctx.ctx, max_checkpoints=4)
    register(manager)
    handler = make_handler("GenericMTMDChatHandler", chat_format="hello")
    handler._qwen_vision_template = True
    llm = SimpleNamespace(_ctx=ctx, is_hybrid=True, speculative=None, _hybrid_cache_mgr=manager,
                          n_tokens=3, input_ids=np.array([1, -9, -9, 0]))
    yield handler, llm, native
    for cache in ctx._checkpoint_caches:
        cache.close()


@pytest.mark.parametrize("conflict", [None, "manager", "speculative", "peer", "host-peer", "closed-peer"])
def test_media_checkpoint_reserves_only_an_unused_device_slot(media_checkpoint, conflict):
    from llama_cpp.llama_cache import HybridCheckpointCache

    handler, llm, _ = media_checkpoint
    if conflict == "manager":
        llm._hybrid_cache_mgr.on_device = True
    elif conflict == "speculative":
        llm.speculative = object()  # Also exclude engines without process_media().
    elif conflict in ("peer", "host-peer", "closed-peer"):
        peer = HybridCheckpointCache(llm._ctx.ctx, on_device=conflict != "host-peer")
        llm._ctx._register_checkpoint_cache(peer)
        if conflict == "closed-peer":
            peer.close()
    cache = handler._get_media_checkpoint_cache(llm)
    assert cache.on_device is (conflict in (None, "host-peer", "closed-peer"))
    assert handler._get_media_checkpoint_cache(llm) is cache
    assert handler._save_media_checkpoint(llm, cache)
    checkpoint = cache.find_best_checkpoint([1, -9, -9])
    assert checkpoint is not None and cache.restore_checkpoint(checkpoint)


@pytest.mark.parametrize("conflict", ["manager", "speculative", "peer"])
def test_media_checkpoint_mode_change_discards_old_device_snapshot(media_checkpoint, conflict):
    from llama_cpp.llama_cache import HybridCheckpointCache

    handler, llm, native = media_checkpoint
    old = handler._get_media_checkpoint_cache(llm)
    assert handler._save_media_checkpoint(llm, old)
    checkpoint = old.checkpoints[0]
    if conflict == "manager":
        llm._hybrid_cache_mgr.on_device = True
    elif conflict == "speculative":
        llm.speculative = object()
    else:
        peer = HybridCheckpointCache(llm._ctx.ctx, on_device=True)
        llm._ctx._register_checkpoint_cache(peer)
    cache = handler._get_media_checkpoint_cache(llm)
    assert cache is not old and not cache.on_device
    assert old._ctx is None and not old.restore_checkpoint(checkpoint)
    native["state_seq_set_data_ext"].assert_not_called()
    assert handler._save_media_checkpoint(llm, cache)
    llm._hybrid_cache_mgr.on_device = False
    llm.speculative = None
    if conflict == "peer":
        peer.close()
    assert handler._get_media_checkpoint_cache(llm).on_device
    assert cache._ctx is None


@pytest.mark.parametrize("failure", ["size", "export"])
def test_media_checkpoint_device_export_failure_retries_host_once(media_checkpoint, failure):
    from llama_cpp import llama_cpp as lib

    handler, llm, native = media_checkpoint
    if failure == "size":
        native["state_seq_get_size_ext"].side_effect = lambda ctx, seq, flags: (
            0 if flags & lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE else 4
        )
    else:
        native["state_seq_get_data_ext"].side_effect = lambda ctx, dst, size, seq, flags: (
            0 if flags & lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE else size
        )
    device_cache = handler._get_media_checkpoint_cache(llm)
    assert handler._save_media_checkpoint(llm, device_cache)
    host_cache = handler._get_media_checkpoint_cache(llm)
    assert host_cache is not device_cache and not host_cache.on_device
    assert device_cache._ctx is None
    assert host_cache.restore_checkpoint(host_cache.checkpoints[0])
    assert handler._save_media_checkpoint(llm, host_cache)
    flags = [call.args[-1] for call in native["state_seq_get_size_ext"].call_args_list]
    assert [bool(flag & lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) for flag in flags] == [True, False, False]
    native["state_seq_get_data_ext"].return_value = 0
    native["state_seq_get_data_ext"].side_effect = None
    assert not handler._save_media_checkpoint(llm, host_cache)
    assert native["state_seq_get_size_ext"].call_count == 4


def test_media_checkpoint_context_change_resets_device_failure(media_checkpoint):
    handler, llm, _ = media_checkpoint
    old = handler._get_media_checkpoint_cache(llm)
    handler._media_checkpoint_device_failed = True
    new_context = SimpleNamespace(ctx=object())
    new_context._register_checkpoint_cache = lambda cache: setattr(cache, "_context_ref", lambda: new_context)
    llm._ctx = new_context
    cache = handler._get_media_checkpoint_cache(llm)
    assert cache is not old and old._ctx is None
    assert cache.on_device and not handler._media_checkpoint_device_failed
    assert cache._ctx is new_context.ctx


@pytest.mark.parametrize("overwrite", ["peer-save", "context-export"])
def test_media_device_snapshot_survives_host_save_but_not_slot_overwrite(media_checkpoint, overwrite):
    from llama_cpp import llama_cpp as lib
    from llama_cpp._internals import LlamaContext
    from llama_cpp.llama_cache import HybridCheckpointCache

    handler, llm, native = media_checkpoint
    cache = handler._get_media_checkpoint_cache(llm)
    assert handler._save_media_checkpoint(llm, cache)
    checkpoint = cache.checkpoints[0]
    assert llm._hybrid_cache_mgr.save_checkpoint(3, [1, -9, -9])
    assert cache.find_best_checkpoint([1, -9, -9]) is checkpoint
    if overwrite == "peer-save":
        peer = HybridCheckpointCache(llm._ctx.ctx, on_device=True)
        llm._ctx._register_checkpoint_cache(peer)
        assert peer.save_checkpoint(3, [1, -9, -9])
    else:
        llm._ctx._assert_ctx = lambda: None
        LlamaContext.get_state_seq_data_ext(
            llm._ctx, (ctypes.c_uint8 * 4)(), 4, 0,
            lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE,
        )
    assert cache.find_best_checkpoint([1, -9, -9]) is None
    assert not cache.restore_checkpoint(checkpoint)
    assert llm._hybrid_cache_mgr.checkpoints
    native["state_seq_set_data_ext"].assert_not_called()


@pytest.mark.parametrize("failure", [None, "encode", "decode", "callback", "interrupt", "overflow", "media-overflow"])
def test_chat_prefill_mtp_callback_and_cleanup(chat_prefill, failure):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    handler, llm, backend = chat_prefill
    embeddings = (ctypes.c_float * 4)(1, 2, 3, 4)
    positions = (ctypes.c_int32 * 8)(0, 0, 0, 1, 0, 1, 0, 0)
    received = []

    def process_media(batch):
        received.append((batch.n_tokens, batch.n_embd, batch.n_pos,
                         list(batch.embd[:4]), list(batch.pos[:8])))
        if failure == "callback":
            raise RuntimeError("MTP callback failed")
        if failure == "interrupt":
            raise KeyboardInterrupt

    llm.speculative = SimpleNamespace(supports_predecoded_media=True,
                                     process_media=process_media, clear=Mock())
    llm.is_hybrid = True
    llm._hybrid_cache_mgr = SimpleNamespace(max_checkpoints=2, clear=Mock(),
                                          save_checkpoint=Mock(), find_best_checkpoint=Mock())
    handler._process_mtmd_prompt.return_value = (
        [-9, -9], [(0, 2, object(), 1, -9)], object(), []
    )
    if failure == "overflow":
        handler._process_mtmd_prompt.return_value = (
            [-9] * 7, [(0, 7, object(), 1, -9)], object(), []
        )
    if failure == "media-overflow":
        backend.mtmd_input_chunk_get_n_tokens = lambda chunk: 7
    backend.mtmd_encode_chunk = Mock(return_value=1 if failure == "encode" else 0)
    backend.mtmd_get_output_embd = Mock(return_value=embeddings)
    backend.mtmd_helper_post_decode_callback = module.mtmd_helper_post_decode_callback

    def decode(mtmd, ctx, chunk, embd, pos, seq, batch_size, output, callback, data):
        # Invoke the real ctypes trampoline, including exception transport.
        view = module.mtmd_helper_embd_batch(2, embd, 2, positions, 4, 0)
        result = callback(ctypes.byref(view), data)
        output._obj.value = 2
        return result or (1 if failure == "decode" else 0)

    backend.mtmd_helper_decode_image_chunk = Mock(side_effect=decode)
    if failure is None:
        assert handler(llama=llm, messages=[]) == "completed"
        assert llm.create_completion.call_args.kwargs["prompt"] == [-9, -9]
        assert llm.reset.call_count == 1
        llm._mark_prefilled_prompt.assert_called_once()
    else:
        with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
            handler(llama=llm, messages=[])
        assert llm.n_tokens == 0 and llm.reset.call_count == 2
        llm.create_completion.assert_not_called()
    if failure not in ("encode", "overflow", "media-overflow"):
        assert received == [(2, 2, 4, [1, 2, 3, 4], list(positions))]
    else:
        backend.mtmd_helper_decode_image_chunk.assert_not_called()
    backend.mtmd_helper_eval_chunk_single.assert_not_called()
    llm._ctx.memory_seq_rm.assert_not_called()
    llm._hybrid_cache_mgr.find_best_checkpoint.assert_not_called()
    llm._hybrid_cache_mgr.save_checkpoint.assert_not_called()
    assert llm.speculative.clear.call_count == llm.reset.call_count
    handler._free_mtmd_resources.assert_called_once()


def test_mtmd_audio_stream_default_params_abi():
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    if getattr(module.mtmd_audio_stream_params_default, "__ctypes_optional__", False):
        pytest.skip("This libmtmd does not provide realtime audio input")
    params = module.mtmd_audio_stream_params_default()
    assert ctypes.sizeof(params) == ctypes.sizeof(ctypes.c_int32)
    assert ctypes.alignment(params) == ctypes.alignment(ctypes.c_int32)
    assert type(params).step_ms.offset == 0
    assert params.step_ms == 320
    module.mtmd_audio_stream_free(None)


@pytest.mark.parametrize("name", [
    "mtmd_audio_stream_get_n_rows", "mtmd_audio_stream_get_n_rows_flushed",
])
def test_mtmd_audio_stream_row_query_abi(name):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    query = getattr(module, name)
    if getattr(query, "__ctypes_optional__", False):
        with pytest.raises(RuntimeError, match=name):
            query(None)
        return
    assert query.argtypes == [module.mtmd_audio_stream_p_ctypes]
    assert query.restype is ctypes.c_int64
    # Native queries require a live model-backed stream; never pass NULL.


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="64-bit MTMD ABI")
@pytest.mark.parametrize("name, offsets", [
    ("mtmd_gen_inp", {
        "type": 0, "code0": 4, "embd": 8, "top_k": 16, "top_p": 20,
        "temp": 24, "codes": 32, "n_codes": 40, "feats": 48, "n_feats": 56,
    }),
    ("mtmd_gen_out", {
        "codes": 0, "n_codes": 8, "feats": 16, "n_feats": 24,
        "embd": 32, "is_eos": 40, "audio": 48, "n_samples": 56,
    }),
])
def test_mtmd_gen_struct_abi(name, offsets):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    struct_type = getattr(module, name)
    # seed moved into state params; serialized decoder state is no longer public.
    assert ctypes.sizeof(struct_type) == 64
    assert ctypes.alignment(struct_type) == 8
    assert {field: getattr(struct_type, field).offset for field, _ in struct_type._fields_} == offsets


def test_mtmd_gen_state_api():
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    params = module.mtmd_gen_state_params_default()
    assert ctypes.sizeof(params) == ctypes.sizeof(ctypes.c_uint32)
    assert ctypes.alignment(params) == ctypes.alignment(ctypes.c_uint32)
    assert type(params).seed.offset == 0
    assert params.seed == 0xFFFFFFFF
    assert module.mtmd_gen_state_init.argtypes == [
        module.mtmd_context_p_ctypes, module.mtmd_gen_state_params,
    ]
    assert module.mtmd_gen_state_init.restype is module.mtmd_gen_state_p_ctypes
    for function in (module.mtmd_gen_state_reset, module.mtmd_gen_state_free):
        assert function.argtypes == [module.mtmd_gen_state_p_ctypes]
        assert function.restype is None
    assert module.mtmd_gen_inp_default.restype is module.mtmd_gen_inp
    assert module.mtmd_gen_audio_process.argtypes == [
        module.mtmd_context_p_ctypes, module.mtmd_gen_state_p_ctypes,
        ctypes.POINTER(module.mtmd_gen_inp), ctypes.POINTER(module.mtmd_gen_out),
    ]
    assert module.mtmd_gen_audio_process.restype is ctypes.c_int32
    module.mtmd_gen_state_free(None)


def test_mtmd_caps_abi():
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    caps = module.mtmd_caps
    assert ctypes.sizeof(caps) == 3
    assert ctypes.alignment(caps) == 1
    assert [getattr(caps, field).offset for field in ("inp_vision", "inp_audio", "gen_audio")] == [0, 1, 2]
    assert module.mtmd_get_cap_from_file.restype is caps


def test_mtmd_helper_gen_audio_input_abi():
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    inp = module.mtmd_helper_gen_audio_inp
    assert inp.stream.offset == 0
    assert inp.seq_id.offset == 4
    assert inp.prompt.offset == 8
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    assert ctypes.sizeof(inp) == 24 + 4 * pointer_size
    assert {field: getattr(inp, field).offset for field in
            ("prompt_len", "speaker_ref", "lang", "top_k", "top_p", "seed", "out_type")} == {
        "prompt_len": 8 + pointer_size, "speaker_ref": 8 + 2 * pointer_size,
        "lang": 8 + 3 * pointer_size, "top_k": 8 + 4 * pointer_size,
        "top_p": 12 + 4 * pointer_size, "seed": 16 + 4 * pointer_size,
        "out_type": 20 + 4 * pointer_size,
    }
    value = inp(stream=False, seq_id=7)
    raw = bytes(value)
    assert raw[0] == 0
    assert ctypes.c_int32.from_buffer_copy(raw[4:8]).value == 7
    value.stream = True
    assert value.seq_id == 7


def test_mtmd_helper_gen_audio_defaults_and_flush_api():
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    default = module.mtmd_helper_gen_audio_inp_default
    assert default.argtypes == []
    assert default.restype is module.mtmd_helper_gen_audio_inp
    inp = default()
    assert not inp.stream and inp.seq_id == 0
    assert inp.prompt is None and inp.prompt_len == 0
    assert inp.speaker_ref is None and inp.lang is None
    assert inp.top_k == 50 and inp.top_p == 1.0
    assert inp.seed == 0xFFFFFFFF
    assert inp.out_type == module.mtmd_helper_gen_audio_outtype.MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV
    assert module.mtmd_helper_gen_audio_flush.argtypes == [module.mtmd_helper_gen_audio_p_ctypes]
    assert module.mtmd_helper_gen_audio_flush.restype is ctypes.c_int32


def test_mtmd_decoder_pos_abi():
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    position = module.mtmd_decoder_pos
    assert ctypes.sizeof(position) == 16
    assert ctypes.alignment(position) == ctypes.alignment(ctypes.c_uint32)
    assert [getattr(position, field).offset for field in ("t", "x", "y", "z")] == [0, 4, 8, 12]

    # Exercise native struct returns and array writes without loading a model.
    chunks = module.mtmd_test_create_input_chunks()
    assert chunks
    try:
        chunk = module.mtmd_input_chunks_get(chunks, 1)
        image = module.mtmd_input_chunk_get_tokens_image(chunk)
        assert image
        count = module.mtmd_image_tokens_get_n_tokens(image)
        assert count > 1
        positions = (position * count)()
        module.mtmd_helper_image_get_decoder_pos(image, 7, positions)
        for index, actual in enumerate(positions):
            returned = module.mtmd_image_tokens_get_decoder_pos(image, 7, index)
            expected = (7 + index,) * 4
            assert (actual.t, actual.x, actual.y, actual.z) == expected
            assert (returned.t, returned.x, returned.y, returned.z) == expected
    finally:
        module.mtmd_input_chunks_free(chunks)


@pytest.mark.parametrize("n_pos", [1, 4])
def test_mtmd_helper_post_decode_callback(n_pos):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    embeddings = (ctypes.c_float * 6)(1, 2, 3, 4, 5, 6)
    positions = (ctypes.c_int32 * (2 * n_pos))(*range(2 * n_pos))
    batch = module.mtmd_helper_embd_batch(2, embeddings, 3, positions, n_pos, 7)
    user_data = ctypes.c_int32(11)
    received = []

    @module.mtmd_helper_post_decode_callback
    def callback(batch_ptr, user_data_ptr):
        received.append((
            ctypes.addressof(batch_ptr.contents),
            batch_ptr.contents.n_tokens,
            batch_ptr.contents.n_embd,
            batch_ptr.contents.n_pos,
            batch_ptr.contents.seq_id,
            list(batch_ptr.contents.embd[:6]),
            list(batch_ptr.contents.pos[:2 * n_pos]),
            ctypes.cast(user_data_ptr, ctypes.POINTER(ctypes.c_int32)).contents.value,
        ))
        return -3

    assert callback(ctypes.byref(batch), ctypes.byref(user_data)) == -3
    assert received == [(
        ctypes.addressof(batch), 2, 3, n_pos, 7,
        [1, 2, 3, 4, 5, 6], list(range(2 * n_pos)), 11,
    )]
    assert module.mtmd_helper_decode_image_chunk.argtypes[8] is module.mtmd_helper_post_decode_callback


def test_mtmd_helper_init_opt_abi():
    module = importlib.import_module("llama_cpp.mtmd_cpp")

    assert module.mtmd_helper_video_init_params._fields_ == [
        ("fps_target", ctypes.c_float),
        ("ffmpeg_bin_dir", ctypes.c_char_p),
        ("timestamp_interval_ms", ctypes.c_int64),
    ]
    assert module.mtmd_helper_init_opt._fields_ == [
        ("video_params", module.mtmd_helper_video_init_params),
    ]
    assert module.mtmd_helper_bitmap_init_from_file.argtypes == [
        module.mtmd_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_bool,
        module.mtmd_helper_init_opt,
    ]
    assert module.mtmd_helper_bitmap_init_from_buf.argtypes == [
        module.mtmd_context_p_ctypes,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_bool,
        module.mtmd_helper_init_opt,
    ]
    assert module.mtmd_helper_video_init.restype is module.mtmd_helper_video_p_ctypes

    opt = module.mtmd_helper_init_opt_default()
    assert opt.video_params.fps_target == 4.0
    assert opt.video_params.ffmpeg_bin_dir is None
    assert opt.video_params.timestamp_interval_ms == 5000


@pytest.mark.parametrize("handler_name", ["MTMDBaseHandler", "MTMDChatHandler"])
def test_mtmd_chat_handler_video_options(tmp_path, make_handler, handler_name):
    module = importlib.import_module("llama_cpp.mtmd_cpp")

    executable_suffix = ".exe" if os.name == "nt" else ""
    for executable_name in ("ffmpeg", "ffprobe"):
        executable_path = tmp_path / (executable_name + executable_suffix)
        executable_path.touch()
        executable_path.chmod(0o755)

    handler = make_handler(
        handler_name,
        video_fps_target=2.5,
        video_ffmpeg_bin_dir=tmp_path,
        video_timestamp_interval_ms=10000,
    )
    video_params = handler._mtmd_helper_init_opt.video_params
    assert video_params.fps_target == 2.5
    assert video_params.ffmpeg_bin_dir == os.fsencode(os.path.abspath(tmp_path))
    assert video_params.timestamp_interval_ms == 10000

    opt = module.mtmd_helper_init_opt_default()
    handler = make_handler(handler_name, mtmd_helper_init_opt=opt)
    assert handler._mtmd_helper_init_opt is opt

    with pytest.raises(ValueError, match="cannot be combined"):
        make_handler(
            handler_name,
            mtmd_helper_init_opt=opt,
            video_fps_target=1.0,
        )


def test_mtmd_chat_handler_rejects_invalid_ffmpeg_bin_dir(tmp_path, make_handler):
    with pytest.raises(ValueError, match="is not an existing directory"):
        make_handler(video_ffmpeg_bin_dir=tmp_path / "missing")

    with pytest.raises(ValueError, match="ffmpeg and ffprobe"):
        make_handler(video_ffmpeg_bin_dir=tmp_path)


@pytest.mark.parametrize("handler_name", ["MTMDBaseHandler", "MTMDChatHandler"])
@pytest.mark.parametrize("flash_attn, expected_flash", [(None, -1), (False, 0), (True, 1)])
def test_mtmd_context_initialization_and_close(tmp_path, handler_name, flash_attn, expected_flash):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = getattr(multimodal, handler_name)(
        clip_model_path=str(tmp_path), verbose=False, use_gpu=False,
        image_min_tokens=32, image_max_tokens=128, batch_max_tokens=256,
        flash_attn=flash_attn,
    )
    native = handler._mtmd_cpp
    backend = SimpleNamespace(
        mtmd_context_params_default=native.mtmd_context_params_default,
        clip_flash_attn_type=native.clip_flash_attn_type,
        mtmd_helper_log_set=Mock(),
        mtmd_default_marker=Mock(return_value=b"<default>"),
        mtmd_get_marker=Mock(return_value=b"<media>"),
        mtmd_init_from_file=Mock(return_value=123),
        mtmd_support_vision=Mock(return_value=True),
        mtmd_support_audio=Mock(return_value=False),
        mtmd_helper_support_video=Mock(return_value=True),
        mtmd_free=Mock(),
    )
    handler._mtmd_cpp = backend
    model = SimpleNamespace(model=456, n_threads=2)
    if handler_name == "MTMDChatHandler":
        model.token_eos = lambda: 2
        model.token_bos = lambda: 1
        model._model = SimpleNamespace(token_get_text=lambda token: {1: "<bos>", 2: "<eos>"}[token])
    try:
        handler._init_mtmd_context(model)
        handler._init_mtmd_context(model)
        backend.mtmd_init_from_file.assert_called_once()
        backend.mtmd_get_marker.assert_called_once_with(123)
        assert handler.media_marker == "<media>"
        assert handler.is_support_vision
        assert not handler.is_support_audio
        assert handler.is_support_video
        assert handler.mctx_params.n_threads == 2
        assert not handler.mctx_params.use_gpu
        assert handler.mctx_params.image_min_tokens == 32
        assert handler.mctx_params.image_max_tokens == 128
        assert handler.mctx_params.batch_max_tokens == 256
        assert handler.mctx_params.flash_attn_type == expected_flash
        if handler_name == "MTMDChatHandler":
            assert handler.mtmd_bos_token == "<bos>"
            assert handler.mtmd_eos_token == "<eos>"
        else:
            assert not hasattr(handler, "chat_template")
            assert not hasattr(handler, "mtmd_bos_token")
    finally:
        handler.close()
        handler.close()
    backend.mtmd_free.assert_called_once_with(123)


def test_mtmd_base_close_releases_context_after_callback_failure(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = multimodal.MTMDBaseHandler(mmproj_path=str(tmp_path), verbose=False)
    events = []
    handler.mtmd_ctx = 123
    handler._mtmd_cpp = SimpleNamespace(mtmd_free=lambda ctx: events.append("context"))

    def release_child():
        events.append("child")
        raise RuntimeError("child cleanup failed")

    handler._exit_stack.callback(release_child)
    with pytest.raises(RuntimeError, match="child cleanup failed"):
        handler.close()
    handler.close()
    assert events == ["child", "context"]
    assert handler.mtmd_ctx is None


def test_mtmd_chat_constructor_preserves_template_options(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")

    class CustomChatHandler(multimodal.MTMDChatHandler):
        chat_format = "custom: {{ value }}"

    arguments = {"value": "hello"}
    handler = CustomChatHandler(
        str(tmp_path), False, True, -1, -1, "override", 512, arguments,
    )
    try:
        assert isinstance(handler, multimodal.MTMDBaseHandler)
        assert handler.chat_template.render(**handler.extra_template_arguments) == "custom: hello"
        arguments["value"] = "changed"
        assert handler.extra_template_arguments == {"value": "hello"}
    finally:
        handler.close()


def test_mtmd_chat_template_raise_exception_preserves_message(make_handler):
    handler = make_handler(
        chat_template_override=(
            "{% if video %}{{ raise_exception('Video not supported') }}"
            "{% else %}text supported{% endif %}"
        ),
    )
    assert handler.chat_template.render(video=False) == "text supported"
    with pytest.raises(TemplateError, match="Video not supported"):
        handler.chat_template.render(video=True)


def test_mtmd_chat_template_strftime_now(make_handler):
    import datetime

    handler = make_handler(
        chat_template_override="{{ strftime_now('%Y') }}",
    )
    assert handler.chat_template.render() == datetime.datetime.now().strftime("%Y")


@pytest.mark.parametrize("kind, handler_name, path, label, item", [
    pytest.param("image", handler_name, "/path/to/image.png", "Picture",
                 {"type": "image_url", "image_url": image_url},
                 id=f"{handler_name}-{form}")
    for handler_name in ("Qwen25VLChatHandler", "Qwen35ChatHandler")
    for form, image_url in (("path", "/path/to/image.png"),
                            ("object", {"url": "/path/to/image.png"}))
] + [
    pytest.param("video", "Qwen35ChatHandler", "/path/to/clip.mp4", "Video", item, id=name)
    for name, item in [
        ("video-path", {"type": "video", "video": "/path/to/clip.mp4"}),
        ("video-object", {"type": "video", "video": {"url": "/path/to/clip.mp4"}}),
        ("video-url-path", {"type": "video_url", "video_url": "/path/to/clip.mp4"}),
        ("video-url-object", {"type": "video_url", "video_url": {"url": "/path/to/clip.mp4"}}),
    ]
] + [
    pytest.param("image", "Qwen3VLChatHandler", "/path/to/image.png", "Picture", item, id=name)
    for name, item in [
        ("image-untyped", {"image": "/path/to/image.png"}),
        ("image-url-untyped", {"image_url": {"url": "/path/to/image.png"}}),
        ("image-object", {"type": "image", "image": {"url": "/path/to/image.png"}}),
        ("image-url-path", {"type": "image_url", "image_url": "/path/to/image.png"}),
    ]
])
def test_qwen_templates_inject_media_marker(make_handler, kind, handler_name, path, label, item):
    handler = make_handler(handler_name)
    handler.mtmd_bos_token = handler.mtmd_eos_token = ""
    handler.media_marker = "<__media__>"
    setattr(handler, "is_support_vision" if kind == "image" else "is_support_video", True)
    messages = [{"role": "user", "content": [item]}]
    rendered = handler._render_mtmd_prompt(messages=messages, add_generation_prompt=False)
    assert "<|vision_start|>" not in rendered and "<|vision_end|>" not in rendered
    assert path in rendered
    media_items = handler._get_media_items(messages)
    prompt = handler._render_and_replace_media(
        messages=messages, media_items=media_items, add_generation_prompt=False,
    )
    assert media_items == [{"url": path, "type": kind}]
    assert f"{label} 1: <__media__>" in prompt
    assert all(token not in prompt for token in
               ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"))
    if kind == "video":
        with pytest.raises(TemplateError, match="System message cannot contain videos"):
            handler._render_mtmd_prompt(
                messages=[{"role": "system", "content": [item]}],
                add_generation_prompt=False,
            )


@pytest.mark.parametrize("placeholder", ["<|image_pad|>", "<|video_pad|>", "/path/to/media"])
@pytest.mark.parametrize("architecture", ["qwen35", "qwen3vl", "gemma4", ""])
def test_generic_media_normalization_preserves_template_boundaries(make_handler, placeholder, architecture):
    template = "literal <|vision_start|>text<|vision_end|> <|vision_start|>\n" + placeholder + " <|vision_end|>"
    handler = make_handler(
        "GenericMTMDChatHandler", chat_format=template,
    )
    handler.media_marker = "<__media__>"
    handler._ensure_chat_template(SimpleNamespace(metadata={"general.architecture": architecture}))
    prompt = handler._replace_media_placeholders(
        handler.chat_template.render(), [{"url": "/path/to/media", "type": "image"}],
    )
    expected = "<__media__>" if architecture.startswith("qwen") else "<|vision_start|>\n<__media__> <|vision_end|>"
    assert prompt == "literal <|vision_start|>text<|vision_end|> " + expected


@pytest.fixture
def chunk_ledger(tmp_path, make_handler, monkeypatch):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"test")
    handler = make_handler(mmproj_path=str(projector))
    handler.is_support_vision = handler.is_support_audio = handler.is_support_video = True
    monkeypatch.setattr(handler, "_render_and_replace_media", lambda **kwargs: "media")
    monkeypatch.setattr(handler, "load_media", lambda url, kind: url.encode())
    monkeypatch.setattr(handler, "_create_bitmap_from_bytes", lambda data: (object(), None))
    monkeypatch.setattr(handler, "_free_mtmd_resources", Mock())
    chunks = []
    monkeypatch.setattr(handler, "_mtmd_tokenize", lambda **kwargs: chunks)
    backend = SimpleNamespace(
        mtmd_input_chunk_type=module.mtmd_input_chunk_type,
        mtmd_input_chunks_size=len,
        mtmd_input_chunks_get=lambda entries, index: entries[index],
        mtmd_input_chunk_get_type=lambda chunk: chunk.kind,
        mtmd_input_chunk_get_n_tokens=lambda chunk: 2,
        mtmd_input_chunk_get_n_pos=lambda chunk: 2,
        mtmd_input_chunk_get_id=lambda chunk: chunk.identity,
    )
    monkeypatch.setattr(handler, "_mtmd_cpp", backend)

    def ledger(identities, media_items):
        chunks[:] = [SimpleNamespace(kind=kind, identity=identity) for kind, identity in identities]
        messages = [{"role": "user", "content": media_items}]
        _, spans, _, _ = handler._process_mtmd_prompt(llama=SimpleNamespace(n_threads=1), messages=messages)
        return [span[4] for span in spans]

    return handler, ledger


@pytest.mark.parametrize("source", ["video", "audio"])
def test_media_chunk_identity_survives_expansion_before_image(chunk_ledger, source):
    _, ledger = chunk_ledger
    kind = 1 if source == "video" else 2
    media = [{"type": source, source: "first.media"}, {"type": "image_url", "image_url": "last.png"}]
    identities = [(kind, b"first+0"), (kind, b"first+2"), (1, b"last-image")]
    first = ledger(identities, media)
    assert len(set(first)) == 3
    assert ledger(identities, media) == first
    changed = ledger([(kind, b"changed+0"), (kind, b"changed+2"), (1, b"last-image")], media)
    assert changed[:2] != first[:2]
    assert changed[2] == first[2]


def test_media_chunk_identity_uses_content_and_subchunk_not_url(chunk_ledger):
    _, ledger = chunk_ledger
    media = [{"type": "image_url", "image_url": "same.png"}]
    original = ledger([(1, b"original"), (1, b"original")], media)
    assert original[0] != original[1]
    assert ledger([(1, b"original"), (1, b"original")], media) == original
    assert ledger([(1, b"changed"), (1, b"changed")], media) != original
    assert ledger([(1, b"original"), (1, b"original")],
                  [{"type": "image_url", "image_url": "renamed.png"}]) == original
    assert ledger([(1, b"")], media) != ledger([(1, b"")], media)


def test_media_chunk_identity_includes_video_sampling(chunk_ledger):
    handler, ledger = chunk_ledger
    media = [{"type": "video", "video": "same.mp4"}]
    first = ledger([(1, b"video+0")], media)
    handler._mtmd_helper_init_opt.video_params.fps_target = 0.5
    assert ledger([(1, b"video+0")], media) != first


def test_media_ledger_and_following_text_use_native_positions(chunk_ledger, monkeypatch):
    handler, _ = chunk_ledger
    backend = handler._mtmd_cpp
    chunks = [SimpleNamespace(kind=0, tokens=[10, 11]),
              SimpleNamespace(kind=1, identity=b"image"),
              SimpleNamespace(kind=0, tokens=[12, 13])]
    monkeypatch.setattr(handler, "_mtmd_tokenize", lambda **kwargs: chunks)
    backend.mtmd_input_chunk_get_n_tokens = lambda chunk: 64
    backend.mtmd_input_chunk_get_n_pos = lambda chunk: 8

    def text_tokens(chunk, count):
        count._obj.value = len(chunk.tokens)
        return chunk.tokens

    backend.mtmd_input_chunk_get_tokens_text = text_tokens
    ledger, spans, _, _ = handler._process_mtmd_prompt(
        llama=SimpleNamespace(n_threads=1),
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": "same.png"}]}],
    )
    assert [(span[0], span[1]) for span in spans] == [(0, 2), (2, 10), (10, 12)]
    assert ledger == [10, 11] + [spans[1][4]] * 8 + [12, 13]


def test_media_cache_token_checks_full_identity_on_hash_collision(chunk_ledger, monkeypatch):
    handler, _ = chunk_ledger
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    monkeypatch.setattr(multimodal.hashlib, "sha256", lambda data: SimpleNamespace(digest=lambda: b"\0" * 32))
    first = handler._media_cache_token(("first",))
    second = handler._media_cache_token(("second",))
    assert first != second
    assert handler._media_cache_token(("first",)) == first
    assert handler._media_cache_token(("second",)) == second
    assert -(2**31) <= second < first < 0


def test_mtmd_base_image_loader_uses_subclass_byte_loader():
    import io
    from PIL import Image
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    payload = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 128)).save(payload, format="PNG")

    class CustomMediaHandler(multimodal.MTMDBaseHandler):
        @staticmethod
        def _load_bytes(media_url, timeout=15, kind="media"):
            assert media_url == "custom-image"
            assert kind == "image"
            return payload.getvalue()

    result = CustomMediaHandler._load_image("custom-image")
    with Image.open(io.BytesIO(result)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (2, 2)


@pytest.fixture
def audio_pipeline(tmp_path, monkeypatch):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    native = importlib.import_module("llama_cpp.mtmd_cpp")
    generator = multimodal.MTMDAudioGenerator(mmproj_path=str(tmp_path), verbose=False)
    hidden = (ctypes.c_float * 2)(1, 2)
    output = ctypes.create_string_buffer(struct.pack("4f", 0.0, 0.1, -0.1, 0.0))
    import io, wave
    state = SimpleNamespace(kind=1, steps=0, fail=None, prompts=[], tokens=[], stop_at=3)

    def set_input(ctx, ptr):
        inp = ctypes.cast(ptr, ctypes.POINTER(native.mtmd_helper_gen_audio_inp)).contents
        assert inp.stream is False and inp.seq_id == 0
        state.prompts.append((ctypes.string_at(inp.prompt, inp.prompt_len), inp.seed))
        state.steps = 0
        state.out_type = inp.out_type
        return 1 if state.fail == "set_input" else 0

    def step_gen(ctx, token, previous, next_ptr, stop_ptr):
        state.steps += 1
        state.tokens.append(token)
        assert previous[0] == 1
        if state.fail == "step_gen":
            return 1
        stop = state.steps >= state.stop_at
        ctypes.cast(stop_ptr, ctypes.POINTER(ctypes.c_bool))[0] = stop
        if not stop and state.fail != "missing_hidden":
            ctypes.cast(next_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))[0] = ctypes.cast(hidden, ctypes.POINTER(ctypes.c_float))
        return 0

    def get_output(ctx, rate, data, size, samples):
        sample_rate = 32000 if state.kind == 3 else 24000
        ctypes.cast(rate, ctypes.POINTER(ctypes.c_int32))[0] = sample_rate
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav:
            wav.setparams((1, 2, sample_rate, 4, "NONE", "not compressed"))
            wav.writeframes(b"\x00\x00\x01\x00\xff\xff\x00\x00")
        state.wav_output = ctypes.create_string_buffer(wav_buffer.getvalue())
        buffer = output if state.out_type == 0 else state.wav_output
        ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = ctypes.cast(buffer, ctypes.c_char_p)
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = len(buffer) - 1
        ctypes.cast(samples, ctypes.POINTER(ctypes.c_int64))[0] = 4
        return 1 if state.fail == "get_output" else 0

    backend = SimpleNamespace(
        mtmd_gen_audio_type=native.mtmd_gen_audio_type,
        mtmd_helper_gen_audio_outtype=native.mtmd_helper_gen_audio_outtype,
        mtmd_helper_gen_audio_inp=native.mtmd_helper_gen_audio_inp,
        mtmd_gen_audio_get_info=lambda ctx: SimpleNamespace(type=state.kind),
        mtmd_helper_gen_audio_init=Mock(return_value=456),
        mtmd_helper_gen_audio_free=Mock(), mtmd_helper_gen_audio_reset=Mock(),
        mtmd_helper_gen_audio_set_input=set_input,
        mtmd_helper_gen_audio_step_prompt=lambda ctx, batch: -1 if state.fail == "step_prompt" else 0,
        mtmd_helper_gen_audio_step_gen=step_gen,
        mtmd_helper_gen_audio_get_output=get_output,
        mtmd_bitmap_is_audio=lambda bitmap: True,
        mtmd_bitmap_free=Mock(), mtmd_helper_video_free=Mock(), mtmd_free=Mock(),
    )
    generator._mtmd_cpp = backend
    generator.mtmd_ctx = 123
    generator.is_support_audio = True
    monkeypatch.setattr(generator, "_create_bitmap_from_bytes", lambda payload: (789, None))
    monkeypatch.setattr(multimodal.llama_cpp_lib, "llama_get_embeddings_ith", lambda ctx, idx: ctypes.cast(hidden, ctypes.POINTER(ctypes.c_float)))
    sampler = Mock()
    sampler.sample.return_value = 42
    sampler_factory = Mock(return_value=sampler)
    monkeypatch.setattr(generator, "_create_audio_sampler", sampler_factory)
    llama = SimpleNamespace(
        _ctx=SimpleNamespace(ctx=321, pooling_type=lambda: 0, n_batch=lambda: 64),
        context_params=SimpleNamespace(embeddings=True),
        _abort_event=threading.Event(), _native_abort_flag=ctypes.c_bool(False),
        reset=Mock(),
    )
    yield generator, llama, backend, state, sampler_factory, output
    generator.close()


@pytest.mark.parametrize("kind", [1, 3], ids=["qwen", "soprano"])
def test_tts_sampled_models_repeated_requests_copy_binary_output(audio_pipeline, kind):
    generator, llama, backend, state, factory, output = audio_pipeline
    state.kind = kind
    generator.is_support_audio = kind != 3
    text = "Hello" if kind == 3 else "你好"
    expected = output.raw[:-1]
    for _ in range(2):
        result = generator.create_speech(llama=llama, text=text, seed=7, response_format="pcm_f32")
        assert result.data == expected
        assert result.finish_reason == "stop"
        assert result.sample_rate == (32000 if kind == 3 else 24000)
        assert result.duration == 4 / result.sample_rate
    output[0] = b"z"
    assert result.data == expected
    assert state.prompts == [(text.encode(), 7)] * 2
    assert state.tokens == [42] * 6
    assert llama.reset.call_count == 4
    assert factory.return_value.accept.call_count == 6
    assert factory.return_value.close.call_count == 2
    assert backend.mtmd_helper_gen_audio_reset.call_count == 2
    backend.mtmd_helper_gen_audio_init.assert_called_once()
    generator.close()
    backend.mtmd_helper_gen_audio_free.assert_called_once_with(456)


def test_tts_pocket_skips_sampler_and_limits_steps(audio_pipeline):
    generator, llama, backend, state, factory, _ = audio_pipeline
    state.kind = 2
    result = generator.create_speech(
        llama=llama, text="hello", speaker_reference=b"RIFF0000WAVE", max_frames=1,
    )
    assert result.finish_reason == "length"
    assert state.tokens == [-1]
    factory.assert_not_called()
    backend.mtmd_bitmap_free.assert_called_once_with(789)


@pytest.mark.parametrize("stage", [
    "set_input", "step_prompt", "step_gen", "get_output", "missing_hidden", "abort", "invalid_output",
])
@pytest.mark.parametrize("kind", [1, 3], ids=["qwen", "soprano"])
def test_tts_failure_can_be_followed_by_success(audio_pipeline, monkeypatch, stage, kind):
    generator, llama, backend, state, _, output = audio_pipeline
    state.kind = kind
    valid = output.raw
    response_format = "pcm_f32" if stage == "invalid_output" else "wav"
    with monkeypatch.context() as patch:
        state.fail = stage
        if stage == "abort":
            def abort(ctx, batch):
                llama._abort_event.set()
                return -1
            patch.setattr(backend, "mtmd_helper_gen_audio_step_prompt", abort)
        elif stage == "invalid_output":
            ctypes.memmove(output, struct.pack("4f", -1, -1, -1, -1), 16)
        with pytest.raises(InterruptedError if stage == "abort" else RuntimeError,
                           match="invalid audio" if stage == "invalid_output" else None):
            generator.create_speech(llama=llama, text="hello", response_format=response_format)
    backend.mtmd_helper_gen_audio_reset.assert_called_once()
    assert llama.reset.call_count == 2
    ctypes.memmove(output, valid, len(valid))
    state.fail = None
    assert generator.create_speech(llama=llama, text="hello", response_format=response_format).finish_reason == "stop"


def test_tts_soprano_constraints_and_frame_limit(audio_pipeline):
    generator, llama, backend, state, factory, _ = audio_pipeline
    state.kind = 3
    for options, message in [
        ({"speaker_reference": "missing.wav"}, "speaker_reference"),
        ({"language": "en"}, "omit language"),
        ({"max_frames": 1}, "max_frames >= 2"),
    ]:
        with pytest.raises(ValueError, match=message):
            generator.create_speech(llama=llama, text="hello", **options)
    backend.mtmd_helper_gen_audio_init.assert_not_called()
    llama.reset.assert_not_called()
    factory.assert_not_called()
    audio = generator.create_speech(llama=llama, text="hello", max_frames=2)
    assert audio.finish_reason == "length" and audio.sample_rate == 32000
    assert state.tokens == [42, 42]


def test_tts_validates_model_and_request_before_decode(audio_pipeline):
    generator, llama, backend, state, _, _ = audio_pipeline
    for options in ({"text": " "}, {"max_frames": 0}, {"seed": -1}, {"response_format": "mp3"}):
        with pytest.raises(ValueError):
            generator.create_speech(**{"llama": llama, "text": "hello", **options})
    state.kind = 2
    with pytest.raises(ValueError, match="requires speaker_reference"):
        generator.create_speech(llama=llama, text="hello")
    state.kind = 0
    with pytest.raises(ValueError, match="supported TTS"):
        generator.create_speech(llama=llama, text="hello")
    backend.mtmd_helper_gen_audio_init.assert_not_called()


def test_tts_rejects_rebinding_and_closed_generator(audio_pipeline):
    generator, llama, _, _, _, _ = audio_pipeline
    generator.create_speech(llama=llama, text="hello")
    with pytest.raises(ValueError, match="different Llama"):
        generator.create_speech(llama=SimpleNamespace(**vars(llama)), text="hello")
    generator.close()
    with pytest.raises(RuntimeError, match="closed"):
        generator.create_speech(llama=llama, text="hello")


@pytest.mark.parametrize("invalid", ["embeddings", "pooling", "closed", "batch"])
def test_tts_rejects_unusable_llama_context(audio_pipeline, invalid):
    generator, llama, backend, _, _, _ = audio_pipeline
    if invalid == "embeddings":
        llama.context_params.embeddings = False
    elif invalid == "pooling":
        llama._ctx.pooling_type = lambda: 1
    elif invalid == "closed":
        llama._ctx.ctx = None
    else:
        generator.batch_max_tokens = 0
    with pytest.raises((ValueError, RuntimeError)):
        generator.create_speech(llama=llama, text="hello")
    backend.mtmd_helper_gen_audio_init.assert_not_called()
    llama.reset.assert_not_called()


def test_tts_rejects_concurrent_use(audio_pipeline):
    generator, llama, backend, _, _, _ = audio_pipeline
    with generator._request_lock:
        with pytest.raises(RuntimeError, match="already synthesizing"):
            generator.create_speech(llama=llama, text="hello")
        with pytest.raises(RuntimeError, match="during synthesis"):
            generator.close()
    assert generator.create_speech(llama=llama, text="hello").finish_reason == "stop"


def test_generated_audio_wav_save(tmp_path):
    import io
    import wave
    from llama_cpp.llama_multimodal import GeneratedAudio

    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00\x01\x00")
    result = GeneratedAudio(data.getvalue(), 24000, 2, "wav", "stop")
    path = tmp_path / "speech.wav"
    result.save(path)
    with wave.open(str(path), "rb") as wav:
        assert wav.getnframes() == result.n_samples
        assert wav.getframerate() == result.sample_rate
        assert wav.readframes(2) == b"\x00\x00\x01\x00"


@pytest.mark.parametrize("values", [[], [float("nan"), 0.0], [float("inf"), 0.0], [-1.0]*4, [0.0]*4, [0.2]*4, [-1.0, 1.0]*50, [2.0, 0.0]])
def test_tts_rejects_invalid_pcm(values):
    from llama_cpp.llama_multimodal import MTMDAudioGenerator
    with pytest.raises(RuntimeError):
        MTMDAudioGenerator._validate_audio(struct.pack(f"{len(values)}f", *values), "pcm_f32", 24000, len(values))


def test_tts_audio_validation_keeps_quiet_speech_and_checks_lengths():
    from llama_cpp.llama_multimodal import MTMDAudioGenerator
    payload = struct.pack("4f", 0, 1e-8, -1e-8, 0)
    MTMDAudioGenerator._validate_audio(payload, "pcm_f32", 24000, 4)
    with pytest.raises(RuntimeError, match="byte length"):
        MTMDAudioGenerator._validate_audio(payload, "pcm_f32", 24000, 5)
    with pytest.raises(RuntimeError, match="malformed WAV"):
        MTMDAudioGenerator._validate_audio(b"bad", "wav", 24000, 4)


@pytest.fixture(scope="module", params=["plain", "ngram", "mtp", "mtp-device"])
def vision_model(request):
    from llama_cpp import Llama
    from llama_cpp.llama_multimodal import Qwen35ChatHandler
    from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
    paths = [os.environ.get(name) for name in
             ("LLAMA_TEST_HYBRID_MODEL", "LLAMA_TEST_MMPROJ")]
    if not all(paths):
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("LLAMA_TEST_HYBRID_MODEL and LLAMA_TEST_MMPROJ are required in Actions")
        pytest.skip("Set LLAMA_TEST_HYBRID_MODEL and LLAMA_TEST_MMPROJ for media tests")
    assert all(os.path.isfile(path) for path in paths), paths
    handler = Qwen35ChatHandler(mmproj_path=paths[1], use_gpu=False,
                               image_max_tokens=64, verbose=False, enable_thinking=False)
    llm = None
    try:
        configs = {
            "ngram": SpecConfig(spec_type=SpeculativeType.NGRAM_MAP_K,
                                ngram_size_n=2, ngram_size_m=4),
            "mtp": SpecConfig(spec_type=SpeculativeType.DRAFT_MTP, draft_n_max=3),
            "mtp-device": SpecConfig(spec_type=SpeculativeType.DRAFT_MTP, draft_n_max=3),
        }
        llm = Llama(model_path=paths[0], chat_handler=handler, n_ctx=512,
                    n_batch=32, n_ubatch=32, n_gpu_layers=0, verbose=False,
                    ctx_checkpoints=4, speculative=configs.get(request.param))
        if request.param == "mtp-device":
            llm._ctx.n_rs_seq = lambda: 0
            llm.speculative._use_native_draft_rollback = False
        yield llm, handler
    finally:
        if llm is not None:
            llm.close()
        handler.close()


def _color_chat(llm, color, stream=False):
    import base64
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return llm.create_chat_completion(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Name the color. Answer in one word."},
    ]}], max_tokens=8, temperature=0, seed=42, stream=stream)


def test_real_media_changed_image_and_interrupted_retry(vision_model, monkeypatch):
    import numpy as np

    llm, _ = vision_model
    references = {}
    ledgers = {}
    engine = llm.speculative
    is_mtp = getattr(engine, "process_media", None) is not None
    media_rows = []
    if is_mtp:
        original = engine.process_media
        def consume(batch):
            media_rows.append(batch.n_tokens)
            target_h = engine._copy_rows(llm._ctx.get_embeddings_nextn(), batch.n_tokens, engine.n_embd)
            result = original(batch)
            np.testing.assert_array_equal(engine.pending_h, target_h[-1])
            assert engine.draft_context.memory_seq_pos_max(0) == llm._ctx.memory_seq_pos_max(0)
            return result
        monkeypatch.setattr(engine, "process_media", consume)
    for color in ("red", "blue"):
        # Keep the cold reference on the same decoding path: quantized target-only
        # and MTP verification batches need not produce identical greedy text.
        llm.reset()
        media_rows.clear()
        references[color] = _color_chat(llm, color)["choices"][0]["message"]["content"]
        assert references[color]
        if is_mtp:
            assert len(media_rows) > 1 and max(media_rows) <= llm.n_batch
            stats = llm.last_speculative_stats
            assert stats["drafted"] > 0 and stats["verified"] > 0
            assert stats["checkpoint_device_captures" if not engine._use_native_draft_rollback
                         else "checkpoint_native_captures"] > 0
        ledgers[color] = [int(t) for t in llm._input_ids if t < 0]
        assert ledgers[color]
    assert ledgers["red"] != ledgers["blue"]
    for color in ("red", "red", "blue"):
        assert _color_chat(llm, color)["choices"][0]["message"]["content"] == references[color]
    stream = _color_chat(llm, "red", stream=True)
    try:
        next(stream)
        next(stream)
    finally:
        stream.close()
    assert _color_chat(llm, "blue")["choices"][0]["message"]["content"] == references["blue"]
    llm.reset()


def test_real_media_partial_decode_failure_then_retry(vision_model, monkeypatch):
    llm, handler = vision_model
    llm.reset()
    reference = _color_chat(llm, "blue")["choices"][0]["message"]["content"]
    helper = ("mtmd_helper_decode_image_chunk" if
              getattr(llm.speculative, "process_media", None) is not None else
              "mtmd_helper_eval_chunk_single")
    original = getattr(handler._mtmd_cpp, helper)
    committed = []
    def fail_after_commit(*args):
        assert original(*args) == 0
        committed.append(llm._ctx.memory_seq_pos_max(0))
        return 1
    with monkeypatch.context() as patch:
        patch.setattr(handler._mtmd_cpp, helper, fail_after_commit)
        with pytest.raises(ValueError, match="Media evaluation failed"):
            _color_chat(llm, "red")
    assert committed and committed[0] >= 0
    assert llm.n_tokens == 0 and llm._ctx.memory_seq_pos_max(0) == -1
    assert llm._prefilled_prompt is None and llm._restored_logits is None
    assert not llm._hybrid_cache_mgr.checkpoints
    if getattr(llm.speculative, "process_media", None) is not None:
        assert llm.speculative.draft_context.memory_seq_pos_max(0) == -1
        assert not llm.speculative.pending_h.any()
    assert _color_chat(llm, "blue")["choices"][0]["message"]["content"] == reference
    llm.reset()
