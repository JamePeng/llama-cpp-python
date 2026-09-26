import ctypes
import importlib
import os
from types import MethodType, SimpleNamespace
from unittest.mock import Mock
import threading
import struct

import pytest
from jinja2 import Template
from jinja2.exceptions import TemplateError


@pytest.fixture
def chat_prefill_builder(tmp_path, monkeypatch):
    import numpy as np
    from llama_cpp import Llama
    from llama_cpp import llama_multimodal as multimodal

    def build(handler_class=multimodal.MTMDChatHandler, **handler_kwargs):
        handler_kwargs.setdefault("mmproj_path", str(tmp_path))
        handler_kwargs.setdefault("verbose", False)
        handler = handler_class(**handler_kwargs)
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
            _n_ctx=6, n_ctx=lambda: 6, n_vocab=lambda: 3, n_batch=1, n_keep=0,
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
        llm._mark_prefilled_prompt = Mock(
            side_effect=lambda: setattr(
                llm, "_restored_logits", np.array([0.25, -0.5, 1.0], dtype=np.float32)
            )
        )

        def evaluate(mtmd, ctx, chunk, pos, seq, batch, logits, output):
            output._obj.value = pos.value + 2
            return 0

        backend = SimpleNamespace(
            mtmd_input_chunk_get_n_tokens=lambda chunk: 2,
            mtmd_helper_eval_chunk_single=Mock(side_effect=evaluate),
        )
        monkeypatch.setattr(handler, "_mtmd_cpp", backend)
        return handler, llm, backend

    return build


@pytest.fixture
def chat_prefill(chat_prefill_builder):
    return chat_prefill_builder()


def test_chat_prefill_returns_owned_logits_without_generation(chat_prefill):
    import numpy as np
    from llama_cpp import PrefillResult

    handler, llm, _ = chat_prefill
    result = handler(llama=llm, messages=[], prefill_only=True)

    assert isinstance(result, PrefillResult)
    assert not hasattr(result, "prompt")
    assert not hasattr(result, "n_tokens")
    assert llm.n_tokens == 4
    assert result.logits.shape == (llm.n_vocab(),)
    assert result.logits.dtype == np.float32
    assert result.logits.flags.owndata
    assert not result.logits.flags.writeable
    assert not np.shares_memory(result.logits, llm._restored_logits)
    np.testing.assert_array_equal(result.logits, [0.25, -0.5, 1.0])
    llm.create_completion.assert_not_called()

    saved_logits = result.logits.copy()
    llm.reset()
    np.testing.assert_array_equal(result.logits, saved_logits)


def test_chat_completion_uses_prefill_and_preserves_generation_args(chat_prefill):
    handler, llm, _ = chat_prefill

    assert handler(
        llama=llm, messages=[], add_generation_prompt=False, temperature=0.7, top_p=0.8
    ) == "completed"

    completion_args = llm.create_completion.call_args.kwargs
    assert completion_args["prompt"] == [1, 2, -9, -9]
    assert completion_args["temperature"] == 0.7
    assert completion_args["top_p"] == 0.8


def test_chat_completion_ignores_legacy_subclass_prefill_name(chat_prefill_builder):
    from llama_cpp import llama_multimodal as multimodal

    class LegacyChatHandler(multimodal.MTMDChatHandler):
        def prefill(self, *args, **kwargs):
            raise AssertionError("legacy prefill override must not intercept generation")

    handler, llm, _ = chat_prefill_builder(LegacyChatHandler)

    assert handler(llama=llm, messages=[]) == "completed"
    assert llm.create_completion.call_args.kwargs["prompt"] == [1, 2, -9, -9]


def test_external_handler_forwards_prefill_only_through_call(chat_prefill_builder):
    from llama_cpp import llama_multimodal as multimodal
    from llama_cpp import PrefillResult

    class WrappedChatHandler(multimodal.MTMDChatHandler):
        def __call__(self, **kwargs):
            self.wrapper_ran = True
            return super().__call__(**kwargs)

    handler, llm, _ = chat_prefill_builder(WrappedChatHandler)
    result = handler(llama=llm, messages=[], prefill_only=True)

    assert handler.wrapper_ran
    assert isinstance(result, PrefillResult)
    llm.create_completion.assert_not_called()


def test_generic_chat_prefill_resolves_model_template(chat_prefill_builder):
    from llama_cpp import llama_multimodal as multimodal
    from llama_cpp import PrefillResult

    handler, llm, _ = chat_prefill_builder(
        multimodal.GenericMTMDChatHandler, chat_format=None
    )
    template = "{% for message in messages %}{{ message.content }}{% endfor %}<|image|>"
    llm._model = SimpleNamespace(model_chat_template=Mock(return_value=template))

    result = handler(llama=llm, messages=[], prefill_only=True)

    assert isinstance(result, PrefillResult)
    llm._model.model_chat_template.assert_called_once_with(None)
    assert handler._template_initialized
    assert handler._chat_format_parser_tags == ["<|image|>"]


@pytest.mark.parametrize(
    "resolution", ["chat_handler", "chat_format", "global_registry"]
)
def test_llama_create_chat_prefill_uses_selected_mtmd_handler(
    chat_prefill_builder, resolution, monkeypatch
):
    import numpy as np
    from llama_cpp import Llama
    from llama_cpp import llama_multimodal as multimodal
    from llama_cpp import PrefillResult

    handler, llm, backend = chat_prefill_builder(
        multimodal.GenericMTMDChatHandler, chat_format=None
    )
    template = "{% for message in messages %}{{ message.content }}{% endfor %}<|image|>"
    llm._model = SimpleNamespace(model_chat_template=Mock(return_value=template))
    llm._get_chat_completion_handler = MethodType(
        Llama._get_chat_completion_handler, llm
    )
    llm.create_chat_prefill = MethodType(Llama.create_chat_prefill, llm)
    llm.chat_handler = handler if resolution == "chat_handler" else None
    llm.chat_format = "test-prefill"
    llm._chat_handlers = (
        {"test-prefill": handler} if resolution == "chat_format" else {}
    )
    registry_lookup = Mock(return_value=handler)
    monkeypatch.setattr(
        "llama_cpp.llama.llama_chat_format.get_chat_completion_handler",
        registry_lookup,
    )

    messages = [{"role": "user", "content": "inspect this"}]
    functions = [{
        "name": "lookup",
        "parameters": {"type": "object", "properties": {}},
    }]
    result = llm.create_chat_prefill(
        messages=messages,
        functions=functions,
        function_call="auto",
        tools=[],
        tool_choice="none",
        add_generation_prompt=False,
    )

    assert isinstance(result, PrefillResult)
    assert llm.n_tokens == 4
    assert llm._restored_logits is not None
    assert not llm.reset.called
    llm._ctx.memory_clear.assert_not_called()
    backend.mtmd_helper_eval_chunk_single.assert_called_once()
    assert not np.shares_memory(result.logits, llm._restored_logits)
    llm.create_completion.assert_not_called()
    llm._model.model_chat_template.assert_called_once_with(None)
    processed = handler._process_mtmd_prompt.call_args.kwargs
    assert processed["messages"] is messages
    assert processed["functions"] is functions
    assert processed["function_call"] == "auto"
    assert processed["tools"] == []
    assert processed["tool_choice"] == "none"
    assert processed["add_generation_prompt"] is False
    if resolution == "global_registry":
        registry_lookup.assert_called_once_with("test-prefill")
    else:
        registry_lookup.assert_not_called()


def test_llama_create_chat_prefill_rejects_non_mtmd_handler():
    from llama_cpp import Llama

    handler = lambda **kwargs: None
    llm = SimpleNamespace(
        chat_handler=handler,
        _chat_handlers={},
        chat_format="unused",
    )
    llm._get_chat_completion_handler = MethodType(
        Llama._get_chat_completion_handler, llm
    )
    llm.create_chat_prefill = MethodType(Llama.create_chat_prefill, llm)

    with pytest.raises(NotImplementedError, match="handler does not support prefill"):
        llm.create_chat_prefill(messages=[])


def test_minicpmv45_prefill_prepares_prompt_and_keeps_generation_stops(
    chat_prefill_builder,
):
    from llama_cpp import llama_multimodal as multimodal
    from llama_cpp import PrefillResult

    handler, llm, _ = chat_prefill_builder(
        multimodal.MiniCPMv45ChatHandler, enable_thinking=False
    )
    llm.input_ids[:] = [1, 2, 3, 4, 5, 6]
    observed = []

    def process_prompt(**kwargs):
        observed.append((llm.input_ids.copy(), handler.extra_template_arguments.copy()))
        return [1, 2, -9, -9], [(2, 4, object(), 1, -9)], object(), []

    handler._process_mtmd_prompt.side_effect = process_prompt
    result = handler.prefill(llama=llm, messages=[])

    assert isinstance(result, PrefillResult)
    llm.create_completion.assert_not_called()
    assert observed[0][0].tolist() == [0, 0, 0, 0, 0, 0]
    assert observed[0][1]["enable_thinking"] is False

    llm.reset()
    handler(llama=llm, messages=[], stop=["caller-stop"])
    assert llm.create_completion.call_args.kwargs["stop"] == [
        handler.MINICPMV_EOS_TOKEN,
        handler.MINICPMV_PAD_TOKEN,
    ]
    llm.create_completion.assert_called_once()


@pytest.mark.parametrize("failure", ["rollback", "shift", "helper", "position", "interrupt"])
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
    elif failure == "position":
        def invalid_position(*args):
            args[-1]._obj.value = 7
            return 0
        backend.mtmd_helper_eval_chunk_single.side_effect = invalid_position
    elif failure == "interrupt":
        backend.mtmd_helper_eval_chunk_single.side_effect = KeyboardInterrupt

    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt)):
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


@pytest.mark.parametrize("mode", ["plain", "hybrid", "shift"])
def test_chat_prefill_success_hands_off_rebuilt_prompt(chat_prefill, mode):
    handler, llm, backend = chat_prefill
    expected = [1, 2, -9, -9]
    if mode == "shift":
        llm.n_tokens = 6
        llm.input_ids[:] = [1, 2, 3, 4, 5, 6]
        handler._process_mtmd_prompt.return_value = (
            [1, 2, 3, 4, 5, 6, -9, -9], [(6, 8, object(), 1, -9)], object(), []
        )
        expected = [4, 5, 6, -9, -9]
    if mode == "hybrid":
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
    if mode == "hybrid":
        llm._hybrid_cache_mgr.clear.assert_not_called()
        llm._hybrid_cache_mgr.save_checkpoint.assert_called_once()


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
def test_mtmd_chat_handler_video_options(tmp_path, handler_name):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler_class = getattr(multimodal, handler_name)

    executable_suffix = ".exe" if os.name == "nt" else ""
    for executable_name in ("ffmpeg", "ffprobe"):
        executable_path = tmp_path / (executable_name + executable_suffix)
        executable_path.touch()
        executable_path.chmod(0o755)

    handler = handler_class(
        mmproj_path=str(tmp_path),
        video_fps_target=2.5,
        video_ffmpeg_bin_dir=tmp_path,
        video_timestamp_interval_ms=10000,
    )
    video_params = handler._mtmd_helper_init_opt.video_params
    assert video_params.fps_target == 2.5
    assert video_params.ffmpeg_bin_dir == os.fsencode(os.path.abspath(tmp_path))
    assert video_params.timestamp_interval_ms == 10000

    opt = module.mtmd_helper_init_opt_default()
    handler = handler_class(
        mmproj_path=str(tmp_path),
        mtmd_helper_init_opt=opt,
    )
    assert handler._mtmd_helper_init_opt is opt

    with pytest.raises(ValueError, match="cannot be combined"):
        handler_class(
            mmproj_path=str(tmp_path),
            mtmd_helper_init_opt=opt,
            video_fps_target=1.0,
        )


def test_mtmd_chat_handler_rejects_invalid_ffmpeg_bin_dir(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")

    with pytest.raises(ValueError, match="is not an existing directory"):
        multimodal.MTMDChatHandler(
            mmproj_path=str(tmp_path),
            video_ffmpeg_bin_dir=tmp_path / "missing",
        )

    with pytest.raises(ValueError, match="ffmpeg and ffprobe"):
        multimodal.MTMDChatHandler(
            mmproj_path=str(tmp_path),
            video_ffmpeg_bin_dir=tmp_path,
        )


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
        mtmd_default_marker=Mock(return_value=b"<media>"),
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
        model.detokenize = lambda tokens: {1: b"<bos>", 2: b"<eos>"}[tokens[0]]
    try:
        handler._init_mtmd_context(model)
        handler._init_mtmd_context(model)
        backend.mtmd_init_from_file.assert_called_once()
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


def test_mtmd_render_prompt_applies_request_template_kwargs_without_mutating_defaults():
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    template_arguments = {"enable_thinking": False}
    handler = SimpleNamespace(
        chat_template=Template("{{ 'enabled' if enable_thinking else 'disabled' }}"),
        extra_template_arguments=template_arguments,
        verbose=False,
        log_prefix="test",
        mtmd_eos_token="",
        mtmd_bos_token="",
    )

    prompt = multimodal.MTMDChatHandler._render_mtmd_prompt(
        handler,
        messages=[],
        chat_template_kwargs={"enable_thinking": True},
    )

    assert prompt == "enabled"
    assert template_arguments == {"enable_thinking": False}


def test_mtmd_chat_template_raise_exception_preserves_message(tmp_path):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = multimodal.MTMDChatHandler(
        mmproj_path=str(tmp_path),
        verbose=False,
        chat_template_override=(
            "{% if video %}{{ raise_exception('Video not supported') }}"
            "{% else %}text supported{% endif %}"
        ),
    )
    try:
        assert handler.chat_template.render(video=False) == "text supported"
        with pytest.raises(TemplateError, match="Video not supported"):
            handler.chat_template.render(video=True)
    finally:
        handler.close()


def test_mtmd_chat_template_strftime_now(tmp_path):
    import datetime

    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = multimodal.MTMDChatHandler(
        mmproj_path=str(tmp_path),
        verbose=False,
        chat_template_override="{{ strftime_now('%Y') }}",
    )
    try:
        assert handler.chat_template.render() == datetime.datetime.now().strftime("%Y")
    finally:
        handler.close()


@pytest.mark.parametrize(
    "video_item",
    [
        {"type": "video", "video": "/path/to/clip.mp4"},
        {"type": "video", "video": {"url": "/path/to/clip.mp4"}},
        {"type": "video_url", "video_url": "/path/to/clip.mp4"},
        {"type": "video_url", "video_url": {"url": "/path/to/clip.mp4"}},
    ],
)
def test_qwen35_video_template_uses_mtmd_marker(tmp_path, video_item):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = multimodal.Qwen35ChatHandler(mmproj_path=str(tmp_path), verbose=False)
    handler.mtmd_bos_token = ""
    handler.mtmd_eos_token = ""
    handler.media_marker = "<__media__>"
    handler.is_support_video = True
    messages = [{"role": "user", "content": [video_item]}]
    try:
        media_items = handler._get_media_items(messages)
        prompt = handler._render_and_replace_media(
            messages=messages,
            media_items=media_items,
            add_generation_prompt=False,
        )
        assert media_items == [{"url": "/path/to/clip.mp4", "type": "video"}]
        assert "Video 1: <|vision_start|><__media__><|vision_end|>" in prompt
        assert "<|video_pad|>" not in prompt
        with pytest.raises(TemplateError, match="System message cannot contain videos"):
            handler._render_mtmd_prompt(
                messages=[{"role": "system", "content": [video_item]}],
                add_generation_prompt=False,
            )
    finally:
        handler.close()


@pytest.mark.parametrize(
    "image_item",
    [
        {"image": "/path/to/image.png"},
        {"image_url": {"url": "/path/to/image.png"}},
        {"type": "image", "image": {"url": "/path/to/image.png"}},
        {"type": "image_url", "image_url": "/path/to/image.png"},
    ],
)
def test_qwen3vl_image_template_injects_media_without_type(tmp_path, image_item):
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")
    handler = multimodal.Qwen3VLChatHandler(mmproj_path=str(tmp_path), verbose=False)
    handler.mtmd_bos_token = ""
    handler.mtmd_eos_token = ""
    handler.media_marker = "<__media__>"
    handler.is_support_vision = True
    messages = [{"role": "user", "content": [image_item]}]
    try:
        media_items = handler._get_media_items(messages)
        prompt = handler._render_and_replace_media(
            messages=messages,
            media_items=media_items,
            add_generation_prompt=False,
        )
        assert media_items == [{"url": "/path/to/image.png", "type": "image"}]
        assert "Picture 1: <|vision_start|><__media__><|vision_end|>" in prompt
    finally:
        handler.close()


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
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setparams((1, 2, 24000, 4, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00\x01\x00\xff\xff\x00\x00")
    wav_output = ctypes.create_string_buffer(wav_buffer.getvalue())
    state = SimpleNamespace(kind=1, steps=0, fail=None, prompts=[], tokens=[], stop_at=2)

    def set_input(ctx, ptr):
        inp = ctypes.cast(ptr, ctypes.POINTER(native.mtmd_helper_gen_audio_inp)).contents
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
        ctypes.cast(rate, ctypes.POINTER(ctypes.c_int32))[0] = 24000
        buffer = output if state.out_type == 0 else wav_output
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


def test_tts_qwen_repeated_requests_copy_binary_output(audio_pipeline):
    generator, llama, backend, state, factory, output = audio_pipeline
    expected = output.raw[:-1]
    for _ in range(2):
        result = generator.create_speech(llama=llama, text="你好", seed=7, response_format="pcm_f32")
        assert result.data == expected
        assert result.finish_reason == "stop"
        assert result.duration == 4 / 24000
    output[0] = b"z"
    assert result.data == expected
    assert state.prompts == [("你好".encode(), 7)] * 2
    assert state.tokens == [42] * 4
    assert llama.reset.call_count == 4
    assert factory.return_value.accept.call_count == 4
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


@pytest.mark.parametrize("stage", ["set_input", "step_prompt", "step_gen", "get_output", "missing_hidden"])
def test_tts_failure_can_be_followed_by_success(audio_pipeline, stage):
    generator, llama, backend, state, factory, _ = audio_pipeline
    state.fail = stage
    with pytest.raises(RuntimeError):
        generator.create_speech(llama=llama, text="hello")
    assert backend.mtmd_helper_gen_audio_reset.call_count == 1
    assert llama.reset.call_count == 2
    state.fail = None
    assert generator.create_speech(llama=llama, text="hello").finish_reason == "stop"


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


def test_tts_abort_releases_request_resources(audio_pipeline):
    generator, llama, backend, state, _, _ = audio_pipeline
    def abort(ctx, batch):
        llama._abort_event.set()
        return -1
    backend.mtmd_helper_gen_audio_step_prompt = abort
    with pytest.raises(InterruptedError):
        generator.create_speech(llama=llama, text="hello")
    assert llama.reset.call_count == 2
    backend.mtmd_helper_gen_audio_reset.assert_called_once()


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


def test_tts_invalid_output_cleans_up_and_can_retry(audio_pipeline):
    generator, llama, backend, _, _, output = audio_pipeline
    valid = output.raw
    ctypes.memmove(output, struct.pack("4f", -1, -1, -1, -1), 16)
    with pytest.raises(RuntimeError, match="invalid audio"):
        generator.create_speech(llama=llama, text="hello", response_format="pcm_f32")
    backend.mtmd_helper_gen_audio_reset.assert_called_once()
    assert llama.reset.call_count == 2
    ctypes.memmove(output, valid, len(valid))
    assert generator.create_speech(llama=llama, text="hello", response_format="pcm_f32").finish_reason == "stop"


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "ngram"])
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
        llm = Llama(model_path=paths[0], chat_handler=handler, n_ctx=512,
                    n_batch=128, n_ubatch=128, n_gpu_layers=0, verbose=False,
                    speculative=SpecConfig(spec_type=SpeculativeType.NGRAM_MAP_K,
                                           ngram_size_n=2, ngram_size_m=4)
                    if request.param else None)
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
    Image.new("RGB", (96, 96), color).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return llm.create_chat_completion(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Name the color. Answer in one word."},
    ]}], max_tokens=8, temperature=0, seed=42, stream=stream)


def test_real_media_changed_image_and_interrupted_retry(vision_model):
    llm, _ = vision_model
    references = {}
    ledgers = {}
    for color in ("red", "blue"):
        llm.reset()
        references[color] = _color_chat(llm, color)["choices"][0]["message"]["content"]
        assert references[color]
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
    original = handler._mtmd_cpp.mtmd_helper_eval_chunk_single
    committed = []
    def fail_after_commit(*args):
        assert original(*args) == 0
        committed.append(llm._ctx.memory_seq_pos_max(0))
        return 1
    with monkeypatch.context() as patch:
        patch.setattr(handler._mtmd_cpp, "mtmd_helper_eval_chunk_single", fail_after_commit)
        with pytest.raises(ValueError, match="Media evaluation failed"):
            _color_chat(llm, "red")
    assert committed and committed[0] >= 0
    assert llm.n_tokens == 0 and llm._ctx.memory_seq_pos_max(0) == -1
    assert llm._prefilled_prompt is None and llm._restored_logits is None
    assert not llm._hybrid_cache_mgr.checkpoints
    assert _color_chat(llm, "blue")["choices"][0]["message"]["content"] == reference
    llm.reset()
