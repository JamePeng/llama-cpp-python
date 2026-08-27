import ctypes
import importlib
import os

import pytest


def test_import_mtmd_cpp():
    module = importlib.import_module("llama_cpp.mtmd_cpp")

    assert module is not None


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


def test_mtmd_chat_handler_video_options(tmp_path):
    module = importlib.import_module("llama_cpp.mtmd_cpp")
    multimodal = importlib.import_module("llama_cpp.llama_multimodal")

    executable_suffix = ".exe" if os.name == "nt" else ""
    for executable_name in ("ffmpeg", "ffprobe"):
        executable_path = tmp_path / (executable_name + executable_suffix)
        executable_path.touch()
        executable_path.chmod(0o755)

    handler = multimodal.MTMDChatHandler(
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
    handler = multimodal.MTMDChatHandler(
        mmproj_path=str(tmp_path),
        mtmd_helper_init_opt=opt,
    )
    assert handler._mtmd_helper_init_opt is opt

    with pytest.raises(ValueError, match="cannot be combined"):
        multimodal.MTMDChatHandler(
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
