import sys
import ctypes
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, TextIO, Union

import llama_cpp._ggml as _ggml
import llama_cpp.llama_cpp as llama_cpp_lib

# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_INFO  = 1,
#     GGML_LOG_LEVEL_WARN  = 2,
#     GGML_LOG_LEVEL_ERROR = 3,
#     GGML_LOG_LEVEL_DEBUG = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };
GGML_LOG_LEVEL_NONE = 0
GGML_LOG_LEVEL_INFO = 1
GGML_LOG_LEVEL_WARN = 2
GGML_LOG_LEVEL_ERROR = 3
GGML_LOG_LEVEL_DEBUG = 4
GGML_LOG_LEVEL_CONT = 5

# common/log.h model:
#
#   LOG_LEVEL_OUTPUT = 0
#   LOG_LEVEL_ERROR  = 1
#   LOG_LEVEL_WARN   = 2
#   LOG_LEVEL_INFO   = 3
#   LOG_LEVEL_TRACE  = 4
#   LOG_LEVEL_DEBUG  = 5
#
# Rule:
#
#   event_verbosity <= verbosity_threshold  => print
#
# Larger threshold means more verbose output.
#
LOG_LEVEL_OUTPUT = 0
LOG_LEVEL_ERROR = 1
LOG_LEVEL_WARN = 2
LOG_LEVEL_INFO = 3
LOG_LEVEL_TRACE = 4
LOG_LEVEL_DEBUG = 5

LOG_DEFAULT_LLAMA = LOG_LEVEL_INFO
LOG_DEFAULT_DEBUG = LOG_LEVEL_DEBUG

# Match the updated common_log_default_callback behavior:
#   INFO -> TRACE
#   CONT -> TRACE
#
# This is slightly more conservative for verbosity=3:
# if the backend emits INFO through ggml_log_callback, Python will hide it unless
# verbosity >= 4. This mirrors the current upstream default callback behavior.
GGML_LEVEL_TO_VERBOSITY = {
    GGML_LOG_LEVEL_NONE: LOG_LEVEL_OUTPUT,
    GGML_LOG_LEVEL_ERROR: LOG_LEVEL_ERROR,
    GGML_LOG_LEVEL_WARN: LOG_LEVEL_WARN,
    GGML_LOG_LEVEL_INFO: LOG_LEVEL_TRACE,
    GGML_LOG_LEVEL_DEBUG: LOG_LEVEL_DEBUG,
    GGML_LOG_LEVEL_CONT: LOG_LEVEL_TRACE,  # fallback only; CONT inherits previous
}

GGML_LEVEL_TO_PYTHON_LEVEL = {
    GGML_LOG_LEVEL_NONE: logging.INFO,
    GGML_LOG_LEVEL_ERROR: logging.ERROR,
    GGML_LOG_LEVEL_WARN: logging.WARNING,
    GGML_LOG_LEVEL_INFO: logging.INFO,
    GGML_LOG_LEVEL_DEBUG: logging.DEBUG,
    GGML_LOG_LEVEL_CONT: logging.INFO,  # fallback only; CONT inherits previous
}


# Default substring filters.
#
# These are intentionally simple substring filters instead of hard-coded
# special branches. Users can replace or clear them with set_log_filters().
DEFAULT_LOG_FILTERS = [
    "CUDA Graph",
    "CUDA graph"
]


VerbosityLike = Union[bool, int, str, None]

logger = logging.getLogger("llama-cpp-python")


@dataclass
class LoggerConfig:
    # 0=output, 1=error, 2=warn, 3=info, 4=trace, 5=debug
    verbosity: int = LOG_DEFAULT_LLAMA

    show_output: bool = True

    stdout: TextIO = sys.stdout
    stderr: TextIO = sys.stderr

    # If any substring is contained in a log message, the message is dropped.
    log_filters: list[str] = field(default_factory=lambda: list(DEFAULT_LOG_FILTERS))
    log_filters_case_sensitive: bool = True


_config = LoggerConfig()
_last_verbosity = LOG_LEVEL_INFO


def _normalize_verbosity(
    value: VerbosityLike,
    *,
    default: int = LOG_DEFAULT_LLAMA,
) -> int:
    """
    Convert user input to llama.cpp-style verbosity 0..5.

    Compatibility:
        verbose=False -> ERROR (1)
        verbose=True  -> DEBUG (5)

    Numeric levels:
        0 = output
        1 = error
        2 = warn
        3 = info
        4 = trace
        5 = debug
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return LOG_LEVEL_DEBUG if value else LOG_LEVEL_ERROR

    if isinstance(value, int):
        return max(LOG_LEVEL_OUTPUT, min(LOG_LEVEL_DEBUG, value))

    if isinstance(value, str):
        key = value.strip().lower()
        aliases = {
            "0": LOG_LEVEL_OUTPUT,
            "output": LOG_LEVEL_OUTPUT,
            "none": LOG_LEVEL_OUTPUT,

            "1": LOG_LEVEL_ERROR,
            "error": LOG_LEVEL_ERROR,
            "err": LOG_LEVEL_ERROR,
            "silent": LOG_LEVEL_ERROR,

            "2": LOG_LEVEL_WARN,
            "warn": LOG_LEVEL_WARN,
            "warning": LOG_LEVEL_WARN,
            "quiet": LOG_LEVEL_WARN,

            "3": LOG_LEVEL_INFO,
            "info": LOG_LEVEL_INFO,
            "default": LOG_DEFAULT_LLAMA,
            "normal": LOG_DEFAULT_LLAMA,

            "4": LOG_LEVEL_TRACE,
            "trace": LOG_LEVEL_TRACE,
            "trc": LOG_LEVEL_TRACE,

            "5": LOG_LEVEL_DEBUG,
            "debug": LOG_LEVEL_DEBUG,
            "verbose": LOG_LEVEL_DEBUG,
        }

        if key in aliases:
            return aliases[key]

        try:
            parsed = int(key)
        except ValueError as exc:
            raise ValueError(
                "_logger._normalize_verbosity: "
                "verbosity must be one of 0..5, bool, None, or "
                "'silent'/'quiet'/'info'/'trace'/'debug'"
            ) from exc

        return max(LOG_LEVEL_OUTPUT, min(LOG_LEVEL_DEBUG, parsed))

    raise TypeError(f"_logger._normalize_verbosity: unsupported verbosity type: {type(value)!r}")


def _verbosity_to_python_level(verbosity: int) -> int:
    if verbosity >= LOG_LEVEL_DEBUG:
        return logging.DEBUG
    if verbosity >= LOG_LEVEL_INFO:
        return logging.INFO
    if verbosity >= LOG_LEVEL_WARN:
        return logging.WARNING
    return logging.ERROR


def _get_verbosity(level: int) -> int:
    """
    Map ggml log level to Python-side verbosity.

    GGML_LOG_LEVEL_INFO maps to LOG_LEVEL_INFO so that verbosity=3 remains
    useful as the default info level.
    """
    if level == GGML_LOG_LEVEL_NONE:
        return LOG_LEVEL_OUTPUT
    if level == GGML_LOG_LEVEL_ERROR:
        return LOG_LEVEL_ERROR
    if level == GGML_LOG_LEVEL_WARN:
        return LOG_LEVEL_WARN
    if level == GGML_LOG_LEVEL_INFO:
        return LOG_LEVEL_INFO
    if level == GGML_LOG_LEVEL_DEBUG:
        return LOG_LEVEL_DEBUG
    if level == GGML_LOG_LEVEL_CONT:
        return LOG_LEVEL_INFO
    return LOG_LEVEL_DEBUG


def _decode_log_text(text: bytes) -> str:
    return text.decode("utf-8", errors="replace")


def _matches_log_filter(msg: str) -> bool:
    filters = _config.log_filters
    if not filters:
        return False

    if _config.log_filters_case_sensitive:
        return any(item and item in msg for item in filters)

    msg_lower = msg.lower()
    return any(item and item.lower() in msg_lower for item in filters)


def _should_drop(level: int, verbosity: int, msg: str) -> bool:
    if verbosity > _config.verbosity:
        return True

    if level == GGML_LOG_LEVEL_NONE and not _config.show_output:
        return True

    if _matches_log_filter(msg):
        return True

    return False


@_ggml.ggml_log_callback
def ggml_log_callback(
    level: int,
    text: bytes,
    user_data: ctypes.c_void_p,
):
    global _last_verbosity

    msg = _decode_log_text(text)

    if level == GGML_LOG_LEVEL_CONT:
        verbosity = _last_verbosity
    else:
        verbosity = _get_verbosity(level)
        _last_verbosity = verbosity

    if _should_drop(level, verbosity, msg):
        return

    out = _config.stdout if level == GGML_LOG_LEVEL_NONE else _config.stderr
    print(msg, end="", flush=True, file=out)


# Keep a global reference to avoid ctypes callback being garbage-collected.
_ggml_log_callback_ref = ggml_log_callback

llama_cpp_lib.llama_log_set(_ggml_log_callback_ref, ctypes.c_void_p(0))


def configure_logging(
    *,
    verbosity: VerbosityLike = None,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
    silent: Optional[bool] = None,
    show_output: Optional[bool] = None,
    log_filters: Optional[Iterable[str]] = None,
    append_log_filters: Optional[Iterable[str]] = None,
    log_filters_case_sensitive: Optional[bool] = None,
):
    """
    Configure native ggml/llama.cpp runtime logging.

    Priority:
        silent > quiet > verbosity > verbose > current config

    Compatibility:
        verbose=False -> ERROR
        verbose=True  -> DEBUG

    Numeric levels:
        0 = output
        1 = error
        2 = warn
        3 = info
        4 = trace
        5 = debug
    """
    if silent is True:
        v = LOG_LEVEL_ERROR
    elif quiet is True:
        v = LOG_LEVEL_WARN
    elif verbosity is not None:
        v = _normalize_verbosity(verbosity)
    elif verbose is not None:
        v = _normalize_verbosity(verbose)
    else:
        v = _config.verbosity

    _config.verbosity = v
    logger.setLevel(_verbosity_to_python_level(v))

    if show_output is not None:
        _config.show_output = show_output

    if log_filters is not None:
        _config.log_filters = [s for s in log_filters if s]

    if append_log_filters is not None:
        _config.log_filters.extend(s for s in append_log_filters if s)

    if log_filters_case_sensitive is not None:
        _config.log_filters_case_sensitive = log_filters_case_sensitive


def set_verbose(verbose: bool):
    """
    Backward-compatible bool API.

    False -> ERROR
    True  -> DEBUG
    """
    configure_logging(verbose=verbose)


def set_verbosity(verbosity: VerbosityLike):
    configure_logging(verbosity=verbosity)


def get_verbosity() -> int:
    return _config.verbosity


def set_quiet(quiet: bool = True):
    configure_logging(quiet=quiet)


def set_silent(silent: bool = True):
    configure_logging(silent=silent)


def set_log_filters(
    filters: Iterable[str],
    *,
    case_sensitive: bool = True,
):
    """
    Replace all substring log filters.

    Example:
        set_log_filters(["CUDA Graph id", "clip_model_loader: tensor"])
    """
    configure_logging(
        log_filters=filters,
        log_filters_case_sensitive=case_sensitive,
    )


def get_log_filters() -> list[str]:
    return list(_config.log_filters)


def add_log_filters(filters: Iterable[str]):
    """
    Append substring log filters.
    """
    configure_logging(append_log_filters=filters)


def clear_log_filters():
    """
    Clear all substring log filters, including default filters.
    """
    _config.log_filters.clear()


def reset_log_filters():
    """
    Restore default substring log filters.
    """
    _config.log_filters = list(DEFAULT_LOG_FILTERS)


def get_log_filters_case_sensitive() -> bool:
    return _config.log_filters_case_sensitive


def reset_logging():
    """
    Reset logging to default llama.cpp-style INFO verbosity and default filters.
    """
    _config.verbosity = LOG_DEFAULT_LLAMA
    _config.show_output = True
    _config.log_filters = list(DEFAULT_LOG_FILTERS)
    _config.log_filters_case_sensitive = True
    logger.setLevel(_verbosity_to_python_level(_config.verbosity))
