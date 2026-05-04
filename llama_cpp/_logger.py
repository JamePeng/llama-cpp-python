import sys
import ctypes
import logging
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
GGML_LOG_LEVEL_TO_LOGGING_LEVEL = {
    0: logging.CRITICAL,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.DEBUG,
    5: logging.DEBUG,
}

logger = logging.getLogger("llama-cpp-python")

_last_log_level = GGML_LOG_LEVEL_TO_LOGGING_LEVEL[0]

# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
@_ggml.ggml_log_callback
def ggml_log_callback(
    level: int,
    text: bytes,
    user_data: ctypes.c_void_p,
):
    # Note(JamePeng): A temporary patch is used to filter out garbage debug information
    # output from the underlying C++ `CUDA Graph id %zu reused`.
    # The logger is planned to be refactored to meet control requirements.
    if text:
        if b"CUDA Graph" in text or b"CUDA graph" in text:
            return
    # TODO: Correctly implement continue previous log
    global _last_log_level
    log_level = GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level] if level != 5 else _last_log_level
    if logger.level <= GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level]:
        print(text.decode("utf-8"), end="", flush=True, file=sys.stderr)
    _last_log_level = log_level


llama_cpp_lib.llama_log_set(ggml_log_callback, ctypes.c_void_p(0))


def set_verbose(verbose: bool):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
