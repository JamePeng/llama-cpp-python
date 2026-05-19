---
title: Logger
class_name: Logger (module)
module_name: llama_cpp._logger
source_file: llama_cpp/_logger.py
last_updated: 2026-05-16
version_target: latest
---

## Overview

The `Logger` module provides configuration for runtime logging in `llama-cpp-python`, wrapping the native `ggml`/`llama.cpp` logging infrastructure. It controls verbosity levels, output streams, substring filtering, and callback integration, allowing fine-grained control over diagnostic and informational output from the underlying bindings.

## Role in the Library

- **Wraps low-level logging**: It intercepts and transforms log events from the C/C++ backend (`ggml_log_callback`).
- **Connects to Python logging**: Maps `ggml` verbosity levels (0–5) to `logging` levels (ERROR, WARNING, INFO, DEBUG), and routes output to `stdout`/`stderr` based on severity.
- **Provides filtering**: Substring-based message filtering to suppress specific log categories (e.g., CUDA Graph output).
- **Extends the API surface**: Offers both explicit configuration functions and convenient shorthand setters (`set_verbose`, `set_quiet`), while preserving full control through `configure_logging`.

## Core Methods

### `configure_logging(*, verbosity=None, verbose=None, quiet=None, silent=None, show_output=None, log_filters=None, append_log_filters=None, log_filters_case_sensitive=None)`

The primary configuration function. Combines multiple parameters into a unified verbosity level.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbosity` | int \| bool \| None | None | Numeric level (0–5). `False` maps to `ERROR` (1), `True` to `DEBUG` (5). |
| `verbose` | bool | None | Shorthand: `True` → `DEBUG`, `False` → `ERROR`. |
| `quiet` | bool | None | Shorthand: `True` → `WARN` (2). |
| `silent` | bool | None | Shorthand: `True` → `ERROR` (1). |
| `show_output` | bool | None | Whether `GGML_LOG_LEVEL_NONE` (output) should be shown. |
| `log_filters` | Iterable[str] | None | List of substring patterns to filter out. |
| `append_log_filters` | Iterable[str] | None | Append additional filter patterns. |
| `log_filters_case_sensitive` | bool | None | Whether filters are case-sensitive. |

### `set_verbose(verbose: bool)`

Shorthand setter. `verbose=True` sets `verbosity=DEBUG`, `verbose=False` sets `verbosity=ERROR`.

### `set_verbosity(verbosity: VerbosityLike)`

Sets verbosity to any value accepted by `configure_logging`.

### `get_verbosity() -> int`

Returns current configured verbosity level (0–5).

### `set_quiet(quiet: bool = True)`

Sets `verbosity=WARN` (`2`).

### `set_silent(silent: bool = True)`

Sets `verbosity=ERROR` (`1`).

### `set_log_filters(filters: Iterable[str], *, case_sensitive: bool = True)`

Replaces all substring log filters.

### `get_log_filters() -> list[str]`

Returns current filter list.

### `add_log_filters(filters: Iterable[str])`

Appends filters to the current list.

### `clear_log_filters()`

Removes all user-defined filters.

### `reset_log_filters()`

Restores the default filter list: `["CUDA Graph", "CUDA graph"]`.

### `reset_logging()`

Resets to default: `verbosity=INFO` (`3`), `show_output=True`, default filters.

## Important Attributes / State

| Attribute | Type | Source | Description |
|-----------|------|--------|-------------|
| `_config` | LoggerConfig | Internal | Holds the current configuration: verbosity, output streams, filters. |
| `_last_verbosity` | int | Internal | Tracks the last verbosity level set by `ggml_log_callback`. |

## Best Practices & Common Patterns

### 1. Default Behavior
Use `reset_logging()` to start with `INFO` verbosity, which shows warnings and errors but hides internal debug output.

```python
from llama_cpp import Llama
from llama_cpp import reset_logging

reset_logging()  # Default verbosity=3 (INFO), show warnings and errors
llm = Llama(model_path="models/qwen3.gguf")
llm("Explain quantum physics.")
```

### 2. Precise Logging via `verbosity`
Replace the legacy `verbose` boolean with the precise `verbosity` parameter. `verbose=False` maps to `ERROR` (1), `verbose=True` to `DEBUG` (5).

```python
from llama_cpp import Llama

# Legacy (coarse control):
llm_quiet = Llama(model_path="models/qwen3.gguf", verbose=False)
llm_quiet("What is a neural network?")

# Modern (fine-grained control):
llm = Llama(model_path="models/qwen3.gguf", verbosity=3)
llm("What is a neural network?")
```

### 3. Low-Level Debugging
For deep backend debugging, set `verbosity=5` (DEBUG) and optionally disable substring filters to see all diagnostic output.

```python
from llama_cpp import Llama

# Debug-level logs, showing all backend diagnostics
llm = Llama(model_path="models/qwen3.gguf", verbosity=5)

# If you want to see normally filtered CUDA Graph messages:
llm = Llama(
    model_path="models/qwen3.gguf",
    verbosity=5,
    log_filters=[],  # Disable all substring filters
)
```

### 4. Substring-Based Backend Noise Filtering
Suppress known noisy backend messages by passing substring filters. This prevents "CUDA Graph" and model loading chatter from flooding the console.

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/qwen3.gguf",
    verbosity=3,  # INFO level
    log_filters=[
        "CUDA Graph id",
        "clip_model_loader: tensor",
        "ggml_cuda_graph_update_required",
        "llama_perf_context_print",
    ],
)
llm("What is a transformer?")
```

### 5. Runtime Logging Adjustments
Since logging is process-global, you can adjust verbosity or filters at runtime — changes apply to all `Llama` instances in the same process.

```python
from llama_cpp import Llama

llm = Llama(model_path="models/qwen3.gguf", verbosity=2)  # QUIET: only show warnings and errors
llm("Quick answer: What is machine learning?")

# Temporarily increase verbosity for diagnostics
llm.set_verbosity(5)
llm("Show me the full debug log for this prompt")
llm.set_verbosity(2)  # Return to QUIET

# Add a specific filter without resetting everything
llm.add_log_filters(["llama_perf_context_print"])
llm("Final answer: What is machine learning?")
```

### 6. Complete Diagnostic Session
For a full diagnostic session, combine precise verbosity, custom filters, and runtime control:

```python
from llama_cpp import Llama

# 1. Start with info-level verbosity
llm = Llama(model_path="models/qwen3.gguf", verbosity=3)

# 2. Suppress backend noise
llm.set_log_filters([
    "CUDA Graph",
    "CUDA graph",
    "clip_model_loader: tensor",
    "ggml_cuda_graph_update_required",
])

# 3. Run inference
llm("Explain the llama.cpp inference pipeline")

# 4. Temporarily increase verbosity for a specific call
llm.set_verbosity(5)
llm("Show debug output for cache hit details")
llm.set_verbosity(2)  # Return to normal

# 5. Remove filters after session
llm.clear_log_filters()
```

## Key Considerations

- **Process-global**: Logging configuration affects all `Llama` instances in the same process. Use `add_log_filters` or `set_log_filters` carefully when multiple instances run concurrently.
- **Flushed immediately**: Every log call flushes to `stdout`/`stderr`, so output appears immediately.
- **Shorthand vs. precise**: Prefer `verbosity`/`set_verbosity` over `verbose`/`set_verbose`/`set_quiet`/`set_silent` for precision, though the shorthands remain for backward compatibility.
- **verbose=False** vs. **verbosity=0**: These have distinct behaviors — `verbose=False` silences Python wrapper prints but not backend diagnostics; `verbosity=0` silences all backend non-error output.

## Deprecated / Changed APIs

None documented.

## Related Links

* [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
