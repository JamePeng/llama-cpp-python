---
title: Llama Speculative Decoding
module_name: llama_cpp.llama_speculative
source_file: llama_cpp/llama_speculative.py
last_updated: 2026-05-23
version_target: "latest"
---

# Llama Speculative Decoding

## Overview

`llama_speculative.py` defines draft-model interfaces and prompt-based speculative decoding helpers for `llama-cpp-python`.

Speculative decoding lets a draft model propose candidate tokens before the main `Llama` model verifies them. In this module, the draft model does not have to be a neural network. It can also be a model-free prompt lookup decoder that predicts future tokens from repeated token patterns in the already verified context.

This module currently defines:

| Class | Status | Description |
|---|---|---|
| `LlamaDraftModel` | public interface | Abstract base class for speculative draft models. |
| `LlamaNGramMapDecoding` | public | Stateful model-free n-gram lookup decoder with `k` and `k4v` modes. |
| `LlamaPromptLookupDecoding` | legacy public | Stateless NumPy sliding-window prompt lookup decoder. |

## Role in the Library

This module defines the draft-model side of speculative decoding.

A draft model receives the verified token sequence so far and returns predicted draft token IDs. These tokens are later verified by the main `Llama` model during generation.

The module provides two prompt-based implementations:

- `LlamaNGramMapDecoding`: optimized, stateful, hash-map based n-gram lookup.
- `LlamaPromptLookupDecoding`: older stateless NumPy sliding-window lookup.

For new usage, prefer `LlamaNGramMapDecoding`. It incrementally maintains an n-gram index, supports memory-oriented lookup modes, and avoids scanning the full token history on every call.

## Choosing Between Related APIs

| API | Recommended Use | Notes |
|---|---|---|
| `LlamaNGramMapDecoding` | Default prompt lookup decoder for new usage. | Uses stateful n-gram maps and supports `k` / `k4v` modes. |
| `LlamaPromptLookupDecoding` | Compatibility with older prompt lookup behavior. | Stateless and simple, but scans token history with NumPy sliding windows. |

## Classes

## `LlamaDraftModel`

```python
class LlamaDraftModel(abc.ABC)
```

Abstract base class for speculative draft models.

A draft model must implement `__call__` and return an array of predicted token IDs.

### Method

```python
def __call__(
    self,
    input_ids: npt.NDArray[np.intc],
    /,
    **kwargs: Any,
) -> npt.NDArray[np.intc]
```

| Parameter | Type | Description |
|---|---|---|
| `input_ids` | `npt.NDArray[np.intc]` | Complete verified token sequence so far. |
| `**kwargs` | `Any` | Additional generation arguments. Implementations may ignore them. |

Returns:

| Type | Description |
|---|---|
| `npt.NDArray[np.intc]` | Draft token IDs proposed by the draft model. |

## `LlamaNGramMapDecoding`

```python
class LlamaNGramMapDecoding(LlamaDraftModel)
```

Fast model-free speculative decoder based on prompt n-gram lookup.

This decoder maintains internal indexes from historical n-grams to either previous positions or cached continuation tokens. When called with the current verified token sequence, it searches for the final n-gram in the already verified history and returns a continuation from the most recent valid historical match.

It does not own or run a separate draft model. Rejected draft tokens do not require manual rollback inside this class, because the next call receives the verified token history through `input_ids`.

### Constructor

```python
def __init__(
    self,
    ngram_size: int = 3,
    num_pred_tokens: int = 10,
    mode: Literal["k", "k4v"] = "k",
    min_hits: int = 2,
    max_entries_per_key: Optional[int] = None,
    sync_check_tokens: int = 16,
) -> None
```

| Parameter | Type | Default | Source | Description |
|---|---|---|---|---|
| `ngram_size` | `int` | `3` | `__init__` signature | Number of tokens used as the lookup key. Larger values require stricter matches and may reduce hit rate. |
| `num_pred_tokens` | `int` | `10` | `__init__` signature | Maximum number of draft tokens to return. |
| `mode` | `Literal["k", "k4v"]` | `"k"` | `__init__` signature | Lookup storage mode. `"k"` stores key-to-position mappings. `"k4v"` stores key-to-continuation mappings. |
| `min_hits` | `int` | `2` | `__init__` signature | Minimum number of historical matches required before returning a draft. Use `1` for maximum recall; use values greater than `1` to reduce low-confidence drafts. |
| `max_entries_per_key` | `Optional[int]` | `None` | `__init__` signature and initialization logic | Optional memory cap per n-gram key. If `mode="k4v"` and this is `None`, it is automatically set to `8`. |
| `sync_check_tokens` | `int` | `16` | `__init__` signature | Number of trailing tokens used to detect whether new input is an incremental append without doing a full prefix comparison. |

### Parameter Validation

The constructor raises `ValueError` when:

| Condition | Error Meaning |
|---|---|
| `ngram_size <= 0` | `ngram_size` must be positive. |
| `num_pred_tokens <= 0` | `num_pred_tokens` must be positive. |
| `min_hits <= 0` | `min_hits` must be positive. |
| `max_entries_per_key is not None and max_entries_per_key <= 0` | The memory cap must be `None` or positive. |
| `sync_check_tokens <= 0` | `sync_check_tokens` must be positive. |
| `mode` is not `"k"` or `"k4v"` after lowercasing | Only the two supported lookup modes are valid. |

### Lookup Modes

| Mode | Internal Storage | Memory Use | Behavior |
|---|---|---|---|
| `"k"` | `key -> [position, position, ...]` | Lower | Stores historical positions and slices continuations from `_history` during lookup. |
| `"k4v"` | `key -> {position: continuation}` | Higher | Stores continuation tokens directly and returns the latest cached continuation. |

Use `"k"` as the general-purpose default. Use `"k4v"` when faster continuation retrieval is preferred and the extra memory use is acceptable. For `"k4v"`, `max_entries_per_key` defaults to `8` when not specified.

### Important Attributes / State

| Attribute | Type | Source | Description |
|---|---|---|---|
| `ngram_size` | `int` | constructor | Number of tokens used as the n-gram lookup key. |
| `num_pred_tokens` | `int` | constructor | Maximum number of predicted draft tokens to return. |
| `mode` | `str` | constructor | Active lookup mode: `"k"` or `"k4v"`. |
| `min_hits` | `int` | constructor | Required number of historical matches before returning a draft. |
| `max_entries_per_key` | `Optional[int]` | constructor / initialization logic | Optional per-key memory cap. Automatically becomes `8` for `k4v` mode when not provided. |
| `sync_check_tokens` | `int` | constructor | Trailing-token window used for incremental append detection. |
| `_history` | `List[int]` | internal state | Verified token history mirrored from `input_ids`. |
| `_map_k` | `DefaultDict[Tuple[int, ...], List[int]]` | internal state | Key-to-position index used in `"k"` mode. |
| `_map_k4v` | `DefaultDict[Tuple[int, ...], Dict[int, Tuple[int, ...]]]` | internal state | Key-to-continuation index used in `"k4v"` mode. |
| `_closed` | `bool` | internal state | Marks the decoder as closed. Calling the decoder after `close()` raises `RuntimeError`. |
| `_last_draft_len` | `int` | internal state | Length of the most recent returned draft. Currently internal diagnostic state. |

Internal state should not be mutated directly.

### Core Methods

#### `__call__`

```python
def __call__(
    self,
    input_ids: npt.NDArray[np.intc],
    /,
    **kwargs: Any,
) -> npt.NDArray[np.intc]
```

Generates draft tokens from verified token history.

| Parameter | Type | Description |
|---|---|---|
| `input_ids` | `npt.NDArray[np.intc]` | Complete verified token sequence so far. |
| `**kwargs` | `Any` | Accepted for interface compatibility and ignored by this implementation. |

Returns:

| Type | Description |
|---|---|
| `npt.NDArray[np.intc]` | Predicted draft tokens. Returns an empty array when no reliable match is found. |

Raises:

| Exception | Condition |
|---|---|
| `RuntimeError` | The decoder has been closed with `close()` and is called again. |

#### `clear`

```python
def clear(self) -> None
```

Clears token history and internal indexes while keeping the decoder reusable.

Use this when starting a completely unrelated generation with the same decoder instance.

#### `close`

```python
def close(self) -> None
```

Clears internal containers and marks the decoder as closed.

This class does not own native memory, but explicit cleanup can be useful in long-running applications that may otherwise keep large Python containers alive.

#### `accept`

```python
def accept(self, n_accepted: int) -> None
```

Compatibility hook for speculative decoding loops.

This implementation is intentionally a no-op. Accepted tokens are reflected by the next `input_ids` passed to `__call__`, so no separate rollback or acceptance state update is required.

### Behavior

When called, `LlamaNGramMapDecoding`:

1. Converts `input_ids` to a flat `np.intc` token list.
2. Synchronizes internal history with the verified token sequence.
3. Uses a fast path when the new input is identical to the stored history.
4. Uses an incremental append path when the trailing tokens indicate that the new input extends the previous input.
5. Rebuilds the index after rollback, prompt switch, truncation, or unsafe mutation.
6. Indexes only n-grams with at least one available continuation token, so the current tail n-gram does not match itself.
7. Looks up the final `ngram_size` tokens as the search key.
8. Requires at least `min_hits` historical matches before returning a draft.
9. Returns up to `num_pred_tokens` tokens from the latest valid historical match.
10. Returns an empty NumPy array if no reliable match is available.

### Example: Direct Prompt Lookup

Use `min_hits=1` in a small standalone example so that one historical match is enough to return a draft.

```python
import numpy as np

from llama_cpp.llama_speculative import LlamaNGramMapDecoding

draft_model = LlamaNGramMapDecoding(
    ngram_size=3,
    num_pred_tokens=2,
    min_hits=1,
)

input_ids = np.array([1, 2, 3, 4, 5, 1, 2, 3], dtype=np.intc)
draft_tokens = draft_model(input_ids)

print(draft_tokens)
# Expected output:
# [4 5]
```

### Example: Use with `Llama`

```python
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaNGramMapDecoding

llm = Llama(
    model_path="path/to/model.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    draft_model=LlamaNGramMapDecoding(
        ngram_size=3,
        num_pred_tokens=10,
        mode="k",
        min_hits=2,
    ),
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": (
                "Write five short Python classes with the same CRUD method layout: "
                "User, Product, Order, Review, and Category."
            ),
        }
    ]
)

print(response["choices"][0]["message"]["content"])
```

### Example: Use `k4v` Mode with a Memory Cap

```python
from llama_cpp.llama_speculative import LlamaNGramMapDecoding

draft_model = LlamaNGramMapDecoding(
    ngram_size=4,
    num_pred_tokens=8,
    mode="k4v",
    min_hits=2,
    max_entries_per_key=8,
)
```

## `LlamaPromptLookupDecoding`

```python
class LlamaPromptLookupDecoding(LlamaDraftModel)
```

Legacy speculative decoder based on NumPy sliding-window lookup.

This implementation is stateless. Each call scans the input token sequence to find previous occurrences of the current n-gram and returns the following tokens as draft predictions.

> Warning: This implementation is not recommended for production. It may have high computational overhead for long contexts and may degrade output quality. Prefer `LlamaNGramMapDecoding` for new usage.

### Constructor

```python
def __init__(
    self,
    max_ngram_size: int = 3,
    num_pred_tokens: int = 10,
)
```

| Parameter | Type | Default | Source | Description |
|---|---|---|---|---|
| `max_ngram_size` | `int` | `3` | `__init__` signature | Maximum n-gram size to search for. The decoder tries larger n-grams first. |
| `num_pred_tokens` | `int` | `10` | `__init__` signature | Maximum number of draft tokens to return. |

### Important Attributes / State

| Attribute | Type | Source | Description |
|---|---|---|---|
| `max_ngram_size` | `int` | constructor | Maximum n-gram window size used during lookup. |
| `num_pred_tokens` | `int` | constructor | Maximum number of predicted draft tokens to return. |

### Static Method

```python
@staticmethod
def find_candidate_pred_tokens(
    input_ids: npt.NDArray[np.intc],
    max_ngram_size: int,
    num_pred_tokens: int,
)
```

Linearly scans `input_ids` using NumPy sliding windows to find matching n-grams.

| Parameter | Type | Description |
|---|---|---|
| `input_ids` | `npt.NDArray[np.intc]` | Complete token sequence. |
| `max_ngram_size` | `int` | Maximum n-gram size to search for. |
| `num_pred_tokens` | `int` | Maximum number of draft tokens to return. |

Returns:

| Type | Description |
|---|---|
| `npt.NDArray[np.intc]` | Candidate draft tokens, or an empty array if no match is found. |

### Method

```python
def __call__(
    self,
    input_ids: npt.NDArray[np.intc],
    /,
    **kwargs: Any,
) -> npt.NDArray[np.intc]
```

Calls `find_candidate_pred_tokens` with the instance's `max_ngram_size` and `num_pred_tokens`.

## Best Practices & Common Patterns

- Prefer `LlamaNGramMapDecoding` for new usage.
- Use `mode="k"` as the default memory-efficient mode.
- Use `mode="k4v"` when cached continuations are useful and the additional memory use is acceptable.
- Keep `max_entries_per_key` set for `k4v` mode unless you intentionally want an unbounded per-key cache.
- Use `min_hits=1` for maximum recall in repetitive prompts or benchmarks.
- Use `min_hits > 1` to reduce low-confidence drafts.
- Increase `ngram_size` for stricter pattern matching.
- Increase `num_pred_tokens` to allow longer draft proposals, but remember that the target model still verifies the tokens.
- Call `clear()` before reusing the same decoder for an unrelated prompt or generation session.
- Do not call the decoder again after `close()` unless you create a new instance.
- Do not mutate `_history`, `_map_k`, `_map_k4v`, or other internal state directly.

## Limitations

- Prompt lookup only predicts tokens that are already implied by repeated patterns in the verified context.
- It is most useful for repetitive, structured, or boilerplate-heavy output.
- It may return an empty draft when the context has too few repeated n-grams or when `min_hits` is too strict.
- It does not replace target-model verification.
- `LlamaPromptLookupDecoding` is kept for compatibility and is not recommended for production use.

## Deprecated / Changed APIs

`LlamaPromptLookupDecoding` is the legacy NumPy sliding-window implementation. It remains available, but `LlamaNGramMapDecoding` is the preferred prompt lookup implementation for new code.

Compared with the older `LlamaNGramMapDecoding` documentation, the current implementation adds:

- `mode`
- `min_hits`
- `max_entries_per_key`
- `sync_check_tokens`
- `clear()`
- `close()`
- `accept()`
- Separate internal indexes for `k` and `k4v` modes

## Related Links

* [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
* [[Benchmark_Speculative](https://github.com/JamePeng/llama-cpp-python/blob/main/examples/benchmark/benchmark_speculative.py)]

