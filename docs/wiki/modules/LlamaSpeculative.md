---
title: Llama Speculative Decoding
module_name: llama_cpp.llama_speculative
source_file: llama_cpp/llama_speculative.py
last_updated: 2026-05-02
version_target: "latest"
---

# Llama Speculative Decoding

## Overview

`llama_speculative.py` provides draft model interfaces and prompt-based speculative decoding helpers for `llama-cpp-python`.

Speculative decoding uses a lightweight draft model to propose candidate tokens before the main model verifies them. In this module, the draft model does not need to be a neural model. It can also be a prompt lookup decoder that predicts future tokens by finding repeated token patterns in the existing context.

This module currently defines:

| Class | Status | Description |
|---|---|---|
| `LlamaDraftModel` | public interface | Abstract base class for draft models used by speculative decoding. |
| `LlamaNGramMapDecoding` | public | Fast stateful n-gram map based speculative decoder. |
| `LlamaPromptLookupDecoding` | legacy public | Stateless NumPy sliding-window prompt lookup decoder. |

## Role in the Library

This module defines the draft-model side of speculative decoding.

A draft model receives the current token sequence and returns predicted draft tokens. These draft tokens can then be verified by the main `Llama` model during generation.

The module provides two prompt-based implementations:

- `LlamaNGramMapDecoding`: optimized, stateful, hash-map based lookup.
- `LlamaPromptLookupDecoding`: older stateless NumPy sliding-window implementation.

For new usage, prefer `LlamaNGramMapDecoding` because it incrementally maintains an n-gram index instead of scanning the full token history on every call.

## Classes

## `LlamaDraftModel`

```python
class LlamaDraftModel(abc.ABC)
````

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

| Parameter   | Type                   | Description                                                       |
| ----------- | ---------------------- | ----------------------------------------------------------------- |
| `input_ids` | `npt.NDArray[np.intc]` | Current token sequence.                                           |
| `**kwargs`  | `Any`                  | Additional generation arguments. Implementations may ignore them. |

Returns:

| Type                   | Description                                  |
| ---------------------- | -------------------------------------------- |
| `npt.NDArray[np.intc]` | Draft token IDs proposed by the draft model. |

## `LlamaNGramMapDecoding`

```python
class LlamaNGramMapDecoding(LlamaDraftModel)
```

Fast speculative decoder based on an n-gram hash map.

This decoder maintains an internal inverted index from historical n-grams to their positions. When called with the current token sequence, it looks up the final n-gram in the history and returns the following tokens from the most recent matching context.

### Constructor

```python
def __init__(
    self,
    ngram_size: int = 3,
    num_pred_tokens: int = 10,
)
```

| Parameter         | Type  | Default | Description                                                                                                                     |
| ----------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `ngram_size`      | `int` | `3`     | Length of the token sequence used as the lookup key. Larger values require stricter context matches but may produce fewer hits. |
| `num_pred_tokens` | `int` | `10`    | Maximum number of draft tokens to return after a matching n-gram is found.                                                      |

### Important Attributes / State

| Attribute         | Type                               | Source         | Description                                                                      |
| ----------------- | ---------------------------------- | -------------- | -------------------------------------------------------------------------------- |
| `ngram_size`      | `int`                              | constructor    | Number of tokens used as the n-gram lookup key.                                  |
| `num_pred_tokens` | `int`                              | constructor    | Maximum number of predicted draft tokens to return.                              |
| `_ngram_map`      | `Dict[Tuple[int, ...], List[int]]` | internal cache | Internal inverted index mapping n-gram tuples to positions in the token history. |
| `_history`        | `List[int]`                        | internal cache | Internal token history used to maintain the n-gram map.                          |

`_ngram_map` and `_history` are internal state and should not be modified directly.

### Behavior

When called, `LlamaNGramMapDecoding`:

1. Synchronizes its internal history with the provided `input_ids`.
2. Incrementally updates the n-gram map when tokens are appended.
3. Rebuilds the map if the input sequence is no longer a simple continuation, such as after rollback or a new prompt.
4. Uses the last `ngram_size` tokens as the search key.
5. Returns up to `num_pred_tokens` tokens following the most recent historical match.
6. Returns an empty NumPy array if no match is found.

### Example

```python
import numpy as np
from llama_cpp.llama_speculative import LlamaNGramMapDecoding

draft_model = LlamaNGramMapDecoding(
    ngram_size=3,
    num_pred_tokens=5,
)

input_ids = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.intc)

draft_tokens = draft_model(input_ids)

print(draft_tokens)
```

## `LlamaPromptLookupDecoding`

```python
class LlamaPromptLookupDecoding(LlamaDraftModel)
```

Legacy speculative decoder based on NumPy sliding-window lookup.

This implementation is stateless. Each call scans the input token sequence to find previous occurrences of the current n-gram and returns the following tokens as draft predictions.

> Warning: This implementation may have high computational overhead for long contexts. Prefer `LlamaNGramMapDecoding` for new usage.

### Constructor

```python
def __init__(
    self,
    max_ngram_size: int = 3,
    num_pred_tokens: int = 10,
)
```

| Parameter         | Type  | Default | Description                                                                |
| ----------------- | ----- | ------- | -------------------------------------------------------------------------- |
| `max_ngram_size`  | `int` | `3`     | Maximum n-gram size to search for. The decoder tries larger n-grams first. |
| `num_pred_tokens` | `int` | `10`    | Maximum number of draft tokens to return.                                  |

### Important Attributes / State

| Attribute         | Type  | Source      | Description                                         |
| ----------------- | ----- | ----------- | --------------------------------------------------- |
| `max_ngram_size`  | `int` | constructor | Maximum n-gram window size used during lookup.      |
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

| Parameter         | Type                   | Description                               |
| ----------------- | ---------------------- | ----------------------------------------- |
| `input_ids`       | `npt.NDArray[np.intc]` | Complete token sequence.                  |
| `max_ngram_size`  | `int`                  | Maximum n-gram size to search for.        |
| `num_pred_tokens` | `int`                  | Maximum number of draft tokens to return. |

Returns:

| Type                   | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `npt.NDArray[np.intc]` | Candidate draft tokens, or an empty array if no match is found. |

### Example

```python
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaNGramMapDecoding

llama = Llama(
    model_path="path/to/qwen-3.6-27b.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    draft_model=LlamaNGramMapDecoding(
        ngram_size=3,
        num_pred_tokens=10
    )
)

response = llama.create_chat_completion(
    messages=[{"role": "user", "content": """
    Write a Python script using `sqlite3` to define CRUD (Create, Read, Update, Delete) operations for an e-commerce database. 
You need to create 5 separate classes for the following entities: `User`, `Product`, `Order`, `Review`, and `Category`. 
Each class MUST have exactly the same internal structure and method names (create, get, update, delete). Do not add extra logic, just the standard boilerplate.
    """}]
)
```

## Best Practices & Common Patterns

* Prefer `LlamaNGramMapDecoding` for new usage.
* Use `LlamaPromptLookupDecoding` only when compatibility with the older stateless prompt lookup behavior is needed.
* Increase `ngram_size` or `max_ngram_size` for stricter context matching.
* Increase `num_pred_tokens` when you want longer draft proposals, but keep in mind that speculative decoding still depends on later verification by the main model.
* Do not mutate `_ngram_map` or `_history` directly.
* If input token history rolls back or changes unexpectedly, `LlamaNGramMapDecoding` automatically rebuilds its internal cache.

## Deprecated / Changed APIs

`LlamaPromptLookupDecoding` is marked as a legacy NumPy sliding-window implementation in the source code. It is still available, but `LlamaNGramMapDecoding` is the preferred implementation for faster repeated calls over long contexts.

## Related Links

* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]

