---
title: Llama Cache
module_name: llama_cpp.llama_cache
source_file: llama_cpp/llama_cache.py
last_updated: 2026-05-02
version_target: "latest"
---

# Llama Cache

## Overview

`llama_cpp.llama_cache` provides cache implementations for storing and restoring `LlamaState` objects or recurrent model state checkpoints.

The module is mainly used to speed up repeated inference workflows by reusing previously computed model state for matching token prefixes.

It defines several cache classes:

| Class | Purpose |
|---|---|
| `BaseLlamaCache` | Abstract base class for llama.cpp state caches. |
| `LlamaRAMCache` | In-memory LRU cache for `LlamaState` objects. |
| `LlamaDiskCache` | Disk-backed cache using the `diskcache` library. |
| `LlamaTrieCache` | Trie-based cache optimized for fast longest-prefix lookup. |
| `HybridCheckpointCache` | Checkpoint manager for RNN/Hybrid model hidden states. |
| `HybridCheckpoint` | Dataclass representing one saved hybrid model checkpoint. |
| `TrieNode` | Internal trie node used by `LlamaTrieCache`. |

The public compatibility alias is:

```python
LlamaCache = LlamaTrieCache
````

This means that code importing `LlamaCache` receives the trie-based cache implementation.

Defined in: `llama_cpp/llama_cache.py`

Related pages: [[Llama]], [[Caching]], [[State Save Load]], [[Hybrid Models]]

---

## Role in the API

The cache module provides reusable storage for model runtime state.

There are two main caching strategies:

1. **Token-prefix state caching**

   Used by:

   * `LlamaRAMCache`
   * `LlamaDiskCache`
   * `LlamaTrieCache`
   * `LlamaCache`

   These caches map token sequences to `llama_core.LlamaState` objects. When queried, they do not require an exact match. Instead, they return the state associated with the longest cached token prefix.

2. **Hybrid / recurrent checkpoint caching**

   Used by:

   * `HybridCheckpoint`
   * `HybridCheckpointCache`

   This is designed for Hybrid or recurrent models where rollback requires saving and restoring hidden state snapshots through low-level llama.cpp state APIs.

---

## Public API Summary

| API                     | Type           |   Public | Description                                     |
| ----------------------- | -------------- | -------: | ----------------------------------------------- |
| `BaseLlamaCache`        | Abstract class |      Yes | Base interface for cache implementations.       |
| `LlamaRAMCache`         | Class          |      Yes | In-memory LRU cache with linear prefix lookup.  |
| `LlamaDiskCache`        | Class          |      Yes | Disk-backed cache using `diskcache.Cache`.      |
| `LlamaTrieCache`        | Class          |      Yes | Trie-based cache with efficient prefix lookup.  |
| `LlamaCache`            | Alias          |      Yes | Backward-compatible alias for `LlamaTrieCache`. |
| `HybridCheckpoint`      | Dataclass      |      Yes | Represents one saved Hybrid/RNN checkpoint.     |
| `HybridCheckpointCache` | Class          |      Yes | Manages Hybrid/RNN state checkpoints.           |
| `TrieNode`              | Class          | Internal | Trie node used by `LlamaTrieCache`.             |

---

# `BaseLlamaCache`

## Overview

`BaseLlamaCache` is the abstract base class for llama.cpp cache implementations.

It defines a common dictionary-like interface for storing and retrieving `llama_core.LlamaState` objects by token sequence.

Subclasses are expected to implement:

* `cache_size`
* `__getitem__`
* `__contains__`
* `__setitem__`

Defined in: `llama_cpp/llama_cache.py`

---

## Role in the API

`BaseLlamaCache` acts as the shared contract for cache implementations used by higher-level llama-cpp-python runtime code.

It is not intended to be used directly. Users should instantiate one of the concrete cache classes instead:

* `LlamaRAMCache`
* `LlamaDiskCache`
* `LlamaTrieCache`
* `LlamaCache`

---

## Constructor: `__init__`

```python
def __init__(self, capacity_bytes: int = (2 << 30)):
    ...
```

| Parameter        | Type  |   Default | Required | Description                                                          |
| ---------------- | ----- | --------: | -------: | -------------------------------------------------------------------- |
| `capacity_bytes` | `int` | `2 << 30` |       No | Maximum cache capacity in bytes. The default is approximately 2 GiB. |

---

## Instance Variables

| Name             | Type  | Description                                                                                                  |
| ---------------- | ----- | ------------------------------------------------------------------------------------------------------------ |
| `capacity_bytes` | `int` | Maximum allowed cache size in bytes. Concrete subclasses use this value to decide when eviction is required. |

---

## Properties

### `cache_size`

```python
@property
@abstractmethod
def cache_size(self) -> int:
    ...
```

Returns the current cache size in bytes.

Concrete implementations define how this value is calculated.

---

## Core Methods

### `_find_longest_prefix_key`

```python
def _find_longest_prefix_key(
    self,
    key: Tuple[int, ...],
) -> Optional[Tuple[int, ...]]:
    ...
```

Finds the cached key with the longest token prefix matching the requested key.

In `BaseLlamaCache`, this method is only a placeholder and does not implement behavior.

Concrete subclasses may override it.

---

### `__getitem__`

```python
@abstractmethod
def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
    ...
```

Retrieves a cached `LlamaState`.

The expected behavior is longest-prefix matching rather than strict exact-key lookup.

---

### `__contains__`

```python
@abstractmethod
def __contains__(self, key: Sequence[int]) -> bool:
    ...
```

Returns whether the cache contains a matching token prefix for the given key.

---

### `__setitem__`

```python
@abstractmethod
def __setitem__(
    self,
    key: Sequence[int],
    value: "llama_core.LlamaState"
) -> None:
    ...
```

Stores a `LlamaState` under a token sequence.

---

# `LlamaRAMCache`

## Overview

`LlamaRAMCache` is an in-memory cache for `llama_core.LlamaState` objects.

It stores token sequences in an `OrderedDict` and maintains an LRU eviction policy. Lookup is based on the longest cached token prefix.

Defined in: `llama_cpp/llama_cache.py`

---

## Role in the API

`LlamaRAMCache` is useful when users want fast in-process caching without writing state to disk.

It keeps all cached states in Python memory. This makes retrieval simple, but memory usage can grow quickly depending on the size of saved `LlamaState` objects.

---

## Constructor: `__init__`

```python
def __init__(self, capacity_bytes: int = (2 << 30), verbose: bool = False):
    ...
```

| Parameter        | Type   |   Default | Required | Description                                                                                                                   |
| ---------------- | ------ | --------: | -------: | ----------------------------------------------------------------------------------------------------------------------------- |
| `capacity_bytes` | `int`  | `2 << 30` |       No | Maximum total size of cached states in bytes.                                                                                 |
| `verbose`        | `bool` |   `False` |       No | Whether to enable verbose behavior when computing token-prefix matches. This value is passed to `Llama.longest_token_prefix`. |

---

## Instance Variables

| Name             | Type                                                  | Description                                                                                                     |
| ---------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `capacity_bytes` | `int`                                                 | Maximum cache capacity in bytes.                                                                                |
| `cache_state`    | `OrderedDict[Tuple[int, ...], llama_core.LlamaState]` | Stores cached token sequences and their corresponding `LlamaState` objects. The order is used for LRU eviction. |
| `_current_size`  | `int`                                                 | Current total size of cached states in bytes.                                                                   |
| `verbose`        | `bool`                                                | Passed to `llama_core.Llama.longest_token_prefix` during prefix comparison.                                     |

---

## Properties

### `cache_size`

```python
@property
def cache_size(self):
    return self._current_size
```

Returns the current tracked memory usage of the cache in bytes.

---

## Core Methods

### `_find_longest_prefix_key`

```python
def _find_longest_prefix_key(
    self,
    key: Tuple[int, ...],
) -> Optional[Tuple[int, ...]]:
    ...
```

Finds the cached token sequence with the longest prefix match against `key`.

This implementation scans every key in `cache_state` and calls:

```python
llama_core.Llama.longest_token_prefix(k, key, self.verbose)
```

### Complexity

| Operation     | Complexity |
| ------------- | ---------: |
| Prefix lookup | `O(N * K)` |
| LRU update    |     `O(1)` |
| Size tracking |     `O(1)` |

Where:

* `N` is the number of cached entries.
* `K` is the token sequence length.

---

### `__getitem__`

```python
def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
    ...
```

Returns the cached `LlamaState` for the longest matching token prefix.

Behavior:

1. Raises `KeyError("Cache is empty")` if the cache has no entries.
2. Converts the input key to a tuple.
3. Finds the longest cached prefix.
4. Raises `KeyError("Key not found")` if no matching prefix exists.
5. Moves the matched key to the end of `cache_state` to mark it as recently used.
6. Returns the matched `LlamaState`.

---

### `__contains__`

```python
def __contains__(self, key: Sequence[int]) -> bool:
    ...
```

Returns `True` if any cached key is a prefix match for the requested token sequence.

Returns `False` if the cache is empty.

---

### `__setitem__`

```python
def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
    ...
```

Stores a `LlamaState` in memory.

Behavior:

1. Converts `key` to a tuple.
2. If the key already exists, deletes the old entry.
3. Inserts the new `LlamaState`.
4. Adds `value.llama_state_size` to `_current_size`.
5. Evicts least-recently-used entries while `_current_size > capacity_bytes`.
6. Resets `_current_size` to `0` if the cache becomes empty.

> Note: The current implementation increments `_current_size` by the new value size when replacing an existing key, but it does not subtract the old value size before deletion. This may cause size tracking to overcount replaced entries.

---

## Example

```python
from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaRAMCache

llm = Llama(
    model_path="./models/model.gguf",
    cache=LlamaRAMCache(capacity_bytes=1 << 30),
)

response = llm("Q: What is llama.cpp?\nA:", max_tokens=64)

print(response["choices"][0]["text"])
```

---

## Best Practices

* Use `LlamaRAMCache` when cache speed is more important than persistence.
* Keep `capacity_bytes` below available system memory.
* Reuse the same cache instance across repeated prompts when prefix reuse is expected.
* Prefer `LlamaTrieCache` or `LlamaCache` when many cached entries are expected and prefix lookup cost matters.

---

# `LlamaDiskCache`

## Overview

`LlamaDiskCache` is a disk-backed cache for `llama_core.LlamaState` objects.

It delegates storage, size limits, and LRU-style eviction behavior to the external `diskcache` library.

Defined in: `llama_cpp/llama_cache.py`

---

## Role in the API

`LlamaDiskCache` is useful when cached model states should persist beyond the current Python process or when RAM usage should be limited.

Compared with `LlamaRAMCache`, it may reduce memory pressure but can be slower due to disk I/O.

---

## Constructor: `__init__`

```python
def __init__(
    self,
    cache_dir: str = ".cache/llama_cache",
    capacity_bytes: int = (2 << 30),
    verbose: bool = False
):
    ...
```

| Parameter        | Type   |                Default | Required | Description                                                                                    |
| ---------------- | ------ | ---------------------: | -------: | ---------------------------------------------------------------------------------------------- |
| `cache_dir`      | `str`  | `".cache/llama_cache"` |       No | Directory used by `diskcache.Cache` to store cached state data.                                |
| `capacity_bytes` | `int`  |              `2 << 30` |       No | Maximum disk cache size in bytes. Passed to `diskcache.Cache(..., size_limit=capacity_bytes)`. |
| `verbose`        | `bool` |                `False` |       No | Passed to `Llama.longest_token_prefix` when searching for the best prefix match.               |

---

## Instance Variables

| Name             | Type              | Description                                                                  |
| ---------------- | ----------------- | ---------------------------------------------------------------------------- |
| `cache_dir`      | `str`             | Filesystem directory for the disk cache.                                     |
| `cache`          | `diskcache.Cache` | SQLite-backed disk cache object.                                             |
| `verbose`        | `bool`            | Passed to token-prefix comparison logic.                                     |
| `capacity_bytes` | `int`             | Maximum configured cache capacity in bytes, inherited from `BaseLlamaCache`. |

---

## Properties

### `cache_size`

```python
@property
def cache_size(self):
    return self.cache.volume()
```

Returns the current disk cache volume in bytes using `diskcache.Cache.volume()`.

---

## Core Methods

### `_find_longest_prefix_key`

```python
def _find_longest_prefix_key(
    self,
    key: Tuple[int, ...],
) -> Optional[Tuple[int, ...]]:
    ...
```

Finds the cached key with the longest token-prefix match.

Behavior:

1. Returns `None` immediately if the disk cache is empty.
2. Iterates over `self.cache.iterkeys()`.
3. Uses `llama_core.Llama.longest_token_prefix(k, key, self.verbose)` to compare each cached key.
4. Stops early if a perfect match is found.

### Complexity

| Operation              |                            Complexity |
| ---------------------- | ------------------------------------: |
| Prefix lookup          |                            `O(N * K)` |
| Disk iteration         | Depends on `diskcache` and filesystem |
| Exact-match early exit |                             Supported |

---

### `__getitem__`

```python
def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
    ...
```

Retrieves the cached state associated with the longest matching prefix.

Behavior:

1. Prints `"LlamaDiskCache.__getitem__: called"` to `stderr`.
2. Raises `KeyError("Cache is empty")` if no entries exist.
3. Converts `key` to a tuple.
4. Finds the longest prefix key.
5. Raises `KeyError("Key not found")` if no match exists.
6. Reads and returns the cached `LlamaState`.

The implementation notes that this read is non-destructive and automatically updates access time for LRU behavior through `diskcache`.

---

### `__contains__`

```python
def __contains__(self, key: Sequence[int]) -> bool:
    ...
```

Returns whether the cache has any longest-prefix match for the given token sequence.

---

### `__setitem__`

```python
def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
    ...
```

Stores a `LlamaState` in the disk cache.

Behavior:

1. Prints `"LlamaDiskCache.__setitem__: called"` to `stderr`.
2. Converts `key` to a tuple.
3. Assigns the value to `self.cache[tuple(key)]`.

`diskcache` handles capacity checks and eviction.

---

## Example

```python
from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaDiskCache

cache = LlamaDiskCache(
    cache_dir=".cache/llama_cache",
    capacity_bytes=2 << 30,
)

llm = Llama(
    model_path="./models/model.gguf",
    cache=cache,
)

response = llm("Q: What is llama.cpp?\nA:", max_tokens=64)

print(response["choices"][0]["text"])
```

---

## Best Practices

* Use `LlamaDiskCache` when cache persistence is useful.
* Place `cache_dir` on a fast local SSD when possible.
* Avoid using slow network filesystems for high-throughput inference.
* Consider `LlamaTrieCache` for workloads where many prefix lookups happen within a single process.

---

## Common Pitfalls

* Disk-backed caching can be slower than RAM caching.
* The cache depends on the third-party `diskcache` package.
* Prefix lookup still scans cached keys linearly, even though storage and eviction are handled by `diskcache`.
* The implementation prints debug messages to `stderr` on get and set operations.

---

# `TrieNode`

## Overview

`TrieNode` is an internal helper class used by `LlamaTrieCache`.

Each node represents one position in a token-prefix tree.

Defined in: `llama_cpp/llama_cache.py`

---

## Role in the API

`TrieNode` is not intended to be used directly by users.

It stores:

* Child nodes keyed by token ID.
* An optional `LlamaState` when the node marks the end of a cached token sequence.

---

## Constructor: `__init__`

```python
def __init__(self):
    ...
```

The constructor takes no parameters.

---

## Instance Variables

| Name       | Type                              | Description                                                                               |
| ---------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| `children` | `Dict[int, TrieNode]`             | Child trie nodes keyed by token ID.                                                       |
| `state`    | `Optional[llama_core.LlamaState]` | Cached state stored at this node if the node represents a complete cached token sequence. |

---

# `LlamaTrieCache`

## Overview

`LlamaTrieCache` is a trie-based cache implementation for `llama_core.LlamaState` objects.

It optimizes longest-prefix lookup by storing token sequences in a prefix tree rather than scanning all cached keys.

Defined in: `llama_cpp/llama_cache.py`

---

## Role in the API

`LlamaTrieCache` is the preferred cache implementation for efficient prefix lookup.

It combines:

* A trie for `O(K)` longest-prefix lookup.
* An `OrderedDict` for `O(1)` LRU tracking.
* Explicit byte-size tracking through `_current_size`.

The compatibility alias `LlamaCache` points to this class:

```python
LlamaCache = LlamaTrieCache
```

---

## Constructor: `__init__`

```python
def __init__(self, capacity_bytes: int = (2 << 30)):
    ...
```

| Parameter        | Type  |   Default | Required | Description                                                                                  |
| ---------------- | ----- | --------: | -------: | -------------------------------------------------------------------------------------------- |
| `capacity_bytes` | `int` | `2 << 30` |       No | Maximum cache size in bytes. Entries are evicted when tracked state size exceeds this value. |

---

## Instance Variables

| Name             | Type                                     | Description                                                                       |
| ---------------- | ---------------------------------------- | --------------------------------------------------------------------------------- |
| `root`           | `TrieNode`                               | Root node of the token-prefix trie.                                               |
| `_current_size`  | `int`                                    | Current total size of cached states in bytes.                                     |
| `lru_tracker`    | `OrderedDict[Tuple[int, ...], TrieNode]` | Tracks cached keys by recency. The value is the terminal `TrieNode` for that key. |
| `capacity_bytes` | `int`                                    | Maximum cache capacity in bytes, inherited from `BaseLlamaCache`.                 |

---

## Properties

### `cache_size`

```python
@property
def cache_size(self) -> int:
    return self._current_size
```

Returns the current total size of cached states in bytes.

This is an `O(1)` operation.

---

## Core Methods

### `_find_longest_prefix_node`

```python
def _find_longest_prefix_node(
    self,
    key: Tuple[int, ...]
) -> Tuple[Optional[TrieNode], Optional[Tuple[int, ...]]]:
    ...
```

Finds the trie node containing the longest cached prefix for the given token sequence.

Returns:

```python
Tuple[Optional[TrieNode], Optional[Tuple[int, ...]]]
```

The first item is the matching trie node.

The second item is the matching cached key.

### Behavior

1. Starts at the root node.
2. Checks whether the empty prefix has a cached state.
3. Walks one token at a time through the trie.
4. Updates the best match each time it reaches a node with a stored state.
5. Stops when the token path no longer exists.

### Complexity

| Operation     | Complexity |
| ------------- | ---------: |
| Prefix lookup |     `O(K)` |

Where `K` is the length of the requested token sequence.

---

### `__getitem__`

```python
def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
    ...
```

Retrieves the `LlamaState` for the longest matching cached prefix.

Behavior:

1. Converts `key` to a tuple.
2. Finds the longest matching trie node.
3. Raises `KeyError` if no prefix match exists.
4. Moves the matched key to the end of `lru_tracker`.
5. Returns the stored `LlamaState`.

---

### `__contains__`

```python
def __contains__(self, key: Sequence[int]) -> bool:
    ...
```

Returns `True` if any prefix of `key` is cached.

This lookup is `O(K)`.

---

### `_prune`

```python
def _prune(self, key: Tuple[int, ...]):
    ...
```

Removes a cached key from the trie and prunes empty parent nodes.

This is an internal helper used during LRU eviction.

Behavior:

1. Walks the trie path for the given key.
2. Returns immediately if the key does not exist.
3. Removes the stored state from the terminal node.
4. Subtracts the state size from `_current_size`.
5. Walks backward through the path and removes empty trie nodes.

---

### `__setitem__`

```python
def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
    ...
```

Stores a `LlamaState` in the trie cache.

Behavior:

1. Converts `key` to a tuple.
2. Creates trie nodes for each token if needed.
3. If the terminal node already has a state, subtracts the old state size.
4. Stores the new state.
5. Adds `value.llama_state_size` to `_current_size`.
6. Updates `lru_tracker`.
7. Evicts least-recently-used items while `_current_size > capacity_bytes`.

---

## Example

```python
from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaCache

llm = Llama(
    model_path="./models/model.gguf",
    cache=LlamaCache(capacity_bytes=2 << 30),
)

response = llm("Q: What is llama.cpp?\nA:", max_tokens=64)

print(response["choices"][0]["text"])
```

Because `LlamaCache` is an alias for `LlamaTrieCache`, this example uses the trie-based cache.

---

## Performance Characteristics

| Cache            | Prefix Lookup |             LRU Tracking | Storage |
| ---------------- | ------------: | -----------------------: | ------- |
| `LlamaRAMCache`  |    `O(N * K)` |                   `O(1)` | RAM     |
| `LlamaDiskCache` |    `O(N * K)` | Delegated to `diskcache` | Disk    |
| `LlamaTrieCache` |        `O(K)` |                   `O(1)` | RAM     |

Where:

* `N` is the number of cached entries.
* `K` is the token sequence length.

---

## Best Practices

* Prefer `LlamaCache` for general use, because it currently aliases `LlamaTrieCache`.
* Use `LlamaTrieCache` directly when you want explicit control over the cache implementation.
* Use a realistic `capacity_bytes` value based on available RAM.
* Use this cache when many prompts share prefixes.

---

## Common Pitfalls

* The cache still stores full `LlamaState` objects, which may be large.
* `capacity_bytes` is based on `value.llama_state_size`; this assumes each stored state reports its size accurately.
* `TrieNode` is internal and should not be manipulated directly.
* Eviction removes entries from both `lru_tracker` and the trie.

---

# `LlamaCache`

## Overview

`LlamaCache` is a backward-compatible alias for `LlamaTrieCache`.

```python
LlamaCache = LlamaTrieCache
```

This means users can import `LlamaCache` and receive the trie-based implementation.

---

## Example

```python
from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaCache

cache = LlamaCache(capacity_bytes=2 << 30)

llm = Llama(
    model_path="./models/model.gguf",
    cache=cache,
)
```

---

## Migration Notes

Older code may expect `LlamaCache` to refer to another cache implementation.

In the current source, `LlamaCache` resolves to `LlamaTrieCache`.

When documenting or debugging cache behavior, treat `LlamaCache` as equivalent to:

```python
from llama_cpp.llama_cache import LlamaTrieCache as LlamaCache
```

---

# `HybridCheckpoint`

## Overview

`HybridCheckpoint` is a dataclass representing one saved snapshot of a Hybrid or recurrent model's hidden state.

It is used by `HybridCheckpointCache`.

Defined in: `llama_cpp/llama_cache.py`

---

## Role in the API

Hybrid or recurrent models may require hidden-state rollback rather than standard KV-cache truncation.

`HybridCheckpoint` stores enough metadata to verify and restore a specific recurrent state snapshot.

---

## Dataclass Definition

```python
@dataclass
class HybridCheckpoint:
    pos: int
    data: bytes
    hash_val: str
    size: int
    seq_id: int
```

---

## Fields

| Field      | Type    | Description                                                     |
| ---------- | ------- | --------------------------------------------------------------- |
| `pos`      | `int`   | Token position where this checkpoint was taken.                 |
| `data`     | `bytes` | Raw binary RNN or Hybrid model state data.                      |
| `hash_val` | `str`   | SHA-256 hash prefix used to verify exact token-prefix matching. |
| `size`     | `int`   | Size of the state data in bytes.                                |
| `seq_id`   | `int`   | Sequence ID associated with this checkpoint.                    |

---

## Notes

`HybridCheckpoint` objects are normally created by `HybridCheckpointCache.save_checkpoint`.

Users usually do not need to instantiate this dataclass manually.

---

# `HybridCheckpointCache`

## Overview

`HybridCheckpointCache` manages RNN or Hybrid model hidden-state checkpoints.

It is designed for models that cannot physically truncate KV cache in the same way as standard transformer-only models.

Instead of implementing dictionary-style cache operations, it provides explicit checkpoint operations:

* `save_checkpoint`
* `find_best_checkpoint`
* `restore_checkpoint`
* `clear`
* `close`

Defined in: `llama_cpp/llama_cache.py`

---

## Role in the API

`HybridCheckpointCache` is a specialized cache manager for Hybrid/Recurrent model rollback.

It stores raw state snapshots extracted from the llama.cpp backend through low-level C API functions:

* `llama_state_seq_get_size_ext`
* `llama_state_seq_get_data_ext`
* `llama_state_seq_set_data_ext`

It is not a drop-in replacement for `LlamaRAMCache`, `LlamaDiskCache`, or `LlamaTrieCache`.

---

## Constructor: `__init__`

```python
def __init__(
    self,
    ctx: llama_cpp_lib.llama_context_p,
    max_checkpoints: int = 16,
    verbose: bool = False
):
    ...
```

| Parameter         | Type                            | Default | Required | Description                                                                                 |
| ----------------- | ------------------------------- | ------: | -------: | ------------------------------------------------------------------------------------------- |
| `ctx`             | `llama_cpp_lib.llama_context_p` |       — |      Yes | Low-level llama.cpp context pointer. Required for extracting and restoring sequence state.  |
| `max_checkpoints` | `int`                           |    `16` |       No | Maximum number of checkpoints to retain. If set to `0` or below, checkpointing is disabled. |
| `verbose`         | `bool`                          | `False` |       No | Enables diagnostic messages printed to `stderr`.                                            |

---

## Constructor Behavior

The constructor raises `ValueError` if `ctx` is `None`.

```python
if ctx is None:
    raise ValueError(
        "HybridCheckpointCache(__init__): Failed to create HybridCheckpointCache with model context"
    )
```

If `max_checkpoints <= 0`, checkpointing is disabled. In verbose mode, the cache reports that rollback capabilities are turned off.

This mode is intended to avoid expensive state extraction for single-turn workflows.

---

## Instance Variables

| Name              | Type                            | Description                                                                                      |
| ----------------- | ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `_ctx`            | `llama_cpp_lib.llama_context_p` | Low-level llama.cpp context pointer used for state extraction and restoration.                   |
| `max_checkpoints` | `int`                           | Maximum number of checkpoints retained. Values less than or equal to zero disable checkpointing. |
| `checkpoints`     | `list[HybridCheckpoint]`        | Stored checkpoint objects.                                                                       |
| `_current_size`   | `int`                           | Total memory used by all stored checkpoints in bytes.                                            |
| `_get_size_ext`   | Callable                        | Cached reference to `llama_state_seq_get_size_ext`.                                              |
| `_get_data_ext`   | Callable                        | Cached reference to `llama_state_seq_get_data_ext`.                                              |
| `_set_data_ext`   | Callable                        | Cached reference to `llama_state_seq_set_data_ext`.                                              |
| `_flag_partial`   | int                             | Cached value of `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`.                                            |
| `verbose`         | `bool`                          | Enables debug output.                                                                            |

---

## Properties

### `cache_size`

```python
@property
def cache_size(self) -> int:
    return self._current_size
```

Returns the total memory used by stored checkpoints in bytes.

---

## Core Methods

### `clear`

```python
def clear(self):
    ...
```

Clears all stored checkpoints and resets `_current_size` to `0`.

If the checkpoint list is already empty, it returns immediately.

In verbose mode, it prints:

```text
HybridCheckpointCache: cleared
```

---

### `close`

```python
def close(self):
    ...
```

Releases references held by the cache.

Behavior:

* Sets `checkpoints` to `None`.
* Sets `_ctx` to `None`.
* Sets cached C API function references to `None`.

This method is also called by `__del__`.

---

### `__del__`

```python
def __del__(self) -> None:
    self.close()
```

Finalizer that calls `close`.

---

### `_hash_prefix`

```python
def _hash_prefix(self, tokens: List[int], length: int) -> str:
    ...
```

Computes a SHA-256 hash for the token prefix up to `length`.

Behavior:

1. Returns `"empty"` if `length <= 0`.
2. Clamps `length` to the actual token list length.
3. Converts the selected token prefix into an `array.array('i')`.
4. Hashes the bytes with SHA-256.
5. Returns the first 32 hex characters.

This hash is used to ensure checkpoints are restored only when the token prefix exactly matches.

---

### `find_best_checkpoint`

```python
def find_best_checkpoint(
    self,
    tokens: List[int],
    seq_id: int = 0
) -> Optional[HybridCheckpoint]:
    ...
```

Finds the longest valid checkpoint matching the given token prefix and sequence ID.

Returns `None` if:

* Checkpointing is disabled.
* There are no checkpoints.
* No checkpoint matches the requested sequence ID and token prefix.

Behavior:

1. Skips checkpoints whose `seq_id` differs.
2. Skips checkpoints whose `pos` is greater than the current token length.
3. Verifies token-prefix integrity using `_hash_prefix`.
4. Returns the checkpoint with the largest matching `pos`.

---

### `save_checkpoint`

```python
def save_checkpoint(
    self,
    current_pos: int,
    tokens: List[int],
    seq_id: int = 0
) -> bool:
    ...
```

Extracts the current recurrent model state from the C++ backend and stores it as a `HybridCheckpoint`.

Returns `True` if the checkpoint was saved successfully.

Returns `False` if:

* Checkpointing is disabled.
* The backend reports state size `0`.
* State extraction writes an unexpected number of bytes.

### Behavior

1. Returns immediately if `max_checkpoints <= 0`.
2. Calls `_get_size_ext` to query the required state buffer size.
3. Allocates a `ctypes.c_uint8` buffer.
4. Calls `_get_data_ext` to extract state data.
5. Copies the state bytes into a Python `bytes` object.
6. Computes a hash of the token prefix.
7. Appends a new `HybridCheckpoint`.
8. Increments `_current_size`.
9. Evicts old checkpoints using FIFO order if the number of checkpoints exceeds `max_checkpoints`.

### Important Performance Note

The implementation intentionally bypasses checkpoint extraction when `max_checkpoints <= 0`.

This avoids potentially large synchronous VRAM-to-RAM transfers for single-turn workflows.

---

### `restore_checkpoint`

```python
def restore_checkpoint(
    self,
    cp: HybridCheckpoint,
    seq_id: int = 0
) -> bool:
    ...
```

Restores a previously saved checkpoint into the C++ backend.

Returns `True` if restoration succeeds.

Returns `False` if:

* The checkpoint sequence ID does not match the requested `seq_id`.
* The current backend state size differs from the checkpoint size.
* The backend does not report the expected number of restored bytes.

### Behavior

1. Verifies `cp.seq_id == seq_id`.
2. Queries current expected state size from the backend.
3. Verifies it matches `cp.size`.
4. Copies checkpoint bytes into a ctypes buffer.
5. Calls `_set_data_ext` to restore the state.
6. Returns whether the number of restored bytes equals `cp.size`.

---

## Disabled Dictionary Interface

`HybridCheckpointCache` inherits from `BaseLlamaCache`, but it intentionally disables the dictionary-style methods.

### `__getitem__`

```python
def __getitem__(self, key):
    raise NotImplementedError(
        "HybridCheckpointCache: pls use save_checkpoint or restore_checkpoint method"
    )
```

### `__setitem__`

```python
def __setitem__(self, key, value):
    raise NotImplementedError(
        "HybridCheckpointCache: pls use save_checkpoint or restore_checkpoint method"
    )
```

### `__contains__`

```python
def __contains__(self, key):
    raise NotImplementedError(
        "HybridCheckpointCache: pls use save_checkpoint or restore_checkpoint method"
    )
```

Users should use checkpoint-specific methods instead.

---

## Example

```python
from llama_cpp.llama_cache import HybridCheckpointCache

# `ctx` must be a valid llama.cpp context pointer.
checkpoint_cache = HybridCheckpointCache(
    ctx=ctx,
    max_checkpoints=16,
    verbose=True,
)

tokens = [1, 2, 3, 4]
current_pos = len(tokens)

saved = checkpoint_cache.save_checkpoint(
    current_pos=current_pos,
    tokens=tokens,
    seq_id=0,
)

if saved:
    checkpoint = checkpoint_cache.find_best_checkpoint(tokens, seq_id=0)

    if checkpoint is not None:
        restored = checkpoint_cache.restore_checkpoint(checkpoint, seq_id=0)
        print("Restored:", restored)
```

> Note: This example assumes `ctx` is already available from lower-level llama.cpp runtime code. Most high-level users do not manually create this cache.

---

## Best Practices

* Use `HybridCheckpointCache` only for Hybrid or recurrent model workflows that require hidden-state rollback.
* Set `max_checkpoints=0` for single-turn workflows where rollback is not needed.
* Keep `max_checkpoints` small if checkpoint states are large.
* Use `find_best_checkpoint` before calling `restore_checkpoint`.
* Do not use dictionary-style cache access with this class.

---

## Common Pitfalls

* Passing `ctx=None` raises `ValueError`.
* `max_checkpoints <= 0` disables checkpointing.
* Restoring a checkpoint with the wrong `seq_id` fails.
* Restore fails if the current backend state size no longer matches the checkpoint size.
* `close()` sets internal references to `None`; the object should not be reused afterward.
* This class is not equivalent to `LlamaCache`.

---

# Module Variables and Constants

## `LlamaCache`

```python
LlamaCache = LlamaTrieCache
```

Backward-compatible alias for `LlamaTrieCache`.

Users can import either:

```python
from llama_cpp.llama_cache import LlamaCache
```

or:

```python
from llama_cpp.llama_cache import LlamaTrieCache
```

Both refer to the trie-based cache implementation in the current source.

---

# How the Cache Implementations Compare

| Class                   | Storage |                   Prefix Lookup | Eviction                 | Persistence | Best For                                     |
| ----------------------- | ------- | ------------------------------: | ------------------------ | ----------: | -------------------------------------------- |
| `LlamaRAMCache`         | RAM     |                      `O(N * K)` | LRU                      |          No | Small in-memory caches.                      |
| `LlamaDiskCache`        | Disk    |                      `O(N * K)` | Delegated to `diskcache` |         Yes | Persistent cache across runs.                |
| `LlamaTrieCache`        | RAM     |                          `O(K)` | LRU                      |          No | Fast prefix lookup with many cached entries. |
| `HybridCheckpointCache` | RAM     | Hash-verified checkpoint search | FIFO by checkpoint count |          No | Hybrid/Recurrent model rollback.             |

---

# Recommended Entry Points

For most users:

```python
from llama_cpp.llama_cache import LlamaCache
```

This currently gives the trie-based implementation.

For explicit cache selection:

```python
from llama_cpp.llama_cache import LlamaRAMCache
from llama_cpp.llama_cache import LlamaDiskCache
from llama_cpp.llama_cache import LlamaTrieCache
```

For Hybrid/Recurrent models:

```python
from llama_cpp.llama_cache import HybridCheckpointCache
```

---

# Related Links

* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
