from abc import ABC, abstractmethod
import array
from collections import OrderedDict
import ctypes
from dataclasses import dataclass
import diskcache
import hashlib
import heapq
import sys
from typing import (
    Any,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import llama_cpp.llama as llama_core
import llama_cpp.llama_cpp as llama_cpp_lib

from .llama_types import *


def _common_prefix_length(
    cached_tokens: Tuple[int, ...], query_tokens: Tuple[int, ...]
) -> int:
    if cached_tokens is query_tokens:
        return len(cached_tokens)
    length = min(len(cached_tokens), len(query_tokens))
    # Avoid allocations for early mismatches. Compare long matching runs in C
    # using bounded tuple slices, without converting entire histories to NumPy.
    probe_length = min(length, 32)
    for index in range(probe_length):
        if cached_tokens[index] != query_tokens[index]:
            return index
    for start in range(probe_length, length, 256):
        stop = min(start + 256, length)
        if cached_tokens[start:stop] != query_tokens[start:stop]:
            for index in range(start, stop):
                if cached_tokens[index] != query_tokens[index]:
                    return index
    return length


class BaseLlamaCache(ABC):
    """Base cache class for a llama.cpp model."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        self.capacity_bytes = capacity_bytes

    @property
    @abstractmethod
    def cache_size(self) -> int:
        raise NotImplementedError

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        pass

    def _get_matching_key(
        self, key: Sequence[int], min_prefix_length: int
    ) -> Tuple[int, ...]:
        if len(key) <= min_prefix_length:
            raise KeyError("No better prefix in cache")
        query_key = tuple(key)
        cached_key = self._find_longest_prefix_key(query_key)
        if cached_key is None or _common_prefix_length(cached_key, query_key) <= min_prefix_length:
            raise KeyError("No better prefix in cache")
        return cached_key

    def get_matching_state(
        self, key: Sequence[int], min_prefix_length: int = 0
    ) -> "llama_core.LlamaState":
        """Read the best match only if its common prefix exceeds min_prefix_length.

        Custom caches retain their dictionary interface. Built-in caches also
        check key metadata before materializing a snapshot.
        """
        state = self[key]
        prefix_length = llama_core.Llama.longest_token_prefix(
            state.input_ids[:state.n_tokens], key
        )
        if prefix_length <= min_prefix_length:
            raise KeyError("No better prefix in cache")
        return state

    @abstractmethod
    def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: Sequence[int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(
        self, key: Sequence[int], value: "llama_core.LlamaState"
    ) -> None:
        raise NotImplementedError


class LlamaDiskCache(BaseLlamaCache):
    """
    Disk cache for a llama.cpp model.
    Delegates eviction and size management to the SQLite-backed `diskcache` library.
    """

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30), verbose: bool = False
    ):
        super().__init__(capacity_bytes)
        self.cache_dir = cache_dir
        # Keep diskcache's configured eviction policy.
        self.cache = diskcache.Cache(cache_dir, size_limit=capacity_bytes)
        self.verbose = verbose

    @property
    def cache_size(self):
        # Native O(1) volume calculation
        return self.cache.volume()  # type: ignore

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        # Early exit if cache is empty
        if len(self.cache) == 0:
            return None

        best_prefix_length = 0
        best_key: Optional[Tuple[int, ...]] = None
        query_length = len(key)
        for cached_key in self.cache.iterkeys():  # type: ignore
            prefix_length = _common_prefix_length(cached_key, key)
            if prefix_length > best_prefix_length:
                best_prefix_length = prefix_length
                best_key = cached_key  # type: ignore
            # Perfect match found, break to prevent full-table disk scan
            if best_prefix_length == query_length:
                break

        return best_key

    def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
        return self.get_matching_state(key)

    def get_matching_state(
        self, key: Sequence[int], min_prefix_length: int = 0
    ) -> "llama_core.LlamaState":
        return self.cache[self._get_matching_key(key, min_prefix_length)]

    def __contains__(self, key: Sequence[int]) -> bool:
        if len(self.cache) == 0:
            return False
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
        # diskcache natively handles capacity check and eviction upon assignment
        self.cache[tuple(key)] = value


class LlamaRAMCache(BaseLlamaCache):
    """
    RAM cache for a llama.cpp model.
    Maintains an LRU eviction policy with O(1) size tracking.
    """

    def __init__(self, capacity_bytes: int = (2 << 30), verbose: bool = False):
        super().__init__(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.cache_state: OrderedDict[
            Tuple[int, ...], "llama_core.LlamaState"
        ] = OrderedDict()
        self._current_size = 0
        self.verbose = verbose
        self._insertion_order: dict[Tuple[int, ...], int] = {}
        self._next_insertion_order = 0

    @property
    def cache_size(self):
        return self._current_size

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        best_prefix_length = 0
        best_key = None
        for cached_key in self.cache_state:
            prefix_length = _common_prefix_length(cached_key, key)
            if prefix_length > best_prefix_length or (
                prefix_length > 0 and prefix_length == best_prefix_length and best_key is not None
                and self._insertion_order[cached_key] < self._insertion_order[best_key]
            ):
                best_prefix_length = prefix_length
                best_key = cached_key
        return best_key

    def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
        return self.get_matching_state(key)

    def get_matching_state(
        self, key: Sequence[int], min_prefix_length: int = 0
    ) -> "llama_core.LlamaState":
        cached_key = self._get_matching_key(key, min_prefix_length)
        value = self.cache_state[cached_key]
        self.cache_state.move_to_end(cached_key)
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        if not self.cache_state:
            return False

        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
        if getattr(value, "nbytes", value.llama_state_size) > self.capacity_bytes:
            return
        key = tuple(key)
        if key in self.cache_state:
            previous = self.cache_state.pop(key)
            self._current_size -= getattr(previous, "nbytes", previous.llama_state_size)

        self.cache_state[key] = value
        if key not in self._insertion_order:
            self._insertion_order[key] = self._next_insertion_order
            self._next_insertion_order += 1
        self._current_size += getattr(value, "nbytes", value.llama_state_size)

        while self._current_size > self.capacity_bytes and len(self.cache_state) > 0:
            popped_key, popped_state = self.cache_state.popitem(last=False)
            del self._insertion_order[popped_key]
            self._current_size -= getattr(popped_state, "nbytes", popped_state.llama_state_size)
            self._current_size = max(0, self._current_size)

        if len(self.cache_state) == 0:
            self._current_size = 0


class _TrieCacheEntry(NamedTuple):
    insertion_order: int
    cache_key: Tuple[int, ...]


class TrieNode:
    """A node in the prefix tree (Trie)."""
    __slots__ = (
        "tokens", "children", "state", "cache_key", "insertion_order", "earliest_entry",
        "_child_entries",
    )

    def __init__(
        self, earliest_entry: Optional[_TrieCacheEntry] = None,
        tokens: Tuple[int, ...] = (),
    ):
        self.tokens = tokens  # Compressed incoming edge; empty only at the root.
        # Child nodes: {token_id: TrieNode}
        self.children: Dict[int, "TrieNode"] = {}
        # Stores the LlamaState if this node marks the end of a cached sequence.
        self.state: Optional["llama_core.LlamaState"] = None
        self.cache_key: Optional[Tuple[int, ...]] = None
        self.insertion_order = 0
        # Earliest inserted cache entry in this node's subtree, including itself.
        self.earliest_entry = earliest_entry
        # Allocate a heap only for wide branches that need repeated reselection.
        self._child_entries: Optional[List[Tuple[int, int]]] = None


class LlamaTrieCache(BaseLlamaCache):
    """
    A Llama cache implementation using a Trie for O(K) prefix lookup
    and an OrderedDict for O(1) LRU eviction.

    - K = length of the query key (number of tokens)
    - N = total number of items in the cache

    This solves the O(N*K) lookup bottleneck of the linear scan cache.
    Unbranched paths are compressed into token-tuple edges, so insertion and
    eviction do not allocate or discard a Python node per token.
    Wide branches use bounded lazy heaps when eviction changes their earliest
    entry; ordinary single-child paths need no heap allocation.
    """

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.root = TrieNode() # The root node of the Trie
        self._current_size = 0  # O(1) tracking of cache size in bytes
        self._next_insertion_order = 0
        self._last_inserted_node: Optional[TrieNode] = None

        # LRU Tracker:
        # Key: Cached token sequence (Tuple[int, ...])
        # Value: The *terminal* TrieNode for that key
        self.lru_tracker: OrderedDict[
            Tuple[int, ...], TrieNode
        ] = OrderedDict()

    @property
    def cache_size(self) -> int:
        """Returns the current total size of the cache in bytes (O(1))."""
        return self._current_size

    def _find_longest_prefix_node(
        self, key: Tuple[int, ...]
    ) -> Tuple[Optional[TrieNode], Optional[Tuple[int, ...]]]:
        """
        Find a state with the longest common prefix, including descendants.

        Returns: (The matching TrieNode, The matching key)
        """
        node = self.root
        depth = 0
        query_length = len(key)
        while depth < query_length:
            child = node.children.get(key[depth])
            if child is None:
                break
            edge_length = len(child.tokens)
            if edge_length == 1:  # The dictionary lookup already matched it.
                node = child
                depth += 1
                continue
            matched = _common_prefix_length(child.tokens, key[depth:depth + edge_length])
            node = child
            depth += matched
            if matched < edge_length:
                break
        entry = node.earliest_entry if depth > 0 else None
        if entry is None:
            return None, None
        return self.lru_tracker[entry.cache_key], entry.cache_key

    def _find_longest_prefix_key(self, key: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        return self._find_longest_prefix_node(key)[1]

    def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
        """
        Retrieves the state for the longest matching prefix in O(K) time.
        Updates the LRU status.
        """
        return self.get_matching_state(key)

    def get_matching_state(
        self, key: Sequence[int], min_prefix_length: int = 0
    ) -> "llama_core.LlamaState":
        cached_key = self._get_matching_key(key, min_prefix_length)
        node = self.lru_tracker[cached_key]
        self.lru_tracker.move_to_end(cached_key)
        assert node.state is not None
        return node.state

    def __contains__(self, key: Sequence[int]) -> bool:
        """Check for a nonempty common prefix without reading the state."""
        node, _ = self._find_longest_prefix_node(tuple(key))
        return node is not None

    @staticmethod
    def _rebuild_child_entries(node: TrieNode) -> None:
        node._child_entries = [
            (child.earliest_entry.insertion_order, token)
            for token, child in node.children.items()
            if child.earliest_entry is not None
        ]
        heapq.heapify(node._child_entries)

    @classmethod
    def _record_child_entry(cls, node: TrieNode, token: int) -> None:
        if node._child_entries is None:
            return
        if len(node.children) <= 8:
            node._child_entries = None
            return
        child = node.children.get(token)
        if child is not None and child.earliest_entry is not None:
            heapq.heappush(node._child_entries, (child.earliest_entry.insertion_order, token))
        # Bound stale heap entries even if the subtree's oldest entry stays live.
        if len(node._child_entries) > 2 * len(node.children):
            cls._rebuild_child_entries(node)

    @classmethod
    def _update_earliest_entry(cls, node: TrieNode) -> None:
        if len(node.children) > 8:
            if node._child_entries is None:
                cls._rebuild_child_entries(node)
            while node._child_entries:
                insertion_order, token = node._child_entries[0]
                child = node.children.get(token)
                if (
                    child is not None and child.earliest_entry is not None
                    and child.earliest_entry.insertion_order == insertion_order
                ):
                    entry = child.earliest_entry
                    break
                heapq.heappop(node._child_entries)
            else:
                entry = None
        else:
            node._child_entries = None
            if not node.children:
                entry = None
            elif len(node.children) == 1:
                entry = next(iter(node.children.values())).earliest_entry
            else:
                entry = min(
                    (child.earliest_entry for child in node.children.values()
                     if child.earliest_entry is not None),
                    key=lambda item: item.insertion_order, default=None,
                )
        if node.state is not None:
            assert node.cache_key is not None
            if entry is None or node.insertion_order < entry.insertion_order:
                entry = _TrieCacheEntry(node.insertion_order, node.cache_key)
        node.earliest_entry = entry

    def _prune(self, key: Tuple[int, ...]):
        """
        (Helper) Removes a key and its state from the Trie.
        Also removes empty parent nodes (branch pruning).
        """
        path: List[Tuple[TrieNode, int]] = []
        node = self.root
        depth = 0
        key_length = len(key)
        while depth < key_length:
            token = key[depth]
            child = node.children.get(token)
            if child is None:
                return
            edge_length = len(child.tokens)
            if edge_length > 1 and key[depth:depth + edge_length] != child.tokens:
                return
            path.append((node, token))
            depth += edge_length
            node = child

        # 2. Remove the state
        if node.state is None:
            return # Node has no state

        self._current_size -= getattr(node.state, "nbytes", node.state.llama_state_size)
        if node is self._last_inserted_node:
            self._last_inserted_node = None
        removed_order = node.insertion_order
        node.state = None
        node.cache_key = None
        # Removing a later terminal entry cannot change any ancestor's best match.
        if node.earliest_entry is None or node.earliest_entry.insertion_order != removed_order:
            if path:
                self._compress_child(*path[-1])
            return
        if node.children:
            self._update_earliest_entry(node)
            if path:
                self._compress_child(*path[-1])
        else:
            node.earliest_entry = None
            if not path:
                return
            parent, token = path[-1]
            del parent.children[token]

        # Only surviving ancestors can need a new prefix candidate.
        for index in range(len(path) - 1, -1, -1):
            parent, token = path[index]
            if parent._child_entries is not None:
                self._record_child_entry(parent, token)
            candidate_changed = (
                parent.earliest_entry is not None
                and parent.earliest_entry.insertion_order == removed_order
            )
            if candidate_changed:
                self._update_earliest_entry(parent)
            if index > 0:
                self._compress_child(*path[index - 1])
            if not candidate_changed:
                break

    @staticmethod
    def _compress_child(parent: TrieNode, token: int) -> None:
        node = parent.children[token]
        if node.state is None:
            if not node.children:
                del parent.children[token]
            elif len(node.children) == 1:
                child = next(iter(node.children.values()))
                child.tokens = node.tokens + child.tokens
                parent.children[token] = child

    def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
        """
        Adds a (key, state) pair, maintaining prefix candidates and LRU order.
        Eviction updates only affected ancestors and wide-branch candidate heaps.
        """
        state_size = getattr(value, "nbytes", value.llama_state_size)
        if state_size > self.capacity_bytes:
            return
        key_tuple = tuple(key)

        # 1. Find a terminal node, splitting compressed edges only at a branch
        # or at a newly cached prefix. Existing candidates precede new entries.
        node = self.root
        entry = None
        depth = 0
        key_length = len(key_tuple)
        previous = self._last_inserted_node
        if previous is not None and previous.cache_key is not None:
            previous_length = len(previous.cache_key)
            if key_length > previous_length and key_tuple[:previous_length] == previous.cache_key:
                node, depth = previous, previous_length
        while depth < key_length:
            token = key_tuple[depth]
            child = node.children.get(token)
            if child is None:
                if entry is None:
                    entry = _TrieCacheEntry(self._next_insertion_order, key_tuple)
                    self._next_insertion_order += 1
                child = node.children[token] = TrieNode(entry, key_tuple[depth:])
                # Insertion orders only increase, so a new entry cannot replace
                # an existing subtree candidate. New nodes start with it directly.
                if node.earliest_entry is None:
                    node.earliest_entry = entry
                if node._child_entries is not None:
                    self._record_child_entry(node, token)
                node = child
                break
            edge_length = len(child.tokens)
            if edge_length == 1:
                node = child
                depth += 1
                continue
            matched = _common_prefix_length(child.tokens, key_tuple[depth:depth + edge_length])
            if matched < edge_length:
                assert matched > 0
                branch = TrieNode(child.earliest_entry, child.tokens[:matched])
                child.tokens = child.tokens[matched:]
                branch.children[child.tokens[0]] = child
                node.children[token] = branch
                node = branch
                depth += matched
                continue
            depth += edge_length
            node = child

        # 2. Check if updating an existing item
        if node.state is not None:
            self._current_size -= getattr(node.state, "nbytes", node.state.llama_state_size)
            self.lru_tracker.move_to_end(key_tuple)
        else:
            if entry is None:
                entry = _TrieCacheEntry(self._next_insertion_order, key_tuple)
                self._next_insertion_order += 1
            node.insertion_order = entry.insertion_order
            if node.earliest_entry is None:
                node.earliest_entry = entry
            self.lru_tracker[key_tuple] = node

        # 3. Set new state and update O(1) size
        node.state = value
        node.cache_key = key_tuple
        self._current_size += state_size

        # 5. Eviction logic
        while self._current_size > self.capacity_bytes and self.lru_tracker:
            # Get the least recently used item in O(1)
            evicted_key, _ = self.lru_tracker.popitem(last=False)

            # Remove the evicted item from the Trie
            self._prune(evicted_key)
        self._last_inserted_node = node

# Alias for backwards compatibility
LlamaCache = LlamaTrieCache


@dataclass
class HybridCheckpoint:
    """
    Represents a single snapshot of the Hybrid/Recurrent model state.

    Notes:
        - When on_device=False, `data` contains the full host-side serialized state.
        - When on_device=True, `data` contains only the host-visible portion of the
          serialized state. The tensor payload is stored in llama_context-owned
          device buffers by llama.cpp, keyed by seq_id.
    """
    pos: int                       # The token position (cursor) where this snapshot was taken.
    data: bytes                    # The raw binary RNN state data.
    hash_val: str                  # SHA-256 hash of the token prefix to ensure exact sequence matching.
    size: int                      # Number of bytes written by llama_state_seq_get_data_ext().
    seq_id: int                    # Sequence id used by llama.cpp state APIs.
    pos_min: int = -1              # Minimum backend memory position covered by the snapshot.
    pos_max: Optional[int] = None  # Maximum backend memory position covered by the snapshot.

class HybridCheckpointCache(BaseLlamaCache):
    """
    Checkpoint manager for Hybrid/Recurrent model states.

    This cache is designed for models whose memory cannot be safely truncated like
    a regular Transformer KV cache. For recurrent/hybrid architectures, rollback is
    implemented by saving and restoring sequence state snapshots.

    Two operating modes are supported:

    1. Host mode: on_device=False
        - Full checkpoint payload is materialized as Python bytes.
        - Multiple checkpoints per seq_id are safe.
        - This mode is suitable for multi-turn rollback and longer conversation reuse.

    2. Device mode: on_device=True
        - LLAMA_STATE_SEQ_FLAGS_ON_DEVICE is forwarded to llama.cpp.
        - Tensor payloads are stored in llama_context-owned device buffers.
        - The device buffers are created per seq_id in llama.cpp.
        - Therefore only one active checkpoint per seq_id is safe.
        - This mode is suitable for fast speculative / branch rollback where avoiding
          device-to-host tensor copies is more important than keeping many historical
          checkpoints.

    Important:
        Do not treat on_device=True as "Python owns a VRAM checkpoint". Python only
        owns the host-visible serialized portion. The tensor payload lives inside the
        llama_context and is keyed by seq_id.
    """
    def __init__(
            self,
            ctx: llama_cpp_lib.llama_context_p,
            max_checkpoints: int = 16,
            on_device: bool = False,
            verbose: bool = False
        ):
        """
        Args:
            ctx (llama_context_p):
                Borrowed llama.cpp context pointer used by the state sequence APIs.
                This cache does not own the context and must not free it.

            max_checkpoints(int): Maximum number of Python-side checkpoint entries to keep.
                - Host mode: This is the maximum number of historical checkpoints across all seq_ids.
                - Device mode: This is still a global upper bound for Python-side metadata entries,
                               but this class also enforces at most one active checkpoint per seq_id,
                               because llama.cpp stores device tensor payloads per seq_id.

            on_device(bool): Whether to request llama.cpp to keep tensor checkpoint payloads in
                             context-owned device buffers via LLAMA_STATE_SEQ_FLAGS_ON_DEVICE.

            verbose(bool): Enables diagnostic logging to stderr for checkpoint save/restore/eviction.
        """
        if ctx is None:
            raise ValueError("HybridCheckpointCache(__init__): Failed to create HybridCheckpointCache with a null model context")
        self._ctx = ctx
        self._context_ref = None
        self.on_device = on_device
        self.verbose = verbose

        # In host mode, max_checkpoints means "maximum number of Python-owned
        # checkpoints across all seq_ids".
        #
        # In device mode, llama.cpp stores tensor payloads in device buffers keyed
        # by seq_id. Multiple Python checkpoint metadata entries for the same seq_id
        # would point to the same mutable device-side slot, so only one checkpoint
        # per seq_id is safe.
        self.max_checkpoints = max_checkpoints

        # Python-side checkpoint registry.
        #
        # Host mode:
        #   Each HybridCheckpoint owns a full serialized checkpoint payload.
        #
        # Device mode:
        #   Each HybridCheckpoint owns only the host-visible serialized portion.
        #   The corresponding tensor payload is owned by llama_context.
        self.checkpoints: list[HybridCheckpoint] = []

        # Total Python-tracked checkpoint size in bytes.
        #
        # Host mode:
        #   Roughly equals the total serialized checkpoint payload size.
        #
        # Device mode:
        #   Tracks only the host-visible part returned by llama.cpp, not the
        #   context-owned device tensor storage.
        self._current_size = 0

        # Cache C API function pointers for faster repeated calls.
        self._get_size_ext = llama_cpp_lib.llama_state_seq_get_size_ext
        self._get_data_ext = llama_cpp_lib.llama_state_seq_get_data_ext
        self._set_data_ext = llama_cpp_lib.llama_state_seq_set_data_ext
        self._get_memory = llama_cpp_lib.llama_get_memory
        self._memory_seq_pos_min = llama_cpp_lib.llama_memory_seq_pos_min
        self._memory_seq_pos_max = llama_cpp_lib.llama_memory_seq_pos_max
        self._memory_seq_rm = llama_cpp_lib.llama_memory_seq_rm

        # State serialization flags forwarded to llama.cpp.
        #
        # LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY:
        #   Save only the sequence-specific / partial state needed for recurrent
        #   rollback instead of a full context state.
        #
        # LLAMA_STATE_SEQ_FLAGS_ON_DEVICE:
        #   Ask llama.cpp to store tensor payloads in context-owned device buffers.
        self._flags = llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
        if on_device:
            self._flags |= llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE

        if self.max_checkpoints <= 0 and self.verbose:
            print("HybridCheckpointCache(__init__): Cache is DISABLED (max_checkpoints <= 0). "
                    "Rollback capabilities are turned off. This is optimal for single-turn workflows.",
                    file=sys.stderr)

        if self.on_device and self.max_checkpoints > 1 and self.verbose:
            print(
                "HybridCheckpointCache(__init__): on_device=True stores tensor payloads "
                "in llama_context-owned device buffers keyed by seq_id. Multiple "
                "historical checkpoints for the same seq_id are unsafe, so this cache "
                "will keep only one checkpoint per seq_id.",
                file=sys.stderr,
            )

    @property
    def cache_size(self) -> int:
        """
        Returns the host-visible checkpoint size tracked by Python.

        In host mode, this is close to the full serialized checkpoint payload size.
        In device mode, this is only the host-visible metadata/payload size returned
        by llama.cpp. Device-side tensor storage is owned by llama_context and is not
        fully represented by this number.
        """
        return self._current_size

    def clear(self):
        """
        Clears Python-side checkpoint metadata.

        This does not explicitly release llama_context-owned device buffers. The
        device buffers are managed by llama.cpp and are associated with the context.
        """
        if not self.checkpoints:
            # Empty Checkpoint: Return immediately, no need to clear.
            return
        self.checkpoints.clear()
        self._current_size = 0
        if self.verbose:
            print("HybridCheckpointCache(clear): cleared", file=sys.stderr)

    def close(self):
        self.clear()
        self._context_ref = None
        self._ctx = None
        self._get_size_ext = None
        self._get_data_ext = None
        self._set_data_ext = None
        self._get_memory = None
        self._memory_seq_pos_min = None
        self._memory_seq_pos_max = None
        self._memory_seq_rm = None

    def __del__(self) -> None:
        self.close()

    # Helper tools

    def _invalidate_related(self, seq_id: int, suffix_start=None, device_only=False):
        owner = self._context_ref() if self._context_ref is not None else None
        if owner is not None:
            owner._invalidate_checkpoints(
                seq_id, suffix_start=suffix_start, device_only=device_only
            )
        else:
            self._invalidate_memory(seq_id, suffix_start)

    def _invalidate_memory(
        self, seq_id: int = -1, suffix_start: Optional[int] = None,
        keep_seq_id: Optional[int] = None,
    ) -> None:
        """Drop partial snapshots whose native prefix is no longer available."""
        self.checkpoints[:] = [
            cp for cp in self.checkpoints
            if (seq_id >= 0 and cp.seq_id != seq_id)
            or cp.seq_id == keep_seq_id
            or (suffix_start is not None and cp.pos_max is not None
                and 0 <= cp.pos_max < suffix_start)
        ]
        self._current_size = sum(cp.size for cp in self.checkpoints)

    def _hash_prefix(self, tokens: List[int], length: int) -> str:
        """
        Computes a SHA-256 hash for a sequence of tokens up to the specified length.
        This ensures that checkpoints are only restored for the EXACT same conversation history.
        """
        if length <= 0:
            return "empty"
        length = min(length, len(tokens))
        data = array.array('i', tokens[:length]).tobytes()
        return hashlib.sha256(data).hexdigest()[:32]

    def _evict_checkpoints_if_needed(self) -> None:
        """
        Evicts old checkpoints if needed

        Host mode:
            This evicts full Python-owned checkpoint payloads, so FIFO historical
            checkpoints are safe and useful.

        Device mode:
            This evicts Python-side metadata only. The device tensor payload is owned
            by llama_context and is keyed by seq_id.
        """
        while len(self.checkpoints) > self.max_checkpoints:
            old_cp = self.checkpoints.pop(0)
            self._current_size -= old_cp.size
            if self._current_size < 0:
                self._current_size = 0

            if self.verbose:
                print(
                    f"HybridCheckpointCache: evicted checkpoint "
                    f"seq_id={old_cp.seq_id}, pos={old_cp.pos}",
                    file=sys.stderr,
                )

    def find_best_checkpoint(self, tokens: List[int], seq_id: int = 0) -> Optional[HybridCheckpoint]:
        """
        Finds the longest valid checkpoint that perfectly matches the provided token prefix.

        The hash check prevents restoring a checkpoint that has the same length but
        belongs to a different prompt/history.

        Returns None if no matching checkpoint is found.
        """
        # Empty Checkpoint: Instant return, no hash calculation needed.
        if self.max_checkpoints <= 0 or len(self.checkpoints) == 0:
            return None

        best_cp: Optional[HybridCheckpoint] = None
        best_pos = -1

        for cp in self.checkpoints:
            if cp.seq_id != seq_id or cp.pos > len(tokens):
                # Skip if sequence ID mismatches or checkpoint is longer than the current prompt
                continue

            # Verify cryptographic integrity of the prompt history
            if self._hash_prefix(tokens, cp.pos) == cp.hash_val:
                if cp.pos > best_pos:
                    # Keep the checkpoint with the longest matching prefix (highest pos)
                    best_pos = cp.pos
                    best_cp = cp
        return best_cp

    def save_checkpoint(
        self,
        current_pos: int,
        tokens: List[int],
        seq_id: int = 0
    ) -> bool:
        """
        Extracts the RNN hidden state from the C++ backend and saves it as a checkpoint.
        Manages eviction (FIFO) if the maximum number of checkpoints is exceeded.
        """

        # 0. Early Exit / Feature Toggle
        # If the user disables checkpoints (max_checkpoints <= 0), we immediately return.
        # This absolutely critical bypass prevents massive (e.g., 150MB+) synchronous
        # VRAM-to-RAM copies over the PCIe bus, eliminating multi-second delays at the
        # end of generation for single-turn workflows.
        # This is more friendly to the single-call ComfyUI ecosystem. :)
        if self.max_checkpoints <= 0:
            if self.verbose:
                print("HybridCheckpointCache(save_checkpoint): Cache is DISABLED (max_checkpoints <= 0). "
                      "Operating in single-turn conversation mode. Skipping state extraction to optimize generation latency.",
                      file=sys.stderr)
            return False

        # In on-device mode, remove old Python metadata for this seq_id before saving
        # the new checkpoint. The underlying llama.cpp device buffer for this seq_id
        # will be overwritten by the get_data_ext() call.
        if self.on_device:
            self._invalidate_related(seq_id, device_only=True)

        flags = self._flags

        # 1. Query the required host-visible buffer size.
        # In on_device mode this may exclude the large tensor payload
        # that stays in device memory.
        size = self._get_size_ext(self._ctx, seq_id, flags)
        if size == 0:
            if self.verbose:
                print("HybridCheckpointCache(save_checkpoint): size=0, skip")
            return False

        # 2. Allocate buffer and extract raw state data
        buffer = (ctypes.c_uint8 * size)()
        n_written = self._get_data_ext(self._ctx, buffer, size, seq_id, flags)

        if n_written != size:
            if self.verbose:
                print(
                    f"HybridCheckpointCache(save_checkpoint): get_data_ext failed "
                    f"({n_written}/{size})",
                    file=sys.stderr,
                )
            return False

        # Note: This deep copy isolates the state from subsequent C++ backend mutations
        data_bytes = bytes(buffer)
        hash_val = self._hash_prefix(tokens, current_pos)

        # Token count and backend memory positions are not interchangeable for all
        # models. Multimodal inputs, custom position IDs, and SWA can make pos_max
        # differ from current_pos - 1, so retain the native range with the snapshot.
        memory = self._get_memory(self._ctx)
        pos_min = self._memory_seq_pos_min(memory, seq_id)
        pos_max = self._memory_seq_pos_max(memory, seq_id)

        # Replace the same cursor only after the new capture succeeds.
        self.checkpoints[:] = [
            cp for cp in self.checkpoints
            if cp.seq_id != seq_id or cp.pos != current_pos
        ]
        self._current_size = sum(cp.size for cp in self.checkpoints)

        # 3. Store the newly extracted checkpoint
        self.checkpoints.append(HybridCheckpoint(
            pos=current_pos,
            data=data_bytes,
            hash_val=hash_val,
            size=n_written,
            seq_id=seq_id,
            pos_min=pos_min,
            pos_max=pos_max)
        )
        self._current_size += n_written

        # 4. Evicts old checkpoints if needed
        self._evict_checkpoints_if_needed()

        if self.verbose:
            mode = "device" if self.on_device else "host"
            print(
                f"HybridCheckpointCache(save_checkpoint): saved {mode} checkpoint "
                f"seq_id={seq_id}, pos={current_pos}, size={size / 1024 / 1024:.2f} MiB, "
                f"hcc_count={len(self.checkpoints)}, "
                f"hcc_mem_used={self._current_size / 1024 / 1024:.2f} MiB",
                file=sys.stderr,
            )

        return True

    def restore_checkpoint(self, cp: HybridCheckpoint, seq_id: int = 0) -> bool:
        """
        Injects a previously saved RNN state checkpoint back into the C++ backend memory.
        """
        # 1. Verify sequence ID matches to prevent cross-sequence contamination
        if cp.seq_id != seq_id:
            if self.verbose:
                print(f"HybridCheckpointCache(restore_checkpoint): [Error] Sequence ID mismatch: checkpoint has {cp.seq_id}, requested {seq_id}", file=sys.stderr)
            return False

        # Host partial snapshots also borrow the current attention prefix.
        if self._ctx is None or not any(item is cp for item in self.checkpoints):
            if self.verbose:
                print(
                    "HybridCheckpointCache(restore_checkpoint): stale checkpoint; refusing restore.",
                    file=sys.stderr,
                )
            return False

        flags = self._flags

        if cp.size <= 0 or cp.size != len(cp.data):
            return False

        # Native consumes const bytes synchronously. Keep their owner alive.
        state_bytes = bytes(cp.data)
        state_pointer = ctypes.cast(ctypes.c_char_p(state_bytes), ctypes.POINTER(ctypes.c_uint8))
        suffix_start = cp.pos_max + 1 if cp.pos_max is not None else cp.pos
        try:
            success = self._set_data_ext(
                self._ctx, state_pointer, cp.size, seq_id, flags
            ) == cp.size
            if success:
                success = bool(self._memory_seq_rm(
                    self._get_memory(self._ctx), seq_id, suffix_start, -1
                ))
        except BaseException:
            self._invalidate_related(seq_id)
            raise
        # Restore and suffix cleanup form one transaction. Partial failure
        # invalidates the old history even when the caller retained objects.
        self._invalidate_related(seq_id, suffix_start if success else None)

        if self.verbose:
            mode = "device" if self.on_device else "host"
            print(
                f"HybridCheckpointCache(restore_checkpoint): restore "
                f"{'OK' if success else 'FAIL'} "
                f"mode={mode}, seq_id={seq_id}, pos={cp.pos}",
                file=sys.stderr,
            )
        return success

    # Disable BaseLlamaCache Dictionary Interfaces

    def __getitem__(self, key):
        raise NotImplementedError("HybridCheckpointCache: pls use save_checkpoint or restore_checkpoint method")

    def __setitem__(self, key, value):
        raise NotImplementedError("HybridCheckpointCache: pls use save_checkpoint or restore_checkpoint method")

    def __contains__(self, key):
        raise NotImplementedError("HybridCheckpointCache: pls use save_checkpoint or restore_checkpoint method")
