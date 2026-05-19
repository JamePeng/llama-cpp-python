from abc import ABC, abstractmethod
import array
from collections import OrderedDict
import ctypes
from dataclasses import dataclass
import diskcache
import hashlib
import sys
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
)

import llama_cpp.llama as llama_core
import llama_cpp.llama_cpp as llama_cpp_lib

from .llama_types import *


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
    Delegates LRU and size management natively to the SQLite-backed `diskcache` library.
    """

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30), verbose: bool = False
    ):
        super().__init__(capacity_bytes)
        self.cache_dir = cache_dir
        # Native SQLite size limit and LRU eviction
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

        min_len = 0
        min_key: Optional[Tuple[int, ...]] = None
        target_len = len(key)
        for k in self.cache.iterkeys():  # type: ignore
            prefix_len = llama_core.Llama.longest_token_prefix(k, key, self.verbose)
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k  # type: ignore
            # Perfect match found, break to prevent full-table disk scan
            if min_len == target_len:
                break

        return min_key

    def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
        print("LlamaDiskCache.__getitem__: called", file=sys.stderr)
        if len(self.cache) == 0:
            raise KeyError("Cache is empty")

        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        # Non-destructive read: automatically updates access time for LRU
        value: "llama_core.LlamaState" = self.cache[_key]  # type: ignore
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        if len(self.cache) == 0:
            return False
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
        print("LlamaDiskCache.__setitem__: called", file=sys.stderr)
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

    @property
    def cache_size(self):
        return self._current_size

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key = None
        keys = (
            (k, llama_core.Llama.longest_token_prefix(k, key, self.verbose))
            for k in self.cache_state.keys()
        )
        for k, prefix_len in keys:
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
        if not self.cache_state:
            raise KeyError("Cache is empty")

        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value = self.cache_state[_key]
        self.cache_state.move_to_end(_key)
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        if not self.cache_state:
            return False

        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
        key = tuple(key)
        if key in self.cache_state:
            del self.cache_state[key]

        self.cache_state[key] = value
        self._current_size += value.llama_state_size

        while self._current_size > self.capacity_bytes and len(self.cache_state) > 0:
            _, popped_state = self.cache_state.popitem(last=False)
            self._current_size -= popped_state.llama_state_size
            self._current_size = max(0, self._current_size)

        if len(self.cache_state) == 0:
            self._current_size = 0


class TrieNode:
    """A node in the prefix tree (Trie)."""
    def __init__(self):
        # Child nodes: {token_id: TrieNode}
        self.children: Dict[int, "TrieNode"] = {}
        # Stores the LlamaState if this node marks the end of a cached sequence.
        self.state: Optional["llama_core.LlamaState"] = None


class LlamaTrieCache(BaseLlamaCache):
    """
    A Llama cache implementation using a Trie for O(K) prefix lookup
    and an OrderedDict for O(1) LRU eviction.

    - K = length of the query key (number of tokens)
    - N = total number of items in the cache

    This solves the O(N*K) lookup bottleneck of the linear scan cache.
    """

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.root = TrieNode() # The root node of the Trie
        self._current_size = 0  # O(1) tracking of cache size in bytes

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
        Finds the longest cached prefix for a given key in O(K) time.

        Returns: (The matching TrieNode, The matching key)
        """
        node = self.root
        longest_prefix_node: Optional[TrieNode] = None
        longest_prefix_key: Optional[Tuple[int, ...]] = None
        current_prefix: List[int] = []

        # Check if the empty prefix (root) is cached
        if node.state is not None:
            longest_prefix_node = node
            longest_prefix_key = tuple(current_prefix)

        for token in key:
            if token not in node.children:
                # Path ends, no further prefix matches
                break

            node = node.children[token]
            current_prefix.append(token)

            if node.state is not None:
                # Found a valid, longer prefix; update our best match
                longest_prefix_node = node
                longest_prefix_key = tuple(current_prefix)

        return longest_prefix_node, longest_prefix_key

    def __getitem__(self, key: Sequence[int]) -> "llama_core.LlamaState":
        """
        Retrieves the state for the longest matching prefix in O(K) time.
        Updates the LRU status.
        """
        key_tuple = tuple(key)
        node, prefix_key = self._find_longest_prefix_node(key_tuple)

        if node is None or node.state is None or prefix_key is None:
            raise KeyError(f"Key prefix not found in cache for: {key_tuple}")

        # Move the accessed key to the end (most recently used) in O(1)
        self.lru_tracker.move_to_end(prefix_key)

        return node.state

    def __contains__(self, key: Sequence[int]) -> bool:
        """Checks if any prefix of the key is cached in O(K) time."""
        node, _ = self._find_longest_prefix_node(tuple(key))
        return node is not None

    def _prune(self, key: Tuple[int, ...]):
        """
        (Helper) Removes a key and its state from the Trie.
        Also removes empty parent nodes (branch pruning).
        """
        path: List[Tuple[TrieNode, int]] = [] # Stores (parent_node, token)
        node = self.root

        # 1. Find the node and record the path
        for token in key:
            if token not in node.children:
                return # Key not found
            path.append((node, token))
            node = node.children[token]

        # 2. Remove the state
        if node.state is None:
            return # Node has no state

        self._current_size -= node.state.llama_state_size
        node.state = None

        # 3. Prune empty parent nodes backward
        for parent, token in reversed(path):
            child = parent.children[token]

            # If the child node is now empty (no children, no state), delete it
            if not child.children and child.state is None:
                del parent.children[token]
            else:
                # Node is still in use, stop pruning
                break

    def __setitem__(self, key: Sequence[int], value: "llama_core.LlamaState"):
        """
        Adds a (key, state) pair to the cache in O(K) time.
        Handles LRU updates and eviction.
        """
        key_tuple = tuple(key)

        # 1. Find or create nodes for the key (O(K))
        node = self.root
        for token in key_tuple:
            node = node.children.setdefault(token, TrieNode())

        # 2. Check if updating an existing item
        if node.state is not None:
            self._current_size -= node.state.llama_state_size

        # 3. Set new state and update O(1) size
        node.state = value
        self._current_size += value.llama_state_size

        # 4. Update LRU tracker (O(1))
        if key_tuple in self.lru_tracker:
            self.lru_tracker.move_to_end(key_tuple)
        else:
            self.lru_tracker[key_tuple] = node

        # 5. Eviction logic
        while self._current_size > self.capacity_bytes and self.lru_tracker:
            # Get the least recently used item in O(1)
            evicted_key, _ = self.lru_tracker.popitem(last=False)

            # Remove the evicted item from the Trie
            self._prune(evicted_key)

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
    pos: int        # The token position (cursor) where this snapshot was taken.
    data: bytes     # The raw binary RNN state data.
    hash_val: str   # SHA-256 hash of the token prefix to ensure exact sequence matching.
    size: int       # Number of bytes written by llama_state_seq_get_data_ext().
    seq_id: int     # Sequence id used by llama.cpp state APIs.

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
        self._ctx = None
        self._get_size_ext = None
        self._get_data_ext = None
        self._set_data_ext = None

    def __del__(self) -> None:
        self.close()

    # Helper tools

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

    def _replace_checkpoint_for_seq_id(self, seq_id: int) -> None:
        """
        Removes all Python-side checkpoints for one seq_id.

        Required for on_device=True because llama.cpp stores the device tensor
        payload per seq_id, not per Python checkpoint object.
        """
        kept: list[HybridCheckpoint] = []
        removed_size = 0

        for cp in self.checkpoints:
            if cp.seq_id == seq_id:
                removed_size += cp.size
            else:
                kept.append(cp)

        self.checkpoints = kept
        self._current_size -= removed_size
        if self._current_size < 0:
            self._current_size = 0

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
            self._replace_checkpoint_for_seq_id(seq_id)

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
        data_bytes = bytes(buffer[:n_written])
        hash_val = self._hash_prefix(tokens, current_pos)

        # 3. Store the newly extracted checkpoint
        self.checkpoints.append(HybridCheckpoint(
            pos=current_pos,
            data=data_bytes,
            hash_val=hash_val,
            size=n_written,
            seq_id=seq_id)
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

        # 2. Guard against stale on-device checkpoint objects.
        #
        # In on_device mode, Python does not own the full checkpoint tensor payload.
        # llama.cpp keeps the large tensor payload in llama_context-owned device
        # buffers keyed by seq_id. Saving a newer checkpoint for the same seq_id may
        # overwrite that device-side payload while an old HybridCheckpoint object can
        # still exist outside this cache.
        #
        # Only checkpoint objects still tracked by this cache are considered valid.
        # This avoids restoring old Python metadata together with newer device tensors.
        if self.on_device and cp not in self.checkpoints:
            if self.verbose:
                print(
                    "HybridCheckpointCache(restore_checkpoint): stale on-device checkpoint; "
                    "refusing restore because device payload may have been overwritten.",
                    file=sys.stderr,
                )
            return False

        flags = self._flags

        # 3. Verify the underlying C++ context still expects the exact same state size.
        # This prevents buffer overflows if the backend context was unexpectedly altered or reallocated.
        current_size = self._get_size_ext(self._ctx, seq_id, flags)
        if current_size != cp.size:
            if self.verbose:
                print(f"HybridCheckpointCache(restore_checkpoint): [Warning] State size mismatch before restore: "
                      f"expected checkpoint size={cp.size}, got current size={current_size} -> possible invalidation")
            return False

        # 4. Copy data back to a ctypes buffer and push to the C++ backend
        buffer = (ctypes.c_uint8 * cp.size).from_buffer_copy(cp.data)
        ret = self._set_data_ext(
            self._ctx, buffer, cp.size, seq_id, flags
        )
        success = (ret == cp.size)

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