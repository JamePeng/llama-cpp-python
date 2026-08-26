import abc
import collections
import ctypes
import enum
import sys
import time

from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

SPEC_VOCAB_MAX_SIZE_DIFFERENCE = 128
SPEC_VOCAB_CHECK_START_TOKEN_ID = 5


def speculative_output_limits(
    n_batch: int, n_parallel: int, n_draft: int
) -> Tuple[int, int]:
    """Mirror ``common_speculative_get_output_limits`` from llama.cpp."""
    if n_batch <= 0:
        raise ValueError("n_batch must be greater than zero")
    if n_parallel <= 0:
        raise ValueError("n_parallel must be greater than zero")
    per_seq = min(n_batch, 1 + max(0, n_draft))
    total = min(n_batch, n_parallel * per_seq)
    return total, per_seq


def _speculative_generation_timing_stats(
    generated_tokens: int,
    time_to_first_token_seconds: float,
    time_to_last_token_seconds: float,
) -> Dict[str, Union[int, float]]:
    """Build speculative generation metrics from active (non-yield) time.

    ``generation_*`` measures from the start of the speculative phase to the
    last generated token. With the speculative-simple flow, this includes
    verification of the held-back final prompt token. ``sustained_*`` excludes
    TTFT and measures only the remaining ``n_gen - 1`` token intervals.
    """
    generation_tokens = max(int(generated_tokens), 0)
    generation_seconds = (
        max(float(time_to_last_token_seconds), 0.0)
        if generation_tokens > 0
        else 0.0
    )
    time_to_first_token_seconds = (
        min(max(float(time_to_first_token_seconds), 0.0), generation_seconds)
        if generation_tokens > 0
        else 0.0
    )
    sustained_tokens = max(generation_tokens - 1, 0)
    sustained_seconds = (
        max(generation_seconds - time_to_first_token_seconds, 0.0)
        if sustained_tokens > 0
        else 0.0
    )

    generation_tokens_per_second = (
        generation_tokens / generation_seconds
        if generation_seconds > 0.0
        else 0.0
    )
    sustained_tokens_per_second = (
        sustained_tokens / sustained_seconds
        if sustained_seconds > 0.0
        else 0.0
    )
    return {
        "generation_tokens": generation_tokens,
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": generation_tokens_per_second,
        "time_to_first_token_seconds": time_to_first_token_seconds,
        "sustained_tokens": sustained_tokens,
        "sustained_seconds": sustained_seconds,
        "sustained_tokens_per_second": sustained_tokens_per_second,
    }


# Base on llama.cpp/common/common.h  enum common_speculative_type
class SpeculativeType(enum.IntEnum):
    # no spec
    NONE          = 0   # no speculative decoding

    # draft-family (model-based)
    DRAFT_SIMPLE  = 1   # standalone draft model speculative decoding
    DRAFT_EAGLE3  = 2   # Eagle3 speculative decoding
    DRAFT_MTP     = 3   # Multi-token prediction
    DRAFT_DFLASH  = 4   # DFlash speculative decoding
    DRAFT_DSPARK  = 5   # DSpark speculative decoding

    # ngram-family (statistical)
    NGRAM_SIMPLE  = 6   # simple self-speculative decoding based on n-grams
    NGRAM_MAP_K   = 7   # self-speculative decoding with n-gram keys only
    NGRAM_MAP_K4V = 8   # self-speculative decoding with n-gram keys and 4 m-gram values
    NGRAM_MOD     = 9   # self-speculative decoding with n-gram mod
    NGRAM_CACHE   = 10  # self-speculative decoding with 3-level n-gram cache

    COUNT         = 11  # number of speculative types, not selectable

    # helper methods
    def is_draft(self) -> bool:
        """
        Draft-family speculative decoding.

        These algorithms generate candidate tokens through
        a draft execution path.

        Note:
            DRAFT does not always mean an external draft model.
            For example:
                - MTP may use target model internal heads
                - or a separate draft model
        """
        return self in {
            SpeculativeType.DRAFT_SIMPLE,
            SpeculativeType.DRAFT_EAGLE3,
            SpeculativeType.DRAFT_MTP,
            SpeculativeType.DRAFT_DFLASH,
            SpeculativeType.DRAFT_DSPARK,
        }


    def is_ngram(self) -> bool:
        """
        NGram based speculative decoding.

        Only classify family.
        Do NOT check implementation status here.
        """
        return self in {
            SpeculativeType.NGRAM_SIMPLE,
            SpeculativeType.NGRAM_MAP_K,
            SpeculativeType.NGRAM_MAP_K4V,
            SpeculativeType.NGRAM_MOD,
            SpeculativeType.NGRAM_CACHE,
        }

    def is_eagle3(self) -> bool:
        return self == SpeculativeType.DRAFT_EAGLE3

    def is_mtp(self) -> bool:
        return self == SpeculativeType.DRAFT_MTP

    def is_dflash(self) -> bool:
        return self == SpeculativeType.DRAFT_DFLASH

    def is_dspark(self) -> bool:
        return self == SpeculativeType.DRAFT_DSPARK

    def is_none(self) -> bool:
        return self == SpeculativeType.NONE

    def __str__(self):
        return self.to_str()

    def to_str(self) -> str:
        return {
            SpeculativeType.NONE: "none",
            SpeculativeType.DRAFT_SIMPLE: "draft-simple",
            SpeculativeType.DRAFT_EAGLE3: "draft-eagle3",
            SpeculativeType.DRAFT_MTP: "draft-mtp",
            SpeculativeType.DRAFT_DFLASH: "draft-dflash",
            SpeculativeType.DRAFT_DSPARK: "draft-dspark",
            SpeculativeType.NGRAM_SIMPLE: "ngram-simple",
            SpeculativeType.NGRAM_MAP_K: "ngram-map-k",
            SpeculativeType.NGRAM_MAP_K4V: "ngram-map-k4v",
            SpeculativeType.NGRAM_MOD: "ngram-mod",
            SpeculativeType.NGRAM_CACHE: "ngram-cache",
        }.get(self, "unknown")

    @classmethod
    def from_str(cls, spec: str) -> "SpeculativeType":
        spec = spec.lower().strip()
        spec = spec.replace("_", "-")
        aliases = {
            "none": cls.NONE,
            # Simple
            "draft-simple": cls.DRAFT_SIMPLE,
            "simple": cls.DRAFT_SIMPLE,
            # EAGLE3
            "eagle3": cls.DRAFT_EAGLE3,
            "draft-eagle3": cls.DRAFT_EAGLE3,
            # MTP
            "mtp": cls.DRAFT_MTP,
            "draft-mtp": cls.DRAFT_MTP,
            # DFLASH
            "dflash": cls.DRAFT_DFLASH,
            "draft-dflash": cls.DRAFT_DFLASH,
            # DSPARK
            "dspark": cls.DRAFT_DSPARK,
            "draft-dspark": cls.DRAFT_DSPARK,
            # NGRAM
            "ngram-simple": cls.NGRAM_SIMPLE,
            "ngram-map-k": cls.NGRAM_MAP_K,
            "ngram-k": cls.NGRAM_MAP_K,
            "ngram-map-k4v": cls.NGRAM_MAP_K4V,
            "ngram-k4v": cls.NGRAM_MAP_K4V,
            "ngram-mod": cls.NGRAM_MOD,
            "ngram-cache": cls.NGRAM_CACHE,
        }
        try:
            return aliases[spec]
        except KeyError:
            raise ValueError(
                f"Unknown speculative type: {spec}"
            )

# llama.cpp/common/common.h and speculative.cpp
@dataclass
class SpecConfig:
    spec_type: SpeculativeType = SpeculativeType.NONE

    # common speculative decoding
    draft_n_max: int = 3        # number of tokens to draft for speculative decoding
    draft_n_min: int = 0        # minimum number of draft tokens to use for speculative decoding
    draft_p_split: float = 0.1  # speculative decoding split probability
    draft_p_min: float = 0.0    # minimum speculative decoding probability (greedy)

    # Optional draft model path
    # Used by:
    #   - DRAFT_SIMPLE
    #   - DRAFT_EAGLE3
    #   - DRAFT_DFLASH
    #   - DRAFT_DSPARK
    #   - DRAFT_MTP (optional)
    #
    draft_model_path: Optional[str] = None # draft model for speculative decoding

    # draft model runtime
    draft_n_gpu_layers: Union[int, Literal["auto", "all"]] = "auto"
    draft_n_threads: Optional[int] = None       # --spec-draft-threads
    draft_n_threads_batch: Optional[int] = None # --spec-draft-threads-batch
    draft_cpu_moe: bool = False         # keep all Mixture of Experts (MoE) weights in the CPU for the draft model
    draft_n_cpu_moe: int = 0            # keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model
    draft_tensor_buft_overrides: Optional[list[Any]] = None  # Draft model tensor buffer override

    # KV cache type
    draft_type_k: Optional[int] = None  # KV cache data type for K for the draft model
    draft_type_v: Optional[int] = None  # KV cache data type for V for the draft model

    # Draft execution device
    # Example:
    #   ["cuda:0"]
    #   ["cuda:1", "cuda:2"]
    draft_devices: list[str] = field(
        default_factory=list
    )

    # Let backend handle draft sampling
    draft_backend_sampling: bool = True  # offload draft sampling to the backend

    # ngram-map
    ngram_size_n: int = 12       # ngram size for lookup
    ngram_size_m: int = 48       # mgram size for speculative tokens
    ngram_min_hits: int = 1      # minimum hits at ngram/mgram lookup for mgram to be proposed
    ngram_max_entries_per_key: Optional[int] = None  # Python backend extension

    # ngram-mod
    ngram_mod_n_match: int = 24
    ngram_mod_n_max: int = 64
    ngram_mod_n_min: int = 48

    # ngram-cache
    lookup_cache_dynamic: Optional[str] = None  # path of dynamic ngram cache file for lookup decoding
    lookup_cache_static: Optional[str] = None   # path of static ngram cache file for lookup decoding

    # Extra model loading options passed to LlamaModel initialization
    draft_model_kwargs: dict = field(
        default_factory=dict
    )

    def enabled(self) -> bool:
        return not self.spec_type.is_none()

    def resolved_draft_n_gpu_layers(self) -> int:
        """Translate arg.cpp's ``auto``/``all`` spellings to native values."""
        if self.draft_n_gpu_layers == "auto":
            return -1
        if self.draft_n_gpu_layers == "all":
            return -2
        if isinstance(self.draft_n_gpu_layers, int):
            return self.draft_n_gpu_layers
        raise ValueError(
            "SpecConfig: draft_n_gpu_layers must be an int, 'auto', or 'all'"
        )

    def max_draft_tokens(self) -> int:
        """Mirror ``common_speculative_n_max`` for the selected implementation."""
        if self.spec_type.is_draft():
            return max(0, self.draft_n_max)
        if self.spec_type in {
            SpeculativeType.NGRAM_SIMPLE,
            SpeculativeType.NGRAM_MAP_K,
            SpeculativeType.NGRAM_MAP_K4V,
        }:
            return max(0, self.ngram_size_m)
        if self.spec_type == SpeculativeType.NGRAM_MOD:
            return max(0, self.ngram_mod_n_max)
        if self.spec_type == SpeculativeType.NGRAM_CACHE:
            return 8
        return 0

    def validate(self) -> None:
        if self.spec_type.is_none():
            return
        supported = {
            SpeculativeType.DRAFT_EAGLE3,
            SpeculativeType.DRAFT_MTP,
            SpeculativeType.DRAFT_DFLASH,
            SpeculativeType.DRAFT_DSPARK,
            SpeculativeType.NGRAM_MAP_K,
            SpeculativeType.NGRAM_MAP_K4V,
        }
        if self.spec_type not in supported:
            raise NotImplementedError(
                f"SpecConfig: speculative type {self.spec_type} "
                "is not implemented"
            )

        # common draft validation
        if self.draft_n_max <= 0:
            raise ValueError(
                "SpecConfig: draft_n_max must be greater than zero"
            )
        if self.draft_n_min < 0:
            raise ValueError(
                "SpecConfig: draft_n_min must be non-negative"
            )
        if self.draft_n_min > self.draft_n_max:
            raise ValueError(
                "SpecConfig: draft_n_min cannot exceed draft_n_max"
            )
        if not 0.0 <= self.draft_p_split <= 1.0:
            raise ValueError(
                "SpecConfig: draft_p_split must be between 0 and 1"
            )
        if not 0.0 <= self.draft_p_min <= 1.0:
            raise ValueError(
                "SpecConfig: draft_p_min must be between 0 and 1"
            )

        # NGram validation
        if self.spec_type.is_ngram():
            if self.ngram_size_n <= 0:
                raise ValueError(
                    "SpecConfig: ngram_size_n must be greater than zero"
                )
            if self.ngram_size_m <= 0:
                raise ValueError(
                    "SpecConfig: ngram_size_m must be greater than zero"
                )
            if self.ngram_min_hits <= 0:
                raise ValueError(
                    "SpecConfig: ngram_min_hits must be greater than zero"
                )

        if self.spec_type in {
            SpeculativeType.DRAFT_EAGLE3,
            SpeculativeType.DRAFT_DFLASH,
            SpeculativeType.DRAFT_DSPARK,
        } and not self.draft_model_path:
            raise ValueError(
                f"SpecConfig: {self.spec_type} requires draft_model_path"
            )
        if self.resolved_draft_n_gpu_layers() < -2:
            raise ValueError(
                "SpecConfig: draft_n_gpu_layers must be >= -2 "
                "(-1='auto', -2='all')"
            )
        if self.draft_n_cpu_moe < 0:
            raise ValueError("SpecConfig: draft_n_cpu_moe must be non-negative")
        if self.draft_n_threads is not None and self.draft_n_threads <= 0:
            raise ValueError("SpecConfig: draft_n_threads must be greater than zero")
        if (
            self.draft_n_threads_batch is not None
            and self.draft_n_threads_batch <= 0
        ):
            raise ValueError(
                "SpecConfig: draft_n_threads_batch must be greater than zero"
            )


class LlamaDraftModel(abc.ABC):
    """Legacy stateless draft-model API kept for compatibility.

    New code should pass :class:`SpecConfig` through ``Llama(speculative=...)``.
    """

    @abc.abstractmethod
    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        raise NotImplementedError()


class LlamaSpecEngine(abc.ABC):
    """Interface for a stateful speculative-decoding backend.

    A generation request normally calls ``begin`` once, then repeats
    ``process -> draft -> accept``. Rejected verification batches may also use
    the checkpoint/rollback methods before the next iteration. Implementations
    may own native resources, but the target model and target context remain
    owned by :class:`Llama`.
    """

    def begin(self, prompt_tokens: Sequence[int], seq_id: int = 0) -> None:
        """Initialize request state from the already-decoded prompt tokens.

        This hook is called once before the first draft. Implementations should
        synchronize their token history here and must not mutate target state.
        """
        _ = prompt_tokens, seq_id

    def process(self, batch: Any, seq_id: int = 0) -> None:
        """Consume a target batch immediately after a successful decode.

        ``batch`` is borrowed from the target context and is only valid for this
        call. Engines can copy its tokens, positions, or target hidden states to
        update the state used by the next ``draft`` call.
        """
        _ = batch, seq_id

    @abc.abstractmethod
    def draft(
        self,
        input_ids: Sequence[int],
        *,
        n_past: int,
        id_last: int,
        n_max: int,
        seq_id: int = 0,
    ) -> npt.NDArray[np.intc]:
        """Propose up to ``n_max`` tokens following ``id_last``.

        ``input_ids`` is verified history including ``id_last``; ``n_past`` is
        the target position assigned to ``id_last``. The returned one-dimensional
        ``np.intc`` array must contain only proposed continuation tokens, never
        ``id_last`` itself. An empty array means that verification should proceed
        without a speculative suffix.
        """
        raise NotImplementedError()

    def accept(self, n_accepted: int, seq_id: int = 0) -> None:
        """Commit the sampled token plus ``n_accepted`` accepted draft tokens."""
        _ = n_accepted, seq_id

    def checkpoint(self, seq_id: int = 0) -> Any:
        """Capture engine state immediately before target verification.

        Stateless engines may return ``None``. Any returned object is opaque to
        :class:`Llama` and is passed back only to rollback methods.
        """
        _ = seq_id
        return None

    def take_verification_checkpoint(self, seq_id: int = 0) -> Any:
        """Return a draft-time checkpoint, capturing one only when necessary.

        This separate hook lets an engine reuse a checkpoint already captured
        and restored inside ``draft``, avoiding a duplicate device sync or copy.
        """
        return self.checkpoint(seq_id)

    def restore(self, checkpoint: Any, seq_id: int = 0) -> None:
        """Restore the exact opaque state returned by ``checkpoint``."""
        _ = checkpoint, seq_id

    def reset_checkpoint_stats(self) -> None:
        """Reset per-request checkpoint counters and accumulated durations."""
        pass

    def checkpoint_stats(self) -> Dict[str, Union[int, float]]:
        """Return per-request checkpoint metrics without resetting them."""
        return {}

    def can_follow_target_native_rollback(self) -> bool:
        """Return whether this engine can realign after target ``seq_rm``.

        This is an engine capability, not a probe of the target context.  The
        caller separately checks that the target has enough recurrent-state
        snapshots for the rejected verification suffix.
        """
        return False

    def rollback_verified(
        self, checkpoint: Any, n_accepted: int, seq_id: int = 0
    ) -> None:
        """Keep the sampled token and accepted prefix after target rollback.

        Called only when the target context has already removed the rejected
        suffix using native recurrent snapshots. Engines must leave their own
        state positioned after ``1 + n_accepted`` verified tokens.
        """
        _ = checkpoint, n_accepted, seq_id
        raise NotImplementedError(
            "This speculative engine cannot replay accepted verification rows"
        )

    def truncate(self, position: int, seq_id: int = 0) -> None:
        """Discard engine state at and after an absolute token position."""
        _ = position, seq_id

    def clear(self) -> None:
        """Clear request state while keeping reusable model resources alive."""
        pass

    def close(self) -> None:
        """Release owned resources; repeated calls should be safe."""
        self.clear()


class LlamaNGramMapDecoding(LlamaSpecEngine):
    """
    Fast model-free speculative decoder based on prompt n-gram lookup.

    It supports two modes:

    - "ngram-map-k":
        Key-only mode. Stores n-gram key -> history positions.
        This is memory-efficient and similar to llama.cpp's ngram-map-k behavior.

    - "ngram-map-k4v":
        Key-to-value mode. Stores n-gram key -> continuation tokens.
        This uses more memory, but can return cached continuations directly.

    This class does not use a draft model. It only speculates from already verified
    token history. Therefore, rejected tokens are handled naturally when the next
    `input_ids` is passed in.

    Aligned with llama.cpp's underlying ngram-map k/k4v algorithm.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        num_pred_tokens: int = 10,
        spec_type: Literal[
            SpeculativeType.NGRAM_MAP_K,
            SpeculativeType.NGRAM_MAP_K4V
        ] = SpeculativeType.NGRAM_MAP_K,
        min_hits: int = 2,
        max_entries_per_key: Optional[int] = None,
        sync_check_tokens: int = 16,
    ) -> None:
        """
        Args:
            ngram_size:
                Number of tokens used as the lookup key.

            num_pred_tokens:
                Maximum number of draft tokens to return.

            spec_type (SpeculativeType):
                "NGRAM_MAP_K" stores only matched positions.
                "NGRAM_MAP_K4V" stores matched continuation values directly.

            min_hits:
                Minimum number of historical matches required before returning a draft.
                Use 1 for maximum recall. Use >1 to reduce low-confidence drafts.

            max_entries_per_key:
                Optional memory cap per n-gram key. K4V defaults to four
                continuations, matching ``COMMON_NGRAM_MAX_VALUES`` in llama.cpp.

            sync_check_tokens:
                Number of trailing tokens used to verify whether the new input is an
                incremental append of the previous input. This avoids expensive full
                prefix comparison while still detecting most rollback/prompt-switch cases.
        """
        if ngram_size <= 0:
            raise ValueError("ngram_size must be greater than 0")
        if num_pred_tokens <= 0:
            raise ValueError("num_pred_tokens must be greater than 0")
        if min_hits <= 0:
            raise ValueError("min_hits must be greater than 0")
        if max_entries_per_key is not None and max_entries_per_key <= 0:
            raise ValueError("max_entries_per_key must be None or greater than 0")
        if sync_check_tokens <= 0:
            raise ValueError("sync_check_tokens must be greater than 0")

        if isinstance(spec_type, str):
            raise ValueError("LlamaNGramMapDecoding: spec_type must be enum SpeculativeType at runtime")
        if spec_type not in {
            SpeculativeType.NGRAM_MAP_K,
            SpeculativeType.NGRAM_MAP_K4V,
        }:
            raise ValueError(
                f"LlamaNGramMapDecoding: unsupported NGram spec type: {spec_type}. "
                f"Expected NGRAM_MAP_K or NGRAM_MAP_K4V."
            )

        self.spec_type = spec_type
        self.ngram_size = int(ngram_size)
        self.num_pred_tokens = int(num_pred_tokens)
        self.min_hits = int(min_hits)
        self.sync_check_tokens = int(sync_check_tokens)

        if spec_type == SpeculativeType.NGRAM_MAP_K4V and max_entries_per_key is None:
            max_entries_per_key = 4
        self.max_entries_per_key = max_entries_per_key

        self._history: List[int] = []

        # In "k" mode:
        #   key -> [position, position, ...]
        self._map_k: DefaultDict[Tuple[int, ...], List[int]] = collections.defaultdict(list)

        # In "k4v" mode:
        #   key -> {position: continuation}
        #
        # A dict is used so that recent entries can be refreshed when more continuation
        # tokens become available. Draft selection is based on continuation frequency,
        # not just the most recent continuation.
        self._map_k4v: DefaultDict[
            Tuple[int, ...], Dict[int, Tuple[int, ...]]
        ] = collections.defaultdict(dict)

        # Acceptance feedback, aligned with llama.cpp's ngram-map behavior:
        # accept(n) stores how many tokens were accepted for the key/value used by
        # the previous draft and limits the future draft length for that key/value.
        self._accepted_k: Dict[Tuple[int, ...], int] = {}
        self._accepted_k4v: DefaultDict[
            Tuple[int, ...], Dict[Tuple[int, ...], int]
        ] = collections.defaultdict(dict)

        self._last_draft_key: Optional[Tuple[int, ...]] = None
        self._last_draft_value: Optional[Tuple[int, ...]] = None

        self._closed = False
        self._last_draft_len = 0

    def clear(self) -> None:
        """
        Clear token history and indexes.

        Use this when starting a completely unrelated generation while keeping the
        decoder instance reusable.
        """
        self._history.clear()
        self._map_k.clear()
        self._map_k4v.clear()
        self._accepted_k.clear()
        self._accepted_k4v.clear()
        self._last_draft_key = None
        self._last_draft_value = None
        self._last_draft_len = 0

    def close(self) -> None:
        """
        Release internal memory.

        This class does not own native memory, but clearing large Python containers
        explicitly is still useful for long-running applications.
        """
        self.clear()
        self._closed = True

    def __del__(self) -> None:
        # Best-effort cleanup. Program correctness must not depend on __del__.
        try:
            self.close()
        except Exception:
            pass

    def begin(self, prompt_tokens: Sequence[int], seq_id: int = 0) -> None:
        if seq_id != 0:
            raise NotImplementedError("N-gram speculative decoding currently supports seq_id=0")
        self.clear()
        self._sync_and_index(np.asarray(prompt_tokens, dtype=np.intc))

    def accept(self, n_accepted: int, seq_id: int = 0) -> None:
        """
        Notify how many draft tokens were accepted by the target model.

        The accepted length is written back to the key/value used by the previous
        draft. Future drafts for the same key/value are truncated to this accepted
        length, matching llama.cpp's ngram-map feedback loop.
        """
        if seq_id != 0:
            raise NotImplementedError("N-gram speculative decoding currently supports seq_id=0")
        if n_accepted < 0:
            raise ValueError("n_accepted must be non-negative")

        if self._last_draft_key is None or self._last_draft_len <= 0:
            return

        accepted = min(int(n_accepted), self._last_draft_len)

        if self.spec_type == SpeculativeType.NGRAM_MAP_K:
            self._accepted_k[self._last_draft_key] = accepted
        else:
            if self._last_draft_value is not None:
                self._accepted_k4v[self._last_draft_key][self._last_draft_value] = accepted

        self._last_draft_key = None
        self._last_draft_value = None
        self._last_draft_len = 0

    def _sync_and_index(self, input_ids: npt.NDArray[np.intc]) -> None:
        """
        Synchronize internal history with input_ids and update the n-gram index.

        The index intentionally stores only n-grams that have at least one continuation
        token. This prevents the current tail n-gram from matching itself and returning
        an empty draft.
        """
        if self._closed:
            raise RuntimeError("LlamaNGramMapDecoding is closed")

        tokens = np.asarray(input_ids, dtype=np.intc).reshape(-1).tolist()

        old_len = len(self._history)
        new_len = len(tokens)

        if new_len == 0:
            self.clear()
            return

        # Fast path: identical input, no update needed.
        if new_len == old_len:
            if self._history == tokens:
                return

        # Incremental append path.
        is_append = False
        if old_len > 0 and new_len > old_len:
            check_len = min(old_len, max(self.ngram_size, self.sync_check_tokens))
            is_append = self._history[old_len - check_len : old_len] == tokens[
                old_len - check_len : old_len
            ]

        if is_append:
            # Append only new tokens.
            self._history.extend(tokens[old_len:])

            if self.spec_type == SpeculativeType.NGRAM_MAP_K:
                # Only newly-valid keys need to be added.
                start = max(0, old_len - self.ngram_size)
            else:
                # K4V must also refresh recent keys because their continuation values
                # can grow as new tokens are appended.
                start = max(0, old_len - self.ngram_size - self.num_pred_tokens + 1)
        else:
            # Rollback, prompt switch, truncation, or unsafe mutation.
            self.clear()
            self._history.extend(tokens)
            start = 0

        # Only index keys that have at least one token after the key.
        # Valid pos satisfies:
        #   pos + ngram_size < len(history)
        end = max(0, len(self._history) - self.ngram_size)

        if start >= end:
            return

        if self.spec_type == SpeculativeType.NGRAM_MAP_K:
            for pos in range(start, end):
                key = tuple(self._history[pos : pos + self.ngram_size])
                bucket = self._map_k[key]

                if not bucket or bucket[-1] != pos:
                    bucket.append(pos)

                if (
                    self.max_entries_per_key is not None
                    and len(bucket) > self.max_entries_per_key
                ):
                    del bucket[: len(bucket) - self.max_entries_per_key]

        else:
            for pos in range(start, end):
                key_start = pos
                value_start = pos + self.ngram_size
                value_end = value_start + self.num_pred_tokens

                # K4V tracks fixed-size continuation m-grams. Partial tail values are
                # intentionally skipped so frequency statistics remain comparable.
                if value_end > len(self._history):
                    continue

                key = tuple(self._history[key_start:value_start])
                value = tuple(self._history[value_start:value_end])

                bucket = self._map_k4v[key]
                bucket[pos] = value

                if (
                    self.max_entries_per_key is not None
                    and len(bucket) > self.max_entries_per_key
                ):
                    # Keep the most recent positions.
                    for old_pos in sorted(bucket)[: len(bucket) - self.max_entries_per_key]:
                        del bucket[old_pos]

    def draft(
        self,
        input_ids: Sequence[int],
        *,
        n_past: int,
        id_last: int,
        n_max: int,
        seq_id: int = 0,
    ) -> npt.NDArray[np.intc]:
        """
        Generate draft tokens from verified token history.

        Args:
            input_ids:
                Complete verified token sequence so far.

        Returns:
            np.ndarray[np.intc]:
                Predicted draft tokens. Empty array means no reliable match was found.
        """
        _ = n_past, id_last
        if seq_id != 0:
            raise NotImplementedError("N-gram speculative decoding currently supports seq_id=0")
        if n_max <= 0:
            return np.array([], dtype=np.intc)

        self._sync_and_index(np.asarray(input_ids, dtype=np.intc))
        self._last_draft_key = None
        self._last_draft_value = None
        self._last_draft_len = 0

        if len(self._history) < self.ngram_size:
            return np.array([], dtype=np.intc)

        search_key = tuple(self._history[-self.ngram_size :])

        if self.spec_type == SpeculativeType.NGRAM_MAP_K:
            positions = self._map_k.get(search_key)
            if not positions:
                return np.array([], dtype=np.intc)

            # Key-only mode follows llama.cpp's ngram-map-k behavior: once a key
            # match is found, draft from the latest valid match. min_hits is not
            # used as a confidence gate for key-only mode.
            draft: List[int] = []
            accepted_limit = self._accepted_k.get(search_key, self.num_pred_tokens)
            if accepted_limit <= 0:
                return np.array([], dtype=np.intc)

            for pos in reversed(positions):
                start = pos + self.ngram_size
                if start < len(self._history):
                    end = min(start + accepted_limit, len(self._history))
                    draft = self._history[start:end]
                    break

            self._last_draft_key = search_key

        else:
            values = self._map_k4v.get(search_key)
            if not values or len(values) < self.min_hits:
                return np.array([], dtype=np.intc)

            # K4V mode chooses the most frequent continuation m-gram rather than the
            # latest one. If the strongest continuation is not at least twice as
            # frequent as all other continuations combined, skip drafting.
            counts = collections.Counter(values.values())
            best_value, best_count = counts.most_common(1)[0]
            other_count = sum(counts.values()) - best_count

            if other_count > 0 and best_count < 2 * other_count:
                return np.array([], dtype=np.intc)

            accepted_limit = self._accepted_k4v[search_key].get(
                best_value, self.num_pred_tokens
            )
            if accepted_limit <= 0:
                return np.array([], dtype=np.intc)

            draft = list(best_value[:accepted_limit])
            self._last_draft_key = search_key
            self._last_draft_value = best_value

        self._last_draft_len = len(draft)
        if self._last_draft_len <= 0:
            self._last_draft_key = None
            self._last_draft_value = None
            return np.array([], dtype=np.intc)

        return np.asarray(draft[: min(n_max, self.num_pred_tokens)], dtype=np.intc)


def create_spec_engine(config: SpecConfig) -> LlamaSpecEngine:
    """Create a speculative engine that does not own native model resources.

    This factory is for algorithms such as NGram lookup that only consume token
    history. They can be initialized from :class:`SpecConfig` alone and do not
    need a target model, target context, hidden states, or a draft context.

    Draft-family algorithms must use :func:`create_native_spec_engine` after
    :class:`Llama` has created the target model and context.
    """
    config.validate()
    if config.spec_type in {
        SpeculativeType.NGRAM_MAP_K,
        SpeculativeType.NGRAM_MAP_K4V,
    }:
        return LlamaNGramMapDecoding(
            ngram_size=config.ngram_size_n,
            num_pred_tokens=config.ngram_size_m,
            spec_type=config.spec_type,
            min_hits=config.ngram_min_hits,
            max_entries_per_key=config.ngram_max_entries_per_key,
        )
    raise RuntimeError(
        f"{config.spec_type} requires target model/context initialization"
    )


class _LlamaModelDraftEngine(LlamaSpecEngine):
    """Infrastructure shared by model-backed speculative-decoding engines.

    This private base class centralizes the native plumbing used by MTP,
    DFlash, and DSpark: draft model parameters, target/draft compatibility,
    context defaults, backend candidate sampling, ctypes buffer lifetimes, and
    resource cleanup. It intentionally does not implement an algorithm's graph
    inputs, checkpoint semantics, draft layout, or acceptance state. Those
    responsibilities remain in the concrete :class:`LlamaSpecEngine` subclass.

    To adapt another model-backed algorithm, its constructor should:

    1. validate its algorithm-specific :class:`SpecConfig` requirements and
       call :meth:`_init_model_draft_engine` before allocating native objects;
    2. load an external sidecar with :meth:`_load_draft_model`, or explicitly
       assign a borrowed model when the target contains the draft heads;
    3. start with :meth:`_build_draft_context_params`, then set only the graph-
       specific context fields before constructing its context and batches;
    4. optionally call :meth:`_enable_backend_sampling` after ``draft_context``
       exists, and implement the abstract request hooks inherited from
       :class:`LlamaSpecEngine`;
    5. implement idempotent ``close()`` using :meth:`_close_draft_resources`,
       passing ``close_model=False`` for a target-owned model.

    The target model and context are always borrowed and must never be closed
    here. Concrete engines own their draft context and batches. An external
    draft model is owned by its engine, whereas internal MTP heads borrow the
    target model. Keeping this ownership boundary explicit is essential because
    ``close()`` may also run while the Python interpreter is shutting down.
    """

    @staticmethod
    def _validate_speculative_vocab_compatibility(
        target_model: Any, draft_model: Any
    ) -> None:
        """Reject target/draft vocabularies that cannot share token IDs.

        The checks mirror llama.cpp's speculative model validation: vocabulary
        type and enabled BOS/EOS behavior must match, vocabulary sizes may only
        differ by the small allowed tail, and every shared normal token must
        decode to identical text. A mismatch raises :class:`ValueError` before
        a draft context is created.
        """
        if target_model.vocab_type() != draft_model.vocab_type():
            raise ValueError(
                "Draft and target models use different vocabulary types"
            )

        for add_name, token_name in (
            ("get_add_bos", "token_bos"),
            ("get_add_eos", "token_eos"),
        ):
            target_add = bool(getattr(target_model, add_name)())
            draft_add = bool(getattr(draft_model, add_name)())
            if target_add != draft_add:
                raise ValueError(
                    f"Draft and target models disagree on {add_name}"
                )
            if target_add and getattr(target_model, token_name)() != getattr(
                draft_model, token_name
            )():
                raise ValueError(
                    f"Draft and target models use different {token_name}"
                )

        target_size = target_model.n_vocab()
        draft_size = draft_model.n_vocab()
        if abs(target_size - draft_size) > SPEC_VOCAB_MAX_SIZE_DIFFERENCE:
            raise ValueError(
                "Draft and target vocabulary sizes differ too much: "
                f"{draft_size} != {target_size}"
            )

        for token in range(
            SPEC_VOCAB_CHECK_START_TOKEN_ID, min(target_size, draft_size)
        ):
            if target_model.token_get_text(token) != draft_model.token_get_text(
                token
            ):
                raise ValueError(
                    f"Draft and target vocabularies differ at token {token}"
                )

    @staticmethod
    def _copy_draft_context_params(
        context_params: Any, pooling_type_unspecified: int
    ) -> Any:
        """Copy target parameters without inheriting embedding or pooling mode.

        This mirrors ``common_base_params_to_speculative`` in llama.cpp. Draft
        contexts configure hidden-state extraction themselves and must not
        inherit a target embedding request or its pooling behavior.
        """
        params = type(context_params).from_buffer_copy(context_params)
        params.embeddings = False
        params.pooling_type = pooling_type_unspecified
        return params

    def _init_model_draft_engine(
        self,
        config: SpecConfig,
        *,
        target_model: Any,
        target_context: Any,
        llama_cpp_lib: Any,
        verbose: bool,
    ) -> None:
        """Initialize shared Python state before allocating draft resources.

        ``target_model`` and ``target_context`` are borrowed. ``llama_cpp_lib``
        is retained so cleanup never needs a lazy import during interpreter
        shutdown. The retained ctypes arrays are initialized here because model
        parameter pointers may continue referencing them after model loading.
        """
        self.config = config
        # close() can run during interpreter shutdown. Retain the already
        # imported module instead of relying on Python's disabled importer.
        self._llama_cpp_lib = llama_cpp_lib
        self.target_model = target_model
        self.target_context = target_context
        self.verbose = verbose
        self._closed = False
        self._backend_sampler = None
        self._backend_sampling = False
        # ctypes arrays and their pattern bytes must outlive model loading.
        self._draft_devices = None
        self._draft_buft_patterns = None
        self._draft_buft_overrides = None

    def _build_draft_model_params(
        self, model_params: Any, *, load_mtp: bool
    ) -> Any:
        """Build sidecar model parameters from target loading defaults.

        The returned ctypes structure applies draft GPU placement, explicit
        devices, CPU MoE tensor overrides, user-supplied parameter overrides,
        and the algorithm-specific ``load_mtp`` flag. Device and buffer-override
        arrays are stored on ``self`` so their native pointers remain valid.
        Unknown device names or parameter fields raise :class:`ValueError`.
        """
        draft_params = type(model_params).from_buffer_copy(model_params)
        draft_params.n_gpu_layers = self.config.resolved_draft_n_gpu_layers()
        draft_params.load_mtp = load_mtp

        if self.config.draft_devices:
            from llama_cpp._ggml import ggml_backend_dev_by_name

            DeviceArray = ctypes.c_void_p * (len(self.config.draft_devices) + 1)
            self._draft_devices = DeviceArray()
            for i, name in enumerate(self.config.draft_devices):
                device = ggml_backend_dev_by_name(name.encode("utf-8"))
                if not device:
                    raise ValueError(f"Unknown draft backend device: {name}")
                self._draft_devices[i] = device
            self._draft_devices[-1] = None
            draft_params.devices = self._draft_devices

        if (
            self.config.draft_cpu_moe
            or self.config.draft_n_cpu_moe > 0
            or self.config.draft_tensor_buft_overrides
        ):
            from llama_cpp._ggml import ggml_backend_cpu_buffer_type

            override_values = list(
                self.config.draft_tensor_buft_overrides or []
            )
            if self.config.draft_cpu_moe:
                patterns = [rb"\.ffn_(up|down|gate|gate_up)_(ch|)exps"]
            else:
                patterns = [
                    f"blk\\.{i}".encode("utf-8")
                    + rb"\.ffn_(up|down|gate|gate_up)_(ch|)exps"
                    for i in range(self.config.draft_n_cpu_moe)
                ]
            self._draft_buft_patterns = patterns
            cpu_buft = ggml_backend_cpu_buffer_type()
            override_type = self._llama_cpp_lib.llama_model_tensor_buft_override
            OverrideArray = override_type * (
                len(override_values) + len(patterns) + 1
            )
            self._draft_buft_overrides = OverrideArray()
            out = 0
            for value in override_values:
                self._draft_buft_overrides[out].pattern = value.pattern
                self._draft_buft_overrides[out].buft = value.buft
                out += 1
            for pattern in patterns:
                self._draft_buft_overrides[out].pattern = pattern
                self._draft_buft_overrides[out].buft = cpu_buft
                out += 1
            self._draft_buft_overrides[out].pattern = None
            self._draft_buft_overrides[out].buft = None
            draft_params.tensor_buft_overrides = self._draft_buft_overrides

        for name, value in self.config.draft_model_kwargs.items():
            if not hasattr(draft_params, name):
                raise ValueError(f"Unknown draft model parameter: {name}")
            setattr(draft_params, name, value)
        return draft_params

    def _load_draft_model(
        self,
        internals: Any,
        model_params: Any,
        *,
        load_mtp: bool,
    ) -> Any:
        """Load and validate an engine-owned external draft model.

        Compatibility is checked immediately after native loading. If checking
        fails, the newly created model is closed before the original exception
        is propagated. The caller owns the returned model and must eventually
        pass ``close_model=True`` to :meth:`_close_draft_resources`.
        """
        draft_model = internals.LlamaModel(
            path_model=self.config.draft_model_path,
            params=self._build_draft_model_params(
                model_params, load_mtp=load_mtp
            ),
            verbose=self.verbose,
        )
        try:
            self._validate_speculative_vocab_compatibility(
                self.target_model, draft_model
            )
        except BaseException:
            draft_model.close()
            raise
        return draft_model

    def _build_draft_context_params(
        self, context_params: Any, target_context: Any
    ) -> Any:
        """Create common draft context parameters from target parameters.

        Embedding mode and pooling are reset first, matching llama.cpp's
        speculative context conversion. Draft-specific thread counts and KV
        cache types are then applied. The draft context follows the initialized
        target context's actual size, which may differ from the originally
        requested ``n_ctx`` after model-dependent resolution. The concrete
        engine must set fields such as ``ctx_type``, ``ctx_other``, batch
        capacity, outputs, attention mode, and recurrent snapshot count before
        constructing ``LlamaContext``.
        """
        params = self._copy_draft_context_params(
            context_params, self._llama_cpp_lib.LLAMA_POOLING_TYPE_UNSPECIFIED
        )
        params.n_ctx = target_context.n_ctx()
        if self.config.draft_n_threads is not None:
            params.n_threads = int(self.config.draft_n_threads)
        if self.config.draft_n_threads_batch is not None:
            params.n_threads_batch = int(self.config.draft_n_threads_batch)
        elif self.config.draft_n_threads is not None:
            params.n_threads_batch = int(self.config.draft_n_threads)
        if self.config.draft_type_k is not None:
            params.type_k = self.config.draft_type_k
        if self.config.draft_type_v is not None:
            params.type_v = self.config.draft_type_v
        return params

    def _enable_backend_sampling(self, internals: Any) -> None:
        """Attach the shared top-k sampler to the initialized draft context.

        This is a best-effort optimization requested by
        ``draft_backend_sampling``. A backend that accepts the sampler can
        expose compact candidates, probabilities, or logits; otherwise the
        temporary sampler is closed and :meth:`_candidate` uses full CPU logits.
        """
        if not self.config.draft_backend_sampling:
            return
        backend_sampler = internals.LlamaSampler()
        backend_sampler.add_top_k(10)
        if self._llama_cpp_lib.llama_set_sampler(
            self.draft_context.ctx, 0, backend_sampler.sampler
        ):
            self._backend_sampler = backend_sampler
            self._backend_sampling = True
        else:
            backend_sampler.close()

    def _reset_backend_sampler(self) -> None:
        """Reset request-local sampler state without detaching the sampler."""
        if self._backend_sampler is not None:
            self._backend_sampler.reset()

    def _close_backend_sampler(self, errors: List[Exception]) -> None:
        """Detach and close the sampler while preserving cleanup failures.

        Cleanup remains non-short-circuiting: detach and close exceptions are
        appended to ``errors`` so callers can release every remaining resource
        and raise the first failure afterward.
        """
        backend_sampler = self._backend_sampler
        self._backend_sampler = None
        self._backend_sampling = False
        if backend_sampler is None:
            return
        if getattr(self, "draft_context", None) is not None:
            try:
                self._llama_cpp_lib.llama_set_sampler(
                    self.draft_context.ctx, 0, None
                )
            except Exception as exc:
                errors.append(exc)
        try:
            backend_sampler.close()
        except Exception as exc:
            errors.append(exc)

    def _close_draft_resources(
        self,
        errors: List[Exception],
        *,
        batch_names: Sequence[str],
        close_model: bool,
    ) -> None:
        """Release draft resources in sampler, batch, context, model order.

        ``batch_names`` lets each algorithm declare its owned batch attributes.
        ``close_model`` must be true only for an external engine-owned model.
        Attributes are cleared before closing to keep repeated or shutdown-time
        cleanup idempotent. All exceptions are collected in ``errors`` rather
        than preventing later native resources from being released.
        """
        self._close_backend_sampler(errors)
        for name in batch_names:
            batch = getattr(self, name, None)
            setattr(self, name, None)
            if batch is not None:
                try:
                    batch.close()
                except Exception as exc:
                    errors.append(exc)

        draft_context = getattr(self, "draft_context", None)
        self.draft_context = None
        if draft_context is not None:
            try:
                draft_context.close()
            except Exception as exc:
                errors.append(exc)

        draft_model = getattr(self, "draft_model", None)
        self.draft_model = None
        if close_model and draft_model is not None:
            try:
                draft_model.close()
            except Exception as exc:
                errors.append(exc)

    @staticmethod
    def _copy_rows(
        pointer: Any, rows: int, width: int
    ) -> npt.NDArray[np.float32]:
        """Copy a contiguous native float matrix into an owned NumPy array.

        Native embedding pointers are only valid until the context performs
        more work. Returning a copy prevents later decode/encode calls from
        mutating hidden states retained for drafting, verification, or replay.
        A null output pointer is treated as a graph/configuration error.
        """
        if not pointer:
            raise RuntimeError("speculative embedding output is unavailable")
        return np.ctypeslib.as_array(pointer, shape=(rows * width,)).reshape(
            rows, width
        ).copy()

    def _candidate(self, output_index: int) -> Tuple[int, float]:
        """Return the most likely token and approximate top-k probability.

        Backend sampling is preferred because it avoids transferring or
        scanning the full vocabulary, which is particularly important for very
        large vocabularies. The method accepts the compact formats exposed by
        llama.cpp: selected token plus probabilities, filter-only candidates
        plus logits, or a selected token without probabilities when ``p_min``
        is disabled. If no usable compact output exists, it selects from the
        top ten entries of the full logits row on the CPU.
        """
        if self._backend_sampling:
            token = int(
                self._llama_cpp_lib.llama_get_sampled_token_ith(
                    self.draft_context.ctx, output_index
                )
            )
            candidates_count = int(
                self._llama_cpp_lib.llama_get_sampled_candidates_count_ith(
                    self.draft_context.ctx, output_index
                )
            )
            probs_count = int(
                self._llama_cpp_lib.llama_get_sampled_probs_count_ith(
                    self.draft_context.ctx, output_index
                )
            )
            logits_count = int(
                self._llama_cpp_lib.llama_get_sampled_logits_count_ith(
                    self.draft_context.ctx, output_index
                )
            )
            candidates_ptr = self._llama_cpp_lib.llama_get_sampled_candidates_ith(
                self.draft_context.ctx, output_index
            )
            probs_ptr = self._llama_cpp_lib.llama_get_sampled_probs_ith(
                self.draft_context.ctx, output_index
            )
            logits_ptr = self._llama_cpp_lib.llama_get_sampled_logits_ith(
                self.draft_context.ctx, output_index
            )

            if candidates_count > 0 and candidates_ptr:
                candidates = np.ctypeslib.as_array(
                    candidates_ptr, shape=(candidates_count,)
                )
                if probs_count > 0 and probs_ptr:
                    count = min(candidates_count, probs_count)
                    probs = np.ctypeslib.as_array(probs_ptr, shape=(count,))
                    if token == self._llama_cpp_lib.LLAMA_TOKEN_NULL:
                        selected = int(np.argmax(probs))
                        return int(candidates[selected]), float(probs[selected])
                    matches = np.flatnonzero(candidates[:count] == token)
                    if matches.size:
                        return token, float(probs[int(matches[0])])
                if logits_count > 0 and logits_ptr:
                    count = min(candidates_count, logits_count)
                    logits = np.ctypeslib.as_array(logits_ptr, shape=(count,))
                    selected = int(np.argmax(logits))
                    shifted = logits.astype(np.float64) - float(np.max(logits))
                    probs = np.exp(shifted)
                    probability = float(probs[selected] / probs.sum())
                    selected_token = int(candidates[selected])
                    if token != self._llama_cpp_lib.LLAMA_TOKEN_NULL:
                        selected_token = token
                    return selected_token, probability
            if (
                token != self._llama_cpp_lib.LLAMA_TOKEN_NULL
                and self.config.draft_p_min <= 0.0
            ):
                return token, 1.0

        logits_ptr = self.draft_context.get_logits_ith(output_index)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self.n_vocab,))
        k = min(10, self.n_vocab)
        top_indices = np.argpartition(logits, -k)[-k:]
        top_logits = logits[top_indices].astype(np.float64)
        best_local = int(np.argmax(top_logits))
        token = int(top_indices[best_local])
        shifted = top_logits - float(np.max(top_logits))
        probability = float(np.exp(shifted[best_local]) / np.exp(shifted).sum())
        return token, probability


class LlamaMTPDecoding(_LlamaModelDraftEngine):
    """Orchestrate llama.cpp's stateful NextN/MTP speculative-decoding graph.

    The engine can use NextN heads embedded in the target GGUF or load an
    external MTP sidecar with ``load_mtp=True``. Both modes create a dedicated
    MTP context linked to the target context through ``ctx_other``. The target
    context owns normal autoregressive verification; this engine owns the
    hidden-state handoff and the speculative draft branch.
    """

    def __init__(
        self,
        config: SpecConfig,
        *,
        target_model: Any,
        target_context: Any,
        model_params: Any,
        context_params: Any,
        verbose: bool = True,
    ) -> None:
        """Create an internal-head or external-sidecar MTP draft context."""
        from llama_cpp import _internals as internals
        from llama_cpp import llama_cpp as llama_cpp_lib

        if config.spec_type != SpeculativeType.DRAFT_MTP:
            raise ValueError("LlamaMTPDecoding requires spec_type=DRAFT_MTP")
        config.validate()

        self._init_model_draft_engine(
            config,
            target_model=target_model,
            target_context=target_context,
            llama_cpp_lib=llama_cpp_lib,
            verbose=verbose,
        )
        self._owns_model = bool(config.draft_model_path)

        if self._owns_model:
            self.draft_model = self._load_draft_model(
                internals, model_params, load_mtp=True
            )
        else:
            self.draft_model = target_model

        try:
            draft_ctx_params = self._build_draft_context_params(
                context_params, target_context
            )
            draft_ctx_params.ctx_type = (
                llama_cpp_lib.llama_context_type.LLAMA_CONTEXT_TYPE_MTP
            )
            draft_ctx_params.ctx_other = target_context.ctx
            # Candidate generation can advance the draft context by at most
            # draft_n_max positions. Keep native recurrent snapshots so the
            # speculative branch can normally be discarded without serializing
            # state through host memory.
            draft_ctx_params.n_rs_seq = max(0, int(config.draft_n_max))
            # llama.cpp's draft context produces one output per sequence.  Keeping
            # this separate from the target verification limits avoids reserving
            # an unnecessary (1 + n_draft) output buffer in the MTP graph.
            draft_ctx_params.n_outputs_max = max(1, int(draft_ctx_params.n_seq_max))
            draft_ctx_params.n_outputs_max_per_seq = 1
            self.draft_context = internals.LlamaContext(
                model=self.draft_model,
                params=draft_ctx_params,
                verbose=verbose,
            )
            self.n_embd = self.draft_model.n_embd_out()
            target_n_embd_out = self.target_model.n_embd_out()
            if self.n_embd != target_n_embd_out:
                raise ValueError(
                    "MTP draft output width does not match target NextN output width: "
                    f"{self.n_embd} != {target_n_embd_out}"
                )
            self.n_vocab = self.draft_model.n_vocab()
            self.n_mtp_layers = max(1, self.draft_model.n_layer_nextn())
            other_addr = ctypes.cast(
                self.draft_context.get_ctx_other(), ctypes.c_void_p
            ).value
            target_addr = ctypes.cast(target_context.ctx, ctypes.c_void_p).value
            self.is_mem_shared = other_addr == target_addr
            self.chain_heads = self.n_mtp_layers > 1 and not self.is_mem_shared
            self.batch = internals.LlamaBatch(
                n_tokens=self.draft_context.n_batch(),
                embd=self.n_embd,
                n_seq_max=1,
                mixed=True,
                verbose=verbose,
            )

            self._enable_backend_sampling(internals)
        except BaseException:
            self._close_draft_resources(
                [], batch_names=("batch",), close_model=self._owns_model
            )
            raise

        self.target_context.set_embeddings_nextn(True, masked=False)
        self.draft_context.set_embeddings_nextn(True, masked=True)
        self.pending_h = np.zeros(self.n_embd, dtype=np.float32)
        self.verify_h = np.empty((0, self.n_embd), dtype=np.float32)
        self.verify_tokens: List[int] = []
        self.verify_positions: List[int] = []
        self._use_native_draft_rollback = (
            not (self.draft_model.is_recurrent() or self.draft_model.is_hybrid())
            or self.draft_context.n_rs_seq() >= config.draft_n_max
        )
        self._pending_verification_checkpoint = None
        self.reset_checkpoint_stats()

        if self.verbose:
            self._print_runtime_configuration(context_params)

    def begin(self, prompt_tokens: Sequence[int], seq_id: int = 0) -> None:
        """Warn when prompt processing did not fully populate the MTP cache."""
        if seq_id != 0:
            raise NotImplementedError(
                "MTP speculative decoding currently supports seq_id=0"
            )
        if not prompt_tokens or self.is_mem_shared:
            return
        pos_max = self.draft_context.memory_seq_pos_max(seq_id)
        expected = len(prompt_tokens) - 1
        if pos_max < expected and self.verbose:
            print(
                "LlamaMTPDecoding: draft context prompt cache is incomplete: "
                f"pos_max={pos_max}, expected at least {expected}; process() may "
                "not have run for every prompt ubatch and draft quality may degrade",
                file=sys.stderr,
            )

    def _print_runtime_configuration(self, target_context_params: Any) -> None:
        """Print requested MTP options together with their resolved runtime state."""
        config = self.config
        model_source = (
            config.draft_model_path
            if config.draft_model_path is not None
            else "target-internal-heads"
        )
        gpu_layers: Union[int, str] = (
            config.resolved_draft_n_gpu_layers()
            if self._owns_model
            else "target-model"
        )
        devices = ",".join(config.draft_devices) if config.draft_devices else "auto"
        threads = (
            str(config.draft_n_threads)
            if config.draft_n_threads is not None
            else "inherit"
        )
        threads_batch = (
            str(config.draft_n_threads_batch)
            if config.draft_n_threads_batch is not None
            else "inherit"
        )
        type_k = str(config.draft_type_k) if config.draft_type_k is not None else "inherit"
        type_v = str(config.draft_type_v) if config.draft_type_v is not None else "inherit"

        print("LlamaMTPDecoding: MTP speculative decoding enabled", file=sys.stderr)
        print(
            "LlamaMTPDecoding: "
            f"type={config.spec_type.to_str()}, model={model_source!r}, "
            f"draft_n_min={config.draft_n_min}, draft_n_max={config.draft_n_max}, "
            f"draft_p_min={config.draft_p_min:g}, "
            f"draft_p_split={config.draft_p_split:g}",
            file=sys.stderr,
        )
        print(
            "LlamaMTPDecoding: "
            f"backend_sampling=requested:{config.draft_backend_sampling}/"
            f"active:{self._backend_sampling}, mtp_heads={self.n_mtp_layers}, "
            f"memory_shared={self.is_mem_shared}, chain_heads={self.chain_heads}",
            file=sys.stderr,
        )
        print(
            "LlamaMTPDecoding: "
            f"draft_gpu_layers={gpu_layers}, devices={devices}, "
            f"threads={threads}, threads_batch={threads_batch}, "
            f"cpu_moe={config.draft_cpu_moe}, n_cpu_moe={config.draft_n_cpu_moe}, "
            f"type_k={type_k}, type_v={type_v}",
            file=sys.stderr,
        )
        print(
            "LlamaMTPDecoding: "
            f"target_n_batch={self.target_context.n_batch()}, "
            f"draft_n_batch={self.draft_context.n_batch()}, "
            f"target_n_rs_seq={int(target_context_params.n_rs_seq)}, "
            f"draft_n_rs_seq={self.draft_context.n_rs_seq()}, "
            f"draft_checkpoint="
            f"{'native-rs' if self._use_native_draft_rollback else 'on-device'}, "
            f"outputs={int(target_context_params.n_outputs_max)}/"
            f"{int(target_context_params.n_outputs_max_per_seq)}",
            file=sys.stderr,
        )

    def checkpoint(self, seq_id: int = 0) -> Any:
        """Capture the draft position, pending hidden row, and recurrent state."""
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")

        started = time.perf_counter()
        try:
            position = self.draft_context.memory_seq_pos_max(seq_id)
            checkpoint: Dict[str, Any] = {
                "position": position,
                "mode": "native",
                "buffer": None,
                "size": 0,
                "flags": 0,
                "pending_h": self.pending_h.copy(),
            }
            if not self._use_native_draft_rollback:
                from llama_cpp import llama_cpp as llama_cpp_lib

                # Only one speculative checkpoint per sequence is live. Keep
                # tensor payloads in llama_context-owned device buffers and
                # retain just the small serialized metadata buffer in Python.
                flags = (
                    llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
                    | llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
                )
                size = self.draft_context.get_state_seq_size_ext(seq_id, flags)
                if size <= 0:
                    raise RuntimeError("MTP draft context returned an empty checkpoint")
                buffer = (ctypes.c_uint8 * size)()
                written = self.draft_context.get_state_seq_data_ext(
                    buffer, size, seq_id, flags
                )
                if written != size:
                    raise RuntimeError(
                        f"MTP draft checkpoint write was incomplete: {written}/{size}"
                    )
                checkpoint.update(
                    mode="on-device",
                    buffer=buffer,
                    size=size,
                    flags=flags,
                )
                self._checkpoint_stats["device_captures"] += 1
                self._checkpoint_stats["buffer_bytes"] += size
            else:
                self._checkpoint_stats["native_captures"] += 1
            return checkpoint
        finally:
            self._checkpoint_stats["captures"] += 1
            self._checkpoint_stats["capture_seconds"] += (
                time.perf_counter() - started
            )

    def take_verification_checkpoint(self, seq_id: int = 0) -> Any:
        """Reuse the checkpoint restored after drafting, or capture a new one."""
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        checkpoint = self._pending_verification_checkpoint
        self._pending_verification_checkpoint = None
        if checkpoint is not None:
            self._checkpoint_stats["verification_reuses"] += 1
            return checkpoint
        return self.checkpoint(seq_id)

    def restore(self, checkpoint: Any, seq_id: int = 0) -> None:
        """Restore draft recurrent/KV state and the pending hidden row."""
        if checkpoint is None:
            return
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")

        started = time.perf_counter()
        try:
            if checkpoint["mode"] == "on-device":
                size = int(checkpoint["size"])
                read = self.draft_context.set_state_seq_data_ext(
                    checkpoint["buffer"], size, seq_id, checkpoint["flags"]
                )
                if read != size:
                    raise RuntimeError(
                        f"MTP draft checkpoint restore was incomplete: {read}/{size}"
                    )
                self._checkpoint_stats["device_restores"] += 1
            else:
                if not self.draft_context.memory_seq_rm(
                    seq_id, int(checkpoint["position"]) + 1, -1
                ):
                    raise RuntimeError("MTP native draft-context rollback failed")
                self._checkpoint_stats["native_restores"] += 1

            self.pending_h[:] = checkpoint["pending_h"]
            self.verify_h = np.empty((0, self.n_embd), dtype=np.float32)
            self.verify_tokens.clear()
            self.verify_positions.clear()
        finally:
            self._checkpoint_stats["restores"] += 1
            self._checkpoint_stats["restore_seconds"] += (
                time.perf_counter() - started
            )

    def reset_checkpoint_stats(self) -> None:
        """Reset MTP checkpoint, restore, and rollback counters."""
        self._checkpoint_stats: Dict[str, Union[int, float]] = {
            "captures": 0,
            "restores": 0,
            "verification_reuses": 0,
            "native_captures": 0,
            "native_restores": 0,
            "device_captures": 0,
            "device_restores": 0,
            "native_verification_rollbacks": 0,
            "buffer_bytes": 0,
            "capture_seconds": 0.0,
            "restore_seconds": 0.0,
        }

    def checkpoint_stats(self) -> Dict[str, Union[int, float]]:
        """Return a snapshot of the current MTP checkpoint counters."""
        return dict(self._checkpoint_stats)

    def truncate(self, position: int, seq_id: int = 0) -> None:
        """Remove a non-recurrent draft-cache suffix from ``position`` onward."""
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        if self.draft_model.is_recurrent() or self.draft_model.is_hybrid():
            raise RuntimeError(
                "Recurrent MTP draft contexts must be restored from a checkpoint"
            )
        if not self.draft_context.memory_seq_rm(seq_id, position, -1):
            raise RuntimeError("MTP draft-context truncation failed")

    def clear(self) -> None:
        """Clear request-local MTP state while keeping native resources loaded."""
        if self._closed:
            return
        self._pending_verification_checkpoint = None
        self.draft_context.memory_clear(True)
        self.pending_h.fill(0.0)
        self.verify_h = np.empty((0, self.n_embd), dtype=np.float32)
        self.verify_tokens.clear()
        self.verify_positions.clear()
        self._reset_backend_sampler()

    def close(self) -> None:
        """Idempotently release MTP batches, context, sampler, and owned model."""
        if self._closed:
            return
        self._closed = True
        self._pending_verification_checkpoint = None
        errors: List[Exception] = []

        self._close_draft_resources(
            errors, batch_names=("batch",), close_model=self._owns_model
        )

        if errors:
            raise errors[0]

    def process(self, batch: Any, seq_id: int = 0) -> None:
        """Consume target token rows and their already-computed NextN states."""
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        n_tokens = int(batch.n_tokens)
        if n_tokens <= 0 or not bool(batch.token) or bool(batch.embd):
            return
        positions = [int(batch.pos[i]) for i in range(n_tokens)]
        tokens = [int(batch.token[i]) for i in range(n_tokens)]
        # MTP currently supports one sequence and target nextn embeddings are
        # unmasked, so output rows are contiguous in physical batch order. Fetch
        # the dense block once instead of crossing ctypes once per row. This also
        # matches the catch-up input path in common/speculative.cpp.
        target_h = self._copy_rows(
            self.target_context.get_embeddings_nextn(), n_tokens, self.n_embd
        )

        self._process_target_rows(tokens, positions, target_h)

    def _process_target_rows(
        self,
        tokens: Sequence[int],
        positions: Sequence[int],
        target_h: npt.NDArray[np.float32],
    ) -> None:
        """Catch the draft context up using already-computed target hidden rows."""
        if len(tokens) == 0:
            return

        if not self.is_mem_shared:
            shifted_h = np.empty_like(target_h)
            shifted_h[0] = self.pending_h
            if len(tokens) > 1:
                shifted_h[1:] = target_h[:-1]

            self.batch.reset()
            for token, pos, hidden in zip(tokens, positions, shifted_h):
                self.batch.add_token_embedding(token, hidden, pos, [0], False)

            try:
                for head in range(self.n_mtp_layers):
                    if self.chain_heads:
                        if not self.draft_context.memory_seq_rm(
                            0, positions[0], -1
                        ):
                            raise RuntimeError(
                                "MTP chained-head catch-up rollback failed"
                            )
                        self.draft_context.set_nextn_layer_offset(head)
                    status = self.draft_context.decode(self.batch)
                    if status != 0:
                        raise RuntimeError(
                            f"MTP catch-up decode failed with status {status}"
                        )
            finally:
                if self.chain_heads:
                    self.draft_context.set_nextn_layer_offset(0)

        self.verify_h = target_h
        self.verify_tokens = list(tokens)
        self.verify_positions = list(positions)
        self.pending_h[:] = target_h[-1]

    def can_follow_target_native_rollback(self) -> bool:
        """Report that MTP can realign after target recurrent-state rollback."""
        return True

    def rollback_verified(
        self, checkpoint: Any, n_accepted: int, seq_id: int = 0
    ) -> None:
        """Replay only the sampled token and accepted draft prefix after RS rollback."""
        count = min(1 + max(0, int(n_accepted)), len(self.verify_tokens))
        tokens = self.verify_tokens[:count]
        positions = self.verify_positions[:count]
        target_h = self.verify_h[:count].copy()
        if self._use_native_draft_rollback and tokens:
            # Target verification has already advanced the draft context through
            # every proposed token. Native snapshots let us discard only the
            # rejected suffix, avoiding a full restore and accepted-prefix replay.
            if not self.is_mem_shared and not self.draft_context.memory_seq_rm(
                seq_id, positions[-1] + 1, -1
            ):
                raise RuntimeError("MTP native verification rollback failed")
            self.pending_h[:] = target_h[-1]
            self._checkpoint_stats["native_verification_rollbacks"] = (
                int(self._checkpoint_stats.get("native_verification_rollbacks", 0))
                + 1
            )
        else:
            self.restore(checkpoint, seq_id)
            if tokens:
                self._process_target_rows(tokens, positions, target_h)

    def draft(
        self,
        input_ids: Sequence[int],
        *,
        n_past: int,
        id_last: int,
        n_max: int,
        seq_id: int = 0,
    ) -> npt.NDArray[np.intc]:
        """Generate and then roll back an MTP continuation of up to ``n_max``."""
        _ = input_ids
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        limit = min(n_max, self.config.draft_n_max)
        if self.chain_heads:
            limit = min(limit, self.n_mtp_layers)
        if limit <= 0:
            return np.empty(0, dtype=np.intc)

        result: List[int] = []
        self._pending_verification_checkpoint = None
        context_checkpoint = self.checkpoint(seq_id)
        chain_h: List[npt.NDArray[np.float32]] = [self.pending_h.copy()]
        self.batch.reset()
        self.batch.add_token_embedding(id_last, self.pending_h, n_past, [0], True)
        output_index = 0

        try:
            for step in range(limit):
                if self.chain_heads:
                    if not self.draft_context.memory_seq_rm(0, n_past, -1):
                        raise RuntimeError("MTP chained-head draft rollback failed")
                    self.draft_context.set_nextn_layer_offset(step)

                status = self.draft_context.decode(self.batch)
                if status != 0:
                    raise RuntimeError(
                        f"MTP draft decode failed at step {step} with status {status}"
                    )
                token, probability = self._candidate(output_index)
                hidden = self._copy_rows(
                    self.draft_context.get_embeddings_nextn_ith(output_index),
                    1,
                    self.n_embd,
                )[0]
                if probability < self.config.draft_p_min:
                    break
                result.append(token)
                if len(result) >= limit:
                    break

                self.batch.reset()
                if self.chain_heads:
                    chain_h.append(hidden.copy())
                    for row, row_hidden in enumerate(chain_h):
                        row_token = id_last if row == 0 else result[row - 1]
                        self.batch.add_token_embedding(
                            row_token,
                            row_hidden,
                            n_past + row,
                            [0],
                            row == len(chain_h) - 1,
                        )
                    output_index = len(chain_h) - 1
                else:
                    next_pos = n_past if self.is_mem_shared else n_past + step + 1
                    self.batch.add_token_embedding(token, hidden, next_pos, [0], True)
                    output_index = 0
        finally:
            if self.chain_heads:
                self.draft_context.set_nextn_layer_offset(0)
            self.restore(context_checkpoint, seq_id)

        if len(result) < self.config.draft_n_min:
            result.clear()
        if result:
            # The draft branch was restored to this exact state above. Reuse the
            # same checkpoint for the upcoming target verification instead of
            # capturing an identical state a second time.
            self._pending_verification_checkpoint = context_checkpoint
        return np.asarray(result, dtype=np.intc)

    def accept(self, n_accepted: int, seq_id: int = 0) -> None:
        """Advance the pending hidden row to the accepted verification prefix."""
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        if self.verify_h.shape[0] == 0:
            return
        row = min(max(0, int(n_accepted)), self.verify_h.shape[0] - 1)
        self.pending_h[:] = self.verify_h[row]


class LlamaDFlashDecoding(_LlamaModelDraftEngine):
    """Python orchestration for llama.cpp's DFlash and DSpark draft graphs.

    Both algorithms extract configured target-layer inputs, fuse them with the
    draft encoder, inject the fused rows into the draft KV cache, and decode a
    non-causal mask block in one pass.  DSpark reuses the same graph and cache
    path but reads its Markov/confidence outputs differently.

    This initial implementation intentionally accepts token batches only.  An
    MTMD target uses M-RoPE position tuples while the current DFlash draft
    context uses scalar positions, so silently forwarding embedding batches
    would leave the draft cache incomplete or invalid.
    """

    def __init__(
        self,
        config: SpecConfig,
        *,
        target_model: Any,
        target_context: Any,
        model_params: Any,
        context_params: Any,
        verbose: bool = True,
    ) -> None:
        """Load a DFlash/DSpark sidecar and create its three-stage draft graph."""
        from llama_cpp import _internals as internals
        from llama_cpp import llama_cpp as llama_cpp_lib

        if config.spec_type not in {
            SpeculativeType.DRAFT_DFLASH,
            SpeculativeType.DRAFT_DSPARK,
        }:
            raise ValueError(
                "LlamaDFlashDecoding requires DRAFT_DFLASH or DRAFT_DSPARK"
            )
        config.validate()

        self._init_model_draft_engine(
            config,
            target_model=target_model,
            target_context=target_context,
            llama_cpp_lib=llama_cpp_lib,
            verbose=verbose,
        )
        self.is_dspark = config.spec_type == SpeculativeType.DRAFT_DSPARK
        self.draft_model = None
        self.draft_context = None
        self.noise_batch = None
        self.encoder_batch = None
        self.inject_batch = None

        self.draft_model = self._load_draft_model(
            internals, model_params, load_mtp=False
        )
        try:
            self.target_layer_ids = self.draft_model.target_layer_ids()
            if not self.target_layer_ids:
                raise ValueError("DFlash draft model has no target_layer_ids")
            self._validate_target_layer_ids(
                self.target_layer_ids, self.target_model.n_layer()
            )

            self.n_embd_tgt = self.target_model.n_embd()
            self.n_embd_dec = self.draft_model.n_embd()
            self.n_embd_enc = len(self.target_layer_ids) * self.n_embd_tgt
            self.n_vocab = self.draft_model.n_vocab()
            self.mask_token_id = self.draft_model.token_mask()
            if self.mask_token_id == llama_cpp_lib.LLAMA_TOKEN_NULL:
                raise ValueError("DFlash draft model has no mask token")

            metadata = self.draft_model.metadata()
            self.block_size = self._metadata_int(
                metadata, "dflash.block_size", 16
            )
            if self.block_size <= 1:
                raise ValueError(
                    f"DFlash block size must be greater than one: {self.block_size}"
                )
            self.sample_from_anchor = self._metadata_bool(
                metadata, "dflash.sample_from_anchor", True
            )
            trained_limit = (
                self.block_size
                if self.is_dspark and self.sample_from_anchor
                else self.block_size - 1
            )
            self.draft_limit = min(config.draft_n_max, trained_limit)
            if self.draft_limit <= 0:
                raise ValueError("DFlash resolved draft length must be positive")

            block_capacity = self.draft_limit + 1
            draft_ctx_params = self._build_draft_context_params(
                context_params, target_context
            )
            draft_ctx_params.ctx_other = target_context.ctx
            draft_ctx_params.n_seq_max = 1
            # Recurrent/hybrid sidecars need one native snapshot per possible
            # speculative position.  Non-recurrent DFlash sidecars clamp this
            # request to zero and use ordinary native KV/SWA suffix removal.
            draft_ctx_params.n_rs_seq = max(0, int(self.draft_limit))
            draft_ctx_params.n_batch = max(
                int(draft_ctx_params.n_batch), block_capacity
            )
            draft_ctx_params.n_ubatch = max(
                int(draft_ctx_params.n_ubatch), block_capacity
            )
            draft_ctx_params.n_outputs_max = block_capacity
            # Keeping a complete block in one sequence prevents the backend
            # from splitting the non-causal DFlash verification graph.
            draft_ctx_params.n_outputs_max_per_seq = block_capacity
            self.draft_context = internals.LlamaContext(
                model=self.draft_model,
                params=draft_ctx_params,
                verbose=verbose,
            )
            self.noise_batch = internals.LlamaBatch(
                n_tokens=block_capacity,
                embd=0,
                n_seq_max=1,
                verbose=verbose,
            )
            self.encoder_batch = internals.LlamaBatch(
                n_tokens=self.draft_context.n_ubatch(),
                embd=self.n_embd_enc,
                n_seq_max=1,
                verbose=verbose,
            )
            self.inject_batch = internals.LlamaBatch(
                n_tokens=self.draft_context.n_ubatch(),
                embd=self.n_embd_dec,
                n_seq_max=1,
                verbose=verbose,
            )

            self._enable_backend_sampling(internals)

            for layer_id in self.target_layer_ids:
                self.target_context.set_embeddings_layer_inp(layer_id, True)
            self.draft_context.set_embeddings_nextn(True, masked=True)
            self.draft_context.set_causal_attn(False)

            self._use_native_draft_rollback = (
                not (
                    self.draft_model.is_recurrent()
                    or self.draft_model.is_hybrid()
                )
                or self.draft_context.n_rs_seq() >= self.draft_limit
            )
            self._pending_verification_checkpoint = None
            self._active_verification_checkpoint = None
            self.verify_positions: List[int] = []
            self.verify_fused = np.empty((0, self.n_embd_dec), dtype=np.float32)
            self.reset_checkpoint_stats()
        except BaseException:
            self.close()
            raise

        if self.verbose:
            self._print_runtime_configuration()

    @staticmethod
    def _metadata_int(metadata: Dict[str, str], key: str, default: int) -> int:
        """Read an integer GGUF metadata value with a fallback default."""
        value = metadata.get(key)
        return default if value is None else int(value)

    @staticmethod
    def _validate_target_layer_ids(
        layer_ids: Sequence[int], n_layers: int
    ) -> None:
        """Validate target taps, including the final head-input tap at n_layers."""
        for layer_id in layer_ids:
            if layer_id < 0 or layer_id > n_layers:
                raise ValueError(
                    "DFlash target layer is outside the target model: "
                    f"{layer_id} not in [0, {n_layers}]"
                )

    @staticmethod
    def _metadata_bool(metadata: Dict[str, str], key: str, default: bool) -> bool:
        """Read and strictly validate a boolean GGUF metadata value."""
        value = metadata.get(key)
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean GGUF metadata {key}={value!r}")

    def _print_runtime_configuration(self) -> None:
        """Print resolved DFlash/DSpark model, block, and context settings."""
        algorithm = "DSpark" if self.is_dspark else "DFlash"
        devices = ",".join(self.config.draft_devices) or "auto"
        print(
            f"LlamaDFlashDecoding: {algorithm} speculative decoding enabled",
            file=sys.stderr,
        )
        print(
            "LlamaDFlashDecoding: "
            f"type={self.config.spec_type.to_str()}, "
            f"model={self.config.draft_model_path!r}, "
            f"draft_n_min={self.config.draft_n_min}, "
            f"draft_n_max={self.config.draft_n_max}, "
            f"resolved_n_max={self.draft_limit}, "
            f"draft_p_min={self.config.draft_p_min:g}",
            file=sys.stderr,
        )
        print(
            "LlamaDFlashDecoding: "
            f"block_size={self.block_size}, mask_token={self.mask_token_id}, "
            f"sample_from_anchor={self.sample_from_anchor}, "
            f"target_layers={self.target_layer_ids}, "
            f"feature_width={self.n_embd_enc}",
            file=sys.stderr,
        )
        print(
            "LlamaDFlashDecoding: "
            f"backend_sampling=requested:{self.config.draft_backend_sampling}/"
            f"active:{self._backend_sampling}, devices={devices}, "
            f"draft_n_ctx={self.draft_context.n_ctx()}, "
            f"draft_n_batch={self.draft_context.n_batch()}, "
            f"draft_n_ubatch={self.draft_context.n_ubatch()}, "
            f"draft_n_rs_seq={self.draft_context.n_rs_seq()}, "
            f"draft_checkpoint="
            f"{'native' if self._use_native_draft_rollback else 'on-device'}, "
            "multimodal=text-only",
            file=sys.stderr,
        )

    def begin(self, prompt_tokens: Sequence[int], seq_id: int = 0) -> None:
        """Verify that target processing populated the complete draft prompt."""
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        if prompt_tokens:
            pos_max = self.draft_context.memory_seq_pos_max(seq_id)
            if pos_max < len(prompt_tokens) - 1:
                raise RuntimeError(
                    "DFlash draft cache did not process the complete prompt: "
                    f"pos_max={pos_max}, expected at least {len(prompt_tokens) - 1}"
                )

    def _gather_target_features(
        self, *, batch_rows: int, offset: int, count: int
    ) -> npt.NDArray[np.float32]:
        """Concatenate configured target-layer inputs for a batch slice."""
        features = np.empty((count, self.n_embd_enc), dtype=np.float32)
        for layer_index, layer_id in enumerate(self.target_layer_ids):
            pointer = self.target_context.get_embeddings_layer_inp(layer_id)
            rows = np.ctypeslib.as_array(
                pointer, shape=(batch_rows * self.n_embd_tgt,)
            ).reshape(batch_rows, self.n_embd_tgt)
            column = layer_index * self.n_embd_tgt
            features[:, column : column + self.n_embd_tgt] = rows[
                offset : offset + count
            ]
        return features

    def process(self, batch: Any, seq_id: int = 0) -> None:
        """Fuse target-layer rows and inject them into the draft context cache."""
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        n_tokens = int(batch.n_tokens)
        if n_tokens <= 0:
            return
        if bool(batch.embd):
            raise NotImplementedError(
                "DFlash/DSpark speculative decoding is currently text-only; "
                "MTMD embedding batches require scalar draft-position remapping"
            )
        if not bool(batch.token):
            return

        for i in range(n_tokens):
            if int(batch.n_seq_id[i]) != 1 or int(batch.seq_id[i][0]) != seq_id:
                raise NotImplementedError(
                    "DFlash speculative decoding currently supports one sequence"
                )

        chunk_size = self.draft_context.n_ubatch()
        fused_chunks: List[npt.NDArray[np.float32]] = []
        for offset in range(0, n_tokens, chunk_size):
            count = min(chunk_size, n_tokens - offset)
            features = self._gather_target_features(
                batch_rows=n_tokens, offset=offset, count=count
            )

            self.encoder_batch.reset()
            self.encoder_batch.add_embeddings(
                features.reshape(-1),
                pos_array=list(range(count)),
                seq_ids=[0],
                logits_array=[True] * count,
            )
            self.draft_context.encode(self.encoder_batch)
            fused = self._copy_rows(
                self.draft_context.get_embeddings_nextn(),
                count,
                self.n_embd_dec,
            )
            fused_chunks.append(fused)

            positions = [int(batch.pos[offset + i]) for i in range(count)]
            self._inject_fused_rows(fused, positions, seq_id=seq_id)

        # Match the server's context-switch boundary: all injected rows must be
        # complete before draft() reads the cache on a potentially other backend.
        self.draft_context.synchronize()
        self.verify_positions = [int(batch.pos[i]) for i in range(n_tokens)]
        self.verify_fused = np.concatenate(fused_chunks, axis=0)

    def _inject_fused_rows(
        self,
        fused: npt.NDArray[np.float32],
        positions: Sequence[int],
        *,
        seq_id: int,
    ) -> None:
        """Inject already-fused target rows, chunking at the draft ubatch."""
        if len(positions) != len(fused):
            raise ValueError(
                "DFlash fused row/position mismatch: "
                f"{len(fused)} != {len(positions)}"
            )
        chunk_size = self.draft_context.n_ubatch()
        for offset in range(0, len(positions), chunk_size):
            count = min(chunk_size, len(positions) - offset)
            self.inject_batch.reset()
            self.inject_batch.add_embeddings(
                fused[offset : offset + count].reshape(-1),
                pos_array=positions[offset : offset + count],
                seq_ids=[seq_id],
            )
            status = self.draft_context.decode(self.inject_batch)
            if status != 0:
                raise RuntimeError(
                    "DFlash KV injection failed with status "
                    f"{status} at fused row offset {offset}"
                )

    def checkpoint(self, seq_id: int = 0) -> Any:
        """Capture the draft position and recurrent state before a noise block."""
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        started = time.perf_counter()
        try:
            checkpoint: Dict[str, Any] = {
                "position": self.draft_context.memory_seq_pos_max(seq_id),
                "mode": "native",
                "buffer": None,
                "size": 0,
                "flags": 0,
            }
            if self._use_native_draft_rollback:
                self._checkpoint_stats["native_captures"] += 1
            else:
                flags = (
                    self._llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
                    | self._llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
                )
                size = self.draft_context.get_state_seq_size_ext(seq_id, flags)
                if size <= 0:
                    raise RuntimeError(
                        "DFlash draft context returned an empty checkpoint"
                    )
                buffer = (ctypes.c_uint8 * size)()
                written = self.draft_context.get_state_seq_data_ext(
                    buffer, size, seq_id, flags
                )
                if written != size:
                    raise RuntimeError(
                        "DFlash draft checkpoint write was incomplete: "
                        f"{written}/{size}"
                    )
                checkpoint.update(
                    mode="on-device", buffer=buffer, size=size, flags=flags
                )
                self._checkpoint_stats["device_captures"] += 1
                self._checkpoint_stats["buffer_bytes"] += size
            return checkpoint
        finally:
            self._checkpoint_stats["captures"] += 1
            self._checkpoint_stats["capture_seconds"] += (
                time.perf_counter() - started
            )

    def take_verification_checkpoint(self, seq_id: int = 0) -> Any:
        """Select and retain the checkpoint used by target verification."""
        checkpoint = self._pending_verification_checkpoint
        self._pending_verification_checkpoint = None
        if checkpoint is not None:
            self._checkpoint_stats["verification_reuses"] += 1
        else:
            checkpoint = self.checkpoint(seq_id)
        self._active_verification_checkpoint = checkpoint
        return checkpoint

    def restore(self, checkpoint: Any, seq_id: int = 0) -> None:
        """Restore draft state and reclaim transient noise-block cache cells."""
        if checkpoint is None:
            return
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        started = time.perf_counter()
        try:
            if checkpoint["mode"] == "on-device":
                size = int(checkpoint["size"])
                read = self.draft_context.set_state_seq_data_ext(
                    checkpoint["buffer"], size, seq_id, checkpoint["flags"]
                )
                if read != size:
                    raise RuntimeError(
                        "DFlash draft checkpoint restore was incomplete: "
                        f"{read}/{size}"
                    )
                # A partial state restore does not reclaim transient SWA/KV
                # cells created by the noise block.  Match llama.cpp server:
                # restore the saved state first, then remove its draft suffix.
                if not self.draft_context.memory_seq_rm(
                    seq_id, int(checkpoint["position"]) + 1, -1
                ):
                    raise RuntimeError(
                        "DFlash device-checkpoint suffix cleanup failed"
                    )
                self._checkpoint_stats["device_restores"] += 1
            else:
                if not self.draft_context.memory_seq_rm(
                    seq_id, int(checkpoint["position"]) + 1, -1
                ):
                    raise RuntimeError("DFlash native draft-context rollback failed")
                self._checkpoint_stats["native_restores"] += 1
            self.verify_positions.clear()
            self.verify_fused = np.empty((0, self.n_embd_dec), dtype=np.float32)
            self._active_verification_checkpoint = None
        finally:
            self._checkpoint_stats["restores"] += 1
            self._checkpoint_stats["restore_seconds"] += (
                time.perf_counter() - started
            )

    def reset_checkpoint_stats(self) -> None:
        """Reset DFlash/DSpark checkpoint and rollback counters."""
        self._checkpoint_stats: Dict[str, Union[int, float]] = {
            "captures": 0,
            "restores": 0,
            "verification_reuses": 0,
            "native_captures": 0,
            "native_restores": 0,
            "device_captures": 0,
            "device_restores": 0,
            "native_verification_rollbacks": 0,
            "buffer_bytes": 0,
            "capture_seconds": 0.0,
            "restore_seconds": 0.0,
        }

    def checkpoint_stats(self) -> Dict[str, Union[int, float]]:
        """Return a snapshot of DFlash/DSpark checkpoint counters."""
        return dict(self._checkpoint_stats)

    def can_follow_target_native_rollback(self) -> bool:
        """Report support for realignment after target native rollback."""
        # Target and draft rollback capabilities are independent.  The target
        # context has recurrent snapshots sized by Llama for draft_n_max, while
        # a recurrent DFlash draft can restore its much smaller on-device
        # checkpoint and replay the accepted fused prefix.
        return True

    def rollback_verified(
        self, checkpoint: Any, n_accepted: int, seq_id: int = 0
    ) -> None:
        """Discard rejected rows or restore and replay the accepted fused prefix."""
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        count = min(1 + max(0, int(n_accepted)), len(self.verify_positions))
        if count <= 0:
            self.restore(checkpoint, seq_id)
            return
        if not self._use_native_draft_rollback:
            accepted_positions = self.verify_positions[:count]
            accepted_fused = self.verify_fused[:count].copy()
            self.restore(checkpoint, seq_id)
            self._inject_fused_rows(
                accepted_fused, accepted_positions, seq_id=seq_id
            )
            self.draft_context.synchronize()
            return
        position = self.verify_positions[count - 1] + 1
        if not self.draft_context.memory_seq_rm(seq_id, position, -1):
            raise RuntimeError("DFlash native verification rollback failed")
        self.verify_positions.clear()
        self.verify_fused = np.empty((0, self.n_embd_dec), dtype=np.float32)
        self._active_verification_checkpoint = None
        self._checkpoint_stats["native_verification_rollbacks"] += 1

    def truncate(self, position: int, seq_id: int = 0) -> None:
        """Truncate native cache state or rebuild a checkpointed accepted prefix."""
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        if self._use_native_draft_rollback:
            if not self.draft_context.memory_seq_rm(seq_id, position, -1):
                raise RuntimeError("DFlash draft-context truncation failed")
        else:
            checkpoint = self._active_verification_checkpoint
            if checkpoint is None:
                raise RuntimeError(
                    "DFlash draft-context truncation has no verification checkpoint"
                )
            keep = sum(pos < position for pos in self.verify_positions)
            accepted_positions = self.verify_positions[:keep]
            accepted_fused = self.verify_fused[:keep].copy()
            self.restore(checkpoint, seq_id)
            if keep:
                self._inject_fused_rows(
                    accepted_fused, accepted_positions, seq_id=seq_id
                )
                self.draft_context.synchronize()
        self.verify_positions.clear()
        self.verify_fused = np.empty((0, self.n_embd_dec), dtype=np.float32)
        self._active_verification_checkpoint = None

    def draft(
        self,
        input_ids: Sequence[int],
        *,
        n_past: int,
        id_last: int,
        n_max: int,
        seq_id: int = 0,
    ) -> npt.NDArray[np.intc]:
        """Decode one masked block and return its confidence-filtered proposals."""
        _ = input_ids
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        limit = min(max(0, int(n_max)), self.draft_limit)
        if limit <= 0:
            return np.empty(0, dtype=np.intc)

        n_block_tokens = limit + (
            0 if self.is_dspark and self.sample_from_anchor else 1
        )
        self.noise_batch.reset()
        tokens = [id_last] + [self.mask_token_id] * (n_block_tokens - 1)
        self.noise_batch.add_sequence(
            token_array=tokens,
            pos_array=[n_past + i for i in range(n_block_tokens)],
            seq_ids=[seq_id],
            logits_array=[True] * n_block_tokens,
        )

        self._pending_verification_checkpoint = None
        checkpoint = self.checkpoint(seq_id)
        result: List[int] = []
        try:
            status = self.draft_context.decode(self.noise_batch)
            if status != 0:
                pos_max = self.draft_context.memory_seq_pos_max(seq_id)
                raise RuntimeError(
                    "DFlash block decode failed with status "
                    f"{status}: n_past={n_past}, block_tokens={n_block_tokens}, "
                    f"draft_pos_max={pos_max}, draft_n_ctx="
                    f"{self.draft_context.n_ctx()}"
                )

            first_output = (
                0 if self.is_dspark and self.sample_from_anchor else 1
            )
            confidence = None
            if self.is_dspark and self.config.draft_p_min > 0.0:
                confidence = self._copy_rows(
                    self.draft_context.get_embeddings_nextn(),
                    n_block_tokens,
                    self.n_embd_dec,
                )[:, 0]

            for output_index in range(first_output, n_block_tokens):
                if (
                    confidence is not None
                    and float(confidence[output_index])
                    < self.config.draft_p_min
                ):
                    break
                token, probability = self._candidate(output_index)
                if (
                    not self.is_dspark
                    and probability < self.config.draft_p_min
                ):
                    break
                result.append(token)
        finally:
            self.restore(checkpoint, seq_id)

        if len(result) < self.config.draft_n_min:
            result.clear()
        if result:
            self._pending_verification_checkpoint = checkpoint
        return np.asarray(result, dtype=np.intc)

    def accept(self, n_accepted: int, seq_id: int = 0) -> None:
        """Finish verification and discard temporary fused-row bookkeeping."""
        _ = n_accepted
        if seq_id != 0:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports seq_id=0"
            )
        self.verify_positions.clear()
        self.verify_fused = np.empty((0, self.n_embd_dec), dtype=np.float32)
        self._active_verification_checkpoint = None

    def clear(self) -> None:
        """Clear request-local draft cache and verification bookkeeping."""
        if self._closed:
            return
        self._pending_verification_checkpoint = None
        self._active_verification_checkpoint = None
        self.verify_positions.clear()
        self.verify_fused = np.empty((0, self.n_embd_dec), dtype=np.float32)
        if self.draft_context is not None:
            self.draft_context.memory_clear(True)
        self._reset_backend_sampler()

    def close(self) -> None:
        """Idempotently release all DFlash/DSpark native resources."""
        if self._closed:
            return
        self._closed = True
        errors: List[Exception] = []

        self._close_draft_resources(
            errors,
            batch_names=("noise_batch", "encoder_batch", "inject_batch"),
            close_model=True,
        )

        if errors:
            raise errors[0]


def create_native_spec_engine(
    config: SpecConfig,
    *,
    target_model: Any,
    target_context: Any,
    model_params: Any,
    context_params: Any,
    verbose: bool = True,
) -> LlamaSpecEngine:
    """Create a model-backed engine bound to an initialized target context.

    ``native`` describes the engine's dependency on llama.cpp model/context
    objects; orchestration may still be implemented in Python. MTP reads target
    hidden states and either uses internal NextN heads or owns an external draft
    model/context. DFlash and DSpark own an external context that fuses selected
    target-layer inputs and generates a non-causal draft block.

    The caller retains ownership of ``target_model`` and ``target_context``.
    Any additional draft-side native resources created by the returned engine
    are released by its ``close()`` method.
    """
    if config.spec_type == SpeculativeType.DRAFT_MTP:
        return LlamaMTPDecoding(
            config,
            target_model=target_model,
            target_context=target_context,
            model_params=model_params,
            context_params=context_params,
            verbose=verbose,
        )
    if config.spec_type in {
        SpeculativeType.DRAFT_DFLASH,
        SpeculativeType.DRAFT_DSPARK,
    }:
        return LlamaDFlashDecoding(
            config,
            target_model=target_model,
            target_context=target_context,
            model_params=model_params,
            context_params=context_params,
            verbose=verbose,
        )
    raise NotImplementedError(f"Native engine {config.spec_type} is not implemented")
