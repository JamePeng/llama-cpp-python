import abc
import collections
import ctypes
import enum
import sys

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
    def from_str(cls, spec: str) -> SpeculativeType:
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


class LlamaSpeculativeEngine(abc.ABC):
    """Stateful speculative-decoding lifecycle shared by all backends.

    ``process`` consumes each successfully decoded target batch. ``draft`` then
    proposes candidates after the last verified token, and ``accept`` feeds the
    target verification result back into the backend.
    """

    def begin(self, prompt_tokens: Sequence[int], seq_id: int = 0) -> None:
        _ = prompt_tokens, seq_id

    def process(self, batch: Any, seq_id: int = 0) -> None:
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
        raise NotImplementedError()

    def accept(self, n_accepted: int, seq_id: int = 0) -> None:
        _ = n_accepted, seq_id

    def checkpoint(self, seq_id: int = 0) -> Any:
        _ = seq_id
        return None

    def restore(self, checkpoint: Any, seq_id: int = 0) -> None:
        _ = checkpoint, seq_id

    def supports_native_target_rollback(self) -> bool:
        return False

    def rollback_verified(
        self, checkpoint: Any, n_accepted: int, seq_id: int = 0
    ) -> None:
        _ = checkpoint, n_accepted, seq_id
        raise NotImplementedError(
            "This speculative engine cannot replay accepted verification rows"
        )

    def truncate(self, position: int, seq_id: int = 0) -> None:
        _ = position, seq_id

    def clear(self) -> None:
        pass

    def close(self) -> None:
        self.clear()


class LlamaNGramMapDecoding(LlamaSpeculativeEngine):
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
                Optional memory cap per n-gram key.
                When set, only the most recent entries are kept.
                For k4v mode, setting max_entries_per_key is strongly recommended.

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
            max_entries_per_key = 8
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


def create_spec_engine(config: SpecConfig) -> LlamaSpeculativeEngine:
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


class LlamaMTPDecoding(LlamaSpeculativeEngine):
    """Python orchestration of llama.cpp's native NextN/MTP graph."""

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
        from llama_cpp import _internals as internals
        from llama_cpp import llama_cpp as llama_cpp_lib

        if config.spec_type != SpeculativeType.DRAFT_MTP:
            raise ValueError("LlamaMTPDecoding requires spec_type=DRAFT_MTP")
        config.validate()

        self.config = config
        # close() can be reached from Llama.__del__ after Python has disabled
        # the import machinery. Keep the module imported during construction
        # instead of importing it lazily while tearing the engine down.
        self._llama_cpp_lib = llama_cpp_lib
        self.target_model = target_model
        self.target_context = target_context
        self.verbose = verbose
        self._closed = False
        self._owns_model = bool(config.draft_model_path)
        self._backend_sampler = None
        self._backend_sampling = False
        self._draft_devices = None
        self._draft_buft_patterns = None
        self._draft_buft_overrides = None

        if self._owns_model:
            draft_params = type(model_params).from_buffer_copy(model_params)
            draft_params.n_gpu_layers = config.resolved_draft_n_gpu_layers()
            draft_params.load_mtp = True

            if config.draft_devices:
                from llama_cpp._ggml import ggml_backend_dev_by_name

                DeviceArray = ctypes.c_void_p * (len(config.draft_devices) + 1)
                self._draft_devices = DeviceArray()
                for i, name in enumerate(config.draft_devices):
                    device = ggml_backend_dev_by_name(name.encode("utf-8"))
                    if not device:
                        raise ValueError(f"Unknown draft backend device: {name}")
                    self._draft_devices[i] = device
                self._draft_devices[-1] = None
                draft_params.devices = self._draft_devices

            if (
                config.draft_cpu_moe
                or config.draft_n_cpu_moe > 0
                or config.draft_tensor_buft_overrides
            ):
                from llama_cpp._ggml import ggml_backend_cpu_buffer_type

                override_values = list(config.draft_tensor_buft_overrides or [])
                if config.draft_cpu_moe:
                    patterns = [rb"\.ffn_(up|down|gate|gate_up)_(ch|)exps"]
                else:
                    patterns = [
                        f"blk\\.{i}".encode("utf-8")
                        + rb"\.ffn_(up|down|gate|gate_up)_(ch|)exps"
                        for i in range(config.draft_n_cpu_moe)
                    ]
                self._draft_buft_patterns = patterns
                cpu_buft = ggml_backend_cpu_buffer_type()
                override_type = llama_cpp_lib.llama_model_tensor_buft_override
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
            for name, value in config.draft_model_kwargs.items():
                if not hasattr(draft_params, name):
                    raise ValueError(f"Unknown draft model parameter: {name}")
                setattr(draft_params, name, value)
            self.draft_model = internals.LlamaModel(
                path_model=config.draft_model_path,
                params=draft_params,
                verbose=verbose,
            )
            try:
                self._validate_vocab_compatibility(target_model, self.draft_model)
            except BaseException:
                self.draft_model.close()
                raise
        else:
            self.draft_model = target_model

        try:
            draft_ctx_params = type(context_params).from_buffer_copy(context_params)
            draft_ctx_params.ctx_type = (
                llama_cpp_lib.llama_context_type.LLAMA_CONTEXT_TYPE_MTP
            )
            draft_ctx_params.ctx_other = target_context.ctx
            draft_ctx_params.n_rs_seq = 0
            # llama.cpp's draft context produces one output per sequence.  Keeping
            # this separate from the target verification limits avoids reserving
            # an unnecessary (1 + n_draft) output buffer in the MTP graph.
            draft_ctx_params.n_outputs_max = max(1, int(draft_ctx_params.n_seq_max))
            draft_ctx_params.n_outputs_max_per_seq = 1
            if config.draft_n_threads is not None:
                draft_ctx_params.n_threads = int(config.draft_n_threads)
            if config.draft_n_threads_batch is not None:
                draft_ctx_params.n_threads_batch = int(
                    config.draft_n_threads_batch
                )
            elif config.draft_n_threads is not None:
                draft_ctx_params.n_threads_batch = int(config.draft_n_threads)
            if config.draft_type_k is not None:
                draft_ctx_params.type_k = config.draft_type_k
            if config.draft_type_v is not None:
                draft_ctx_params.type_v = config.draft_type_v

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

            if config.draft_backend_sampling:
                backend_sampler = internals.LlamaSampler()
                backend_sampler.add_top_k(10)
                if llama_cpp_lib.llama_set_sampler(
                    self.draft_context.ctx, 0, backend_sampler.sampler
                ):
                    self._backend_sampler = backend_sampler
                    self._backend_sampling = True
                else:
                    backend_sampler.close()
        except BaseException:
            if self._backend_sampler is not None:
                if getattr(self, "draft_context", None) is not None:
                    llama_cpp_lib.llama_set_sampler(
                        self.draft_context.ctx, 0, None
                    )
                self._backend_sampler.close()
                self._backend_sampler = None
                self._backend_sampling = False
            if getattr(self, "draft_context", None) is not None:
                self.draft_context.close()
            if self._owns_model:
                self.draft_model.close()
            raise

        self.target_context.set_embeddings_nextn(True, masked=False)
        self.draft_context.set_embeddings_nextn(True, masked=True)
        self.pending_h = np.zeros(self.n_embd, dtype=np.float32)
        self.verify_h = np.empty((0, self.n_embd), dtype=np.float32)
        self.verify_tokens: List[int] = []
        self.verify_positions: List[int] = []

        if self.verbose:
            self._print_runtime_configuration(context_params)

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
            f"n_rs_seq={int(target_context_params.n_rs_seq)}, "
            f"outputs={int(target_context_params.n_outputs_max)}/"
            f"{int(target_context_params.n_outputs_max_per_seq)}",
            file=sys.stderr,
        )

    @staticmethod
    def _validate_vocab_compatibility(target_model: Any, draft_model: Any) -> None:
        """Mirror common_speculative_are_compatible vocab checks."""
        if target_model.vocab_type() != draft_model.vocab_type():
            raise ValueError("Draft and target models use different vocabulary types")

        for add_name, token_name in (
            ("get_add_bos", "token_bos"),
            ("get_add_eos", "token_eos"),
        ):
            target_add = bool(getattr(target_model, add_name)())
            draft_add = bool(getattr(draft_model, add_name)())
            if target_add != draft_add:
                raise ValueError(f"Draft and target models disagree on {add_name}")
            if target_add and getattr(target_model, token_name)() != getattr(
                draft_model, token_name
            )():
                raise ValueError(f"Draft and target models use different {token_name}")

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
            if target_model.token_get_text(token) != draft_model.token_get_text(token):
                raise ValueError(
                    f"Draft and target vocabularies differ at token {token}"
                )

    @staticmethod
    def _copy_rows(pointer: Any, rows: int, width: int) -> npt.NDArray[np.float32]:
        if not pointer:
            raise RuntimeError("speculative embedding output is unavailable")
        return np.ctypeslib.as_array(pointer, shape=(rows * width,)).reshape(
            rows, width
        ).copy()

    def _candidate(self, output_index: int) -> Tuple[int, float]:
        if self._backend_sampling:
            from llama_cpp import llama_cpp as llama_cpp_lib

            token = int(
                llama_cpp_lib.llama_get_sampled_token_ith(
                    self.draft_context.ctx, output_index
                )
            )
            candidates_count = int(
                llama_cpp_lib.llama_get_sampled_candidates_count_ith(
                    self.draft_context.ctx, output_index
                )
            )
            probs_count = int(
                llama_cpp_lib.llama_get_sampled_probs_count_ith(
                    self.draft_context.ctx, output_index
                )
            )
            logits_count = int(
                llama_cpp_lib.llama_get_sampled_logits_count_ith(
                    self.draft_context.ctx, output_index
                )
            )
            candidates_ptr = llama_cpp_lib.llama_get_sampled_candidates_ith(
                self.draft_context.ctx, output_index
            )
            probs_ptr = llama_cpp_lib.llama_get_sampled_probs_ith(
                self.draft_context.ctx, output_index
            )
            logits_ptr = llama_cpp_lib.llama_get_sampled_logits_ith(
                self.draft_context.ctx, output_index
            )

            if candidates_count > 0 and candidates_ptr:
                candidates = np.ctypeslib.as_array(
                    candidates_ptr, shape=(candidates_count,)
                )

                if probs_count > 0 and probs_ptr:
                    count = min(candidates_count, probs_count)
                    probs = np.ctypeslib.as_array(probs_ptr, shape=(count,))
                    if token == llama_cpp_lib.LLAMA_TOKEN_NULL:
                        selected = int(np.argmax(probs))
                        return int(candidates[selected]), float(probs[selected])
                    matches = np.flatnonzero(candidates[:count] == token)
                    if matches.size:
                        return token, float(probs[int(matches[0])])

                # A filter-only backend chain such as top-k returns compact
                # candidates/logits but no selected token or probabilities.
                # common_sampler_sample() finishes this selection on the CPU;
                # mirror that behavior instead of treating the compact logits
                # buffer as a full n_vocab row.
                if logits_count > 0 and logits_ptr:
                    count = min(candidates_count, logits_count)
                    logits = np.ctypeslib.as_array(logits_ptr, shape=(count,))
                    selected = int(np.argmax(logits))
                    shifted = logits.astype(np.float64) - float(np.max(logits))
                    probs = np.exp(shifted)
                    probability = float(probs[selected] / probs.sum())
                    selected_token = int(candidates[selected])
                    if token != llama_cpp_lib.LLAMA_TOKEN_NULL:
                        selected_token = token
                    return selected_token, probability

                # A backend may return only the selected token. This is enough
                # when p_min is disabled, otherwise use the CPU probability path.
            if (
                token != llama_cpp_lib.LLAMA_TOKEN_NULL
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

    def checkpoint(self, seq_id: int = 0) -> Any:
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")

        position = self.draft_context.memory_seq_pos_max(seq_id)
        native_state: Optional[bytes] = None
        if self.draft_model.is_recurrent() or self.draft_model.is_hybrid():
            from llama_cpp import llama_cpp as llama_cpp_lib

            flags = llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
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
            native_state = bytes(buffer)

        return {
            "position": position,
            "native_state": native_state,
            "pending_h": self.pending_h.copy(),
        }

    def restore(self, checkpoint: Any, seq_id: int = 0) -> None:
        if checkpoint is None:
            return
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")

        native_state = checkpoint["native_state"]
        if native_state is not None:
            from llama_cpp import llama_cpp as llama_cpp_lib

            flags = llama_cpp_lib.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
            buffer = (ctypes.c_uint8 * len(native_state)).from_buffer_copy(native_state)
            read = self.draft_context.set_state_seq_data_ext(
                buffer, len(native_state), seq_id, flags
            )
            if read != len(native_state):
                raise RuntimeError(
                    f"MTP draft checkpoint restore was incomplete: {read}/{len(native_state)}"
                )
        else:
            self.draft_context.memory_seq_rm(
                seq_id, int(checkpoint["position"]) + 1, -1
            )

        self.pending_h[:] = checkpoint["pending_h"]
        self.verify_h = np.empty((0, self.n_embd), dtype=np.float32)
        self.verify_tokens.clear()
        self.verify_positions.clear()

    def truncate(self, position: int, seq_id: int = 0) -> None:
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        if self.draft_model.is_recurrent() or self.draft_model.is_hybrid():
            raise RuntimeError(
                "Recurrent MTP draft contexts must be restored from a checkpoint"
            )
        self.draft_context.memory_seq_rm(seq_id, position, -1)

    def clear(self) -> None:
        if self._closed:
            return
        self.draft_context.memory_clear(True)
        self.pending_h.fill(0.0)
        self.verify_h = np.empty((0, self.n_embd), dtype=np.float32)
        self.verify_tokens.clear()
        self.verify_positions.clear()
        if self._backend_sampler is not None:
            self._backend_sampler.reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: List[Exception] = []

        if self._backend_sampler is not None:
            backend_sampler = self._backend_sampler
            self._backend_sampler = None
            self._backend_sampling = False
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

        if getattr(self, "batch", None) is not None:
            batch = self.batch
            self.batch = None
            try:
                batch.close()
            except Exception as exc:
                errors.append(exc)

        if getattr(self, "draft_context", None) is not None:
            draft_context = self.draft_context
            self.draft_context = None
            try:
                draft_context.close()
            except Exception as exc:
                errors.append(exc)

        if self._owns_model and getattr(self, "draft_model", None) is not None:
            draft_model = self.draft_model
            try:
                draft_model.close()
            except Exception as exc:
                errors.append(exc)
        self.draft_model = None

        if errors:
            raise errors[0]

    def process(self, batch: Any, seq_id: int = 0) -> None:
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
                        self.draft_context.memory_seq_rm(0, positions[0], -1)
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

    def supports_native_target_rollback(self) -> bool:
        return True

    def rollback_verified(
        self, checkpoint: Any, n_accepted: int, seq_id: int = 0
    ) -> None:
        """Replay only the sampled token and accepted draft prefix after RS rollback."""
        count = min(1 + max(0, int(n_accepted)), len(self.verify_tokens))
        tokens = self.verify_tokens[:count]
        positions = self.verify_positions[:count]
        target_h = self.verify_h[:count].copy()
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
        _ = input_ids
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        limit = min(n_max, self.config.draft_n_max)
        if self.chain_heads:
            limit = min(limit, self.n_mtp_layers)
        if limit <= 0:
            return np.empty(0, dtype=np.intc)

        result: List[int] = []
        context_checkpoint = self.checkpoint(seq_id)
        chain_h: List[npt.NDArray[np.float32]] = [self.pending_h.copy()]
        self.batch.reset()
        self.batch.add_token_embedding(id_last, self.pending_h, n_past, [0], True)
        output_index = 0

        try:
            for step in range(limit):
                if self.chain_heads:
                    self.draft_context.memory_seq_rm(0, n_past, -1)
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
        return np.asarray(result, dtype=np.intc)

    def accept(self, n_accepted: int, seq_id: int = 0) -> None:
        if seq_id != 0:
            raise NotImplementedError("MTP speculative decoding currently supports seq_id=0")
        if self.verify_h.shape[0] == 0:
            return
        row = min(max(0, int(n_accepted)), self.verify_h.shape[0] - 1)
        self.pending_h[:] = self.verify_h[row]


def create_native_spec_engine(
    config: SpecConfig,
    *,
    target_model: Any,
    target_context: Any,
    model_params: Any,
    context_params: Any,
    verbose: bool = True,
) -> LlamaSpeculativeEngine:
    """Create a model-backed engine bound to an initialized target context.

    ``native`` describes the engine's dependency on llama.cpp model/context
    objects; orchestration may still be implemented in Python. For MTP, the
    engine reads target hidden states and either uses the target model's internal
    NextN heads or creates and owns a separate draft model/context when
    ``config.draft_model_path`` is set.

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
    raise NotImplementedError(f"Native engine {config.spec_type} is not implemented")
