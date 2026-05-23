import abc
import collections

from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt


class LlamaDraftModel(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        raise NotImplementedError()


class LlamaNGramMapDecoding(LlamaDraftModel):
    """
    Fast model-free speculative decoder based on prompt n-gram lookup.

    It supports two modes:

    - "k":
        Key-only mode. Stores n-gram key -> history positions.
        This is memory-efficient and similar to llama.cpp's ngram-map-k behavior.

    - "k4v":
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
        mode: Literal["k", "k4v"] = "k",
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

            mode:
                "k" stores only matched positions.
                "k4v" stores matched continuation values directly.

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

        mode = mode.lower()
        if mode not in ("k", "k4v"):
            raise ValueError("mode must be either 'k' or 'k4v'")

        self.ngram_size = int(ngram_size)
        self.num_pred_tokens = int(num_pred_tokens)
        self.mode = mode
        self.min_hits = int(min_hits)
        self.sync_check_tokens = int(sync_check_tokens)

        if mode == "k4v" and max_entries_per_key is None:
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
        # tokens become available.
        self._map_k4v: DefaultDict[
            Tuple[int, ...], Dict[int, Tuple[int, ...]]
        ] = collections.defaultdict(dict)

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

    def accept(self, n_accepted: int) -> None:
        """
        Notify how many draft tokens were accepted by the target model.

        This implementation does not need to update internal state here, because the
        next call receives the verified token history through `input_ids`.

        The method is kept for API symmetry and future extensions, such as acceptance
        statistics, adaptive reset, or low-acceptance fallback.
        """
        return

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

            if self.mode == "k":
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

        if self.mode == "k":
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
                value_end = min(value_start + self.num_pred_tokens, len(self._history))

                if value_start >= value_end:
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

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
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
        _ = kwargs

        self._sync_and_index(input_ids)
        self._last_draft_len = 0

        if len(self._history) < self.ngram_size:
            return np.array([], dtype=np.intc)

        search_key = tuple(self._history[-self.ngram_size :])

        if self.mode == "k":
            positions = self._map_k.get(search_key)
            if not positions or len(positions) < self.min_hits:
                return np.array([], dtype=np.intc)

            # Use the latest valid match with an available continuation.
            draft: List[int] = []
            for pos in reversed(positions):
                start = pos + self.ngram_size
                if start < len(self._history):
                    end = min(start + self.num_pred_tokens, len(self._history))
                    draft = self._history[start:end]
                    break

        else:
            values = self._map_k4v.get(search_key)
            if not values or len(values) < self.min_hits:
                return np.array([], dtype=np.intc)

            # Use the continuation from the latest historical position.
            latest_pos = max(values)
            draft = list(values[latest_pos])

        self._last_draft_len = len(draft)
        return np.asarray(draft, dtype=np.intc)


# Legacy Numpy sliding window implementation
# Fast in some cases, but may degrade output quality.
# Not recommended for production.
class LlamaPromptLookupDecoding(LlamaDraftModel):
    """
    Stateless speculative decoding based on Numpy sliding window
    Warning: High computational overhead for long contexts.

    Based on https://github.com/apoorvumang/prompt-lookup-decoding
    """

    def __init__(self, max_ngram_size: int = 3, num_pred_tokens: int = 10):
        """
        Initializes the legacy sliding window speculative decoder.

        Args:
            max_ngram_size (int): The maximum n-gram size to search for. Defaults to 3.
            num_pred_tokens (int): The maximum number of tokens to predict. Defaults to 10.
        """
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens

    @staticmethod
    def find_candidate_pred_tokens(
        input_ids: npt.NDArray[np.intc],
        max_ngram_size: int,
        num_pred_tokens: int,
    ):
        """
        Linearly scans the input_ids using sliding windows to find pattern matches.

        Args:
            input_ids (npt.NDArray[np.intc]): The complete sequence of token IDs.
            max_ngram_size (int): Maximum size of the n-gram window.
            num_pred_tokens (int): Maximum draft tokens to return.

        Returns:
            npt.NDArray[np.intc]: The predicted draft tokens.
        """
        input_length = input_ids.shape[0]

        for ngram_size in range(min(max_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = np.lib.stride_tricks.sliding_window_view(input_ids, (ngram_size,))

            # Convert ngram to an array for comparison
            ngram_array = input_ids[-ngram_size:]

            # Find where the windows match the ngram
            matches = np.all(windows == ngram_array, axis=1)

            # Get the indices of matches
            match_indices = np.nonzero(matches)[0]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + num_pred_tokens
                end_idx = min(end_idx, input_length)

                if start_idx < end_idx:
                    return input_ids[start_idx:end_idx]

        # If no match is found, return an empty array
        return np.array([], dtype=np.intc)

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        """Generates draft tokens using the legacy sliding window search."""
        return self.find_candidate_pred_tokens(
            input_ids=input_ids,
            max_ngram_size=self.max_ngram_size,
            num_pred_tokens=self.num_pred_tokens,
        )
