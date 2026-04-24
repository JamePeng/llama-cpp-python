import abc
import collections

from typing import Any, Dict, List, Tuple

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
    Ultra-fast speculative decoder based on hash inverted index and incremental updates.
    O(1) time complexity, aligned with llama.cpp's underlying ngram-map algorithm.
    """

    def __init__(self, ngram_size: int = 3, num_pred_tokens: int = 10):
        """
        Initializes the N-Gram Map speculative decoder.

        Args:
            ngram_size (int): The length of the token sequence used as the search key.
                Larger values provide strictly accurate context matching but may result
                in fewer cache hits. Defaults to 3.
            num_pred_tokens (int): The maximum number of future tokens to draft (predict)
                and return once a match is found in the history. Defaults to 10.
        """
        self.ngram_size = ngram_size
        self.num_pred_tokens = num_pred_tokens

        # Core state cache
        # Mapping format: (token_1, ..., token_N) -> [index_1, index_2, ...]
        self._ngram_map: Dict[Tuple[int, ...], List[int]] = collections.defaultdict(list)
        self._history: List[int] = []

    def _update_cache(self, input_ids: npt.NDArray[np.intc]) -> None:
        """
        Smart state synchronization and incremental build (Extreme O(1) optimization).

        Args:
            input_ids (npt.NDArray[np.intc]): The complete sequence of current token IDs
                generated or processed so far.
        """
        new_len = len(input_ids)
        old_len = len(self._history)

        # Check if it's a perfect incremental append (verify if the previous token matches)
        is_incremental = False
        if new_len > old_len and old_len > 0:
            if self._history[-1] == input_ids[old_len - 1]:
                is_incremental = True

        if is_incremental:
            # Only extract, convert, and append new tokens.
            # Never copy or touch the entire historical array!
            new_tokens = input_ids[old_len:].tolist()
            self._history.extend(new_tokens)
            start_idx = max(0, old_len - self.ngram_size)
        else:
            # Rollback occurred (wrong prediction) or a completely new Prompt. Trigger full rebuild.
            self._ngram_map.clear()
            self._history = input_ids.tolist()
            start_idx = 0

        # Build/update the hash inverted index
        for i in range(start_idx, new_len - self.ngram_size):
            key = tuple(self._history[i : i + self.ngram_size])
            self._ngram_map[key].append(i)

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        """
        Generates draft tokens based on historical N-Gram frequency.

        Args:
            input_ids (npt.NDArray[np.intc]): The current sequence of token IDs.
            **kwargs: Additional generation arguments (ignored in this implementation).

        Returns:
            npt.NDArray[np.intc]: An array of predicted draft tokens. Returns an empty
            array if no matching context is found.
        """
        # 1. Ultra-fast state synchronization
        self._update_cache(input_ids)

        # 2. Cannot speculate if the history is too short
        if len(self._history) < self.ngram_size:
            return np.array([], dtype=np.intc)

        # 3. Extract the Search Key (the last N tokens)
        search_key = tuple(self._history[-self.ngram_size:])

        # 4. O(1) instant lookup
        match_indices = self._ngram_map.get(search_key)

        if not match_indices:
            return np.array([], dtype=np.intc)

        # 5. Get the context of the last match and extract draft tokens
        best_match_idx = match_indices[-1]
        draft_start = best_match_idx + self.ngram_size
        draft_end = min(draft_start + self.num_pred_tokens, len(self._history))

        return np.array(self._history[draft_start:draft_end], dtype=np.intc)


# Legacy Numpy sliding window implementation
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
