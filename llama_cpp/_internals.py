from __future__ import annotations

import ctypes
import enum
import os

from typing import (
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Sequence,
    Union,
    TYPE_CHECKING
)

from dataclasses import dataclass, field
from collections import deque
from contextlib import ExitStack

import numpy as np
import numpy.typing as npt

from .llama_types import *
from .llama_grammar import LlamaGrammar
from ._utils import suppress_stdout_stderr

import llama_cpp.llama_cpp as llama_cpp

if TYPE_CHECKING:
    from llama_cpp._ctypes_extensions import (
        CtypesArray,
        CtypesPointer,
    )

# Python wrappers over llama.h structs


class LlamaModel:
    """Intermediate Python wrapper for a llama.cpp llama_model.
    NOTE: For stability it's recommended you use the Llama class instead."""

    def __init__(
        self,
        *,
        path_model: str,
        params: llama_cpp.llama_model_params,
        verbose: bool = True,
    ):
        self.path_model = path_model
        self.params = params
        self.verbose = verbose
        self._exit_stack = ExitStack()

        model = None

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        with suppress_stdout_stderr(disable=verbose):
            model = llama_cpp.llama_model_load_from_file(
                self.path_model.encode("utf-8"), self.params
            )

        if model is None:
            raise ValueError(f"Failed to load model from file: {path_model}")

        vocab = llama_cpp.llama_model_get_vocab(model)

        if vocab is None:
            raise ValueError(f"Failed to get vocab from model: {path_model}")

        self.model = model
        self.vocab = vocab

    def close(self):
        """Manually free LlamaModel and Vocab resources."""
        if getattr(self, "model", None) is not None:
            try:
                llama_cpp.llama_model_free(self.model)
            except Exception:
                pass
            self.model = None
        self.vocab = None

        self._exit_stack.close()

    def __del__(self):
        self.close()

    def vocab_type(self) -> int:
        return llama_cpp.llama_vocab_type(self.model)

    def n_vocab(self) -> int:
        return llama_cpp.llama_n_vocab(self.vocab)

    def n_ctx_train(self) -> int:
        return llama_cpp.llama_model_n_ctx_train(self.model)

    def n_cls_out(self) -> int:
        return llama_cpp.llama_model_n_cls_out(self.model)

    def n_embd(self) -> int:
        return llama_cpp.llama_model_n_embd(self.model)

    def n_embd_inp(self) -> int:
        return llama_cpp.llama_model_n_embd_inp(self.model)

    def n_embd_out(self) -> int:
        return llama_cpp.llama_model_n_embd_out(self.model)

    def n_layer(self) -> int:
        return llama_cpp.llama_model_n_layer(self.model)

    def n_head(self) -> int:
        return llama_cpp.llama_model_n_head(self.model)

    def n_head_kv(self) -> int:
        return llama_cpp.llama_model_n_head_kv(self.model)

    def n_swa(self) -> int:
        return llama_cpp.llama_model_n_swa(self.model)

    def n_params(self) -> int:
        return llama_cpp.llama_model_n_params(self.model)

    def has_encoder(self) -> bool:
        return llama_cpp.llama_model_has_encoder(self.model)

    def has_decoder(self) -> bool:
        return llama_cpp.llama_model_has_decoder(self.model)

    def decoder_start_token(self) -> int:
        return llama_cpp.llama_model_decoder_start_token(self.model)

    def is_recurrent(self) -> bool:
        return llama_cpp.llama_model_is_recurrent(self.model)

    def is_hybrid(self) -> bool:
        return llama_cpp.llama_model_is_hybrid(self.model)

    def is_diffusion(self) -> bool:
        return llama_cpp.llama_model_is_diffusion(self.model)

    def rope_freq_scale_train(self) -> float:
        return llama_cpp.llama_model_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        buf = ctypes.create_string_buffer(1024)
        llama_cpp.llama_model_desc(self.model, buf, 1024)
        return buf.value.decode("utf-8")

    def size(self) -> int:
        return llama_cpp.llama_model_size(self.model)

    def get_tensor(self, name: str) -> ctypes.c_void_p:
        raise NotImplementedError("get_tensor is not implemented in llama.cpp")

    # Vocab

    def token_get_text(self, token: int) -> str:
        return llama_cpp.llama_vocab_get_text(self.vocab, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        return llama_cpp.llama_vocab_get_score(self.vocab, token)

    def token_get_attr(self, token: int) -> int:
        return llama_cpp.llama_vocab_get_attr(self.vocab, token)

    def token_is_eog(self, token: int) -> bool:
        return llama_cpp.llama_vocab_is_eog(self.vocab, token)

    def token_is_control(self, token: int) -> bool:
        return llama_cpp.llama_vocab_is_control(self.vocab, token)

    # Special tokens

    def token_bos(self) -> int:
        return llama_cpp.llama_vocab_bos(self.vocab)

    def token_eos(self) -> int:
        return llama_cpp.llama_vocab_eos(self.vocab)

    def token_eot(self) -> int:
        return llama_cpp.llama_vocab_eot(self.vocab)

    def token_sep(self) -> int:
        return llama_cpp.llama_vocab_sep(self.vocab)

    def token_nl(self) -> int:
        return llama_cpp.llama_vocab_nl(self.vocab)

    def token_pad(self) -> int:
        return llama_cpp.llama_vocab_pad(self.vocab)

    def token_mask(self) -> int:
        return llama_cpp.llama_vocab_mask(self.vocab)

    def token_cls(self) -> int:
        return llama_cpp.llama_vocab_cls(self.vocab)

    def token_fim_pre(self) -> int:
        return llama_cpp.llama_vocab_fim_pre(self.vocab)

    def token_fim_suf(self) -> int:
        return llama_cpp.llama_vocab_fim_suf(self.vocab)

    def token_fim_mid(self) -> int:
        return llama_cpp.llama_vocab_fim_mid(self.vocab)

    def token_fim_pad(self) -> int:
        return llama_cpp.llama_vocab_fim_pad(self.vocab)

    def token_fim_rep(self) -> int:
        return llama_cpp.llama_vocab_fim_rep(self.vocab)

    def token_fim_sep(self) -> int:
        return llama_cpp.llama_vocab_fim_sep(self.vocab)

    def get_add_bos(self) -> bool:
        return llama_cpp.llama_vocab_get_add_bos(self.vocab)

    def get_add_eos(self) -> bool:
        return llama_cpp.llama_vocab_get_add_eos(self.vocab)

    def get_add_sep(self) -> bool:
        return llama_cpp.llama_vocab_get_add_sep(self.vocab)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool):
        """
        Tokenize a string.
        Optimized to use dynamic buffer allocation.
        """
        n_tokens_alloc = len(text) + 2
        tokens = (llama_cpp.llama_token * n_tokens_alloc)()

        n_tokens = llama_cpp.llama_tokenize(
            self.vocab, text, len(text), tokens, n_tokens_alloc, add_bos, special
        )

        # If the buffer is insufficient (returns a negative number), reallocate the buffer.
        if n_tokens < 0:
            n_tokens_alloc = -n_tokens
            tokens = (llama_cpp.llama_token * n_tokens_alloc)()
            n_tokens = llama_cpp.llama_tokenize(
                self.vocab, text, len(text), tokens, n_tokens_alloc, add_bos, special
            )
            if n_tokens < 0:
                raise RuntimeError(
                    f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                )

        # return a buffer of n_tokens size.
        return list(tokens[:n_tokens])

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        """
        Convert a single token to bytes.
        Optimized to handle dynamic resizing for ultra-long tokens.
        """
        size = 32
        buf = (ctypes.c_char * size)()
        n = llama_cpp.llama_token_to_piece(self.vocab, token, buf, size, 0, special)

        # If the token is very long (returns a negative number), redistribute it according to the returned size.
        if n < 0:
            size = -n
            buf = (ctypes.c_char * size)()
            n = llama_cpp.llama_token_to_piece(self.vocab, token, buf, size, 0, special)
            if n < 0:
                raise RuntimeError(f"Failed to get piece for token {token}")

        # return a buffer of n size.
        return bytes(buf[:n])

    def detokenize(self, tokens: List[int], special: bool = False) -> bytes:
        """
        Convert a list of tokens to bytes.
        Optimized to handle dynamic resizing for ultra-long tokens.
        """
        if not tokens:
            return b""

        n_tokens = len(tokens)
        # Convert a Python list to a C int array
        tokens_array = (llama_cpp.llama_token * n_tokens)(*tokens)

        # Initial buffer size estimation
        buffer_size = max(n_tokens, 64)
        buffer = (ctypes.c_char * buffer_size)()

        n_chars = llama_cpp.llama_detokenize(
            self.vocab, tokens_array, n_tokens, buffer, buffer_size, False, special
        )

        # If the buffer is insufficient, expand it and retry.
        if n_chars < 0:
            buffer_size = -n_chars
            buffer = (ctypes.c_char * buffer_size)()
            n_chars = llama_cpp.llama_detokenize(
                self.vocab, tokens_array, n_tokens, buffer, buffer_size, False, special
            )
            if n_chars < 0:
                raise RuntimeError("Failed to detokenize")

        return bytes(buffer[:n_chars])


    # Extra
    def metadata(self) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        # Pre-allocate a 16KB buffer. This is large enough to handle almost all
        # metadata values (including gpt-oss large chat templates ~15KB) in a single pass,
        # eliminating the need for resize-and-retry in most cases.
        buffer_size = 16384
        buffer = ctypes.create_string_buffer(buffer_size)

        # Caching function references reduces the overhead of property lookups within loops.
        get_key_by_index = llama_cpp.llama_model_meta_key_by_index
        get_val_by_index = llama_cpp.llama_model_meta_val_str_by_index
        metadata_count = llama_cpp.llama_model_meta_count(self.model)
        # iterate over model keys
        for i in range(metadata_count):
            # 1. Get Key
            nbytes = get_key_by_index(self.model, i, buffer, buffer_size)
            # Handle buffer resize if the key exceeds current size
            if nbytes > buffer_size:
                buffer_size = nbytes + 1024
                buffer = ctypes.create_string_buffer(buffer_size)
                # Retry with the larger buffer
                nbytes = get_key_by_index(self.model, i, buffer, buffer_size)
            key = buffer.value.decode("utf-8")

            # 2. Get Value
            nbytes = get_val_by_index(self.model, i, buffer, buffer_size)
            # Handle buffer resize if the value exceeds current size
            if nbytes > buffer_size:
                buffer_size = nbytes + 1024
                buffer = ctypes.create_string_buffer(buffer_size)
                # Retry with the larger buffer
                nbytes = get_val_by_index(self.model, i, buffer, buffer_size)
            value = buffer.value.decode("utf-8")

            metadata[key] = value
        return metadata

    @staticmethod
    def default_params():
        """Get the default llama_model_params."""
        return llama_cpp.llama_model_default_params()


class LlamaContext:
    """Intermediate Python wrapper for a llama.cpp llama_context.
    NOTE: For stability it's recommended you use the Llama class instead."""

    def __init__(
        self,
        *,
        model: LlamaModel,
        params: llama_cpp.llama_context_params,
        verbose: bool = True,
    ):
        self.model = model
        self.params = params
        self.verbose = verbose
        self._exit_stack = ExitStack()

        ctx = llama_cpp.llama_init_from_model(self.model.model, self.params)

        if ctx is None:
            llama_cpp.llama_model_free(self.model.model)
            raise ValueError("Failed to create context with model")

        self.ctx = ctx

    def close(self):
        """Manually free LlamaContext resources."""
        if getattr(self, "ctx", None) is not None:
            try:
                llama_cpp.llama_free(self.ctx)
            except Exception:
                pass
            self.ctx = None

        self._exit_stack.close()

    def __del__(self):
        self.close()

    def n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx(self.ctx)

    def n_ctx_seq(self) -> int:
        return llama_cpp.llama_n_ctx_seq(self.ctx)

    def n_batch(self) -> int:
        return llama_cpp.llama_n_batch(self.ctx)

    def n_ubatch(self) -> int:
        return llama_cpp.llama_n_ubatch(self.ctx)

    def n_seq_max(self) -> int:
        return llama_cpp.llama_n_seq_max(self.ctx)

    def pooling_type(self) -> int:
        return llama_cpp.llama_pooling_type(self.ctx)

    # // Memory API

    def get_memory(self):
        return llama_cpp.llama_get_memory(self.ctx)

    def memory_clear(self, data: bool):
        llama_cpp.llama_memory_clear(self.get_memory(), data)

    def memory_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        if self.ctx is not None:
            return llama_cpp.llama_memory_seq_rm(self.get_memory(), seq_id, p0, p1)
        else:
            return False

    def memory_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        llama_cpp.llama_memory_seq_cp(self.get_memory(), seq_id_src, seq_id_dst, p0, p1)

    def memory_seq_keep(self, seq_id: int):
        llama_cpp.llama_memory_seq_keep(self.get_memory(), seq_id)

    def memory_seq_add(self, seq_id: int, p0: int, p1: int, delta: int):
        llama_cpp.llama_memory_seq_add(self.get_memory(), seq_id, p0, p1, delta)

    def memory_seq_div(self, seq_id: int, p0: int, p1: int, d: int):
        llama_cpp.llama_memory_seq_div(self.get_memory(), seq_id, p0, p1, d)

    def memory_seq_pos_max(self, seq_id: int) -> int:
        return llama_cpp.llama_memory_seq_pos_max(self.get_memory(), seq_id)

    def memory_seq_pos_min(self, seq_id: int) -> int:
        return llama_cpp.llama_memory_seq_pos_min(self.get_memory(), seq_id)

    # // State / sessions API

    def get_state_size(self) -> int:
        return llama_cpp.llama_state_get_size(self.ctx)

    def get_state_data(self, dst:ctypes.Array[ctypes.c_uint8], size: int) -> int:
        return llama_cpp.llama_state_get_data(self.ctx, dst, size)

    def set_state_data(self, src:ctypes.Array[ctypes.c_uint8], size: int) -> int:
        return llama_cpp.llama_state_set_data(self.ctx, src, size)

    def load_state_file(
        self,
        path_session: bytes,
        tokens_out: ctypes.Array[llama_cpp.llama_token],
        n_token_capacity: ctypes.c_size_t,
        n_token_count_out: CtypesPointer[ctypes.c_size_t]
    ) -> bool:
        return llama_cpp.llama_state_load_file(self.ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)

    def save_state_file(
        self,
        path_session: bytes,
        tokens: ctypes.Array[llama_cpp.llama_token],
        n_token_count: ctypes.c_size_t
    ) -> bool:
        return llama_cpp.llama_state_save_file(self.ctx, path_session, tokens, n_token_count)

    def get_state_seq_size(self, seq_id: int) -> int:
        return llama_cpp.llama_state_seq_get_size(self.ctx, seq_id)

    def get_state_seq_data(self, dst: ctypes.Array[ctypes.c_uint8], size: int, seq_id: int) -> int:
        return llama_cpp.llama_state_seq_get_data(self.ctx, dst, size, seq_id)

    def set_state_seq_data(self, src: ctypes.Array[ctypes.c_uint8], size: int, dest_seq_id: int) -> int:
        return llama_cpp.llama_state_seq_set_data(self.ctx, src, size, dest_seq_id)

    def load_state_seq_file(
        self,
        filepath: bytes,
        dest_seq_id: int,
        tokens_out: ctypes.Array[llama_cpp.llama_token],
        n_token_capacity: ctypes.c_size_t,
        n_token_count_out: CtypesPointer[ctypes.c_size_t]
    ) -> int:
        return llama_cpp.llama_state_seq_load_file(self.ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)

    def save_state_seq_file(
        self,
        filepath: bytes,
        seq_id: int,
        tokens: ctypes.Array[llama_cpp.llama_token],
        n_token_count: ctypes.c_size_t
    ) -> int:
        return llama_cpp.llama_state_seq_save_file(self.ctx, filepath, seq_id, tokens, n_token_count)

    def get_state_seq_size_ext(self, seq_id: int, flags: llama_cpp.llama_state_seq_flags) -> int:
        return llama_cpp.llama_state_seq_get_size_ext(self.ctx, seq_id, flags)

    def get_state_seq_data_ext(
        self,
        dst:ctypes.Array[ctypes.c_uint8],
        size: int,
        seq_id: int,
        flags: llama_cpp.llama_state_seq_flags
    ) -> int:
        return llama_cpp.llama_state_seq_get_data_ext(self.ctx, dst, size, seq_id, flags)

    def set_state_seq_data_ext(
        self,
        src:ctypes.Array[ctypes.c_uint8],
        size: int,
        dest_seq_id: int,
        flags: llama_cpp.llama_state_seq_flags
    ) -> int:
        return llama_cpp.llama_state_seq_set_data_ext(self.ctx, src, size, dest_seq_id, flags)

    # // Decoding API

    def encode(self, batch: LlamaBatch):
        return_code = llama_cpp.llama_encode(
            self.ctx,
            batch.batch,
        )
        if return_code != 0:
            raise RuntimeError(f"llama_encode returned {return_code}")

    def decode(self, batch: 'LlamaBatch') -> int:
        """
        Evaluate the batch of tokens using the transformer model.

        This method executes the forward pass. If the KV cache is heavily fragmented
        or out of space, it may return 1, indicating the caller should try to reduce
        the batch size or evict idle sequences.

        Returns:
            0: Success.
            1: No KV slot available (Recoverable). The caller should implement a
               fallback strategy, such as reducing the batch size and retrying.

        Raises:
            RuntimeError: If a fatal, non-recoverable error occurs during decoding
                          (e.g., negative error codes or invalid batch structures).
        """
        return_code = llama_cpp.llama_decode(self.ctx, batch.batch)

        if return_code == 0:
            return 0

        # 1 means "No KV slot available".
        # We explicitly return 1 instead of raising an exception so that the caller
        # can gracefully handle it via dynamic batch sizing (batch_size //= 2).
        elif return_code == 1:
            return 1

        # Any other code indicates a fatal failure.
        error_map = {
             2: "Decoding aborted by user callback",
            -1: "Invalid input batch (e.g. n_tokens == 0 or exceeding capacity)",
            -2: "Could not allocate space for the compute graph (VRAM exhausted)",
            -3: "Graph computation failed internally",
        }

        msg = error_map.get(return_code, "Unknown fatal internal error")
        raise RuntimeError(f"llama_decode failed (code {return_code}): {msg}")

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        llama_cpp.llama_set_n_threads(self.ctx, n_threads, n_threads_batch)

    def n_threads(self) -> int:
        return llama_cpp.llama_n_threads(self.ctx)

    def n_threads_batch(self) -> int:
        return llama_cpp.llama_n_threads_batch(self.ctx)

    def set_causal_attn(self, causal_attn: bool):
        llama_cpp.llama_set_causal_attn(self.ctx, causal_attn)

    def set_warmup(self, warmup: bool):
        llama_cpp.llama_set_warmup(self.ctx, warmup)

    def synchronize(self):
        llama_cpp.llama_synchronize(self.ctx)

    def get_logits(self):
        return llama_cpp.llama_get_logits(self.ctx)

    def get_logits_ith(self, i: int):
        return llama_cpp.llama_get_logits_ith(self.ctx, i)

    def set_embeddings(self, embeddings: bool):
        llama_cpp.llama_set_embeddings(self.ctx, embeddings)

    def get_embeddings(self):
        return llama_cpp.llama_get_embeddings(self.ctx)

    def get_embeddings_ith(self, i: int):
        return llama_cpp.llama_get_embeddings_ith(self.ctx, i)

    def get_embeddings_seq(self, seq_id: int):
        return llama_cpp.llama_get_embeddings_seq(self.ctx, seq_id)

    def reset_timings(self):
        llama_cpp.llama_perf_context_reset(self.ctx)

    def print_timings(self):
        llama_cpp.llama_perf_context_print(self.ctx)

    def print_memory_breakdown(self):
        llama_cpp.llama_memory_breakdown_print(self.ctx)

    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return llama_cpp.llama_context_default_params()


class LlamaBatch:
    def __init__(
        self,
        *,
        n_tokens: int,
        embd: int,
        n_seq_max: int,
        verbose: bool = True
    ):
        # logical validity of parameters
        if n_tokens <= 0:
            raise ValueError(f"n_tokens must be positive, got {n_tokens}")
        if n_seq_max <= 0:
            raise ValueError(f"n_seq_max must be positive, got {n_seq_max}")

        self.n_tokens_capacity = n_tokens
        self.embd = embd
        self.n_seq_max = n_seq_max
        self.verbose = verbose
        self._exit_stack = ExitStack()

        batch = llama_cpp.llama_batch_init(self.n_tokens_capacity, self.embd, self.n_seq_max)

        if batch is None:
            raise MemoryError(
                f"Failed to allocate memory for llama_batch via llama_batch_init({n_tokens},{embd},{n_seq_max})"
            )

        self.batch = batch

    def close(self):
        """Manually free LlamaBatch resources."""
        if getattr(self, "batch", None) is not None:
            try:
                llama_cpp.llama_batch_free(self.batch)
            except Exception:
                pass
            self.batch = None

        self._exit_stack.close()

    def __del__(self):
        self.close()

    def n_tokens(self) -> int:
        """
        Current number of tokens stored in the batch.
        """
        if self.batch is None: return 0
        return self.batch.n_tokens

    def capacity(self) -> int:
        """
        Total capacity of the batch.
        """
        return self.n_tokens_capacity

    def space_left(self) -> int:
        """
        Returns the number of empty slots remaining in the batch.
        Throws a RuntimeError if internal state implies an overflow.
        """
        if self.batch is None: return 0
        elif self.n_tokens_capacity >= self.batch.n_tokens:
            return self.n_tokens_capacity - self.batch.n_tokens
        else:
            raise RuntimeError(
                f"LlamaBatch Critical Error: n_tokens ({self.batch.n_tokens}) exceeds capacity ({self.n_tokens_capacity}). "
                "This implies a buffer overflow or corrupted internal state."
            )

    def reset(self):
        """
        Resets the batch counter to 0. Does not free memory, just resets the index.
        Call this before starting a new decoding step.
        """
        if self.batch is not None:
            self.batch.n_tokens = 0

    def add_token(self, token: int, pos: int, seq_ids: Sequence[int], logits: bool):
        """
        Adds a single token to the batch.
        This is a high-performance method for appending a single token during the generation loop,
        avoiding the overhead of creating temporary lists required by add_sequence.

        Args:
            token: The integer ID of the token to add.
            pos: The logical sequence position (n_past) of this token.
            seq_ids: A sequence of sequence IDs this token belongs to (e.g., [0] for a standard single chat).
                     A single token can be part of multiple sequences simultaneously.
            logits: A boolean flag indicating whether the backend should compute logits for this token.
        """
        idx = self.batch.n_tokens
        if idx >= self.n_tokens_capacity:
            raise IndexError(f"LlamaBatch overflow[add_token]: Cannot add token. Capacity {self.n_tokens_capacity} reached.")

        self.batch.token[idx] = token
        self.batch.pos[idx] = pos

        n_seq_id = len(seq_ids)
        if n_seq_id > self.n_seq_max:
            raise ValueError(f"LlamaBatch Error[add_token]: Token belongs to {n_seq_id} sequences, "
                             f"but n_seq_max was initialized to {self.n_seq_max}.")
        self.batch.n_seq_id[idx] = n_seq_id

        for i, seq_id in enumerate(seq_ids):
            self.batch.seq_id[idx][i] = seq_id
        self.batch.logits[idx] = logits

        self.batch.n_tokens += 1

    def add_sequence(
        self,
        token_array: Sequence[int],
        pos_array: Sequence[int],
        seq_ids: Sequence[Sequence[int]],
        logits_array: Sequence[bool]
    ):
        """
        Adds a sequence of tokens to the batch in a vectorized manner.
        Strictly maps the provided arrays to the underlying C++ batch structure without subjective overriding.

        Args:
            token_array: A sequence of token IDs to be evaluated.
            pos_array: A sequence of logical positions corresponding to each token.
            seq_id_array: A sequence of lists, where each list contains the sequence IDs for the respective token.
                          (e.g., [[0], [0], [0]] for 3 tokens belonging to sequence 0).
            logits_array: A sequence of boolean flags indicating whether to compute logits for each token.
        """
        n_tokens = len(token_array)
        current_count = self.batch.n_tokens

        if current_count + n_tokens > self.n_tokens_capacity:
            raise IndexError(
                f"LlamaBatch overflow[add_sequence]: Cannot add {n_tokens} tokens. "
                f"Space left: {self.n_tokens_capacity - current_count}"
            )

        n_seq_id = len(seq_ids)
        if n_seq_id > self.n_seq_max:
            raise ValueError(f"LlamaBatch Error[add_sequence]: Token belongs to {n_seq_id} sequences, "
                             f"but n_seq_max was initialized to {self.n_seq_max}.")

        for i in range(n_tokens):
            j = current_count + i
            self.batch.token[j] = token_array[i]
            self.batch.pos[j] = pos_array[i]

            self.batch.n_seq_id[j] = n_seq_id
            for k, seq_id in enumerate(seq_ids):
                self.batch.seq_id[j][k] = seq_id

            self.batch.logits[j] = logits_array[i]

        self.batch.n_tokens += n_tokens


# Embedding functions
def normalize_embedding(embedding):
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return [v / norm for v in embedding]


class LlamaTokenDataArray:
    """
    Performance-optimized wrapper for llama_token_data_array.
    This class minimizes Python overhead by caching memory views and avoiding
    redundant memory allocations during the inference loop.
    """
    def __init__(self, *, n_vocab: int):
        self.n_vocab = n_vocab

        # Define the structure of llama_token_data to match the C++ memory layout.
        # id: token identifier (int32)
        # logit: raw prediction score (float32)
        # p: probability score (float32)
        self.candidates_data = np.empty(
            self.n_vocab,
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)],
                align=True
            ),
        )

        # Optimization: Cache field views to bypass NumPy's expensive field lookup overhead.
        # Using these cached views allows for direct memory access in the inference loop.
        self._id_view = self.candidates_data["id"]
        self._logit_view = self.candidates_data["logit"]
        self._p_view = self.candidates_data["p"]

        # Initialization: Pre-generate a standard token ID sequence (0 to n_vocab - 1).
        # This acts as the 'golden' reference to reset the buffer after sorting operations.
        self._default_ids = np.arange(self.n_vocab, dtype=np.intc)
        self._id_view[:] = self._default_ids

        # Construct the llama_cpp C structure.
        # 'data' is assigned a direct pointer to the underlying NumPy memory buffer.
        self.candidates = llama_cpp.llama_token_data_array(
            data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=self.n_vocab,
            selected=-1,
            sorted=False,
        )

    def copy_logits(self, logits: npt.NDArray[np.single]):
        """
        Synchronizes the memory buffer with new logit data from the model.
        """
        # Step 1: Transfer new logits from the model output to our working buffer.
        self._logit_view[:] = logits

        # Step 2: Critical Reset.
        # Samplers (like top-k or top-p) reorder elements in memory during processing.
        # We must reset token IDs every step to ensure logical consistency for the next run.
        self._id_view[:] = self._default_ids

        # Step 3: Metadata update.
        # Inform the llama.cpp backend that the buffer is full and currently unsorted.
        self.candidates.size = self.n_vocab
        self.candidates.sorted = False
        self.candidates.selected = -1

    def close(self):
        """
        Release internal NumPy buffers and C-structure references.
        """
        # Main structured NumPy buffer holding token data (id, logit, prob)
        self.candidates_data = None

        # Cached NumPy field views (avoid dangling references)
        self._id_view = None
        self._logit_view = None
        self._p_view = None

        # Precomputed default token id array
        self._default_ids = None

        # Setting to None ensures no stale pointer references remain.
        self.candidates = None

    def __del__(self):
        # Ensures memory cleanup in case close() was not called explicitly.
        try:
            self.close()
        except Exception:
            pass


# Python wrappers over common/sampling structs
# common/common.h common_params_sampling

# enum common_sampler_type {
#     COMMON_SAMPLER_TYPE_NONE        = 0,
#     COMMON_SAMPLER_TYPE_DRY         = 1,
#     COMMON_SAMPLER_TYPE_TOP_K       = 2,
#     COMMON_SAMPLER_TYPE_TOP_P       = 3,
#     COMMON_SAMPLER_TYPE_MIN_P       = 4,
#   //COMMON_SAMPLER_TYPE_TFS_Z       = 5,
#     COMMON_SAMPLER_TYPE_TYPICAL_P   = 6,
#     COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
#     COMMON_SAMPLER_TYPE_XTC         = 8,
#     COMMON_SAMPLER_TYPE_INFILL      = 9,
#     COMMON_SAMPLER_TYPE_PENALTIES   = 10,
#     COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11,
#     COMMON_SAMPLER_TYPE_ADAPTIVE_P  = 12,
# };

class CommonSamplerType(enum.IntEnum):
    NONE        = 0
    DRY         = 1
    TOP_K       = 2
    TOP_P       = 3
    MIN_P       = 4
    TYPICAL_P   = 6
    TEMPERATURE = 7
    XTC         = 8
    INFILL      = 9
    PENALTIES   = 10
    TOP_N_SIGMA = 11
    ADAPTIVE_P  = 12

    CUSTOM      = 99

@dataclass
class LlamaSamplingParams:
    seed: int = llama_cpp.LLAMA_DEFAULT_SEED  # the seed used to initialize llama_sampler

    n_prev: int = 64                 # number of previous tokens to remember
    n_probs: int = 0                 # if greater than 0, output the probabilities of top n_probs tokens.
    min_keep: int = 0                # 0 = disabled, otherwise samplers should return at least min_keep tokens
    top_k: int = 40                  # <= 0 to use vocab size
    top_p: float = 0.95              # 1.0 = disabled
    min_p: float = 0.05              # 0.0 = disabled
    xtc_probability: float = 0.0     # 0.0 = disabled
    xtc_threshold: float = 0.1       # > 0.5 disables XTC
    typical_p: float = 1.00          # typical_p, 1.0 = disabled
    temp: float = 0.80               # <= 0.0 to sample greedily, 0.0 to not output probabilities
    dynatemp_range: float = 0.00     # 0.0 = disabled
    dynatemp_exponent: float = 1.00  # controls how entropy maps to temperature in dynamic temperature sampler

    penalty_last_n: int = 64         # last n tokens to penalize (0 = disable penalty, -1 = context size)
    penalty_repeat: float = 1.0      # 1.0 = disabled
    penalty_freq: float = 0.00       # 0.0 = disabled
    penalty_present: float = 0.00    # 0.0 = disabled

    dry_multiplier: float = 0.0      # 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    dry_base: float = 1.75           # 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    dry_allowed_length: int = 2      # tokens extending repetitions beyond this receive penalty
    dry_penalty_last_n: int = -1     # how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)

    adaptive_target: float = -1.0    # select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    adaptive_decay: float = 0.90     # EMA decay for adaptation; history ≈ 1/(1-decay) tokens (0.0 - 0.99)
    mirostat: int = 0                # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    top_n_sigma: float = -1.00       # -1.0 = disabled
    mirostat_tau: float = 5.00       # target entropy
    mirostat_eta: float = 0.10       # learning rate

    ignore_eos: bool = False
    no_perf: bool = False            # disable performance metrics
    timing_per_token: bool = False
    backend_sampling: bool = False
    user_sampling_config: int = 0    # bitfield to track user-specified samplers

    dry_sequence_breakers: List[str] = field(
        default_factory=lambda: ["\n", ":", "\"", "*"]  # default sequence breakers for DRY
    )

    custom_samplers: List['CustomSampler'] = field(default_factory=list)

    samplers: List[CommonSamplerType] = field(
        default_factory=lambda: [
            CommonSamplerType.PENALTIES,
            CommonSamplerType.DRY,
            CommonSamplerType.TOP_N_SIGMA,
            # CommonSamplerType.CUSTOM,  # When logits_processor is used, CommonSamplerType.CUSTOM is automatically injected into the samplers.
            CommonSamplerType.TOP_K,
            CommonSamplerType.TYPICAL_P,
            CommonSamplerType.TOP_P,
            CommonSamplerType.MIN_P,
            CommonSamplerType.XTC,
            CommonSamplerType.TEMPERATURE,
        ]
    )

    grammar: str = ""
    grammar_lazy: bool = False
    grammar_triggers: List[Any] = field(default_factory=list)
    preserved_tokens: Set[int] = field(default_factory=set)

    logit_bias: List[llama_cpp.llama_logit_bias] = field(default_factory=list)
    logit_bias_eog: List[llama_cpp.llama_logit_bias] = field(default_factory=list)

    @property
    def has_logit_bias(self) -> bool:
        return len(self.logit_bias) > 0

    def print_params(self) -> str:
        result = (
            f"\trepeat_last_n = {self.penalty_last_n}, repeat_penalty = {self.penalty_repeat:.3f}, "
            f"frequency_penalty = {self.penalty_freq:.3f}, present_penalty = {self.penalty_present:.3f}\n"

            f"\tdry_multiplier = {self.dry_multiplier:.3f}, dry_base = {self.dry_base:.3f}, "
            f"dry_allowed_length = {self.dry_allowed_length}, dry_penalty_last_n = {self.dry_penalty_last_n}\n"

            f"\ttop_k = {self.top_k}, top_p = {self.top_p:.3f}, min_p = {self.min_p:.3f}, "
            f"xtc_probability = {self.xtc_probability:.3f}, xtc_threshold = {self.xtc_threshold:.3f}, "
            f"typical_p = {self.typ_p:.3f}, top_n_sigma = {self.top_n_sigma:.3f}, temp = {self.temp:.3f}\n"

            f"\tmirostat = {self.mirostat}, mirostat_lr = {self.mirostat_eta:.3f}, "
            f"mirostat_ent = {self.mirostat_tau:.3f}, adaptive_target = {self.adaptive_target:.3f}, "
            f"adaptive_decay = {self.adaptive_decay:.3f}"
        )
        return result

    def __repr__(self) -> str:
        return self.print_params()

class GrammarSampler:

    def __init__(self, model, grammar_str, lazy=False, triggers=None):

        if model is None:
            raise ValueError("model must not be None")

        self.model = model
        self.vocab = model.vocab

        if not grammar_str:
            raise ValueError("grammar_str must not be empty")

        self.grammar = llama_cpp.llama_sampler_init_grammar(
            self.vocab,
            grammar_str.encode("utf-8"),
            b"root"
        )

        if not self.grammar:
            raise RuntimeError("Failed to initialize grammar sampler")

    def apply(self, token_data):
        llama_cpp.llama_sampler_apply(self.grammar, token_data)

    def accept(self, token):
        llama_cpp.llama_sampler_accept(self.grammar, token)

    def reset(self):
        llama_cpp.llama_sampler_reset(self.grammar)

    def close(self):
        if self.grammar:
            try:
                llama_cpp.llama_sampler_free(self.grammar)
            except Exception:
                pass

        self.model = None
        self.vocab = None
        self.grammar = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

@dataclass
class LlamaSamplingContext:
    """
    High-level Python wrapper that manages the lifecycle and configuration
    of the llama.cpp sampler chain.
    """
    def __init__(
        self,
        params: LlamaSamplingParams = field(default_factory=LlamaSamplingParams),
        model: Optional[LlamaModel] = None,
        _existing_sampler: Optional[LlamaSampler] = None, # Internal use for cloning
    ):
        if model is None:
            raise RuntimeError("model must not be None")
        self.model = model

        self.params = params
        self.vocab = llama_cpp.llama_model_get_vocab(model.model)
        self.n_vocab = model.n_vocab()

        lparams = llama_cpp.llama_sampler_chain_default_params()
        lparams.no_perf = params.no_perf

        # history (bounded)
        # last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)
        if self.params.penalty_last_n == -1:
            # full context
            self.params.penalty_last_n = self.model.n_ctx_train()

        # params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
        if self.params.penalty_last_n > 0:
            self.params.n_prev = max(
                self.params.n_prev,
                self.params.penalty_last_n
            )
        self.prev = deque(maxlen=max(self.params.n_prev, 32))

        # reusable token data array
        self._cur_p = LlamaTokenDataArray(n_vocab=self.n_vocab)

        # reusable numpy logits view
        self._logits_view = None

        self._single_token = llama_cpp.llama_token_data()
        self._single_array = llama_cpp.llama_token_data_array(
            data=ctypes.pointer(self._single_token),
            size=1,
            selected=-1,
            sorted=False,
        )

        # sampler chain
        if _existing_sampler:
            self.sampler_chain = _existing_sampler
        else:
            self.sampler_chain = LlamaSampler()
            self._build_sampler_chain()

        # grammar sampler
        self.grammar_sampler = None
        if params.grammar:
            self.grammar_sampler = GrammarSampler(
                model,
                params.grammar,
                params.grammar_lazy,
                params.grammar_triggers,
            )

    def _build_sampler_chain(self):
        """
        Build sampler chain aligned with llama.cpp common_sampler_init
        Grammar is intentionally NOT part of the chain.
        """

        s = self.sampler_chain
        p = self.params
        m = self.model

        if m is None:
            raise RuntimeError("Model required to build sampler chain firstly")

        use_adaptive_p = False

        # --- 1. Logit Bias (Always applied first to mask/boost tokens) ---
        if p.logit_bias and m:
            s.add_logit_bias(m.n_vocab(), p.logit_bias)

        # --- 2. Usage-Specific Samplers (Infill) ---
        # If Infill is required, it often modifies logits based on prefix/suffix
        if CommonSamplerType.INFILL in p.samplers and m:
             s.add_infill(m)

        # --- 3. Penalties (Repetition) ---
        # Note: In some implementations, penalties come before other samplers
        if CommonSamplerType.PENALTIES in p.samplers:
            s.add_penalties(
                p.penalty_last_n,
                p.penalty_repeat,
                p.penalty_freq,
                p.penalty_present
            )

        # --- 4. DRY (Don't Repeat Yourself) ---
        if CommonSamplerType.DRY in p.samplers and m:
            s.add_dry(
                m,
                p.dry_multiplier,
                p.dry_base,
                p.dry_allowed_length,
                p.dry_penalty_last_n,
                p.dry_sequence_breakers
            )

        # --- 5. Core Sampling Strategies (The "Filter" Loop) ---
        # We iterate through the list to preserve user-defined order for these specific samplers
        for stype in p.samplers:
            if stype == CommonSamplerType.CUSTOM:
                if p.custom_samplers:
                    for cs in p.custom_samplers:
                        s.add_custom(cs)

            elif stype == CommonSamplerType.TOP_K:
                s.add_top_k(p.top_k)

            elif stype == CommonSamplerType.TOP_P:
                s.add_top_p(p.top_p, p.min_keep)

            elif stype == CommonSamplerType.MIN_P:
                s.add_min_p(p.min_p, p.min_keep)

            elif stype == CommonSamplerType.TYPICAL_P:
                s.add_typical(p.typical_p, p.min_keep)

            elif stype == CommonSamplerType.TEMPERATURE:
                s.add_temp(p.temp)

            elif stype == CommonSamplerType.XTC:
                s.add_xtc(p.xtc_probability, p.xtc_threshold, p.min_keep, p.seed)

            elif stype == CommonSamplerType.TOP_N_SIGMA:
                s.add_top_n_sigma(p.top_n_sigma)

            elif stype == CommonSamplerType.ADAPTIVE_P:
                use_adaptive_p = True

        # --- 6. Final Distribution / Selection ---
        # Mirostat overrides standard greedy/dist sampling
        if p.mirostat == 1 and m:
            s.add_mirostat(m.n_vocab(), p.seed, p.mirostat_tau, p.mirostat_eta, 100)
        elif p.mirostat == 2:
            s.add_mirostat_v2(p.seed, p.mirostat_tau, p.mirostat_eta)
        else:
            if use_adaptive_p:
                s.add_adaptive_p(p.adaptive_target, p.adaptive_decay, p.seed)
            else:
                if p.temp == 0:
                    s.add_greedy()
                else:
                    s.add_dist(p.seed)

    def reset(self):
        """
        Resets the internal state of all samplers in the chain.
        """
        self.prev.clear()

        if self.grammar_sampler:
            self.grammar_sampler.reset()

        if self.sampler_chain:
            self.sampler_chain.reset()

    def cp(self) -> 'LlamaSamplingContext':
        """
        Creates a deep copy of the sampling context.
        This clones the sampler chain state
        """
        # 1. Clone the sampler chain using llama_sampler_clone
        new_sampler_chain = self.sampler_chain.clone()

        # 2. Create new context wrapping the cloned chain
        new_ctx = LlamaSamplingContext(
            self.params,
            self.model,
            _existing_sampler=new_sampler_chain
        )

        # 3. Copy Python-side history
        new_ctx.prev = self.prev.copy()

        return new_ctx

    def accept(self, token: int, accept_grammar: bool):
        """
        Accepts a token into the sampler state.
        MUST be called after sampling to update repetition penalties, grammar state, etc.

        Args:
            token: The token ID selected.
        """
        if self.grammar_sampler and accept_grammar:
            self.grammar_sampler.accept(token)
        self.sampler_chain.accept(token)
        self.prev.append(token)

    def sample(
        self,
        ctx: LlamaContext,
        idx: int = -1,
        grammar_first: bool = True,
    ) -> int:

        # 1. Synchronize
        llama_cpp.llama_synchronize(ctx.ctx)

        # 2. Backend sampler shortcut
        sampled = llama_cpp.llama_get_sampled_token_ith(ctx.ctx, idx)
        if sampled != llama_cpp.LLAMA_TOKEN_NULL:
            if self.grammar_sampler:
                raise RuntimeError("Backend sampling + grammar unsupported")
            return int(sampled)

        # 3. build cur_p
        logits_ptr = llama_cpp.llama_get_logits_ith(ctx.ctx, idx)

        if self._logits_view is None:
            self._logits_view = np.ctypeslib.as_array(
                logits_ptr,
                shape=(self.n_vocab,),
            )

        logits_array = self._logits_view
        cur_p = self._cur_p

        cur_p.copy_logits(logits_array)

        # logit bias
        if self.params.logit_bias:
            for item in self.params.logit_bias:
                cur_p._logit_view[item.token] += item.bias


        # 4. grammar first
        if self.grammar_sampler and grammar_first:
            llama_cpp.llama_sampler_apply(
                self.grammar_sampler.grammar,
                ctypes.byref(cur_p.candidates)
            )

            llama_cpp.llama_sampler_apply(
                self.sampler_chain.sampler,
                ctypes.byref(cur_p.candidates)
            )
            # grammar-first return directly
            selected = cur_p.candidates.selected
            return int(cur_p._id_view[selected])


        # 5. sampling chain
        llama_cpp.llama_sampler_apply(
            self.sampler_chain.sampler,
            ctypes.byref(cur_p.candidates)
        )

        selected = cur_p.candidates.selected
        token = int(cur_p._id_view[selected])

        # 6. grammar rejection sampling
        if self.grammar_sampler:

            self._single_token.id = token
            self._single_token.logit = 1.0
            self._single_token.p = 0.0
            self._single_array.selected = -1
            self._single_array.sorted = False

            llama_cpp.llama_sampler_apply(
                self.grammar_sampler.grammar,
                ctypes.byref(self._single_array)
            )

            if not np.isneginf(self._single_token.logit):
                return token

            # 7. resample
            cur_p.copy_logits(logits_array)

            llama_cpp.llama_sampler_apply(
                self.grammar_sampler.grammar,
                ctypes.byref(cur_p.candidates)
            )

            llama_cpp.llama_sampler_apply(
                self.sampler_chain.sampler,
                ctypes.byref(cur_p.candidates)
            )

            selected = cur_p.candidates.selected
            token = int(cur_p._id_view[selected])

        return token

    def close(self):
        """
        Release all sampling-related resources and break references
        to large buffers to allow Python GC to reclaim memory.

        This method must be called when the sampling context is no longer needed,
        especially in long-running services, to prevent memory retention.
        """

        # Free grammar sampler if it was initialized.
        # This releases underlying llama.cpp sampler memory.
        if self.grammar_sampler:
            self.grammar_sampler.close()
            self.grammar_sampler = None

        # Free the sampler chain and all attached C samplers.
        if self.sampler_chain:
            self.sampler_chain.close()
            self.sampler_chain = None

        # Release large token data buffer used during sampling.
        # Important for high-vocab models to avoid memory retention.
        if hasattr(self, "_cur_p"):
            try:
                self._cur_p.close()
            except Exception:
                pass
            self._cur_p = None

        # Clear token history deque to drop references.
        if hasattr(self, "prev"):
            self.prev.clear()
            self.prev = None

        # Remove NumPy view pointing to llama logits buffer.
        self._logits_view = None

        # Break references to small C structs used in grammar rejection sampling.
        self._single_token = None
        self._single_array = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- Utilities ---

    def last(self) -> Optional[int]:
        """Returns the last sampled token."""
        return self.prev[-1] if self.prev else None

    def prev_str(self, ctx_main: LlamaContext, n: int) -> str:
        """
        Decodes the last n tokens into a string.
        Useful for debugging what the sampler chain "sees" as context.
        """
        if not self.prev:
            return ""
        # Get the last n tokens
        last_n_tokens = self.prev[-n:]
        # Use the model linked to the context to detokenize
        return ctx_main.model.detokenize(last_n_tokens).decode("utf-8", errors="replace")


class CustomSampler:
    """
    Python wrapper for llama.cpp custom sampler.

    apply_func:
        Callable receiving llama_token_data_array
        and modifying logits in-place.
    """

    def __init__(
        self,
        apply_func: Callable[[llama_cpp.llama_token_data_array], None],
        name: str = "custom",
        accept_func: Optional[Callable] = None,
        reset_func: Optional[Callable] = None,
        free_func: Optional[Callable] = None,
        clone_func: Optional[Callable] = None,
    ):
        if not callable(apply_func):
            raise TypeError("apply_func must be callable")

        self.apply_func = apply_func
        # Convert the name to bytes for C compatibility
        self.name_bytes = name.encode("utf-8")

        # Define internal Python callbacks
        def _cb_name(_):
            return self.name_bytes

        def _cb_apply(_, cur_p):
            if cur_p:
                self.apply_func(cur_p.contents)

        def _cb_accept(_, token):
            if accept_func:
                accept_func(token)

        def _cb_reset(_):
            if reset_func:
                reset_func()

        def _cb_free(_):
            if free_func:
                free_func()

        def _cb_clone(_):
            if clone_func:
                return clone_func()
            return None

        self._cb_name_ref = llama_cpp.llama_sampler_name_fn(_cb_name)
        self._cb_apply_ref = llama_cpp.llama_sampler_apply_fn(_cb_apply)
        self._cb_accept_ref = llama_cpp.llama_sampler_accept_fn(_cb_accept)
        self._cb_reset_ref = llama_cpp.llama_sampler_reset_fn(_cb_reset)
        self._cb_free_ref = llama_cpp.llama_sampler_free_fn(_cb_free)
        self._cb_clone_ref = llama_cpp.llama_sampler_clone_fn(_cb_clone)

        # Build llama_sampler_i
        self.llama_sampler_i = llama_cpp.llama_sampler_i()

        self.llama_sampler_i.name = self._cb_name_ref
        self.llama_sampler_i.apply = self._cb_apply_ref
        self.llama_sampler_i.accept = self._cb_accept_ref
        self.llama_sampler_i.reset = self._cb_reset_ref
        self.llama_sampler_i.free = self._cb_free_ref
        self.llama_sampler_i.clone = self._cb_clone_ref

        # Disable backend hooks
        self.llama_sampler_i.backend_init = ctypes.cast(
            0, llama_cpp.llama_sampler_backend_init_fn
        )
        self.llama_sampler_i.backend_accept = ctypes.cast(
            0, llama_cpp.llama_sampler_backend_accept_fn
        )
        self.llama_sampler_i.backend_apply = ctypes.cast(
            0, llama_cpp.llama_sampler_backend_apply_fn
        )
        self.llama_sampler_i.backend_set_input = ctypes.cast(
            0, llama_cpp.llama_sampler_backend_set_input_fn
        )

        self.sampler_p = llama_cpp.llama_sampler_init(
            ctypes.pointer(self.llama_sampler_i),
            None
        )

        if not self.sampler_p:
            raise RuntimeError("Failed to initialize custom sampler")

    def get_sampler(self) -> llama_cpp.llama_sampler_p:
        """Returns the underlying C pointer to the initialized sampler."""
        return self.sampler_p

    def close(self):
        """Safely releases C memory and breaks Python reference cycles."""
        if hasattr(self, 'sampler_p') and self.sampler_p:
            try:
                llama_cpp.llama_sampler_free(self.sampler_p)
            except Exception:
                pass
            self.sampler_p = None

        self.llama_sampler_i = None
        self._cb_name_ref = None
        self._cb_apply_ref = None
        self._cb_accept_ref = None
        self._cb_reset_ref = None
        self._cb_free_ref = None
        self._cb_clone_ref = None
        self.apply_func = None

    def __del__(self):
        """Fallback cleanup if the object is GC before close() is called."""
        self.close()


class LlamaSampler:
    def __init__(self, existing_sampler_p: Optional[llama_cpp.llama_sampler_p] = None):
        if existing_sampler_p:
            self.sampler = existing_sampler_p
        else:
            # Initialize new chain
            params = llama_cpp.llama_sampler_chain_default_params()
            params.no_perf = False
            self.sampler = llama_cpp.llama_sampler_chain_init(params)

        self.samplers: List[llama_cpp.llama_sampler_p] = []
        self.custom_samplers: List["CustomSampler"] = []
        self._keep_alive: List[Any] = []

    def _add_sampler(self, sampler: llama_cpp.llama_sampler_p):
        if not sampler:
            raise RuntimeError("Failed to initialize sampler")
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)
        self.samplers.append(sampler)

    # --- Core Sampling Methods ---

    def accept(self, token: int):
        """
        Updates the sampler state (e.g. repetition penalty history).
        """
        assert self.sampler is not None

        if token is None: raise RuntimeError("Sampler returned None token")

        if token < 0: raise RuntimeError(f"Invalid token sampled: {token}")

        try:
            llama_cpp.llama_sampler_accept(self.sampler, token)
        except Exception as e:
            raise RuntimeError(
                f"Sampler accept crashed. token={token}"
            ) from e

    def clone(self) -> 'LlamaSampler':
        """
        Clones the sampler chain and its internal state.
        """
        if not self.sampler:
            raise RuntimeError("Cannot clone: sampler is closed or not initialized")

        # Call C-level llama.cpp clone
        new_sampler_p = llama_cpp.llama_sampler_clone(self.sampler)
        if not new_sampler_p:
            raise RuntimeError("llama_sampler_clone failed")

        new_sampler = LlamaSampler(existing_sampler_p=new_sampler_p)

        # copy _keep_alive and custom_samplers list to new sampler
        if self._keep_alive:
            new_sampler._keep_alive = self._keep_alive.copy()

        if self.custom_samplers:
            new_sampler.custom_samplers = self.custom_samplers.copy()

        return new_sampler

    def sample(self, ctx: LlamaContext, idx: int = -1) -> int:
        """
        Sample and accept a token from the idx-th output of the last evaluation
        """
        assert self.sampler is not None
        assert ctx.ctx is not None
        return llama_cpp.llama_sampler_sample(self.sampler, ctx.ctx, idx)

    def reset(self):
        """
        Resets the sampler state.
        """
        assert self.sampler is not None
        llama_cpp.llama_sampler_reset(self.sampler)

    def reset_timings(self):
        """
        Reset the performance timings for the sampler chain.
        """
        assert self.sampler is not None
        llama_cpp.llama_perf_sampler_reset(self.sampler)

    def print_timings(self):
        """
        Print the performance timings for each sampler in the chain.
        """
        assert self.sampler is not None
        llama_cpp.llama_perf_sampler_print(self.sampler)

    def close(self):
        if self.sampler:
            # Iterate backwards to safely remove samplers without shifting indices
            for index, custom_sampler in reversed(self.custom_samplers):
                # Detach the custom sampler from the C-level chain
                llama_cpp.llama_sampler_chain_remove(self.sampler, index)

                # Explicitly free the custom sampler's C memory and Python callbacks
                if custom_sampler:
                    custom_sampler.close()

            # Free the main official sampler chain
            llama_cpp.llama_sampler_free(self.sampler)
            self.sampler = None

        # Clear cache lists
        self.samplers.clear()
        self.custom_samplers.clear()
        self._keep_alive.clear()

    def __del__(self):
        self.close()

    # --- Specific Samplers (aligning with llama-sampler.cpp) ---

    def add_greedy(self):
        self._add_sampler(llama_cpp.llama_sampler_init_greedy())

    def add_dist(self, seed: int):
        self._add_sampler(llama_cpp.llama_sampler_init_dist(seed))

    def add_top_k(self, k: int):
        self._add_sampler(llama_cpp.llama_sampler_init_top_k(k))

    def add_top_p(self, p: float, min_keep: int):
        self._add_sampler(llama_cpp.llama_sampler_init_top_p(p, min_keep))

    def add_min_p(self, p: float, min_keep: int):
        self._add_sampler(llama_cpp.llama_sampler_init_min_p(p, min_keep))

    def add_typical(self, p: float, min_keep: int):
        self._add_sampler(llama_cpp.llama_sampler_init_typical(p, min_keep))

    def add_temp(self, temp: float):
        self._add_sampler(llama_cpp.llama_sampler_init_temp(temp))

    def add_temp_ext(self, t: float, delta: float, exponent: float):
        self._add_sampler(llama_cpp.llama_sampler_init_temp_ext(t, delta, exponent))

    def add_xtc(self, p: float, t: float, min_keep: int, seed: int):
        self._add_sampler(llama_cpp.llama_sampler_init_xtc(p, t, min_keep, seed))

    def add_top_n_sigma(self, n: float):
        self._add_sampler(llama_cpp.llama_sampler_init_top_n_sigma(n))

    def add_mirostat(self, n_vocab: int, seed: int, tau: float, eta: float, m: int):
        self._add_sampler(llama_cpp.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m))

    def add_mirostat_v2(self, seed: int, tau: float, eta: float):
        self._add_sampler(llama_cpp.llama_sampler_init_mirostat_v2(seed, tau, eta))

    def add_grammar(
        self,
        model: LlamaModel,
        grammar_str: str,
        lazy: bool = False,
        triggers: List[Union[str, int]] = None
    ):
        """
        Adds a grammar sampler.
        Args:
            grammar_str: The BNF grammar string.
            root: The root rule name.
            lazy: If True, enables lazy evaluation.
            triggers: List of trigger words (str) or tokens (int) for lazy evaluation.
        """
        c_grammar_str = grammar_str.encode('utf-8')
        c_root = "root".encode('utf-8')

        self._keep_alive.append(c_grammar_str)
        self._keep_alive.append(c_root)

        if not lazy:
            self._add_sampler(llama_cpp.llama_sampler_init_grammar(
                model.vocab, c_grammar_str, c_root
            ))
        else:
            trigger_patterns = []
            trigger_tokens = []
            if triggers:
                for t in triggers:
                    if isinstance(t, str):
                        trigger_patterns.append(t)
                    elif isinstance(t, int):
                        trigger_tokens.append(t)

            c_trigger_patterns = (ctypes.c_char_p * len(trigger_patterns))()
            c_trigger_patterns[:] = [w.encode('utf-8') for w in trigger_patterns]
            c_trigger_tokens = (llama_cpp.llama_token * len(trigger_tokens))(*trigger_tokens)

            self._keep_alive.append(c_trigger_patterns)
            self._keep_alive.append(c_trigger_tokens)

            self._add_sampler(llama_cpp.llama_sampler_init_grammar_lazy_patterns(
                model.vocab, c_grammar_str, c_root,
                c_trigger_patterns, len(trigger_patterns),
                c_trigger_tokens, len(trigger_tokens)
            ))

    def add_penalties(self, penalty_last_n: int, penalty_repeat: float, penalty_freq: float, penalty_present: float):
        self._add_sampler(llama_cpp.llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present))

    def add_dry(self, model: LlamaModel, multiplier: float, base: float, allowed_len: int, last_n: int, breakers: List[str]):
        """DRY (Don't Repeat Yourself) sampler."""
        # Convert python string list to C char**
        c_breakers = (ctypes.c_char_p * len(breakers))()
        c_breakers[:] = [b.encode('utf-8') for b in breakers]

        self._add_sampler(llama_cpp.llama_sampler_init_dry(
            model.vocab,
            model.n_ctx_train(),
            multiplier,
            base,
            allowed_len,
            last_n,
            c_breakers,
            len(breakers)
        ))

    def add_logit_bias(self, n_vocab: int, bias_dict: List[llama_cpp.llama_logit_bias]):
        """Logit bias sampler."""
        if not bias_dict: return

        c_bias = (llama_cpp.llama_logit_bias * len(bias_dict))()
        for i, bias in enumerate(bias_dict):
            c_bias[i].token = bias.token
            c_bias[i].bias = bias.bias

        self._add_sampler(llama_cpp.llama_sampler_init_logit_bias(n_vocab, len(bias_dict), c_bias))

    def add_infill(self, model: LlamaModel):
        self._add_sampler(llama_cpp.llama_sampler_init_infill(model.vocab))

    def add_adaptive_p(self, target: float, decay: float, seed: int):
        self._add_sampler(llama_cpp.llama_sampler_init_adaptive_p(target, decay, seed))

    def add_custom(self, custom_sampler: CustomSampler):
        if not isinstance(custom_sampler, CustomSampler):
            raise TypeError("add_custom expects a CustomSampler instance")

        sampler = custom_sampler.get_sampler()
        self._add_sampler(sampler)

        self.custom_samplers.append(
            [llama_cpp.llama_sampler_chain_n(self.sampler) - 1, custom_sampler]
        )

    def get_seed(self) -> int:
        assert self.sampler is not None
        return llama_cpp.llama_sampler_get_seed(self.sampler)
