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

        def free_model():
            if self.model is None:
                return
            llama_cpp.llama_model_free(self.model)
            self.model = None

        self._exit_stack.callback(free_model)

    def close(self):
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

        def free_ctx():
            if self.ctx is None:
                return
            llama_cpp.llama_free(self.ctx)
            self.ctx = None

        self._exit_stack.callback(free_ctx)

    def close(self):
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

    def decode(self, batch: LlamaBatch):
        return_code = llama_cpp.llama_decode(self.ctx, batch.batch)

        if return_code == 0:
            return

        error_map = {
             1: "No KV slot available: try reducing batch size or increasing context window",
             2: "Decoding aborted",
            -1: "Invalid input batch",
        }

        msg = error_map.get(return_code, "Fatal internal error")
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

        def free_batch():
            if self.batch is None:
                return
            llama_cpp.llama_batch_free(self.batch)
            self.batch = None

        self._exit_stack.callback(free_batch)

    def close(self):
        """Manually free resources."""
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

    def set_batch(self, batch: Sequence[int], n_past: llama_cpp.llama_pos, logits_all: bool):
        if len(batch) > self.n_tokens_capacity:
             raise IndexError(f"Input batch size {len(batch)} exceeds capacity {self.n_tokens_capacity}")

        n_tokens = len(batch)
        self.batch.n_tokens = n_tokens
        for i in range(n_tokens):
            self.batch.token[i] = batch[i]
            self.batch.pos[i] = n_past + i
            self.batch.seq_id[i][0] = 0
            self.batch.n_seq_id[i] = 1
            self.batch.logits[i] = logits_all
        self.batch.logits[n_tokens - 1] = True

    def add_sequence(self, batch: Sequence[int], seq_id: int, logits_all: bool):
        n_tokens = len(batch)
        current_count = self.batch.n_tokens
        if current_count + n_tokens > self.n_tokens_capacity:
            raise IndexError(
                f"LlamaBatch overflow: Cannot add {n_tokens} tokens. "
                f"Space left: {self.n_tokens_capacity - current_count}"
            )
        self.batch.n_tokens += n_tokens
        for i in range(n_tokens):
            j = current_count + i
            self.batch.token[j] = batch[i]
            self.batch.pos[j] = i
            self.batch.seq_id[j][0] = seq_id
            self.batch.n_seq_id[j] = 1
            self.batch.logits[j] = logits_all
        self.batch.logits[current_count + n_tokens - 1] = True


class LlamaTokenDataArray:
    def __init__(self, *, n_vocab: int):
        self.n_vocab = n_vocab
        self.candidates_data = np.recarray(
            (self.n_vocab,),
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
            ),
        )
        self.candidates = llama_cpp.llama_token_data_array(
            data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=self.n_vocab,
            selected=-1,
            sorted=False,
        )
        self.default_candidates_data_id = np.arange(self.n_vocab, dtype=np.intc)  # type: ignore
        self.default_candidates_data_p = np.zeros(self.n_vocab, dtype=np.single)

    def copy_logits(self, logits: npt.NDArray[np.single]):
        self.candidates_data.id[:] = self.default_candidates_data_id
        self.candidates_data.logit[:] = logits
        self.candidates_data.p[:] = self.default_candidates_data_p
        self.candidates.sorted = False
        self.candidates.size = self.n_vocab


# Embedding functions


def normalize_embedding(embedding):
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return [v / norm for v in embedding]


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
            CommonSamplerType.CUSTOM,
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
            f"frequency_penalty = {self.penalty_freq:.3f}, presence_penalty = {self.penalty_present:.3f}\n"

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
        self.params = params
        self.model = model

        # Keep track of generated tokens for Python-side debugging/decoding
        self.prev: List[int] = []

        if _existing_sampler:
            # Use the provided sampler (already configured/cloned)
            self.sampler = _existing_sampler
        else:
            # Build a new chain from scratch
            self.sampler = LlamaSampler()
            self._build_sampler_chain()

    def _build_sampler_chain(self):
        """
        Constructs the sampler chain based on the parameters.
        The order generally follows common.cpp practices:
        Bias -> Grammar -> Penalties -> DRY -> [Configurable Samplers] -> Dist/Greedy
        """
        s = self.sampler
        p = self.params
        m = self.model

        # --- 1. Logit Bias (Always applied first to mask/boost tokens) ---
        if p.logit_bias and m:
            s.add_logit_bias(m.n_vocab(), p.logit_bias)

        # --- 2. Usage-Specific Samplers (Infill) ---
        # If Infill is required, it often modifies logits based on prefix/suffix
        if CommonSamplerType.INFILL in p.samplers and m:
             s.add_infill(m)

        # --- 3. Grammar / Syntax Constraints ---
        if p.grammar and m:
            # Use "root" as default rule name if not specified
            root_rule = "root"
            s.add_grammar(
                model=m,
                grammar_str=p.grammar,
                root=root_rule,
                lazy=p.grammar_lazy,
                triggers=p.grammar_triggers
            )

        # --- 4. Penalties (Repetition) ---
        # Note: In some implementations, penalties come before other samplers
        if CommonSamplerType.PENALTIES in p.samplers:
            s.add_penalties(
                p.penalty_last_n,
                p.penalty_repeat,
                p.penalty_freq,
                p.penalty_present
            )

        # --- 5. DRY (Don't Repeat Yourself) ---
        if CommonSamplerType.DRY in p.samplers and m:
            s.add_dry(
                m,
                p.dry_multiplier,
                p.dry_base,
                p.dry_allowed_length,
                p.dry_penalty_last_n,
                p.dry_sequence_breakers
            )

        # --- 6. Core Sampling Strategies (The "Filter" Loop) ---
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
                s.add_adaptive_p(p.adaptive_target, p.adaptive_decay, p.seed)

        # --- 7. Final Distribution / Selection ---
        # Mirostat overrides standard greedy/dist sampling
        if p.mirostat == 1 and m:
            s.add_mirostat(m.n_vocab(), p.seed, p.mirostat_tau, p.mirostat_eta, 100)
        elif p.mirostat == 2:
            s.add_mirostat_v2(p.seed, p.mirostat_tau, p.mirostat_eta)
        else:
            # If not using Mirostat, use Greedy (if temp=0) or Random Distribution
            if p.temp == 0:
                s.add_greedy()
            else:
                s.add_dist(p.seed)

    def reset(self):
        """
        Resets the internal state of all samplers in the chain.
        """
        self.sampler.reset()
        self.prev = []

    def cp(self) -> 'LlamaSamplingContext':
        """
        Creates a deep copy of the sampling context.
        This clones the sampler chain state
        """
        # 1. Clone the sampler chain using llama_sampler_clone
        new_sampler = self.sampler.clone()

        # 2. Create new context wrapping the cloned chain
        new_ctx = LlamaSamplingContext(
            self.params,
            self.model,
            _existing_sampler=new_sampler
        )

        # 3. Copy Python-side history
        new_ctx.prev = self.prev.copy()

        return new_ctx

    def accept(self, token: int):
        """
        Accepts a token into the sampler state.
        MUST be called after sampling to update repetition penalties, grammar state, etc.

        Args:
            token: The token ID selected.
        """
        self.sampler.accept(token)
        self.prev.append(token)

    def sample(
        self,
        ctx: LlamaContext,
        idx: int = -1,
    ) -> int:
        """
        Samples a token from the model's current logits.

        Args:
            ctx_main: The context containing the logits.
            idx: The batch index to sample from (defaults to last token: -1).
        """
        return self.sampler.sample(ctx, idx)

    # --- Utilities ---

    def last(self) -> Optional[int]:
        """Returns the last sampled token."""
        if len(self.prev) > 0:
            return self.prev[-1]
        else:
            return None

    def prev_str(self, ctx_main: LlamaContext, n: int) -> str:
        """
        Decodes the last n tokens into a string.
        Useful for debugging what the sampler chain "sees" as context.
        """
        if not self.prev:
            return ""
        # Get the last n tokens
        last_tokens = self.prev[-n:]
        # Use the model linked to the context to detokenize
        return ctx_main.model.detokenize(last_tokens).decode("utf-8", errors="replace")


class CustomSampler:
    def __init__(
        self,
        apply_func: Callable[[llama_cpp.llama_token_data_array], None],
        name: str = "custom",
        **kwargs
    ):
        self.apply_func = apply_func
        self.name_bytes = name.encode('utf-8')

        def _cb_name(smpl):
            return self.name_bytes

        def _cb_apply(smpl, cur_p):
            if cur_p and self.apply_func:
                self.apply_func(cur_p.contents)

        self._cb_accept = kwargs.get('accept_func') or (lambda smpl, token: None)
        self._cb_reset  = kwargs.get('reset_func')  or (lambda smpl: None)
        self._cb_free   = kwargs.get('free_func')   or (lambda smpl: None)
        self._cb_clone  = kwargs.get('clone_func')  or (lambda smpl: None)

        self.llama_sampler_i = llama_cpp.llama_sampler_i()

        self.llama_sampler_i.name  = llama_cpp.llama_sampler_name_fn(_cb_name)
        self.llama_sampler_i.accept = llama_cpp.llama_sampler_accept_fn(lambda s, t: self._cb_accept(s, t))
        self.llama_sampler_i.apply = llama_cpp.llama_sampler_apply_fn(_cb_apply)
        self.llama_sampler_i.reset  = llama_cpp.llama_sampler_reset_fn(lambda s: self._cb_reset(s))
        self.llama_sampler_i.clone  = llama_cpp.llama_sampler_clone_fn(lambda s: self._cb_clone(s))
        self.llama_sampler_i.free   = llama_cpp.llama_sampler_free_fn(lambda s: self._cb_free(s))

        self.llama_sampler_i.backend_init = ctypes.cast(0, llama_cpp.llama_sampler_backend_init_fn)
        self.llama_sampler_i.backend_accept = ctypes.cast(0, llama_cpp.llama_sampler_backend_accept_fn)
        self.llama_sampler_i.backend_apply = ctypes.cast(0, llama_cpp.llama_sampler_backend_apply_fn)
        self.llama_sampler_i.backend_set_input = ctypes.cast(0, llama_cpp.llama_sampler_backend_set_input_fn)

        self.sampler_p = llama_cpp.llama_sampler_init(ctypes.pointer(self.llama_sampler_i), None)

    def get_sampler(self) -> llama_cpp.llama_sampler_p:
        return self.sampler_p


class LlamaSampler:
    def __init__(self, existing_sampler_p: Optional[llama_cpp.llama_sampler_p] = None):
        if existing_sampler_p:
            self.sampler = existing_sampler_p
        else:
            # Initialize new chain
            params = llama_cpp.llama_sampler_chain_params()
            params.no_perf = False
            self.sampler = llama_cpp.llama_sampler_chain_init(params)

        self.samplers: List[llama_cpp.llama_sampler_p] = []
        self.custom_samplers: List["CustomSampler"] = []

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
        llama_cpp.llama_sampler_accept(self.sampler, token)

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

        return LlamaSampler(existing_sampler_p=new_sampler_p)

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

    def close(self):
        if self.sampler:
            # NOTE: Must remove custom samplers before free or llama.cpp will try to free them
            for i, _ in reversed(self.custom_samplers):
                llama_cpp.llama_sampler_chain_remove(self.sampler, i)
            llama_cpp.llama_sampler_free(self.sampler)
            self.sampler = None
        self.samplers.clear()
        self.custom_samplers.clear()

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
        grammar: LlamaGrammar,
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
        c_grammar_str = grammar._grammar.encode('utf-8')
        c_root = grammar._root.encode('utf-8')

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

            self._add_sampler(llama_cpp.llama_sampler_init_grammar_lazy_patterns(
                model.vocab,
                c_grammar_str,
                c_root,
                c_trigger_patterns,
                len(trigger_patterns),
                c_trigger_tokens,
                len(trigger_tokens)
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

    def add_logit_bias(self, n_vocab: int, bias_dict: Dict[int, float]):
        """Logit bias sampler."""
        if not bias_dict: return

        c_bias = (llama_cpp.llama_logit_bias * len(bias_dict))()
        for i, (token, bias) in enumerate(bias_dict.items()):
            c_bias[i].token = token
            c_bias[i].bias = bias

        self._add_sampler(llama_cpp.llama_sampler_init_logit_bias(n_vocab, len(bias_dict), c_bias))

    def add_infill(self, model: LlamaModel):
        self._add_sampler(llama_cpp.llama_sampler_init_infill(model.vocab))

    def add_adaptive_p(self, target: float, decay: float, seed: int):
        self._add_sampler(llama_cpp.llama_sampler_init_adaptive_p(target, decay, seed))

    def add_custom(
        self, apply_func: Callable[[llama_cpp.llama_token_data_array], None]
    ):
        custom_sampler = CustomSampler(apply_func)
        sampler = custom_sampler.get_sampler()
        self._add_sampler(sampler)
        # NOTE: Must remove custom samplers before free or llama.cpp will try to free them
        self.custom_samplers.append(
            [llama_cpp.llama_sampler_chain_n(self.sampler) - 1, custom_sampler]
        )

    def get_seed(self) -> int:
        assert self.sampler is not None
        return llama_cpp.llama_sampler_get_seed(self.sampler)
