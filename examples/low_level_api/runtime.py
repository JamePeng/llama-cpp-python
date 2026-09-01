from __future__ import annotations

import codecs
import ctypes
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from typing_extensions import Self

# Use this checkout when an older llama_cpp is installed elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llama_cpp
from llama_cpp._ggml import ggml_backend_load_all_from_path
from llama_cpp._logger import configure_logging

_BACKEND_READY = False


def initialize_backend() -> None:
    """Initialize process-global backends once and discover packaged plugins."""
    global _BACKEND_READY
    if _BACKEND_READY:
        return

    llama_cpp.llama_backend_init()
    lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
    if not lib_dir.is_dir():
        raise RuntimeError(f"llama_cpp backend directory does not exist: {lib_dir}")

    # Dynamic wheels keep optimized CPU/GPU backends beside the Python package.
    ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))
    _BACKEND_READY = True


class LowLevelLlama:
    """Small owner for the raw model, context, batch, and sampler APIs."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int,
        n_batch: int,
        n_ubatch: int,
        n_threads: int,
        n_gpu_layers: int,
        verbose: bool | None = None,
        verbosity: str | None = None,
    ) -> None:
        if not 0 < n_ubatch <= n_batch <= n_ctx:
            raise ValueError("expected 0 < n_ubatch <= n_batch <= n_ctx")

        self.model = None
        self.context = None
        self.vocab = None
        self.memory = None
        self.batch = None
        self.n_batch = n_batch
        self._closed = False

        # Fine-grained verbosity takes precedence over the compatibility flag.
        configure_logging(verbose=verbose, verbosity=verbosity)
        initialize_backend()
        try:
            model_params = llama_cpp.llama_model_default_params()
            model_params.n_gpu_layers = n_gpu_layers
            self.model = llama_cpp.llama_model_load_from_file(
                str(model_path).encode("utf-8"), model_params
            )
            if not self.model:
                raise RuntimeError(f"failed to load model: {model_path}")

            self.vocab = llama_cpp.llama_model_get_vocab(self.model)
            if not self.vocab:
                raise RuntimeError("the model does not expose a vocabulary")
            if (
                llama_cpp.llama_model_has_encoder(self.model)
                or not llama_cpp.llama_model_has_decoder(self.model)
                or llama_cpp.llama_model_is_diffusion(self.model)
            ):
                raise RuntimeError("these examples require a decoder-only text model")

            context_params = llama_cpp.llama_context_default_params()
            # n_batch is logical capacity; n_ubatch limits each physical graph.
            context_params.n_ctx = n_ctx
            context_params.n_batch = n_batch
            context_params.n_ubatch = n_ubatch
            context_params.n_threads = n_threads
            context_params.n_threads_batch = n_threads
            self.context = llama_cpp.llama_init_from_model(self.model, context_params)
            if not self.context:
                raise RuntimeError("failed to create the llama context")

            self.memory = llama_cpp.llama_get_memory(self.context)
            if not self.memory:
                raise RuntimeError("the context does not expose token memory")

            self.batch = llama_cpp.llama_batch_init(n_batch, 0, 1)
        except BaseException:
            self.close()
            raise

    @property
    def context_size(self) -> int:
        assert self.context is not None
        return llama_cpp.llama_n_ctx(self.context)

    def tokenize(
        self, text: str, *, add_special: bool, parse_special: bool
    ) -> list[int]:
        assert self.vocab is not None
        data = text.encode("utf-8")
        capacity = max(16, len(data) + 16)
        tokens = (llama_cpp.llama_token * capacity)()
        count = llama_cpp.llama_tokenize(
            self.vocab,
            data,
            len(data),
            tokens,
            capacity,
            add_special,
            parse_special,
        )
        if count < 0:
            capacity = -count
            tokens = (llama_cpp.llama_token * capacity)()
            count = llama_cpp.llama_tokenize(
                self.vocab,
                data,
                len(data),
                tokens,
                capacity,
                add_special,
                parse_special,
            )
        if count < 0:
            raise RuntimeError("tokenization failed")
        return list(tokens[:count])

    def token_piece(self, token: int) -> bytes:
        assert self.vocab is not None
        capacity = 32
        buffer = ctypes.create_string_buffer(capacity)
        count = llama_cpp.llama_token_to_piece(
            self.vocab, token, buffer, capacity, 0, False
        )
        if count < 0:
            capacity = -count
            buffer = ctypes.create_string_buffer(capacity)
            count = llama_cpp.llama_token_to_piece(
                self.vocab, token, buffer, capacity, 0, False
            )
        if count < 0:
            raise RuntimeError(f"failed to decode token {token}")
        return bytes(buffer.raw[:count])

    def render_chat(
        self, messages: Sequence[tuple[str, str]], *, add_assistant: bool = True
    ) -> str:
        assert self.model is not None
        template = llama_cpp.llama_model_chat_template(self.model, None)
        if not template:
            raise RuntimeError("the model does not contain a supported chat template")

        encoded = [(role.encode(), content.encode()) for role, content in messages]
        chat = (llama_cpp.llama_chat_message * len(encoded))(
            *(llama_cpp.llama_chat_message(role, content) for role, content in encoded)
        )
        capacity = max(
            1024, sum(len(role) + len(content) for role, content in encoded) * 2
        )
        buffer = ctypes.create_string_buffer(capacity)
        count = llama_cpp.llama_chat_apply_template(
            template, chat, len(chat), add_assistant, buffer, capacity
        )
        if count >= capacity:
            capacity = count + 1
            buffer = ctypes.create_string_buffer(capacity)
            count = llama_cpp.llama_chat_apply_template(
                template, chat, len(chat), add_assistant, buffer, capacity
            )
        if count < 0:
            raise RuntimeError("failed to apply the model chat template")
        return bytes(buffer.raw[:count]).decode("utf-8")

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        seed: int,
        add_special: bool = True,
        parse_special: bool = False,
    ) -> Iterator[str]:
        assert self.context is not None
        assert self.memory is not None
        assert self.vocab is not None

        prompt_tokens = self.tokenize(
            prompt, add_special=add_special, parse_special=parse_special
        )
        if not prompt_tokens:
            raise ValueError("the prompt produced no tokens")
        if len(prompt_tokens) + max_tokens > self.context_size:
            available = max(0, self.context_size - len(prompt_tokens))
            raise ValueError(
                f"prompt uses {len(prompt_tokens)} tokens, leaving {available} "
                "generation tokens; reduce --max-tokens or increase --n-ctx"
            )

        llama_cpp.llama_memory_clear(self.memory, True)
        sampler = self._new_sampler(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        try:
            self._decode(prompt_tokens, start_pos=0)

            position = len(prompt_tokens)
            for index in range(max_tokens):
                # Sampling also accepts the selected token into sampler history.
                token = llama_cpp.llama_sampler_sample(sampler, self.context, -1)
                if llama_cpp.llama_vocab_is_eog(self.vocab, token):
                    break

                text = decoder.decode(self.token_piece(token), final=False)
                if text:
                    yield text

                if index + 1 < max_tokens:
                    self._decode([token], start_pos=position)
                    position += 1

            try:
                tail = decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                # A token limit can stop in the middle of a byte-fallback sequence.
                tail = ""
            if tail:
                yield tail
        finally:
            llama_cpp.llama_sampler_free(sampler)

    def print_timings(self) -> None:
        assert self.context is not None
        llama_cpp.llama_perf_context_print(self.context)

    def _decode(self, tokens: Sequence[int], *, start_pos: int) -> None:
        assert self.batch is not None
        assert self.context is not None

        offset = 0
        chunk_size = min(self.n_batch, len(tokens))
        while offset < len(tokens):
            chunk = tokens[offset : offset + chunk_size]
            self.batch.n_tokens = len(chunk)
            for index, token in enumerate(chunk):
                self.batch.token[index] = token
                self.batch.pos[index] = start_pos + offset + index
                self.batch.n_seq_id[index] = 1
                self.batch.seq_id[index][0] = 0
                self.batch.logits[index] = index == len(chunk) - 1

            result = llama_cpp.llama_decode(self.context, self.batch)
            if result == 0:
                offset += len(chunk)
                continue
            if result == 1 and chunk_size > 1:
                # A smaller retry can fit when the memory has no slot for a large batch.
                chunk_size = max(1, chunk_size // 2)
                continue

            messages = {
                1: "no context-memory slot is available",
                2: "decoding was aborted by a callback",
                -1: "the input batch is invalid",
                -2: "the compute graph could not be allocated",
                -3: "graph computation failed",
            }
            detail = messages.get(result, "an unknown native error occurred")
            raise RuntimeError(f"llama_decode failed with code {result}: {detail}")

    def _new_sampler(
        self,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        seed: int,
    ):
        assert self.vocab is not None
        params = llama_cpp.llama_sampler_chain_default_params()
        sampler = llama_cpp.llama_sampler_chain_init(params)
        if not sampler:
            raise RuntimeError("failed to create sampler chain")

        def add(child) -> None:
            if not child:
                raise RuntimeError("failed to initialize a sampler")
            # The chain owns child after this call and frees it with the chain.
            llama_cpp.llama_sampler_chain_add(sampler, child)

        try:
            n_vocab = llama_cpp.llama_vocab_n_tokens(self.vocab)
            add(
                llama_cpp.llama_sampler_init_penalties(
                    n_vocab, 64, repeat_penalty, 0.0, 0.0
                )
            )
            if temperature <= 0:
                add(llama_cpp.llama_sampler_init_greedy())
            else:
                add(llama_cpp.llama_sampler_init_top_k(top_k))
                add(llama_cpp.llama_sampler_init_top_p(top_p, 1))
                add(llama_cpp.llama_sampler_init_temp(temperature))
                actual_seed = seed if seed >= 0 else llama_cpp.LLAMA_DEFAULT_SEED
                add(llama_cpp.llama_sampler_init_dist(actual_seed))
            return sampler
        except BaseException:
            llama_cpp.llama_sampler_free(sampler)
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.batch is not None:
            llama_cpp.llama_batch_free(self.batch)
            self.batch = None
        if self.context is not None:
            llama_cpp.llama_free(self.context)
            self.context = None
        if self.model is not None:
            llama_cpp.llama_model_free(self.model)
            self.model = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
