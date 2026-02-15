import llama_cpp
from llama_cpp import LLAMA_TOKEN_NULL

import llama_cpp.mtmd_cpp as mtmd
from .mtmd_cpp import mtmd_input_chunk_type, mtmd_free
from ._internals import LlamaContext, LlamaBatch

import ctypes
from typing import Union, List

class TextChunk:
    def __init__(self, tokens: List[int]):
        self.tokens = tokens
        self.n_tokens = len(tokens)

class MediaChunk:
    def __init__(self, chunk_ptr: ctypes.c_void_p):
        self.chunk_ptr = mtmd.mtmd_input_chunk_copy(chunk_ptr)
        self.n_tokens = mtmd.mtmd_input_chunk_get_n_tokens(self.chunk_ptr)

    def __del__(self):
        if self.chunk_ptr:
            mtmd.mtmd_input_chunk_free(self.chunk_ptr)

class MultimodalTokenList:
    def __init__(self):
        self.chunks: List[Union[TextChunk, MediaChunk]] = []
        self.total_tokens = 0

    def add(self, chunk_ptr: mtmd.mtmd_input_chunk_p_ctypes):
        chunk_type = mtmd.mtmd_input_chunk_get_type(chunk_ptr)

        if chunk_type in [
            mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE,
            mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_AUDIO
        ]:
            m_chunk = MediaChunk(chunk_ptr)
            self.chunks.append(m_chunk)
            self.total_tokens += m_chunk.n_tokens

        elif chunk_type == mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT:
            n_tokens_ref = ctypes.c_size_t()
            text_tokens_ptr = mtmd.mtmd_input_chunk_get_tokens_text(chunk_ptr, ctypes.byref(n_tokens_ref))
            tokens = [text_tokens_ptr[j] for j in range(n_tokens_ref.value)]
            self.add_text(tokens)

        else:
            raise ValueError(f"Invalid chunk type {chunk_type}")

    def add_text(self, tokens: List[int]):
        if not tokens: return
        # combine text nodes
        if self.chunks and isinstance(self.chunks[-1], TextChunk):
            self.chunks[-1].tokens.extend(tokens)
            self.chunks[-1].n_tokens += len(tokens)
        else:
            self.chunks.append(TextChunk(tokens))
        self.total_tokens += len(tokens)

    def __len__(self):
        return self.total_tokens


class MultiModalContext:
    def __init__(
            self,
            ctx
    ):
        self.ctx = ctx

    def close(self):
        if self.ctx is None:
            return
        mtmd_free(self.ctx)
        self.ctx = None

    def __del__(self):
        self.close()


# Simple FNV-1a hash implementation to match fnv_hash in C++
def fnv_hash(data: bytes) -> str:
    h = 0x811c9dc5
    for b in data:
        h = (h ^ b) * 0x01000193
        h &= 0xffffffff
    return f"{h:08x}"

def mtmd_tokenize(
    mctx: mtmd.mtmd_context_p,
    prompt: str,
    files_data: list[bytes | str]) -> MultimodalTokenList:

    bitmaps = []
    do_hash = False

    for data in files_data:

        bmp = None
        if isinstance(data, str):
            bmp = mtmd.mtmd_helper_bitmap_init_from_file(mctx, data.encode("utf-8"))
        elif isinstance(data, bytes):
            buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
            bmp = mtmd.mtmd_helper_bitmap_init_from_buf(mctx, buf, len(buf))
        elif isinstance(data, bytearray):
            buf = (ctypes.c_ubyte * len(data)).from_buffer(data)
            bmp = mtmd.mtmd_helper_bitmap_init_from_buf(mctx, buf, len(buf))

        if bmp is None:
            raise RuntimeError("Failed to load image or audio file")

        if do_hash:
            data_ptr = mtmd.mtmd_bitmap_get_data(bmp)
            data_size = mtmd.mtmd_bitmap_get_n_bytes(bmp)

            raw_node_data = ctypes.string_at(data_ptr, data_size)
            h = fnv_hash(raw_node_data)
            mtmd.mtmd_bitmap_set_id(bmp, h.encode('utf-8'))

        bitmaps.append(bmp)

    inp_txt = mtmd.mtmd_input_text(
        text=prompt.encode('utf-8'),
        add_special=True,
        parse_special=True
    )

    chunks_ptr = mtmd.mtmd_input_chunks_init()

    n_bitmaps = len(bitmaps)
    if n_bitmaps > 0:
        BitmapPtr = mtmd.mtmd_bitmap_p_ctypes * n_bitmaps
        bitmaps_ptr = BitmapPtr(*bitmaps)
    else:
        bitmaps_ptr = None

    res = mtmd.mtmd_tokenize(
        mctx,
        chunks_ptr,
        ctypes.pointer(inp_txt),
        bitmaps_ptr,
        n_bitmaps
    )

    # TODO Hash based cache
    for data in bitmaps:
        mtmd.mtmd_bitmap_free(bmp)

    if res != 0:
        mtmd.mtmd_input_chunks_free(chunks_ptr)
        raise RuntimeError(f"Tokenization failed with code {res}")

    st = MultimodalTokenList()

    n_chunks = mtmd.mtmd_input_chunks_size(chunks_ptr)
    for i in range(n_chunks):
        chunk_ptr = mtmd.mtmd_input_chunks_get(chunks_ptr, i)
        st.add(chunk_ptr)

    mtmd.mtmd_input_chunks_free(chunks_ptr)

    return st

def mtmd_prefill(
    ctx: LlamaContext,
    mctx: mtmd.mtmd_context_p,
    batch: LlamaBatch,
    mtmd_tokens: MultimodalTokenList
) -> int:
    n_past = 0
    n_batch = ctx.n_batch()
    total_chunks = len(mtmd_tokens.chunks)

    for i, chunk in enumerate(mtmd_tokens.chunks):
        is_last_chunk = (i == total_chunks - 1)

        if isinstance(chunk, TextChunk):
            batch.set_batch(
                chunk.tokens,
                n_past,
                logits_all=False,
                logits_last=is_last_chunk
            )
            ctx.decode(batch)

            n_past += chunk.n_tokens
        else:
            new_n_past = llama_cpp.llama_pos(0)
            result = mtmd.mtmd_helper_eval_chunk_single(
                mctx,
                ctx.ctx,
                chunk.chunk_ptr,
                llama_cpp.llama_pos(n_past),
                llama_cpp.llama_seq_id(0),
                n_batch,
                False, # logits_last
                ctypes.byref(new_n_past)
            )
            if result != 0:
                raise RuntimeError(f"MTMD eval error: {result}")

            n_past = new_n_past.value
