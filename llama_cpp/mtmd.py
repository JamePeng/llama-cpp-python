import llama_cpp
from llama_cpp import LLAMA_TOKEN_NULL

import llama_cpp.mtmd_cpp as mtmd
from .mtmd_cpp import mtmd_input_chunk_type, mtmd_free
from ._internals import LlamaContext, LlamaBatch

import ctypes
from typing import Union, List, Optional, Any, Tuple

import llama_cpp.llama_types as llama_types
import llama_cpp.llama as llama
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
import copy
import numpy as np
import numpy.typing as npt
import os

from .llama_chat_format import ChatFormatter, ChatFormatterResponse

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

DEFAULT_MEDIA_MARKER = mtmd.mtmd_default_marker().decode('utf-8')

class Jinja2MultimodalChatFormatter(ChatFormatter):
    def __init__(
            self,
            template: str,
            eos_token: str,
            bos_token: str,
            add_generation_prompt: bool = True,
            stop_token_ids: Optional[List[int]] = None,
            placeholders: List[str] = None
    ):
        """A chat formatter that uses jinja2 templates to format the prompt."""
        self.template = template
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.add_generation_prompt = add_generation_prompt
        self.stop_token_ids = (
            set(stop_token_ids) if stop_token_ids is not None else None
        )

        self.chat_template = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True
        ).from_string(template)

        # Placeholder mapping, mtmd_tokenize requires <__media__>
        self.placeholders = placeholders if placeholders else [
            "<|vision_start|><|image_pad|><|vision_end|>", # Qwen3-VL
            "<image>",            # LLaVA / Yi
            "<image_placeholder>",# DeepSeek
        ]

    def __call__(
            self,
            messages: List[llama_types.ChatCompletionRequestMessage],
            functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
            function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
            tools: Optional[List[llama_types.ChatCompletionTool]] = None,
            tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
            **kwargs: Any,
    ) -> Tuple[str, List[Union[str, bytes, bytearray]], List[str]]:
        def raise_exception(message: str):
            raise ValueError(message)

        def strftime_now(format_string="%Y-%m-%d %H:%M:%S") -> str:
            """
            Returns the current time formatted as a string.
            """
            return datetime.datetime.now().strftime(format_string)

        messages = copy.deepcopy(messages)
        media_urls, media_types = self.split_media(messages)
        medias = []

        for url, m_type in zip(media_urls, media_types):
            if m_type == "video":
                raise ValueError("Video input is not supported yet.")

            data = self._fetch_media(url, m_type)

            #if m_type == "image" and isinstance(data, bytes):
            #    data = self._compress_image(data)

            medias.append(data)

        prompt = self.chat_template.render(
            messages=messages,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            raise_exception=raise_exception,
            strftime_now=strftime_now,
            add_generation_prompt=self.add_generation_prompt,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
        )

        for p in self.placeholders:
            prompt = prompt.replace(p, DEFAULT_MEDIA_MARKER)

        stopping_criteria = None
        if self.stop_token_ids is not None:

            def stop_on_last_token(
                    tokens: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
            ) -> bool:
                return tokens[-1] in self.stop_token_ids

            stopping_criteria = llama.StoppingCriteriaList([stop_on_last_token])

        return ChatFormatterResponse(
            prompt=prompt,
            stop=[self.eos_token],
            stopping_criteria=stopping_criteria,
            added_special=True,
            medias=medias,
            media_types=media_types
        )

    @staticmethod
    def split_media(messages: List[llama_types.ChatCompletionRequestMessage]):
        media_urls: List[Union[str, bytes, bytearray]] = []
        media_types: List[str] = []

        for message in messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue

            for content in message["content"]:
                if not (isinstance(content, dict) and "type" in content):
                    continue

                c_type = content["type"]
                if c_type == "text":
                    continue

                value = content[c_type]

                if isinstance(value, dict) and "url" in value:
                    media_urls.append(value["url"])
                    value["url"] = DEFAULT_MEDIA_MARKER
                else:
                    media_urls.append(value)
                    content[c_type] = DEFAULT_MEDIA_MARKER

                if c_type == "image" or c_type == "image_url":
                    media_types.append("image")

                elif c_type == "audio" or c_type == "audio_url":
                    media_types.append("audio")

                elif c_type == "video" or c_type == "video_url":
                    media_types.append("video")

                else:
                    raise ValueError(f"Unsupported content type {c_type}")

        return media_urls, media_types

    @staticmethod
    def _fetch_media(media_input: Union[str, bytes], media_type: str) -> Union[str, bytes, bytearray]:
        """
        Fetch media (audio, image, video...) from local disk, memory, or internet
        """

        # --- from_buffer fast path ---
        if isinstance(media_input, bytes) or isinstance(media_input, bytearray):
            return media_input

        if not isinstance(media_input, str):
            raise ValueError(f"Unsupported media input type: {type(media_input)}")

        # --- from_file fast path ---
        if media_input.startswith("file://"):
            parsed_path = urllib.parse.urlparse(media_input).path
            # unquote 处理 URL 编码的字符
            abs_path = os.path.abspath(urllib.parse.unquote(parsed_path))
            if os.path.exists(abs_path):
                return abs_path
            else:
                raise FileNotFoundError(f"Local file not found: {abs_path}")

        # --- base64 or remote url ---
        raw_bytes = b""
        if media_input.startswith("data:"):
            import base64
            # Split only once from the right to correctly handle mime types containing commas
            comma_pos = media_input.find(",")
            if comma_pos == -1:
                raise ValueError("Invalid data URI: missing comma separator")

            raw_bytes = base64.b64decode(media_input[comma_pos+1:])
        elif "://" in media_input:
            import urllib.request
            from urllib.error import URLError, HTTPError

            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(media_input, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=15) as f:
                    raw_bytes = f.read()
            except (URLError, HTTPError) as e:
                raise ConnectionError(f"Failed to fetch media from {media_input}: {e}")

        else:
            # try direct path
            if os.path.exists(media_input):
                return os.path.abspath(media_input)
            raise ValueError("Unrecognized media string format")

        if not raw_bytes:
            raise ValueError("Empty data received")

        return raw_bytes

    @staticmethod
    def _compress_image(image_bytes: bytes) -> bytes:
        try:
            from PIL import Image, ImageStat
        except ImportError:
            raise ImportError("Pillow is required for image processing. Install with: pip install pillow")

        import io
        image = Image.open(io.BytesIO(image_bytes))

        # 4. Handle transparency (RGBA, LA, P with transparency, etc.)
        if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
            # Use alpha channel as mask
            if image.mode == "P":
                image = image.convert("RGBA")

            alpha = image.split()[-1]  # Last channel is alpha
            # Compute average brightness of visible (non-transparent) pixels
            stat = ImageStat.Stat(image.convert("L"), mask=alpha)

            # Choose background: white for dark content, black for bright content
            bg_color = (255, 255, 255)  # white
            if stat.count[0] > 0 and stat.mean[0] > 127:
                bg_color = (0, 0, 0)  # black

            background = Image.new("RGB", image.size, bg_color)
            background.paste(image, mask=alpha)
            image = background

        # 5. Ensure RGB mode for formats like CMYK, palette, etc.
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 6. Save as high-quality JPEG, suitable for most vision models.
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95, optimize=True, progressive=True)
        return output.getvalue()

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
    medias_data: list[Union[str, bytes, bytearray]]) -> MultimodalTokenList:

    bitmaps = []
    do_hash = False

    for data in medias_data:

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

    return n_past