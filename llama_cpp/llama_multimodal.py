from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
import zlib

from contextlib import ExitStack
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Protocol,
    TYPE_CHECKING,
    cast,
)

import urllib.request
from urllib.error import URLError, HTTPError

import llama_cpp.llama_cpp as llama_cpp_lib
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar

if TYPE_CHECKING:
    import llama_cpp.llama as llama_core

from ._logger import ggml_log_callback

from llama_cpp.llama_chat_format import (
    _convert_completion_to_chat,
    _convert_completion_to_chat_function,
    _grammar_for_response_format,
    ImmutableSandboxedEnvironment
)

class MTMDChatHandler:
    DEFAULT_SYSTEM_MESSAGE: Optional[str] = (
"You are an exceptionally capable, precise, and helpful multimodal AI assistant that excels at deeply understanding and richly describing images, charts, diagrams, text in images, scenes, and any visual content, "
"while also answering every question accurately, clearly, and step-by-step when appropriate — always responding in the same language as the user's question, remaining polite, professional, and maximally helpful."
    )

    CHAT_FORMAT = (
        "{{ bos_token if bos_token is defined else '' }}"
        "{% for message in messages %}"
            "{% if message.role == 'system' %}"
                "{{ message.content }}"
            "{% elif message.role == 'user' %}"
                "USER: "
                "{% if message.content is string %}"
                    "{{ message.content }}"
                "{% elif message.content is iterable %}"
                    "{% for content in message.content %}"
                        "{% if content.type == 'image_url' %}"
                            "{{ content.image_url if content.image_url is string else content.image_url.url }}"
                        "{% elif content.type == 'audio_url' %}"
                            "{{ content.audio_url if content.audio_url is string else content.audio_url.url }}"
                        "{% elif content.type == 'input_audio' %}"
                            "{% if content.input_audio is string %}"
                                "{{ content.input_audio }}"
                            "{% else %}"
                                "data:audio/{{ content.input_audio.format }};base64,{{ content.input_audio.data }}"
                            "{% endif %}"
                        "{% elif content.type == 'video_url' %}"
                            "{{ content.video_url if content.video_url is string else content.video_url.url }}"
                        "{% elif content.type == 'text' %}"
                            "{{ content.text }}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"

            "{% elif message.role == 'assistant' and message.content is not none %}"
                "ASSISTANT: {{ message.content }}"
            "{% endif %}"
            "{{ \"\n\" }}"
        "{% endfor %}"

        "{% if eos_token is defined %}"
            "{{ eos_token }}"
        "{% endif %}"

        "{% if add_generation_prompt %}"
            "ASSISTANT: "
        "{% endif %}"
    )

    KNOWN_MEDIA_TAGS: List[str] = []

    def __init__(
        self,
        mmproj_path: Optional[str] = None,
        verbose: bool = True,
        use_gpu: bool = True,
        image_min_tokens: int = -1,
        image_max_tokens: int = -1,
        chat_template_override: Optional[str] = None,
        batch_max_tokens: int = 1024,
        extra_template_arguments: Optional[Dict[str, Any]] = None,
        **kwargs
    ):

        self.log_prefix = self.__class__.__name__
        self.verbose = verbose

        # Backward compatibility: `clip_model_path` was the old name for `mmproj_path`.
        # Accept it for existing user code, warn during initialization, and normalize
        # all internal usage to `mmproj_path`.
        clip_model_path = kwargs.pop("clip_model_path", None)
        if mmproj_path is None and clip_model_path is not None:
            mmproj_path = clip_model_path
            if self.verbose:
                print(
                    f"{self.log_prefix}(__init__): `clip_model_path` is deprecated; "
                    "please use `mmproj_path` instead.",
                    file=sys.stderr,
                )

        if kwargs:
            unexpected_args = ", ".join(f"'{k}'" for k in kwargs.keys())
            raise TypeError(
                f"Initialization Error in {self.log_prefix}: Received unexpected keyword argument(s) {unexpected_args}.\n"
                f"If you are passing model-specific parameters, ensure they are supported by {self.log_prefix}."
            )

        if mmproj_path is None:
            raise ValueError(
                f"{self.log_prefix}(__init__): `mmproj_path` is required. "
                "`clip_model_path` is accepted only as a deprecated compatibility alias."
            )

        self.mmproj_path = mmproj_path
        if not os.path.exists(self.mmproj_path):
            raise ValueError(
                f"{self.log_prefix}(__init__): mmproj path does not exist: {self.mmproj_path}"
            )

        self.image_min_tokens = image_min_tokens
        self.image_max_tokens = image_max_tokens
        self.batch_max_tokens = batch_max_tokens
        self.use_gpu = use_gpu

        import llama_cpp.mtmd_cpp as mtmd_cpp
        self._mtmd_cpp = mtmd_cpp
        self.mtmd_ctx: Optional[mtmd_cpp.mtmd_context_p] = None

        if extra_template_arguments is not None and not isinstance(extra_template_arguments, dict):
            raise TypeError(
                f"{self.log_prefix}(__init__): `extra_template_arguments` must be a dict."
            )

        # Preserve subclass attributes
        if not hasattr(self, "chat_format"):
            self.chat_format = None

        self.chat_format_override = chat_template_override
        self.extra_template_arguments: dict[str, Any] = dict(extra_template_arguments or {})

        self.is_support_vision = False
        self.is_support_audio = False
        self.is_support_video = False

        self.chat_template = None
        self._chat_format_parser_tags = []
        self._template_initialized = False

        # Pre-compile Jinja template
        if self.chat_format is None:
            if self.chat_format_override is not None:
                self.chat_format = self.chat_format_override
            else:
                self.chat_format = self.CHAT_FORMAT

        self._change_chat_template(self.chat_format)

        self._exit_stack = ExitStack()

    def _change_chat_template(self, new_template: str):
        self.chat_template = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True
        ).from_string(new_template)

    def _init_mtmd_context(self, llama_model: llama_core.Llama):
        """Initialize mtmd context with the llama model."""
        if self.mtmd_ctx is not None:
            return  # Already initialized

        self._mtmd_cpp.mtmd_helper_log_set(ggml_log_callback, ctypes.c_void_p(0))

        # Get default parameters
        self.mctx_params = self._mtmd_cpp.mtmd_context_params_default()
        self.mctx_params.use_gpu = self.use_gpu
        self.mctx_params.print_timings = self.verbose
        self.mctx_params.n_threads = llama_model.n_threads
        self.mctx_params.flash_attn_type = self._mtmd_cpp.clip_flash_attn_type.CLIP_FLASH_ATTN_TYPE_AUTO
        self.mctx_params.warmup = True
        if self.image_min_tokens > 0:
            self.mctx_params.image_min_tokens = self.image_min_tokens
        if self.image_max_tokens > 0:
            self.mctx_params.image_max_tokens = self.image_max_tokens
        if (self.image_max_tokens < self.image_min_tokens) and self.image_max_tokens > 0:
            raise ValueError(f"{self.log_prefix}(_init_mtmd_context): Configuration Error! image_max_tokens ({self.image_max_tokens}) "
                                f"cannot be less than image_min_tokens ({self.image_min_tokens}).")
        self.mctx_params.batch_max_tokens = self.batch_max_tokens

        # Cache the model's eos token and bos token
        self.mtmd_eos_token=llama_model.detokenize([llama_model.token_eos()]).decode('utf-8', errors='ignore')
        self.mtmd_bos_token=llama_model.detokenize([llama_model.token_bos()]).decode('utf-8', errors='ignore')

        # Cache the mtmd_default_marker
        self.media_marker = self._mtmd_cpp.mtmd_default_marker().decode('utf-8')

        # Initialize mtmd context
        self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
            self.mmproj_path.encode(),
            llama_model.model,
            self.mctx_params
        )

        if self.mtmd_ctx is None:
            raise ValueError(f"{self.log_prefix}(_init_mtmd_context): Failed to load mtmd context from: {self.mmproj_path}")

        # Check if vision is supported
        self.is_support_vision = self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx)
        if self.is_support_vision:
            if self.verbose:
                print(f"{self.log_prefix}(_init_mtmd_context): Vision support detected.", file=sys.stderr)
        else:
            if self.verbose:
                print(f"{self.log_prefix}(_init_mtmd_context): Vision is NOT supported by this mmproj model backend.", file=sys.stderr)

        # Check if audio is supported
        self.is_support_audio = self._mtmd_cpp.mtmd_support_audio(self.mtmd_ctx)
        if self.is_support_audio:
            if self.verbose:
                print(f"{self.log_prefix}(_init_mtmd_context): Audio support detected.", file=sys.stderr)
        else:
            if self.verbose:
                print(f"{self.log_prefix}(_init_mtmd_context): Audio is NOT supported by this mmproj model backend.", file=sys.stderr)

        # Check if video is supported
        self.is_support_video = self._mtmd_cpp.mtmd_helper_support_video(self.mtmd_ctx)
        if self.is_support_video:
            if self.verbose:
                print(f"{self.log_prefix}(_init_mtmd_context): Video support detected.", file=sys.stderr)
        else:
            if self.verbose:
                print(f"{self.log_prefix}(_init_mtmd_context): Video support is NOT available in this build.", file=sys.stderr)

    def close(self) -> None:
        """Explicitly free the mtmd context and vision model resources."""
        if getattr(self, "mtmd_ctx", None) is not None:
            try:
                self._mtmd_cpp.mtmd_free(self.mtmd_ctx)
                self.mtmd_ctx = None
            except Exception:
                pass
        self.mctx_params = None
        self.chat_format = None
        self.chat_template = None
        self.chat_template_override = None
        self._template_initialized = False
        self._chat_format_parser_tags = []

        if getattr(self, "_exit_stack", None) is not None and hasattr(self._exit_stack, "close"):
            self._exit_stack.close()
            self._exit_stack = None

    def __del__(self) -> None:
        self.close()

    def _get_media_url(
        self,
        content: Dict[str, Any],
        keys: Tuple[str, ...],
        media_type: str,
    ) -> str:
        """
        Extract a media URL or data URI from a multimodal content item.

        Different chat templates and client APIs may represent the same media
        payload with slightly different keys. For example, an image may appear as
        `image`, `image_url`, or a typed chunk with `{"type": "image", ...}`.
        This helper checks the provided keys in order and returns the first usable
        media payload.

        Returns an empty string when none of the requested keys exist or when the
        payload shape is unsupported. The caller is responsible for raising a
        media-type-specific error when an empty value is not acceptable.
        """
        # Try keys in priority order. This lets callers prefer canonical fields
        # such as "image" over compatibility aliases such as "image_url", while
        # still accepting either representation.
        value = None
        for key in keys:
            if key in content:
                value = content[key]
                break

        # String payloads may already be URLs, local paths, or data URIs.
        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            # Common OpenAI-style shape:
            # {"image_url": {"url": "..."}}
            if "url" in value:
                return value["url"]

            # Forward-compatible inline media shape:
            # {"audio": {"data": "...", "format": "wav"}}
            #
            # Convert it to a data URI so downstream media loading does not need
            # separate branches for raw base64 payloads.
            if "data" in value and "format" in value:
                media_format = value.get("format", "")
                media_data = value.get("data", "")
                if media_format and media_data:
                    return f"data:{media_type}/{media_format};base64,{media_data}"

        return ""

    def _get_media_items(
        self,
        messages: List[llama_types.ChatCompletionRequestMessage],
    ) -> List[Dict[str, str]]:
        """
        Extract media payloads from chat messages in message/content order.

        Supports OpenAI-style typed media chunks as well as template-friendly
        variants used by multimodal chat templates, such as:
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image", "image": "..."}
        - {"image": "..."}
        - {"type": "audio_url", "audio_url": {"url": "..."}}
        - {"type": "audio", "audio": "..."}
        - {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}
        - {"type": "video_url", "video_url": {"url": "..."}}
        - {"type": "video", "video": "..."}
        - {"video": "..."}

        The returned order must match the media placeholders emitted by the rendered
        chat template as closely as possible.
        """
        media_items: List[Dict[str, str]] = []

        for message in messages:
            content_list = message.get("content")
            if not isinstance(content_list, list):
                continue

            for content in content_list:
                if not isinstance(content, dict):
                    continue

                content_type = content.get("type", "")

                has_image = (
                    content_type in ("image", "image_url")
                    or "image" in content
                    or "image_url" in content
                )
                has_audio = (
                    content_type in ("audio", "audio_url", "input_audio")
                    or "audio" in content
                    or "audio_url" in content
                    or "input_audio" in content
                )
                has_video = (
                    content_type in ("video", "video_url")
                    or "video" in content
                    or "video_url" in content
                )

                media_kind_count = int(has_image) + int(has_audio) + int(has_video)
                if media_kind_count > 1:
                    raise ValueError(
                        f"{self.log_prefix}: content item contains multiple media types; "
                        "each content item must contain only one of image, audio, or video."
                    )

                # 1. Vision Processing
                if has_image:
                    if not self.is_support_vision:
                        raise ValueError(
                            f"{self.log_prefix}: This mmproj model instance does not support image inputs."
                        )

                    url = self._get_media_url(
                        content,
                        keys=("image", "image_url"),
                        media_type="image",
                    )
                    if not url:
                        raise ValueError(f"{self.log_prefix}: missing image url/data.")

                    media_items.append({"url": url, "type": "image"})

                # 2. Audio Processing
                elif has_audio:
                    if not self.is_support_audio:
                        raise ValueError(
                            f"{self.log_prefix}: This mmproj model instance does not support audio inputs."
                        )

                    if content_type == "input_audio" or "input_audio" in content:
                        input_audio = content.get("input_audio", {})

                        if isinstance(input_audio, dict) and "data" in input_audio:
                            audio_data = input_audio.get("data", "")
                            audio_format = input_audio.get("format", "")

                            # Strictly align with llama.cpp.
                            if audio_format not in ["wav", "mp3"]:
                                raise ValueError(
                                    f"{self.log_prefix}: input_audio.format must be either 'wav' or 'mp3'"
                                )

                            url = f"data:audio/{audio_format};base64,{audio_data}"
                        else:
                            url = input_audio if isinstance(input_audio, str) else ""
                    else:
                        url = self._get_media_url(
                            content,
                            keys=("audio", "audio_url"),
                            media_type="audio",
                        )

                    if not url:
                        raise ValueError(f"{self.log_prefix}: missing audio url/data.")

                    media_items.append({"url": url, "type": "audio"})

                # 3. Video Processing
                elif has_video:
                    if not self.is_support_video:
                        raise ValueError(
                            f"{self.log_prefix}: This libmtmd build does not support video inputs."
                        )

                    url = self._get_media_url(
                        content,
                        keys=("video", "video_url"),
                        media_type="video",
                    )
                    if not url:
                        raise ValueError(f"{self.log_prefix}: missing video url/data.")

                    media_items.append({"url": url, "type": "video"})

                # 4. Text & Unknown Types
                elif content_type == "text" or "text" in content:
                    continue
                else:
                    if self.verbose:
                        print(
                            f"{self.log_prefix}: ignored unknown content type '{content_type}'.",
                            file=sys.stderr,
                        )

        return media_items

    def _create_bitmap_from_bytes(self, media_bytes: bytes):
        """
        Constructs an mtmd_bitmap structure from a raw byte buffer containing media data.

        Supported formats:
          - Images (via stb_image): jpg, png, bmp, etc.
          - Audio (via miniaudio): wav, mp3, flac.
          - Video: depends on whether MTMD_VIDEO was enabled at build time.

        Note:
          - Media types (Image vs. Audio) are auto-detected by the C++ backend using magic bytes.
          - The underlying C++ helper function is thread-safe, making it suitable for concurrent preprocessing.

        Args:
            media_bytes (bytes): The raw byte content of the media file.

        Returns:
            bitmap: mtmd_bitmap *
            video_ctx: mtmd_helper_video * or NULL
        """
        if self.mtmd_ctx is None:
            raise ValueError(f"{self.log_prefix}(_create_bitmap_from_bytes): mtmd context not initialized.")

        if not media_bytes:
            raise ValueError(f"{self.log_prefix}(_create_bitmap_from_bytes): empty media bytes.")

        buf = (ctypes.c_uint8 * len(media_bytes)).from_buffer_copy(media_bytes)

        wrapper = self._mtmd_cpp.mtmd_helper_bitmap_init_from_buf(
            self.mtmd_ctx,
            buf,
            len(media_bytes),
            False,
        )

        if not wrapper.bitmap:
            if wrapper.video_ctx:
                self._mtmd_cpp.mtmd_helper_video_free(wrapper.video_ctx)

            raise ValueError(
                f"{self.log_prefix}(_create_bitmap_from_bytes): "
                "Failed to load media from bytes "
                "(unsupported media format, corrupted data, or missing helper support)."
            )

        return wrapper.bitmap, wrapper.video_ctx

    def _is_text_chunk(self, chunk_type: int) -> bool:
        """Return True if `chunk_type` is the MTMD text chunk type enum value."""
        return (
            chunk_type
            == self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT
        )

    def _is_image_chunk(self, chunk_type: int) -> bool:
        """Return True if `chunk_type` is the MTMD image chunk type enum value."""
        return (
            chunk_type
            == self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE
        )

    def _is_audio_chunk(self, chunk_type: int) -> bool:
        """Return True if `chunk_type` is the MTMD audio chunk type enum value."""
        return (
            chunk_type
            == self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_AUDIO
        )

    def _render_mtmd_prompt(
        self,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render the chat template into plain prompt text.

        This stage only renders the Jinja template. It does not normalize media
        placeholders or replace media URLs with the MTMD runtime marker.
        """
        return self.chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            eos_token=self.mtmd_eos_token,
            bos_token=self.mtmd_bos_token,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            **getattr(self, "extra_template_arguments", {}),
        )

    def _replace_media_placeholders(
        self,
        text: str,
        media_items: List[Dict[str, str]],
    ) -> str:
        """
        Normalize rendered media placeholders and media URLs into the MTMD runtime marker.

        llama.cpp MTMD tokenization recognizes the canonical media marker, usually
        `<__media__>`. Model chat templates may render media as model-specific tags
        such as `<image>`, `<|image|>`, `[IMG]`, `<|image_pad|>`, or as the original
        URL/data URI. This stage converts those rendered forms into the canonical
        MTMD marker and validates that the final marker count matches the number of
        media payloads.
        """
        media_marker = self.media_marker
        if not media_marker:
            raise ValueError(
                f"{self.log_prefix}(_replace_media_placeholders): media marker must not be empty."
            )

        # 1. Replace known template-specific media tags first.
        #
        # This handles templates that render placeholders such as:
        #   <image>, <|image|>, [IMG], <|image_pad|>, <|media_pad|>, etc.
        for tag in self._chat_format_parser_tags:
            if tag in text:
                text = text.replace(tag, media_marker)

        # 2. Replace rendered media URLs/data URIs.
        #
        # This handles templates that directly render the original image/audio/video
        # URL or data URI instead of a symbolic placeholder.
        for item in media_items:
            url = item.get("url", "")
            if url and url in text:
                text = text.replace(url, media_marker, 1)

        # 3. Validate after all normalization is complete.
        marker_count = text.count(media_marker)
        media_count = len(media_items)

        if marker_count != media_count:
            raise ValueError(
                f"{self.log_prefix}(_replace_media_placeholders): media marker mismatch\n"
                f"- marker_count={marker_count}\n"
                f"- media_count={media_count}\n"
                f"- media_marker={media_marker!r}\n"
                "Each media item must render to exactly one MTMD media marker. "
                "Check whether the chat template rendered both a media tag and the "
                "original URL/data URI, or failed to render a media placeholder."
            )

        return text

    def _render_and_replace_media(
        self,
        messages: List[llama_types.ChatCompletionRequestMessage],
        media_items: List[Dict[str, str]],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render chat messages and normalize rendered media placeholders into MTMD markers.
        """
        text = self._render_mtmd_prompt(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            add_generation_prompt=add_generation_prompt,
        )

        return self._replace_media_placeholders(
            text=text,
            media_items=media_items,
        )

    def _validate_mtmd_inputs(
        self,
        *,
        text: str,
        bitmaps: Optional[List[Any]] = None,
    ) -> None:
        """
        Validate Python-side MTMD tokenizer inputs before calling mtmd_tokenize.

        This mirrors the most important checks in llama.cpp mtmd_tokenizer:
        - mtmd context must be initialized
        - rendered text must be a string
        - media marker must not be empty
        - media marker count must match bitmap count
        - bitmap entries must not be None

        Pure text input is valid:
            bitmaps is None or []
            marker_count == 0
        """
        if self.mtmd_ctx is None:
            raise ValueError(
                f"{self.log_prefix}(_validate_mtmd_inputs): mtmd context not initialized."
            )

        if not isinstance(text, str):
            raise TypeError(
                f"{self.log_prefix}(_validate_mtmd_inputs): text must be str, "
                f"got {type(text).__name__}."
            )

        if not self.media_marker:
            raise ValueError(
                f"{self.log_prefix}(_validate_mtmd_inputs): media marker must not be empty."
            )

        if bitmaps is None:
            bitmaps = []

        marker_count = text.count(self.media_marker)
        bitmap_count = len(bitmaps)

        if marker_count != bitmap_count:
            raise ValueError(
                f"{self.log_prefix}(_validate_mtmd_inputs): media marker mismatch\n"
                f"- marker_count={marker_count}\n"
                f"- bitmap_count={bitmap_count}\n"
                f"- media_marker={self.media_marker!r}\n"
                "The rendered prompt must contain exactly one media marker per decoded media input."
            )

        for i, bitmap in enumerate(bitmaps):
            if bitmap is None:
                raise ValueError(
                    f"{self.log_prefix}(_validate_mtmd_inputs): bitmap[{i}] is None."
                )

    def _mtmd_tokenize(
        self,
        llama: "llama_core.Llama",
        text: str,
        bitmaps: Optional[List[Any]] = None,
        chunks: Optional[Any] = None,
    ) -> Any:
        """
        Perform MTMD hybrid tokenization.

        This function isolates the llama.cpp mtmd_tokenize call
        so that prompt construction logic is decoupled from runtime execution.

        It guarantees:
        - stable interface for future async/batch decoding
        - isolated error handling for tokenizer failures
        - clean separation between prompt building and C++ binding
        - strict Python-side marker/bitmap validation before native tokenization

        Pure text input is valid:
            bitmaps is None or []
            marker_count == 0
        """
        if bitmaps is None:
            bitmaps = []

        self._validate_mtmd_inputs(
            text=text,
            bitmaps=bitmaps,
        )

        if chunks is None:
            chunks = self._mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise ValueError(
                    f"{self.log_prefix}(_mtmd_tokenize): failed to init mtmd_input_chunks"
                )

        input_text = self._mtmd_cpp.mtmd_input_text()
        encoded_text = text.encode("utf-8")
        input_text.text = ctypes.c_char_p(encoded_text)
        input_text.text_len = len(encoded_text)
        input_text.add_special = (llama.n_tokens == 0)
        input_text.parse_special = True

        n_bitmaps = len(bitmaps)

        if n_bitmaps > 0:
            bitmap_array = (
                self._mtmd_cpp.mtmd_bitmap_p_ctypes * n_bitmaps
            )(*bitmaps)
        else:
            bitmap_array = None

        result = self._mtmd_cpp.mtmd_tokenize(
            self.mtmd_ctx,
            chunks,
            ctypes.byref(input_text),
            bitmap_array,
            n_bitmaps,
        )

        if result != 0:
            marker_count = text.count(self.media_marker)
            raise ValueError(
                f"{self.log_prefix}(_mtmd_tokenize): mtmd_tokenize failed\n"
                f"- result={result}\n"
                f"- text_len={len(text)}\n"
                f"- marker_count={marker_count}\n"
                f"- n_bitmaps={n_bitmaps}\n"
                f"- supports_vision={self.is_support_vision}\n"
                f"- supports_audio={self.is_support_audio}\n"
                f"- supports_video={self.is_support_video}\n"
                "Possible causes: marker/bitmap mismatch, invalid image/audio data, "
                "unsupported vision/audio projector, failed media preprocessing, "
                "or text tokenization failure."
            )

        return chunks

    def _process_mtmd_prompt(
        self,
        llama: llama_core.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[List[int], List[tuple], Any, List[Any]]:
        """
        Core multimodal preprocessing pipeline.
        Converts raw chat messages into C++ MTMD chunk structures and a virtual token ledger.

        Features:
        - Thread-safe concurrent media decoding to eliminate I/O bottlenecks.
        - "Negative Reverse Vocabulary" mapping for O(1) prefix matching of media tokens.
        - Strict RAII-style C++ memory management to prevent leaks on failure.

        Returns:
            full_prompt_ids: Ledger of text tokens and negative media IDs for prefix matching.
            chunk_token_spans: Tuples of (start_idx, end_idx, chunk_ptr, chunk_type, media_id).
            chunks: Allocated C++ mtmd_input_chunks pointer (must be freed by the caller).
            bitmap_cleanup: List of C++ bitmap pointers to be freed after evaluation.
        """
        # 1. Inject default system prompt if omitted by the user
        system_prompt = next((msg["content"] for msg in messages if msg.get("role") == "system"), "")
        if system_prompt == "" and self.DEFAULT_SYSTEM_MESSAGE is not None:
            messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_MESSAGE}] + messages

        media_items = self._get_media_items(messages)

        # 2. Render chat template and normalize media placeholders to MTMD markers.
        text = self._render_and_replace_media(
            messages=messages,
            media_items=media_items,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            add_generation_prompt=add_generation_prompt,
        )

        if self.verbose:
            print(
                f"{self.log_prefix}(_process_mtmd_prompt): "
                f"Rendered prompt length: {len(text)} chars, Media count: {len(media_items)}.\n"
                f"Rendered prompt: {text}",
                file=sys.stderr,
            )

        # 3. Pre-allocate bitmap array to guarantee chronological order during concurrent decoding
        bitmaps = [None] * len(media_items)
        bitmap_cleanup = []
        video_cleanup = []
        chunks = None

        try:
            # Concurrent Media Decoding
            import concurrent.futures
            if media_items:
                def _create_bitmap_func(idx: int, item: dict):
                    media_bytes = self.load_media(item["url"], item["type"])
                    bitmap, video_ctx = self._create_bitmap_from_bytes(media_bytes)
                    return idx, bitmap, video_ctx
                # This method uses multi-threaded parallel processing to convert images or audio to bitmaps,
                # which can be used in the future to process large numbers of video frames.
                max_workers = min(llama.n_threads, len(media_items))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_create_bitmap_func, i, item) for i, item in enumerate(media_items)]

                    for future in concurrent.futures.as_completed(futures):
                        idx, bitmap, video_ctx = future.result()

                        bitmaps[idx] = bitmap
                        bitmap_cleanup.append(bitmap)

                        if video_ctx:
                            video_cleanup.append(video_ctx)

                # Strict validation: Abort if any thread failed to decode its assigned media
                if any(b is None for b in bitmaps):
                    raise RuntimeError(f"{self.log_prefix}(_create_bitmap_func): Failed to decode one or more media files.")
                else:
                    if self.verbose:
                        print(f"{self.log_prefix}(_create_bitmap_func with {max_workers} threads): {len(media_items)} bitmaps were successfully created.")
            else:
                # If there are no images, set the bitmaps to empty.
                bitmaps = []

            # 4. Hybrid Tokenization (Text + Media)
            chunks = self._mtmd_tokenize(
                llama=llama,
                text=text,
                bitmaps=bitmaps,
                chunks=None,
            )

            # Video helper contexts only need to stay alive until mtmd_tokenize() completes.
            if video_cleanup:
                for video_ctx in video_cleanup:
                    self._mtmd_cpp.mtmd_helper_video_free(video_ctx)
                video_cleanup.clear()

            # 5. Virtual Token Ledger Construction
            full_prompt_ids = []
            chunk_token_spans = []
            current_idx = 0
            n_chunks = self._mtmd_cpp.mtmd_input_chunks_size(chunks)

            # Cursor to track the actual media contents (URLs or base64 data) provided by the user
            media_items_count = len(media_items)
            media_items_cur = 0
            last_media_id = None

            for i in range(n_chunks):
                chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                if chunk is None: continue
                chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)

                if self._is_text_chunk(chunk_type):
                    # Extract standard text token IDs
                    n_tokens_out = ctypes.c_size_t()
                    tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n_tokens_out))
                    if tokens_ptr and n_tokens_out.value > 0:
                        tokens = [tokens_ptr[j] for j in range(n_tokens_out.value)]
                        chunk_token_spans.append((current_idx, current_idx + len(tokens), chunk, chunk_type, None))
                        full_prompt_ids.extend(tokens)
                        current_idx += len(tokens)
                elif self._is_image_chunk(chunk_type) or self._is_audio_chunk(chunk_type):
                    # Extract media properties
                    # Note(JamePeng):
                    # The M-RoPE model is based on `n_pos` instead of `n_tokens` (of course, there's no difference in non-M-RoPE models).
                    # However, I still keep `n_tokens` because if `n_pos` is used, the underlying system will assume it is a full-match and will skip eval and sample.
                    # chunk_n_pos = self._mtmd_cpp.mtmd_input_chunk_get_n_pos(chunk) # equals to max(t,h,w) for M-RoPE; equals to `n_tokens` otherwise
                    chunk_n_tokens = self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)

                    if media_items_cur < media_items_count:
                        # The C++ parser only sees identical placeholders (e.g., "<__media__>").
                        # We MUST inject the actual media content's identity here.
                        real_media_url = media_items[media_items_cur]["url"]
                        # Vocabulary Positive forward: 0 to 248,319 (Qwen3.5)
                        # Generate a deterministic, unique negative ID for this specific image/audio.
                        # - zlib.crc32 ensures cross-platform and cross-run consistency (unlike Python's hash()).
                        # - We map it to a negative space (-100 to -16,777,316) to avoid colliding with
                        #   positive text token IDs (e.g., Qwen3.5 vocab goes up to ~152k).
                        # This empowers `longest_token_prefix` to correctly identify and reuse cached images,
                        # while instantly breaking the match if the image content changes.
                        # media_id = - (zlib.crc32(real_media_url.encode('utf-8')) % (2**24)) - 100
                        media_id = - (zlib.crc32(real_media_url.encode('utf-8')) & 0xFFFFFF) - 100
                        last_media_id = media_id
                        media_items_cur += 1
                    elif last_media_id is not None:
                        # video may expand into multiple image chunks from one media marker
                        media_id = last_media_id
                    else:
                        # Magic Negative Number as fallback :)
                        media_id = -314159

                    if self.verbose:
                        print(f"{self.log_prefix}(mtmd_input_chunk_media_id): chunk_n_tokens: {chunk_n_tokens}, media_id: {media_id}, ")

                    chunk_token_spans.append((current_idx, current_idx + chunk_n_tokens, chunk, chunk_type, media_id))

                    # Pad the ledger with the pseudo-ID to mimic the physical space taken in the KV cache
                    full_prompt_ids.extend([media_id] * chunk_n_tokens)
                    current_idx += chunk_n_tokens
                else:
                    raise TypeError(f"{self.log_prefix}(mtmd_input_chunk_get_type): Invalid chunk type, chunk_type = {chunk_type}.")

            if media_items_cur != media_items_count:
                raise RuntimeError(
                    f"{self.log_prefix}(_process_mtmd_prompt): not all media inputs were consumed by MTMD chunks\n"
                    f"- consumed={media_items_cur}\n"
                    f"- media_items={media_items_count}\n"
                    "This usually means the rendered prompt did not produce enough media chunks, "
                    "or the chat template/media marker normalization is incorrect."
                )

            return full_prompt_ids, chunk_token_spans, chunks, bitmap_cleanup

        except Exception as e:
            # Ensure no useless pointers remain upon any failure
            # Free chunks
            if chunks is not None:
                self._mtmd_cpp.mtmd_input_chunks_free(chunks)
                chunks = None
            # Free bitmaps
            if len(bitmap_cleanup) > 0:
                for bitmap in bitmap_cleanup:
                    self._mtmd_cpp.mtmd_bitmap_free(bitmap)
                bitmap_cleanup = None
            # Free videos
            if len(video_cleanup) > 0:
                for video_ctx in video_cleanup:
                    self._mtmd_cpp.mtmd_helper_video_free(video_ctx)
                video_cleanup = None

            bitmaps = None

            raise e

    def __call__(
        self,
        *,
        llama: llama_core.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        present_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_n_sigma: float = -1.00,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        xtc_threshold: float = 0.1,
        xtc_probability: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n:int = 0,
        dry_seq_breakers: list[str] = ["\n", ":", "\"", "*"],
        adaptive_target : float = -1.0,
        adaptive_decay : float = 0.9,
        use_infill: bool = False,
        model: Optional[str] = None,
        logits_processor: Optional[llama_core.LogitsProcessorList] = None,
        grammar: Optional[llama_grammar.LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        add_generation_prompt: bool = True,
        reasoning_budget: int = -1,
        reasoning_start: str = "<think>",
        reasoning_end: str = "</think>",
        reasoning_budget_message: Optional[str] = None,
        reasoning_start_in_prompt: bool = False,
        reasoning_start_max_tokens: Optional[int] = 32,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        # 1. Initialize mtmd context
        self._init_mtmd_context(llama)
        assert self.mtmd_ctx is not None

        # 2. Concurrent Preprocessing & Ledger Construction
        full_prompt_ids, chunk_token_spans, chunks, bitmap_cleanup = self._process_mtmd_prompt(
            llama=llama,
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            add_generation_prompt=add_generation_prompt,
        )

        if self.verbose:
            print(f"{self.log_prefix}(__call__): Prepared virtual token ledger of length {len(full_prompt_ids)}.", file=sys.stderr)

        try:
            # 3. KV Cache Synchronization & State Rollback
            # Compares the virtual ledger with physical history to prevent Cache Poisoning.
            current_history = llama.input_ids[:llama.n_tokens].tolist()
            longest_prefix = llama.longest_token_prefix(current_history, full_prompt_ids, self.verbose)

            if longest_prefix < llama.n_tokens:
                if llama.is_hybrid and llama._hybrid_cache_mgr is not None:
                    if llama._hybrid_cache_mgr.max_checkpoints > 0:
                        if self.verbose:
                            print(f"{self.log_prefix}(__call__): Hybrid prefix mismatch (matched {longest_prefix}/{llama.n_tokens}). "
                                f"Searching for nearest checkpoint...", file=sys.stderr)

                        best_ckpt = llama._hybrid_cache_mgr.find_best_checkpoint(full_prompt_ids, seq_id=0)
                        if best_ckpt and llama._hybrid_cache_mgr.restore_checkpoint(best_ckpt, seq_id=0):
                            llama.n_tokens = best_ckpt.pos
                            if self.verbose:
                                print(f"{self.log_prefix}(__call__): Successfully rolled back to checkpoint at pos {llama.n_tokens}.", file=sys.stderr)
                        else:
                            if self.verbose:
                                print(f"{self.log_prefix}(__call__): No suitable checkpoint found or restore failed. Clearing hybrid cache entirely.", file=sys.stderr)
                            llama._hybrid_cache_mgr.clear()
                            llama._ctx.memory_clear(True)
                            llama.n_tokens = 0
                    else:
                        if self.verbose:
                            print(f"{self.log_prefix}(__call__): Hybrid cache enabled but max_checkpoints is 0. Clearing cache entirely.", file=sys.stderr)
                        llama._hybrid_cache_mgr.clear()
                        llama._ctx.memory_clear(True)
                        llama.n_tokens = 0
                else:
                    if self.verbose:
                        print(f"{self.log_prefix}(__call__): Prefix mismatch. Truncating KV cache from {llama.n_tokens} to {longest_prefix}.", file=sys.stderr)
                    llama._ctx.memory_seq_rm(0, longest_prefix, -1)
                    llama.n_tokens = longest_prefix

            n_past = llama.n_tokens

            for start_idx, end_idx, chunk_ptr, chunk_type, media_id in chunk_token_spans:
                # Skip previously matched chunks
                if end_idx <= n_past:
                    continue

                if self._is_text_chunk(chunk_type):
                    unprocessed_start = max(start_idx, n_past) - start_idx
                    n_tokens_out = ctypes.c_size_t()
                    tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk_ptr, ctypes.byref(n_tokens_out))

                    if tokens_ptr and n_tokens_out.value > 0:
                        all_tokens = [tokens_ptr[j] for j in range(n_tokens_out.value)]
                        tokens_to_eval = all_tokens[unprocessed_start:]

                        if tokens_to_eval:
                            if self.verbose:
                                print(
                                    f"{self.log_prefix}(__call__): Evaluating TEXT chunk "
                                    f"({len(tokens_to_eval)} tokens) at pos {llama.n_tokens}...",
                                    file=sys.stderr,
                                )

                            # Text evaluation delegates shift and chunking to native llama.eval
                            llama.eval(tokens_to_eval)
                            n_past = llama.n_tokens

                elif self._is_image_chunk(chunk_type) or self._is_audio_chunk(chunk_type):
                    chunk_n_tokens = self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk_ptr)

                    if self.verbose:
                        media_str = "IMAGE" if self._is_image_chunk(chunk_type) else "AUDIO"
                        print(f"{self.log_prefix}(__call__): Evaluating {media_str} chunk ({chunk_n_tokens} tokens) at pos {llama.n_tokens}...", file=sys.stderr)

                    # Stage 5: Multimodal Physical OOM Defense
                    if n_past + chunk_n_tokens > llama.n_ctx():
                        if not llama._ctx.memory_can_shift():
                            raise RuntimeError(
                                f"{self.log_prefix}(__call__): Context Shift is explicitly disabled by the C++ backend "
                                f"(n_pos_per_embd > 1 or incompatible M-RoPE). "
                                f"Multimodal chunk exceeded context limit(currently n_ctx={llama._n_ctx}), "
                                f"You MUST increase n_ctx to fit the dialogue."
                            )
                        else:
                            # Safely discard oldest tokens while preserving system prompts
                            n_discard = (n_past + chunk_n_tokens) - llama.n_ctx() + llama.n_batch
                            n_keep = min(llama.n_keep, n_past)
                            n_discard = min(n_discard, n_past - n_keep)

                            if n_discard <= 0:
                                raise RuntimeError(f"{self.log_prefix}(__call__): Critical Overflow. Not enough unpinned tokens to discard for Context Shift.")

                            if self.verbose:
                                print(f"{self.log_prefix}(__call__): OOM risk detected. Shifting multimodal context: keeping {n_keep}, discarding {n_discard}...", file=sys.stderr)

                            # Execute physical memory shift
                            llama._ctx.memory_seq_rm(0, n_keep, n_keep + n_discard)
                            llama._ctx.memory_seq_add(0, n_keep + n_discard, n_past, -n_discard)

                            # Shift python virtual array to match
                            remaining_len = n_past - (n_keep + n_discard)
                            if remaining_len > 0:
                                llama.input_ids[n_keep : n_keep + remaining_len] = llama.input_ids[n_keep + n_discard : n_past]

                            n_past -= n_discard
                            llama.n_tokens = n_past

                    # Execute C++ Multimodal Black-box Extraction
                    new_n_past = llama_cpp_lib.llama_pos(0)
                    result = self._mtmd_cpp.mtmd_helper_eval_chunk_single(
                        self.mtmd_ctx,
                        llama._ctx.ctx,
                        chunk_ptr,
                        llama_cpp_lib.llama_pos(n_past),
                        llama_cpp_lib.llama_seq_id(0),
                        llama.n_batch,
                        True, # logits_last = True, drastically saves computational overhead
                        ctypes.byref(new_n_past)
                    )

                    if result != 0:
                        raise ValueError(f"{self.log_prefix}(mtmd_helper_eval_chunk_single): Media evaluation failed with error code {result}.")

                    # Update Ledger with "Negative Reverse Vocabulary" IDs
                    llama.input_ids[n_past : new_n_past.value] = media_id
                    n_past = new_n_past.value
                    llama.n_tokens = n_past

            # Extract the final, perfectly synchronized prompt sequence
            prompt = llama.input_ids[: llama.n_tokens].tolist()

            # End-of-Turn Checkpoint
            # Anchors the state ONLY after the entire multi-modal turn is processed
            if (
                llama.is_hybrid
                and llama._hybrid_cache_mgr is not None
                and llama._hybrid_cache_mgr.max_checkpoints > 0
            ):
                if self.verbose:
                    print(f"{self.log_prefix}(__call__): [End-of-Turn Checkpoint] Anchoring full prompt state at pos {llama.n_tokens}.", file=sys.stderr)

                llama._hybrid_cache_mgr.save_checkpoint(
                    current_pos=llama.n_tokens,
                    tokens=prompt,
                    seq_id=0
                )
        finally:
            # Cleanup chunks
            if chunks is not None:
                self._mtmd_cpp.mtmd_input_chunks_free(chunks)
                chunks = None
            # Cleanup bitmaps
            if bitmap_cleanup:
                for bitmap in bitmap_cleanup:
                    self._mtmd_cpp.mtmd_bitmap_free(bitmap)
                bitmap_cleanup.clear()
            bitmap_array = None

        # Handle response format and tools (same as before)
        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        # Convert legacy functions to tools
        if functions is not None:
            tools = [
                {
                    "type": "function",
                    "function": function,
                }
                for function in functions
            ]

        # Convert legacy function_call to tool_choice
        if function_call is not None:
            if isinstance(function_call, str) and (
                function_call == "none" or function_call == "auto"
            ):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                    },
                }

        tool = None
        if (
            tool_choice is not None
            and isinstance(tool_choice, dict)
            and tools is not None
        ):
            name = tool_choice["function"]["name"]
            tool = next((t for t in tools if t["function"]["name"] == name), None)
            if tool is None:
                raise ValueError(f"Tool choice '{name}' not found in tools.")
            schema = tool["function"]["parameters"]
            try:
                # create grammar from json schema
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=llama.verbose
                )
            except Exception as e:
                if llama.verbose:
                    print(str(e), file=sys.stderr)
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )

        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=top_logprobs if logprobs else None,
            stream=stream,
            stop=stop,
            seed=seed,
            max_tokens=max_tokens,
            present_penalty=present_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            top_n_sigma=top_n_sigma,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            xtc_threshold=xtc_threshold,
            xtc_probability=xtc_probability,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            dry_seq_breakers=dry_seq_breakers,
            adaptive_target=adaptive_target,
            adaptive_decay=adaptive_decay,
            use_infill=use_infill,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
            reasoning_budget=reasoning_budget,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
            reasoning_budget_message=reasoning_budget_message,
            reasoning_start_in_prompt=reasoning_start_in_prompt,
            reasoning_start_max_tokens=reasoning_start_max_tokens,
        )

        if tool is not None:
            tool_name = tool["function"]["name"]
            return _convert_completion_to_chat_function(
                tool_name, completion_or_chunks, stream
            )
        return _convert_completion_to_chat(completion_or_chunks, stream=stream)

    def load_media(self, media_url: str, media_type: str) -> bytes:
        """
        Unified dispatcher for loading media payloads.
        Routes the URL/URI to the specific image, audio, or video processor based on the media_type.
        """
        if media_type == "image":
            return self._load_image(media_url)

        elif media_type == "audio":
            audio_bytes = self._load_bytes(media_url, timeout=15, kind="audio")
            try:
                self.detect_audio_format(audio_bytes)
            except ValueError as e:
                raise ValueError(f"{self.log_prefix}(load_media): {e}")
            return audio_bytes

        elif media_type == "video":
            return self._load_bytes(media_url, timeout=30, kind="video")

        else:
            raise ValueError(f"{self.log_prefix}(load_media): Unknown media type '{media_type}'")

    @staticmethod
    def detect_audio_format(audio_bytes: bytes) -> str:
        """
        Pure utility function: Detects the audio format from magic bytes.
        Strictly translated from llama.cpp's `is_audio_file` to ensure 100% compatibility
        and avoid false positives (e.g., AVI files disguised as RIFF).
        """
        length = len(audio_bytes)

        if length < 12:
            raise ValueError("Audio data is corrupted or too small (less than 12 bytes).")

        # RIFF & WAVE magic bytes verification
        is_wav = audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE"

        # ID3 metadata or MPEG sync word verification
        is_mp3 = length >= 3 and (
            audio_bytes.startswith(b"ID3") or
            (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0)
        )

        # FLAC magic bytes verification
        is_flac = audio_bytes.startswith(b"fLaC")

        if is_wav:
            return "wav"
        elif is_mp3:
            return "mp3"
        elif is_flac:
            return "flac"
        else:
            raise ValueError(
                "Unsupported audio format detected via magic bytes. "
                "The underlying C++ miniaudio backend ONLY supports WAV, MP3, and FLAC."
            )

    DEFAULT_HTTP_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/148.0.0.0 Safari/537.36"
        ),
    }

    @staticmethod
    def _load_bytes(media_url: str, timeout: int = 15, kind: str = "media") -> bytes:
        """
        Load raw bytes from a data URI, local file path, or remote HTTP/HTTPS URL.
        """
        media_bytes = b""

        # 1. Handle data URI
        if media_url.strip().startswith("data:"):
            comma_pos = media_url.find(",")
            if comma_pos == -1:
                raise ValueError("Invalid data URI: missing comma separator")

            base64_data = media_url[comma_pos + 1:]
            media_bytes = base64.b64decode(base64_data)

        # 2. Handle local file path
        elif os.path.exists(media_url):
            with open(media_url, "rb") as f:
                media_bytes = f.read()

        # 3. Handle remote URL via HTTP/HTTPS
        else:
            req = urllib.request.Request(
                media_url,
                headers=MTMDChatHandler.DEFAULT_HTTP_HEADERS,
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as f:
                    media_bytes = f.read()
            except (URLError, HTTPError) as e:
                raise ConnectionError(f"Failed to download {kind} from {media_url}: {e}")

        if not media_bytes:
            raise ValueError(f"Empty {kind} data received")

        return media_bytes

    @staticmethod
    def _load_image(image_url: str) -> bytes:
        """
        Load an image from either a URL or a data URI and return it as JPEG bytes.

        Supports:
        - Remote images via HTTP/HTTPS (with proper User-Agent)
        - Data URIs (base64-encoded, e.g., data:image/png;base64,...)
        - Images with alpha channel (PNG, WebP, etc.) → automatically composites on white/black background
        - Any format that Pillow can open. See: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

        Returns:
            JPEG-encoded bytes (quality=95) in RGB mode, suitable for most vision models.
        """
        # 1. Load image bytes from image_url
        image_bytes = MTMDChatHandler._load_bytes(
            image_url,
            timeout=15,
            kind="image",
        )

        # 2. Check if image_bytes is empty.
        if not image_bytes:
            raise ValueError("Empty image data received")

        # 3. Open image with Pillow
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

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str],
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        **kwargs: Any,
    ) -> "MTMDChatHandler":
        import fnmatch
        from pathlib import Path

        try:
            from huggingface_hub import hf_hub_download, HfFileSystem  # type: ignore
            from huggingface_hub.utils import validate_repo_id  # type: ignore
        except ImportError:
            raise ImportError(
                "Llama.from_pretrained requires the huggingface_hub package. "
                "You can install it with `pip install --upgrade huggingface_hub`."
            )

        validate_repo_id(repo_id)

        hffs = HfFileSystem()

        files = [
            file["name"] if isinstance(file, dict) else file
            for file in hffs.ls(repo_id)  # type: ignore
        ]

        # split each file into repo_id, subfolder, filename
        file_list: List[str] = []
        for file in files:
            rel_path = Path(file).relative_to(repo_id)
            file_list.append(str(rel_path))

        matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

        if len(matching_files) == 0:
            raise ValueError(
                f"No file found in {repo_id} that match {filename}\n\n"
                f"Available Files:\n{json.dumps(file_list)}"
            )

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found in {repo_id} matching {filename}\n\n"
                f"Available Files:\n{json.dumps(files)}"
            )

        (matching_file,) = matching_files

        subfolder = str(Path(matching_file).parent)
        filename = Path(matching_file).name

        # download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=cast(Union[str, Path, None], local_dir),
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cast(Union[str, Path, None], cache_dir),
        )

        if local_dir is None:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=cast(Union[str, Path, None], cache_dir),
                local_files_only=True,
            )
        else:
            model_path = os.path.join(local_dir, filename)

        return cls(
            mmproj_path=model_path,
            **kwargs,
        )

# Generic template-driven MTMD handler.
class GenericMTMDChatHandler(MTMDChatHandler):
    """
    Generic MTMD chat handler backed by the model-provided chat template.

    This handler is intentionally template-driven. It renders the model's
    tokenizer.chat_template first, then normalizes rendered media URLs or
    placeholder tokens into MTMD media markers before tokenization.

    It is designed for model templates that emit media placeholders such as
    <|image_pad|>, <|image|>, <image>, [IMG], or Kimi-style <|media_pad|>.
    Model-specific handlers may still be preferable when a model requires
    special stop tokens, generation flags, or non-standard template arguments.
    """

    KNOWN_MEDIA_TAGS = [
        # Pad placeholders inside model-specific wrappers.
        "<|image_pad|>",
        "<|audio_pad|>",
        "<|video_pad|>",

        # Direct placeholders inside Gemma/Llama/GLM-style wrappers.
        "<|image|>",
        "<|audio|>",
        "<|video|>",

        # LLaVA / LFM / Mistral-style placeholders.
        "<image>",
        "<audio>",
        "<video>",
        "[IMG]",

        # Kimi-style placeholders.
        "<|media_pad|>",
        "<|kimi_k25_video_placeholder|>",
    ]

    def __init__(
        self,
        chat_format: Optional[str],
        mmproj_path: str,
        verbose: bool = True,
        chat_template_name: Optional[str] = None,
        **kwargs
    ) -> None:
        self.chat_format = chat_format
        self.chat_template_name = chat_template_name
        self.verbose = verbose

        if self.verbose and self.chat_format is not None:
            print(
                f"{self.__class__.__name__}.__init__: using provided chat template:\n"
                f"```jinja\n{self.chat_format}\n```",
                file=sys.stderr,
            )

        super().__init__(mmproj_path = mmproj_path, verbose = verbose, **kwargs)

    def _resolve_chat_format(self, llama: llama_core.Llama) -> str:
        # Highest priority: use the template explicitly provided by the caller.
        if self.chat_format is not None:
            return self.chat_format

        chat_format = None

        # The Llama instance is only available at call time, so query llama.cpp here
        # for either the requested named template or the model's default template.
        try:
            name = (
                self.chat_template_name.encode("utf-8")
                if self.chat_template_name is not None
                else None
            )
            chat_format = llama._model.model_chat_template(name)
        except Exception as exc:
            if self.verbose:
                print(
                    f"{self.log_prefix}: failed to load chat template"
                    f"{f' {self.chat_template_name!r}' if self.chat_template_name else ''} "
                    f"from llama model: {exc}",
                    file=sys.stderr,
                )

        # If a named template is unavailable, try the default model template.
        if chat_format is None and self.chat_template_name is not None:
            try:
                chat_format = llama._model.model_chat_template(None)
                if self.verbose and chat_format is not None:
                    print(
                        f"{self.log_prefix}: chat template {self.chat_template_name!r} "
                        "not found; using default model chat template.",
                        file=sys.stderr,
                    )
            except Exception as exc:
                if self.verbose:
                    print(
                        f"{self.log_prefix}: failed to load default model chat template: {exc}",
                        file=sys.stderr,
                    )

        # Last resort: use the generic built-in MTMD template.
        if chat_format is None:
            chat_format = self.CHAT_FORMAT
            if self.verbose:
                print(
                    f"{self.log_prefix}: no model chat template found; "
                    "using MTMDChatHandler built-in CHAT_FORMAT.",
                    file=sys.stderr,
                )

        self.chat_format = chat_format
        return chat_format

    def _ensure_chat_template(
        self,
        llama: llama_core.Llama,
    ) -> None:
        """
        Resolve and analyze chat template once.

        Chat template metadata is static for a model instance,
        so it should not be recomputed for every request.
        """
        if self._template_initialized:
            return

        self._resolve_chat_format(llama)

        if self.chat_format is None:
            raise ValueError(
                f"{self.log_prefix}: failed to resolve a chat template. "
                "`chat_format` must be a Jinja chat template string. You may pass it "
                "directly, read it from a chat_template.jinja file, set a valid "
                "`chat_template_name` for a named template stored in the model, or use "
                "a model that provides tokenizer.chat_template metadata."
            )

        self._chat_format_parser_tags = [
            tag
            for tag in self.KNOWN_MEDIA_TAGS
            if tag in self.chat_format
        ]

        self._template_initialized = True

    def __call__(self, **kwargs):
        llama = kwargs["llama"]

        self._ensure_chat_template(llama)

        if self.verbose:
            print(f"{self.log_prefix} - Start processing", file=sys.stderr)

        # Use parent implementation
        return super().__call__(**kwargs)

class Llava15ChatHandler(MTMDChatHandler):
    CHAT_FORMAT = (
        "{% for message in messages %}"
            "{% if message.role == 'system' %}"
                "{{ message.content }}"
            "{% endif %}"

            "{% if message.role == 'user' %}"
                "{% if message.content is string %}"
                    "\nUSER: {{ message.content }}"
                "{% elif message.content is iterable %}"
                    "\nUSER: "
                    "{% for content in message.content %}"
                        "{% if content.type == 'image_url' %}"
                            "{{ content.image_url if content.image_url is string else content.image_url.url }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "{% for content in message.content %}"
                        "{% if content.type == 'text' %}"
                            "{{ content.text }}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endif %}"

            "{% if message.role == 'assistant' and message.content is not none %}"
                "\nASSISTANT: {{ message.content }}"
            "{% endif %}"
        "{% endfor %}"

        "{% if add_generation_prompt %}"
            "\nASSISTANT: "
        "{% endif %}"
    )


class ObsidianChatHandler(MTMDChatHandler):
    # Prompt Format
    # The model followed ChatML format. However, with ### as the seperator

    # <|im_start|>user
    # What is this sign about?\n<image>
    # ###
    # <|im_start|>assistant
    # The sign is about bullying, and it is placed on a black background with a red background.
    # ###

    CHAT_FORMAT = (
        "{% for message in messages %}"
        # System message
        "{% if message.role == 'system' %}"
        "<|im_start|>system\n"
        "{{ message.content }}\n"
        "###\n"
        "{% endif %}"
        # User message
        "{% if message.role == 'user' %}"
        "<|im_start|>user\n"
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% if message.content is iterable %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "###\n"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{{ message.content }}"
        "###\n"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


class MoondreamChatHandler(MTMDChatHandler):
    # Chat Format:
    # f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "{% if message.content is iterable %}"
        # <image>
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}\n\n"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{{ content.image_url.url }}\n\n"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        # Question:
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "Question: {{ content.text }}\n\n"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        # Question:
        "{% if message.content is string %}"
        "Question: {{ message.content }}\n\n"
        "{% endif %}"
        "{% endif %}"
        # Answer:
        "{% if message.role == 'assistant' %}"
        "Answer:{{ message.content }}\n\n"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "Answer:"
        "{% endif %}"
    )


class Llava16ChatHandler(MTMDChatHandler):
    # Example prompt
    # "DEFAULT_SYSTEM_MESSAGE + USER: <image>\nWhat is shown in this image? ASSISTANT:"

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% if message.role == 'user' %}"
        "{% if message.content is iterable %}"
        # <image>
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}\n"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{{ content.image_url.url }}\n"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        # Question:
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        # Question:
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% endif %}"
        # Answer:
        "{% if message.role == 'assistant' %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "Answer:"
        "{% endif %}"
    )


class NanoLlavaChatHandler(MTMDChatHandler):
    # Prompt Format
    # The model follow the ChatML standard, however, without \n at the end of <|im_end|>:

    # <|im_start|>system
    # Answer the question<|im_end|><|im_start|>user
    # <image>
    # What is the picture about?<|im_end|><|im_start|>assistant
    DEFAULT_SYSTEM_MESSAGE = "Answer the question"

    CHAT_FORMAT = (
        "{% for message in messages %}"
        # System message
        "{% if message.role == 'system' %}"
        "<|im_start|>system\n"
        "{{ message.content }}"
        "<|im_end|>"
        "{% endif %}"
        # User message
        "{% if message.role == 'user' %}"
        "<|im_start|>user\n"
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% if message.content is iterable %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "<|im_end|>"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{{ message.content }}"
        "<|im_end|>"
        "{% endif %}"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


class Llama3VisionAlphaChatHandler(MTMDChatHandler):
    # question = "<image>" + q

    # prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "<|start_header_id|>"
        "{% if message.role == 'user' %}"
        "user<|end_header_id|>\n\n"
        "{% if message.content is iterable %}"
        # <image>
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        # Question:
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        # Question:
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% endif %}"
        # Answer:
        "{% if message.role == 'assistant' %}"
        "assistant<|end_header_id|>\n\n"
        "{{ message.content }}"
        "{% endif %}"
        "<|eot_id|>"
        "{% endfor %}"
        # Generation prompt
        "{% if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% endif %}"
    )


# alias
Llama3VisionAlpha = Llama3VisionAlphaChatHandler


class MiniCPMv26ChatHandler(MTMDChatHandler):

    CHAT_FORMAT = (
        "{% set image_count = namespace(value=0) %}"
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "{% endif %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is iterable %}"
        "{% for content in message['content'] %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{% set image_count.value = image_count.value + 1 %}"
        "<image_id>{{ image_count.value }}</image_id>: <image>{{ content.image_url }}</image>"
        "{% endif %}"
        "{% if content.image_url is mapping %}"
        "{% set image_count.value = image_count.value + 1 %}"
        "<image_id>{{ image_count.value }}</image_id>: <image>{{ content.image_url.url }}</image>"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"

        "{% for content in message['content'] %}"
        "{% if content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


class MiniCPMv45ChatHandler(MTMDChatHandler):
    """
    Handler for MiniCPM-V 4.5 models.

    Supports:
    - Multi-step tool calls with <tool_call> and <tool_response> XML tags.
    - Integrated reasoning (thinking) process with <think> tags.
    - Specialized system prompt handling with tool definitions.
    - Global image numbering for multi-image processing.
    """

    # Model specific control tokens
    MINICPMV_BOS_TOKEN = "<|im_start|>"
    MINICPMV_EOS_TOKEN = "<|im_end|>"
    MINICPMV_PAD_TOKEN = "<|endoftext|>"

    # Image placeholder tags
    MINICPMV_IMAGE_START_TOKEN = "<image>"
    MINICPMV_IMAGE_END_TOKEN = "</image>"
    MINICPMV_IMAGE_ID_START_TOKEN = "<image_id>"
    MINICPMV_IMAGE_ID_END_TOKEN = "</image_id>"

    CHAT_FORMAT = (
        # --- 1. First System Message & Tools Definitions ---
        "{%- if tools %}"
            "{{- '" + MINICPMV_BOS_TOKEN + "system\\n' }}"
            "{%- if messages[0].role == 'system' %}{{- messages[0].content + '\\n\\n' }}{%- endif %}"
            "{{- '# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\n' }}"
            "{{- 'You are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}"
            "{%- for tool in tools %}{{- '\\n' + (tool | tojson) }}{%- endfor %}"
            "{{- '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call>" + MINICPMV_EOS_TOKEN + "\\n' }}"
        "{%- elif messages[0].role == 'system' %}"
            "{{- '" + MINICPMV_BOS_TOKEN + "system\\n' + messages[0].content + '" + MINICPMV_EOS_TOKEN + "\\n' }}"
        "{%- endif %}"

        # --- 2. Message Stream Processing ---
        "{% set image_count = namespace(value=0) %}"
        "{%- for message in messages %}"
            # --- Unified Role Handling (User, Assistant, and subsequent Systems) ---
            "{%- if message.role in ['user', 'assistant'] or (message.role == 'system' and not loop.first) %}"
                "{{- '" + MINICPMV_BOS_TOKEN + "' + message.role + '\\n' }}"

                "{%- set content = message.content %}"
                "{%- if content is not string %}"
                    "{%- set ns = namespace(content_str='') %}"
                    "{%- for item in content %}"
                        # --- Explicit image_url type and value checking ---
                        "{%- if item.type == 'image_url' %}"
                            "{%- set image_url = item.image_url if item.image_url is string else item.image_url.url %}"
                            "{%- set image_count.value = image_count.value + 1 %}"
                            # Format: <image_id>N</image_id>: <image>IMAGE_URL</image>
                            "{%- set ns.content_str = ns.content_str + '<image_id>' + (image_count.value | string) + '</image_id>: <image>' + image_url + '</image>' %}"
                        "{%- elif item.type == 'text' %}"
                            "{%- set ns.content_str = ns.content_str + item.text %}"
                        "{%- endif %}"
                    "{%- endfor %}"
                    "{%- set content = ns.content_str %}"
                "{%- endif %}"

                "{{- content -}}"

                # Append tool_calls to assistant messages if they exist
                "{%- if message.role == 'assistant' and message.tool_calls %}"
                    "{%- for tool_call in message.tool_calls %}"
                        "{%- set tc = tool_call.function if tool_call.function else tool_call %}"
                        "{{- '\\n<tool_call>\\n{\"name\": \"' + tc.name + '\", \"arguments\": ' }}"
                        "{{- tc.arguments if tc.arguments is string else tc.arguments | tojson }}"
                        "{{- '}\\n</tool_call>' }}"
                    "{%- endfor %}"
                "{%- endif %}"
                "{{- '" + MINICPMV_EOS_TOKEN + "\\n' }}"

            # --- Specialized Tool Response Handling ---
            # Group consecutive tool responses under a single user-like block
            "{%- elif message.role == 'tool' %}"
                "{%- if loop.first or (messages[loop.index0 - 1].role != 'tool') %}"
                    "{{- '" + MINICPMV_BOS_TOKEN + "user' }}"
                "{%- endif %}"
                "{{- '\\n<tool_response>\\n' + message.content + '\\n</tool_response>' }}"
                "{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}"
                    "{{- '" + MINICPMV_EOS_TOKEN + "\\n' }}"
                "{%- endif %}"
            "{%- endif %}"
        "{%- endfor %}"

        # --- 3. Generation Prompt ---
        "{%- if add_generation_prompt %}"
            "{{- '" + MINICPMV_BOS_TOKEN + "assistant\\n' }}"
            # Handle thinking/reasoning block visibility based on configuration
            "{%- if enable_thinking is defined and enable_thinking is false %}"
                "{{- '<think>\\n\\n</think>\\n\\n' }}"
            "{%- elif enable_thinking is defined and enable_thinking is true %}"
                "{{- '<think>\\n' }}"
            "{%- endif %}"
        "{%- endif %}"
    )

    def __init__(self, enable_thinking: bool = True, **kwargs):
        """
        Initializes the MiniCPM-V 4.5 Handler.

        Args:
            enable_thinking (bool): If True, model generates reasoning before the final answer.
            **kwargs: Additional arguments for the base MTMDChatHandler.
        """
        self.enable_thinking = enable_thinking
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        # Inject thinking control flag into the template
        self.extra_template_arguments["enable_thinking"] = self.enable_thinking

        # Set stop token patch
        kwargs['stop'] = [self.MINICPMV_EOS_TOKEN, self.MINICPMV_PAD_TOKEN]

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix}(enable_thinking={self.enable_thinking}) - Start processing")
        return super().__call__(**kwargs)


class MiniCPMV46ChatHandler(MTMDChatHandler):
    """
    Handler for MiniCPM-V-4.6 models.

    Features:
    - Aligned with official tokenizer_config.json special tokens.
    - Custom `<|image_pad|>` and `<|video_pad|>` multimodal tokens.
    - Integrated MTMD-style URL and Base64 injection for visual content.
    - Specialized `<tool_call>` and `<tool_response>` block generation.
    - Autonomously folds previous reasoning paths using `last_query_index`.
    - Toggles `<think>` block generation via `enable_thinking` (Defaults to False).
    """

    # Core tokens
    MINICPM_BOS_TOKEN = "<|im_start|>"
    MINICPM_EOS_TOKEN = "<|im_end|>"
    MINICPM_PAD_TOKEN = "<|endoftext|>"

    # Vision tokens
    MINICPM_VISION_BOS_TOKEN = "<|vision_start|>"
    MINICPM_VISION_EOS_TOKEN = "<|vision_end|>"
    MINICPM_IMAGE_TOKEN = "<|image_pad|>"
    MINICPM_VIDEO_TOKEN = "<|video_pad|>"

    CHAT_FORMAT = (
        "{%- if enable_thinking is not defined -%}\n"
        "    {%- set enable_thinking = false -%}\n"
        "{%- endif -%}\n"
        "{%- macro render_content(content, is_system_content=false) -%}\n"
        "    {%- if content is string -%}\n"
        "        {{- content -}}\n"
        "    {%- elif content is iterable and content is not mapping -%}\n"
        "        {%- set ns = namespace(parts=[]) -%}\n"
        "        {%- for item in content -%}\n"
        "            {%- if 'image' in item or 'image_url' in item or item.type == 'image' -%}\n"
        "                {%- if is_system_content -%}\n"
        "                    {{- raise_exception('System message cannot contain images.') -}}\n"
        "                {%- endif -%}\n"
        "                {%- set url_val = '' -%}\n"
        "                {%- if item.type == 'image_url' -%}\n"
        "                    {%- set url_val = item.image_url if item.image_url is string else item.image_url.url -%}\n"
        "                {%- endif -%}\n"
        "                {%- set ns.parts = ns.parts + ['<|image_pad|>' + url_val] -%}\n"
        # "            {%- elif 'video' in item or 'video_url' in item or item.type == 'video' -%}\n"
        # "                {%- if is_system_content -%}\n"
        # "                    {{- raise_exception('System message cannot contain videos.') -}}\n"
        # "                {%- endif -%}\n"
        # "                {%- set url_val = '' -%}\n"
        # "                {%- if item.type == 'video_url' -%}\n"
        # "                    {%- set url_val = item.video_url if item.video_url is string else item.video_url.url -%}\n"
        # "                {%- endif -%}\n"
        # "                {%- set ns.parts = ns.parts + ['<|video_pad|>' + url_val] -%}\n"
        "            {%- elif 'text' in item -%}\n"
        "                {%- set ns.parts = ns.parts + [item.text] -%}\n"
        "            {%- else -%}\n"
        "                {{- raise_exception('Unexpected item type in content.') -}}\n"
        "            {%- endif -%}\n"
        "        {%- endfor -%}\n"
        "        {{- ns.parts | join('\\n') -}}\n"
        "    {%- elif content is none or content is undefined -%}\n"
        "        {{- '' -}}\n"
        "    {%- else -%}\n"
        "        {{- raise_exception('Unexpected content type.') -}}\n"
        "    {%- endif -%}\n"
        "{%- endmacro -%}\n"
        "{%- if not messages %}\n"
        "    {{- raise_exception('No messages provided.') }}\n"
        "{%- endif %}\n"
        "{%- if tools and tools is iterable and tools is not mapping %}\n"
        "    {{- '<|im_start|>system\\n' }}\n"
        "    {{- '# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>' }}\n"
        "    {%- for tool in tools %}\n"
        "        {{- '\\n' }}\n"
        "        {{- tool | tojson }}\n"
        "    {%- endfor %}\n"
        "    {{- '\\n</tools>' }}\n"
        "    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {%- set content = render_content(messages[0].content, true)|trim %}\n"
        "        {%- if content %}\n"
        "            {{- '\\n\\n' + content }}\n"
        "        {%- endif %}\n"
        "    {%- endif %}\n"
        "    {{- '<|im_end|>\\n' }}\n"
        "{%- else %}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {%- set content = render_content(messages[0].content, true)|trim %}\n"
        "        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}\n"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
        "{%- for message in messages[::-1] %}\n"
        "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
        "    {%- if ns.multi_step_tool and message.role == 'user' %}\n"
        "        {%- set content = render_content(message.content)|trim %}\n"
        "        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}\n"
        "            {%- set ns.multi_step_tool = false %}\n"
        "            {%- set ns.last_query_index = index %}\n"
        "        {%- endif %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if ns.multi_step_tool %}\n"
        "    {{- raise_exception('No user query found in messages.') }}\n"
        "{%- endif %}\n"
        "{%- for message in messages %}\n"
        "    {%- set content = render_content(message.content)|trim %}\n"
        "    {%- if message.role == 'system' %}\n"
        "        {%- if not loop.first %}\n"
        "            {{- raise_exception('System message must be at the beginning.') }}\n"
        "        {%- endif %}\n"
        "    {%- elif message.role == 'user' %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
        "    {%- elif message.role == 'assistant' %}\n"
        "        {%- set reasoning_content = '' %}\n"
        "        {%- if message.reasoning_content is string %}\n"
        "            {%- set reasoning_content = message.reasoning_content %}\n"
        "        {%- else %}\n"
        "            {%- if '</think>' in content %}\n"
        "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
        "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
        "            {%- endif %}\n"
        "        {%- endif %}\n"
        "        {%- set reasoning_content = reasoning_content|trim %}\n"
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "        {%- endif %}\n"
        "        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}\n"
        "            {%- for tool_call in message.tool_calls %}\n"
        "                {%- if tool_call.function is defined %}\n"
        "                    {%- set tool_call = tool_call.function %}\n"
        "                {%- endif %}\n"
        "                {%- if loop.first %}\n"
        "                    {%- if content|trim %}\n"
        "                        {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n"
        "                    {%- else %}\n"
        "                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n"
        "                    {%- endif %}\n"
        "                {%- else %}\n"
        "                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n"
        "                {%- endif %}\n"
        "                {%- if tool_call.arguments is defined %}\n"
        "                    {%- for args_name, args_value in tool_call.arguments|items %}\n"
        "                        {{- '<parameter=' + args_name + '>\\n' }}\n"
        "                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}\n"
        "                        {{- args_value }}\n"
        "                        {{- '\\n</parameter>\\n' }}\n"
        "                    {%- endfor %}\n"
        "                {%- endif %}\n"
        "                {{- '</function>\\n</tool_call>' }}\n"
        "            {%- endfor %}\n"
        "        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}\n"
        "    {%- elif message.role == 'tool' %}\n"
        "        {%- if loop.previtem and loop.previtem.role != 'tool' %}\n"
        "            {{- '<|im_start|>user' }}\n"
        "        {%- endif %}\n"
        "        {{- '\\n<tool_response>\\n' }}\n"
        "        {{- content }}\n"
        "        {{- '\\n</tool_response>' }}\n"
        "        {%- if not loop.last and loop.nextitem.role != 'tool' %}\n"
        "            {{- '<|im_end|>\\n' }}\n"
        "        {%- elif loop.last %}\n"
        "            {{- '<|im_end|>\\n' }}\n"
        "        {%- endif %}\n"
        "    {%- else %}\n"
        "        {{- raise_exception('Unexpected message role.') }}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>assistant\\n' }}\n"
        "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
        "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
        "    {%- else %}\n"
        "        {{- '<think>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}\n"
    )

    def __init__(self, enable_thinking: bool = True, **kwargs):
        """
        Initializes the MiniCPM-V-4.6 Handler.

        Args:
            enable_thinking (bool): Controls whether to open a `<think>` block for reasoning.
                                    Defaults to False as per the standard template logic.
        """
        self.enable_thinking = enable_thinking
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        # Inject the thinking variable into the Jinja environment
        self.extra_template_arguments["enable_thinking"] = self.enable_thinking

        # MiniCPM uses standard <|im_end|> ChatML stop formatting
        kwargs['stop'] = [self.MINICPM_PAD_TOKEN, self.MINICPM_EOS_TOKEN]

        if self.verbose:
            print(f"{self.log_prefix}(enable_thinking={self.enable_thinking}) - Start processing")

        return super().__call__(**kwargs)


class Gemma3ChatHandler(MTMDChatHandler):

    GEMMA3_BOI_TOKEN  = "<start_of_image>"
    GEMMA3_EOI_TOKEN = "<end_of_image>"
    GEMMA3_BOS_TOKEN = "<bos>"
    GEMMA3_EOS_TOKEN = "<eos>"

    CHAT_FORMAT = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"
        "{% if messages[0]['content'] is string %}"
        "{% set first_user_prefix = messages[0]['content'] + '\n\n' %}"
        "{% else %}"
        "{% set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' %}"
        "{% endif %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set first_user_prefix = '' %}"
        "{% endif %}"

        "{% for message in loop_messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
        "{% endif %}"

        "{% if message['role'] == 'assistant' %}"
        "{% set role = 'model' %}"
        "{% else %}"
        "{% set role = message['role'] %}"
        "{% endif %}"

        "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else '') }}"

        "{% if message['content'] is string %}"
        "{{ message['content'] | trim }}"
        "{% elif message['content'] is iterable %}"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'image_url' and item['image_url'] is string %}"
        "{{ '<start_of_image>' + item['image_url'] + '<end_of_image>' }}"
        "{% elif item['type'] == 'image_url' and item['image_url'] is mapping %}"
        "{{ '<start_of_image>' + item['image_url']['url'] + '<end_of_image>' }}"
        "{% elif item['type'] == 'text' %}"
        "{{ item['text'] | trim }}"
        "{% endif %}"
        "{% endfor %}"
        "{% else %}"
        "{{ raise_exception('Invalid content type') }}"
        "{% endif %}"

        "<end_of_turn>\n"
        "{% endfor %}"

        "{% if add_generation_prompt %}"
        "<start_of_turn>model\n"
        "{% endif %}"
    )


class Gemma4ChatHandler(MTMDChatHandler):
    """
    Handler for Gemma 4 models.

    Note on `enable_thinking`:
        The `enable_thinking` toggle is currently ONLY supported by Gemma4 31B and 26BA4B models.
        It is NOT supported by Gemma4 E2B and E4B models.

    [Important Note for Audio Processing!]
        It is recommended to use BF16 mmproj for Gemma4 E2B and E4B models.
        Other quantizations are known to have degraded performance;
        ref comment: https://github.com/ggml-org/llama.cpp/pull/21421#issuecomment-4230306463
    """

    # The special token in Gemma 4
    GEMMA4_BOI_TOKEN  = "<|image>"
    GEMMA4_EOI_TOKEN = "<image|>"
    GEMMA4_BOA_TOKEN  = "<|audio>"
    GEMMA4_EOA_TOKEN = "<audio|>"
    GEMMA4_BOS_TOKEN = "<bos>"
    GEMMA4_EOS_TOKEN = "<eos>"
    GEMMA4_SOT_TOKEN = "<|turn>"
    GEMMA4_EOT_TOKEN = "<turn|>"
    GEMMA4_SOC_TOKEN = "<|channel>"
    GEMMA4_EOC_TOKEN = "<channel|>"
    GEMMA4_STC_TOKEN = "<|tool_call>"
    GEMMA4_ETC_TOKEN = "<tool_call|>"
    GEMMA4_STD_TOKEN = "<|tool>"
    GEMMA4_ETD_TOKEN = "<tool|>"
    GEMMA4_STR_TOKEN = "<|tool_response>"
    GEMMA4_ETR_TOKEN = "<tool_response|>"

    CHAT_FORMAT = (
        "{%- macro format_parameters(properties, required, filter_keys=false) -%}\n"
        "    {%- set standard_keys = ['description', 'type', 'properties', 'required', 'nullable'] -%}\n"
        "    {%- set ns = namespace(found_first=false) -%}\n"
        "    {%- for key, value in properties | dictsort -%}\n"
        "        {%- set add_comma = false -%}\n"
        "        {%- if not filter_keys or key not in standard_keys -%}\n"
        "            {%- if ns.found_first %},{% endif -%}\n"
        "            {%- set ns.found_first = true -%}\n"
        "            {{ key }}:{\n"
        "            {%- if value['description'] -%}\n"
        "                description:<|\"|>{{ value['description'] }}<|\"|>\n"
        "                {%- set add_comma = true -%}\n"
        "            {%- endif -%}\n"
        "            {%- if value['type'] | upper == 'STRING' -%}\n"
        "                {%- if value['enum'] -%}\n"
        "                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}\n"
        "                    enum:{{ format_argument(value['enum']) }}\n"
        "                {%- endif -%}\n"
        "            {%- elif value['type'] | upper == 'ARRAY' -%}\n"
        "                {%- if value['items'] is mapping and value['items'] -%}\n"
        "                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}\n"
        "                    items:{\n"
        "                    {%- set ns_items = namespace(found_first=false) -%}\n"
        "                    {%- for item_key, item_value in value['items'] | dictsort -%}\n"
        "                        {%- if item_value is not none -%}\n"
        "                            {%- if ns_items.found_first %},{% endif -%}\n"
        "                            {%- set ns_items.found_first = true -%}\n"
        "                            {%- if item_key == 'properties' -%}\n"
        "                                properties:{\n"
        "                                {%- if item_value is mapping -%}\n"
        "                                    {{- format_parameters(item_value, value['items']['required'] | default([])) -}}\n"
        "                                {%- endif -%}\n"
        "                                }\n"
        "                            {%- elif item_key == 'required' -%}\n"
        "                                required:[\n"
        "                                {%- for req_item in item_value -%}\n"
        "                                    <|\"|>{{- req_item -}}<|\"|>\n"
        "                                    {%- if not loop.last %},{% endif -%}\n"
        "                                {%- endfor -%}\n"
        "                                ]\n"
        "                            {%- elif item_key == 'type' -%}\n"
        "                                {%- if item_value is string -%}\n"
        "                                    type:{{ format_argument(item_value | upper) }}\n"
        "                                {%- else -%}\n"
        "                                    type:{{ format_argument(item_value | map('upper') | list) }}\n"
        "                                {%- endif -%}\n"
        "                            {%- else -%}\n"
        "                                {{ item_key }}:{{ format_argument(item_value) }}\n"
        "                            {%- endif -%}\n"
        "                        {%- endif -%}\n"
        "                    {%- endfor -%}\n"
        "                    }\n"
        "                {%- endif -%}\n"
        "            {%- endif -%}\n"
        "            {%- if value['nullable'] %}\n"
        "                {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}\n"
        "                nullable:true\n"
        "            {%- endif -%}\n"
        "            {%- if value['type'] | upper == 'OBJECT' -%}\n"
        "                {%- if value['properties'] is defined and value['properties'] is mapping -%}\n"
        "                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}\n"
        "                    properties:{\n"
        "                    {{- format_parameters(value['properties'], value['required'] | default([])) -}}\n"
        "                    }\n"
        "                {%- elif value is mapping -%}\n"
        "                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}\n"
        "                    properties:{\n"
        "                    {{- format_parameters(value, value['required'] | default([]), filter_keys=true) -}}\n"
        "                    }\n"
        "                {%- endif -%}\n"
        "                {%- if value['required'] -%}\n"
        "                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}\n"
        "                    required:[\n"
        "                    {%- for item in value['required'] | default([]) -%}\n"
        "                        <|\"|>{{- item -}}<|\"|>\n"
        "                        {%- if not loop.last %},{% endif -%}\n"
        "                    {%- endfor -%}\n"
        "                    ]\n"
        "                {%- endif -%}\n"
        "            {%- endif -%}\n"
        "            {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}\n"
        "            type:<|\"|>{{ value['type'] | upper }}<|\"|>}\n"
        "        {%- endif -%}\n"
        "    {%- endfor -%}\n"
        "{%- endmacro -%}\n"
        "{%- macro format_function_declaration(tool_data) -%}\n"
        "    declaration:{{- tool_data['function']['name'] -}}{description:<|\"|>{{- tool_data['function']['description'] -}}<|\"|>\n"
        "    {%- set params = tool_data['function']['parameters'] -%}\n"
        "    {%- if params -%}\n"
        "        ,parameters:{\n"
        "        {%- if params.get('properties') -%}\n"
        "            properties:{ {{- format_parameters(params['properties'], params['required']) -}} },\n"
        "        {%- endif -%}\n"
        "        {%- if params.get('required') -%}\n"
        "            required:[\n"
        "            {%- for item in params['required'] -%}\n"
        "                <|\"|>{{- item -}}<|\"|>\n"
        "                {{- ',' if not loop.last -}}\n"
        "            {%- endfor -%}\n"
        "            ],\n"
        "        {%- endif -%}\n"
        "        {%- if params.get('type') -%}\n"
        "            type:<|\"|>{{- params['type'] | upper -}}<|\"|>}\n"
        "        {%- endif -%}\n"
        "    {%- endif -%}\n"
        "    {%- if 'response' in tool_data['function'] -%}\n"
        "        {%- set response_declaration = tool_data['function']['response'] -%}\n"
        "        ,response:{\n"
        "        {%- if response_declaration['description'] -%}\n"
        "            description:<|\"|>{{- response_declaration['description'] -}}<|\"|>,\n"
        "        {%- endif -%}\n"
        "        {%- if response_declaration['type'] | upper == 'OBJECT' -%}\n"
        "            type:<|\"|>{{- response_declaration['type'] | upper -}}<|\"|>}\n"
        "        {%- endif -%}\n"
        "    {%- endif -%}\n"
        "    }\n"
        "{%- endmacro -%}\n"
        "{%- macro format_argument(argument, escape_keys=True) -%}\n"
        "    {%- if argument is string -%}\n"
        "        {{- '<|\"|>' + argument + '<|\"|>' -}}\n"
        "    {%- elif argument is boolean -%}\n"
        "        {{- 'true' if argument else 'false' -}}\n"
        "    {%- elif argument is mapping -%}\n"
        "        {{- '{' -}}\n"
        "        {%- set ns = namespace(found_first=false) -%}\n"
        "        {%- for key, value in argument | dictsort -%}\n"
        "            {%- if ns.found_first %},{% endif -%}\n"
        "            {%- set ns.found_first = true -%}\n"
        "            {%- if escape_keys -%}\n"
        "                {{- '<|\"|>' + key + '<|\"|>' -}}\n"
        "            {%- else -%}\n"
        "                {{- key -}}\n"
        "            {%- endif -%}\n"
        "            :{{- format_argument(value, escape_keys=escape_keys) -}}\n"
        "        {%- endfor -%}\n"
        "        {{- '}' -}}\n"
        "    {%- elif argument is sequence -%}\n"
        "        {{- '[' -}}\n"
        "        {%- for item in argument -%}\n"
        "            {{- format_argument(item, escape_keys=escape_keys) -}}\n"
        "            {%- if not loop.last %},{% endif -%}\n"
        "        {%- endfor -%}\n"
        "        {{- ']' -}}\n"
        "    {%- else -%}\n"
        "        {{- argument -}}\n"
        "    {%- endif -%}\n"
        "{%- endmacro -%}\n"
        "{%- macro strip_thinking(text) -%}\n"
        "    {%- set ns = namespace(result='') -%}\n"
        "    {%- for part in text.split('<channel|>') -%}\n"
        "        {%- if '<|channel>' in part -%}\n"
        "            {%- set ns.result = ns.result + part.split('<|channel>')[0] -%}\n"
        "        {%- else -%}\n"
        "            {%- set ns.result = ns.result + part -%}\n"
        "        {%- endif -%}\n"
        "    {%- endfor -%}\n"
        "    {{- ns.result | trim -}}\n"
        "{%- endmacro -%}\n"
        "\n"
        "{%- macro format_tool_response_block(tool_name, response) -%}\n"
        "    {{- '<|tool_response>' -}}\n"
        "    {%- if response is mapping -%}\n"
        "        {{- 'response:' + tool_name + '{' -}}\n"
        "        {%- for key, value in response | dictsort -%}\n"
        "            {{- key -}}:{{- format_argument(value, escape_keys=False) -}}\n"
        "            {%- if not loop.last %},{% endif -%}\n"
        "        {%- endfor -%}\n"
        "        {{- '}' -}}\n"
        "    {%- else -%}\n"
        "        {{- 'response:' + tool_name + '{value:' + format_argument(response, escape_keys=False) + '}' -}}\n"
        "    {%- endif -%}\n"
        "    {{- '<tool_response|>' -}}\n"
        "{%- endmacro -%}\n"
        "\n"
        "{%- set ns = namespace(prev_message_type=None) -%}\n"
        "{%- set loop_messages = messages -%}\n"
        "{{- bos_token -}}\n"
        "{#- Handle System/Tool Definitions Block -#}\n"
        "{%- if (enable_thinking is defined and enable_thinking) or tools or messages[0]['role'] in ['system', 'developer'] -%}\n"
        "    {{- '<|turn>system\\n' -}}\n"
        "    {#- Inject Thinking token at the very top of the FIRST system turn -#}\n"
        "    {%- if enable_thinking is defined and enable_thinking -%}\n"
        "        {{- '<|think|>\\n' -}}\n"
        "        {%- set ns.prev_message_type = 'think' -%}\n"
        "    {%- endif -%}\n"
        "    {%- if messages[0]['role'] in ['system', 'developer'] -%}\n"
        "        {%- if messages[0]['content'] is string -%}\n"
        "            {{- messages[0]['content'] | trim -}}\n"
        "        {%- elif messages[0]['content'] is sequence -%}\n"
        "            {%- for item in messages[0]['content'] -%}\n"
        "                {{- item['text'] | trim + ' '-}}\n"
        "            {%- endfor -%}\n"
        "        {%- endif -%}\n"
        "        {%- set loop_messages = messages[1:] -%}\n"
        "    {%- endif -%}\n"
        "    {%- if tools -%}\n"
        "        {%- for tool in tools %}\n"
        "            {{- '<|tool>' -}}\n"
        "            {{- format_function_declaration(tool) | trim -}}\n"
        "            {{- '<tool|>' -}}\n"
        "        {%- endfor %}\n"
        "        {%- set ns.prev_message_type = 'tool' -%}\n"
        "    {%- endif -%}\n"
        "    {{- '<turn|>\\n' -}}\n"
        "{%- endif %}\n"
        "\n"
        "{#- Pre-scan: find last user message index for reasoning guard -#}\n"
        "{%- set ns_turn = namespace(last_user_idx=-1) -%}\n"
        "{%- for i in range(loop_messages | length) -%}\n"
        "    {%- if loop_messages[i]['role'] == 'user' -%}\n"
        "        {%- set ns_turn.last_user_idx = i -%}\n"
        "    {%- endif -%}\n"
        "{%- endfor -%}\n"
        "\n"
        "{#- Loop through messages -#}\n"
        "{%- for message in loop_messages -%}\n"
        "    {%- if message['role'] != 'tool' -%}\n"
        "    {%- set ns.prev_message_type = None -%}\n"
        "    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}\n"
        "    {#- Detect continuation: suppress duplicate <|turn>model when previous non-tool message was also assistant -#}\n"
        "    {%- set prev_nt = namespace(role=None, found=false) -%}\n"
        "    {%- if loop.index0 > 0 -%}\n"
        "        {%- for j in range(loop.index0 - 1, -1, -1) -%}\n"
        "            {%- if not prev_nt.found -%}\n"
        "                {%- if loop_messages[j]['role'] != 'tool' -%}\n"
        "                    {%- set prev_nt.role = loop_messages[j]['role'] -%}\n"
        "                    {%- set prev_nt.found = true -%}\n"
        "                {%- endif -%}\n"
        "            {%- endif -%}\n"
        "        {%- endfor -%}\n"
        "    {%- endif -%}\n"
        "    {%- set continue_same_model_turn = (role == 'model' and prev_nt.role == 'assistant') -%}\n"
        "    {%- if not continue_same_model_turn -%}\n"
        "        {{- '<|turn>' + role + '\\n' }}\n"
        "    {%- endif -%}\n"
        "\n"
        "    {#- Render reasoning/reasoning_content as thinking channel -#}\n"
        "    {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}\n"
        "    {%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}\n"
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        "    {%- endif -%}\n"
        "\n"
        "            {%- if message.get('tool_calls') -%}\n"
        "                {%- for tool_call in message['tool_calls'] -%}\n"
        "                    {%- set function = tool_call['function'] -%}\n"
        "                    {{- '<|tool_call>call:' + function['name'] + '{' -}}\n"
        "                    {%- if function['arguments'] is mapping -%}\n"
        "                        {%- set ns_args = namespace(found_first=false) -%}\n"
        "                        {%- for key, value in function['arguments'] | dictsort -%}\n"
        "                            {%- if ns_args.found_first %},{% endif -%}\n"
        "                            {%- set ns_args.found_first = true -%}\n"
        "                            {{- key -}}:{{- format_argument(value, escape_keys=False) -}}\n"
        "                        {%- endfor -%}\n"
        "                    {%- elif function['arguments'] is string -%}\n"
        "                        {{- function['arguments'] -}}\n"
        "                    {%- endif -%}\n"
        "                    {{- '}<tool_call|>' -}}\n"
        "                {%- endfor -%}\n"
        "                {%- set ns.prev_message_type = 'tool_call' -%}\n"
        "            {%- endif -%}\n"
        "\n"
        "            {%- set ns_tr_out = namespace(flag=false) -%}\n"
        "            {%- if message.get('tool_responses') -%}\n"
        "                {#- Legacy: tool_responses embedded on the assistant message (Google/Gemma native) -#}\n"
        "                {%- for tool_response in message['tool_responses'] -%}\n"
        "                    {{- format_tool_response_block(tool_response['name'] | default('unknown'), tool_response['response']) -}}\n"
        "                    {%- set ns_tr_out.flag = true -%}\n"
        "                    {%- set ns.prev_message_type = 'tool_response' -%}\n"
        "                {%- endfor -%}\n"
        "            {%- elif message.get('tool_calls') -%}\n"
        "                {#- OpenAI Chat Completions: forward-scan consecutive role:tool messages -#}\n"
        "                {%- set ns_tool_scan = namespace(stopped=false) -%}\n"
        "                {%- for k in range(loop.index0 + 1, loop_messages | length) -%}\n"
        "                    {%- if ns_tool_scan.stopped -%}\n"
        "                    {%- elif loop_messages[k]['role'] != 'tool' -%}\n"
        "                        {%- set ns_tool_scan.stopped = true -%}\n"
        "                    {%- else -%}\n"
        "                        {%- set follow = loop_messages[k] -%}\n"
        "                        {#- Resolve tool_call_id to function name -#}\n"
        "                        {%- set ns_tname = namespace(name=follow.get('name') | default('unknown')) -%}\n"
        "                        {%- for tc in message['tool_calls'] -%}\n"
        "                            {%- if tc.get('id') == follow.get('tool_call_id') -%}\n"
        "                                {%- set ns_tname.name = tc['function']['name'] -%}\n"
        "                            {%- endif -%}\n"
        "                        {%- endfor -%}\n"
        "                        {#- Handle content as string or content-parts array -#}\n"
        "                        {%- set tool_body = follow.get('content') -%}\n"
        "                        {%- if tool_body is string -%}\n"
        "                            {{- format_tool_response_block(ns_tname.name, tool_body) -}}\n"
        "                        {%- elif tool_body is sequence and tool_body is not string -%}\n"
        "                            {%- set ns_txt = namespace(s='') -%}\n"
        "                            {%- for part in tool_body -%}\n"
        "                                {%- if part.get('type') == 'text' -%}\n"
        "                                    {%- set ns_txt.s = ns_txt.s + (part.get('text') | default('')) -%}\n"
        "                                {%- endif -%}\n"
        "                            {%- endfor -%}\n"
        "                            {{- format_tool_response_block(ns_tname.name, ns_txt.s) -}}\n"
        "                            {%- for part in tool_body -%}\n"
        "                                {%- if part.get('type') == 'image_url' -%}\n"
        "                                    {%- set url_val = part['image_url'] if part['image_url'] is string else part['image_url']['url'] -%}\n"
        "                                    {{- '<|image|>' + url_val -}}\n"
        "                                {%- elif part.get('type') in ['audio_url', 'input_audio'] -%}\n"
        "                                    {%- if part.get('type') == 'audio_url' -%}\n"
        "                                        {%- set audio_val = part['audio_url'] if part['audio_url'] is string else part['audio_url']['url'] -%}\n"
        "                                        {{- '<|audio|>' + audio_val -}}\n"
        "                                    {%- elif part.get('type') == 'input_audio' -%}\n"
        "                                        {%- set audio_val = part['input_audio'] if part['input_audio'] is string else ('data:audio/' + part['input_audio']['format'] + ';base64,' + part['input_audio']['data']) -%}\n"
        "                                        {{- '<|audio|>' + audio_val -}}\n"
        "                                    {%- endif -%}\n"
        # "                              {%- elif part.get('type') == 'video_url' -%}\n"
        # "                                  {%- set video_val = part['video_url'] if part['video_url'] is string else part['video_url']['url'] -%}\n"
        # "                                  {{- '<|video|>' + video_val -}}\n"
        "                                {%- endif -%}\n"
        "                            {%- endfor -%}\n"
        "                        {%- else -%}\n"
        "                            {{- format_tool_response_block(ns_tname.name, tool_body) -}}\n"
        "                        {%- endif -%}\n"
        "                        {%- set ns_tr_out.flag = true -%}\n"
        "                        {%- set ns.prev_message_type = 'tool_response' -%}\n"
        "                    {%- endif -%}\n"
        "                {%- endfor -%}\n"
        "            {%- endif -%}\n"
        "\n"
        "            {%- set captured_content -%}\n"
        "            {%- if message['content'] is string -%}\n"
        "                {%- if role == 'model' -%}\n"
        "                    {{- strip_thinking(message['content']) -}}\n"
        "                {%- else -%}\n"
        "                    {{- message['content'] | trim -}}\n"
        "                {%- endif -%}\n"
        "            {%- elif message['content'] is sequence -%}\n"
        "                {%- for item in message['content'] -%}\n"
        "                    {%- if item['type'] == 'text' -%}\n"
        "                        {%- if role == 'model' -%}\n"
        "                            {{- strip_thinking(item['text']) -}}\n"
        "                        {%- else -%}\n"
        "                            {{- item['text'] | trim -}}\n"
        "                        {%- endif -%}\n"
        "                    {%- elif item['type'] == 'image_url' -%}\n"
        "                        {%- set url_val = item['image_url'] if item['image_url'] is string else item['image_url']['url'] -%}\n"
        "                        {{- '<|image|>' + url_val -}}\n"
        "                        {%- set ns.prev_message_type = 'image' -%}\n"
        "                    {%- elif item['type'] in ['audio_url', 'input_audio'] -%}\n"
        "                        {%- if item['type'] == 'audio_url' -%}\n"
        "                            {%- set audio_val = item['audio_url'] if item['audio_url'] is string else item['audio_url']['url'] -%}\n"
        "                            {{- '<|audio|>' + audio_val -}}\n"
        "                        {%- elif item['type'] == 'input_audio' -%}\n"
        "                            {%- set audio_val = item['input_audio'] if item['input_audio'] is string else ('data:audio/' + item['input_audio']['format'] + ';base64,' + item['input_audio']['data']) -%}\n"
        "                            {{- '<|audio|>' + audio_val -}}\n"
        "                        {%- endif -%}\n"
        "                        {%- set ns.prev_message_type = 'audio' -%}\n"
        "                    {%- endif -%}\n"
        # "                    {%- elif item['type'] == 'video_url' -%}\n"
        # "                        {%- set video_val = item['video_url'] if item['video_url'] is string else item['video_url']['url'] -%}\n"
        # "                        {{- '<|video|>' + video_val -}}\n"
        # "                        {%- set ns.prev_message_type = 'video' -%}\n"
        "                {%- endfor -%}\n"
        "            {%- endif -%}\n"
        "            {%- endset -%}\n"
        "\n"
        "            {{- captured_content -}}\n"
        "            {%- set has_content = captured_content | trim | length > 0 -%}\n"
        "\n"
        "        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}\n"
        "            {{- '<|tool_response>' -}}\n"
        "        {%- elif not (ns_tr_out.flag and not has_content) -%}\n"
        "            {{- '<turn|>\\n' -}}\n"
        "        {%- endif -%}\n"
        "    {%- endif -%}\n"
        "{%- endfor -%}\n"
        "\n"
        "{%- if add_generation_prompt -%}\n"
        "    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}\n"
        "        {{- '<|turn>model\\n' -}}\n"
        "        {%- if not enable_thinking | default(false) -%}\n"
        "            {{- '<|channel>thought\\n<channel|>' -}}\n"
        "        {%- endif -%}\n"
        "    {%- endif -%}\n"
        "{%- endif -%}\n"
    )

    def __init__(self, enable_thinking: bool = True, **kwargs):
        """
        Initializes the Gemma 4 Handler.

        Args:
            enable_thinking (bool): Controls whether the <|think|> tag is injected and
                                    manages <|channel>thought behavior.
                                    Note: ONLY supported on Gemma4 31B and 26BA4B models.
                                    NOT supported on Gemma4 E2B and E4B models.
        """
        self.enable_thinking = enable_thinking
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        # Inject the thinking variable into the Jinja environment
        self.extra_template_arguments["enable_thinking"] = self.enable_thinking

        # Set the stop token based on Gemma 4's format (<turn|>)
        # generation_config.json:   "eos_token_id": [1, 106, 50]
        kwargs['stop'] = [self.GEMMA4_EOS_TOKEN, self.GEMMA4_EOT_TOKEN, self.GEMMA4_STR_TOKEN]

        if self.verbose:
            print(f"{self.log_prefix}(enable_thinking={self.enable_thinking}) - Start processing")

        return super().__call__(**kwargs)


class GLM41VChatHandler(MTMDChatHandler):
    # Note: Make sure the GGUF files of your converted model and mmproj are F16 or F32.

    GLM41V_EOS_TOKEN = "<|endoftext|>"
    GLM41V_PAD_TOKEN = "<|endoftext|>"
    GLM41V_IMAGE_START_TOKEN = "<|begin_of_image|>"
    GLM41V_IMAGE_END_TOKEN = "<|end_of_image|>"

    CHAT_FORMAT = (
        "[gMASK]<sop>\n"
        "{%- for msg in messages -%}"
            "{%- if msg.role == 'system' -%}"
                "<|system|>\n{{ msg.content }}{{ GLM41V_EOS_TOKEN }}"
            "{%- elif msg.role == 'user' -%}"
                "<|user|>\n"
                "{%- if msg.content is string -%}"
                    "{{ msg.content }}"
                "{%- else -%}"
                    "{%- for item in msg.content -%}"
                        "{%- if item.type == 'image_url' or 'image_url' in item -%}"
                            "<|begin_of_image|>"
                            "{%- if item.image_url is string -%}"
                                "{{- item.image_url -}}"
                            "{%- else -%}"
                                "{{- item.image_url.url -}}"
                            "{%- endif -%}"
                            "<|end_of_image|>"
                        "{%- elif item.type == 'text' -%}"
                            "{{ item.text }}"
                        "{%- endif -%}"
                    "{%- endfor -%}"
                "{%- endif -%}{{ GLM41V_EOS_TOKEN }}"
            "{%- elif msg.role == 'assistant' -%}"
                "{%- if msg.metadata -%}"
                    "<|assistant|>{{ msg.metadata }}\n{{ msg.content }}{{ GLM41V_EOS_TOKEN }}"
                "{%- else -%}"
                    "<|assistant|>\n{{ msg.content }}{{ GLM41V_EOS_TOKEN }}"
                "{%- endif -%}"
            "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
            "<|assistant|>\n"
        "{%- endif -%}"
    )

    def __call__(self, **kwargs):
        self.extra_template_arguments["GLM41V_EOS_TOKEN"] = self.GLM41V_EOS_TOKEN
        # https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/blob/main/generation_config.json
        stop_tokens = [self.GLM41V_EOS_TOKEN, "<|user|>", "<|observation|>", "</answer>"] # Stop token patch
        kwargs['stop'] = stop_tokens

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix} - Start processing")

        # Use parent implementation
        return super().__call__(**kwargs)


class GLM46VChatHandler(MTMDChatHandler):
    GLM46V_EOS_TOKEN = "<|endoftext|>"
    GLM46V_PAD_TOKEN = "<|endoftext|>"
    GLM46V_IMAGE_START_TOKEN = "<|begin_of_image|>"
    GLM46V_IMAGE_END_TOKEN = "<|end_of_image|>"

    CHAT_FORMAT = (
        "[gMASK]<sop>"
        "{%- if tools -%}"
            "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            "{%- for tool in tools -%}"
                "{{ tool | tojson(ensure_ascii=False) }}\n"
            "{%- endfor -%}"
            "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
            "<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n...\n</tool_call>"
        "{%- endif -%}"

        "{%- for m in messages -%}"
            "{%- if m.role == 'system' -%}"
                "<|system|>\n{{ m.content }}"
            "{%- elif m.role == 'user' -%}"
                "<|user|>\n"
                "{%- if m.content is string -%}"
                    "{{ m.content }}"
                "{%- else -%}"
                    "{%- for item in m.content -%}"
                        "{%- if item.type == 'image_url' or 'image_url' in item -%}"
                            "<|begin_of_image|>"
                            "{%- if item.image_url is string -%}"
                                "{{- item.image_url -}}"
                            "{%- else -%}"
                                "{{- item.image_url.url -}}"
                            "{%- endif -%}"
                            "<|end_of_image|>"
                        "{%- elif item.type == 'text' -%}"
                            "{{ item.text }}"
                        "{%- endif -%}"
                    "{%- endfor -%}"
                "{%- endif -%}"
                # If enable_thinking is disabled, insert `/nothink` according to the source code logic.
                "{{ '/nothink' if not enable_thinking else '' }}"
            "{%- elif m.role == 'assistant' -%}"
                "<|assistant|>"
                "{%- if enable_thinking -%}"
                    "{%- set reasoning = m.reasoning_content if m.reasoning_content is string else '' -%}"
                    "\n<think>{{ reasoning.strip() }}</think>"
                "{%- else -%}"
                    "\n<think></think>"
                "{%- endif -%}"
                "{{ '\n' + m.content.strip() if m.content.strip() else '' }}"
            "{%- endif -%}"
            "{{ GLM46V_EOS_TOKEN }}"
        "{%- endfor -%}"

        "{%- if add_generation_prompt -%}"
            "<|assistant|>\n"
            "{{ '<think>' if enable_thinking else '<think></think>\n' }}"
        "{%- endif -%}"
    )

    def __init__(self, enable_thinking: bool = True, **kwargs):
        """
        GLM-4.6V Handler
        Parameters:
        - enable_thinking (bool): Whether to enable the model's think process. The default is True.
        """
        self.enable_thinking = enable_thinking
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        self.extra_template_arguments["enable_thinking"] = self.enable_thinking
        self.extra_template_arguments["GLM46V_EOS_TOKEN"] = self.GLM46V_EOS_TOKEN

        # https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/main/generation_config.json
        kwargs['stop'] = [self.GLM46V_EOS_TOKEN, "<|user|>", "<|observation|>", "<|code_middle|>"] # Stop token patch

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix}(enable_thinking={self.enable_thinking}) - Start processing")

        return super().__call__(**kwargs)


class GraniteDoclingChatHandler(MTMDChatHandler):
    """
    Handler for Granite-Docling models.

    Format(512x512): <loc_xmin><loc_ymin><loc_xmax><loc_ymax>Content

    Note(JamePeng): The GGUF files for Model and MMPROJ should be BF16 version !!!
                    Since the model does not have special tokens for the start and end of an image,
                    it is recommended to process only one image at a time.
                    You can iterate through the images individually for recognition.

    """
    GRANITE_BOS_TOKEN = "<|start_of_role|>"
    GRANITE_EOS_TOKEN = "<|end_of_text|>"
    GRANITE_PAD_TOKEN = "<|end_of_text|>"
    GRANITE_IMAGE_TOKEN = "<image>"

    CHAT_FORMAT = (
        "{%- for message in messages -%}"
            "{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' -}}"
            "{%- if message['content'] is string -%}"
                "{{- message['content'] -}}"
            "{%- else -%}"
                "{%- for part in message['content'] -%}"
                    "{%- if part['type'] == 'text' -%}"
                        "{{- part['text'] -}}"
                    "{%- elif part['type'] == 'image_url' -%}"
                        "{%- if part.image_url is string -%}"
                            "{{- part.image_url -}}"
                        "{%- else -%}"
                            "{{- part.image_url.url -}}"
                        "{%- endif -%}"
                    "{%- endif -%}"
                "{%- endfor -%}"
            "{%- endif -%}"
            "{{- '<|end_of_text|>\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
            "{{- '<|start_of_role|>assistant' -}}"
            # Support the 'controls' parameter if present in generation arguments
            "{%- if controls -%}{{- ' ' + controls | tojson() -}}{%- endif -%}"
            "{{- '<|end_of_role|>' -}}"
        "{%- endif -%}"
    )

    def __init__(self, controls: dict = None, **kwargs):
        """
        Granite-Docling Handler
        Args:
            controls (dict, optional): Operational parameters passed to the assistant role.

            The 'controls' parameter is used to guide the model's behavior or output format.
            Common examples for 'controls' include:
             - Document Parsing: {"mode": "document_parsing", "format": "json"}
        """
        self.controls = controls
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        # Inject controls into the template environment
        self.extra_template_arguments["controls"] = self.controls
        self.DEFAULT_SYSTEM_MESSAGE = None
        kwargs['stop'] = [self.GRANITE_EOS_TOKEN]

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix} - Start processing")


        return super().__call__(**kwargs)


class LFM2VLChatHandler(MTMDChatHandler):
    LFM2VL_BOS_TOKEN = "<|startoftext|>"
    LFM2VL_EOS_TOKEN = "<|im_end|>"
    LFM2VL_IMAGE_START_TOKEN = "<|image_start|>"
    LFM2VL_IMAGE_END_TOKEN = "<|image_end|>"

    CHAT_FORMAT = (
        "{%- for message in messages -%}"
            "{{ '<|im_start|>' + message['role'] + '\n' }}"
            "{%- if message['content'] is string -%}"
                "{{ message['content'] }}"
            "{%- else -%}"
                "{%- for content in message['content'] -%}"
                    "{%- if 'image_url' in content -%}"
                        "{%- if content.image_url is string -%}"
                            "<|image_start|>{{ content.image_url }}<|image_end|>"
                        "{%- else -%}"
                            "<|image_start|>{{ content.image_url.url }}<|image_end|>"
                        "{%- endif -%}"
                    "{%- elif content['type'] == 'text' -%}"
                        "{{ content['text'] }}"
                    "{%- endif -%}"
                "{%- endfor -%}"
            "{%- endif -%}"
            "{{ '<|im_end|>\n' }}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
            "{{ '<|im_start|>assistant\n' }}"
        "{%- endif -%}"
    )

    def __init__(self, image_min_tokens: int = -1, image_max_tokens: int = -1, **kwargs):
        """
        LFM2-VL Handler
        LiquidAI officially recommends configuring LFM2-VL with the following Vision parameters: min_image_tokens=64, max_image_tokens=256
        """
        self.image_min_tokens = image_min_tokens
        self.image_max_tokens = image_max_tokens
        super().__init__(image_min_tokens=self.image_min_tokens, image_max_tokens=self.image_max_tokens, **kwargs)

    def __call__(self, **kwargs):

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix} - Start processing")

        return super().__call__(**kwargs)


class LFM25VLChatHandler(MTMDChatHandler):
    """
    Handler for LFM2.5-VL multimodal models.

    Note(JamePeng): The suggestion is to compress the input image to 512x512 pixels to achieve native resolution processing.
    """
    # Aligned with LFM2.5-VL tokenizer_config
    LFM25VL_BOS_TOKEN = "<|startoftext|>"
    LFM25VL_EOS_TOKEN = "<|im_end|>"
    LFM25VL_PAD_TOKEN = "<|pad|>"

    # Image specific tokens
    LFM25VL_IMAGE_TOKEN = "<image>"
    LFM25VL_IMAGE_START_TOKEN = "<|image_start|>"
    LFM25VL_IMAGE_END_TOKEN = "<|image_end|>"
    LFM25VL_IMAGE_THUMBNAIL = "<|img_thumbnail|>"

    CHAT_FORMAT = (
        "{{- bos_token -}}\n"
        "{%- set keep_past_thinking = keep_past_thinking | default(false) -%}\n"
        "{%- set ns = namespace(system_prompt='', content='') -%}\n"
        "{%- if messages[0]['role'] == 'system' -%}\n"
        "    {%- set ns.system_prompt = messages[0]['content'] -%}\n"
        "    {%- set messages = messages[1:] -%}\n"
        "{%- endif -%}\n"
        "{%- if tools -%}\n"
        "    {%- set ns.system_prompt = ns.system_prompt + ('\\n' if ns.system_prompt else '') + 'List of tools: [' -%}\n"
        "    {%- for tool in tools -%}\n"
        "        {%- if tool is not string -%}\n"
        "            {%- set tool = tool | tojson -%}\n"
        "        {%- endif -%}\n"
        "        {%- set ns.system_prompt = ns.system_prompt + tool -%}\n"
        "        {%- if not loop.last -%}\n"
        "            {%- set ns.system_prompt = ns.system_prompt + ', ' -%}\n"
        "        {%- endif -%}\n"
        "    {%- endfor -%}\n"
        "    {%- set ns.system_prompt = ns.system_prompt + ']' -%}\n"
        "{%- endif -%}\n"
        "{%- if ns.system_prompt -%}\n"
        "    {{- '<|im_start|>system\\n' + ns.system_prompt + '<|im_end|>\\n' -}}\n"
        "{%- endif -%}\n"
        "{%- set ns.last_assistant_index = -1 -%}\n"
        "{%- for message in messages -%}\n"
        "    {%- if message['role'] == 'assistant' -%}\n"
        "        {%- set ns.last_assistant_index = loop.index0 -%}\n"
        "    {%- endif -%}\n"
        "{%- endfor -%}\n"
        "{%- for message in messages -%}\n"
        "    {{- '<|im_start|>' + message['role'] + '\\n' -}}\n"
        "    {%- set content = message['content'] -%}\n"
        "    {%- if content is not string -%}\n"
        "        {%- set ns.content = '' -%}\n"
        "        {#- MTMD-style Multimodal Injection (Audio stripped for VL model) -#}\n"
        "        {%- for item in content -%}\n"
        "            {%- if item['type'] == 'image_url' -%}\n"
        "                {%- set img_val = item['image_url'] if item['image_url'] is string else item['image_url']['url'] -%}\n"
        "                {%- set ns.content = ns.content + img_val -%}\n"
        "            {%- elif item['type'] == 'text' -%}\n"
        "                {%- set ns.content = ns.content + item['text'] -%}\n"
        "            {%- else -%}\n"
        "                {%- set ns.content = ns.content + (item | tojson) -%}\n"
        "            {%- endif -%}\n"
        "        {%- endfor -%}\n"
        "        {%- set content = ns.content -%}\n"
        "    {%- endif -%}\n"
        "    {%- if message['role'] == 'assistant' and not keep_past_thinking and loop.index0 != ns.last_assistant_index -%}\n"
        "        {%- if '</think>' in content -%}\n"
        "            {%- set content = content.split('</think>')[-1] | trim -%}\n"
        "        {%- endif -%}\n"
        "    {%- endif -%}\n"
        "    {{- content + '<|im_end|>\\n' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "    {{- '<|im_start|>assistant\\n' -}}\n"
        "{%- endif -%}\n"
    )

    def __init__(self, keep_past_thinking: bool = False, **kwargs):
        self.keep_past_thinking = keep_past_thinking
        super().__init__(**kwargs)


    def __call__(self, **kwargs):
        if self.image_min_tokens > 256:
            if self.verbose:
                print(f"{self.log_prefix}: For LFM2.5-VL, using values higher than 256 for `image_min_tokens` could cause errors. Please reset it to between 64 and 256.")
            self.image_min_tokens = -1

        self.extra_template_arguments["keep_past_thinking"] = self.keep_past_thinking

        kwargs['stop'] = [self.LFM25VL_EOS_TOKEN]

        if self.verbose:
            print(f"{self.log_prefix}(keep_past_thinking={self.keep_past_thinking}) - Start processing")
        return super().__call__(**kwargs)


class PaddleOCRChatHandler(MTMDChatHandler):
    """
    Handler for PaddleOCR 1.5/1.6 multimodal models.
    """

    PADDLEOCR_CLS_TOKEN = "<|begin_of_sentence|>"
    PADDLEOCR_BOS_TOKEN = "<s>"
    PADDLEOCR_EOS_TOKEN = "</s>"
    PADDLEOCR_SEP_TOKEN = "<|end_of_sentence|>"
    PADDLEOCR_IMAGE_BOS_TOKEN = "<|IMAGE_START|>"
    PADDLEOCR_IMAGE_EOS_TOKEN = "<|IMAGE_END|>"

    CHAT_FORMAT = (
        "{%- if not add_generation_prompt is defined -%}{%- set add_generation_prompt = true -%}{%- endif -%}"
        "{%- if not cls_token is defined -%}{%- set cls_token = '" + PADDLEOCR_CLS_TOKEN + "' -%}{%- endif -%}"
        "{%- if not eos_token is defined -%}{%- set eos_token = '" + PADDLEOCR_EOS_TOKEN + "' -%}{%- endif -%}"

        "{{- cls_token -}}"
        "{%- for message in messages -%}"
            "{%- if message['role'] == 'user' -%}"
                "{{- 'User: ' -}}"

                # Robust parsing: Check if content is string or list
                "{%- if message['content'] is string -%}"
                    "{{- message['content'] -}}"
                "{%- else -%}"
                    # Pass 1: Render all images first
                    "{%- for content in message['content'] -%}"
                        "{%- if content['type'] == 'image_url' and 'image_url' in content -%}"
                            "{{- '<|IMAGE_START|>' -}}"
                                "{%- if content.image_url is string -%}"
                                    "{{- content.image_url -}}"
                                "{%- else -%}"
                                    "{{- content.image_url.url -}}"
                                "{%- endif -%}"
                            "{{- '<|IMAGE_END|>' -}}"
                        "{%- endif -%}"
                    "{%- endfor -%}"

                    # Pass 2: Render all text second
                    "{%- for content in message['content'] -%}"
                        "{%- if content['type'] == 'text' -%}"
                            "{{- content['text'] -}}"
                        "{%- endif -%}"
                    "{%- endfor -%}"
                "{%- endif -%}"
                "{{- '\\n' -}}"

            "{%- elif message['role'] == 'assistant' -%}"
                "{{- 'Assistant:\\n' -}}"
                "{%- if message['content'] is string -%}"
                    "{{- message['content'] -}}"
                "{%- else -%}"
                    "{%- for content in message['content'] -%}"
                        "{%- if content['type'] == 'text' -%}"
                            "{{- content['text'] -}}"
                        "{%- endif -%}"
                    "{%- endfor -%}"
                "{%- endif -%}"
                "{{- eos_token -}}"

            "{%- elif message['role'] == 'system' -%}"
                "{%- if message['content'] is string -%}"
                    "{{- message['content'] + '\\n' -}}"
                "{%- else -%}"
                    "{%- for content in message['content'] -%}"
                        "{%- if content['type'] == 'text' -%}"
                            "{{- content['text'] + '\\n' -}}"
                        "{%- endif -%}"
                    "{%- endfor -%}"
                "{%- endif -%}"
            "{%- endif -%}"
        "{%- endfor -%}"

        "{%- if add_generation_prompt -%}"
            "{{- 'Assistant:\\n' -}}"
        "{%- endif -%}"
    )

    def __init__(
        self,
        image_min_tokens: int = -1,
        image_max_tokens: int = -1,
        **kwargs
    ):
        self.image_min_tokens = image_min_tokens
        self.image_max_tokens = image_max_tokens
        super().__init__(
            image_min_tokens=self.image_min_tokens,
            image_max_tokens=self.image_max_tokens,
            **kwargs
        )

    def __call__(self, **kwargs):
        # Set the specific stop token defined in the PaddleOCR template
        kwargs['stop'] = [self.PADDLEOCR_EOS_TOKEN]

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix} - Start processing")

        return super().__call__(**kwargs)


class Qwen25VLChatHandler(MTMDChatHandler):

    QWEN25_VL_BOS_TOKEN = "<|endoftext|>"
    QWEN25_VL_PAD_TOKEN = "<|endoftext|>"
    QWEN25_VL_EOS_TOKEN = "<|im_end|>"

    CHAT_FORMAT = (
        "{% set image_count = namespace(value=0) %}"
        "{% for message in messages %}"
        "{% if loop.first and message['role'] != 'system' %}"
        "<|im_start|>system\n"
        "{{ self.DEFAULT_SYSTEM_MESSAGE }}<|im_end|>\n"
        "{% endif %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}<|im_end|>\n"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{% set image_count.value = image_count.value + 1 %}"
        "Picture {{ image_count.value }}: <|vision_start|> {{ content.image_url }} <|vision_end|>"
        "{% else %}"
        "{% set image_count.value = image_count.value + 1 %}"
        "Picture {{ image_count.value }}: <|vision_start|> {{ content.image_url.url }} <|vision_end|>"
        "{% endif %}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "<|im_start|>assistant\n"
    )

    def __call__(self, **kwargs):
        kwargs['stop'] = [self.QWEN25_VL_EOS_TOKEN, self.QWEN25_VL_PAD_TOKEN]

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix} - Start processing")

        # Use parent implementation
        return super().__call__(**kwargs)

class Qwen3ASRChatHandler(MTMDChatHandler):
    """
    Handler for Qwen 3 ASR (Automatic Speech Recognition) models.

    Features:
    - Highly specialized for Speech-to-Text tasks.
    - Aggregates all system text into a single cohesive system block.
    - Drops user text entirely, extracting ONLY audio data into a unified user turn.
    - Wraps audio with <|audio_start|><|audio_pad|>[DATA]<|audio_end|>.
    - Integrated MTMD-style URL and Base64 injection for input_audio and audio_url.
    """

    DEFAULT_SYSTEM_MESSAGE = """
    You are an advanced multilingual Speech-to-Text model. Accurately transcribe the audio into text in its original spoken language.
    You should ignore background noise, filler words, and stutters where possible, and format the final output with correct grammar and capitalization.
    """

    QWEN3_ASR_BOS_TOKEN = "<|im_start|>"
    QWEN3_ASR_PAD_TOKEN = "<|endoftext|>"
    QWEN3_ASR_EOS_TOKEN = "<|im_end|>"


    QWEN3_ASR_AUDIO_BOS_TOKEN = "<|audio_start|>"
    QWEN3_ASR_AUDIO_PAD_TOKEN = "<|audio_pad|>"
    QWEN3_ASR_AUDIO_EOS_TOKEN = "<|audio_end|>"

    CHAT_FORMAT = (
        "{%- set ns = namespace(system_text='') -%}\n"
        "{%- for m in messages -%}\n"
        "    {%- if m.role == 'system' -%}\n"
        "        {%- if m.content is string -%}\n"
        "            {%- set ns.system_text = ns.system_text + m.content -%}\n"
        "        {%- else -%}\n"
        "            {%- for c in m.content -%}\n"
        "                {%- if c.type == 'text' and (c.text is defined) -%}\n"
        "                    {%- set ns.system_text = ns.system_text + c.text -%}\n"
        "                {%- endif -%}\n"
        "            {%- endfor -%}\n"
        "        {%- endif -%}\n"
        "    {%- endif -%}\n"
        "{%- endfor -%}\n"
        "\n"
        "{%- set ns2 = namespace(audio_tokens='') -%}\n"
        "{%- for m in messages -%}\n"
        "    {%- if m.content is not string -%}\n"
        "        {%- for c in m.content -%}\n"
        "            {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) or c.type == 'input_audio' -%}\n"
        "                {#- MTMD Audio Injection -#}\n"
        "                {%- set audio_val = '' -%}\n"
        "                {%- if c.type == 'audio_url' or 'audio_url' in c -%}\n"
        "                    {%- set audio_val = c.audio_url if c.audio_url is string else c.audio_url.url -%}\n"
        "                {%- elif c.type == 'input_audio' or 'input_audio' in c -%}\n"
        "                    {%- set audio_val = c.input_audio if c.input_audio is string else ('data:audio/' + c.input_audio.format + ';base64,' + c.input_audio.data) -%}\n"
        "                {%- endif -%}\n"
        "                {%- set ns2.audio_tokens = ns2.audio_tokens + '<|audio_start|><|audio_pad|>' + audio_val + '<|audio_end|>' -%}\n"
        "            {%- endif -%}\n"
        "        {%- endfor -%}\n"
        "    {%- endif -%}\n"
        "{%- endfor -%}\n"
        "\n"
        "{{- '<|im_start|>system\\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\\n' -}}\n"
        "{{- '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' -}}\n"
        "{%- if add_generation_prompt -%}\n"
        "    {{- '<|im_start|>assistant\\n' -}}\n"
        "{%- endif -%}\n"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        # Qwen3 models universally use `<|endoftext|>` and `<|im_end|>` as the stop token
        kwargs['stop'] = [self.QWEN3_ASR_AUDIO_PAD_TOKEN, self.QWEN3_ASR_AUDIO_EOS_TOKEN]

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix} - Start processing Qwen3-ASR (Audio Only)")

        return super().__call__(**kwargs)

class Qwen3VLChatHandler(MTMDChatHandler):

    QWEN3_VL_BOS_TOKEN = "<|endoftext|>"
    QWEN3_VL_PAD_TOKEN = "<|endoftext|>"
    QWEN3_VL_EOS_TOKEN = "<|im_end|>"

    CHAT_FORMAT = (
        "{{- '<|im_start|>system\n' -}}"
        "{%- if messages[0].content is string and messages[0].role == 'system' -%}"
            "{{- messages[0].content -}}"
        "{%- elif messages[0].role == 'system' -%}"
            "{%- if 'text' in messages[0].content -%}"
                "{{- messages[0].content.text -}}"
            "{%- else -%}"
                "{{- 'You are a helpful assistant.' -}}"
            "{%- endif -%}"
        "{%- endif -%}"
        "{%- if tools -%}"
            "{{- '\n\n' -}}"
            "{{- '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' -}}"
            "{%- for tool in tools -%}"
                "{{- '\n' -}}"
                "{{- tool | tojson -}}"
            "{%- endfor -%}"
            "{{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <arguments-json-object>}\n</tool_call>\n\nYou can also return a response for the user alongside a function call:\nRESPONSE FOR THE USER HERE\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <arguments-json-object>}\n</tool_call>' -}}"
        "{%- endif -%}"
        "{{- '<|im_end|>\n' -}}"
        "{%- set image_count = namespace(value=0) -%}"
        #"{%- set video_count = namespace(value=0) -%}"
        "{%- for message in messages -%}"
            "{%- if message.role == 'tool' -%}"
                "{{- '<|im_start|>user\n<tool_response>\n' -}}"
            "{%- elif message.role != 'system' -%}"
                "{{- '<|im_start|>' + message.role + '\n' -}}"
            "{%- endif -%}"
            "{%- if message.content is string and message.role != 'system' -%}"
                "{{- message.content -}}"
            "{%- elif message.role != 'system' -%}"
                "{%- for content in message.content -%}"
                    "{%- if 'image_url' in content -%}"
                        "{%- set image_count.value = image_count.value + 1 -%}"
                        "{%- if add_vision_id -%}"
                            "{{- 'Picture ' -}}"
                            "{{- image_count.value | string -}}"
                            "{{- ': ' -}}"
                        "{%- endif -%}"
                        "{{- '<|vision_start|>' -}}"
                        "{%- if content.image_url is string -%}"
                            "{{- content.image_url -}}"
                        "{%- else -%}"
                            "{{- content.image_url.url -}}"
                        "{%- endif -%}"
                        "{{- '<|vision_end|>' -}}"
                    "{%- endif -%}"
                    # Video not supported yet
                    "{%- if 'text' in content -%}"
                        "{{- content.text -}}"
                    "{%- endif -%}"
                "{%- endfor -%}"
            "{%- endif -%}"
            "{%- if message.role == 'assistant' -%}"
                "{%- if message.tool_calls -%}"
                    "{%- for tool_call in message.tool_calls -%}"
                        "{%- if (loop.first and message.content) or (not loop.first) -%}"
                            "{{- '\n' -}}"
                        "{%- endif -%}"
                        "{%- if tool_call.function -%}"
                            "{%- set tool_call = tool_call.function -%}"
                        "{%- endif -%}"
                        "{{- '<tool_call>\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' -}}"
                        "{%- if tool_call.arguments is string -%}"
                            "{{- tool_call.arguments -}}"
                        "{%- else -%}"
                            "{{- tool_call.arguments | tojson -}}"
                        "{%- endif -%}"
                        "{{- '}\n</tool_call>' -}}"
                    "{%- endfor -%}"
                "{%- endif -%}"
            "{%- elif message.role == 'tool' -%}"
                "{{- '</tool_response>' -}}"
            "{%- endif -%}"
            "{%- if message.role != 'system' -%}"
                "{{- '<|im_end|>\n' -}}"
            "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
            "{{- '<|im_start|>assistant\n' -}}"
            "{%- if force_reasoning -%}"
                "{{- '<think>\n' -}}"
            "{%- endif -%}"
        "{%- endif -%}"
    )

    def __init__(
        self,
        force_reasoning: bool = False,
        add_vision_id: bool = True,
        **kwargs,
    ):
        """
        Parameters:
        - force_reasoning (bool):
            - True: Force the reasoning in the model by adding <think> to the chat template.
            - False (default): Don't force the reasoning.
        - add_vision_id (bool):
            - True (default): Count all the images. Recommended for multi-image.
            - False: Doesn't count the images. Can save tokens with single-image.
        """
        super().__init__(**kwargs)
        self.force_reasoning = force_reasoning
        self.extra_template_arguments["force_reasoning"] = force_reasoning
        self.extra_template_arguments["add_vision_id"] = add_vision_id

    def __call__(self, **kwargs):
        kwargs['stop'] = [self.QWEN3_VL_EOS_TOKEN, self.QWEN3_VL_PAD_TOKEN]

        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix}(force_reasoning={self.force_reasoning}) - Start processing")

        # Use parent implementation
        return super().__call__(**kwargs)

class Qwen35ChatHandler(MTMDChatHandler):
    """
    Handler for Qwen3.5/Qwen3.6 models.
    """
    CHAT_FORMAT = (
        "{%- set image_count = namespace(value=0) -%}"
        "{%- set video_count = namespace(value=0) -%}"
        "{%- macro render_content(content, do_vision_count, is_system_content=false) -%}"
        "    {%- if content is string -%}"
        "        {{- content -}}"
        "    {%- elif content is iterable and content is not mapping -%}"
        "        {%- for item in content -%}"
        "            {%- if 'image_url' in item or item.type == 'image_url' -%}"
        "                {%- if is_system_content -%}"
        "                    {{- raise_exception('System message cannot contain images.') -}}"
        "                {%- endif -%}"
        "                {%- if do_vision_count -%}"
        "                    {%- set image_count.value = image_count.value + 1 -%}"
        "                {%- endif -%}"
        "                {%- if add_vision_id -%}"
        "                    {{- 'Picture ' -}}"
        "                    {{- image_count.value | string -}}"
        "                    {{- ': ' -}}"
        "                {%- endif -%}"
        "                {{- '<|vision_start|>' -}}"
        "                {%- if item.image_url is string -%}"
        "                    {{- item.image_url -}}"
        "                {%- else -%}"
        "                    {{- item.image_url.url -}}"
        "                {%- endif -%}"
        "                {{- '<|vision_end|>' -}}"
        "            {%- elif 'video' in item -%}"
        "                {{- raise_exception('llama.cpp does not currently support video.') -}}"  # Video not supported, raise exception
        "                {%- if is_system_content -%}"
        "                    {{- raise_exception('System message cannot contain videos.') -}}"
        "                {%- endif -%}"
        "                {%- if do_vision_count -%}"
        "                    {%- set video_count.value = video_count.value + 1 -%}"
        "                {%- endif -%}"
        "                {%- if add_vision_id -%}"
        "                    {{- 'Video ' ~ video_count.value ~ ': ' -}}"
        "                {%- endif -%}"
        "                {{- '<|vision_start|>' -}}"
        "                {{- item.video -}}"
        "                {{- '<|vision_end|>' -}}"
        "            {%- elif 'text' in item -%}"
        "                {{- item.text -}}"
        "            {%- else -%}"
        "                {{- raise_exception('Unexpected item type in content.') -}}"
        "            {%- endif -%}"
        "        {%- endfor -%}"
        "    {%- elif content is none or content is undefined -%}"
        "        {{- '' -}}"
        "    {%- else -%}"
        "        {{- raise_exception('Unexpected content type.') -}}"
        "    {%- endif -%}"
        "{%- endmacro -%}"
        "{%- if not messages -%}"
        "    {{- raise_exception('No messages provided.') -}}"
        "{%- endif -%}"
        "{%- if tools and tools is iterable and tools is not mapping -%}"
        "    {{- '<|im_start|>system\n' -}}"
        "    {{- '# Tools\n\nYou have access to the following functions:\n\n<tools>' -}}"
        "    {%- for tool in tools -%}"
        "        {{- '\n' -}}"
        "        {{- tool | tojson -}}"
        "    {%- endfor -%}"
        "    {{- '\n</tools>' -}}"
        "    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' -}}"
        "    {%- if messages[0].role == 'system' -%}"
        "        {%- set content = render_content(messages[0].content, false, true) | trim -%}"
        "        {%- if content -%}"
        "            {{- '\n\n' + content -}}"
        "        {%- endif -%}"
        "    {%- endif -%}"
        "    {{- '<|im_end|>\n' -}}"
        "{%- elif messages[0].role == 'system' -%}"
        "    {%- set content = render_content(messages[0].content, false, true) -%}"
        "    {{- '<|im_start|>system\n' + content + '<|im_end|>\n' -}}"
        "{%- endif -%}"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages | length - 1) -%}"
        "{%- for message in messages[::-1] -%}"
        "    {%- set index = messages | length - 1 - loop.index0 -%}"
        "    {%- if ns.multi_step_tool and message.role == 'user' -%}"
        "        {%- set content = render_content(message.content, false) | trim -%}"
        "        {%- if not (content.startswith('<tool_response>') and content.endswith('</tool_response>')) -%}"
        "            {%- set ns.multi_step_tool = false -%}"
        "            {%- set ns.last_query_index = index -%}"
        "        {%- endif -%}"
        "    {%- endif -%}"
        "{%- endfor -%}"
        "{%- if ns.multi_step_tool -%}"
        "    {{- raise_exception('No user query found in messages.') -}}"
        "{%- endif -%}"
        "{%- for message in messages -%}"
        "    {%- set content = render_content(message.content, true) | trim -%}"
        "    {%- if message.role == 'system' -%}"
        "        {%- if not loop.first -%}"
        "            {{- raise_exception('System message must be at the beginning.') -}}"
        "        {%- endif -%}"
        "    {%- elif message.role == 'user' -%}"
        "        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' -}}"
        "    {%- elif message.role == 'assistant' -%}"
        "        {%- set reasoning_content = '' -%}"
        "        {%- if message.reasoning_content is string -%}"
        "            {%- set reasoning_content = message.reasoning_content -%}"
        "        {%- elif '</think>' in content -%}"
        "            {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}"
        "            {%- set content = content.split('</think>')[-1].lstrip('\n') -%}"
        "        {%- endif -%}"
        "        {%- set reasoning_content = reasoning_content | trim -%}"
        "        {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) -%}"
        "            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content -}}"
        "        {%- else -%}"
        "            {{- '<|im_start|>' + message.role + '\n' + content -}}"
        "        {%- endif -%}"
        "        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping -%}"
        "            {%- for tool_call in message.tool_calls -%}"
        "                {%- if tool_call.function is defined -%}"
        "                    {%- set tool_call = tool_call.function -%}"
        "                {%- endif -%}"
        "                {%- if loop.first -%}"
        "                    {%- if content | trim -%}"
        "                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' -}}"
        "                    {%- else -%}"
        "                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' -}}"
        "                    {%- endif -%}"
        "                {%- else -%}"
        "                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' -}}"
        "                {%- endif -%}"
        "                {%- if tool_call.arguments is defined -%}"
        "                    {%- for (args_name, args_value) in tool_call.arguments | items -%}"
        "                        {{- '<parameter=' + args_name + '>\n' -}}"
        "                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}"
        "                        {{- args_value -}}"
        "                        {{- '\n</parameter>' -}}"
        "                    {%- endfor -%}"
        "                {%- endif -%}"
        "                {{- '</function>\n</tool_call>' -}}"
        "            {%- endfor -%}"
        "        {%- endif -%}"
        "        {{- '<|im_end|>\n' -}}"
        "    {%- elif message.role == 'tool' -%}"
        "        {%- if loop.previtem and loop.previtem.role != 'tool' -%}"
        "            {{- '<|im_start|>user' -}}"
        "        {%- endif -%}"
        "        {{- '\n<tool_response>\n' -}}"
        "        {{- content -}}"
        "        {{- '\n</tool_response>' -}}"
        "        {%- if not loop.last and loop.nextitem.role != 'tool' -%}"
        "            {{- '<|im_end|>\n' -}}"
        "        {%- elif loop.last -%}"
        "            {{- '<|im_end|>\n' -}}"
        "        {%- endif -%}"
        "    {%- else -%}"
        "        {{- raise_exception('Unexpected message role.') -}}"
        "    {%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "    {{- '<|im_start|>assistant\n' -}}"
        "    {%- if enable_thinking is defined and enable_thinking is false -%}"
        "        {{- '<think>\n\n</think>\n\n' -}}"
        "    {%- else -%}"
        "        {{- '<think>\n' -}}"
        "    {%- endif -%}"
        "{%- endif -%}"
    )

    def __init__(
        self,
        add_vision_id: bool = True,
        enable_thinking: bool = True,
        preserve_thinking: bool = False,
        **kwargs,
    ):
        """
        Parameters:
        - add_vision_id (bool):
            - True (default): Count all the images. Recommended for multi-image.
            - False: Doesn't count the images. Can save tokens with single-image.
        - enable_thinking (bool):
            - True (default): Enables reasoning for better results.
            - False: Disables reasoning for faster results.
        - preserve_thinking (bool):
            - True: Keeps <think> reasoning process for ALL historical conversational turns.
            - False (default): Only keeps <think> for the latest assistant reply to save tokens.
        """
        super().__init__(**kwargs)
        self.enable_thinking = enable_thinking
        self.preserve_thinking = preserve_thinking
        self.extra_template_arguments["add_vision_id"] = add_vision_id
        self.extra_template_arguments["enable_thinking"] = enable_thinking
        self.extra_template_arguments["preserve_thinking"] = preserve_thinking

    def __call__(self, **kwargs):
        llama = kwargs['llama']

        if hasattr(llama, 'input_ids'):
            llama.input_ids.fill(0)

        if self.verbose:
            print(f"{self.log_prefix}(enable_thinking={self.enable_thinking}, preserve_thinking={self.preserve_thinking}) - Start processing")

        # Use parent implementation
        return super().__call__(**kwargs)


class Step3VLChatHandler(MTMDChatHandler):
    """
    Handler for Step3-VL models.
    """

    STEP3VL_BOS_TOKEN = "<|im_start|>"
    STEP3VL_EOS_TOKEN = "<|im_end|>"
    STEP3VL_PAD_TOKEN = "<|endoftext|>"
    STEP3VL_IMAGE_TOKEN = "<im_patch>"

    CHAT_FORMAT = (
        "{%- macro render_content(content) -%}\n"
        "    {%- if content is none -%}{{- '' -}}\n"
        "    {%- elif content is string -%}{{- content -}}\n"
        "    {%- elif content is mapping -%}{{- content['value'] if 'value' in content else content['text'] -}}\n"
        "    {%- elif content is iterable -%}\n"
        "        {%- for item in content -%}\n"
        "            {%- if item.type == 'text' -%}\n"
        "                {{- item['value'] if 'value' in item else item['text'] -}}\n"
        "            {%- elif item.type in ['image', 'image_url'] -%}\n"
        "                {%- set url_val = '' -%}\n"
        "                {%- if item.image_url -%}\n"
        "                    {%- set url_val = item.image_url if item.image_url is string else item.image_url.url -%}\n"
        "                {%- endif -%}\n"
        "                {{- '<im_patch>' + url_val -}}\n"
        "            {%- endif -%}\n"
        "        {%- endfor -%}\n"
        "    {%- endif -%}\n"
        "{%- endmacro -%}\n"
        "\n"
        "{%- if tools -%}\n"
        "    {{- '<|im_start|>system\\n' -}}\n"
        "    {%- if messages[0].role == 'system' -%}\n"
        "        {{- render_content(messages[0].content) + '\\n\\n' -}}\n"
        "    {%- endif -%}\n"
        "    {{- '# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>' -}}\n"
        "    {%- for tool in tools -%}\n"
        "        {{- '\\n' -}}\n"
        "        {{- tool | tojson -}}\n"
        "    {%- endfor -%}\n"
        "    {{- '\\n</tools>\\n\\nAlways adhere to this exact format for tool use:\\n<tool_calls>\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call>\\n{additional_tool_calls}</tool_calls>\\n\\nNote:\\n- For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.\\n- `<function-name>` must be an exact match to one of the available tools.\\n- `<args-json-object>` must be valid JSON that strictly follows the tool\\'s parameters schema.<|im_end|>\\n' -}}\n"
        "{%- else -%}\n"
        "    {%- if messages[0].role == 'system' -%}\n"
        "        {{- '<|im_start|>system\\n' + render_content(messages[0].content) + '<|im_end|>\\n' -}}\n"
        "    {%- endif -%}\n"
        "{%- endif -%}\n"
        "\n"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) -%}\n"
        "{%- for message in messages[::-1] -%}\n"
        "    {%- set index = (messages|length - 1) - loop.index0 -%}\n"
        "    {%- if ns.multi_step_tool and message.role == 'user' and render_content(message.content) is string and not(render_content(message.content).startswith('<tool_response>') and render_content(message.content).endswith('</tool_response>')) -%}\n"
        "        {%- set ns.multi_step_tool = false -%}\n"
        "        {%- set ns.last_query_index = index -%}\n"
        "    {%- endif -%}\n"
        "{%- endfor -%}\n"
        "\n"
        "{%- for message in messages -%}\n"
        "    {%- set content = render_content(message.content) -%}\n"
        "    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) -%}\n"
        "        {%- set role_name = 'observation' if (message.role == 'system' and not loop.first and message.name == 'observation') else message.role -%}\n"
        "        {{- '<|im_start|>' + role_name + '\\n' + content + '<|im_end|>' + '\\n' -}}\n"
        "    {%- elif message.role == 'assistant' -%}\n"
        "        {%- if message.reasoning_content is string -%}\n"
        "            {%- set reasoning_content = render_content(message.reasoning_content) -%}\n"
        "        {%- else -%}\n"
        "            {%- if '</think>' in content -%}\n"
        "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') -%}\n"
        "                {%- set content = content.split('</think>')[-1].lstrip('\\n') -%}\n"
        "            {%- else -%}\n"
        "                {%- set reasoning_content = '' -%}\n"
        "            {%- endif -%}\n"
        "        {%- endif -%}\n"
        "        {%- if loop.index0 > ns.last_query_index -%}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n' + content -}}\n"
        "        {%- else -%}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content -}}\n"
        "        {%- endif -%}\n"
        "        {%- if message.tool_calls -%}\n"
        "            {{- '\\n<tool_calls>' -}}\n"
        "            {%- for tool_call in message.tool_calls -%}\n"
        "                {{- '\\n' -}}\n"
        "                {%- if tool_call.function -%}\n"
        "                    {%- set tool_call = tool_call.function -%}\n"
        "                {%- endif -%}\n"
        "                {{- '<tool_call>\\n{\"name\": \"' -}}\n"
        "                {{- tool_call.name -}}\n"
        "                {{- '\", \"arguments\": ' -}}\n"
        "                {%- if tool_call.arguments is string -%}\n"
        "                    {{- tool_call.arguments -}}\n"
        "                {%- else -%}\n"
        "                    {{- tool_call.arguments | tojson -}}\n"
        "                {%- endif -%}\n"
        "                {{- '}\\n</tool_call>' -}}\n"
        "            {%- endfor -%}\n"
        "            {{- '\\n</tool_calls>' -}}\n"
        "        {%- endif -%}\n"
        "        {{- '<|im_end|>\\n' -}}\n"
        "    {%- elif message.role == 'tool' -%}\n"
        "        {%- if loop.first or (messages[loop.index0 - 1].role != 'tool') -%}\n"
        "            {{- '<|im_start|>tool_response' -}}\n"
        "        {%- endif -%}\n"
        "        {{- '\\n<tool_response>\\n' -}}\n"
        "        {{- content -}}\n"
        "        {{- '\\n</tool_response>' -}}\n"
        "        {%- if loop.last or (messages[loop.index0 + 1].role != 'tool') -%}\n"
        "            {{- '<|im_end|>\\n' -}}\n"
        "        {%- endif -%}\n"
        "    {%- endif -%}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n' if (enable_thinking is defined and not enable_thinking) else '<|im_start|>assistant\\n<think>' -}}\n"
        "{%- endif -%}\n"
    )

    def __init__(self, enable_thinking: bool = True, **kwargs):
        """
        Initializes the Step3-VL Handler.

        Args:
            enable_thinking (bool): If False, injects an empty <think> block to bypass reasoning.
        """
        self.enable_thinking = enable_thinking
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        # Pass thinking toggle into Jinja
        self.extra_template_arguments["enable_thinking"] = self.enable_thinking

        # Step3 uses standard <|im_end|> ChatML stop formatting
        kwargs['stop'] = [self.STEP3VL_PAD_TOKEN, self.STEP3VL_EOS_TOKEN]

        if self.verbose:
            print(f"{self.log_prefix}(enable_thinking={self.enable_thinking}) - Start processing")

        return super().__call__(**kwargs)
