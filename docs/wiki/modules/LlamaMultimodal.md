---
title: Llama Multimodal
module_name: llama_cpp.llama_multimodal
source_file: llama_cpp/llama_multimodal.py
last_updated: 2026-11-21
version_target: "latest"
---

# Llama Multimodal

Use a matching backbone GGUF and mmproj GGUF to chat about images, audio or
video. Choose a handler supported by the model, pass it as `chat_handler`, and
submit media through `create_chat_completion`. Native libraries must match the
checked-out vendor API.

## MTMDBaseHandler

`MTMDBaseHandler` is the shared projector/media layer behind the chat, embedding
and TTS handlers. You can instantiate it directly when an application needs
projected media features rather than a chat response. It handles:

- Lazy projector initialization and capability queries on the first model-bound operation.
- Shared GPU, Flash Attention, resolution, batching and video configuration.
- Media loading/decoding, scoped native chunks, and ordered batch encoding.
- Native resource cleanup and custom decoder callback ownership.

The shared layer is inherited by the higher-level interfaces:

```text
MTMDBaseHandler
├── MTMDChatHandler
│   ├── GenericMTMDChatHandler
│   └── Model-specific chat handlers
├── MTMDEmbeddingHandler
└── MTMDAudioGenerator
```

The base encoding methods return features; they do not generate text or populate
the backbone's KV cache.

### Shared configuration

Media support depends on the projector and loaded native library. A handler
cannot add a modality that the model does not implement.

| Setting/input | Meaning |
| --- | --- |
| `mmproj_path` | Matching multimodal projector GGUF. |
| `use_gpu`, `flash_attn` | Projector execution; `flash_attn=None/True/False` selects auto/on/off independently of backbone settings. |
| `image_min_tokens`, `image_max_tokens` | Resolution/token limits for projectors that support them. |
| `batch_max_tokens` | MTMD batch token budget. |
| `video_fps_target` | Requested frame rate; a model's recommended rate can take precedence. |
| `video_timestamp_interval_ms` | Timestamp interval; nonpositive disables interval timestamps. Model-specific video layouts may still emit their own timestamps. |
| `video_no_soundtrack` | Suppress video audio where supported. |
| `video_ffmpeg_bin_dir` | Directory containing ffmpeg and ffprobe; otherwise use PATH. |
| `video_decoder` | Application-defined decoder table; see [custom video decoders](#custom-video-decoders). |

For ordinary videos, include `{"type": "video", "video": "/path/to/video.mp4"}`
in message content and install ffmpeg/ffprobe with an MTMD_VIDEO-enabled library.
A custom decoder can provide video frames without that built-in backend.
The current chat path loads the video into bytes before passing it to MTMD.
Native file-path reading is available through the low-level API.

Base, chat, embedding and TTS handlers bind to a specific `Llama` instance
when projector initialization succeeds. Reuse that pair across requests.
Passing another `Llama` instance raises `ValueError`, even if it loaded the
same GGUF. Create a separate handler for each model instance.

Keep the pair open until the response stream finishes. Use one pair per
concurrent generation and close a manually created handler before its model.

### Tokenize and encode

| Method | Use |
| --- | --- |
| `tokenize(model, prompt, media=...)` | Encode a prompt containing one runtime media marker per image/audio payload. Media must be encoded file bytes. |
| `tokenize_parts(model, parts)` | Supply ordered text strings and image/audio file bytes without writing media markers. |
| `request.encode()` / `handler.encode(request)` | Encode the live request and return a tuple of owned `MTMDEncodedChunk` results. |
| `from_pretrained(repo_id, filename, **kwargs)` | Download the projector and construct the handler; load the backbone separately. |
| `close()` / context exit | Release dependent native resources and the projector; repeated close is safe. |

Both tokenization methods are context managers and default to `add_special=True`
and `parse_special=True`. Call `encode` inside the request scope. Use
`tokenize_parts` for custom feature consumers when the model's runtime media
marker is not yet known; a model-bound handler exposes it as `handler.media_marker`.

This example requires matching vision backbone/mmproj files and an image:

```python
from contextlib import closing
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_multimodal import MTMDBaseHandler

image = Path("/path/to/image.jpg").read_bytes()
with closing(Llama(
    model_path="/path/to/model/backbone.gguf", n_gpu_layers=-1,
)) as llm, MTMDBaseHandler(
    mmproj_path="/path/to/model/mmproj.gguf", use_gpu=True, image_max_tokens=512,
) as handler:
    with handler.tokenize_parts(llm, ["Describe this image: ", image]) as request:
        print("Input rows:", request.n_tokens, "Position span:", request.n_pos)
        encoded = request.encode()

    # Native request storage is gone; the copied results remain usable.
    for chunk in encoded:
        if chunk.embeddings is not None:
            print("Media features:", chunk.embeddings.shape, chunk.embeddings.dtype)
        else:
            print("Text tokens:", chunk.tokens)
        print("Positions:", chunk.positions, "Position span:", chunk.n_pos)
```

`n_tokens` counts input rows; `n_pos` counts decoder positions. They can differ
for M-RoPE, so do not substitute one for the other. Each result carries its
`kind`, text `tokens` or a copied read-only float32 `embeddings` matrix,
`positions`, `n_pos`, and `use_non_causal` attention metadata. Positions start
at zero for each encoding request; a custom backbone consumer must map them to
its own sequence position. Results remain usable after the request and handler
close, but the request itself cannot be encoded after its scope exits.

Requests hold the handler lock until their scope exits. Close the handler
after leaving that scope and returning from any decoder callbacks. Base
tokenization accepts images and audio. Chat handlers also expand video.
For pooled vectors, use [the embedding API](LlamaEmbedding.md#shared-mtmd-encoding-interfaces).

## MTMDChatHandler and model-specific handlers

`MTMDChatHandler` builds on the base layer with chat-template rendering, media
expansion, backbone evaluation and response generation. Model-specific handlers
inherit this pipeline and supply their model's template and generation options.
Use a dedicated handler when those options are needed; for example,
`Qwen35ChatHandler`, `Qwen3VLChatHandler` or `Gemma4ChatHandler`.
See [Qwen image chat](../examples/vision/vision-qwen.md) for a dedicated example.

## GenericMTMDChatHandler

Use this handler when the model's Jinja template describes its chat/media
format. The backbone and projector must already be supported by native MTMD;
recognizing a placeholder does not add a new model architecture.

### Image chat

This example requires a vision-capable model whose GGUF includes a compatible
chat template, plus an accessible image. Replace the paths before running.

```python
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_multimodal import GenericMTMDChatHandler

handler = GenericMTMDChatHandler(
    chat_format=None,
    mmproj_path="/path/to/model/mmproj.gguf",
    image_max_tokens=512,
)
llm = Llama(
    model_path="/path/to/model/backbone.gguf",
    chat_handler=handler,
    n_ctx=8192, n_batch=2048, n_ubatch=1024, n_gpu_layers=-1,
)
try:
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": Path("/path/to/image.jpg").resolve().as_uri()}},
            {"type": "text", "text": "Describe this image briefly."},
        ]}],
        max_tokens=512, temperature=0,
    )
    print(result["choices"][0]["message"]["content"])
finally:
    handler.close()
    llm.close()
```

### Template selection and media injection

| Option | Behavior |
| --- | --- |
| `chat_format=None` | Use the model's default template. Pass this explicitly: the constructor requires the argument. |
| `chat_template_name="name"` | Select a named model template; if unavailable, try the default. |
| `chat_format=template_text` | Use this Jinja source directly; it takes precedence over model templates. Read a file with `Path(...).read_text(encoding="utf-8")`. |
| `extra_template_arguments={...}` | Supply template-specific flags; their meaning depends on that template. |

If the model has no usable stored template, the handler falls back to the
built-in MTMD chat template. Supply the correct template when the fallback does
not match the model. Templates are resolved/analyzed once per handler.

The handler extracts media in message order, renders the selected template,
replaces recognized tags and rendered media URLs with the loaded context's
`handler.media_marker`, and validates one marker per media item. It then lets
MTMD expand images, audio and video before backbone evaluation. Common tags
include `<|image_pad|>`, `<|image|>`, `<image>`, `[IMG]`, `<|patch|>` and
`<|kimi_image_placeholder|>`. The full list is `KNOWN_MEDIA_TAGS` in
[`llama_multimodal.py`](../../../llama_cpp/llama_multimodal.py).

Custom chat handlers can override `_get_media_items(messages)`. The chat path
calls it once with the full message history, including any default system
message, and uses the returned media descriptors in prompt order.

For generic Qwen3-ASR templates that take string content, the override
can filter the descriptors from `super()` and rewrite a URL with
`{**item, "url": new_url}`. This keeps the marker at the original audio part,
even when two clips share a URL. If the override builds fresh descriptors and
both filters and renames clips, include `"_mtmd_source": (message_index, part_index)`
using zero-based indices into the messages received by the hook.

### Adapting an unrecognized placeholder

For a template that emits one independent tag per image, extend the list in a
small subclass. For example, if your template literally contains
`<|custom_image|>`, define:

```python
from llama_cpp.llama_multimodal import GenericMTMDChatHandler


class CustomMediaChatHandler(GenericMTMDChatHandler):
    KNOWN_MEDIA_TAGS = [
        *GenericMTMDChatHandler.KNOWN_MEDIA_TAGS,
        "<|custom_image|>",
    ]
```

Use `CustomMediaChatHandler` in place of `GenericMTMDChatHandler` in the image
chat example, with the GGUF template or explicit `chat_format` that emits that
tag. The custom tag is illustrative; Kimi K3.1's actual image placeholder is
already recognized. Define the subclass before construction; do not modify the
shared class list or a live handler's cached parser configuration.

Tags are selected by their literal presence in the template source. If Jinja
constructs a tag dynamically, rewrite it as a literal or add model-specific
normalization in a subclass. A tag repeated for every patch, or a numbered tag
such as `<|image_1|>`, needs a scoped normalization rule rather than an unbounded
list of tag names. Preserve media order and produce exactly one marker per
source media item, not per output patch/frame.

Check boundary ownership before adapting a template. MTMD can insert image
start/end tokens, sizes and tile markers itself. Adding their names to the
placeholder list would turn boundaries into extra media inputs. If the template
already wraps its placeholder in boundaries supplied by MTMD, normalize only
the complete media wrapper in a model-specific `_replace_media_placeholders`
override. Preserve surrounding text, and use the superclass validation of marker
counts. Generic handling already removes the recognized Qwen/LFM3 vision wrapper;
it does not remove arbitrary wrappers for every architecture.

### Diagnose `media marker mismatch`

Enable `verbose=True` to inspect the rendered prompt and compare the error's
`marker_count` and `media_count`. Verbose logs can contain message/media data.

| Symptom | Check |
| --- | --- |
| Too few markers | Unknown or dynamically constructed tag; template skipped a media content type. |
| Too many markers | Template emitted both a tag and URL, or expanded one input into repeated patch tags. |
| Counts match, but visual input behaves incorrectly | Duplicate boundaries, wrong template/projector pairing, or unsupported model-specific behavior. |

Before using the adapter, check single media, multiple media, surrounding text,
and repeated turns. Counts alone do not prove that boundary placement is correct.
If the model needs special stop tokens or generation behavior, prefer a dedicated
handler instead of expanding the generic replacement rules.

## Common multimodal examples: Gemma4

These plain Python examples cover file/byte inputs, multiple attachments,
conversation history and streaming. Replace the paths with matching Gemma4
backbone/mmproj GGUF files and your media. Run the loading block once, use the
request snippets with that open `llm`, and call `llm.close()` when finished.
No UI framework is required.

### Load once

Passing `mmproj_path` to `Llama` creates the generic MTMD handler automatically.
Use `chat_handler_kwargs` for its shared options; do not also pass a separate
`chat_handler` when using this shortcut.

```python
from llama_cpp import Llama

llm = Llama(
    model_path="/path/to/model/gemma4-e4b.gguf",
    mmproj_path="/path/to/model/mmproj-gemma4-e4b.gguf",
    n_ctx=8192, n_batch=512, n_ubatch=512, n_gpu_layers="all",
    ctx_checkpoints=0,
    chat_handler_kwargs={
        "image_max_tokens": 512,
        "batch_max_tokens": 512,
        "video_fps_target": 2.0,
        "video_timestamp_interval_ms": 3000,
        "video_no_soundtrack": True,
        "verbose": False,
    },
    verbose=False,
)
```

For normal video decoding, install ffmpeg/ffprobe and use an MTMD_VIDEO build.
Add `video_ffmpeg_bin_dir="/path/to/ffmpeg/bin"` to `chat_handler_kwargs` if the
binaries are not on PATH. Set video options at construction and serialize use of
the model. Image/audio support depends on the projector; capability fields are
populated lazily on its first model-bound operation.

### Images, audio, video and mixed attachments

Gemma4's template uses `image_url`, `audio` and `video` content parts as below.
Each tuple is an independent request; run only the cases for which you have the
files and projector support. Lists preserve attachment order.

```python
from pathlib import Path


def file_uri(path):
    return Path(path).resolve().as_uri()


image = {"type": "image_url", "image_url": {"url": file_uri("/path/to/image.jpg")}}
second_image = {"type": "image_url", "image_url": {"url": file_uri("/path/to/second.jpg")}}
audio = {"type": "audio", "audio": file_uri("/path/to/reference.wav")}
video = {"type": "video", "video": file_uri("/path/to/clip.mp4")}

examples = [
    ([image], "Describe this image."),
    ([image, second_image], "What differs between the two images?"),
    ([audio], "Transcribe the speech."),
    ([video], "Describe the main events in order."),
    ([image, audio], "How does the speech relate to the image?"),
]
for attachments, question in examples:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": [
            *attachments, {"type": "text", "text": question},
        ]}],
        max_tokens=256, temperature=0,
    )
    print(response["choices"][0]["message"]["content"])
```

These examples generate text. Audio transcription is distinct from TTS output.
Current vendor video soundtrack extraction applies to LFM3-Omni, not Gemma4;
submit a separate audio part when Gemma4 needs to hear a video's soundtrack.
A different model template may require different part names even when Python
can extract those media types.

### In-memory files: base64 data URIs

Use encoded file bytes from an upload, recording or application buffer:

```python
import base64
from pathlib import Path


def data_uri(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


audio_bytes = Path("/path/to/reference.wav").read_bytes()
audio = {"type": "audio", "audio": data_uri(audio_bytes, "audio/wav")}
```

Use this `audio` part in the request block above. The same helper accepts JPEG,
PNG or MP4 file bytes with the correct MIME type. Raw float samples or RGB arrays
must first be encoded, or supplied through a custom integration such as
[the RGB frame decoder](#complete-rgb-frame-decoder-example). Data URIs and file
paths both reach native media decoding; base64 also adds memory overhead.

### Follow-up questions about the same media

Keep the original media message and the returned assistant message in history:

```python
messages = [{"role": "user", "content": [
    image, {"type": "text", "text": "Describe this image briefly."},
]}]
first = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0)
messages.append(first["choices"][0]["message"])
messages.append({"role": "user", "content": "What stands out most?"})
second = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0)
messages.append(second["choices"][0]["message"])
print(second["choices"][0]["message"]["content"])
```

Stable history and the same loaded instance allow prefix/media reuse when the
model and cache path support it; this does not guarantee a cache hit on every
request. Omitting the original media removes it from the conversation. To start
fresh, clear your `messages` list and call `llm.reset()`; weights stay loaded.
Shorten clips, reduce video fps/image tokens or trim history if the context fills.

### Save and restore message history

The `messages` list can be stored as JSON without serializing the loaded model:

```python
import json
from pathlib import Path

Path("conversation.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
messages = json.loads(Path("conversation.json").read_text(encoding="utf-8"))
```

This saves conversation content, not native KV state. Referenced files must still
exist; embedded data URIs travel with the JSON and increase its size.

### Stream output and measure response time

This snippet assumes the `messages` list from the previous example and adds
a short follow-up question:

```python
import time

messages.append({"role": "user", "content": "Summarize your answer in one sentence."})
started = time.perf_counter()
first_content_seconds = None
response = llm.create_chat_completion(messages=messages, stream=True, max_tokens=256, temperature=0)
try:
    for chunk in response:
        for choice in chunk.get("choices", []):
            text = choice.get("delta", {}).get("content") or ""
            if text:
                if first_content_seconds is None:
                    first_content_seconds = time.perf_counter() - started
                print(text, end="", flush=True)
finally:
    response.close()
print()
print("First visible text (s):", first_content_seconds)
print("Request duration (s):", time.perf_counter() - started)
```

The first-visible-text measurement includes media loading, prefill and any delay
before a content delta; it is not necessarily the first generated token. Other
fields, such as separate reasoning deltas, are not printed here. To interrupt
from another control thread, call `llm.abort()`; keep the response consumer's
`finally` cleanup. Do not close or reset the model concurrently with generation.
If sharing an instance, hold an application lock from request creation until the
stream is consumed/closed, rather than locking individual chunks.

### Optional Gemma4 Assistant MTP

Before constructing `Llama`, create this configuration and pass it as
`speculative=mtp` in the loading block. A compatible `gemma4-assistant` sidecar
GGUF is required in addition to the backbone and projector:

```python
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

mtp = SpecConfig(
    spec_type=SpeculativeType.DRAFT_MTP,
    draft_model_path="/path/to/model/gemma4-e4b-assistant.gguf",
    draft_n_max=3, draft_n_min=0, draft_p_min=0.0,
    draft_n_gpu_layers="all", draft_backend_sampling=True,
)
# Add speculative=mtp to Llama(...) before loading the instance.
```

The matching Gemma4 assistant shares target KV through the native MTP path;
there is no separate media submission to the assistant. Keep
`n_batch >= 1 + draft_n_max` and measure whether speculation improves your
workload. `ctx_checkpoints=0` disables periodic prompt checkpoints; it does not
turn off MTP rollback or all other state reuse. See
[shared-KV MTP](LlamaSpeculative.md#external-mtp-gguf) and
[the speculative decoding guide](../examples/speculative-decoding.md).

After consuming a request, inspect the available generation statistics:

```python
stats = llm.last_speculative_stats
for key in ("generation_tokens", "sustained_tokens_per_second",
            "draft_token_acceptance_rate", "mean_accepted_length"):
    print(key, stats.get(key))
```

Acceptance statistics describe speculative requests, not media cache hits. The
end-to-end request duration above includes preprocessing; sustained generation
throughput measures a different phase. When done, use `llm.close()` (preferably
in your application's `finally` block); it also closes the auto-created handler.

## Embeddings and speech synthesis

`MTMDEmbeddingHandler` uses the base tokenization/encoding interfaces and adds
pooling and normalization. See [the embedding API](LlamaEmbedding.md#shared-mtmd-encoding-interfaces)
for configuration and return values.

`MTMDAudioGenerator` produces non-streaming speech. If the projector fails to
load, its capability query fails, or the native TTS helper fails to initialize
on first use, the generator releases the partial initialization. You can retry
`create_speech()` on the same open generator with the same `Llama` instance
or another instance compatible with its projector. For model setup, reference
audio and sampling options, see [the TTS guide](../examples/audio/audio-tts.md).

For integrations that manage native handles directly, see
[the MTMD ctypes bindings](MTMDCppBindings.md).

## Custom video decoders

`mtmd_helper_video_init_params.decoder` accepts a pointer to a
`mtmd_helper_video_decoder`; NULL selects built-in ffmpeg. The decoder must
provide `open`, `read_frame` and `close`; `start` and `read_audio` are optional.
`mtmd_helper_video_stream_info` describes displayed dimensions (rotation and
pixel aspect ratio applied), source fps, frame count and duration.

Use the callback types in `mtmd_cpp.py`. Keep callbacks, `user_data` and decoder
state alive until `mtmd_helper_video_free` finishes. `open` receives input bytes
valid only during that callback, so copy them if needed later. `read_frame`
provides scaled RGB24 frames in presentation order; NULL RGB storage asks it to
skip a frame. `start` can set `fps_picked` when it handles fps resampling itself.

`read_audio` returns mono float32 PCM at the requested rate. Its ctypes callback
returns a buffer address (or `None`), because ctypes cannot construct Python
callbacks returning a typed float pointer. Retain the audio buffer until the
next decoder call. Catch exceptions inside callbacks and return failure values.

The `video` capability flag reports the built-in decoder only; it does not
account for an application-provided decoder. `MTMDBaseHandler` and its subclasses
accept `video_decoder=decoder`, where `decoder` is an
`mtmd_helper_video_decoder` instance. A visual projector plus a custom decoder
enables high-level video input even when built-in decoding is unavailable.

The handler retains the decoder and its ctypes ownership graph until `close()`
finishes releasing child resources and the MTMD context. Each successful native
`open` must return an independent state; native video cleanup calls the decoder's
`close` for that state. Raw addresses do not establish Python ownership: retain
external objects yourself if they are not referenced by the callbacks or ctypes.
Multiple media inputs can be decoded concurrently, so shared callback data needs
appropriate synchronization. Reentrant handler closure during media preparation
is rejected. Do not mutate the decoder table or options while the handler is open.

Pass either `video_decoder` with individual `video_*` options, or an entire
`mtmd_helper_init_opt` whose `video_params.decoder` is set. The two configuration
styles cannot be combined. Both paths retain the decoder ownership chain.

### Complete RGB frame decoder example

<details>
<summary>Expand the runnable example: supply already-decoded RGB frames</summary>


This runnable callback example requires Pillow and matching vision backbone/mmproj
GGUF files. It creates red and blue frames, then supplies them to MTMD without
encoding an MP4 or launching ffmpeg. Replace the demo frames with frames decoded
by your application (for example, ComfyUI images converted to Pillow RGB images).
The temporary file contains a private source marker recognized by this example;
it is not an MP4 decoder. Replace the two model paths before running.

```python
import ctypes
import itertools
import logging
import math
import os
import threading
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from llama_cpp import Llama, mtmd_cpp as mtmd
from llama_cpp.llama_multimodal import GenericMTMDChatHandler

SOURCE_MARKER = b"mtmd-rgb-frame-demo"


def make_frame_decoder(frames, fps):
    # Snapshot pixels; each open gets an independent iterator over these frames.
    frames = tuple(frame.convert("RGB").copy() for frame in frames)
    if not frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Supply frames and a positive finite fps")
    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("Frames must have the same dimensions")
    states, handles, lock = {}, itertools.count(1), threading.Lock()

    @mtmd.mtmd_helper_video_decoder_open
    def open_video(user_data, path, buf, size, info):
        try:
            # Buffer storage expires when open returns; read it here.
            source = ctypes.string_at(buf, size) if buf else Path(os.fsdecode(path)).read_bytes()
            if source != SOURCE_MARKER:
                return None
            info[0] = mtmd.mtmd_helper_video_stream_info(
                width, height, fps, len(frames), len(frames) / fps,
            )
            with lock:
                handle = next(handles)
                states[handle] = iter(frames)
            return handle  # An opaque integer handle, not a Python object's address.
        except Exception:
            logging.exception("Video open failed")
            return None

    @mtmd.mtmd_helper_video_decoder_read_frame
    def read_frame(state, rgb, out_width, out_height):
        try:
            with lock:
                frame = next(states[state], None)
            if frame is None:
                return False
            if rgb:  # NULL means skip: advance without copying pixels.
                pixels = frame.resize((out_width, out_height)).tobytes()
                ctypes.memmove(rgb, pixels, len(pixels))
            return True
        except Exception:
            logging.exception("Video frame read failed")
            return False

    @mtmd.mtmd_helper_video_decoder_close
    def close_video(state):
        with lock:
            states.pop(state, None)

    # Omitting start lets MTMD select frames for its requested fps.
    # Omitting read_audio makes this a frames-only decoder.
    return mtmd.mtmd_helper_video_decoder(
        open=open_video, read_frame=read_frame, close=close_video,
    )


frames = [Image.new("RGB", (160, 160), color) for color in ("red", "red", "blue", "blue")]
decoder = make_frame_decoder(frames, fps=2.0)

with TemporaryDirectory() as directory:
    source = Path(directory) / "frames.bin"
    source.write_bytes(SOURCE_MARKER)
    with GenericMTMDChatHandler(
        chat_format=None,
        mmproj_path="/path/to/model/mmproj.gguf",
        video_decoder=decoder,
        image_max_tokens=512,
        verbose=False,
    ) as handler, closing(Llama(
        model_path="/path/to/model/backbone.gguf",
        chat_handler=handler,
        n_ctx=8192, n_batch=2048, n_ubatch=1024, n_gpu_layers=-1,
        verbose=False,
    )) as llm:
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "video", "video": str(source)},
                {"type": "text", "text": "What colors appear, in order?"},
            ]}],
            max_tokens=512, temperature=0,
        )
        print(result["choices"][0]["message"]["content"])
```

The answer should describe red followed by blue; wording depends on the model.
Callbacks and the state registry are retained through the returned ctypes table.
Native video cleanup calls `close_video` for each opened state, including failure
paths after a successful open. The model closes before the handler, and the
source file remains available until both have closed. A separate frame source
requires its own decoder instance in this minimal example.

For real files, implement `open_video` using your chosen video library, store
its reader in `states`, decode/scale RGB24 frames in `read_frame`, and close the
reader in `close_video`. Report displayed dimensions and apply rotation and pixel
aspect ratio in that implementation. Copy buffer input or retain an owned reader
before `open_video` returns; the original native buffer cannot be borrowed later.


</details>

Decoder identity is included in the media cache namespace. Repeated requests with
the same handler can reuse media; different decoder handlers do not share those
cache identities. Video bytes still pass through the native format detector, so
valid image/audio inputs can be handled before the custom video callback.

## Related guides

- [Image chat with Qwen](../examples/vision/vision-qwen.md)
- [Speech synthesis](../examples/audio/audio-tts.md)
- [Embedding APIs](LlamaEmbedding.md)
- [MTMD ctypes bindings](MTMDCppBindings.md)

## Fixed-template handlers: compatibility note

For new integrations, fixed-template handlers are **not the recommended default**.
Prefer `GenericMTMDChatHandler(chat_format=None, ...)` after verifying the model's
stored template and media placeholders. The following is the complete list of
model-specific handlers with a built-in `CHAT_FORMAT` in the current source:

| Handlers | Model/template family |
| --- | --- |
| `Llava15ChatHandler`, `Llava16ChatHandler` | LLaVA USER/ASSISTANT prompts. |
| `ObsidianChatHandler`, `NanoLlavaChatHandler` | Model-specific ChatML-style prompts and separators. |
| `MoondreamChatHandler` | Question/Answer prompts. |
| `Llama3VisionAlphaChatHandler` | Llama 3 vision-alpha role format. |
| `MiniCPMv26ChatHandler`, `MiniCPMv45ChatHandler`, `MiniCPMV46ChatHandler` | MiniCPM-V version-specific templates. |
| `Gemma3ChatHandler`, `Gemma4ChatHandler` | Gemma turn formats. |
| `GLM41VChatHandler`, `GLM46VChatHandler` | GLM vision templates. |
| `GraniteDoclingChatHandler` | Granite Docling document prompts. |
| `LFM2VLChatHandler`, `LFM25VLChatHandler` | LFM vision templates. |
| `PaddleOCRChatHandler` | PaddleOCR task prompts. |
| `Qwen25VLChatHandler`, `Qwen3VLChatHandler`, `Qwen35ChatHandler` | Qwen vision/chat templates. |
| `Qwen3ASRChatHandler` | Qwen audio-transcription template. |
| `Step3VLChatHandler` | Step vision template and patch placeholder handling. |

`MTMDChatHandler` also provides a fixed fallback template, used by the generic
handler when no model template is available. That fallback is not guaranteed to
match every model.

Fixed templates do not automatically follow changes to a GGUF's stored template.
These classes remain available for existing matching-model workflows; this is a
usage recommendation, not a formal API deprecation or a claim that every listed
model is obsolete. Some handlers also supply stop rules, reasoning flags or task
behavior, so verify those before switching. If the stored template is absent or
incompatible, retain the matching handler or supply a verified explicit template.
