---
title: MTMD ctypes Bindings
module_name: llama_cpp.mtmd_cpp
source_file: llama_cpp/mtmd_cpp.py
last_updated: 2026-11-16
version_target: "latest"
---

# MTMD ctypes Bindings

`llama_cpp.mtmd_cpp` exposes the vendored MTMD C API through ctypes. For image,
audio and video chat, start with [Llama Multimodal](LlamaMultimodal.md). Use these
bindings when integrating a custom runtime that manages native handles directly.

## Open and query a projector

Replace the paths with matching backbone/mmproj GGUF files. Keep the backbone
alive until the MTMD context is released.

```python
from contextlib import closing

from llama_cpp import Llama, mtmd_cpp as mtmd

with closing(Llama(model_path="/path/to/model/backbone.gguf", n_gpu_layers=-1)) as llm:
    params = mtmd.mtmd_context_params_default()
    ctx = mtmd.mtmd_init_from_file(b"/path/to/model/mmproj.gguf", llm._model.model, params)
    if not ctx:
        raise RuntimeError("Failed to load the multimodal projector")
    try:
        print("Media marker:", mtmd.mtmd_get_marker(ctx).decode("utf-8"))
        print("Vision:", mtmd.mtmd_support_vision(ctx))
        print("Audio:", mtmd.mtmd_support_audio(ctx))
    finally:
        mtmd.mtmd_free(ctx)
```

`llm._model.model` is an internal native handle used here for direct C API access.
Use `mtmd_get_marker(ctx)` for the loaded model rather than assuming the default
marker matches every projector.

## Processing flow

| Step | API |
| --- | --- |
| Decode encoded media | `mtmd_helper_bitmap_init_from_file` / `mtmd_helper_bitmap_init_from_buf`. |
| Prepare inputs | `mtmd_input_chunks_init`, then `mtmd_tokenize` or `mtmd_tokenize_from_parts`. |
| Encode media | `mtmd_encode_chunk` or `mtmd_batch_*`; text chunks already contain token IDs. |
| Evaluate with the backbone | `mtmd_helper_eval_chunks`, or submit copied embeddings and positions to the llama context. |
| Release resources | `mtmd_input_chunks_free`, `mtmd_bitmap_free`, `mtmd_helper_video_free`, then `mtmd_free`. |

Check return codes and release partially created resources on failure. Copy
borrowed embedding/output buffers before the owning context is reused or freed.
`mtmd_helper_get_n_tokens` counts input rows; `mtmd_helper_get_n_pos` counts decoder
positions. These counts can differ for M-RoPE models.

Video options accept a `mtmd_helper_video_decoder` callback table. See the
[custom decoder contract and runnable example](LlamaMultimodal.md#custom-video-decoders).
The native `video` capability reports built-in decoding only; it does not count
custom decoders.

## Inspect media without loading a model

`mtmd_helper_get_media_meta` reads encoded media metadata: image dimensions,
audio source sample rate/duration, or video dimensions/duration/frame count.
It returns `0` on success and `1` for an unrecognized format. Unknown duration
and video frame count are `-1`; fields not applicable to the media type keep
their defaults. Video probing uses ffprobe or the configured custom decoder's
`open`/`close`, without reading frames. Keep custom callbacks alive for the call.

```python
import ctypes
from pathlib import Path
from llama_cpp import mtmd_cpp as mtmd

payload = Path("/path/to/reference.wav").read_bytes()
buffer = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
metadata = mtmd.mtmd_helper_media_meta()
options = mtmd.mtmd_helper_init_opt_default()
if mtmd.mtmd_helper_get_media_meta(buffer, len(buffer), options, ctypes.byref(metadata)) != 0:
    raise ValueError("Unrecognized media")
print(mtmd.mtmd_helper_media_type(metadata.type).name)
print(metadata.width, metadata.height, metadata.sample_rate, metadata.duration, metadata.n_frames)
```

The TTS input ABI includes `speed` (default `1.0`) and `loudness` (default `0.0`,
model-defined loudness). Zero-initializing the structure leaves an invalid
`speed=0`; use `mtmd_helper_gen_audio_inp_default` or explicitly set both defaults.
`mtmd_helper_gen_audio_set_input` returns `2` for model-invalid input, including
unsupported delivery overrides, and `1` for other errors. Non-default speed and
loudness are currently supported only by Zonos3. The high-level
`MTMDAudioGenerator.create_speech` exposes both settings; see
[Zonos3 synthesis](../examples/audio/audio-tts.md#zonos3).

The input also includes `speaker_ref_text` (optional reference transcript),
`cfg_scale` and `flow_steps` (`0` selects model defaults). These options are
model-specific; non-default CFG/flow settings are rejected by unsupported models.
`mtmd_gen_inp` includes serialized backbone-state input and flow/guidance fields
for YuE2 and MiniMax-Music3; initialize it with `mtmd_gen_inp_default(ctx)`.
YuE2 is also supported by the high-level generator; see
[song synthesis](../examples/audio/audio-tts.md#yue2).
MiniMax-Music3 and Fish S2 enums are bound, but those pipelines are not yet
enabled by `create_speech`.

`mtmd_gen_audio_get_info` reports `n_channels`. Low-level `mtmd_gen_out.n_samples`
counts samples across all channels; helper `get_output` counts samples **per
channel**. Raw float32 PCM is interleaved, and WAV headers carry the channel count.
`mtmd_helper_gen_audio_max_frames` returns the model's default helper-step limit;
`0` lets the pipeline stop at EOS or its own limits. Helper initialization can return NULL
when the backbone context cannot run the model, including insufficient capacity.

## Source and compatibility

[`mtmd_cpp.py`](../../../llama_cpp/mtmd_cpp.py) documents signatures, callbacks,
return codes and ownership next to their original C declarations. The matching
[vendor headers](../../../vendor/llama.cpp/tools/mtmd/mtmd-helper.h) define the
helper ABI. Use a library built from the matching vendor revision: MTMD helpers
can change structure layouts and are not a stable ABI.
