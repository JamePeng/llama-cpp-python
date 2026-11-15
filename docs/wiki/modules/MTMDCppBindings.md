---
title: MTMD ctypes Bindings
module_name: llama_cpp.mtmd_cpp
source_file: llama_cpp/mtmd_cpp.py
last_updated: 2026-11-15
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

## Source and compatibility

[`mtmd_cpp.py`](../../../llama_cpp/mtmd_cpp.py) documents signatures, callbacks,
return codes and ownership next to their original C declarations. The matching
[vendor headers](../../../vendor/llama.cpp/tools/mtmd/mtmd-helper.h) define the
helper ABI. Use a library built from the matching vendor revision: MTMD helpers
can change structure layouts and are not a stable ABI.
