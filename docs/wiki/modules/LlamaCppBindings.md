---
title: llama.cpp ctypes Bindings
module_name: llama_cpp.llama_cpp
source_file: llama_cpp/llama_cpp.py
last_updated: 2026-11-04
version_target: "latest"
---

# llama.cpp ctypes Bindings

## Overview

The low-level Python bindings for `llama.cpp` and ggml are maintained directly
in the source tree. Because this interface follows the bundled native code, the
source files are the authoritative and most current reference.

## Source References

Refer to the following files according to the level of API you need:

- [`llama_cpp/llama_cpp.py`](../../../llama_cpp/llama_cpp.py) for the direct
  Python bindings corresponding to the `llama.cpp` C API.
- [`llama_cpp/_ggml.py`](../../../llama_cpp/_ggml.py) for the direct Python
  bindings and shared definitions corresponding to ggml.
- [`llama_cpp/_internals.py`](../../../llama_cpp/_internals.py) for the managed
  internal Python wrappers built on top of those bindings.
- [`llama_cpp/llama.py`](../../../llama_cpp/llama.py) for the high-level public
  `Llama` interface used by applications.

For the native declarations against which these bindings are maintained, refer
to the headers under [`vendor/llama.cpp`](../../../vendor/llama.cpp). General
application usage is documented in [Llama](../core/Llama.md).
For managed internal batch builders and their native conversion paths, see
[Llama Batch](LlamaBatch.md).

## Optional device feature extensions

The staging APIs in [`llama-ext.h`](../../../vendor/llama.cpp/src/llama-ext.h)
include the following Python bindings for DFlash-family feature transfer:

| Binding | Purpose |
|---|---|
| `llama_set_embeddings_layer_inp_dev(ctx, value)` | Capture enabled target layer inputs in a device buffer, with rows in input batch order and layers in ascending ID order. Configure before decoding. |
| `llama_supports_embd_row(ctx)` | Report whether the draft context's backends can read the feature buffer of its `ctx_other`. Enable capture on the peer before querying. |
| `llama_batch_ext_add_embd_row(batch, seq_id, row)` | Append a borrowed device feature row. Returns its batch index or a negative add error code; positions must be set separately. |

These bindings resolve platform-specific C++ symbol aliases and are optional at
import. An explicit call raises `RuntimeError` when its symbol is unavailable;
importing the package does not make an older native build support the API.
The managed speculative engine checks availability and backend support before
selecting device transfer, with host copies as the fallback. See
[device feature transfer](LlamaSpeculative.md#device-feature-transfer) and
[batch ownership](LlamaBatch.md#device-feature-rows).
