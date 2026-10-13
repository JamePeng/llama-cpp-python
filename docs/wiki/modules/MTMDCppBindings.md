---
title: MTMD ctypes Bindings
module_name: llama_cpp.mtmd_cpp
source_file: llama_cpp/mtmd_cpp.py
last_updated: 2026-10-13
version_target: "latest"
---

# MTMD ctypes Bindings

## Overview

This is the low-level ctypes interface to the bundled MTMD multimodal C API.
Most applications should use `mmproj_path`, the multimodal chat handlers, or
the higher-level APIs in `llama_cpp.llama_multimodal` instead.

## Source References

This page intentionally stays source-oriented so it does not become a stale
copy of the native API:

- [`llama_cpp/mtmd_cpp.py`](../../../llama_cpp/mtmd_cpp.py) is the authoritative
  list of MTMD types, enums, callbacks, structures, and bound functions.
- [`llama_cpp/llama_multimodal.py`](../../../llama_cpp/llama_multimodal.py)
  provides `MTMDBaseHandler`, multimodal chat/TTS handlers, and
  `MTMDEmbeddingHandler` for managed encoding and inference.
- [`llama_cpp/llama_chat_format.py`](../../../llama_cpp/llama_chat_format.py)
  provides chat-format registration and shared completion/message helpers.

For text/image/audio embeddings with a matching projector, see
[LlamaEmbedding](LlamaEmbedding.md). The loaded context's media marker is queried
with `mtmd_get_marker()`; `mtmd_default_marker()` is only the default marker.

## Stability

These bindings track the vendored MTMD headers and are intended for advanced
integrations. Native handles and returned pointers remain subject to the
ownership and lifetime rules documented beside their declarations in
`mtmd_cpp.py` and in the corresponding upstream headers.
