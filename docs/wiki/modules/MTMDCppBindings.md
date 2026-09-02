---
title: MTMD ctypes Bindings
module_name: llama_cpp.mtmd_cpp
source_file: llama_cpp/mtmd_cpp.py
last_updated: 2026-09-03
version_target: "latest"
---

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
  provides the managed multimodal integration used by the high-level API.
- [`llama_cpp/llama_chat_format.py`](../../../llama_cpp/llama_chat_format.py)
  contains the chat handlers that prepare multimodal messages and inputs.

## Stability

These bindings track the vendored MTMD headers and are intended for advanced
integrations. Native handles and returned pointers remain subject to the
ownership and lifetime rules documented beside their declarations in
`mtmd_cpp.py` and in the corresponding upstream headers.
