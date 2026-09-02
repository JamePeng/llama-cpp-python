---
title: llama.cpp ctypes Bindings
module_name: llama_cpp.llama_cpp
source_file: llama_cpp/llama_cpp.py
last_updated: 2026-09-03
version_target: "latest"
---

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
application usage is documented in [[Llama]].
