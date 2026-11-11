---
title: llama.cpp ctypes Bindings
module_name: llama_cpp.llama_cpp
source_file: llama_cpp/llama_cpp.py
last_updated: 2026-11-11
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

## Optional staging extensions

The staging APIs in [`llama-ext.h`](../../../vendor/llama.cpp/src/llama-ext.h)
allow breaking changes and C++ types. Python binds the 39 declarations whose
signatures can be represented directly with ctypes, including model queries,
quantization helpers, graph reservation, and speculative feature transfer.

Each binding accepts a plain C symbol and platform-specific MSVC, Mach-O, and
ELF C++ symbol names. These are alternative export names for one function,
not separate APIs. For the NextN row-copy functions, the Mach-O aliases encode
`int64_t` as `long long`, while the ELF aliases encode it as `long`.
The argument types remain `ctypes.c_int64` on both platforms.

These bindings are optional at import. An explicit call raises `RuntimeError`
when its symbol is unavailable; importing the package does not make an older
native build support the API. Successful symbol resolution establishes neither
model nor backend support.

### Architecture and decision-head queries

| Binding | Result |
|---|---|
| `llama_arch_name(arch)` | Canonical architecture name as Python `bytes`, or `b"unknown"`. |
| `llama_arch_supports_sm_tensor(arch)` | Whether the architecture supports `LLAMA_SPLIT_MODE_TENSOR`. |
| `llama_arch_supports_mixed_batch(arch)` | Whether token and embedding inputs can share a native microbatch. |
| `llama_arch_supports_save(arch)` | Whether native model saving supports the architecture. |
| `llama_model_decision_label(model, i)` | Decision-head label as Python `bytes`, or `None` past the label list. |

The architecture queries take integer `llm_arch` values from the loaded native
build's [`llama-arch.h`](../../../vendor/llama.cpp/src/llama-arch.h). These IDs
are not model handles and can change between builds. Python does not provide a
stable architecture-ID enumeration. Decision-label indices are `uint32_t`;
the model must remain open during the call. Returned strings are copied into
Python bytes and require no native free.

`llama_ftype_get_default_type(ftype)` returns a native `ggml_type` enum as a
Python integer; its ctypes return type is `ctypes.c_int`.

### Device feature transfer

The following bindings support DFlash-family feature transfer:

| Binding | Purpose |
|---|---|
| `llama_set_embeddings_layer_inp_dev(ctx, value)` | Capture enabled target layer inputs in a device buffer, with rows in input batch order and layers in ascending ID order. Configure before decoding. |
| `llama_supports_embd_row(ctx)` | Report whether the draft context's backends can read the feature buffer of its `ctx_other`. Enable capture on the peer before querying. |
| `llama_batch_ext_add_embd_row(batch, seq_id, row)` | Append a borrowed device feature row. Returns its batch index or a negative add error code; positions must be set separately. |

NextN/MTP state transfer uses these additional bindings:

| Binding | Purpose |
|---|---|
| `llama_set_embeddings_nextn_dev(ctx, value)` | Enable unmasked NextN capture in a device buffer; returns whether the buffer can be used. |
| `llama_get_embeddings_nextn_rows(ctx, row0, n_rows, out)` | Copy device state rows to caller-owned float32 storage. |
| `llama_set_embeddings_nextn_rows(ctx, row0, n_rows, data)` | Copy host float32 rows into the device state buffer, including reserved pending rows. |
| `llama_set_embd_state_src(ctx, src)` | Borrow `src`'s device state buffer for draft inputs; `None` detaches it. |
| `llama_batch_ext_set_embd_state_row(batch, idx, row)` | Assign an entry's state row. All submitted entries must use device state rows, or none may use them. |

The managed speculative engine checks availability and backend support before
selecting device transfer, with host copies as the fallback. See
[device feature transfer](LlamaSpeculative.md#device-feature-transfer) and
[batch ownership](LlamaBatch.md#device-feature-rows).

### APIs requiring a C++ bridge

Five staging declarations are intentionally unavailable as direct Python
bindings: `llama_arch_all`, `llama_arch_from_string`, `llama_arch_kv_name`,
`llama_get_memory_breakdown`, and `llama_memory_get_buffers`. Their signatures
use `std::string`, `std::vector`, or `std::map`; exported symbols alone do not
make those types safe to pass through ctypes. They require a C++ bridge that
exposes C-compatible arguments, results, and ownership.
