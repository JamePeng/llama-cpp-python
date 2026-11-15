---
title: llama.cpp ctypes Bindings
module_name: llama_cpp.llama_cpp
source_file: llama_cpp/llama_cpp.py
last_updated: 2026-11-13
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
  bindings and shared definitions corresponding to a selected ggml backend and
  tensor subset. The full tensor-operator and GGUF editing APIs remain outside
  this wrapper's scope.
- [`llama_cpp/_internals.py`](../../../llama_cpp/_internals.py) for the managed
  internal Python wrappers built on top of those bindings.
- [`llama_cpp/llama.py`](../../../llama_cpp/llama.py) for the high-level public
  `Llama` interface used by applications.

For the native declarations against which these bindings are maintained, refer
to the headers under [`vendor/llama.cpp`](../../../vendor/llama.cpp). General
application usage is documented in [Llama](../core/Llama.md).
For managed internal batch builders and their native conversion paths, see
[Llama Batch](LlamaBatch.md).

## Native 1.0 API compatibility

Use native libraries built from the vendor revision shipped with the Python
package. The 1.0 API removes deprecated exports; Python no longer binds those
names. Existing low-level callers should migrate as follows:

| Removed API | Replacement |
|---|---|
| `llama_load_model_from_file`, `llama_free_model`, `llama_new_context_with_model` | `llama_model_load_from_file`, `llama_model_free`, `llama_init_from_model` |
| `llama_n_ctx_train`, `llama_n_embd`, `llama_n_layer`, `llama_n_head`, `llama_n_vocab` | Corresponding `llama_model_n_*` queries; use `llama_vocab_n_tokens` for vocabulary size. |
| `llama_token_get_*`, `llama_token_is_*` | Corresponding `llama_vocab_get_*` and `llama_vocab_is_*` queries. |
| `llama_token_bos/eos/eot/sep/nl/pad` and `llama_token_fim_*` | Corresponding `llama_vocab_*` accessors. Use `llama_vocab_bos` for the removed CLS accessors. |
| `llama_add_bos_token`, `llama_add_eos_token` | `llama_vocab_get_add_bos`, `llama_vocab_get_add_eos` |
| `llama_get_state_size`, `llama_copy_state_data`, `llama_set_state_data` | `llama_state_get_size`, `llama_state_get_data`, `llama_state_set_data`; copy/set calls require an explicit buffer size. |
| `llama_load_session_file`, `llama_save_session_file` | `llama_state_load_file`, `llama_state_save_file` |
| `llama_set_warmup` | Perform warmup evaluations explicitly. There is no replacement toggle. |
| `llama_sampler_init_grammar_lazy` | `llama_sampler_init_grammar_lazy_patterns`; triggers are regex patterns rather than literal trigger words. |

The managed `LlamaModel.token_cls()` accessor is also removed. Call
`token_bos()` explicitly when the model uses BOS for its classification token.

`llama_context_params` no longer contains `defrag_thold`. GGML removes
`GGML_OP_ADD1`, `GGML_OP_DIAG_MASK_INF`, and `GGML_OP_DIAG_MASK_ZERO`; subsequent
operation IDs change and `GGML_OP_COUNT` becomes 99. Use the current `GGML_OP`
members when inspecting native tensors instead of retaining old numeric IDs.
The removed `GGML_PREC_DEFAULT` alias is replaced by `GGML_PREC_UNDEFINED`.

## Sampler parameters

The current penalties constructor is
`llama_sampler_init_penalties(n_vocab, penalty_last_n, penalty_repeat,
penalty_freq, penalty_present)`: pass the vocabulary size first. The Python
binding and managed `LlamaSampler.add_penalties()` use this five-argument form.
Repeat penalty must be positive; frequency and presence penalties must be finite.

The raw penalties and DRY constructors clamp negative history lengths to zero,
disabling their histories. Resolve a context-size sentinel before calling the
C API. The public `Llama` methods resolve `-1` to `n_ctx_seq()` and preserve
`0` as disabled. Standalone internal `LlamaSamplingContext` construction only
has a model, so its `-1` fallback uses `model.n_ctx_train()`.

DRY takes a vocabulary handle, not a model or context handle. With DRY enabled,
native code skips empty sequence breakers, truncates each to 40 bytes before
deduplication, and processes at most `LLAMA_DRY_MAX_SEQ_BREAKERS = 64` distinct
breakers. Each distinct breaker requires a vocabulary scan during initialization.
This constant is a macro exposed by Python, not a new DLL function.

Use `llama_sampler_init_temp_ext(t, delta, exponent)` for dynamic temperature.
Adaptive-P, Mirostat, greedy, and distribution samplers select a token and belong
at the end of a manually assembled chain. `llama_sampler_sample()` both samples
and accepts a token; do not call `llama_sampler_accept()` again for that token.
For several outputs from one sampler, consume outputs in order without gaps.
Chain addition transfers ownership; free the chain once rather than freeing its
children separately. See [high-level sampling](../core/Llama.md#sampling-parameters-and-precedence)
for request defaults and Mirostat precedence.

## Vocabulary and tokenization

`llama_model_get_vocab(model)` returns a borrowed vocabulary handle; keep the
model open while querying it. `llama_vocab_type(vocab)` returns a member of
`LlamaVocabType`, including `LLAMA_VOCAB_TYPE_PLAMO3 = 8`. The separately exposed
`llama_vocab_pre_type` enum follows the internal `llama-vocab.h`, including
`SOPRANO = 65` and `TEKKEN_NUM3 = 66`; it is not the return type of
`llama_vocab_type()` and does not imply a public pre-tokenizer query function.

Special-token queries preserve `LLAMA_TOKEN_NULL` (`-1`) for absent tokens.
Use `llama_vocab_is_eog()` to test generation termination, since EOS and EOT
can both end generation. `llama_vocab_get_text()` returns copied native token
text, while `llama_token_to_piece()` produces rendered token bytes. The
suppressed-token query returns borrowed storage; `LlamaModel.suppress_tokens()`
copies it into a Python list.

`llama_tokenize()` takes an explicit input byte length, so embedded NUL bytes
are supported. Its `add_special` flag enables model-configured special-token
insertion; the high-level native tokenizer retains the historical `add_bos`
parameter name for this flag. `parse_special` separately controls recognition
of special-token text. A negative result usually gives the required token
capacity, but `INT32_MIN` indicates overflow and must not become an allocation
size. The managed tokenizer raises `RuntimeError` for that sentinel.

The output arguments of `llama_token_to_piece()` and `llama_detokenize()`
require writable storage, such as `ctypes.create_string_buffer()`, with lengths
in bytes. Do not pass immutable Python bytes as output buffers. Token pieces
are not NUL-terminated; use the returned byte count instead of `.value` when
the result may contain embedded NUL bytes.

## Optional staging extensions

The staging APIs in [`llama-ext.h`](../../../vendor/llama.cpp/src/llama-ext.h)
allow breaking changes and C++ types. Python binds the 21 declarations whose
signatures can be represented directly with ctypes, including model queries,
quantization helpers, and graph reservation. Speculative feature transfer is
now part of the public C API described below.

Each binding accepts a plain C symbol and platform-specific MSVC, Mach-O, and
ELF C++ symbol names. These are alternative export names for one function,
not separate APIs. These aliases apply to the remaining staging declarations;
promoted functions resolve their plain C export names.

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

## Public speculative decoding API

Eighteen former staging declarations now live in
[`llama.h`](../../../vendor/llama.cpp/include/llama.h), including NextN output
capture, layer-input capture, device rows, peer-context access, and the three
DFlash selector/target-layer queries. These are required C bindings with no
C++ symbol aliases. `llama_set_embd_state_src` is renamed to
`llama_set_ctx_state_src`; the managed context method is now
`LlamaContext.set_ctx_state_src(source)` as well. NextN row offsets and counts
remain `ctypes.c_int64` on every platform.

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
| `llama_set_ctx_state_src(ctx, ctx_src)` | Borrow `ctx_src`'s device state buffer for draft inputs; `None` detaches it. |
| `llama_batch_ext_set_embd_state_row(batch, idx, row)` | Assign an entry's state row. All submitted entries must use device state rows, or none may use them. |

The managed speculative engine checks backend support before selecting device
transfer, with host copies as the fallback when device transfer is rejected.
A missing public C export prevents import and requires matching native libraries. See
[device feature transfer](LlamaSpeculative.md#device-feature-transfer) and
[batch ownership](LlamaBatch.md#device-feature-rows).

## APIs requiring a C++ bridge

Five staging declarations are intentionally unavailable as direct Python
bindings: `llama_arch_all`, `llama_arch_from_string`, `llama_arch_kv_name`,
`llama_get_memory_breakdown`, and `llama_memory_get_buffers`. Their signatures
use `std::string`, `std::vector`, or `std::map`; exported symbols alone do not
make those types safe to pass through ctypes. They require a C++ bridge that
exposes C-compatible arguments, results, and ownership.

MTMD has the same limitation for `mtmd_get_memory_usage()`: its C++ result
contains a `std::map`, so it requires a bridge rather than a direct ctypes
binding. The public C functions in `mtmd.h` and `mtmd-helper.h` are bound in
[`mtmd_cpp.py`](../../../llama_cpp/mtmd_cpp.py).

The exposed `mtmd_pos_type` enum mirrors an internal definition in
[`mtmd.cpp`](../../../vendor/llama.cpp/tools/mtmd/mtmd.cpp). It includes
`MTMD_POS_TYPE_SENSENOVA_U1 = 3`, `MTMD_POS_TYPE_CANVAS = 4`, and
`MTMD_POS_TYPE_COUNT = 5`. These constants do not add a public position-type
query or establish high-level support for a model on their own.
