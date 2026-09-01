---
title: Llama Speculative Decoding
module_name: llama_cpp.llama_speculative
source_file: llama_cpp/llama_speculative.py
last_updated: 2026-09-02
version_target: "latest"
---

# Llama Speculative Decoding

## Overview

`llama_cpp.llama_speculative` implements the Python side of the stateful
speculative-decoding lifecycle used by `llama.cpp`. A speculative engine proposes
one or more tokens, the target model verifies `[id_last, draft...]` in one batch,
and the generation loop accepts the matching prefix or rolls rejected state back.

New code should pass a `SpecConfig` to `Llama(speculative=...)`. The old
`Llama(draft_model=...)` callback path and `LlamaDraftModel` are deprecated
compatibility APIs.

The current engines support one sequence (`seq_id=0`). MTP and n-gram engines
are text-only. The DFlash family can consume target token or embedding batches
after the target context has extracted the configured layer inputs.

## Implementation Status

`SpeculativeType` mirrors `common_speculative_type` from `llama.cpp`, so the enum
contains algorithms that do not yet have Python engines.

| Type | Python engine | Status | Availability |
|---|---|---|---|
| `DRAFT_MTP` | `LlamaMTPDecoding` | Implemented for built-in and external MTP | Since `0.3.48` |
| `DRAFT_DFLASH` | `LlamaDFlashDecoding` | Implemented for external DFlash and selector-based DFlash2 GGUFs | Since `0.3.49` |
| `DRAFT_DSPARK` | `LlamaDFlashDecoding` | Implemented for external DSpark GGUFs | Since `0.3.49` |
| `NGRAM_MAP_K` | `LlamaNGramMapDecoding` | Implemented | Since `0.3.48` |
| `NGRAM_MAP_K4V` | `LlamaNGramMapDecoding` | Implemented | Since `0.3.48` |
| `DRAFT_EAGLE3` | none | Declared, not implemented | — |
| `DRAFT_SIMPLE` | none | Declared, not implemented | — |
| `NGRAM_SIMPLE` | none | Declared, not implemented | — |
| `NGRAM_MOD` | none | Declared, not implemented | — |
| `NGRAM_CACHE` | none | Declared, not implemented | — |

Selecting an unimplemented type raises `NotImplementedError` during validation or
engine creation. Eagle3, DFlash, and DSpark require `draft_model_path`; Eagle3
is accepted by configuration validation but does not yet have a native Python
engine.

### Version availability

- `0.3.48` introduced the stateful, text-only MTP path for both target-internal
  NextN heads and compatible external MTP GGUFs. The K and K4V n-gram engines
  were also available in this release.
- `0.3.49` introduced external DFlash, DFlash2, and DSpark support
  through `LlamaDFlashDecoding`. DFlash2 uses `DRAFT_DFLASH` and is detected
  from the sidecar's selector metadata. This functionality was not part of the
  `0.3.48` release.

## Recommended Entry Point

```python
from llama_cpp import Llama
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

llm = Llama(
    model_path="path/to/model-with-mtp.gguf",
    n_ctx=4096,
    n_batch=512,
    n_gpu_layers="all",
    speculative=SpecConfig(
        spec_type=SpeculativeType.DRAFT_MTP,
        draft_n_max=2,
        draft_p_min=0.0,
    ),
)
```

`Llama` validates the configuration, reserves enough target output rows for the
verification batch, creates the correct engine, and closes owned draft resources
with the target model.

Do not pass both `draft_model` and `speculative`; `Llama` rejects that combination.

## `SpeculativeType`

```python
class SpeculativeType(enum.IntEnum):
    ...
```

Useful helpers include:

| Method | Description |
|---|---|
| `is_draft()` | True for model-backed draft-family types. |
| `is_ngram()` | True for n-gram-family types. |
| `is_eagle3()` | True only for `DRAFT_EAGLE3`. |
| `is_mtp()` | True only for `DRAFT_MTP`. |
| `is_dflash()` | True only for `DRAFT_DFLASH`. |
| `is_dspark()` | True only for `DRAFT_DSPARK`. |
| `is_none()` | True only for `NONE`. |
| `to_str()` | Returns the `llama.cpp`-style name, such as `draft-mtp`. |
| `from_str(value)` | Parses canonical names and aliases such as `mtp`, `ngram-k`, and `ngram-k4v`. |

## `SpecConfig`

```python
@dataclass
class SpecConfig:
    ...
```

`SpecConfig` mirrors the relevant `llama.cpp --spec-*` settings and contains
additional Python-engine settings.

### Common draft settings

| Field | Default | Description |
|---|---:|---|
| `spec_type` | `NONE` | Selected speculative algorithm. |
| `draft_n_max` | `3` | Maximum proposed tokens for draft-family engines. |
| `draft_n_min` | `0` | Discard a proposal shorter than this value. |
| `draft_p_split` | `0.1` | Reserved split probability matching `llama.cpp`. |
| `draft_p_min` | `0.0` | MTP/DFlash token-probability threshold, DFlash2 selector-transition threshold, or DSpark predicted acceptance-confidence threshold. |
| `draft_model_path` | `None` | External draft GGUF. Omit it for built-in target MTP heads. |
| `draft_backend_sampling` | `True` | Let supported model-backed engines select compact candidates on the backend instead of copying full logits rows to Python. DFlash2 uses its selector lattice instead, so this request remains inactive for DFlash2. |

### External draft runtime settings

| Field | Default | Description |
|---|---:|---|
| `draft_n_gpu_layers` | `"auto"` | Draft offload setting: integer, `"auto"`, or `"all"`. |
| `draft_n_threads` | `None` | Draft generation thread count. |
| `draft_n_threads_batch` | `None` | Draft prompt/batch thread count. When omitted, it follows `draft_n_threads` if that field is set; otherwise it inherits the target default. |
| `draft_cpu_moe` | `False` | Keep all draft MoE expert tensors on CPU. |
| `draft_n_cpu_moe` | `0` | Keep the first N draft MoE layers on CPU. |
| `draft_devices` | `[]` | Ordered backend device names for the draft model. |
| `draft_tensor_buft_overrides` | `None` | Native tensor buffer-type overrides retained for the lifetime of the external draft model. |
| `draft_type_k`, `draft_type_v` | `None` | Optional draft KV-cache data types. |
| `draft_model_kwargs` | `{}` | Additional native draft model parameters. |

### N-gram settings

| Field | Default | Description |
|---|---:|---|
| `ngram_size_n` | `12` | Number of verified tokens in the lookup key. |
| `ngram_size_m` | `48` | Maximum continuation length. This is independent of `draft_n_max`. |
| `ngram_min_hits` | `1` | Minimum cached continuations required by K4V. |
| `ngram_max_entries_per_key` | `None` | Optional Python cache cap. K4V resolves `None` to 4. |

`max_draft_tokens()` resolves the active algorithm's real verification length:
draft-family engines use `draft_n_max`, while K and K4V use `ngram_size_m`.
The resulting length must not exceed `Llama.n_batch - 1` because `id_last` and
all draft tokens must remain in one verification batch.

Other configuration helpers are `enabled()`, which tests for a non-`NONE`
algorithm; `resolved_draft_n_gpu_layers()`, which maps `"auto"` and `"all"` to
their native values; and `validate()`, which checks implementation status,
ranges, and required sidecar paths before model loading.

`ngram_mod_n_match`, `ngram_mod_n_max`, `ngram_mod_n_min`,
`lookup_cache_dynamic`, and `lookup_cache_static` are retained for alignment
with the corresponding `llama.cpp` algorithm families. Their Python engines are
not implemented, so these fields are not currently actionable.

## `speculative_output_limits()`

```python
total_outputs, outputs_per_sequence = speculative_output_limits(
    n_batch, n_parallel, n_draft
)
```

This public helper mirrors `common_speculative_get_output_limits()` from
`llama.cpp`. It reserves at most `1 + n_draft` target outputs per sequence for
the complete `[id_last, draft...]` verification block, capped by `n_batch`, and
then caps the combined capacity for all configured sequences. `Llama` uses the
result while constructing its target context; callers normally do not need to
invoke it themselves. Non-positive `n_batch` or `n_parallel` values raise
`ValueError`.

## `LlamaSpecEngine`

```python
class LlamaSpecEngine(abc.ABC):
    ...
```

This is the public base interface for stateful engines. Applications normally
provide `SpecConfig` instead of constructing or driving an engine directly.

| Method | Generation-loop role |
|---|---|
| `begin(prompt_tokens, seq_id=0)` | Initialize request state after the prompt prefix is decoded. |
| `process(batch, seq_id=0)` | Consume target tokens and hidden rows after target decode has completed and synchronized. |
| `draft(input_ids, n_past, id_last, n_max, seq_id=0)` | Return only continuation tokens, never `id_last`. |
| `accept(n_accepted, seq_id=0)` | Commit acceptance feedback for the last proposal. |
| `checkpoint(seq_id=0)` | Capture opaque draft-side state before verification. |
| `take_verification_checkpoint(seq_id=0)` | Reuse a draft-time checkpoint when possible. |
| `restore(checkpoint, seq_id=0)` | Restore rejected draft-side state. |
| `reset_checkpoint_stats()` | Reset per-request checkpoint counters and accumulated durations. |
| `can_follow_target_native_rollback()` | Report whether the engine can realign its own state after target native rollback. This does not test target snapshot capacity. |
| `rollback_verified(checkpoint, n_accepted, seq_id=0)` | Keep the sampled token and accepted prefix after native target rollback. |
| `truncate(position, seq_id=0)` | Remove state at and after an absolute position. |
| `clear()` | Clear request state but keep reusable model resources. |
| `close()` | Release owned resources. Calls are expected to be idempotent. |
| `checkpoint_stats()` | Return per-request capture and restore metrics. |

The target model and target context always remain owned by `Llama`.

## `_LlamaModelDraftEngine`

```python
class _LlamaModelDraftEngine(LlamaSpecEngine):
    ...
```

This private base class contains the native plumbing shared by
`LlamaMTPDecoding` and `LlamaDFlashDecoding`. It is an implementation extension
point, not a public engine that applications should instantiate. It deliberately
does not define graph inputs, draft layout, checkpoint semantics, or acceptance
state; concrete engines continue to implement those algorithm-specific parts.

Its shared responsibilities are:

| Helper area | Responsibility |
|---|---|
| Model compatibility | Check vocabulary type, enabled BOS/EOS behavior, vocabulary-size tolerance, shared token text, and close a rejected sidecar safely. |
| Model parameters | Apply draft GPU layers, devices, CPU MoE placement, tensor buffer overrides, arbitrary model parameter overrides, and the algorithm-specific `load_mtp` flag. |
| Context parameters | Copy target defaults while forcing `embeddings=False` and unspecified pooling, then apply draft threads and KV types. |
| Candidate selection | Prefer backend top-k candidate buffers; fall back to selecting from the top ten entries of the full CPU logits row. |
| Native buffers | Copy transient ctypes embedding rows into owned NumPy arrays before another graph execution invalidates their pointers. |
| Lifecycle | Retain ctypes arrays, detach samplers, and close batches, draft context, and an engine-owned model in a shutdown-safe, idempotent order. |

To add another model-backed algorithm, subclass `_LlamaModelDraftEngine`, call
`_init_model_draft_engine()` before allocating native resources, load and
validate an external sidecar with `_load_draft_model()` when needed, derive its
context from `_build_draft_context_params()`, and enable backend sampling only
after `draft_context` exists. The subclass must still implement the
`LlamaSpecEngine` request hooks and call `_close_draft_resources()` from an
idempotent `close()`.

Ownership is intentionally asymmetric: the target model and context are always
borrowed from `Llama`; every concrete engine owns its draft context and batches;
an external draft model is engine-owned, while built-in MTP borrows the target
model containing its NextN heads.

## MTP Engine

`LlamaMTPDecoding` orchestrates the native NextN/MTP graph while participating in
the same `LlamaSpecEngine` lifecycle.

The built-in and external MTP paths have been tested with the Qwen3.5,
Qwen3.6, and Qwen3.8 model families. External MTP has also been tested with a
`gemma4` target paired with a compatible `gemma4-assistant` GGUF. Other model
families may work when their GGUF tensors are compatible, but they have not yet
been validated.

The engine reads target hidden-state rows, sizes them with the models'
`n_embd_out`, and advances a dedicated MTP context. Native recurrent snapshots
are used when the draft context provides enough slots; otherwise the engine
uses a partial on-device checkpoint. The checkpoint captured and restored while
drafting is reused for target verification instead of being captured twice.
Backend sampling avoids copying a full vocabulary-sized logits row to Python
when the backend exposes compact candidate buffers. This is especially
important for large vocabularies.

### Lifecycle

1. **Initialize:** borrow the target model for built-in heads or load an
   external MTP sidecar, create an MTP context linked through `ctx_other`, and
   enable target/draft NextN hidden outputs. `begin()` is currently the base
   no-op because MTP state is initialized from decoded target rows.
2. **Process:** after each synchronized target decode, `process()` copies the
   target NextN rows and advances the draft context. Shared-memory assistants
   such as `gemma4-assistant` skip the ordinary catch-up decode.
3. **Draft:** `draft()` checkpoints the draft context, predicts up to
   `draft_n_max` continuation tokens from `id_last` and the pending hidden row,
   then restores the speculative branch. That same checkpoint is retained for
   the upcoming target verification.
4. **Verify and commit:** the target verifies `[id_last, draft...]`. `accept()`
   advances the pending hidden row after a fully handled step; rejection uses
   `rollback_verified()` or `restore()` to discard the rejected suffix while
   retaining the sampled token and accepted prefix.
5. **Reset or close:** `clear()` removes request-local memory and sampler state;
   `close()` additionally releases the draft batch/context and an externally
   owned sidecar model.

### Built-in target MTP heads

```python
llm = Llama(
    model_path="path/to/model-with-mtp.gguf",
    n_batch=512,
    n_gpu_layers="all",
    speculative=SpecConfig(
        spec_type=SpeculativeType.DRAFT_MTP,
        draft_n_max=2,
        draft_p_min=0.0,
    ),
)
```

When `draft_model_path` is absent, `Llama` automatically enables target MTP
tensor loading. The draft context uses the target model's NextN heads.

### External MTP GGUF

```python
llm = Llama(
    model_path="path/to/target.gguf",
    n_batch=512,
    n_gpu_layers="all",
    speculative=SpecConfig(
        spec_type=SpeculativeType.DRAFT_MTP,
        draft_model_path="path/to/mtp.gguf",
        draft_n_max=2,
        draft_n_gpu_layers="all",
        draft_backend_sampling=True,
    ),
)
```

The engine owns a separate external model and draft context. Initialization
verifies vocabulary type, vocabulary size/token compatibility, and `n_embd_out`.
Multiple NextN heads are chained only when the model exposes more than one head
and its memory is not shared with the target. Shared-memory assistants execute
their layers through their native graph instead.

The `gemma4-assistant` architecture is a special external-MTP case: its context
is linked to the `gemma4` target through `ctx_other`, it consumes target hidden
states and target token embeddings, and its attention memory shares selected
target KV layers. `LlamaMTPDecoding` detects this shared-memory relationship and
automatically skips ordinary draft-context catch-up while using the same target
position for successive draft proposals. No architecture-specific user option
is required beyond supplying the compatible assistant GGUF. This tested path is
currently text-only.

For Qwen3.8 27B, testing so far suggests `draft_n_max=2` as the best starting
point. This is not a universal optimum: GPU, backend, quantization, prompt,
sampling settings, and whether MTP is built in or external can change the
result. Run `examples.benchmark.benchmark_speculative` in the intended
deployment environment and choose the fastest stable value.

Larger `draft_n_max` values increase the verification batch and rollback
exposure and are only useful when later-position acceptance remains high.

## DFlash, DFlash2, and DSpark Engine

`LlamaDFlashDecoding` implements the DFlash-family block-diffusion paths. It
extracts the target layers requested by the draft GGUF, runs the draft encoder,
injects the fused rows into the draft KV cache, and decodes the complete
non-causal mask block in one call. DFlash2 adds selector candidate and transition
outputs; DSpark uses the same cache and block path with its additional Markov
and confidence heads.

### Variant detection and execution

DFlash2 does not have a separate `SpeculativeType`. Configure
`SpeculativeType.DRAFT_DFLASH`; after loading the sidecar, the engine reads
`dflash.selector_top_k`. A positive value selects DFlash2. The selector width is
fixed by the GGUF and is not a runtime tuning parameter.

The resolved variant controls the draft output path:

| Variant | NextN output | Candidate selection | `draft_backend_sampling` |
|---|---|---|---|
| DFlash v1 | Masked rows | Top token from vocabulary logits | Supported |
| DFlash2 | Unmasked rows | Packed selector lattice | Not activated |
| DSpark | Masked rows | Markov/confidence output | Supported where applicable |

For each DFlash2 position, the packed output contains `K` candidate token IDs
and `K x K` transition scores. The engine starts from predecessor index zero,
selects the strongest transition for that predecessor, emits the corresponding
candidate ID, and carries the selected index into the next row. When
`draft_p_min > 0`, the selected transition is normalized across that row's `K`
scores and drafting stops before a transition below the threshold.

### Lifecycle

1. **Initialize:** load and validate the external sidecar, read its target-layer,
   block, selector, RoPE, and attention metadata, create injection/noise batches,
   enable selected target input taps, and configure the resolved variant's
   output mode.
2. **Process prompt and begin:** each synchronized target decode calls
   `process()`, which gathers the configured target-layer rows and passes them
   to the fused encoder/KV-injection draft decode. After prompt evaluation,
   `begin()` verifies that this processing populated the complete draft prompt;
   later target decodes continue through the same `process()` path.
3. **Draft:** `draft()` captures a native or partial on-device checkpoint,
   decodes one anchor-plus-mask block, applies DFlash token probability,
   DFlash2 selector-transition probability, or DSpark confidence filtering,
   restores the transient noise branch, and retains the checkpoint for
   verification.
4. **Verify and commit:** the target verifies `[id_last, draft...]`. `accept()`
   clears temporary feature-row bookkeeping; rejection either removes the draft
   suffix natively or restores the checkpoint and replays only the sampled
   token plus accepted target features.
5. **Reset or close:** `clear()` empties request-local cache and sampler state;
   `close()` releases both batches, the draft context, sampler, and
   external sidecar model.

All variants currently require an external draft GGUF:

```python
llm = Llama(
    model_path="path/to/target.gguf",
    n_ctx=8192,
    n_batch=512,
    n_gpu_layers="all",
    speculative=SpecConfig(
        # DFlash2 uses DRAFT_DFLASH and is detected from selector metadata.
        spec_type=SpeculativeType.DRAFT_DFLASH,  # or DRAFT_DSPARK
        draft_model_path="path/to/dflash-dflash2-or-dspark.gguf",
        draft_n_max=7,
        draft_p_min=0.0,
        draft_n_gpu_layers="all",
        # Used by DFlash v1/DSpark; remains inactive for DFlash2.
        draft_backend_sampling=True,
    ),
)
```

Testing covers compatible Qwen3.6 DFlash and Qwen3.8 DSpark target/draft pairs.
DFlash2 has been validated with `Qwen3.8-27B-Q5_K_M.gguf` and the compatible
`Qwen3.8-27B-DFlash2-Q8_0.gguf` sidecar. Short greedy comparisons matched the
ordinary decoder's output tokens, and selector drafting, verification,
acceptance, and native rollback all completed successfully. Other GGUF pairs
may work when their vocabulary, target-layer metadata, selector layout, and
embedding dimensions are compatible, but they have not yet been validated.

The requested minimum and maximum draft lengths are clamped to the GGUF's
`dflash.block_size`.
DFlash and DFlash2 can produce at most `block_size - 1` tokens. DSpark can
produce `block_size` tokens when `dflash.sample_from_anchor=true`; otherwise it
also uses `block_size - 1`. For DFlash, `draft_p_min` filters the selected
top-token probability. For DFlash2, it filters the selected transition
probability within the current predecessor row. For DSpark, it filters the
model's predicted acceptance confidence.
When a DSpark sidecar declares `dflash.has_confidence_head=false`, probability
filtering requires `draft_p_min=0`.

The engine keeps verification blocks atomic and checks every draft-memory
removal. Transformer drafts and recurrent/hybrid drafts with enough native
snapshot slots use suffix removal; other recurrent/hybrid drafts use a partial
on-device checkpoint plus accepted-prefix replay. A failed native removal is a
fatal state-alignment error rather than a runtime fallback trigger.

For token or embedding target batches, a sidecar that declares M-RoPE receives
four plane-major positions `[text, text, text, 0]` in the fused injection batch,
matching the current `llama.cpp` DFlash driver. A target model using M-RoPE does
not by itself mark the draft sidecar as M-RoPE; inspect the resolved runtime
configuration instead of inferring it from the target architecture.

Target layer IDs come from the sidecar GGUF. Values from `0` through
`target_model.n_layer()` are valid: the inclusive final value denotes the final
head-input tap required by architectures such as Nemotron DFlash. Target rows
are copied into draft-`n_ubatch` chunks. The draft graph fuses its encoder with
KV injection in one decode, avoiding the previous device-to-host-to-device
round trip and the separate encoder graph build.

## N-gram Map Engines

`LlamaNGramMapDecoding` incrementally indexes verified token history and does not
load a draft model.

### K mode

`NGRAM_MAP_K` stores `n-gram key -> historical positions`. It drafts from the
latest valid match and stores acceptance feedback so a previously rejected
continuation is shortened on later attempts. In this mode, `ngram_min_hits` is
not used as a confidence gate, matching the current `llama.cpp` K behavior.

K mode retains all matching positions unless `ngram_max_entries_per_key` is set.
It generally has the highest recall and can benefit from long `M` values on
highly repetitive content.

### K4V mode

`NGRAM_MAP_K4V` stores `n-gram key -> fixed-size continuation values`. It:

1. keeps at most four recent continuations per key by default, matching
   `COMMON_NGRAM_MAX_VALUES` in `llama.cpp`;
2. chooses the most frequent continuation;
3. skips drafting unless the strongest continuation is at least twice as
   frequent as all alternatives combined; and
4. applies `ngram_min_hits` and previous acceptance feedback.

K4V is more selective and uses more token storage per key. Shorter continuation
lengths may work better because only complete M-token values are indexed.

### Direct construction

Direct construction is useful for custom engines and tests:

```python
from llama_cpp.llama_speculative import (
    LlamaNGramMapDecoding,
    SpeculativeType,
)

engine = LlamaNGramMapDecoding(
    ngram_size=8,
    num_pred_tokens=16,
    spec_type=SpeculativeType.NGRAM_MAP_K4V,
    min_hits=1,
    max_entries_per_key=4,
    sync_check_tokens=16,
)
```

The old string `mode="k"` / `mode="k4v"` argument is not supported. Pass the
corresponding `SpeculativeType` enum.

### Hybrid and recurrent targets

Transformer targets can usually discard rejected verification rows with native
KV removal. Hybrid or recurrent targets need checkpoint-backed rollback for
n-gram speculation:

```python
llm = Llama(
    model_path="path/to/hybrid-model.gguf",
    speculative=SpecConfig(
        spec_type=SpeculativeType.NGRAM_MAP_K,
        ngram_size_n=8,
        ngram_size_m=16,
    ),
    ctx_checkpoints=16,
    checkpoint_on_device=True,
)
```

On-device checkpoints avoid copying recurrent tensor payloads through host
memory. Checkpoint count and timing are exposed in `last_speculative_stats`.

## Engine Factories

### `create_spec_engine(config)`

Creates token-history-only engines that do not require initialized native model
resources. It currently creates K and K4V n-gram engines.

### `create_native_spec_engine(...)`

Creates engines that need an initialized target model/context. It currently
creates MTP, DFlash, and DSpark engines and may own an external draft
model/context. The word `native` describes those resource dependencies; the
lifecycle orchestration is still implemented in Python.

Most callers should not invoke either factory directly. `Llama` selects the
correct one from `SpecConfig`.

## Runtime Statistics

After generation, `Llama.last_speculative_stats` contains the most recent
request's metrics. Important keys include:

| Key | Meaning |
|---|---|
| `drafted`, `verified`, `accepted` | Proposal and verification token counts. |
| `begin_calls`, `draft_calls`, `process_calls`, `accept_calls` | Speculative lifecycle call counts. |
| `generated_drafts`, `accepted_drafts` | Draft-batch counts. |
| `draft_batch_acceptance_rate` | Draft batches with at least one accepted token divided by generated draft batches. |
| `accepted_draft_tokens` | Number of proposed draft tokens accepted by target verification. |
| `draft_token_acceptance_rate` | Accepted draft tokens divided by proposed tokens. |
| `mean_accepted_length` | Sampled token plus mean accepted draft prefix. |
| `acceptance_rate_per_position` | Acceptance probability at each draft position. |
| `begin_seconds`, `draft_seconds`, `accept_seconds` | Speculative engine lifecycle time outside target verification. |
| `target_decode_seconds` | Host time spent submitting target decode work. |
| `target_sync_seconds` | Time spent waiting for target decode/verification to complete. |
| `process_seconds` | Hidden-state processing and draft-context catch-up after the target synchronization boundary. |
| `checkpoint_captures`, `checkpoint_restores`, `checkpoint_verification_reuses` | Checkpoint operations and draft-time checkpoints reused for verification. |
| `checkpoint_native_captures`, `checkpoint_native_restores` | Native suffix-removal checkpoint operations. |
| `checkpoint_device_captures`, `checkpoint_device_restores` | Partial on-device state checkpoint operations. |
| `checkpoint_native_verification_rollbacks` | Draft contexts realigned directly after target native rollback. |
| `checkpoint_buffer_bytes` | Total serialized metadata buffer bytes allocated for on-device checkpoints. |
| `checkpoint_capture_seconds`, `checkpoint_restore_seconds` | Checkpoint overhead. |
| `verification_steps` | Target verification batches that contained draft tokens. |
| `rollbacks`, `native_rollbacks`, `checkpoint_rollbacks` | Target rollback paths. |
| `acceptance_rate` | Accepted draft tokens divided by target-verified draft tokens. |
| `generation_tokens`, `generation_seconds`, `generation_tokens_per_second` | Delivered-token throughput from speculative-phase start through the last token, including TTFT. |
| `time_to_first_token_seconds` | Time to first generated token. |
| `sustained_tokens`, `sustained_seconds`, `sustained_tokens_per_second` | Throughput after the first token, excluding TTFT. |

`decode_tokens`, `decode_seconds`, and `decode_tokens_per_second` remain aliases
for the corresponding generation measurements.

With `verbose=True`, the same information is printed in a multi-line summary at
the end of `Llama.generate`.

## Benchmarking and Tuning

MTP and n-gram draft lengths are independent parameters. The general benchmark
CLI uses `--mtp-draft-tokens` for MTP and `--ngram-draft-tokens` for n-gram
methods. DFlash, DFlash2, and DSpark use `--draft-tokens` in their dedicated
example.

```bash
# Scan N={6,8,10,12} x M={8,16,32,48} for K and K4V
python -m examples.benchmark.benchmark_speculative \
  --model model.gguf \
  --methods ngram-k ngram-k4v \
  --ngram-grid

# Compare ordinary, built-in MTP, and external MTP
python -m examples.high_level_api.high_level_api_mtp_speculative \
  --model target.gguf \
  --mtp-mode both \
  --draft-model mtp.gguf \
  --draft-tokens 2

# Compare ordinary decoding with external DFlash2
python -m examples.high_level_api.high_level_api_dflash_dspark_speculative \
  --algorithm dflash2 \
  --model target.gguf \
  --draft-model dflash2.gguf \
  --draft-tokens 7 \
  --draft-p-min 0 \
  --temperature 0 \
  --ignore-eos
```

N-gram speedups are workload-sensitive. Repetitive structured output can benefit
substantially, while low-repetition prose can be neutral or slower. Model-backed
acceptance also depends on the target, draft tensors, sampling settings, prompt,
and configured draft length. Measure TTFT, sustained speed, acceptance by
position, target decode/sync time, and rollback cost together. For DFlash v1,
DSpark, and other vocabulary-sampling engines, keep backend sampling enabled
unless profiling shows otherwise. DFlash2 always selects from its compact
selector lattice and therefore reports backend sampling as inactive.

## Limitations and Lifecycle Notes

* MTP and n-gram engines are text-only. DFlash-family engines accept target
  embedding batches, but end-to-end multimodal model coverage is still limited.
* Current engines support only `seq_id=0`; parallel sequence decoding is not yet
  supported.
* Speculative resets clear target and engine state together. Public prompt-cache
  state does not currently serialize the speculative engine's context.
* Speculation still runs target verification. Low acceptance or expensive
  rollback can make it slower than ordinary decoding.
* Greedy runs can diverge from ordinary output because different verification
  batch shapes may change floating-point tie-breaking. Benchmarks report the
  first divergent token as a diagnostic rather than hiding it.
* Explicitly call `Llama.close()` in long-running applications so external draft
  resources are released before interpreter shutdown.

## Deprecated APIs

`LlamaDraftModel` and `Llama(draft_model=...)` remain for legacy stateless
callbacks. They do not use the stateful engine lifecycle, native recurrent
rollback, phase statistics, or MTP resource management. New code should use
`SpecConfig`.

`LlamaPromptLookupDecoding` is no longer part of this module. Use
`NGRAM_MAP_K` or `NGRAM_MAP_K4V` instead.

## Related Links

* [Index/Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)
* [Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)
* [MTP example](https://github.com/JamePeng/llama-cpp-python/blob/main/examples/high_level_api/high_level_api_mtp_speculative.py)
* [DFlash/DFlash2/DSpark example](https://github.com/JamePeng/llama-cpp-python/blob/main/examples/high_level_api/high_level_api_dflash_dspark_speculative.py)
* [DFlash2 walkthrough](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/examples/dflash2-speculative-decoding.md)
* [DFlash2 overview](https://inco.ai/blog/dflash2/)
* [Speculative benchmark](https://github.com/JamePeng/llama-cpp-python/blob/main/examples/benchmark/benchmark_speculative.py)
