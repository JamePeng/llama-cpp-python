---
title: Llama Speculative Decoding
module_name: llama_cpp.llama_speculative
source_file: llama_cpp/llama_speculative.py
last_updated: 2026-08-19
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

The current engines are text-only and support one sequence (`seq_id=0`).

## Implementation Status

`SpeculativeType` mirrors `common_speculative_type` from `llama.cpp`, so the enum
contains algorithms that do not yet have Python engines.

| Type | Python engine | Status |
|---|---|---|
| `DRAFT_MTP` | `LlamaMTPDecoding` | Implemented for built-in and external MTP |
| `NGRAM_MAP_K` | `LlamaNGramMapDecoding` | Implemented |
| `NGRAM_MAP_K4V` | `LlamaNGramMapDecoding` | Implemented |
| `DRAFT_EAGLE3` | none | Declared, not implemented |
| `DRAFT_DFLASH` | none | Declared, not implemented |
| `DRAFT_DSPARK` | none | Declared, not implemented |
| `DRAFT_SIMPLE` | none | Declared, not implemented |
| `NGRAM_SIMPLE` | none | Declared, not implemented |
| `NGRAM_MOD` | none | Declared, not implemented |
| `NGRAM_CACHE` | none | Declared, not implemented |

Selecting an unimplemented type raises `NotImplementedError` during validation or
engine creation. Eagle3, DFlash, and DSpark also require `draft_model_path`.

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
| `is_mtp()` | True only for `DRAFT_MTP`. |
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
| `draft_p_min` | `0.0` | Minimum probability retained by MTP drafting. |
| `draft_model_path` | `None` | External draft GGUF. Omit it for built-in target MTP heads. |
| `draft_backend_sampling` | `True` | Let the backend select MTP candidates where supported. |

### External draft runtime settings

| Field | Default | Description |
|---|---:|---|
| `draft_n_gpu_layers` | `"auto"` | Draft offload setting: integer, `"auto"`, or `"all"`. |
| `draft_n_threads` | `None` | Draft generation thread count. |
| `draft_n_threads_batch` | `None` | Draft prompt/batch thread count. |
| `draft_cpu_moe` | `False` | Keep all draft MoE expert tensors on CPU. |
| `draft_n_cpu_moe` | `0` | Keep the first N draft MoE layers on CPU. |
| `draft_devices` | `[]` | Ordered backend device names for the draft model. |
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
| `process(batch, seq_id=0)` | Consume target tokens and NextN hidden rows after a successful decode. |
| `draft(input_ids, n_past, id_last, n_max, seq_id=0)` | Return only continuation tokens, never `id_last`. |
| `accept(n_accepted, seq_id=0)` | Commit acceptance feedback for the last proposal. |
| `checkpoint(seq_id=0)` | Capture opaque draft-side state before verification. |
| `take_verification_checkpoint(seq_id=0)` | Reuse a draft-time checkpoint when possible. |
| `restore(checkpoint, seq_id=0)` | Restore rejected draft-side state. |
| `rollback_verified(checkpoint, n_accepted, seq_id=0)` | Keep the sampled token and accepted prefix after native target rollback. |
| `truncate(position, seq_id=0)` | Remove state at and after an absolute position. |
| `clear()` | Clear request state but keep reusable model resources. |
| `close()` | Release owned resources. Calls are expected to be idempotent. |
| `checkpoint_stats()` | Return per-request capture and restore metrics. |

The target model and target context always remain owned by `Llama`.

## MTP Engine

`LlamaMTPDecoding` orchestrates the native NextN/MTP graph while participating in
the same `LlamaSpecEngine` lifecycle.

The built-in and external MTP paths have currently been tested only with the
Qwen3.5, Qwen3.6, and Qwen3.8 model families. Other model families may work
when their GGUF tensors are compatible, but they have not yet been validated.

The engine reads target hidden-state rows, sizes them with the models'
`n_embd_out`, advances a dedicated MTP context, and uses recurrent snapshots to
discard rejected speculative branches. Backend sampling avoids copying a full
vocabulary-sized logits row to Python when the backend exposes compact candidate
buffers. This is especially important for large vocabularies.

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

The external model owns a separate model and context. Initialization verifies
vocabulary type, vocabulary size/token compatibility, and `n_embd_out`. Multiple
NextN layers are chained when the model exposes more than one MTP head.

For Qwen3.8 27B, testing so far suggests `draft_n_max=2` as the best starting
point. This is not a universal optimum: GPU, backend, quantization, prompt,
sampling settings, and whether MTP is built in or external can change the
result. Run `examples.benchmark.benchmark_speculative` in the intended
deployment environment and choose the fastest stable value.

Larger `draft_n_max` values increase the verification batch and rollback
exposure and are only useful when later-position acceptance remains high.

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
creates MTP engines and may also own an external draft model/context. The word
`native` describes those resource dependencies; the lifecycle orchestration is
still implemented in Python.

Most callers should not invoke either factory directly. `Llama` selects the
correct one from `SpecConfig`.

## Runtime Statistics

After generation, `Llama.last_speculative_stats` contains the most recent
request's metrics. Important keys include:

| Key | Meaning |
|---|---|
| `drafted`, `verified`, `accepted` | Proposal and verification token counts. |
| `generated_drafts`, `accepted_drafts` | Draft-batch counts. |
| `draft_token_acceptance_rate` | Accepted draft tokens divided by proposed tokens. |
| `mean_accepted_length` | Sampled token plus mean accepted draft prefix. |
| `acceptance_rate_per_position` | Acceptance probability at each draft position. |
| `begin_seconds`, `draft_seconds`, `process_seconds`, `accept_seconds` | Engine phase time. |
| `checkpoint_captures`, `checkpoint_restores` | Checkpoint operations. |
| `checkpoint_capture_seconds`, `checkpoint_restore_seconds` | Checkpoint overhead. |
| `rollbacks`, `native_rollbacks`, `checkpoint_rollbacks` | Target rollback paths. |
| `generation_tokens_per_second` | Delivered-token throughput from speculative-phase start through the last token, including TTFT. |
| `time_to_first_token_seconds` | Time to first generated token. |
| `sustained_tokens_per_second` | Throughput after the first token, excluding TTFT. |

With `verbose=True`, the same information is printed in a multi-line summary at
the end of `Llama.generate`.

## Benchmarking and Tuning

MTP and n-gram draft lengths are independent parameters. The benchmark CLI uses
`--mtp-draft-tokens` for MTP and `--ngram-draft-tokens` for n-gram methods.

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
```

N-gram speedups are workload-sensitive. Repetitive structured output can benefit
substantially, while low-repetition prose can be neutral or slower. MTP acceptance
also depends on the target, draft tensors, sampling settings, and prompt. Measure
TTFT, sustained speed, acceptance by position, and rollback cost together.

## Limitations and Lifecycle Notes

* Current stateful engines are text-only. Do not use them with MTMD/multimodal
  embedding batches or negative placeholder token IDs.
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

* [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
* [[MTP example](https://github.com/JamePeng/llama-cpp-python/blob/main/examples/high_level_api/high_level_api_mtp_speculative.py)]
* [[Speculative benchmark](https://github.com/JamePeng/llama-cpp-python/blob/main/examples/benchmark/benchmark_speculative.py)]
