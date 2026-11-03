# Stateful Speculative Decoding

Use `Llama(speculative=SpecConfig(...))` to propose draft tokens and verify them
with the target model. The target still determines the returned tokens; the
benefit depends on how much verification work the accepted draft prefix saves.

Choose a source compatible with the target:

| Source | Configuration | Requirements |
|---|---|---|
| Built-in MTP | `DRAFT_MTP`, without `draft_model_path` | Target GGUF with supported NextN/MTP heads. |
| External MTP | `DRAFT_MTP`, with `draft_model_path` | Compatible MTP sidecar and target. |
| DFlash / DFlash2 / LiliCorr | `DRAFT_DFLASH` | Compatible external GGUF; metadata selects the variant. |
| DSpark | `DRAFT_DSPARK` | Compatible external GGUF with Markov/confidence heads. |
| N-gram | `NGRAM_MAP_K` or `NGRAM_MAP_K4V` | Repeated continuations in verified history; no draft model. |

Current engines support one sequence, `seq_id=0`. Keep
`n_batch >= 1 + draft_n_max` so a model-backed proposal fits one target
verification batch. For n-gram engines, use `ngram_size_m` as the proposal limit
instead. See the [API reference](../modules/LlamaSpeculative.md) for rollback
limits, supported models, MTP image chat, and engine lifecycle.

## Run a DSpark sidecar

This complete example requires an installed package with a working native
library, a compatible target and DSpark GGUF pair, and enough GPU memory for
both models and their contexts. Replace the portable paths with your local
files. Use `DRAFT_DFLASH` for a DFlash-family sidecar without DSpark heads.

```python
from contextlib import closing

from llama_cpp import Llama
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

with closing(Llama(
    model_path="path/to/target.gguf",
    n_ctx=1024,
    n_batch=128,
    n_ubatch=64,
    n_gpu_layers="all",
    ctx_checkpoints=0,
    speculative=SpecConfig(
        spec_type=SpeculativeType.DRAFT_DSPARK,
        draft_model_path="path/to/dspark.gguf",
        draft_n_max=7,
        draft_n_min=0,
        draft_p_min=0.0,
        draft_n_gpu_layers="all",
        draft_backend_sampling=True,
    ),
    verbose=True,
)) as llm:
    response = llm.create_completion(
        prompt="Explain how speculative decoding verifies a draft block.\n",
        temperature=0.0,
        seed=42,
        max_tokens=256,
    )
    print(response["choices"][0]["text"])
    stats = llm.last_speculative_stats
    print("accepted draft tokens:", stats["accepted_draft_tokens"])
    print("draft acceptance:", stats["draft_token_acceptance_rate"])
    print("feature processing seconds:", stats["process_seconds"])
```

`ctx_checkpoints=0` disables periodic target prompt checkpoints. Native
speculative rollback still operates within the target's available rollback
window. It does not disable all checkpoints or make draft-state restoration
unnecessary.

The engine clamps the draft length to the sidecar's trained block size.
DFlash2 and LiliCorr read selector lattices and do not activate the backend
vocabulary sampler. For DSpark, `draft_p_min` filters predicted acceptance
confidence; for MTP/DFlash it filters token probability, and for DFlash2/LiliCorr
it filters the selected transition. Start at `0` and measure the effect of
raising it on total throughput, as well as acceptance.

## Compare host and device feature transfer

In the current source, compatible DFlash, DFlash2, and DSpark contexts can pass
target features directly through a native device buffer. With `verbose=True`,
startup output reports `target_features=device` or `target_features=host`.
Set `LLAMA_DFLASH_DEV=0` before model construction to force host copies; `1` or
an unset variable allows automatic device transfer. See the
[selection and fallback conditions](../modules/LlamaSpeculative.md#device-feature-transfer).

Use the dedicated benchmark with identical arguments for each path. For example,
in PowerShell, replace the model paths and run:

```powershell
$benchmarkArgs = @(
    "-m", "examples.high_level_api.high_level_api_dflash_dspark_speculative",
    "--algorithm", "dspark",
    "--model", "path/to/target.gguf",
    "--draft-model", "path/to/dspark.gguf",
    "--prompt", "Write a detailed explanation of speculative decoding with examples.",
    "--n-ctx", "1024", "--n-batch", "128", "--n-ubatch", "64",
    "--draft-tokens", "7", "--draft-p-min", "0",
    "--warmup-tokens", "64", "--max-tokens", "256", "--runs", "3",
    "--temperature", "0", "--ignore-eos", "--verbose"
)
$env:LLAMA_DFLASH_DEV = "0"
python @benchmarkArgs
$env:LLAMA_DFLASH_DEV = "1"
python @benchmarkArgs
```

For DFlash or DFlash2, change `--algorithm` and the model paths. The example
also measures ordinary decoding on each invocation. Its printed `speedup`
compares speculative decoding with ordinary decoding; to compare feature
transfer, compare the speculative mode's aggregate token/s across the two
invocations. Check startup output to confirm that the second invocation
actually enabled device transfer.

Repeat or alternate measurement order to reduce effects from warmup, clocks,
and other workloads. Keep prompts, output lengths, sampling, and draft settings
constant. A lower `process_seconds` can coexist with unchanged overall
throughput, and a higher acceptance fraction can coexist with slower decoding.
Compare output tokens and rollback behavior as well as speed. Quantized models
can produce different greedy output when ordinary and speculative decoding use
different batch shapes; see [quantization and batch shapes](../features/caching.md#quantization-and-batch-shapes).

## State and reuse

Fresh speculative text requests reset the target and draft together. Public
`LlamaState` snapshots do not include the draft context or MTP's pending hidden
row. After `load_state()`, start speculative generation with `reset=True` and
the full text prompt; `reset=False` is rejected for that restored state.
Transient checkpoints used within generation are separate from public snapshots
and cross-request prompt caching.

High-level DFlash-family chat remains text-only. For compatible MTP image
requests, use the [Qwen image-chat workflow](vision/vision-qwen.md#mtp-image-chat).
For selector-specific setup, see [DFlash2 speculative decoding](dflash2-speculative-decoding.md).
