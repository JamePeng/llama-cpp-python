# Caching and state reuse

For repeated text requests, pass the full updated prompt to `create_completion()`.
With speculation disabled, `Llama` automatically reuses a matching live prefix.
An exact prefix extension evaluates only the new suffix; changing earlier tokens
requires truncation, checkpoint restoration, or a fresh prefill.

`Llama.cache` defaults to `None`. Live native KV reuse and Hybrid checkpoints
work independently of the optional RAM, disk, and trie snapshot caches below.
For an existing `llm`, attach an optional snapshot cache explicitly:

```python
from llama_cpp.llama_cache import LlamaRAMCache

llm.set_cache(LlamaRAMCache(capacity_bytes=1 << 30))
```

RAM, disk, and trie caches select the longest nonempty common token prefix,
including a shorter request or a branch from an older history. Equal matches
prefer the first inserted entry. RAM and trie reads update LRU recency without
changing this tie; disk eviction follows its configured diskcache policy.
Built-in caches support `get_matching_state(tokens, min_prefix_length=...)` to reject a
candidate before reading its snapshot when the live context matches as well.
Completion cache keys contain only tokens already decoded into native memory.

## Choose the state you need

| Mechanism | What it preserves | Lifetime and limits |
|---|---|---|
| Live context prefix | Current native memory and Python token history | Belongs to one `Llama` instance; reset discards it |
| `LlamaState` | Complete native export plus owned token/output data | Independent host snapshot; compatibility checks apply; no sampler or draft continuation |
| `LlamaRAMCache`, `LlamaDiskCache`, `LlamaTrieCache` | Token-prefix mappings to `LlamaState` objects | External cache storage; configuring it does not remove native restore requirements |
| `HybridCheckpointCache` | Partial recurrent/SWA state and native position metadata | Depends on a valid live attention prefix; invalidation or eviction makes old objects unusable |
| Engine draft checkpoint | Temporary MTP/DFlash verification state | Engine-local and capture-local; not a persistent prompt cache |

See [Llama state methods](../core/Llama.md#save_state-and-load_statestate) and
[cache APIs](../modules/LlamaCache.md) for the detailed contracts.

For example, with only `(1, 2, 3)` cached, a query for `(1, 2, 4)` hits all
three backends with a common-prefix length of two. A cache hit returns the
stored snapshot; generation must still reconcile the matching prefix.

## Saving a text prefix

This example requires a compatible text GGUF and runs without speculation.
It restores the same evaluated prefix before evaluating each alternative suffix.
The sample is one token, not a complete answer.

```python
from llama_cpp import Llama

llm = Llama(model_path="./model.gguf", n_ctx=512, verbose=False)
try:
    llm.eval(llm.tokenize(b"The capital of"))
    prefix = llm.save_state()
    for suffix in (b" France is", b" Japan is"):
        llm.load_state(prefix)
        llm.eval(llm.tokenize(suffix, add_bos=False))
        token = llm.sample(temp=0)
        print(llm.detokenize([token]).decode("utf-8", errors="replace"))
finally:
    llm.close()
```

Use `save_state()` rather than constructing `LlamaState` by hand. The snapshot
owns its arrays and bytes, but does not save sampler history or draft features.
A target snapshot loaded with a speculative engine requires a new full-prompt
request with `reset=True`; it cannot resume that engine through `reset=False`.

## Hybrid checkpoints and cache misses

`ctx_checkpoints` limits the number of partial snapshots. Host storage retains
multiple historical payloads; `checkpoint_on_device=True` uses a context-owned
slot per sequence, so a new capture invalidates the previous device snapshot for
that sequence. Raising the entry limit does not create more native device slots.

A saved checkpoint is useful only if its token prefix matches, its object is
still registered, and the required native prefix remains valid. `clear()`,
eviction, context mutation, or closure can invalidate it even when application
code still holds the object. A failed restore may have changed native state;
the high-level generation path clears and rebuilds when it cannot reuse it.

Fresh text requests with a speculative engine reset target and engine together.
Their within-request rollback checkpoints do not imply cross-request prompt
reuse. Automatic completion snapshot reads and writes are skipped for these
engines. A full hybrid snapshot can be reused by appending to its complete prefix
or by sampling its owned final logits; it does not restore historical checkpoints.
Speculative requests also skip the automatic target-only final checkpoint;
verification still captures target checkpoints when native rollback cannot be used.
Supported MTMD handlers have a separate prefill handoff for MTP and n-gram
engines. Qwen3.5 MTP image requests rebuild both contexts before media prefill;
the complete prompt must fit without context shifting. See
[MTP image chat](../examples/vision/vision-qwen.md#mtp-image-chat) for model requirements.
See [speculative lifecycle](../modules/LlamaSpeculative.md#prefix-cache-reuse-and-reset).

Set `ctx_checkpoints=0` to disable Python prompt checkpoints. Keep it positive
for n-gram speculation on a hybrid target. Checkpoint
sizes reported by Python exclude context-owned device tensor storage.
MTP can use zero Python checkpoints when the target has native recurrent rollback
slots and the engine can follow target rollback. The draft context may use native
rollback or device checkpoint restoration and replay. Verification length is
capped by the target's native rollback window; native recurrent snapshots and
temporary draft checkpoint metadata remain active.
Zero Python checkpoints avoid target checkpoint exports and history searches.
Appending to the live prefix still evaluates only the suffix, but branching or
shortening a Hybrid history requires a full prefill without a usable checkpoint.
With checkpoints disabled, an identical prompt bypasses evaluation only when
the final output is valid. Missing output requires a fresh text prefill. Score
callbacks receive refreshed logits even when evaluation is skipped. Custom stop
criteria require logits by default; `requires_logits=False` opts out of that
copy when the callback does not inspect scores, as with `AbortCriteria`.
During speculative verification, score callbacks receive the current sampling
row and its decoded prefix, excluding unaccepted future draft tokens.

Wrapped native operations notify registered caches on the same context. Raw C
calls, automatic SWA eviction, and cross-context shared-KV changes are outside
that notification mechanism; do not assume arbitrary low-level mutations are safe.

## Quantization and batch shapes

Evaluating a full prompt in one batch can produce different logits from evaluating
its prefix and suffix separately. Backend kernels and reduced precision, including
quantized GGUF weights such as Q4_K_M, can change numerical rounding. When leading
logits are close, the next greedy token can change even with `temperature=0` and a
fixed seed. Speculative verification also changes batch shapes. FP16/BF16 weights
do not guarantee identical results across batch shapes or backends.

Different generated text alone does not establish a cache or rollback error.
To isolate state reuse, compare against an independent native context using the
same model, runtime settings, batch boundaries, and requested output rows. Compare
the logits as well as token history and native positions; separately check that
only the intended suffix was evaluated to distinguish numerical differences from
incorrect restoration.
