# Caching and state reuse

For repeated text requests, pass the full updated prompt to `create_completion()`.
With speculation disabled, `Llama` automatically reuses a matching live prefix.
An exact prefix extension evaluates only the new suffix; changing earlier tokens
requires truncation, checkpoint restoration, or a fresh prefill.

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

RAM and disk caches select the stored key with the longest nonempty common
prefix, which may diverge later in the query. Trie lookup requires a complete
stored key to be a prefix of the query. For example, with only `(1, 2, 3)`
cached, a query for `(1, 2, 4)` can hit RAM/disk but misses the trie. A cache hit
returns the stored snapshot; generation must still reconcile the matching prefix.

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
reuse. Supported MTMD handlers have a separate prefill handoff for n-gram engines.
See [speculative lifecycle](../modules/LlamaSpeculative.md#prefix-cache-reuse-and-reset).

Set `ctx_checkpoints=0` only when the workflow does not need checkpoint-backed
rollback. Keep it positive for n-gram speculation on a hybrid target. Checkpoint
sizes reported by Python exclude context-owned device tensor storage.

Wrapped native operations notify registered caches on the same context. Raw C
calls, automatic SWA eviction, and cross-context shared-KV changes are outside
that notification mechanism; do not assume arbitrary low-level mutations are safe.
