---
title: Llama Batch
module_name: llama_cpp._internals
source_file: llama_cpp/_internals.py
last_updated: 2026-10-13
version_target: "latest"
---

# Llama Batch

`LlamaBatch` and `LlamaBatchExt` prepare inputs for `LlamaContext.encode()` and
`decode()`. These are internal wrappers imported from `llama_cpp._internals`;
ordinary applications should use [Llama](../core/Llama.md),
[LlamaEmbedding](LlamaEmbedding.md), or the multimodal handlers.

## Choosing a builder

| Builder | Representation | Use |
|---|---|---|
| `LlamaBatch` | Fixed-capacity parallel arrays in a native `llama_batch`. | Existing token, embedding, and specialized mixed-input paths. |
| `LlamaBatchExt` | Logical entries rendered into a context-bound native `llama_batch_ext`. | Explicit per-entry tokens, external features, positions, sequence membership, and outputs. |

The extended builder separates input construction from submission. Each entry
can carry a token ID, an embedding, or both. This makes heterogeneous prompts
easier to represent, but interleaving still requires model/context support.
It does not encode images or audio, configure speculative decoding, or add new
positional dimensions to a model.

## Native conversion paths

Both public native entry points converge on extended inputs:

```text
LlamaBatch    -> llama_encode/decode -> llama_batch_compat -> llama_batch_ext
LlamaBatchExt -> render              -> llama_process     -> llama_batch_ext
                                                         -> encode/decode
                                                         -> llama_batch_allocr
                                                         -> llama_ubatch -> graph
```

Python chooses the entry point from the supplied object's type. There is no
automatic builder selection based on the model or media. Existing `Llama` text
and text-embedding paths still use `LlamaBatch`; `MTMDEmbeddingHandler` explicitly
builds `LlamaBatchExt` for encoded text/media chunks.

The legacy compatibility adapter constructs a temporary extended batch for each
call and releases it afterward. It preserves explicit sequence IDs, positions,
and output flags. Missing sequence membership defaults to sequence `0`; missing
positions continue from context memory; a missing logits array requests only the
last input's output. Extended inputs supply these fields explicitly and default
to no outputs.

`llama_batch_allocr` validates and flattens extended entries into internal arrays,
then builds physical micro-batches. It converts entry-major positions to native
section-major layout; scalar text positions expand to `[p, p, p, 0]` for four-axis
positions. Token/feature interleaving requires an architecture that supports
mixed batches and a default context. Paired token/feature entries cannot share
a batch with unpaired entries, and state features must be present on all entries
or none.

For MTP contexts accepting state features, the adapter interprets legacy batches
with both `token` and `embd` as token IDs plus **state embeddings**. Python's
`add_token_embedding()` sets a **token embedding**, while `state=` supplies the
separate **state embedding**. Use `add_token(token, position, state=hidden)` to
replace a legacy MTP token/state row, or `add_embedding(features, position,
state=hidden)` for an image feature/state row. Extended
submission avoids the legacy adapter but still performs rendering and internal
flattening; it does not guarantee fewer copies or faster inference.

## Construction and insertion

Constructor signatures:

```text
LlamaBatchExt(*, context: LlamaContext)
LlamaBatch(*, n_tokens: int, embd: int, n_seq_max: int,
           mixed: bool = False, verbose: bool = True)
```

For `LlamaBatchExt`, `context` must be an open owning context. It supplies the
model, vocabulary, position layout, and batch/sequence limits. The builder keeps
it alive and can only be submitted to that context.

For `LlamaBatch`, `n_tokens` is allocated entry capacity, `embd=0` selects token
storage, and positive `embd` allocates that many floats per entry. `n_seq_max`
sets sequence capacity; `mixed=True` adds token storage to an embedding batch
for specialized callers. Token insertion uses `add_token()` or `add_sequence()`;
embedding and mixed inputs require the corresponding specialized methods.

`LlamaBatchExt` insertion signatures:

```text
add_token(token, position, seq_ids=(0,), output=False, *, decision_order=0, state=None)
add_tokens(tokens, *, positions, seq_ids=(0,), outputs=None, states=None)
add_embedding(embedding, position, seq_ids=(0,), output=False, *, state=None)
add_embeddings(embeddings, *, positions, seq_ids=(0,), outputs=None, states=None)
add_token_embedding(token, embedding, position, seq_ids=(0,), output=False, *, state=None)
```

| Input | Type and meaning |
|---|---|
| `token` / `tokens` | Vocabulary integer ID / sequence of IDs, resolved through the model's embedding table. |
| `embedding` | Array-like vector or matrix forming one entry's feature payload. |
| `embeddings` | Matrix `[entries, width]`; each row becomes one entry. |
| `state` / `states` | Separate hidden state: vector or matrix for one entry, or matrix `[entries, state_width]` for bulk insertion. Copied into owned storage; currently accepted by native MTP contexts. |
| `position` / `positions` | Nonnegative int32 model position / one position per entry. Token and paired entries use scalars; embedding-only entries use four coordinates for M-RoPE/IM-RoPE, otherwise scalars. Bulk coordinates are entry-major: `[entries, 4]`. |
| `seq_ids` | Nonempty sequence of integer IDs in `[0, context.n_seq_max())`. Multiple IDs share the same input and position; bulk calls apply the same IDs to every entry. |
| `output` / `outputs` | Boolean flag / one flag per entry. Defaults request no outputs. |
| `decision_order` | Native decision-head metadata for `add_token()`; `0` is ordinary text. Use a defined `llama_decision_order` value when the model requires it. |

Single-entry methods return an integer logical index. Bulk methods return a tuple
of logical indices and validate the complete span before appending it.

Use single-entry calls for differing metadata; bulk calls are convenient for
contiguous text or feature spans. `add_token_embedding()` attaches both a token
ID and external features to one entry. A submitted range containing such paired
entries must contain only paired entries. All embeddings in a range must have
the same flattened width, and native validation checks it against the context.
States are rendered through `llama_batch_ext_set_embd_state()` independently of
token embeddings. Every entry in a submitted range must carry a state, or none
may carry one. State widths must agree within that range and are checked
separately from input embedding widths. The native context validates the exact
width and whether states are supported; a rejected state raises `RuntimeError`
and clears the partial native batch so the logical inputs can be retried.
Bulk states are copied once and shared by read-only row views.

For an MTP context, text and image catch-up inputs can be prepared as separate
submissions (support for interleaving in one submission depends on the context):

```python
with LlamaBatchExt(context=draft_context) as batch:
    batch.add_token(token_id, position, state=previous_hidden, output=True)
    draft_context.decode(batch)
    batch.reset()
    batch.add_embeddings(image_features, positions=image_positions, states=shifted_hidden)
    draft_context.decode(batch)
```

Input feature width and hidden-state width may differ. The caller supplies the
correctly shifted target states and consumes outputs before the next decode;
the builder does not compute states or enable MTP in the high-level MTMD handler.

## Submission and outputs

`len(batch)` and `batch.n_tokens()` count logical entries, including features.
`capacity()` is the maximum entries per submission: `context.n_batch()`.
The logical collection may exceed that limit. Submit nonempty ranges explicitly:

```python
context.decode(batch, start=0, count=None)
context.encode(batch, start=0, count=None)
context.process(batch, process_type, start=0, count=None)
```

`count=None` selects all remaining entries. There is no automatic splitting;
llama.cpp performs physical micro-batching within each submitted range using
`n_ubatch`. Encoder and non-causal embedding inputs can require a complete
sequence in one physical batch, so arbitrary splitting is not valid for every
model. `decode()` returns `0` on success or `1` when no KV slot is available;
the caller handles retries. An abort raises `LlamaDecodeAbort`; fatal errors
raise `RuntimeError`. `encode()` raises `RuntimeError` for any nonzero status.
`process()` dispatches to these same semantics.

Context output limits are separate from input capacity: `n_outputs_max` controls
outputs per physical batch, and `n_outputs_max_per_seq` limits outputs for a
sequence with an attached backend sampler. Flag only the outputs you need.

After rendering, `submitted_indices` identifies the logical entries in that
range, and `output_indices` identifies those explicitly flagged for output.
They describe the rendered input, not proof of successful processing. For native
output accessors, a nonnegative index is relative to the submitted range's input
entries; negative indices count backward through produced output rows. For
example, logical index `10` in a range starting at `8` is accessed with index
`2`, if that entry requests output. Pooling can instead produce sequence outputs.

Read results through the context before its next processing call. Accessors such
as `get_logits_ith()` synchronize; call `synchronize()` explicitly when measuring
completed device execution. With backend sampling, `set_sampler()` success means
attachment succeeded, and `get_sampled_token_ith()` may still return
`LLAMA_TOKEN_NULL` when the backend did not select a token.

This standalone example requires a decoder GGUF model at `./model.gguf`:

```python
from contextlib import closing

import numpy as np
from llama_cpp import Llama
from llama_cpp._internals import LlamaBatchExt

with closing(Llama(model_path="./model.gguf", n_ctx=128, n_batch=32)) as llm:
    tokens = llm.tokenize(b"Hello", add_bos=True)
    with LlamaBatchExt(context=llm._ctx) as batch:
        batch.add_tokens(
            tokens, positions=range(len(tokens)),
            outputs=[False] * (len(tokens) - 1) + [True],
        )
        for start in range(0, len(batch), batch.capacity()):
            count = min(batch.capacity(), len(batch) - start)
            if llm._ctx.decode(batch, start=start, count=count) != 0:
                raise RuntimeError("No KV slot available")
            if batch.output_indices:
                logits = np.ctypeslib.as_array(
                    llm._ctx.get_logits_ith(-1), shape=(llm.n_vocab(),),
                )
                print("Next token ID:", int(logits.argmax()))
```

Direct context submissions do not update `Llama`'s Python token ledger. Keep a
manual inference workflow consistent, or reset the high-level model before
returning to its completion APIs.

## Ownership and reuse

Embedding insertion copies into owned, contiguous float32 arrays. Bulk insertion
copies the matrix once and shares its rows between entries; rendering copies
features into native storage. This is not a zero-copy input path. Retaining a
large logical collection retains its feature memory even when submitting small
ranges.

`entries` returns a tuple snapshot; treat its feature arrays as read-only.
`render(context=None, *, start=0, count=None)` prepares input without inference
and returns a borrowed native pointer valid until the next render, reset, or
close. Prefer context submission methods to direct pointer handling.

`reset()` (also `clear()`) drops logical inputs and index mappings while retaining
native capacity for reuse. It does not clear context memory or KV state.
`close()` releases native storage, inputs, and the context reference, and is safe
to repeat. Closing the context also closes its extended batches. Use
`with LlamaBatchExt(...)` for deterministic cleanup; the builder and context
must not be used concurrently.

For native signatures, see [llama.cpp bindings](LlamaCppBindings.md) and
[`llama.h`](../../../vendor/llama.cpp/include/llama.h). For media encoding and
speculation workflows, see [embeddings](../features/embeddings-rerank.md) and
[speculative decoding](LlamaSpeculative.md).
