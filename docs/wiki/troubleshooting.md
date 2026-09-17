# Runtime state troubleshooting

## A saved checkpoint does not produce a cache hit

Check whether the request still matches the saved token prefix, whether the
checkpoint remains in the cache, and whether its native prefix is still valid.
Host payloads are partial snapshots; retaining their Python objects is not enough.
Device mode keeps one active native snapshot per sequence, so a later capture can
replace the one needed by an earlier prompt. Fresh speculative text requests
reset target and engine together and do not perform ordinary cross-request prefix
reuse. See [caching and state reuse](features/caching.md).

## A state or checkpoint cannot be restored

`LlamaState` restore checks model/context compatibility and payload shape before
native mutation. Use a snapshot from a compatible model file and configuration.
Do not remove metadata to bypass the check. An ordinary exception during restoration
resets the high-level model; resend the complete prompt.
Direct `load_state()` does not catch `KeyboardInterrupt`.

Partial checkpoints must still belong to the live cache. MTP/DFlash draft
checkpoints must belong to the current engine capture. After clear, closure,
eviction, or a newer draft capture, acquire a fresh checkpoint rather than retrying
an old object. Target snapshots do not restore the draft engine; see
[state restoration](core/Llama.md#save_state-and-load_statestate).

## Generation was interrupted

An in-loop `KeyboardInterrupt` becomes a cancelled generation with
`finish_reason="abort"`; `verbose=True` emits a diagnostic on stderr. A native
mid-decode abort also clears partial state. Already delivered stream text remains
available, but continuation requires a new prompt evaluation. Direct low-level
calls and media prefill can still propagate exceptions; see
[cancellation](core/Llama.md#abort) and [media recovery](examples/vision/vision-qwen.md#reuse-and-failure-recovery).

## Image input fails or cannot fit

Use a projector matching the model and a valid image payload. If a media chunk
cannot fit alongside retained context, reduce the image token budget or increase
context capacity. The handler will not advance its ledger after failed native
removal or accept an out-of-range helper position. A partial prefill failure
clears uncertain state, so retry with the full messages. Use ordinary or n-gram
decoding for the Qwen MTMD handler, not MTP/DFlash.

## Embedding changed the generation context

This is intentional: embedding batches reuse sequence IDs and positions from
zero. Both embedding APIs clear generation state when execution starts and at final
cleanup; errors rejected during initial validation do not trigger this reset.
Returned vectors remain owned Python data. Use separate instances when
an ongoing conversation's live context must survive embedding requests.
