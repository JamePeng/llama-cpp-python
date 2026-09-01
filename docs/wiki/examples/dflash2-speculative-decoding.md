---
title: DFlash2 Speculative Decoding
source_file: examples/high_level_api/high_level_api_dflash_dspark_speculative.py
last_updated: 2026-09-02
version_target: "0.3.49"
---

# DFlash2 Speculative Decoding

## Goal

Run a compatible target model and DFlash2 sidecar through the stateful Python
speculative-decoding API, then compare it with ordinary decoding using the
included benchmark. DFlash2 proposes a block of tokens, but the target model
still verifies every proposal before it is returned to the caller.

DFlash2 uses `SpeculativeType.DRAFT_DFLASH`. It is selected automatically when
the sidecar GGUF contains a positive `dflash.selector_top_k`; there is no
separate `DRAFT_DFLASH2` enum value.

## Prerequisites

- `llama-cpp-python` `0.3.49` or newer.
- A target GGUF and a DFlash2 GGUF trained for that exact target family.
- Enough combined memory for the target model, sidecar, contexts, and compute
  buffers.
- `n_batch >= 1 + draft_n_max` so `[id_last, draft...]` fits in one target
  verification batch.

The validated Qwen3.8 pair is:

- `Qwen3.8-27B-Q5_K_M.gguf`
- `Qwen3.8-27B-DFlash2-Q8_0.gguf`

The sidecar and target vocabularies, token text, embedding widths, and requested
target-layer taps must be compatible. Similar filenames alone do not guarantee
compatibility.

## Complete Runnable Code

```python
from llama_cpp import Llama
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType


TARGET_MODEL = "path/to/Qwen3.8-27B-Q5_K_M.gguf"
DFLASH2_MODEL = "path/to/Qwen3.8-27B-DFlash2-Q8_0.gguf"

llm = Llama(
    model_path=TARGET_MODEL,
    n_ctx=8192,
    n_batch=512,
    n_ubatch=512,
    n_gpu_layers="all",
    ctx_checkpoints=0,
    speculative=SpecConfig(
        spec_type=SpeculativeType.DRAFT_DFLASH,
        draft_model_path=DFLASH2_MODEL,
        draft_n_max=7,
        draft_n_min=0,
        draft_p_min=0.0,
        draft_n_gpu_layers="all",
        # DFlash2 uses its selector lattice, not vocabulary sampling.
        draft_backend_sampling=False,
    ),
    verbose=True,
)

try:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": "Explain speculative decoding in five concise points.",
            }
        ],
        temperature=0.0,
        max_tokens=256,
    )
    print(response["choices"][0]["message"]["content"])

    stats = llm.last_speculative_stats
    print("drafted:", stats["drafted"])
    print("accepted draft tokens:", stats["accepted_draft_tokens"])
    print("draft acceptance:", stats["draft_token_acceptance_rate"])
    print("generation tok/s:", stats["generation_tokens_per_second"])
finally:
    llm.close()
```

## Expected Output

With `verbose=True`, model initialization identifies the resolved variant and
its metadata. For the validated sidecar, the important fields include:

```text
LlamaDFlashDecoding: DFlash2 speculative decoding enabled
LlamaDFlashDecoding: ... selector_top_k=16, mrope=False, ...
LlamaDFlashDecoding: backend_sampling=requested:False/active:False, ...
```

The generated text and exact statistics depend on the prompt, sampling
settings, hardware, quantization, and accepted draft prefixes. A successful run
should have non-zero `drafted`, `verified`, and `verification_steps` values.

## Compare With Ordinary Decoding

Use the dedicated example for a repeatable correctness and throughput check:

```bash
python -m examples.high_level_api.high_level_api_dflash_dspark_speculative \
  --algorithm dflash2 \
  --model path/to/Qwen3.8-27B-Q5_K_M.gguf \
  --draft-model path/to/Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --max-tokens 512 \
  --runs 3 \
  --warmup-tokens 64 \
  --n-ctx 4096 \
  --n-batch 512 \
  --n-ubatch 512 \
  --draft-tokens 7 \
  --draft-n-min 0 \
  --draft-p-min 0 \
  --temperature 0 \
  --ignore-eos
```

`--algorithm dflash2` requires selector metadata and fails early for a DFlash v1
sidecar. `--algorithm dflash` remains backward compatible and automatically
labels a selector-equipped sidecar as DFlash2.

Read these output fields together:

- `output tokens match` checks the first ordinary and speculative runs.
- `selector top-k` confirms the sidecar's fixed selector width.
- `nextn output: unmasked` confirms the DFlash2 output layout.
- `backend sampling: ... active=False` confirms selector-based selection.
- `draft accepted`, acceptance by position, and rollback counts explain whether
  the configured draft length is useful.
- `speedup` is the end-to-end result; a high internal draft rate alone does not
  guarantee a faster application.

## Configuration Notes

- Start with `draft_n_max=7` for an eight-token DFlash2 block, then benchmark
  shorter values such as 3 and 5. The engine clamps the request to
  `dflash.block_size - 1`.
- Start with `draft_p_min=0`. Raising it stops a path before low-confidence
  selector transitions, which may reduce verification work but also shorten
  useful drafts.
- Keep `temperature=0` while comparing correctness and performance. Add sampling
  only after the deterministic path is stable.
- DFlash2 always reads unmasked NextN selector rows and does not attach the
  backend vocabulary sampler. `draft_backend_sampling=True` is harmless but
  remains inactive for this variant.
- `selector_top_k` is trained into the sidecar and cannot be tuned from Python.
- The engine currently supports text and `seq_id=0` only.
- Text-token M-RoPE positions are supported when the draft sidecar declares
  M-RoPE. Direct MTMD image/video embedding batches are not supported. The
  validated Qwen3.8 DFlash2 sidecar reports `mrope=False`; this is expected from
  its current metadata and does not prevent its tested text workflow.

## Tips

- Measure at least a few hundred generated tokens after warmup; two- or
  eight-token smoke tests validate execution but do not establish speedup.
- Use `--ignore-eos` for fixed-length throughput comparisons.
- Keep the ordinary and DFlash2 prompts, seed, temperature, and stopping rules
  identical.
- Watch GPU memory when increasing `n_ctx`, `n_batch`, or both; the sidecar and
  its draft context consume memory in addition to the target.
- Explicitly close `Llama` in long-running applications to release the sidecar
  model and context deterministically.

## Related Links

- [Llama Speculative Decoding](../modules/LlamaSpeculative.md)
- [Llama high-level API](../core/Llama.md)
- [DFlash2 benchmark example](../../../examples/high_level_api/high_level_api_dflash_dspark_speculative.py)
- [DFlash2 overview](https://inco.ai/blog/dflash2/)
