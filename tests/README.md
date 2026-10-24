# Tests

## Running

From the repository root, with the compiled package installed:

```sh
python -m pytest
python -m pytest tests/test_state.py
```

Tests use local model files and never download them. Unconfigured model tests
skip locally; the core Transformer, Hybrid and vision paths are required in
[Actions](../.github/workflows/test.yaml). Configured paths must exist and load
successfully. Optional embedding and DFlash model pairs skip when unset.

## Models

| Environment variable | Model / purpose |
| --- | --- |
| `LLAMA_TEST_TRANSFORMER_MODEL` | Qwen2.5-0.5B-Instruct Q4_K_M; Transformer runtime, KV state and grammar |
| `LLAMA_TEST_HYBRID_MODEL` | Qwen3.5 with MTP weights; Hybrid checkpoints and speculative decoding |
| `LLAMA_TEST_MMPROJ` | Matching Qwen3.5 projector; vision and MTP image tests |
| `LLAMA_TEST_MULTIMODAL_EMBEDDING_MODEL` / `LLAMA_TEST_MULTIMODAL_EMBEDDING_MMPROJ` | EmbeddingGemma 2 model and projector |
| `LLAMA_TEST_QWEN3VL_EMBEDDING_MODEL` / `LLAMA_TEST_QWEN3VL_EMBEDDING_MMPROJ` | Qwen3-VL-Embedding model and projector |
| `LLAMA_TEST_DFLASH_MODEL` / `LLAMA_TEST_DFLASH_DRAFT` | DFlash target and draft model |
| `LLAMA_TEST_DFLASH2_MODEL` / `LLAMA_TEST_DFLASH2_DRAFT` | DFlash2 target and draft model |
| `LLAMA_TEST_LILICORR_MODEL` / `LLAMA_TEST_LILICORR_DRAFT` | LiliCorr target and draft model |

CI pins [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
and [Qwen3.5-0.8B with its projector](https://huggingface.co/JamePeng2023/Qwen3.5-0.8B-GGUF).
Larger compatible local models can use `LLAMA_TEST_N_GPU_LAYERS=-1` for full GPU
offload; the default is `0`. Tokenizer tests use the vocabulary-only GGUF in
`vendor/llama.cpp/models`.

## Coverage

| Module | Main checks |
| --- | --- |
| `test_cache.py` | RAM/disk/trie prefix matching, thresholds, ties, eviction and memory bounds |
| `test_state.py` | Snapshot ownership, Hybrid restore, zero checkpoints, cache reuse and failure recovery |
| `test_speculation.py` | MTP/DFlash verification, rollback, interrupted continuation and score callbacks |
| `test_runtime.py` | Native API and batch parity, resource lifetimes, sampling, KV state and embeddings |
| `test_media.py` | Vision/audio handling, MTP image requests and partial-decode recovery |
| `test_formats.py` | Chat formatting and grammar constraints |
| `test_rpc.py` | RPC backend, endpoint validation and device ownership |

Sampling regressions compare suppression, merged user biases, and candidate
probabilities against native sampler chains. They cover ordinary sampling,
restored logits, sampler clones, grammar resampling, reasoning-budget forcing,
MTP internal/sidecar list ownership, CPU confidence cutoffs, and backend-selected
token bypasses.

## Test design and efficiency

Development time, maintenance effort, and CI execution time are part of the test
design budget. Aggregate, compact, and decouple tests as much as practical while
preserving meaningful coverage and readable failure reports.

- Group equivalent inputs with parametrization or local loops. Reuse reference
  results across equivalent paths and avoid redundant Cartesian products,
  repeated model loads, and duplicate native computation.
- Extract repeated setup and assertions into small, focused fixtures or helpers.
  Keep behavioral expectations independent of the helper implementation; avoid
  abstractions or dense expressions that make failures harder to understand.
- Share model weights and owned cold references, but reset mutable contexts
  between cases. Isolate patches and disk caches, release owned resources, and
  keep tests independent of execution order and other tests' side effects.
- Preserve boundary, failure-recovery, ownership, and probability assertions when
  consolidating tests. Label aggregated cases so failures identify the affected
  input and backend; fewer lines or collected cases alone do not justify losing
  coverage.

## Quantized model comparisons

With quantized GGUFs such as Q4_K_M, full-prompt and prefix/suffix decoding can
produce different logits because their batch shapes differ. Close scores can
swap order and change greedy output even at `temperature=0` with a fixed seed.
Results can also vary by backend; different text alone does not establish
incorrect cache reuse.

Compare reuse against an independent native context with the same model, runtime
settings, batch boundaries and output flags. Keep assertions on logits, evaluated
suffixes, token histories and native positions to detect actual state errors.

## Benchmarks

See the scripts' `--help` for model, baseline and result options:

```sh
python -m examples.benchmark.benchmark_trie_cache --help
python -m examples.benchmark.benchmark_default_reuse --help
```

The first compares Trie maintenance and cache reuse. The second measures the
default `cache=None` prefix reuse and MTP rollback path. Both support comparison
against saved baseline sources and check generated tokens against cold references.
