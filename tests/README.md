# Tests

## Models

| Environment variable | Model file | Purpose |
| --- | --- | --- |
| `LLAMA_TEST_TRANSFORMER_MODEL` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Ordinary Transformer execution and KV state |
| `LLAMA_TEST_HYBRID_MODEL` | `Qwen3.5-0.8B-Q4_K_M.gguf` | Hybrid memory, checkpoints and MTP decoding |
| `LLAMA_TEST_MMPROJ` | `mmproj-Qwen3.5-0.8b-Q8_0.gguf` | Vision input with the Qwen3.5 model |
| `LLAMA_TEST_MULTIMODAL_EMBEDDING_MODEL` | `embeddinggemma-2-BF16.gguf` | Optional non-causal embedding tests |
| `LLAMA_TEST_MULTIMODAL_EMBEDDING_MMPROJ` | `mmproj-embeddinggemma-2-BF16.gguf` | Optional image embedding test |
| `LLAMA_TEST_QWEN3VL_EMBEDDING_MODEL` / `LLAMA_TEST_QWEN3VL_EMBEDDING_MMPROJ` | Qwen3-VL-Embedding-8B Q8 target / BF16 projector | Optional causal LAST-pooling text/image embeddings and native submission comparison |
| `LLAMA_TEST_DFLASH_MODEL` / `LLAMA_TEST_DFLASH_DRAFT` | Qwen3.6-27B Q5 target / DFlash Q8 sidecar | Optional DFlash block limits, verification and interrupted retry |
| `LLAMA_TEST_DFLASH2_MODEL` / `LLAMA_TEST_DFLASH2_DRAFT` | Qwen3.8-27B Q5 target / DFlash2 Q8 sidecar | Optional selector drafting with the same recovery checks |

Qwen2.5 comes from [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).
The Qwen3.5 model, including MTP weights, and its matching projector come from
[JamePeng2023/Qwen3.5-0.8B-GGUF](https://huggingface.co/JamePeng2023/Qwen3.5-0.8B-GGUF).
[Actions](../.github/workflows/test.yaml) prepares pinned models in one job, reusing a cache
across runs. Each test environment downloads the same model artifact and receives local paths.

## Test coverage

Each module combines focused unit tests with model tests where needed.

| File | Real-model entry points | Model and coverage |
| --- | --- | --- |
| `test_runtime.py` | `llama_cpp_model_path`, `completion_model` | Qwen2.5: native decode, sampling, prefix append/rollback, save/load and embedding |
| `test_runtime.py` | `LLAMA_TEST_HYBRID_MODEL` | Qwen3.5: extended token/state inputs vs legacy MTP, and embedding/state inputs vs the direct native API |
| `test_state.py` | `hybrid_model` | Qwen3.5: host/device checkpoints, prefix reuse, snapshots and failure recovery |
| `test_speculation.py` | `mtp_model` | Qwen3.5: MTP verification, embedding catch-up vs direct native submission, native/device checkpoint paths and interrupted-request recovery |
| `test_speculation.py` | `dflash_model` | DFlash/DFlash2: full-block prefix equivalence under runtime caps, minimum-length ordering, verification and interrupted retry |
| `test_media.py` | `vision_model` | Qwen3.5 + mmproj: plain/NGRAM/MTP image requests, image changes, native/device rollback and partial-decode recovery |
| `test_formats.py` | `grammar_model` | Qwen2.5: native grammar acceptance and constrained generation; unit tests cover chat formatting |

Tokenization also uses the vocabulary-only GGUF in `vendor/llama.cpp/models`.
DFlash/DSpark unit tests use test doubles, with optional DFlash/DFlash2 target
and sidecar model tests. Generated-audio tests do not load dedicated TTS models.
`test_runtime.py` checks multimodal entrypoint initialization and projector configuration.
Its optional EmbeddingGemma 2 test covers lazy projector loading, text/image embeddings,
marker-free parts, output dimensions, embedding-mode restoration, unsupported image
input, resource closure and rejection after closure.
The same parameterized test covers Qwen3-VL-Embedding with its causal attention
mode, scalar text positions, four-axis media positions and direct native batch
equivalence. Set `LLAMA_TEST_N_GPU_LAYERS=-1` for target GPU offload in these tests.
Aggregated unit cases cover attention-mode compatibility, failure cleanup and
normalization modes, including large-value Lp inputs that overflow float32 powers.
High-level MTP image tests use MTMD post-decode callbacks for every media batch,
compare greedy output with target-only generation, and cover repeated/changed
images, interrupted requests and partial-decode recovery under native and device
checkpoint modes. These requests rebuild both contexts; target-only prefix
checkpoints cannot restore draft state, and the prompt must fit without context
shifting. Lower-level MTP tests submit synthetic feature
rows and compare continuation hidden outputs and candidates with the direct
native API; unit tests cover scalar/M-RoPE/IM-RoPE positions, cross-batch state
handoff, shared-memory/chained-head paths, failures and media checkpoint replay.
With `LLAMA_TEST_MMPROJ` configured, the lower-level MTP image test also encodes
a real test image, processes text/media chunks, drafts tokens, verifies them and
rolls both contexts back under native and device checkpoint modes.

## Running

Set the first three variables above to local GGUF paths, then run:

```sh
python -m pytest
```

Tests read these paths without downloading models. Missing configuration skips
the corresponding local model tests but fails in Actions. Configured paths must
exist, and model load failures are errors. A single module can be run with
`python -m pytest tests/test_state.py`.
Set both `LLAMA_TEST_MULTIMODAL_EMBEDDING_*` variables to enable the optional
EmbeddingGemma 2 test; missing configuration skips it locally and in Actions.
Each optional DFlash pair also skips when its paths are unset. Set
`LLAMA_TEST_N_GPU_LAYERS=-1` to offload both target and draft for those tests;
the default is `0`. Run only these model cases with
`python -m pytest tests/test_speculation.py -k real_dflash`.

Keep fixtures local, parameterize equivalent cases, and preserve ABI, boundary,
failure-recovery and ownership coverage. Avoid tests of private structure,
fixed log text or algorithms reimplemented only inside tests.
Runtime cases aggregate normalization inputs within each mode, graph-reservation
validation and recovery, initialization ownership failures, borrowed-reference
lifetimes, legacy batch layouts and sampling constraints. Native sampler checks
reuse the shared model and verify valid output/state progression without relying
on a particular randomly generated sentence.

`examples/multimodal_embedding.py` uses `LlamaEmbedding.embed_multimodal()`.
For marker-free inputs, pass `parts=["title: none | text: ", image_bytes]`.
`MTMDBaseHandler.tokenize()` / `tokenize_parts()` are context managers; call
`request.encode()` inside the scope to obtain owned tokens, features and positions.
The whole embedding input must fit `n_ctx`, `n_batch` and `n_ubatch` without truncation.
