# Tests

## Models

| Environment variable | Model file | Purpose |
| --- | --- | --- |
| `LLAMA_TEST_TRANSFORMER_MODEL` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Ordinary Transformer execution and KV state |
| `LLAMA_TEST_HYBRID_MODEL` | `Qwen3.5-0.8B-Q4_K_M.gguf` | Hybrid memory, checkpoints and MTP decoding |
| `LLAMA_TEST_MMPROJ` | `mmproj-Qwen3.5-0.8b-Q8_0.gguf` | Vision input with the Qwen3.5 model |
| `LLAMA_TEST_MULTIMODAL_EMBEDDING_MODEL` | `embeddinggemma-2-BF16.gguf` | Optional non-causal embedding tests |
| `LLAMA_TEST_MULTIMODAL_EMBEDDING_MMPROJ` | `mmproj-embeddinggemma-2-BF16.gguf` | Optional image embedding test |

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
| `test_state.py` | `hybrid_model` | Qwen3.5: host/device checkpoints, prefix reuse, snapshots and failure recovery |
| `test_speculation.py` | `mtp_model` | Qwen3.5: MTP verification, native/device checkpoint paths and interrupted-request recovery |
| `test_media.py` | `vision_model` | Qwen3.5 + mmproj: plain/NGRAM image requests, image changes and partial-decode recovery |
| `test_formats.py` | `grammar_model` | Qwen2.5: native grammar acceptance and constrained generation; unit tests cover chat formatting |

Tokenization also uses the vocabulary-only GGUF in `vendor/llama.cpp/models`.
DFlash and generated-audio paths use test doubles; they do not load dedicated draft or TTS models.
`test_runtime.py` checks multimodal entrypoint initialization and projector configuration.
Its optional EmbeddingGemma 2 test covers lazy projector loading, text/image embeddings,
marker-free parts, output dimensions, embedding-mode restoration, unsupported image
input, resource closure and rejection after closure.
MTP with image prefill is not a supported combination; real image tests use plain or NGRAM decoding.

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

Keep fixtures local, parameterize equivalent cases, and preserve ABI, boundary,
failure-recovery and ownership coverage. Avoid tests of private structure,
fixed log text or algorithms reimplemented only inside tests.

`examples/multimodal_embedding.py` uses `LlamaEmbedding.embed_multimodal()`.
For marker-free inputs, pass `parts=["title: none | text: ", image_bytes]`.
`MTMDBaseHandler.tokenize()` / `tokenize_parts()` are context managers; call
`request.encode()` inside the scope to obtain owned tokens, features and positions.
The whole embedding input must fit `n_ctx`, `n_batch` and `n_ubatch` without truncation.
