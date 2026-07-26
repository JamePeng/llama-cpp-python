---
title: Embeddings and Reranking
feature_name: Embeddings and Reranking
source_files:
  - llama_cpp/llama.py
  - llama_cpp/llama_embedding.py
  - llama_cpp/_internals.py
last_updated: 2026-07-26
version_target: "latest"
---

# Embeddings and Reranking

## Overview

`llama-cpp-python` can use compatible GGUF models for three related inference
workflows:

- **Sentence or document embeddings** produce one vector per input.
- **Token embeddings** produce one vector per token.
- **Reranking** scores each query/document pair with a cross-encoder model.

The general-purpose `Llama` class and the specialized `LlamaEmbedding` class
share the same native model and context implementation. Both support streaming
batches, pre-tokenized inputs, multiple pooling modes, and configurable vector
normalization.

`LlamaEmbedding` adds embedding-oriented defaults, extra output formats, and
the `rank()` helper. The standard `Llama` API is useful when an application
already manages models through the main class or needs both generation and
embedding capabilities.

## When to Use

| Goal | Recommended API | Pooling |
|---|---|---|
| Store one vector per sentence or document | `Llama.embed()` or `LlamaEmbedding.embed()` | `LLAMA_POOLING_TYPE_UNSPECIFIED`, or the model-required MEAN/CLS/LAST mode |
| Return an OpenAI-style embedding response | `create_embedding()` | Sequence pooling |
| Inspect a vector for every token | `embed()` | `LLAMA_POOLING_TYPE_NONE` |
| Score documents against a query | `LlamaEmbedding.rank()` | `LLAMA_POOLING_TYPE_RANK` |
| Return raw arrays or a cosine-similarity matrix | `LlamaEmbedding.create_embedding()` | Sequence pooling |

Use the pooling configuration documented by the model author whenever one is
provided. `LLAMA_POOLING_TYPE_UNSPECIFIED` lets model metadata select the
sequence-pooling behavior and is the safest general default for ordinary
sentence embeddings.

## Supported Models

The project README currently lists the following GGUF model families as working
with the embedding and reranking APIs:

| Model family | Task | GGUF model |
|---|---|---|
| `bge-m3` | Embedding | [bge-m3-GGUF](https://huggingface.co/gpustack/bge-m3-GGUF) |
| `jina-embeddings-v2-base-zh` | Embedding | [jina-embeddings-v2-base-zh-GGUF](https://huggingface.co/gpustack/jina-embeddings-v2-base-zh-GGUF) |
| `jina-embeddings-v3` | Embedding | [jina-embeddings-v3-GGUF](https://huggingface.co/second-state/jina-embeddings-v3-GGUF) |
| `bge-reranker-v2-m3` | Reranking | [bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF) |
| `qwen3-reranker` | Reranking | [Qwen3-Reranker-GGUF](https://huggingface.co/JamePeng2023/Qwen3-Reranker-GGUF) |

This is a known-compatible list, not an exhaustive compatibility matrix.
Support for a specific file still depends on its GGUF metadata, pooling
configuration, classifier head, tokenizer, and reranking template. Validate
the output shape and quality before deploying a new model or quantization.

## Related APIs

| API | Role |
|---|---|
| `Llama(..., embeddings=True)` | General-purpose model interface with maintained `embed()` and `create_embedding()` methods |
| `LlamaEmbedding(...)` | Specialized subclass that forces `embeddings=True` and `kv_unified=True` |
| `Llama.embed()` | Raw sequence, token-level, or rank output with optional token counting |
| `Llama.create_embedding()` | OpenAI-compatible response wrapper; defaults to raw vectors |
| `LlamaEmbedding.embed()` | Specialized raw embedding API; defaults to L2 normalization |
| `LlamaEmbedding.create_embedding()` | Adds `json`, `json+`, and `array` output formats |
| `LlamaEmbedding.rank()` | Formats query/document pairs and returns reranking scores |
| `Llama.tokenize()` | Converts text into token IDs for pre-tokenized embedding input |

See [[core/Llama|Llama]] for the general model lifecycle and
[[modules/LlamaEmbedding|Llama Embedding]] for the complete specialized class
reference.

## Code Examples

All examples assume that `MODEL_PATH` points to a compatible GGUF embedding or
reranking model. Pooling requirements and output dimensions are model-specific.

### Sentence Embeddings with `Llama`

```python
from llama_cpp import Llama, LLAMA_POOLING_TYPE_UNSPECIFIED


MODEL_PATH = "path/to/embedding-model.gguf"

model = Llama(
    model_path=MODEL_PATH,
    embeddings=True,
    pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED,
    n_ctx=512,
    n_batch=512,
    n_ubatch=512,
    n_seq_max=8,
    kv_unified=True,
    n_gpu_layers=-1,
    verbose=False,
)

try:
    documents = [
        "The weather is pleasant today.",
        "A storm is expected tomorrow.",
        "Vector search compares semantic meaning.",
    ]

    vectors, token_count = model.embed(
        documents,
        normalize=True,
        return_count=True,
    )

    print("vectors:", len(vectors))
    print("dimension:", len(vectors[0]))
    print("processed tokens:", token_count)

    response = model.create_embedding(
        documents,
        normalize=2,
    )
    print(response["usage"])
finally:
    model.close()
```

For `Llama.embed()`, `normalize=False` is the backward-compatible default.
`True` and integer mode `2` both select L2 normalization.

### Specialized Batch Embeddings and Similarity

```python
from llama_cpp import LLAMA_POOLING_TYPE_UNSPECIFIED
from llama_cpp.llama_embedding import (
    LlamaEmbedding,
    NORM_MODE_EUCLIDEAN,
)


MODEL_PATH = "path/to/embedding-model.gguf"

model = LlamaEmbedding(
    model_path=MODEL_PATH,
    pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED,
    n_ctx=512,
    n_batch=512,
    n_ubatch=512,
    n_seq_max=8,
    n_gpu_layers=-1,
    verbose=False,
)

try:
    texts = ["apple", "fruit", "automobile"]

    # "array" always returns one vector entry per input.
    vectors = model.create_embedding(
        texts,
        normalize=NORM_MODE_EUCLIDEAN,
        output_format="array",
    )
    print("first vector dimension:", len(vectors[0]))

    response = model.create_embedding(
        texts,
        normalize=NORM_MODE_EUCLIDEAN,
        output_format="json+",
    )
    print(response["cosineSimilarity"])
finally:
    model.close()
```

`json+` extends the OpenAI-style response with `cosineSimilarity` when at least
two compatible sequence vectors are available.

### Token-Level Embeddings

```python
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE


model = Llama(
    model_path="path/to/embedding-model.gguf",
    embeddings=True,
    pooling_type=LLAMA_POOLING_TYPE_NONE,
    n_ctx=256,
    n_batch=256,
    verbose=False,
)

try:
    token_vectors = model.embed("Token-level example", normalize=True)

    print("tokens:", len(token_vectors))
    print("dimension per token:", len(token_vectors[0]))
finally:
    model.close()
```

Token-level output is a matrix, not one flat vector per document. It is useful
for token analysis and custom pooling, but it is not the normal shape expected
by OpenAI-compatible vector-store clients.

### Pre-tokenized and Separator-Split Inputs

```python
from llama_cpp import Llama, LLAMA_POOLING_TYPE_UNSPECIFIED


model = Llama(
    model_path="path/to/embedding-model.gguf",
    embeddings=True,
    pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED,
    n_ctx=256,
    n_batch=256,
    verbose=False,
)

try:
    token_batches = [
        model.tokenize(b"first document"),
        model.tokenize(b"second document"),
    ]
    vectors = model.embed(token_batches, normalize=2)

    split_vectors = model.embed(
        "first document\nsecond document",
        separator="\n",
        normalize=2,
    )

    print(len(vectors), len(split_vectors))
finally:
    model.close()
```

When `separator` is set, a single string is treated as a batch and the return
value uses the batch shape.

### Reranking Query/Document Pairs

```python
from llama_cpp import LLAMA_POOLING_TYPE_RANK
from llama_cpp.llama_embedding import LlamaEmbedding


RERANK_MODEL_PATH = "path/to/reranker-model.gguf"

ranker = LlamaEmbedding(
    model_path=RERANK_MODEL_PATH,
    pooling_type=LLAMA_POOLING_TYPE_RANK,
    n_ctx=1024,
    n_batch=1024,
    n_ubatch=512,
    n_seq_max=8,
    n_gpu_layers=-1,
    verbose=False,
)

try:
    query = "What causes rain?"
    documents = [
        "Rain forms when atmospheric water vapor condenses and falls.",
        "A cake is made from flour, eggs, and sugar.",
        "Cloud droplets grow until gravity pulls them toward the ground.",
    ]

    scores = ranker.rank(query, documents)
    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    for document, score in ranked:
        print(f"{score:.6f}  {document}")
finally:
    ranker.close()
```

`rank()` first checks for a model-provided `rerank` chat template. If no
template exists, it constructs a sequence from the model's BOS, separator, and
EOS tokens.

## Configuration Notes

### Pooling Modes

| Constant | Output behavior | Typical use |
|---|---|---|
| `LLAMA_POOLING_TYPE_UNSPECIFIED` | Uses the model-configured pooling behavior | Default for sentence embedding models |
| `LLAMA_POOLING_TYPE_NONE` | One vector per token | Token analysis or custom pooling |
| `LLAMA_POOLING_TYPE_MEAN` | Mean-pooled sequence vector | Models trained for mean pooling |
| `LLAMA_POOLING_TYPE_CLS` | Vector from the classification token | Models trained with CLS pooling |
| `LLAMA_POOLING_TYPE_LAST` | Vector from the final token | Models trained with last-token pooling |
| `LLAMA_POOLING_TYPE_RANK` | Classifier or reranking output | Cross-encoder reranking models |

Do not select `LLAMA_POOLING_TYPE_NONE` when one vector per input is required.
It changes both the amount of output and its nesting depth.

### Normalization Modes

| Mode | Value | Behavior |
|---|---:|---|
| `NORM_MODE_NONE` | `-1` | Return raw values |
| `NORM_MODE_MAX_INT16` | `0` | Scale the maximum absolute component to `32760` |
| `NORM_MODE_TAXICAB` | `1` | L1/taxicab normalization |
| `NORM_MODE_EUCLIDEAN` | `2` | L2/Euclidean normalization |
| p-norm | Any integer greater than `2` | Normalize using the corresponding p-norm |

The constant `NORM_MODE_PNORM` currently has value `6`; callers may also pass a
different integer greater than `2`.

Normalization defaults differ between the two classes:

| API | Default |
|---|---|
| `Llama.embed()` / `Llama.create_embedding()` | Raw output (`False`) |
| `LlamaEmbedding.embed()` / `LlamaEmbedding.create_embedding()` | L2 (`NORM_MODE_EUCLIDEAN`) |
| Rank output | Never normalized |

L2-normalized vectors are convenient for cosine similarity because their dot
product is their cosine similarity.

### Batch and Context Capacity

| Parameter | Controls |
|---|---|
| `n_ctx` | Maximum context length available to an input sequence |
| `n_batch` | Maximum tokens in one logical decode batch |
| `n_ubatch` | Physical token micro-batch size used by llama.cpp |
| `n_seq_max` | Maximum independent sequences decoded together |

Embedding input lists are streamed through multiple decode batches. The
default `n_seq_max=1` is valid and processes inputs sequentially. Increasing it
allows more independent sequences to be decoded together, but may use more
context resources.

Each individual tokenized sequence must fit the configured logical batch
capacity. Choose `n_batch` large enough for the longest intended input and use
`truncate=True` when truncation is acceptable.

### Input and Return Shapes

| Input and mode | Direct `embed()` result |
|---|---|
| Single string with sequence pooling | `List[float]` |
| String list with sequence pooling | `List[List[float]]` |
| Separator-split string with sequence pooling | `List[List[float]]` |
| Single string with token-level pooling | `List[List[float]]` |
| String list with token-level pooling | `List[List[List[float]]]` |
| Rank model with one classifier output | Scalar for one string; list of scalars for a batch |
| Rank model with multiple classifier outputs | Classifier vector per input |
| Any input with `return_count=True` | `(result, processed_token_count)` |

Token counts are measured after tokenization and any applied truncation.

### Output Wrappers

`Llama.create_embedding()` returns an OpenAI-compatible dictionary containing
`object`, `data`, `model`, and token `usage`.

`LlamaEmbedding.create_embedding()` supports:

| `output_format` | Result |
|---|---|
| `"json"` | OpenAI-style response |
| `"json+"` | OpenAI-style response plus a cosine-similarity matrix when available |
| `"array"` | Raw list containing one output entry per input |

For OpenAI-compatible vector-store clients, use sequence pooling so each
`data[i]["embedding"]` value is a flat vector.

### Common Configuration Problems

| Symptom | Cause | Action |
|---|---|---|
| `Llama model must be created with embeddings=True` | Standard `Llama` was initialized without embedding extraction | Recreate it with `embeddings=True` |
| Output is a matrix for each document | `LLAMA_POOLING_TYPE_NONE` selects token-level output | Use `UNSPECIFIED` or the pooling mode required by the model |
| `seq_id` exceeds `n_seq_max` in custom batch code | A manual sequence ID is outside the configured capacity | Increase `n_seq_max` or use IDs within `0..n_seq_max-1` |
| A long input exceeds `n_batch` | One tokenized sequence is larger than the logical batch | Increase `n_batch`, shorten the input, or enable truncation |
| Local source changes are not visible | Python imported an installed `site-packages` build | Print `llama_cpp.__file__`, then reinstall or adjust the development environment |

## Limitations

- Embedding dimensions, valid pooling modes, tokenization, and reranking heads
  are determined by the GGUF model. A model that was not exported for the
  requested task may not produce meaningful output.
- `rank()` returns raw model scores. They are not automatically calibrated as
  probabilities and should primarily be compared within the same query.
- For a two-output reranking head, `rank()` uses the first output as the score.
  It does not apply softmax.
- The fallback reranking prompt depends on the model's BOS, separator, and EOS
  tokens. Prefer a GGUF model containing a suitable `rerank` chat template.
- `json+` similarity output is intended for at least two compatible,
  fixed-length sequence vectors. It is not suitable for ragged token-level
  matrices or scalar rank scores.
- Embedding calls clear the context memory used by the operation. Do not expect
  a previous completion KV-cache state to remain reusable after embedding on
  the same model instance.
- Model and reranking support still requires broader testing across GGUF
  architectures. Validate output quality and shape before production use.

## Related Features
- [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
- [[core/Llama|Llama](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)] — General model lifecycle and built-in embedding APIs.
- [[modules/LlamaEmbedding|Llama Embedding](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaEmbedding.md)] — Specialized API reference,
  normalization constants, and reranking methods.
- [[install|Installation](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/install.md)] — Backend selection, GPU acceleration, and source
  installation.
