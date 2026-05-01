---
title: LlamaEmbedding
class_name: LlamaEmbedding
last_updated: 2026-05-01
version_target: "latest"
---

# LlamaEmbedding

## Overview

`LlamaEmbedding` is a specialized class for high-performance Text Embedding and Reranking. It inherits from the base `Llama` class but is optimized for vector operations.

### Support Embeddings & Rerank Model:


|  Model             |  Type     |  Link                                                  |  Status      |
|--------------------|-----------|--------------------------------------------------------|--------------|
|      `bge-m3`      | Embedding |[bge-m3-GGUF](https://huggingface.co/gpustack/bge-m3-GGUF)             |  Useful ✅  |
|`bge-reranker-v2-m3`|   Rerank  |[bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF) |  Useful ✅  |
|`qwen3-reranker`|   Rerank  |[Qwen3-Reranker-GGUF](https://huggingface.co/JamePeng2023/Qwen3-Reranker-GGUF) |  Useful ✅  |

**Core Features:**
1. **Auto-configuration**: Automatically sets `embeddings=True`.
2. **Streaming Batch**: Handles massive datasets without OOM (Out Of Memory).
3. **Native Reranking Support**: Specifically handles `LLAMA_POOLING_TYPE_RANK` models (like BGE-Reranker, Qwen3-Reranker). It correctly identifies classification heads to output scalar relevance scores instead of high-dimensional vectors.
4. **Advanced Normalization**: Implements MaxInt16, Taxicab (L1), and Euclidean (L2) normalization strategies using NumPy for optimal performance and compatibility with various vector databases.

## Constructor `__init__`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to the GGUF model file. |
| `n_ctx` | int | 0 | Text context window size (0 = model default). |
| `n_batch` | int | 512 | Maximum prompt processing batch size. |
| `n_ubatch` | int | 512 | Physical batch size. |
| `pooling_type` | int | `LLAMA_POOLING_TYPE_UNSPECIFIED` (-1) | Pooling strategy used by the model: `LLAMA_POOLING_TYPE_RANK` (4) for rerankers, `LLAMA_POOLING_TYPE_UNSPECIFIED` (-1) for embeddings. |
| `n_gpu_layers` | int | 0 | Number of layers offloaded to GPU (0 = CPU only, -1 = all layers). |
| `verbose` | bool | True | Whether to print debug information. |
| `**kwargs` | Any | — | Extra arguments passed to the `Llama` base class (e.g., `n_batch`, `n_ctx`, `verbose`). |

### Initialization Logic

1. Forces `embeddings=True` to enable embedding support.
2. Sets `kv_unified=True` to enable unified KV Cache, allowing arbitrary sequence IDs in a batch without "invalid seq_id" errors.
3. Passes `pooling_type` to the parent class constructor.

## Core Methods

### `embed(input, normalize=NORM_MODE_EUCLIDEAN, truncate=True, separator=None, return_count=False)`

**Description**: Computes embedding vectors for input text (standard embeddings or reranking scores).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `Union[str, List[str], List[List[int]]]` | — | Input format: string (can be split), list of strings, or list of integer lists (token IDs). |
| `normalize` | int | `NORM_MODE_EUCLIDEAN` (2) | Vector normalization mode (see below). |
| `truncate` | bool | True | Whether to truncate input. |
| `separator` | str | None | Separator for splitting string input into multiple documents. |
| `return_count` | bool | False | If True, returns `(embeddings, token_count)`. |

**Normalization Modes:**
- `NORM_MODE_NONE` (-1): No normalization.
- `NORM_MODE_MAX_INT16` (0): Max absolute value normalization (scaled to 32760).
- `NORM_MODE_TAXICAB` (1): L1 Taxicab norm.
- `NORM_MODE_EUCLIDEAN` (2): L2 Euclidean norm.
- `NORM_MODE_PNORM` (>2): p-norm normalization.

**Returns:**
- `return_count=False`: List of embedding vectors.
- `return_count=True`: Tuple `(embeddings, token_count)`.

**Internal Logic:**
1. Determines mode based on `pooling_type`: `LLAMA_POOLING_TYPE_NONE` (token-level), `LLAMA_POOLING_TYPE_RANK` (rerank), or other (sequence-level).
2. Uses streaming batch decoding to process embeddings in chunks.
3. For token-level mode, extracts and normalizes per-token vectors.
4. For sequence-level mode, extracts sequence vectors and normalizes.
5. Supports `separator` for splitting input into multiple documents.

### `rank(query, documents)`

**Description**: Calculates relevance scores for a list of documents against a query using a Reranking model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | str | Search query string. |
| `documents` | `List[str]` | List of candidate document strings to be scored. |

**Returns**: List of float scores, where higher values indicate greater relevance.

**Internal Logic:**
1. Checks if model is a reranker (`pooling_type == LLAMA_POOLING_TYPE_RANK`).
2. Attempts to retrieve the built-in 'rerank' chat template.
3. If template exists: dynamically replaces `{query}` and `{document}` and tokenizes; otherwise, manually constructs `[BOS] Query [SEP] Doc [EOS]` sequence.
4. Executes embedding inference (`embed`), returning raw logits/scores.
5. For generative rerankers (e.g., Qwen3-Reranker, output dim = 2), uses `yes_logit` as relevance score.

### `create_embedding(input, model=None, normalize=NORM_MODE_EUCLIDEAN, output_format="json")`

**Description**: High-level API compatible with OpenAI format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `Union[str, List[str]]` | — | Input text or list of texts. |
| `model` | str | None | Model name (optional, uses `self.model_path` if None). |
| `normalize` | int | `NORM_MODE_EUCLIDEAN` (2) | Normalization mode. |
| `output_format` | str | "json" | Output format: 'json', 'json+', or 'array'. |

**Output Formats:**
- `'json'`: OpenAI-style dictionary list.
- `'json+'`: OpenAI dictionary list + cosine similarity matrix.
- `'array'`: Raw Python list (`List[float]` or `List[List[float]]`).

**Returns**: Data structure according to `output_format`.

## Deprecated / Changed APIs

- **Note**: The TODO comments `# TODO(JamePeng): Needs more extensive testing with various embedding and reranking models.` indicate that support for various embedding and reranking models may be incomplete. Further testing is recommended.

## Best Practices & Common Patterns

1. **Select Correct `pooling_type`**:
   - Standard embeddings: `LLAMA_POOLING_TYPE_UNSPECIFIED (-1)`.
   - Reranker models: `LLAMA_POOLING_TYPE_RANK (4)`.
   - Token-level embeddings: `LLAMA_POOLING_TYPE_NONE (0)`.

2. **Batch Optimization for Large Datasets**:
   - Adjust `n_batch` and `n_ubatch` to balance performance and memory.
   - Streaming processing avoids OOM for large datasets.

3. **Normalization Selection**:
   - Vector databases typically prefer L2 normalization (Euclidean), but other norms may be needed in specific scenarios.

4. **Reranker Models**:
   - Ensure `pooling_type` is set to `LLAMA_POOLING_TYPE_RANK`.
   - Note that output is scalar scores, not vectors.

5. **Performance Tuning**:
   - For GPU acceleration, set `n_gpu_layers` to -1 (recommended).
   - Use `verbose=True` for debugging configuration.

## Example Code

### 1. Text Embeddings (Vector Search)

To generate embeddings, use the `LlamaEmbedding` class. It automatically configures the model for vector generation.

```python
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(model_path="path/to/bge-m3.gguf", n_gpu_layers=-1, pooling_type=LLAMA_POOLING_TYPE_NONE)

# 1. Simple usage (OpenAI-compatible format)
response = llm.create_embedding("Hello, world!")
print(response['data'][0]['embedding'])

# 2. Batch processing (High Performance)
# You can pass a large list of strings; the streaming batcher handles memory automatically.
documents = ["Hello, world!", "Goodbye, world!", "Llama is cute."] * 100
embeddings = llm.embed(documents) # Returns a list of lists (vectors)

print(f"Generated {len(embeddings)} vectors.")
```

**Advanced Output Formats:**
You can request raw arrays or cosine similarity matrices directly:

```python
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(model_path="path/to/bge-m3.gguf", n_gpu_layers=-1, pooling_type=LLAMA_POOLING_TYPE_NONE)

# Returns raw List[float] instead of a dictionary wrapper
vector = llm.create_embedding("Text", output_format="array")

# Returns a similarity matrix (A @ A.T) in the response
# Note: Requires numpy installed
response = llm.create_embedding(
    ["apple", "fruit", "car"],
    output_format="json+"
)
print(response["cosineSimilarity"])
```

### 2. Reranking (Cross-Encoder Scoring)

Reranking models (like `bge-reranker`) take a **Query** and a list of **Documents** as input and output a relevance score (scalar) for each document.

> **Important:** You must explicitly set `pooling_type` to `LLAMA_POOLING_TYPE_RANK` (4) when initializing the model.

```python
import llama_cpp
from llama_cpp.llama_embedding import LlamaEmbedding

# Initialize a Reranking model
ranker = LlamaEmbedding(
    model_path="path/to/qwen3-reranker-0.6b-q8_0.gguf",
    pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,  # Crucial for Rerankers!
    n_gpu_layers=-1,
    n_ctx=0
)

query = "What causes Rain?"
docs = [
    "Clouds are made of water droplets...", # Relevant
    "To bake a cake you need flour...",     # Irrelevant
    "Rain is liquid water in the form of droplets..." # Highly Relevant
]

# Calculate relevance scores
# Logic: Constructs inputs like "[BOS] query [SEP] doc [EOS]" automatically
scores = ranker.rank(query, docs)

# Result: List of floats (higher means more relevant)
print(scores) 
# e.g., [0.0011407170677557588, 5.614783731289208e-05, 0.7173627614974976] -> The 3rd doc is the best match
```

### 3. Normalization

The `embed` method supports various mathematical normalization strategies via the `normalize` parameter.

| Normalization modes | $Integer$ | Description         | Formula |
|---------------------|-----------|---------------------|---------|
| NORM_MODE_NONE | $-1$      | none                |
| NORM_MODE_MAX_INT16 | $0$       | max absolute int16  | $\Large{{32760 * x_i} \over\max \lvert x_i\rvert}$
| NORM_MODE_TAXICAB | $1$       | taxicab             | $\Large{x_i \over\sum \lvert x_i\rvert}$
| NORM_MODE_EUCLIDEAN | $2$       | euclidean (default) | $\Large{x_i \over\sqrt{\sum x_i^2}}$
| NORM_MODE_PNORM | $>2$      | p-norm              | $\Large{x_i \over\sqrt[p]{\sum \lvert x_i\rvert^p}}$

This is useful for optimizing storage or preparing vectors for cosine similarity search (which requires L2 normalization).

```python
from llama_cpp.llama_embedding import (
  LLAMA_POOLING_TYPE_NONE,
  NORM_MODE_MAX_INT16,
  NORM_MODE_TAXICAB,
  NORM_MODE_EUCLIDEAN
)

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(model_path="path/to/bge-m3.gguf", n_gpu_layers=-1, pooling_type=LLAMA_POOLING_TYPE_NONE)

# Taxicab (L1)
vec_l1 = llm.embed("text", normalize=NORM_MODE_TAXICAB)

# Default is Euclidean (L2) - Standard for vector databases
vec_l2 = llm.embed("text", normalize=NORM_MODE_EUCLIDEAN)

# Max Absolute Int16 - Useful for quantization/compression
vec_int16 = llm.embed("text", normalize=NORM_MODE_MAX_INT16)

# Raw Output (No Normalization) - Get the raw floating point values from the model
embeddings_raw = llm.embed(["search query", "document text"], normalize=NORM_MODE_NONE)
```

## Notes

- This class is in development; some features may be unstable, especially reranking model support.
- Performance issues can be addressed by adjusting `n_batch`, `n_ubatch`, and `n_gpu_layers`.
- For custom models, manual `pooling_type` configuration may be required to match model behavior.

## Related Links

* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
