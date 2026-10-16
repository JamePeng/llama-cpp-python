---
title: Llama Embedding
module_name: llama_cpp.llama_embedding
source_file: llama_cpp/llama_embedding.py
class_name: LlamaEmbedding
last_updated: 2026-10-14
version_target: "latest"
---

# Llama Embedding

## Overview

`LlamaEmbedding` provides embedding-oriented defaults and reranking helpers on
top of `Llama`. Its `embed()` method uses the base implementation while retaining
L2 normalization as the default.
With a compatible MTMD projector, `embed_multimodal()` also pools text, images
and audio into one vector. It uses `MTMDEmbeddingHandler` without chat templates
or a generation prompt.

### Support Embeddings & Rerank Model:


|  Model             |  Type     |  Link                                                  |  Status      |
|--------------------|-----------|--------------------------------------------------------|--------------|
|`bge-m3`| Embedding |[bge-m3-GGUF](https://huggingface.co/gpustack/bge-m3-GGUF) |  Useful ✅  |
|`jina-embeddings-v2-base-zh`| Embedding |[jina-embeddings-v2-base-zh-GGUF](https://huggingface.co/gpustack/jina-embeddings-v2-base-zh-GGUF)  |  Useful ✅  |
|`jina-embeddings-v3`| Embedding |[jina-embeddings-v3-GGUF](https://huggingface.co/second-state/jina-embeddings-v3-GGUF) |  Useful ✅  |
|`bge-reranker-v2-m3`|   Rerank  |[bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF) |  Useful ✅  |
|`qwen3-reranker`|   Rerank  |[Qwen3-Reranker-GGUF](https://huggingface.co/JamePeng2023/Qwen3-Reranker-GGUF) |  Useful ✅  |

**Core Features:**
1. **Auto-configuration**: Automatically sets `embeddings=True`.
2. **Streaming Batch**: Limits inputs active in native inference; returned results still accumulate in Python memory.
3. **Native Reranking Support**: Specifically handles `LLAMA_POOLING_TYPE_RANK` models (like BGE-Reranker, Qwen3-Reranker). It correctly identifies classification heads to output scalar relevance scores instead of high-dimensional vectors.
4. **Advanced Normalization**: Implements MaxInt16, Taxicab (L1), and Euclidean (L2) normalization strategies using NumPy for optimal performance and compatibility with various vector databases.

## Constructor `__init__`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to the GGUF model file. |
| `n_ctx` | int | 0 | Text context window size (0 = model default). |
| `n_batch` | int | 512 | Maximum prompt processing batch size. |
| `n_ubatch` | int | 512 | Physical batch size. |
| `n_seq_max` | int | 1 (inherited) | Maximum number of independent sequence IDs available in a decode batch. Increase this for parallel embedding batches. |
| `pooling_type` | int | `LLAMA_POOLING_TYPE_UNSPECIFIED` (-1) | Pooling strategy used by the model: `LLAMA_POOLING_TYPE_RANK` (4) for rerankers, `LLAMA_POOLING_TYPE_UNSPECIFIED` (-1) for embeddings. |
| `n_gpu_layers` | int | 0 | Number of model layers offloaded to GPU. `0` disables layer offload but does not guarantee that every operation avoids an available GPU backend; `-1` uses automatic offload. |
| `verbose` | bool | True | Whether to print debug information. |
| `mmproj_path` | `Optional[str]` | `None` | Matching MTMD projector GGUF for multimodal embeddings. Loaded lazily on first use. |
| `embedding_handler` | `Optional[MTMDEmbeddingHandler]` | `None` | Custom projector settings; use instead of `mmproj_path`. The model owns and closes this handler. |
| `**kwargs` | Any | — | Extra arguments passed to the `Llama` base class (e.g., `n_batch`, `n_ctx`, `verbose`). |

### Initialization Logic

1. Forces `embeddings=True` to enable embedding support.
2. Sets `kv_unified=True` to enable unified KV Cache. Sequence IDs must still
   fit within the configured `n_seq_max`.
3. Passes `pooling_type` to the parent class constructor.
4. Configures an optional embedding projector. Passing both `mmproj_path` and
   `embedding_handler` raises `ValueError`; a different handler type raises `TypeError`.

### Parallel Batch Capacity

`n_batch`, `n_ubatch`, and `n_seq_max` control different limits:

- `n_batch`: maximum number of input tokens in a logical decode batch.
- `n_ubatch`: physical token batch size used by llama.cpp.
- `n_seq_max`: number of independent sequence IDs that may coexist in a decode
  batch.

For multiple documents, set `n_seq_max` to the desired parallel sequence
capacity:

```python
model = LlamaEmbedding(
    model_path="path/to/model.gguf",
    n_batch=512,
    n_ubatch=512,
    n_seq_max=8,
)
```

If the configuration is too small, the error includes the current capacity,
valid ID range, and required minimum:

```text
LlamaBatch.add_sequence: seq_id=1 exceeds the configured sequence capacity
(n_seq_max=1; valid IDs are 0 through 0). For parallel batching, initialize
Llama or LlamaEmbedding with n_seq_max>=2 ...
```

`n_seq_max` is not the total number of documents passed to `embed()`; it is the
number that can be active in one decode batch. Increase it carefully because
larger values may require more context resources.
For non-causal models, each text input must also fit `n_ubatch`: attention needs
the whole sequence in one physical batch. `embed()` applies truncation against
`min(n_batch, n_ubatch)` for these models; `truncate=False` rejects oversized inputs.

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
- `NORM_MODE_PNORM` (6): L6 normalization; any integer mode greater than 2 selects that p-norm.

**Returns:**
- Sequence pooling: one vector for a single string, or a list of vectors for a batch.
- NONE pooling: one token-vector matrix per input.
- RANK pooling: a scalar per input for one classifier output, otherwise a classifier vector.
- `return_count=True`: Tuple `(result, token_count)`.

**Implementation and lifecycle:**

`LlamaEmbedding.embed()` delegates to [Llama.embed()](../core/Llama.md#embedinput-normalizefalse-truncatetrue-separatornone-return_countfalse).
It retains the L2 default and treats an empty separator as no splitting. The
shared implementation packs whole inputs into batches, copies borrowed output
vectors, and resets generation state before execution and during final cleanup.
Decode or output-extraction failures cannot leave a reusable partial request.
Rank output remains unnormalized; returned vectors are independent Python data.

### `embed_multimodal(prompt=None, media=(), *, parts=None, normalize=2, return_count=False)`

Available starting with `0.4.3-preview`.

Produce one pooled vector for a text/image/audio input. This requires an open
model, a configured projector, and active MEAN, CLS or LAST pooling. The model's
attention mode is preserved: EmbeddingGemma 2 uses bidirectional attention and
MEAN pooling; Qwen3-VL-Embedding uses causal attention and LAST pooling.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `Optional[str]` | `None` | Text containing one native media marker per payload in `media`. |
| `media` | `Sequence[bytes]` | `()` | Encoded image/audio files, in marker order. Paths and URLs must be loaded into bytes first. |
| `parts` | `Optional[Sequence[Union[str, bytes]]]` | `None` | Ordered text and media bytes, without manually inserting markers. Use instead of `prompt`/`media`. |
| `normalize` | `Union[bool, int]` | `2` | Same normalization modes as `embed()`; `True` selects L2 and `False` preserves raw values. |
| `return_count` | `bool` | `False` | Return `(vector, total_token_count)`, including both text and media tokens. |

The result is an owned `List[float]` with the model's output embedding width,
which may differ from its input feature width. EmbeddingGemma 2 projects
512-dimensional input features to a 768-dimensional output vector.
`create_embedding()` remains a text API; call `embed_multimodal()` directly for media.

The projector initializes automatically on first use. Native library loading
checks the required MTMD symbols; context initialization checks model binding.
Closed or uninitialized objects, unsupported pooling and unsupported
media raise before encoding or decoding. Images and audio are detected from
their encoded bytes. Video bytes are currently rejected by this embedding path.

The complete input must fit `n_ctx`, `n_batch` and `n_ubatch`, including media
feature tokens. There is no automatic truncation or splitting of a multimodal
embedding request. MTMD may batch compatible media chunks separately during
encoding; this does not relax the decoder microbatch limit.

The native media helper can temporarily enable bidirectional attention for a
media chunk (for example, some Gemma projectors). Joint pooled decoding cannot
apply that chunk-local switch to a causal model. Such requests raise
`NotImplementedError` instead of silently changing the attention semantics.
Globally bidirectional embedding models and causal projectors such as
Qwen3-VL-Embedding retain their existing behavior.

Requests reset inference state once validation/tokenization succeeds and during
final cleanup. Embedding output is explicitly enabled before decoding. Native
chunks, bitmaps and encoding batches are released on success or failure; returned
vectors remain valid. Closing the model releases its projector before the model
context. Do not share the model or handler with concurrent generation requests.

See the [multimodal workflow](../features/embeddings-rerank.md#multimodal-embeddings-with-embeddinggemma-2)
for a complete example and task-prefix conventions.

### Shared MTMD encoding interfaces

`MTMDBaseHandler` in `llama_cpp.llama_multimodal` exposes media preparation for
custom consumers. With an open model and handler, use either
`handler.tokenize(model, prompt, media)` or `handler.tokenize_parts(model, parts)`
as a context manager. Both accept `add_special=True` and `parse_special=True`.
Call `request.encode()` inside that scope; the request cannot be encoded after
the scope exits.

`request.n_tokens` counts input rows and `request.n_pos` counts decoder positions.
The returned tuple of `MTMDEncodedChunk` objects preserves input order. Each chunk
holds text `tokens` or a copied, read-only float32 `embeddings` matrix, absolute
decoder `positions`, its position span `n_pos`, and `use_non_causal` metadata.
Text positions are scalars, as required by native token entries, even for M-RoPE
models. Media positions have one or four axes according to the projector.
These owned results
remain usable after request cleanup and subsequent encoding.

`MTMDEmbeddingHandler.embed(model, ...)` applies pooling and normalization on top
of these interfaces. A standalone handler binds to its first model and must be
closed before that model; it cannot be reused with a different model.

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
5. For a two-output classifier head, uses the first output as the relevance score, without softmax. Its label meaning depends on the model.

### `create_embedding(input, model=None, normalize=NORM_MODE_EUCLIDEAN, output_format="json")`

**Description**: High-level API compatible with OpenAI format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `Union[str, List[str]]` | — | Input text or list of texts. |
| `model` | str | None | Model name (optional, uses `self.model_path` if None). |
| `normalize` | int | `NORM_MODE_EUCLIDEAN` (2) | Normalization mode. |
| `output_format` | str | "json" | Output format: 'json', 'json+', or 'array'. |

**Output Formats:**
- `'json'`: OpenAI-style response dictionary containing a `data` list.
- `'json+'`: The same response, with `cosineSimilarity` when at least two compatible vectors are available.
- `'array'`: Raw list with one result per input, including a single string. Sequence pooling returns `List[List[float]]`; NONE pooling returns token matrices.

**Returns**: Data structure according to `output_format`.

## Deprecated / Changed APIs

- **Note**: The TODO comments `# TODO(JamePeng): Needs more extensive testing with various embedding and reranking models.` indicate that support for various embedding and reranking models may be incomplete. Further testing is recommended.

## Best Practices & Common Patterns

1. **Select Correct `pooling_type`**:
   - Standard embeddings: `LLAMA_POOLING_TYPE_UNSPECIFIED (-1)`.
   - Reranker models: `LLAMA_POOLING_TYPE_RANK (4)`.
   - Token-level embeddings: `LLAMA_POOLING_TYPE_NONE (0)`.

2. **Batch Optimization for Large Datasets**:
   - Adjust `n_batch`, `n_ubatch`, and `n_seq_max` to balance parallelism,
     performance, and memory.
   - If `seq_id` exceeds the configured capacity, increase `n_seq_max` to at
     least `seq_id + 1`.
   - Batching limits active native inputs; large returned collections can still exhaust host memory.

3. **Normalization Selection**:
   - Vector databases typically prefer L2 normalization (Euclidean), but other norms may be needed in specific scenarios.

4. **Reranker Models**:
   - Ensure `pooling_type` is set to `LLAMA_POOLING_TYPE_RANK`.
   - `rank()` returns scalar scores; direct `embed()` can return classifier vectors for multi-output heads.

5. **Performance Tuning**:
   - For GPU acceleration, set `n_gpu_layers` to -1 (recommended).
   - Use `verbose=True` for debugging configuration.

## Example Code

### 1. Text Embeddings (Vector Search)

To generate embeddings, use the `LlamaEmbedding` class. It automatically configures the model for vector generation.

```python
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_UNSPECIFIED

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(
    model_path="path/to/bge-m3.gguf",
    n_gpu_layers=-1,
    pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED,
    n_seq_max=128,
)

# 1. Simple usage (OpenAI-compatible format)
response = llm.create_embedding("Hello, world!")
print(response['data'][0]['embedding'])

# 2. Batch processing (High Performance)
# Native inference is batched; the returned vectors still occupy host memory.
documents = ["Hello, world!", "Goodbye, world!", "Llama is cute."] * 100
embeddings = llm.embed(documents) # Returns a list of lists (vectors)

print(f"Generated {len(embeddings)} vectors.")
```

**Advanced Output Formats:**
You can request raw arrays or cosine similarity matrices directly:

```python
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_UNSPECIFIED

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(model_path="path/to/bge-m3.gguf", n_gpu_layers=-1, pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED)

# Returns a list containing one sequence vector
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
| NORM_MODE_PNORM | $6$ (or any integer $p>2$) | p-norm | $\Large{x_i \over\sqrt[p]{\sum \lvert x_i\rvert^p}}$

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
- Performance issues can be addressed by adjusting `n_batch`, `n_ubatch`,
  `n_seq_max`, and `n_gpu_layers`.
- For custom models, manual `pooling_type` configuration may be required to match model behavior.

## Related Links

* [Wiki index](../index.md)
* [Llama Core](../core/Llama.md)
