import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
import llama_cpp.llama_cpp as llama_cpp_lib
from .llama_types import Embedding
from .llama import Llama
# Pooling types from .llama_cpp
from .llama_cpp import (
    LLAMA_POOLING_TYPE_UNSPECIFIED,
    LLAMA_POOLING_TYPE_NONE,
    LLAMA_POOLING_TYPE_MEAN,
    LLAMA_POOLING_TYPE_CLS,
    LLAMA_POOLING_TYPE_LAST,
    LLAMA_POOLING_TYPE_RANK, # Specifically for Reranking models
)

# Normalization modes for embedding vectors
# See: https://github.com/ggml-org/llama.cpp/tree/master/examples/embedding#--embd-normalize-integer
NORM_MODE_NONE = -1
NORM_MODE_MAX_INT16 = 0
NORM_MODE_TAXICAB = 1
NORM_MODE_EUCLIDEAN = 2
NORM_MODE_PNORM = 6

# TODO(JamePeng): Needs more extensive testing with various embedding and reranking models.
class LlamaEmbedding(Llama):
    """
    A specialized class for high-performance Text Embedding and Reranking.
    Inherits from the base Llama class but is optimized for vector operations.

    Key Features:
    1. Auto-configuration: Automatically sets embeddings=True.
    2. Streaming Batch: Handles massive datasets without OOM (Out Of Memory).
    3. Native Reranking Support: Specifically handles `LLAMA_POOLING_TYPE_RANK` models (like BGE-Reranker, Qwen3-Reranker). /
       It correctly identifies classification heads to output scalar relevance scores instead of high-dimensional vectors.
    4. Advanced Normalization: Implements MaxInt16, Taxicab (L1), and Euclidean (L2) normalization strategies /
       using NumPy for optimal performance and compatibility with various vector databases.
    """

    def __init__(
            self,
            model_path: str,
            n_ctx: int = 0,
            n_batch: int = 512,
            n_ubatch: int = 512,
            pooling_type: int = LLAMA_POOLING_TYPE_UNSPECIFIED,
            n_gpu_layers: int = 0,
            verbose: bool = True,
            **kwargs):
        """
        Initialize the embedding model with enforced configuration.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Text context, 0 = from model
            n_batch: Prompt processing maximum batch size
            n_ubatch: Physical batch size
            pooling_type: The pooling strategy used by the model.
                          - Use `LLAMA_POOLING_TYPE_RANK` (4) for Reranker models.
                          - Use `LLAMA_POOLING_TYPE_UNSPECIFIED` (-1) to let the model metadata decide (for standard embeddings).
            n_gpu_layers: Number of model layers to offload to GPU.
                          - Set to 0 for CPU only.
                          - Set to -1 for all layers (recommended for best performance).
            **kwargs: Additional arguments passed to the Llama base class (e.g., n_batch, n_ctx, verbose).
        """
        kwargs["embeddings"] = True
        kwargs["n_gpu_layers"] = n_gpu_layers
        kwargs["n_ctx"] = n_ctx
        kwargs["n_batch"] = n_batch
        kwargs["n_ubatch"] = n_ubatch
        kwargs["verbose"] = verbose

        # Enable Unified KV Cache (Crucial for Batching)
        # This allows us to assign arbitrary seq_ids in a batch, enabling the parallel /
        #     encoding of multiple unrelated documents without "invalid seq_id" errors.
        kwargs["kv_unified"] = True

        # Set pooling type
        kwargs["pooling_type"] = pooling_type

        super().__init__(model_path=model_path, **kwargs)

        if self.verbose:
            print(f"LlamaEmbedding initialized with pooling_type: {self.pooling_type()}")

    def embed(
        self,
        input: Union[str, List[str], List[List[int]]],
        normalize: int = NORM_MODE_EUCLIDEAN,
        truncate: bool = True,
        separator: Optional[str] = None,
        return_count: bool = False,
    ) -> Union[List[float], List[List[float]], Tuple[Any, int]]:

        return super().embed(
            input, normalize=normalize, truncate=truncate,
            separator=separator or None, return_count=return_count,
        )

    def rank(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate relevance scores for a list of documents against a query using a Reranking model.

        This method follows the implementation logic of the latest llama.cpp embedding example,
        supporting both specialized chat templates and manual sequence construction.

        Link: https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp

        Args:
            query: The search query string.
            documents: A list of candidate document strings to be scored.

        Returns:
            A list of float scores, where higher values indicate greater relevance.
        """
        # Ensure the model is configured for Reranking (Cross-Encoding)
        if self.pooling_type() != LLAMA_POOLING_TYPE_RANK:
            raise ValueError(f"Model pooling_type is {self.pooling_type()}, but LLAMA_POOLING_TYPE_RANK is required.")

        # 1. Attempt to retrieve the built-in 'rerank' chat template from model metadata.
        # Modern GGUF models often include a template for formatting query/document pairs.
        rerank_template = self._model.model_chat_template(b"rerank")

        batch_inputs: List[List[int]] = []

        # 2. Case A: Using Model-Specific Template
        # If a template exists, we perform dynamic string replacement for {query} and {document}.
        if rerank_template:
            for doc in documents:
                final_prompt = rerank_template.replace("{query}", query).replace("{document}", doc)
                # Tokenize the full formatted prompt. Template usually dictates BOS/EOS placement.
                tokens = self.tokenize(final_prompt.encode("utf-8"), add_bos=False, special=True)
                batch_inputs.append(tokens)

        # 3. Case B: Manual Sequence Construction (Fallback)
        # If no template is found, construct the standard [BOS] Query [SEP] Doc [EOS] sequence.
        else:
            # Determine separator and end-of-sequence tokens
            sep_id = self.token_sep() if self.token_sep() != -1 else self.token_eos()
            eos_id = self.token_eos()

            # Pre-tokenize the query with BOS (Beginning of Sequence)
            q_tokens = self.tokenize(query.encode("utf-8"), add_bos=True, special=True)

            # Remove the automatically added EOS token from the query to allow concatenation.
            if q_tokens and q_tokens[-1] == eos_id:
                q_tokens.pop()

            for doc in documents:
                # Tokenize document without an additional BOS token
                d_tokens = self.tokenize(doc.encode("utf-8"), add_bos=False, special=True)

                # Combine: [BOS] Query [SEP] Document
                full_seq = q_tokens + [sep_id] + d_tokens

                # Ensure the sequence is properly terminated with an EOS token for inference.
                if not full_seq or full_seq[-1] != eos_id:
                    full_seq.append(eos_id)

                batch_inputs.append(full_seq)

        # Execute embedding inference. Rerankers output raw logits/scores, so we skip normalization.
        raw_results = self.embed(batch_inputs, normalize=NORM_MODE_NONE)
        results_list = [raw_results] if (len(batch_inputs) == 1 and isinstance(raw_results[0], float)) else raw_results

        # 5. Output Post-Processing
        # For generative rerankers like Qwen3-Reranker, output dim is 2 ([yes_logit, no_logit]).
        final_scores = []
        # Ensure we iterate through results (embed returns List[Any] for batch inputs)
        for res in results_list:
            if isinstance(res, (list, np.ndarray)) and len(res) == 2:
                final_scores.append(float(res[0])) # Standard scalar score in list form yes_logit
            else:
                final_scores.append(float(res))    # Raw scalar score

        return final_scores

    def create_embedding(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        normalize: int = NORM_MODE_EUCLIDEAN,
        output_format: str = "json"
    ) -> Union[Dict[str, Any], List[float], List[List[float]]]:
        """
        High-level API compatible with OpenAI format.

        Args:
            output_format:
                - 'json': OpenAI style dict (Default)
                - 'json+': OpenAI style dict + cosineSimilarity matrix
                - 'array': Raw python list (List[float] or List[List[float]])
        """
        model_name = model if model is not None else self.model_path

        # Normalize input to list
        inputs_list = [input] if isinstance(input, str) else input

        # Generate Embeddings(and get token count)
        embeddings, token_count = self.embed(
            inputs_list,
            normalize=normalize,
            return_count=True
        )

        if output_format == "array":
            return embeddings

        # Structure the OpenAI-style response ('json' or 'json+')
        # Ensure embeddings is a list for iteration
        # (If input was single string, embeddings is List[float], wrap it for the loop)
        iter_embeddings = [embeddings] if isinstance(embeddings[0], float) else embeddings

        data: List[Embedding] = [
            {
                "object": "embedding",
                "embedding": emb,
                "index": idx,
            }
            for idx, emb in enumerate(iter_embeddings)
        ]

        response = {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": token_count,  # Input consumption
                "completion_tokens": 0,        # The Embedding task does not generate text, so the value is 0.
                "total_tokens": token_count,   # Total consumption = Input consumption + Output
            }
        }

        # Calculate Cosine Similarity Matrix (Optimized via Numpy)
        # Only if output_format is 'json+' and we have vectors
        if output_format == "json+" and len(embeddings) > 1 and isinstance(embeddings[0], list):
            try:
                # Assuming embeddings are already L2 normalized if normalize=2
                mat = np.array(embeddings)

                # Safety check: Force normalize if not already done, to ensure Cosine (not Dot Product)
                if normalize != NORM_MODE_EUCLIDEAN:
                    norm = np.linalg.norm(mat, axis=1, keepdims=True)
                    # Avoid division by zero
                    norm[norm == 0] = 1e-10
                    mat = mat / norm

                # Matrix multiplication: A @ A.T
                sim_matrix = np.dot(mat, mat.T)
                response["cosineSimilarity"] = sim_matrix.tolist()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to calculate similarity matrix: {e}")

        return response
