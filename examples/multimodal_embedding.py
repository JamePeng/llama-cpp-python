"""EmbeddingGemma 2 text/image/audio embeddings with LlamaEmbedding.

Pass one <__media__> marker per image/audio file in --prompt, in file order.
Use task prefixes from the model card; do not add a chat generation prompt.
The full input must fit n_ctx/n_batch/n_ubatch for bidirectional attention.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from llama_cpp import Llama, llama_cpp as lib
from llama_cpp.llama_embedding import LlamaEmbedding
from llama_cpp.llama_multimodal import MTMDEmbeddingHandler


def embed_multimodal(model: Llama, encoder: MTMDEmbeddingHandler, prompt: str,
                     media: tuple[bytes, ...] = ()) -> np.ndarray:
    """Return an owned L2 vector through the reusable embedding handler."""
    return np.asarray(encoder.embed(model, prompt, media), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--prompt", default="title: none | text: hello")
    parser.add_argument("--media", nargs="*", default=[])
    parser.add_argument("--n-ctx", type=int, default=1024)
    args = parser.parse_args()
    with ExitStack() as cleanup:
        encoder = MTMDEmbeddingHandler(mmproj_path=args.mmproj, use_gpu=False,
                                      image_max_tokens=70, verbose=False)
        cleanup.callback(encoder.close)
        model = LlamaEmbedding(args.model, n_gpu_layers=0, embedding_handler=encoder,
                               pooling_type=lib.LLAMA_POOLING_TYPE_MEAN, n_ctx=args.n_ctx,
                               n_batch=args.n_ctx, n_ubatch=args.n_ctx, verbose=False)
        cleanup.callback(model.close)
        vector = model.embed_multimodal(args.prompt,
                                       tuple(Path(path).read_bytes() for path in args.media))
        print(vector)


if __name__ == "__main__":
    main()
