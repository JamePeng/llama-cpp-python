"""EmbeddingGemma 2 and Qwen3-VL text/media embeddings with LlamaEmbedding.

Pass one <__media__> marker per image/audio file in --prompt, in file order.
EmbeddingGemma 2 uses task prefixes; --profile qwen3-vl wraps the instruction,
user input and assistant suffix required by Qwen3-VL-Embedding.
The full input must fit n_ctx/n_batch/n_ubatch.
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
    parser.add_argument("--profile", choices=("embeddinggemma2", "qwen3-vl"), default="embeddinggemma2")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--instruction", default="Represent the user's input.")
    parser.add_argument("--media", nargs="*", default=[])
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="0 for CPU, -1 for all GPU layers")
    parser.add_argument("--image-max-tokens", type=int, default=None)
    args = parser.parse_args()
    qwen = args.profile == "qwen3-vl"
    prompt = args.prompt if args.prompt is not None else ("hello" if qwen else "title: none | text: hello")
    if qwen:
        prompt = (f"<|im_start|>system\n{args.instruction}<|im_end|>\n"
                  f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    with ExitStack() as cleanup:
        encoder = MTMDEmbeddingHandler(mmproj_path=args.mmproj, use_gpu=args.n_gpu_layers != 0,
                                      image_max_tokens=args.image_max_tokens if args.image_max_tokens is not None else (64 if qwen else 70),
                                      verbose=False)
        cleanup.callback(encoder.close)
        model = LlamaEmbedding(args.model, n_gpu_layers=args.n_gpu_layers, embedding_handler=encoder,
                               pooling_type=lib.LLAMA_POOLING_TYPE_LAST if qwen else lib.LLAMA_POOLING_TYPE_MEAN, n_ctx=args.n_ctx,
                               n_batch=args.n_ctx, n_ubatch=args.n_ctx, verbose=False)
        cleanup.callback(model.close)
        vector = model.embed_multimodal(prompt,
                                       tuple(Path(path).read_bytes() for path in args.media))
        print(vector)


if __name__ == "__main__":
    main()
