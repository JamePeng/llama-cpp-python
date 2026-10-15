"""Qwen3.5 image chat with built-in MTP heads and a matching MTMD projector.

The complete prompt must fit n_ctx. With MTP enabled, each request rebuilds
target and draft state and forwards media batches and NextN states automatically.
Use --no-mtp with the same image and prompt for an ordinary-decoding baseline.
"""

import argparse
import base64
from contextlib import ExitStack
import mimetypes
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Target GGUF containing MTP heads")
    parser.add_argument("--mmproj", type=Path, required=True, help="Matching vision projector GGUF")
    parser.add_argument("--image", type=Path, required=True, help="Local PNG, JPEG or WebP image")
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Target offload layers; default -1 (all)")
    parser.add_argument("--projector-gpu", action=argparse.BooleanOptionalAction,
                        default=True, help="Run the projector on GPU (default: enabled)")
    parser.add_argument("--image-max-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--mtp", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable MTP proposals (default: enabled)")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--draft-min", type=int, default=0)
    parser.add_argument("--draft-p-min", type=float, default=0.0)
    args = parser.parse_args()
    for path in (args.model, args.mmproj, args.image):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    mime = mimetypes.guess_type(args.image.name)[0] or "image/png"
    image_url = f"data:{mime};base64," + base64.b64encode(args.image.read_bytes()).decode("ascii")

    from llama_cpp import Llama
    from llama_cpp.llama_multimodal import Qwen35ChatHandler
    from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

    class TimedLlama(Llama):
        def generate(self, *args, **kwargs):
            # Time each next() in both modes: MTMD prefill precedes generate(),
            # and time spent by the caller handling a yielded token is excluded.
            tokens, active_seconds, first_seconds = 0, 0.0, 0.0
            self.decode_timing = (tokens, active_seconds, first_seconds)
            iterator = super().generate(*args, **kwargs)
            try:
                while True:
                    started = time.perf_counter()
                    try:
                        token = next(iterator)
                    except StopIteration:
                        return
                    active_seconds += time.perf_counter() - started
                    tokens += 1
                    if tokens == 1:
                        first_seconds = active_seconds
                    self.decode_timing = (tokens, active_seconds, first_seconds)
                    yield token
            finally:
                iterator.close()

    config = SpecConfig(spec_type=SpeculativeType.DRAFT_MTP,
                        draft_n_max=args.draft_tokens, draft_n_min=args.draft_min,
                        draft_p_min=args.draft_p_min)
    with ExitStack() as cleanup:
        handler = Qwen35ChatHandler(
            mmproj_path=str(args.mmproj), use_gpu=args.projector_gpu,
            image_max_tokens=args.image_max_tokens, enable_thinking=False, verbose=False,
        )
        cleanup.callback(handler.close)
        llm = TimedLlama(
            model_path=str(args.model), chat_handler=handler, n_ctx=args.n_ctx,
            n_batch=args.n_batch, n_ubatch=args.n_batch, n_gpu_layers=args.n_gpu_layers,
            ctx_checkpoints=4, verbose=False,
            speculative=config if args.mtp else None,
        )
        cleanup.callback(llm.close)
        print(
            f"Decode mode: {'MTP' if args.mtp else 'ordinary (--no-mtp)'}", flush=True,
        )
        if args.mtp:
            print(
                f"MTP config: type={config.spec_type.name}, source=built-in heads, "
                f"draft_n_max={config.draft_n_max}, draft_n_min={config.draft_n_min}, "
                f"draft_p_min={config.draft_p_min:g}, "
                f"backend_sampling_requested={config.draft_backend_sampling}",
                flush=True,
            )
        print(
            f"Runtime: n_ctx={llm.n_ctx()}, n_batch={llm.n_batch}, "
            f"target_gpu_layers={args.n_gpu_layers}, projector_gpu={args.projector_gpu}, "
            f"image_max_tokens={args.image_max_tokens}, max_tokens={args.max_tokens}, "
            "temperature=0, seed=42",
            flush=True,
        )
        started = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": args.prompt},
            ]}], max_tokens=args.max_tokens, temperature=0, seed=42,
        )
        elapsed = time.perf_counter() - started
        print(response["choices"][0]["message"]["content"])
        stats = llm.last_speculative_stats
        if args.mtp:
            print(
                f"MTP: drafted={stats['drafted']}, verified={stats['verified']}, "
                f"accepted={stats['accepted']}, verified_acceptance={stats['acceptance_rate']:.1%}"
            )
            print(f"Verification: steps={stats['verification_steps']}, rollbacks={stats['rollbacks']}")
        tokens, seconds, first_seconds = llm.decode_timing
        rate = tokens / seconds if seconds > 0 else 0.0
        sustained_seconds = seconds - first_seconds
        sustained_rate = (tokens - 1) / sustained_seconds if sustained_seconds > 0 else 0.0
        print(
            f"Decode (excludes media prefill): {tokens} tokens / "
            f"{seconds:.3f} s = {rate:.2f} tok/s"
        )
        print(
            f"Sustained decode (excludes first token): {sustained_rate:.2f} tok/s; "
            f"decode TTFT={first_seconds * 1000:.2f} ms"
        )
        if args.mtp:
            print(
                f"Decode phases: draft={stats['draft_seconds']:.3f} s, "
                f"target_decode={stats['target_decode_seconds']:.3f} s, "
                f"target_sync={stats['target_sync_seconds']:.3f} s, "
                f"process={stats['process_seconds']:.3f} s, accept={stats['accept_seconds']:.3f} s"
            )
        print(f"Request wall time (includes media prefill, excludes model loading): {elapsed:.3f} s")


if __name__ == "__main__":
    main()
