"""Run one video-understanding request through the high-level MTMD API.

Run this script with ``-h`` or ``--help`` for setup guidance and examples.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Union

# Prefer this checkout over an independently installed llama_cpp package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve guide formatting and display argument defaults."""


def existing_file(value: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return str(path)


def existing_directory(value: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return str(path)


def n_gpu_layers(value: str) -> Union[int, str]:
    if value in ("auto", "all"):
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer, 'auto', or 'all'"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test video understanding through llama-cpp-python MTMD.",
        formatter_class=HelpFormatter,
        epilog="""
Quick start:
  1. Use a main GGUF model and its matching multimodal projector (mmproj).
  2. Make ffmpeg and ffprobe available on PATH, or pass --ffmpeg-bin-dir.
  3. Start with --fps 1 for a short video, then adjust FPS and context size.

Minimal example (ffmpeg and ffprobe are on PATH):
  python examples/high_level_api/mtmd_video_chat.py --model path/to/model.gguf --mmproj path/to/mmproj.gguf --video path/to/sample.mp4

Example with an explicit ffmpeg directory:
  python examples/high_level_api/mtmd_video_chat.py --model path/to/model.gguf --mmproj path/to/mmproj.gguf --video path/to/sample.mp4 --ffmpeg-bin-dir path/to/ffmpeg/bin --fps 1

Template compatibility:
  Gemma 4 expects --video-content-type video (the default). Use video_url only
  when the model's chat template explicitly supports that schema.

Resource guidance:
  Video frames are expanded during MTMD tokenization. Test with a short video
  and a low FPS first; long or high-resolution videos can require substantial
  RAM, context space, and preprocessing time.
""",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=existing_file,
        metavar="FILE",
        help="Path to the main GGUF language model.",
    )
    parser.add_argument(
        "--mmproj",
        required=True,
        type=existing_file,
        metavar="FILE",
        help="Path to the multimodal projector matching the main model.",
    )
    parser.add_argument(
        "--video",
        required=True,
        type=existing_file,
        metavar="FILE",
        help="Path to the local video file to analyze.",
    )
    parser.add_argument(
        "--ffmpeg-bin-dir",
        type=existing_directory,
        default=None,
        metavar="DIR",
        help="Directory containing ffmpeg and ffprobe; omit to search PATH.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Describe this video in detail."
        ),
        help="Question or instruction sent with the video.",
    )
    parser.add_argument(
        "--video-content-type",
        choices=("video", "video_url"),
        default="video",
        help=(
            "Message content schema passed to the model chat template. Gemma 4 "
            "expects 'video'; use 'video_url' only for templates that support it."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames sampled per second. Start low to control memory and tokens.",
    )
    parser.add_argument(
        "--timestamp-interval-ms",
        type=int,
        default=5000,
        help="Interval for inserting timestamp text; <= 0 disables timestamps.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=32768,
        help="Model context size, including text and video tokens.",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=2048,
        help="Maximum logical batch size used by the language model.",
    )
    parser.add_argument(
        "--batch-max-tokens",
        type=int,
        default=1024,
        help="Maximum MTMD media tokens processed in one decode batch.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=n_gpu_layers,
        default="auto",
        help="Layers offloaded to GPU: an integer, 'auto', or 'all'.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of response tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the response.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose llama.cpp and MTMD logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Delay the native library import so -h/--help stays fast and does not emit
    # machine-specific shared-library paths before argparse exits.
    from llama_cpp import Llama

    if args.fps <= 0:
        print(
            "Warning: --fps <= 0 uses the video's native FPS and may consume "
            "a very large amount of memory and context.",
            file=sys.stderr,
        )

    chat_handler_kwargs = {
        "video_fps_target": args.fps,
        "video_timestamp_interval_ms": args.timestamp_interval_ms,
        "batch_max_tokens": args.batch_max_tokens,
    }
    if args.ffmpeg_bin_dir is not None:
        chat_handler_kwargs["video_ffmpeg_bin_dir"] = args.ffmpeg_bin_dir

    print("MTMD video test", file=sys.stderr)
    print(f"  model:  {args.model}", file=sys.stderr)
    print(f"  mmproj: {args.mmproj}", file=sys.stderr)
    print(f"  video:  {args.video}", file=sys.stderr)
    print(f"  size:   {os.path.getsize(args.video) / (1024 * 1024):.2f} MiB", file=sys.stderr)
    print(f"  fps:    {args.fps}", file=sys.stderr)
    print(f"  schema: {args.video_content_type}", file=sys.stderr)
    print(
        f"  ffmpeg: {args.ffmpeg_bin_dir or 'PATH'}",
        file=sys.stderr,
    )

    llama = Llama(
        model_path=args.model,
        mmproj_path=args.mmproj,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
        chat_handler_kwargs=chat_handler_kwargs,
        verbosity=2,
        verbose=not args.quiet,
    )

    try:
        if args.video_content_type == "video":
            video_content = {
                "type": "video",
                "video": args.video,
            }
        else:
            video_content = {
                "type": "video_url",
                "video_url": {"url": args.video},
            }

        response = llama.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        video_content,
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=True,
        )

        print("\nAssistant:\n", end="", flush=True)
        for chunk in response:
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            text = delta.get("content") or delta.get("reasoning_content")
            if text:
                print(text, end="", flush=True)
    finally:
        llama.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
