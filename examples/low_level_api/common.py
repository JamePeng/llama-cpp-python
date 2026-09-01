from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Keep examples readable while showing defaults."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help or ""
        return super()._get_help_string(action)


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def gpu_layers(value: str) -> int:
    names = {"auto": -1, "all": -2}
    if value.lower() in names:
        return names[value.lower()]
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use an integer, 'auto', or 'all'") from exc


def micro_batch(value: str) -> int:
    if value.lower() == "auto":
        return 0
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use a positive integer or 'auto'") from exc


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable debug-level native logs (compatibility switch).",
    )
    parser.add_argument(
        "--verbosity",
        metavar="LEVEL",
        default=None,
        help="Native log level 0-5 or output/error/warning/info/trace/debug.",
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m", "--model", required=True, type=existing_file, help="GGUF model path."
    )
    parser.add_argument(
        "--n-ctx", type=int, default=4096, help="Context size in tokens."
    )
    parser.add_argument(
        "--n-batch", type=int, default=512, help="Maximum prompt decode batch size."
    )
    parser.add_argument(
        "--n-ubatch",
        type=micro_batch,
        default="auto",
        metavar="N|auto",
        help="Physical micro-batch size; auto uses at most 512 tokens.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Threads used for generation and prompt decoding.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=gpu_layers,
        default="auto",
        metavar="N|auto|all",
        help="Maximum layers to offload: an integer, 'auto', or 'all'.",
    )
    add_logging_arguments(parser)


def add_generation_arguments(
    parser: argparse.ArgumentParser, *, max_tokens: int = 256
) -> None:
    parser.add_argument(
        "-n",
        "--max-tokens",
        type=int,
        default=max_tokens,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Zero selects greedy output."
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="Keep the K most likely tokens."
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Nucleus sampling probability."
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to recent generated tokens.",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Negative values select a random seed."
    )


def validate_positive(parser: argparse.ArgumentParser, **values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")


def validate_generation(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.temperature < 0:
        parser.error("--temperature must be zero or greater")
    if args.top_k < 0:
        parser.error("--top-k must be zero or greater")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than zero and at most one")
    if args.repeat_penalty <= 0:
        parser.error("--repeat-penalty must be greater than zero")
    if args.seed > 0xFFFFFFFF:
        parser.error("--seed must fit in an unsigned 32-bit integer")


def validate_model_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n_ubatch == 0:
        args.n_ubatch = min(args.n_batch, 512)
    validate_positive(
        parser,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        threads=args.threads,
    )
    if args.n_batch > args.n_ctx:
        parser.error("--n-batch must not exceed --n-ctx")
    if args.n_ubatch > args.n_batch:
        parser.error("--n-ubatch must not exceed --n-batch")


def run_cli(entrypoint: Callable[[], int]) -> int:
    try:
        return entrypoint()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
