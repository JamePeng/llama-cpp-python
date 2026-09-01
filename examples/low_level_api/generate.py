"""Stream a text completion with the current low-level llama.cpp API."""

from __future__ import annotations

import argparse

from common import (
    HelpFormatter,
    add_generation_arguments,
    add_model_arguments,
    run_cli,
    validate_generation,
    validate_model_arguments,
    validate_positive,
)


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Low-level streaming text generation.",
        formatter_class=HelpFormatter,
        epilog="""Example:
  python examples/low_level_api/generate.py -m model.gguf -p "Explain why the sky is blue:" --n-ctx 4096 --max-tokens 256 --n-gpu-layers all
""",
    )
    add_model_arguments(parser)
    add_generation_arguments(parser)
    parser.add_argument(
        "-p",
        "--prompt",
        default="Question: Why is the sky blue?\nAnswer:",
    )
    parser.add_argument("--show-timings", action="store_true")
    return parser, parser.parse_args()


def main() -> int:
    parser, args = parse_args()
    validate_model_arguments(parser, args)
    validate_positive(parser, max_tokens=args.max_tokens)
    validate_generation(parser, args)

    # Delay native imports so `-h` stays fast and machine-independent.
    from runtime import LowLevelLlama

    with LowLevelLlama(
        args.model,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
        verbosity=args.verbosity,
    ) as model:
        print(args.prompt, end="", flush=True)
        for text in model.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            seed=args.seed,
        ):
            print(text, end="", flush=True)
        print()
        if args.show_timings:
            model.print_timings()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
