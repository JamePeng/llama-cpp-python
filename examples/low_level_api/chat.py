"""Interactive chat using the template stored in a GGUF model."""

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
        description="Low-level interactive chat.",
        formatter_class=HelpFormatter,
        epilog="""Commands:
  /reset  clear the conversation
  /exit   leave the chat

Example:
  python examples/low_level_api/chat.py -m instruct-model.gguf --n-ctx 4096 --max-tokens 512 --n-gpu-layers all
""",
    )
    add_model_arguments(parser)
    add_generation_arguments(parser, max_tokens=512)
    parser.add_argument(
        "--system",
        default="You are a helpful, concise assistant.",
        help="System message; pass an empty string to disable it.",
    )
    return parser, parser.parse_args()


def main() -> int:
    parser, args = parse_args()
    validate_model_arguments(parser, args)
    validate_positive(parser, max_tokens=args.max_tokens)
    validate_generation(parser, args)

    from runtime import LowLevelLlama

    messages: list[tuple[str, str]] = []
    if args.system:
        messages.append(("system", args.system))

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
        print("Chat ready. Enter /reset or /exit at any time.")
        while True:
            try:
                user_text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_text:
                continue
            if user_text == "/exit":
                break
            if user_text == "/reset":
                messages = [("system", args.system)] if args.system else []
                print("Conversation cleared.")
                continue

            messages.append(("user", user_text))
            try:
                # Re-render history so the model's own template supplies delimiters.
                prompt = model.render_chat(messages)
                print("Assistant: ", end="", flush=True)
                chunks: list[str] = []
                for text in model.generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty,
                    seed=args.seed,
                    add_special=False,
                    parse_special=True,
                ):
                    chunks.append(text)
                    print(text, end="", flush=True)
                print()
                messages.append(("assistant", "".join(chunks)))
            except (RuntimeError, ValueError) as exc:
                messages.pop()
                print(f"Error: {exc}")
                print("Use /reset, shorten the conversation, or increase --n-ctx.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
