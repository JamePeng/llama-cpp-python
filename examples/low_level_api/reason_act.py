"""A small reason/action loop with one safe calculator tool."""

from __future__ import annotations

import argparse
import ast
import math
import operator
import re

from common import (
    HelpFormatter,
    add_generation_arguments,
    add_model_arguments,
    run_cli,
    validate_generation,
    validate_model_arguments,
    validate_positive,
)

SYSTEM_PROMPT = """You can answer directly or use a calculator.
To calculate, output exactly: Action: calculate[expression]
After receiving an Observation, answer the original question.
Do not invent an Observation."""

ACTION_PATTERN = re.compile(r"Action:\s*calculate\[([^\]\n]+)\]", re.IGNORECASE)
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> float:
    def checked(value: object) -> float:
        if type(value) is int and value.bit_length() <= 1024:
            return value
        if type(value) is float and math.isfinite(value):
            return value
        raise ValueError("result is outside the calculator's safe range")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return checked(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent is outside the calculator's safe range")
            return checked(BINARY_OPERATORS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
            return checked(UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
        raise ValueError("only numbers and arithmetic operators are allowed")

    if len(expression) > 120:
        raise ValueError("expression is too long")
    return evaluate(ast.parse(expression, mode="eval").body)


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Low-level reason/action loop with a calculator tool.",
        formatter_class=HelpFormatter,
        epilog="""Example:
  python examples/low_level_api/reason_act.py -m instruct-model.gguf --n-ctx 4096 --max-tokens 256 "What is (17 * 23) / 4?"
""",
    )
    add_model_arguments(parser)
    add_generation_arguments(parser, max_tokens=256)
    parser.add_argument("question", nargs="?", default="What is 4 * 7 / 3?")
    parser.add_argument("--max-steps", type=int, default=3)
    return parser, parser.parse_args()


def main() -> int:
    parser, args = parse_args()
    validate_model_arguments(parser, args)
    validate_positive(parser, max_tokens=args.max_tokens, max_steps=args.max_steps)
    validate_generation(parser, args)

    from runtime import LowLevelLlama

    messages: list[tuple[str, str]] = [
        ("system", SYSTEM_PROMPT),
        ("user", args.question),
    ]
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
        print(f"Question: {args.question}")
        for _ in range(args.max_steps):
            prompt = model.render_chat(messages)
            chunks: list[str] = []
            print("Assistant: ", end="", flush=True)
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
            response = "".join(chunks).strip()
            messages.append(("assistant", response))

            match = ACTION_PATTERN.search(response)
            if not match:
                return 0
            try:
                observation = str(calculate(match.group(1)))
            except (ArithmeticError, SyntaxError, ValueError) as exc:
                observation = f"calculator error: {exc}"
            print(f"Tool: {observation}")
            # A user observation works with templates that have no tool role.
            messages.append(
                (
                    "user",
                    f"Observation: {observation}\nNow answer the original question.",
                )
            )

    print("Stopped after reaching --max-steps.")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
