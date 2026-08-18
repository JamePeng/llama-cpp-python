"""Compare ordinary decoding with built-in or external MTP decoding.

Run with ``-h`` to see portable command examples and every tuning option.

Author: JamePeng
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from llama_cpp import Llama


DecodeMode = Literal["ordinary", "builtin", "external"]


@dataclass(frozen=True)
class GenerationResult:
    tokens: tuple[int, ...]
    text: str
    elapsed: float
    drafted: int = 0
    verified: int = 0
    accepted: int = 0
    verification_steps: int = 0
    rollbacks: int = 0

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.elapsed if self.elapsed > 0.0 else 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    load_seconds: float
    prompt_tokens: int
    runs: tuple[GenerationResult, ...]

    @property
    def aggregate_tokens_per_second(self) -> float:
        elapsed = sum(run.elapsed for run in self.runs)
        tokens = sum(len(run.tokens) for run in self.runs)
        return tokens / elapsed if elapsed > 0.0 else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ordinary decoding against built-in MTP heads, an external "
            "MTP draft model, or both. The ordinary baseline is always measured."
        ),
        epilog="""examples:
  Built-in MTP:
    python -m examples.high_level_api.high_level_api_mtp_speculative --model model.gguf --mtp-mode builtin

  External MTP draft model:
    python -m examples.high_level_api.high_level_api_mtp_speculative --model model.gguf --mtp-mode external --draft-model mtp-model.gguf

  Compare all three modes:
    python -m examples.high_level_api.high_level_api_mtp_speculative --model model.gguf --mtp-mode both --draft-model mtp-model.gguf --draft-tokens 2 --runs 3
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the target GGUF model.",
    )
    parser.add_argument(
        "--mtp-mode",
        choices=("builtin", "external", "both"),
        default="builtin",
        help=(
            "MTP implementation to benchmark against ordinary decoding "
            "(default: builtin)."
        ),
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        help="External MTP GGUF model; required for --mtp-mode external or both.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Write a concise Python function that returns the first n Fibonacci "
            "numbers, followed by a short explanation:\n"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens in each measured run (default: 512).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured runs per decoding mode (default: 3).",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="Untimed tokens generated once per mode to initialize compute graphs.",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=4096, help="Context size (default: 4096)."
    )
    parser.add_argument(
        "--n-batch", type=int, default=512, help="Logical batch size (default: 512)."
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=3,
        help="Maximum MTP tokens proposed per draft step (default: 3).",
    )
    parser.add_argument(
        "--draft-p-min",
        type=float,
        default=0.0,
        help="Minimum probability for retaining an MTP proposal (default: 0).",
    )
    parser.add_argument(
        "--baseline-load-mtp",
        action="store_true",
        help="Diagnostic: load NextN weights for ordinary decoding without enabling MTP.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Keep this at 0 for deterministic output-equivalence checks.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Keep model layers on the CPU instead of offloading all layers.",
    )
    parser.add_argument(
        "--no-backend-sampling",
        action="store_true",
        help="Select MTP candidates on the CPU instead of the backend sampler.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed llama.cpp logging."
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def generate_once(
    llm: Llama,
    prompt_tokens: list[int],
    *,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    # reset=True below clears the native target context. reset() also clears the
    # stateful MTP draft context, preventing prefix-cache reuse from biasing A/B.
    llm.reset()
    generated: list[int] = []
    stream = llm.generate(
        prompt_tokens,
        temp=temperature,
        top_k=1 if temperature <= 0.0 else 40,
        top_p=1.0 if temperature <= 0.0 else 0.95,
        min_p=0.0,
        repeat_penalty=1.0,
        seed=42,
        reset=True,
    )
    started = time.perf_counter()
    try:
        for token in stream:
            if token == llm.token_eos():
                break
            generated.append(token)
            if len(generated) >= max_tokens:
                break
    finally:
        # Ensure Llama.generate's checkpoint/finalization path runs immediately.
        stream.close()
    elapsed = time.perf_counter() - started

    text = llm.detokenize(generated, prev_tokens=prompt_tokens).decode(
        "utf-8", errors="replace"
    )
    stats = llm.last_speculative_stats
    return GenerationResult(
        tuple(generated),
        text,
        elapsed,
        drafted=int(stats["drafted"]),
        verified=int(stats["verified"]),
        accepted=int(stats["accepted"]),
        verification_steps=int(stats["verification_steps"]),
        rollbacks=int(stats["rollbacks"]),
    )


def benchmark_mode(args: argparse.Namespace, mode: DecodeMode) -> BenchmarkResult:
    # Keep llama.cpp and its native libraries unloaded for lightweight CLI paths
    # such as -h/--help and argument validation failures.
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

    labels = {
        "ordinary": "ordinary",
        "builtin": "built-in MTP",
        "external": "external MTP",
    }
    label = labels[mode]
    speculative: Optional[SpecConfig] = None
    if mode != "ordinary":
        speculative = SpecConfig(
            spec_type=SpeculativeType.DRAFT_MTP,
            draft_n_max=args.draft_tokens,
            draft_p_min=args.draft_p_min,
            draft_model_path=(str(args.draft_model) if mode == "external" else None),
            draft_backend_sampling=not args.no_backend_sampling,
        )

    load_started = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=0 if args.cpu else "all",
        load_mtp=mode == "builtin" or (mode == "ordinary" and args.baseline_load_mtp),
        ctx_checkpoints=0,
        speculative=speculative,
        verbose=args.verbose,
    )
    load_seconds = time.perf_counter() - load_started

    try:
        prompt_tokens = llm.tokenize(
            args.prompt.encode("utf-8"), add_bos=True, special=True
        )
        if args.warmup_tokens:
            generate_once(
                llm,
                prompt_tokens,
                max_tokens=args.warmup_tokens,
                temperature=args.temperature,
            )

        results = tuple(
            generate_once(
                llm,
                prompt_tokens,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for _ in range(args.runs)
        )
    finally:
        llm.close()

    return BenchmarkResult(label, load_seconds, len(prompt_tokens), results)


def print_mode(result: BenchmarkResult) -> None:
    rates = [run.tokens_per_second for run in result.runs]
    print(f"\n[{result.label}]")
    print(f"  model load       : {result.load_seconds:.3f} s (not benchmarked)")
    for index, run in enumerate(result.runs, 1):
        print(
            f"  run {index:<2}          : {len(run.tokens):>4} tokens / "
            f"{run.elapsed:.3f} s = {run.tokens_per_second:.2f} token/s"
        )
    print(f"  aggregate        : {result.aggregate_tokens_per_second:.2f} token/s")
    print(f"  median run       : {statistics.median(rates):.2f} token/s")
    verified = sum(run.verified for run in result.runs)
    accepted = sum(run.accepted for run in result.runs)
    if verified:
        drafted = sum(run.drafted for run in result.runs)
        steps = sum(run.verification_steps for run in result.runs)
        rollbacks = sum(run.rollbacks for run in result.runs)
        print(f"  draft proposed   : {drafted}")
        print(f"  draft verified   : {verified} in {steps} steps")
        print(f"  draft accepted   : {accepted} ({accepted / verified:.1%})")
        print(f"  draft rollbacks  : {rollbacks}")


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.model = args.model.expanduser().resolve()
    if not args.model.is_file():
        parser.error(f"target GGUF model not found: {args.model}")

    if args.draft_model is not None:
        args.draft_model = args.draft_model.expanduser().resolve()
    if args.mtp_mode in {"external", "both"}:
        if args.draft_model is None:
            parser.error(f"--draft-model is required for --mtp-mode {args.mtp_mode}")
        if not args.draft_model.is_file():
            parser.error(f"external MTP GGUF model not found: {args.draft_model}")

    for name in ("max_tokens", "runs", "draft_tokens", "n_ctx", "n_batch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be non-negative")
    if not 0.0 <= args.draft_p_min <= 1.0:
        parser.error("--draft-p-min must be between 0 and 1")


def selected_mtp_modes(args: argparse.Namespace) -> tuple[DecodeMode, ...]:
    if args.mtp_mode == "both":
        return ("builtin", "external")
    return (args.mtp_mode,)


def print_comparison(ordinary: BenchmarkResult, candidate: BenchmarkResult) -> None:
    speedup = (
        candidate.aggregate_tokens_per_second / ordinary.aggregate_tokens_per_second
        if ordinary.aggregate_tokens_per_second > 0.0
        else 0.0
    )
    outputs_match = all(
        run.tokens == ordinary.runs[0].tokens for run in ordinary.runs + candidate.runs
    )

    print(f"\n[comparison: {candidate.label} vs ordinary]")
    print(f"  output tokens match: {outputs_match}")
    print(f"  speedup            : {speedup:.3f}x")
    print(f"  throughput change  : {(speedup - 1.0) * 100.0:+.1f}%")
    if not outputs_match:
        baseline = ordinary.runs[0].tokens
        result = candidate.runs[0].tokens
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(baseline, result))
                if left != right
            ),
            min(len(baseline), len(result)),
        )
        print(f"  first divergence   : generated token {mismatch + 1}")


def main() -> None:
    args = parse_args()

    print(f"Model         : {args.model}")
    if args.draft_model is not None:
        print(f"Draft model   : {args.draft_model}")
    print(f"MTP mode      : {args.mtp_mode}")
    print(f"Prompt        : {args.prompt!r}")
    print(f"Measured runs : {args.runs} x {args.max_tokens} tokens per mode")
    print(f"MTP draft max : {args.draft_tokens}")
    print(
        "Sampling      : greedy"
        if args.temperature <= 0.0
        else "Sampling      : random"
    )
    print("Rollback      : native recurrent-state snapshots")

    ordinary = benchmark_mode(args, "ordinary")
    mtp_results = tuple(benchmark_mode(args, mode) for mode in selected_mtp_modes(args))
    print_mode(ordinary)
    for result in mtp_results:
        print_mode(result)
    for result in mtp_results:
        print_comparison(ordinary, result)

    print("\n--- ordinary output ---")
    print(ordinary.runs[0].text)
    for result in mtp_results:
        print(f"--- {result.label} output ---")
        print(result.runs[0].text)


if __name__ == "__main__":
    main()
