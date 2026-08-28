"""Compare ordinary decoding with an external DFlash, DFlash2, or DSpark model.

Run with ``-h`` for portable command examples and tuning options.

The current DFlash-family implementation is text-only and supports one sequence.

Author: JamePeng
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from llama_cpp import Llama


DecodeMode = Literal["ordinary", "dflash", "dflash2", "dspark"]


@dataclass(frozen=True)
class GenerationResult:
    tokens: tuple[int, ...]
    text: str
    elapsed: float
    speculative_stats: dict[str, Any]

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.elapsed if self.elapsed > 0.0 else 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    load_seconds: float
    prompt_tokens: int
    runs: tuple[GenerationResult, ...]
    runtime_config: Optional[dict[str, Any]] = None

    @property
    def aggregate_tokens_per_second(self) -> float:
        elapsed = sum(run.elapsed for run in self.runs)
        tokens = sum(len(run.tokens) for run in self.runs)
        return tokens / elapsed if elapsed > 0.0 else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ordinary decoding against an external DFlash, DFlash2, or DSpark "
            "draft model. The ordinary baseline is always measured."
        ),
        epilog="""examples:
  DFlash:
    python -m examples.high_level_api.high_level_api_dflash_dspark_speculative --algorithm dflash --model target.gguf --draft-model dflash.gguf

  DFlash2 (requires selector metadata in the sidecar):
    python -m examples.high_level_api.high_level_api_dflash_dspark_speculative --algorithm dflash2 --model target.gguf --draft-model dflash2.gguf --draft-tokens 7 --draft-p-min 0

  DSpark with a shorter draft:
    python -m examples.high_level_api.high_level_api_dflash_dspark_speculative --algorithm dspark --model target.gguf --draft-model dspark.gguf --draft-tokens 3 --runs 3

  Fixed-length throughput comparison that ignores end-of-generation tokens:
    python -m examples.high_level_api.high_level_api_dflash_dspark_speculative --algorithm dflash --model target.gguf --draft-model dflash.gguf --max-tokens 512 --warmup-tokens 128 --ignore-eos
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        choices=("dflash", "dflash2", "dspark"),
        default="dflash",
        help=(
            "Draft algorithm described by the external GGUF. 'dflash' also "
            "auto-detects and labels DFlash2; 'dflash2' requires selector metadata "
            "(default: dflash)."
        ),
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to the target GGUF model."
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        required=True,
        help="Path to the compatible DFlash, DFlash2, or DSpark GGUF sidecar.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Write a detailed 1800-word technical article about speculative decoding. "
            "Use ten titled sections, include pseudocode, performance analysis, limitations, and a conclusion. Do not finish early."
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
        default=128,
        help="Untimed tokens generated once per mode to initialize graph shapes.",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=4096, help="Context size (default: 4096)."
    )
    parser.add_argument(
        "--n-batch", type=int, default=512, help="Logical batch size (default: 512)."
    )
    parser.add_argument(
        "--n-ubatch",
        type=int,
        default=512,
        help="Physical micro-batch size (default: 512).",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=7,
        help=(
            "Maximum draft tokens per verification step (default: 7). Benchmark "
            "this value for each model and deployment environment."
        ),
    )
    parser.add_argument(
        "--draft-n-min",
        type=int,
        default=0,
        help=(
            "Minimum number of proposals required to use a speculative block "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--draft-p-min",
        type=float,
        default=0.0,
        help=(
            "DFlash top-token, DFlash2 selector-path, or DSpark confidence "
            "threshold (default: 0)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Keep this at 0 for deterministic output comparison.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Generate exactly --max-tokens unless the context limit is reached.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Keep target-model layers on the CPU.",
    )
    parser.add_argument(
        "--draft-cpu",
        action="store_true",
        help="Keep draft-model layers on the CPU.",
    )
    parser.add_argument(
        "--no-backend-sampling",
        action="store_true",
        help=(
            "Select draft candidates on the CPU. This is usually much slower for "
            "large vocabularies."
        ),
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


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.model = args.model.expanduser().resolve()
    args.draft_model = args.draft_model.expanduser().resolve()
    if not args.model.is_file():
        parser.error(f"target GGUF model not found: {args.model}")
    if not args.draft_model.is_file():
        parser.error(f"draft GGUF model not found: {args.draft_model}")

    for name in (
        "max_tokens",
        "runs",
        "n_ctx",
        "n_batch",
        "n_ubatch",
        "draft_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be non-negative")
    if args.draft_n_min < 0:
        parser.error("--draft-n-min must be non-negative")
    if args.draft_n_min > args.draft_tokens:
        parser.error("--draft-n-min must not exceed --draft-tokens")
    if args.n_ubatch > args.n_batch:
        parser.error("--n-ubatch must not exceed --n-batch")
    if args.draft_tokens > args.n_batch - 1:
        parser.error("--draft-tokens must not exceed --n-batch - 1")
    if not 0.0 <= args.draft_p_min <= 1.0:
        parser.error("--draft-p-min must be between 0 and 1")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")


def generate_once(
    llm: Llama,
    prompt_tokens: list[int],
    *,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
) -> GenerationResult:
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
        ignore_eos=ignore_eos,
    )
    started = time.perf_counter()
    try:
        for token in stream:
            if not ignore_eos and llm._model.token_is_eog(token):
                break
            generated.append(int(token))
            if len(generated) >= max_tokens:
                break
    finally:
        stream.close()
    elapsed = time.perf_counter() - started
    text = llm.detokenize(generated, prev_tokens=prompt_tokens).decode(
        "utf-8", errors="replace"
    )
    return GenerationResult(
        tokens=tuple(generated),
        text=text,
        elapsed=elapsed,
        speculative_stats=dict(llm.last_speculative_stats),
    )


def benchmark_mode(args: argparse.Namespace, mode: DecodeMode) -> BenchmarkResult:
    # Keep native libraries unloaded for lightweight -h and validation paths.
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

    speculative: Optional[SpecConfig] = None
    if mode != "ordinary":
        spec_type = (
            SpeculativeType.DRAFT_DSPARK
            if mode == "dspark"
            else SpeculativeType.DRAFT_DFLASH
        )
        speculative = SpecConfig(
            spec_type=spec_type,
            draft_model_path=str(args.draft_model),
            draft_n_max=args.draft_tokens,
            draft_n_min=args.draft_n_min,
            draft_p_min=args.draft_p_min,
            draft_n_gpu_layers=0 if args.draft_cpu else "all",
            draft_backend_sampling=not args.no_backend_sampling,
        )

    load_started = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_gpu_layers=0 if args.cpu else "all",
        ctx_checkpoints=0,
        speculative=speculative,
        verbose=args.verbose,
    )
    load_seconds = time.perf_counter() - load_started
    runtime_config: Optional[dict[str, Any]] = None
    try:
        label = mode
        if mode != "ordinary":
            engine = llm.speculative
            is_dflash2 = bool(getattr(engine, "is_dflash2", False))
            if mode == "dflash2" and not is_dflash2:
                raise ValueError(
                    "--algorithm dflash2 requires a sidecar with a non-zero "
                    "DFlash2 selector_top_k"
                )
            if mode in {"dflash", "dflash2"} and is_dflash2:
                label = "DFlash2"
            elif mode == "dflash":
                label = "DFlash"
            elif mode == "dspark":
                label = "DSpark"
            runtime_config = {
                "selector_top_k": int(getattr(engine, "selector_top_k", 0)),
                "nextn_masked": not is_dflash2,
                "backend_requested": not args.no_backend_sampling,
                "backend_active": bool(getattr(engine, "_backend_sampling", False)),
                "mrope": bool(getattr(engine, "is_mrope", False)),
                "block_size": int(getattr(engine, "block_size", 0)),
                "draft_n_min": args.draft_n_min,
                "draft_n_max": int(getattr(engine, "draft_limit", args.draft_tokens)),
                "draft_p_min": args.draft_p_min,
            }
        prompt_tokens = llm.tokenize(
            args.prompt.encode("utf-8"), add_bos=True, special=True
        )
        if args.warmup_tokens:
            generate_once(
                llm,
                prompt_tokens,
                max_tokens=args.warmup_tokens,
                temperature=args.temperature,
                ignore_eos=args.ignore_eos,
            )
        results = tuple(
            generate_once(
                llm,
                prompt_tokens,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                ignore_eos=args.ignore_eos,
            )
            for _ in range(args.runs)
        )
    finally:
        llm.close()
    return BenchmarkResult(
        label, load_seconds, len(prompt_tokens), results, runtime_config
    )


def stat_total(result: BenchmarkResult, name: str) -> float:
    return sum(float(run.speculative_stats.get(name, 0)) for run in result.runs)


def print_mode(result: BenchmarkResult) -> None:
    rates = [run.tokens_per_second for run in result.runs]
    print(f"\n[{result.label}]")
    print(f"  model load          : {result.load_seconds:.3f} s (not benchmarked)")
    for index, run in enumerate(result.runs, 1):
        print(
            f"  run {index:<2}             : {len(run.tokens):>4} tokens / "
            f"{run.elapsed:.3f} s = {run.tokens_per_second:.2f} token/s"
        )
    print(f"  aggregate           : {result.aggregate_tokens_per_second:.2f} token/s")
    print(f"  median run          : {statistics.median(rates):.2f} token/s")

    config = result.runtime_config
    if config is not None:
        print(
            "  draft range         : "
            f"{config['draft_n_min']}..{config['draft_n_max']}, "
            f"p_min={config['draft_p_min']:g}"
        )
        print(f"  block size          : {config['block_size']}")
        print(f"  selector top-k      : {config['selector_top_k']}")
        print(
            "  nextn output        : "
            + ("masked" if config["nextn_masked"] else "unmasked")
        )
        print(
            "  backend sampling    : "
            f"requested={config['backend_requested']}, "
            f"active={config['backend_active']}"
        )
        print(f"  M-RoPE              : {config['mrope']}")

    verified = int(stat_total(result, "verified"))
    if not verified:
        return
    drafted = int(stat_total(result, "drafted"))
    accepted = int(stat_total(result, "accepted"))
    steps = int(stat_total(result, "verification_steps"))
    rollbacks = int(stat_total(result, "rollbacks"))
    native_rollbacks = int(stat_total(result, "native_rollbacks"))
    target_seconds = stat_total(result, "target_decode_seconds") + stat_total(
        result, "target_sync_seconds"
    )
    checkpoint_seconds = stat_total(result, "checkpoint_capture_seconds") + stat_total(
        result, "checkpoint_restore_seconds"
    )
    print(f"  draft proposed      : {drafted}")
    print(f"  draft verified      : {verified} in {steps} steps")
    print(f"  draft accepted      : {accepted} ({accepted / verified:.1%})")
    print(f"  draft rollbacks     : {rollbacks} (native: {native_rollbacks})")
    print(f"  target decode+sync  : {target_seconds:.3f} s")
    print(f"  draft phase         : {stat_total(result, 'draft_seconds'):.3f} s")
    print(f"  process phase       : {stat_total(result, 'process_seconds'):.3f} s")
    print(f"  checkpoint          : {checkpoint_seconds * 1000.0:.3f} ms")


def first_divergence(left: tuple[int, ...], right: tuple[int, ...]) -> Optional[int]:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def print_comparison(ordinary: BenchmarkResult, speculative: BenchmarkResult) -> None:
    speedup = (
        speculative.aggregate_tokens_per_second / ordinary.aggregate_tokens_per_second
        if ordinary.aggregate_tokens_per_second > 0.0
        else 0.0
    )
    reference = ordinary.runs[0].tokens
    divergence = first_divergence(reference, speculative.runs[0].tokens)
    deterministic = all(
        run.tokens == result.runs[0].tokens
        for result in (ordinary, speculative)
        for run in result.runs
    )
    print(f"\n[comparison: {speculative.label} vs ordinary]")
    print(f"  output tokens match : {divergence is None}")
    print(f"  runs deterministic  : {deterministic}")
    if divergence is not None:
        print(f"  first divergence    : generated token {divergence + 1}")
    print(f"  speedup             : {speedup:.3f}x")
    print(f"  throughput change   : {(speedup - 1.0) * 100.0:+.1f}%")


def main() -> None:
    args = parse_args()
    print(f"Target model   : {args.model}")
    print(f"Draft model    : {args.draft_model}")
    print(f"Algorithm      : {args.algorithm}")
    print(f"Prompt         : {args.prompt!r}")
    print(f"Measured runs  : {args.runs} x {args.max_tokens} tokens per mode")
    print(f"Warmup         : {args.warmup_tokens} tokens per mode")
    print(
        "Draft settings : "
        f"min={args.draft_n_min}, max={args.draft_tokens}, "
        f"p_min={args.draft_p_min:g}"
    )
    print(
        "Backend sample : "
        + ("disabled (CPU)" if args.no_backend_sampling else "enabled")
    )
    print(f"Ignore EOG     : {args.ignore_eos}")
    print("Limitations     : text-only, seq_id=0")

    ordinary = benchmark_mode(args, "ordinary")
    speculative = benchmark_mode(args, args.algorithm)
    print_mode(ordinary)
    print_mode(speculative)
    print_comparison(ordinary, speculative)

    print("\n--- ordinary output ---")
    print(ordinary.runs[0].text)
    print(f"--- {speculative.label} output ---")
    print(speculative.runs[0].text)


if __name__ == "__main__":
    main()
