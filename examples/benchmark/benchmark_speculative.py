"""Benchmark current stateful speculative-decoding implementations.

The ordinary decoder is always measured first as the deterministic baseline.
Run with ``-h`` for method descriptions and portable examples.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

if TYPE_CHECKING:
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import SpecConfig


MethodName = Literal["ordinary", "ngram-k", "ngram-k4v", "builtin-mtp", "external-mtp"]
SPECULATIVE_METHODS: tuple[MethodName, ...] = (
    "ngram-k",
    "ngram-k4v",
    "builtin-mtp",
    "external-mtp",
)
NGRAM_GRID_N = (6, 8, 10, 12)
NGRAM_GRID_M = (8, 16, 32, 48)


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    category: str
    expected_behavior: str
    prompt: str


@dataclass(frozen=True)
class BenchmarkCase:
    method: MethodName
    draft_tokens: int = 0
    ngram_size: Optional[int] = None

    @property
    def key(self) -> str:
        if self.ngram_size is not None:
            return f"{self.method}-n{self.ngram_size}-m{self.draft_tokens}"
        if self.method in {"builtin-mtp", "external-mtp"}:
            return f"{self.method}-m{self.draft_tokens}"
        return self.method


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="crud",
        name="CRUD boilerplate",
        category="code_boilerplate",
        expected_behavior="Repeated class and method structure should favor n-gram lookup.",
        prompt=(
            "Write only Python code using sqlite3. Define Account, Transaction, "
            "Customer, Loan, Portfolio, and AuditLog classes. Each class must use "
            "the same create, get, update, delete, and list_all method structure."
        ),
    ),
    Scenario(
        key="jsonl",
        name="Repetitive JSONL records",
        category="structured_logs",
        expected_behavior="Highly regular records should strongly favor n-gram lookup.",
        prompt=(
            "Continue this trading log for 40 lines. Return only JSONL and preserve "
            "the exact field order while changing values:\n"
            '{"timestamp":"2026-05-23T09:30:01Z","level":"INFO",'
            '"event":"trade_filled","symbol":"AAPL","side":"BUY",'
            '"quantity":100,"price":175.50,"execution_time_ms":12}\n'
            '{"timestamp":"2026-05-23T09:30:02Z","level":"INFO",'
            '"event":"trade_filled","symbol":"MSFT","side":"SELL",'
            '"quantity":50,"price":410.25,"execution_time_ms":15}'
        ),
    ),
    Scenario(
        key="table",
        name="Markdown course table",
        category="markdown_table",
        expected_behavior="Repeated columns and row syntax should favor speculation.",
        prompt=(
            "Create a 30-row Markdown postgraduate course table. Return only the "
            "table with these columns: Course ID, Course Title, Department, Credits, "
            "Prerequisites, Grading Basis, Core Objective. Keep every row concise."
        ),
    ),
    Scenario(
        key="report",
        name="Structured market report",
        category="structured_report",
        expected_behavior="Repeated headings and bullets should provide moderate reuse.",
        prompt=(
            "Write a professional Q3 macroeconomic and equity outlook with exactly "
            "eight sections. Each section must contain one heading and three bullet "
            "points covering rates, inflation, volatility, supply chains, and duration."
        ),
    ),
    Scenario(
        key="essay",
        name="Low-repetition historical essay",
        category="low_repetition_control",
        expected_behavior="Varied prose is a negative control for n-gram speculation.",
        prompt=(
            "Write a continuous academic essay on how Weimar hyperinflation changed "
            "institutional trust. Use varied syntax and vocabulary; do not use lists, "
            "parallel phrasing, or repetitive paragraph structures."
        ),
    ),
    Scenario(
        key="reasoning",
        name="Quantitative-finance explanation",
        category="reasoning_explanation",
        expected_behavior="Less template-like reasoning may have lower acceptance.",
        prompt=(
            "Explain the assumptions and limitations of Black-Scholes in connected "
            "academic paragraphs. Discuss log-normal prices, constant volatility and "
            "rates, frictionless markets, and European versus American options."
        ),
    ),
)
SCENARIO_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}


@dataclass(frozen=True)
class GenerationResult:
    tokens: tuple[int, ...]
    text: str
    request_seconds: float
    ttft_seconds: float
    sustained_seconds: float
    request_tokens_per_second: float
    sustained_tokens_per_second: float
    speculative_stats: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkRecord:
    scenario: str
    category: str
    expected_behavior: str
    method: str
    configuration: str
    ngram_size: Optional[int]
    draft_tokens: int
    repeat: int
    model_load_seconds: float
    prompt_tokens: int
    generated_tokens: int
    request_seconds: float
    sustained_seconds: float
    ttft_ms: float
    request_tokens_per_second: float
    sustained_tokens_per_second: float
    drafted_tokens: int
    accepted_tokens: int
    draft_acceptance_rate: float
    verification_steps: int
    rollbacks: int
    checkpoint_capture_ms: float
    checkpoint_restore_ms: float
    matches_baseline: Optional[bool]
    first_divergence: Optional[int]
    output_sha256: str
    output_preview: str


def build_parser() -> argparse.ArgumentParser:
    scenario_keys = ", ".join(SCENARIO_BY_KEY)
    parser = argparse.ArgumentParser(
        description=(
            "Compare stateful speculative methods with ordinary decoding. Models are "
            "loaded once per parameter configuration, TTFT is measured separately, "
            "and sustained speed starts after delivery of the first generated token."
        ),
        epilog="""examples:
  Compare both n-gram map modes:
    python -m examples.benchmark.benchmark_speculative --model model.gguf --methods ngram-k ngram-k4v

  Benchmark built-in MTP:
    python -m examples.benchmark.benchmark_speculative --model model-with-mtp.gguf --methods builtin-mtp --mtp-draft-tokens 2

  Compare built-in and external MTP:
    python -m examples.benchmark.benchmark_speculative --model model.gguf --methods builtin-mtp external-mtp --draft-model mtp-model.gguf

  Scan the recommended n-gram N x M grid:
    python -m examples.benchmark.benchmark_speculative --model model.gguf --methods ngram-k ngram-k4v --ngram-grid
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True, help="Target GGUF model.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SPECULATIVE_METHODS,
        default=["ngram-k", "ngram-k4v"],
        help=(
            "Speculative methods measured after the ordinary baseline "
            "(default: ngram-k ngram-k4v)."
        ),
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        help="External MTP GGUF; required when external-mtp is selected.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIO_BY_KEY),
        default=list(SCENARIO_BY_KEY),
        metavar="SCENARIO",
        help=f"Scenario keys to run (default: all). Available: {scenario_keys}.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Generated tokens per run."
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="Measured runs per scenario."
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="Untimed warmup tokens generated once per parameter configuration.",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size.")
    parser.add_argument("--n-batch", type=int, default=512, help="Logical batch size.")
    parser.add_argument(
        "--mtp-draft-tokens",
        type=int,
        default=2,
        help="Maximum proposed tokens for built-in and external MTP (default: 2).",
    )
    parser.add_argument(
        "--draft-p-min",
        type=float,
        default=0.0,
        help="Minimum probability retained by MTP drafting.",
    )
    parser.add_argument(
        "--ngram-sizes",
        type=int,
        nargs="+",
        default=[12],
        metavar="N",
        help="N-gram lookup key lengths to scan (default: 12).",
    )
    parser.add_argument(
        "--ngram-draft-tokens",
        type=int,
        nargs="+",
        default=[48],
        metavar="M",
        help="N-gram continuation lengths to scan (default: 48).",
    )
    parser.add_argument(
        "--ngram-grid",
        action="store_true",
        help="Scan N={6,8,10,12} x M={8,16,32,48} for each n-gram method.",
    )
    parser.add_argument(
        "--ngram-min-hits",
        type=int,
        default=1,
        help="Minimum matching histories required for an n-gram proposal.",
    )
    parser.add_argument(
        "--ngram-max-entries",
        type=int,
        default=4,
        help="Maximum K4V continuations per key (default: 4, matching llama.cpp).",
    )
    parser.add_argument(
        "--ctx-checkpoints",
        type=int,
        default=16,
        help=(
            "Target checkpoints available to n-gram rollback on hybrid/recurrent "
            "models (default: 16). MTP uses native recurrent snapshots instead."
        ),
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-on-device",
        dest="checkpoint_on_device",
        action="store_true",
        help="Keep hybrid target checkpoint tensor payloads on the device (default).",
    )
    checkpoint_group.add_argument(
        "--checkpoint-on-host",
        dest="checkpoint_on_device",
        action="store_false",
        help="Serialize hybrid target checkpoint tensor payloads through host memory.",
    )
    parser.set_defaults(checkpoint_on_device=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; zero enables deterministic comparison.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Sampling and shuffle seed."
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Run the target model on the CPU."
    )
    parser.add_argument(
        "--draft-cpu",
        action="store_true",
        help="Run an external MTP draft model on the CPU.",
    )
    parser.add_argument(
        "--no-backend-sampling",
        action="store_true",
        help="Select MTP candidates on the CPU instead of the backend sampler.",
    )
    parser.add_argument(
        "--shuffle-methods",
        action="store_true",
        help="Shuffle speculative parameter configurations after the baseline.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("speculative_benchmark_results.csv"),
        help="CSV result path (default: speculative_benchmark_results.csv).",
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Do not write CSV output."
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
    if not args.model.is_file():
        parser.error(f"target GGUF model not found: {args.model}")

    args.methods = list(dict.fromkeys(args.methods))
    args.scenarios = list(dict.fromkeys(args.scenarios))
    if args.ngram_grid:
        args.ngram_sizes = list(NGRAM_GRID_N)
        args.ngram_draft_tokens = list(NGRAM_GRID_M)
    else:
        args.ngram_sizes = list(dict.fromkeys(args.ngram_sizes))
        args.ngram_draft_tokens = list(dict.fromkeys(args.ngram_draft_tokens))
    if "external-mtp" in args.methods:
        if args.draft_model is None:
            parser.error("--draft-model is required for method external-mtp")
        args.draft_model = args.draft_model.expanduser().resolve()
        if not args.draft_model.is_file():
            parser.error(f"external MTP GGUF model not found: {args.draft_model}")

    positive = (
        "max_tokens",
        "repeats",
        "n_ctx",
        "n_batch",
        "mtp_draft_tokens",
        "ngram_min_hits",
        "ngram_max_entries",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    for option, values in (
        ("--ngram-sizes", args.ngram_sizes),
        ("--ngram-draft-tokens", args.ngram_draft_tokens),
    ):
        if any(value <= 0 for value in values):
            parser.error(f"all {option} values must be greater than zero")
    mtp_selected = {"builtin-mtp", "external-mtp"}.intersection(args.methods)
    ngram_selected = {"ngram-k", "ngram-k4v"}.intersection(args.methods)
    if mtp_selected and args.mtp_draft_tokens > args.n_batch - 1:
        parser.error("--mtp-draft-tokens must not exceed --n-batch - 1")
    if ngram_selected and max(args.ngram_draft_tokens) > args.n_batch - 1:
        parser.error("all --ngram-draft-tokens values must not exceed --n-batch - 1")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be non-negative")
    if not 0.0 <= args.draft_p_min <= 1.0:
        parser.error("--draft-p-min must be between 0 and 1")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    if args.ctx_checkpoints < 0:
        parser.error("--ctx-checkpoints must be non-negative")
    if ngram_selected and args.ctx_checkpoints == 0:
        parser.error("n-gram methods require --ctx-checkpoints > 0 on hybrid targets")

    if not args.no_csv:
        args.csv = args.csv.expanduser().resolve()
        args.csv.parent.mkdir(parents=True, exist_ok=True)


def method_label(case: BenchmarkCase) -> str:
    label = {
        "ordinary": "ordinary",
        "ngram-k": "n-gram map K",
        "ngram-k4v": "n-gram map K4V",
        "builtin-mtp": "built-in MTP",
        "external-mtp": "external MTP",
    }[case.method]
    if case.ngram_size is not None:
        return f"{label} (N={case.ngram_size}, M={case.draft_tokens})"
    if case.method in {"builtin-mtp", "external-mtp"}:
        return f"{label} (M={case.draft_tokens})"
    return label


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    cases = [BenchmarkCase("ordinary")]
    for method in args.methods:
        if method in {"ngram-k", "ngram-k4v"}:
            cases.extend(
                BenchmarkCase(method, draft_tokens=m, ngram_size=n)
                for n in args.ngram_sizes
                for m in args.ngram_draft_tokens
            )
        else:
            cases.append(BenchmarkCase(method, draft_tokens=args.mtp_draft_tokens))
    return cases


def create_spec_config(
    case: BenchmarkCase, args: argparse.Namespace
) -> Optional[SpecConfig]:
    from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

    if case.method == "ordinary":
        return None
    if case.method in {"ngram-k", "ngram-k4v"}:
        spec_type = (
            SpeculativeType.NGRAM_MAP_K
            if case.method == "ngram-k"
            else SpeculativeType.NGRAM_MAP_K4V
        )
        assert case.ngram_size is not None
        return SpecConfig(
            spec_type=spec_type,
            ngram_size_n=case.ngram_size,
            ngram_size_m=case.draft_tokens,
            ngram_min_hits=args.ngram_min_hits,
            ngram_max_entries_per_key=args.ngram_max_entries,
        )
    return SpecConfig(
        spec_type=SpeculativeType.DRAFT_MTP,
        draft_n_max=case.draft_tokens,
        draft_p_min=args.draft_p_min,
        draft_model_path=(
            str(args.draft_model) if case.method == "external-mtp" else None
        ),
        draft_n_gpu_layers=0 if args.draft_cpu else "all",
        draft_backend_sampling=not args.no_backend_sampling,
    )


def load_model(case: BenchmarkCase, args: argparse.Namespace) -> tuple[Llama, float]:
    from llama_cpp import Llama

    uses_target_checkpoints = case.method in {"ngram-k", "ngram-k4v"}
    started = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=0 if args.cpu else "all",
        load_mtp=case.method == "builtin-mtp",
        ctx_checkpoints=args.ctx_checkpoints if uses_target_checkpoints else 0,
        checkpoint_on_device=(
            args.checkpoint_on_device if uses_target_checkpoints else False
        ),
        speculative=create_spec_config(case, args),
        verbose=args.verbose,
    )
    return llm, time.perf_counter() - started


def generate_once(
    llm: Llama,
    prompt_tokens: Sequence[int],
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
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
        seed=seed,
        reset=True,
    )
    started = time.perf_counter()
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    try:
        for token in stream:
            delivered_at = time.perf_counter()
            if token == llm.token_eos():
                break
            if first_token_at is None:
                first_token_at = delivered_at
            last_token_at = delivered_at
            generated.append(int(token))
            if len(generated) >= max_tokens:
                break
    finally:
        stream.close()

    finished = last_token_at if last_token_at is not None else time.perf_counter()
    request_seconds = max(0.0, finished - started)
    ttft_seconds = (
        max(0.0, first_token_at - started) if first_token_at is not None else 0.0
    )
    sustained_seconds = (
        max(0.0, last_token_at - first_token_at)
        if first_token_at is not None and last_token_at is not None
        else 0.0
    )
    sustained_tokens = max(0, len(generated) - 1)
    text = llm.detokenize(generated, prev_tokens=list(prompt_tokens)).decode(
        "utf-8", errors="replace"
    )
    return GenerationResult(
        tokens=tuple(generated),
        text=text,
        request_seconds=request_seconds,
        ttft_seconds=ttft_seconds,
        sustained_seconds=sustained_seconds,
        request_tokens_per_second=(
            len(generated) / request_seconds if request_seconds > 0.0 else 0.0
        ),
        sustained_tokens_per_second=(
            sustained_tokens / sustained_seconds if sustained_seconds > 0.0 else 0.0
        ),
        speculative_stats=dict(llm.last_speculative_stats),
    )


def first_divergence(
    baseline: Sequence[int], candidate: Sequence[int]
) -> Optional[int]:
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        if left != right:
            return index + 1
    if len(baseline) != len(candidate):
        return min(len(baseline), len(candidate)) + 1
    return None


def output_hash(tokens: Sequence[int]) -> str:
    payload = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def make_record(
    scenario: Scenario,
    case: BenchmarkCase,
    repeat: int,
    load_seconds: float,
    prompt_tokens: Sequence[int],
    result: GenerationResult,
    baseline_tokens: Optional[Sequence[int]],
) -> BenchmarkRecord:
    stats = result.speculative_stats
    drafted = int(stats.get("drafted", 0))
    accepted = int(stats.get("accepted", 0))
    divergence = (
        first_divergence(baseline_tokens, result.tokens)
        if baseline_tokens is not None
        else None
    )
    return BenchmarkRecord(
        scenario=scenario.key,
        category=scenario.category,
        expected_behavior=scenario.expected_behavior,
        method=case.method,
        configuration=case.key,
        ngram_size=case.ngram_size,
        draft_tokens=case.draft_tokens,
        repeat=repeat + 1,
        model_load_seconds=load_seconds,
        prompt_tokens=len(prompt_tokens),
        generated_tokens=len(result.tokens),
        request_seconds=result.request_seconds,
        sustained_seconds=result.sustained_seconds,
        ttft_ms=result.ttft_seconds * 1000.0,
        request_tokens_per_second=result.request_tokens_per_second,
        sustained_tokens_per_second=result.sustained_tokens_per_second,
        drafted_tokens=drafted,
        accepted_tokens=accepted,
        draft_acceptance_rate=accepted / drafted if drafted > 0 else 0.0,
        verification_steps=int(stats.get("verification_steps", 0)),
        rollbacks=int(stats.get("rollbacks", 0)),
        checkpoint_capture_ms=(
            float(stats.get("checkpoint_capture_seconds", 0.0)) * 1000.0
        ),
        checkpoint_restore_ms=(
            float(stats.get("checkpoint_restore_seconds", 0.0)) * 1000.0
        ),
        matches_baseline=divergence is None if baseline_tokens is not None else None,
        first_divergence=divergence,
        output_sha256=output_hash(result.tokens),
        output_preview=result.text[:160].replace("\n", "\\n"),
    )


def run_case(
    case: BenchmarkCase,
    scenarios: Sequence[Scenario],
    args: argparse.Namespace,
    baseline_outputs: dict[tuple[str, int], tuple[int, ...]],
) -> list[BenchmarkRecord]:
    print(f"\n[{method_label(case)}] loading model")
    llm: Optional[Llama] = None
    records: list[BenchmarkRecord] = []
    try:
        llm, load_seconds = load_model(case, args)
        print(f"  model loaded in {load_seconds:.3f} s")
        if args.warmup_tokens > 0:
            warmup = llm.tokenize(b"Continue the sequence: 1, 2, 3,", add_bos=True)
            generate_once(
                llm,
                warmup,
                max_tokens=args.warmup_tokens,
                temperature=0.0,
                seed=args.seed,
            )

        for scenario in scenarios:
            prompt_tokens = llm.tokenize(
                scenario.prompt.encode("utf-8"), add_bos=True, special=True
            )
            for repeat in range(args.repeats):
                result = generate_once(
                    llm,
                    prompt_tokens,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                key = (scenario.key, repeat)
                baseline = baseline_outputs.get(key)
                if case.method == "ordinary":
                    baseline_outputs[key] = result.tokens
                record = make_record(
                    scenario,
                    case,
                    repeat,
                    load_seconds,
                    prompt_tokens,
                    result,
                    baseline,
                )
                records.append(record)
                acceptance = (
                    f" | accept {record.draft_acceptance_rate:.1%}"
                    if record.drafted_tokens > 0
                    else ""
                )
                match = (
                    " | output exact"
                    if record.matches_baseline is True
                    else (
                        f" | output diverged at token {record.first_divergence}"
                        if record.matches_baseline is False
                        else ""
                    )
                )
                print(
                    f"  {scenario.key:<10} run {repeat + 1:<2} | "
                    f"TTFT {record.ttft_ms:8.2f} ms | "
                    f"sustained {record.sustained_tokens_per_second:8.2f} tok/s"
                    f"{acceptance}{match}"
                )
    finally:
        if llm is not None:
            llm.close()
        del llm
        gc.collect()
    return records


def summarize(records: Sequence[BenchmarkRecord]) -> None:
    print("\n[summary: sustained generation speed]")
    for scenario in SCENARIOS:
        scenario_records = [row for row in records if row.scenario == scenario.key]
        if not scenario_records:
            continue
        grouped: dict[str, list[BenchmarkRecord]] = {}
        for row in scenario_records:
            grouped.setdefault(row.configuration, []).append(row)
        baseline = statistics.mean(
            row.sustained_tokens_per_second for row in grouped["ordinary"]
        )
        print(f"\n  {scenario.key}: {scenario.name}")
        print(
            f"    {'configuration':<24} {'mean tok/s':>12} {'min':>10} {'max':>10} "
            f"{'speedup':>10} {'output':>12}"
        )
        for configuration, method_records in grouped.items():
            speeds = [row.sustained_tokens_per_second for row in method_records]
            mean = statistics.mean(speeds)
            minimum = min(speeds)
            maximum = max(speeds)
            speedup = mean / baseline if baseline > 0.0 else 0.0
            divergences = [
                row.first_divergence
                for row in method_records
                if row.first_divergence is not None
            ]
            if configuration == "ordinary":
                output_text = "reference"
            elif not divergences:
                output_text = "exact"
            else:
                output_text = f"div@{min(divergences)}"
            print(
                f"    {configuration:<24} {mean:12.2f} {minimum:10.2f} {maximum:10.2f} "
                f"{speedup:9.3f}x {output_text:>12}"
            )
    print(
        "\n  Output is diagnostic only: div@N is the first token that differs "
        "from ordinary decoding. It does not invalidate the measured throughput."
    )


def save_csv(records: Sequence[BenchmarkRecord], path: Path) -> None:
    if not records:
        return
    rows = [asdict(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {path}")


def main() -> None:
    args = parse_args()
    scenarios = tuple(SCENARIO_BY_KEY[key] for key in args.scenarios)
    cases = build_cases(args)
    if args.shuffle_methods:
        speculative_cases = cases[1:]
        random.Random(args.seed).shuffle(speculative_cases)
        cases = [cases[0], *speculative_cases]

    print("llama-cpp-python speculative decoding benchmark")
    print(f"  target model : {args.model}")
    if args.draft_model is not None:
        print(f"  draft model  : {args.draft_model}")
    print(f"  methods      : ordinary, {', '.join(args.methods)}")
    print(f"  cases        : {len(cases)} total")
    if {"ngram-k", "ngram-k4v"}.intersection(args.methods):
        print(f"  n-gram grid  : N={args.ngram_sizes}, " f"M={args.ngram_draft_tokens}")
    if {"builtin-mtp", "external-mtp"}.intersection(args.methods):
        print(f"  MTP draft    : M={args.mtp_draft_tokens}")
    print(f"  scenarios    : {', '.join(args.scenarios)}")
    print(f"  workload     : {args.repeats} x {args.max_tokens} tokens per scenario")
    print("  timing       : TTFT includes prompt eval; sustained excludes first token")
    if {"ngram-k", "ngram-k4v"}.intersection(args.methods):
        checkpoint_location = "device" if args.checkpoint_on_device else "host"
        print(
            f"  n-gram undo  : {args.ctx_checkpoints} target checkpoints on "
            f"{checkpoint_location}"
        )

    baseline_outputs: dict[tuple[str, int], tuple[int, ...]] = {}
    records: list[BenchmarkRecord] = []
    for case in cases:
        records.extend(run_case(case, scenarios, args, baseline_outputs))

    summarize(records)
    if not args.no_csv:
        save_csv(records, args.csv)


if __name__ == "__main__":
    main()
