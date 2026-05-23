import csv
import gc
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from llama_cpp import Llama
from llama_cpp.llama_speculative import (
    LlamaPromptLookupDecoding,
    LlamaNGramMapDecoding,
)


# ============================================================
# Model Configuration
# ============================================================

MODEL_PATH = r"/path/to/your/model.GGUF"

N_CTX = 4096
MAX_TOKENS = 1024
REPEATS = 2
CSV_OUTPUT = "speculative_benchmark_results.csv"

RANDOMIZE_ENGINE_ORDER = False


# ============================================================
# Benchmark Scenario Definition
# ============================================================

@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    prompt: str
    expected_behavior: str


TEST_SCENARIOS: List[Scenario] = [
    Scenario(
        name="A1. Medium-High Repetition - CRUD Boilerplate Code",
        category="code_boilerplate",
        expected_behavior="Should benefit from n-gram lookup because class and method structures repeat.",
        prompt="""<|im_start|>system
You are a senior backend developer. Write highly structured and consistent boilerplate code.<|im_end|>
<|im_start|>user
Write a Python script using `sqlite3` to define CRUD operations for a core banking system database.

Create 6 separate classes:
- Account
- Transaction
- Customer
- Loan
- Portfolio
- AuditLog

Each class MUST use the same internal method structure:
- create
- get
- update
- delete
- list_all

Do not add extra explanations. Output only code.<|im_end|>
<|im_start|>assistant
""",
    ),
    Scenario(
        name="A2. Extreme Repetition - JSONL Trading Logs",
        category="structured_logs",
        expected_behavior="Should strongly favor n-gram methods, especially K/K4V.",
        prompt="""<|im_start|>system
You are a deterministic data generation script. Output only raw JSON lines.<|im_end|>
<|im_start|>user
Continue this algorithmic trading execution log for 40 more lines.
Only change timestamp seconds, symbol, quantity, price, and execution_time_ms.

{"timestamp":"2026-05-23T09:30:01Z","level":"INFO","module":"exec_engine","event":"trade_filled","symbol":"AAPL","side":"BUY","quantity":100,"price":175.50,"execution_time_ms":12}
{"timestamp":"2026-05-23T09:30:02Z","level":"INFO","module":"exec_engine","event":"trade_filled","symbol":"MSFT","side":"SELL","quantity":50,"price":410.25,"execution_time_ms":15}
{"timestamp":"2026-05-23T09:30:03Z","level":"INFO","module":"exec_engine","event":"trade_filled","symbol":"TSLA","side":"BUY","quantity":200,"price":180.10,"execution_time_ms":11}<|im_end|>
<|im_start|>assistant
""",
    ),
    Scenario(
        name="A3. Markdown Table - Repetitive Course Catalog",
        category="markdown_table",
        expected_behavior="Repeated table columns and row structure should benefit from speculative lookup.",
        prompt="""<|im_start|>system
You generate clean Markdown tables with consistent formatting.<|im_end|>
<|im_start|>user
Create a Markdown comparison table for 30 university postgraduate courses.

Columns:
| Course ID | Course Title | Department | Credits | Prerequisites | Grading Basis | Core Objective |

The row format must stay consistent.
Use concise but realistic academic descriptions.
Do not add explanation outside the table.<|im_end|>
<|im_start|>assistant
| Course ID | Course Title | Department | Credits | Prerequisites | Grading Basis | Core Objective |
|---:|---|---|---:|---|---|---|
""",
    ),
    Scenario(
        name="A4. Structured Financial Market Report",
        category="structured_report",
        expected_behavior="Heading and bullet patterns repeat; n-gram lookup should help moderately.",
        prompt="""<|im_start|>system
You are a quantitative macroeconomic analyst. Output structured, clear, and professional financial reports.<|im_end|>
<|im_start|>user
Write a Q3 Macroeconomic & Equity Strategy Outlook Report for institutional investors.

Requirements:
1. Divide the report into exactly 8 sections.
2. Each section MUST contain exactly one heading and 3 bullet points.
3. Repeatedly emphasize the following themes across the sections: interest rate trajectory, inflation stickiness, equity market volatility, supply chain realignment, and fixed-income duration strategies.
4. Keep the tone highly professional and analytical.<|im_end|>
<|im_start|>assistant
""",
    ),
    Scenario(
        name="B1. Low Repetition - Macroeconomic Historical Essay",
        category="low_repetition_creative",
        expected_behavior="Should show limited or no speedup; useful as a negative control.",
        prompt="""<|im_start|>system
You are an academic historian of economics. Write with varied sentence structures, rich vocabulary, and analytical depth.<|im_end|>
<|im_start|>user
Write a comprehensive essay exploring the psychological and sociological impacts of hyperinflation on institutional trust during the Weimar Republic in the 1920s.

Requirements:
- Use highly academic and varied language.
- Do NOT use repetitive paragraph structures.
- Do NOT use bullet points or lists.
- Avoid parallel phrasing; favor complex, flowing narrative analysis.
- Make it a long, continuous essay.<|im_end|>
<|im_start|>assistant
The catastrophic devaluation of the Papiermark in the early 1920s fundamentally fractured the psychological bedrock of the Weimar Republic. """,
    ),
    Scenario(
        name="B2. Reasoning-Like Explanation - Quantitative Finance",
        category="reasoning_explanation",
        expected_behavior="May show smaller speedup because content is less template-like.",
        prompt="""<|im_start|>system
You are a careful technical explainer. Avoid repetitive phrasing.<|im_end|>
<|im_start|>user
Explain the foundational assumptions and inherent limitations of the Black-Scholes option pricing model.

Discuss the following concepts contextually:
- Log-normal distribution of asset prices
- The assumption of constant volatility and risk-free rates
- Frictionless markets (no transaction costs or taxes)
- The difference in applicability between European and American options

Write in clear, academic paragraphs. Do not use bullet points or lists.<|im_end|>
<|im_start|>assistant
""",
    ),
    Scenario(
        name="C1. Long Context Copy-Edit - High Local Reuse",
        category="copy_edit",
        expected_behavior="Prompt contains repeated phrases; n-gram lookup should exploit local reuse.",
        prompt="""<|im_start|>system
You are a precise academic editing assistant. Preserve the structure while improving the wording.<|im_end|>
<|im_start|>user
Rewrite the following academic grant proposal abstract in a cleaner professional style.
Keep the same repetitive sentence layout but fix the grammar and flow.

Draft Proposal:
The proposed research will investigate the efficiency of machine learning in high-frequency trading.
The proposed research will demonstrate the risk vectors of automated market making.
The methodology will utilize massive historical limit order book datasets.
The methodology will require significant computational cluster resources.
The expected outcomes will provide a new framework for liquidity provisioning.
The expected outcomes will establish a baseline for regulatory compliance monitoring.
The budget will allocate funds for data acquisition from major exchanges.
The budget will allocate funds for two postdoctoral researchers.
The timeline will span twenty-four months of continuous data analysis.
The timeline will include three major peer-reviewed journal submissions.
The significance will address the growing instability in algorithmic flash crashes.
The significance will ensure safer automated trading environments.<|im_end|>
<|im_start|>assistant
""",
    ),
]


# ============================================================
# Engine Definition
# ============================================================

@dataclass(frozen=True)
class EngineConfig:
    name: str
    draft_factory: Callable[[], Optional[object]]
    note: str


ENGINE_CONFIGS: List[EngineConfig] = [
    EngineConfig(
        name="Baseline",
        draft_factory=lambda: None,
        note="No speculative decoding.",
    ),
    EngineConfig(
        name="PromptLookup-Numpy-n10",
        draft_factory=lambda: LlamaPromptLookupDecoding(
            max_ngram_size=3,
            num_pred_tokens=10,
        ),
        note="Legacy sliding-window prompt lookup.",
    ),
    EngineConfig(
        name="NGramMap-K-n6",
        draft_factory=lambda: LlamaNGramMapDecoding(
            ngram_size=3,
            num_pred_tokens=6,
            mode="k",
            min_hits=1,
        ),
        note="Key-only n-gram map, shorter draft.",
    ),
    EngineConfig(
        name="NGramMap-K-n10",
        draft_factory=lambda: LlamaNGramMapDecoding(
            ngram_size=3,
            num_pred_tokens=10,
            mode="k",
            min_hits=1,
        ),
        note="Key-only n-gram map, default draft length.",
    ),
    EngineConfig(
        name="NGramMap-K4V-n10-cap8",
        draft_factory=lambda: LlamaNGramMapDecoding(
            ngram_size=3,
            num_pred_tokens=10,
            mode="k4v",
            min_hits=1,
            max_entries_per_key=8,
        ),
        note="K4V with bounded per-key memory.",
    ),
    EngineConfig(
        name="NGramMap-K4V-n16-cap8",
        draft_factory=lambda: LlamaNGramMapDecoding(
            ngram_size=3,
            num_pred_tokens=16,
            mode="k4v",
            min_hits=1,
            max_entries_per_key=8,
        ),
        note="Longer K4V draft; can be faster on highly repetitive outputs.",
    ),
    EngineConfig(
        name="NGramMap-K-minhits2-n10",
        draft_factory=lambda: LlamaNGramMapDecoding(
            ngram_size=3,
            num_pred_tokens=10,
            mode="k",
            min_hits=2,
        ),
        note="More conservative K mode.",
    ),
]


# ============================================================
# Measurement Helpers
# ============================================================

def cleanup_model(llm: Optional[Llama]) -> None:
    if llm is not None:
        del llm
    gc.collect()


def create_llama(draft_model: Optional[object]) -> Llama:
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        draft_model=draft_model,
        verbose=False,
    )


def measure_once(
    scenario: Scenario,
    engine: EngineConfig,
    repeat_idx: int,
) -> Dict[str, object]:
    draft_model = engine.draft_factory()

    print(f"\n⏳ [{scenario.name}] Engine={engine.name} | Repeat={repeat_idx + 1}")
    print(f"   Note: {engine.note}")

    llm: Optional[Llama] = None

    try:
        llm = create_llama(draft_model)

        # Warmup: force backend initialization and first-token path.
        llm.create_completion(
            prompt=scenario.prompt,
            max_tokens=1,
            temperature=0.0,
            echo=False,
        )

        start = time.perf_counter()

        response = llm.create_completion(
            prompt=scenario.prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.0,
            echo=False,
        )

        end = time.perf_counter()

        duration = end - start
        usage = response.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", 0))
        prompt_tokens = int(usage.get("prompt_tokens", 0))

        text = response["choices"][0]["text"]
        tps = completion_tokens / duration if duration > 0 else 0.0

        print(
            f"✅ {engine.name:<28} "
            f"{tps:8.2f} tok/s | "
            f"time={duration:7.2f}s | "
            f"gen={completion_tokens:4d} | "
            f"prompt={prompt_tokens:4d}"
        )
        print(f"   Snippet: {text[:120].replace(chr(10), ' ')}...")

        return {
            "scenario": scenario.name,
            "category": scenario.category,
            "expected_behavior": scenario.expected_behavior,
            "engine": engine.name,
            "engine_note": engine.note,
            "repeat": repeat_idx + 1,
            "duration_sec": duration,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sec": tps,
            "snippet": text[:160].replace("\n", "\\n"),
        }

    finally:
        if hasattr(draft_model, "close"):
            draft_model.close()
        cleanup_model(llm)


# ============================================================
# Reporting
# ============================================================

def summarize_results(rows: List[Dict[str, object]]) -> None:
    print("\n\n" + "=" * 90)
    print("📊 Benchmark Summary")
    print("=" * 90)

    by_scenario: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    for scenario_name, scenario_rows in by_scenario.items():
        print(f"\n📂 {scenario_name}")
        print("-" * 90)

        grouped: Dict[str, List[float]] = {}
        for row in scenario_rows:
            grouped.setdefault(str(row["engine"]), []).append(float(row["tokens_per_sec"]))

        baseline_avg = statistics.mean(grouped.get("Baseline", [0.0]))

        print(
            f"{'Engine':<32} | {'Avg tok/s':>10} | {'Best':>10} | "
            f"{'Worst':>10} | {'Speedup':>8}"
        )
        print("-" * 90)

        for engine_name, speeds in grouped.items():
            avg = statistics.mean(speeds)
            best = max(speeds)
            worst = min(speeds)
            speedup = avg / baseline_avg if baseline_avg > 0 else 1.0

            print(
                f"{engine_name:<32} | "
                f"{avg:10.2f} | "
                f"{best:10.2f} | "
                f"{worst:10.2f} | "
                f"{speedup:8.2f}x"
            )


def save_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n💾 CSV saved to: {path}")


# ============================================================
# Main Benchmark Flow
# ============================================================

def run_benchmark() -> None:
    print("=" * 90)
    print("🏆 llama-cpp-python Speculative Decoding Benchmark")
    print("=" * 90)
    print(f"Model: {MODEL_PATH}")
    print(f"n_ctx={N_CTX}, max_tokens={MAX_TOKENS}, repeats={REPEATS}")
    print("=" * 90)

    rows: List[Dict[str, object]] = []

    for scenario in TEST_SCENARIOS:
        print("\n\n" + "#" * 90)
        print(f"📂 Scenario: {scenario.name}")
        print(f"📌 Category: {scenario.category}")
        print(f"🧠 Expected: {scenario.expected_behavior}")
        print("#" * 90)

        engines = list(ENGINE_CONFIGS)
        if RANDOMIZE_ENGINE_ORDER:
            baseline = [e for e in engines if e.name == "Baseline"]
            others = [e for e in engines if e.name != "Baseline"]
            random.shuffle(others)
            engines = baseline + others

        for engine in engines:
            for repeat_idx in range(REPEATS):
                row = measure_once(
                    scenario=scenario,
                    engine=engine,
                    repeat_idx=repeat_idx,
                )
                rows.append(row)

    summarize_results(rows)
    save_csv(rows, CSV_OUTPUT)


if __name__ == "__main__":
    run_benchmark()