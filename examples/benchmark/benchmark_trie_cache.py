"""Compare Trie maintenance and real-model throughput against a saved source file.

Save llama_cpp/llama_cache.py before editing, then run this module with
--baseline-source PATH. Add --model PATH to measure completion cache reuse and
eviction with the same model, requests, sampling settings, and alternating order.
"""

import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import timeit
from types import SimpleNamespace

from llama_cpp import Llama
import llama_cpp.llama_cache as cache_module


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("llama_cpp._trie_benchmark_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LlamaTrieCache


def microbench(baseline, repeats):
    state = SimpleNamespace(nbytes=1, llama_state_size=1)
    cases = {
        "wide_insert": (512, [(1, i) + tuple(range(14)) for i in range(512)]),
        "wide_churn": (512, [(1, i) + tuple(range(14)) for i in range(1024)]),
        "long_shared_churn": (64, [tuple(range(512)) + (i + 1000,) + tuple(range(32)) for i in range(128)]),
        "replacement": (32, [tuple(range(512)) for _ in range(1024)]),
        "prefix_ladder": (512, [(1,) * length for length in range(1, 513)]),
    }
    report = []
    for name, (capacity, keys) in cases.items():
        def run(cache_type):
            cache = cache_type(capacity_bytes=capacity)
            for key in keys:
                cache[key] = state
            return cache

        row = {"case": name, "writes": len(keys)}
        for label, cache_type in (("baseline", baseline), ("optimized", cache_module.LlamaTrieCache)):
            samples = timeit.repeat(lambda: run(cache_type), number=1, repeat=repeats)
            row[label + "_ms"] = statistics.median(samples) * 1000
            cache = run(cache_type)
            pending, node_count = [cache.root], 0
            while pending:
                node = pending.pop()
                node_count += 1
                pending.extend(node.children.values())
            row[label + "_nodes"] = node_count
        row["speedup"] = row["baseline_ms"] / row["optimized_ms"]
        report.append(row)
        print(json.dumps(row), flush=True)
    return report


def timed_completion(llm, prompt, max_tokens):
    native_generate, native_load = llm.generate, llm.load_state
    delivered, timestamps, loads = [], [], []

    def generate(*args, **kwargs):
        with closing(native_generate(*args, **kwargs)) as stream:
            for token in stream:
                timestamps.append(time.perf_counter())
                delivered.append(int(token))
                yield token

    def load(state):
        loads.append(state.n_tokens)
        return native_load(state)

    llm.generate, llm.load_state = generate, load
    started = time.perf_counter()
    try:
        llm.create_completion(
            prompt, max_tokens=max_tokens, temperature=0, seed=42,
            repeat_penalty=1.0,
        )
        llm._ctx.synchronize()
    finally:
        llm.generate, llm.load_state = native_generate, native_load
    return {
        "tokens": delivered,
        "wall_seconds": time.perf_counter() - started,
        "ttft_ms": (timestamps[0] - started) * 1000,
        "decode_intervals": max(0, len(delivered) - 1),
        "decode_seconds": timestamps[-1] - timestamps[0],
        "loads": len(loads),
    }


def model_bench(args, baseline):
    with closing(Llama(
        model_path=str(args.model), n_ctx=1024, n_batch=128, n_ubatch=128,
        n_gpu_layers=args.n_gpu_layers, ctx_checkpoints=0, verbose=False,
    )) as llm:
        seeds = {}
        for label, offset in (("a", 0), ("b", 10), ("c", 20)):
            text = "Continue the JSONL sequence. Output only JSONL.\n"
            text += "".join(json.dumps({"i": i, "value": i * i}) + "\n" for i in range(1 + offset, 7 + offset))
            text += '{"i":%d,"value":' % (7 + offset)
            seeds[label] = llm.tokenize(text.encode())
        prompts = dict(seeds)
        prompts["append"] = seeds["a"] + llm.tokenize(b'49}\n{"i":8,"value":', add_bos=False)
        prompts["branch"] = seeds["a"][:-1] + llm.tokenize(b'"other":', add_bos=False)
        trace = ["a", "b", "a", "append", "branch", "c", "b", "a"]
        llm.reset()
        timed_completion(llm, seeds["a"], 8)  # Warm model/graphs before timing.
        references, snapshots = {}, {}
        for name, tokens in prompts.items():
            llm.reset()
            references[name] = timed_completion(llm, tokens, args.max_tokens)["tokens"]
        for name, tokens in seeds.items():
            llm.reset()
            llm.eval(tokens, copy_logits=False)
            snapshots[name] = llm.save_state()
        capacity = max(state.nbytes for state in snapshots.values()) * 5
        rounds = []
        variants = [("baseline", baseline), ("optimized", cache_module.LlamaTrieCache)]
        for repeat in range(args.repeats):
            for label, cache_type in (variants if repeat % 2 == 0 else variants[::-1]):
                writes, reads, evictions = [], [], []

                class MeasuredCache(cache_type):
                    def __setitem__(self, key, state):
                        before = len(self.lru_tracker)
                        new_entry = tuple(key) not in self.lru_tracker
                        started = time.perf_counter()
                        super().__setitem__(key, state)
                        writes.append(time.perf_counter() - started)
                        evictions.append(before + int(new_entry) - len(self.lru_tracker))

                    def get_matching_state(self, key, min_prefix_length=0):
                        started = time.perf_counter()
                        try:
                            return super().get_matching_state(key, min_prefix_length)
                        finally:
                            reads.append(time.perf_counter() - started)

                cache = MeasuredCache(capacity_bytes=capacity)
                for name, state in snapshots.items():
                    cache[seeds[name]] = state
                writes.clear()
                evictions.clear()
                llm.set_cache(cache)
                samples = []
                try:
                    for name in trace:
                        llm.reset()  # Exercise external snapshot reuse each time.
                        sample = timed_completion(llm, prompts[name], args.max_tokens)
                        assert sample.pop("tokens") == references[name], (label, repeat, name)
                        samples.append(sample)
                finally:
                    llm.set_cache(None)
                tokens = sum(len(references[name]) for name in trace)
                wall = sum(sample["wall_seconds"] for sample in samples)
                row = {
                    "variant": label, "repeat": repeat, "tokens": tokens,
                    "end_to_end_tps": tokens / wall,
                    "decode_tps": sum(s["decode_intervals"] for s in samples) / sum(s["decode_seconds"] for s in samples),
                    "median_ttft_ms": statistics.median(s["ttft_ms"] for s in samples),
                    "cache_write_ms": sum(writes) * 1000,
                    "cache_read_ms": sum(reads) * 1000,
                    "loads": sum(s["loads"] for s in samples), "evictions": sum(evictions),
                    "matches_cold_tokens": True, "samples": samples,
                }
                assert row["loads"] > 0 and row["evictions"] > 0
                rounds.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
        summary = {}
        for label, _ in variants:
            rows = [r for r in rounds if r["variant"] == label]
            summary[label] = {
                key: statistics.median(r[key] for r in rows)
                for key in ("end_to_end_tps", "decode_tps", "median_ttft_ms", "cache_write_ms", "cache_read_ms", "loads", "evictions")
            }
        return {"model": str(args.model), "ctx_checkpoints": 0, "capacity_bytes": capacity,
                "trace": trace, "rounds": rounds, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--micro-repeats", type=int, default=11)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2 or args.micro_repeats < 1 or args.max_tokens < 2:
        parser.error("Require repeats >= 2, micro-repeats >= 1, max-tokens >= 2")
    baseline = load_baseline(args.baseline_source)
    report = {
        "baseline_source": str(args.baseline_source),
        "baseline_sha256": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        "optimized_sha256": hashlib.sha256(Path(cache_module.__file__).read_bytes()).hexdigest(),
        "micro": microbench(baseline, args.micro_repeats),
    }
    if args.model:
        report["model_benchmark"] = model_bench(args, baseline)
        print(json.dumps(report["model_benchmark"]["summary"]), flush=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
