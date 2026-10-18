"""Measure cache=None live-prefix reuse and MTP rollback with a local model.

An optional --baseline-dir contains saved llama.py and llama_speculative.py.
Only the methods under comparison are installed temporarily; both versions use
the same loaded model, native libraries, settings and alternating request order.
"""

import argparse
import ast
import __future__
from contextlib import closing, contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from llama_cpp import Llama
import llama_cpp.llama as llama_module
import llama_cpp.llama_speculative as spec_module
from examples.benchmark.benchmark_speculative import SCENARIO_BY_KEY


def baseline_methods(folder):
    methods = []
    for filename, module, cls, names in (
        ("llama.py", llama_module, Llama, ("generate", "_limit_speculative_draft_n_max")),
        ("llama_speculative.py", spec_module, spec_module.LlamaMTPDecoding, ("rollback_verified",)),
    ):
        source = folder / filename
        tree = ast.parse(source.read_text(encoding="utf-8"))
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls.__name__)
        for name in names:
            node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)
            namespace = dict(vars(module))
            code = compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec", flags=__future__.annotations.compiler_flag)
            exec(code, namespace)
            methods.append((cls, name, namespace[name]))
    return methods


@contextmanager
def implementation(version, methods):
    with ExitStack() as stack:
        if version == "before":
            for cls, name, method in methods:
                stack.enter_context(patch.object(cls, name, method))
        yield


def prepare(llm, tokens, case):
    llm.reset()
    if case == "exact":
        llm.eval(tokens, copy_logits=False)
    elif case == "append":
        llm.eval(tokens[:-min(16, len(tokens) - 1)], copy_logits=False)
    elif case in ("branch", "restore"):
        llm.eval(tokens[:-1], copy_logits=False)
        if case == "restore":
            assert llm._hybrid_cache_mgr.save_checkpoint(llm.n_tokens, tokens[:-1])
        alternate = (tokens[-1] + 1) % llm.n_vocab()
        llm.eval([alternate], copy_logits=False)
    elif case == "shorter":
        llm.eval(tokens + llm.tokenize(b" Extra text that must be discarded.", add_bos=False), copy_logits=False)


def measure(llm, tokens, max_tokens):
    assert llm.cache is None
    native_eval = llm.eval
    manager = llm._hybrid_cache_mgr
    timestamps, output = [], []
    counters = {"evaluated_tokens": 0, "evaluated_before_first_token": 0,
                "checkpoint_captures": 0, "checkpoint_bytes": 0,
                "checkpoint_searches": 0, "checkpoint_restores": 0}

    def evaluate(inputs, **kwargs):
        counters["evaluated_tokens"] += len(inputs)
        if not timestamps:
            counters["evaluated_before_first_token"] += len(inputs)
        return native_eval(inputs, **kwargs)

    def capture(*args):
        written = native_capture(*args)
        counters["checkpoint_captures"] += 1
        counters["checkpoint_bytes"] += int(written)
        return written

    def search(*args, **kwargs):
        counters["checkpoint_searches"] += 1
        return native_search(*args, **kwargs)

    def restore(*args, **kwargs):
        result = native_restore(*args, **kwargs)
        counters["checkpoint_restores"] += int(result)
        return result

    if manager is not None:
        native_capture, native_search, native_restore = manager._get_data_ext, manager.find_best_checkpoint, manager.restore_checkpoint
    with ExitStack() as stack:
        stack.enter_context(patch.object(llm, "eval", evaluate))
        if manager is not None:
            stack.enter_context(patch.object(manager, "_get_data_ext", capture))
            stack.enter_context(patch.object(manager, "find_best_checkpoint", search))
            stack.enter_context(patch.object(manager, "restore_checkpoint", restore))
        started = time.perf_counter()
        with closing(llm.generate(tokens, temp=0, top_k=1, top_p=1, min_p=0, repeat_penalty=1, seed=42)) as stream:
            for token in stream:
                if token == llm.token_eos():
                    break
                timestamps.append(time.perf_counter())
                output.append(int(token))
                if len(output) >= max_tokens:
                    break
            delivered = time.perf_counter()
        llm._ctx.synchronize()
        finished = time.perf_counter()
    assert output
    if llm.ctx_checkpoints == 0:
        assert counters["checkpoint_captures"] == counters["checkpoint_bytes"] == 0
    decode_seconds = timestamps[-1] - timestamps[0]
    return output, {
        "tokens": len(output), "request_tps": len(output) / (finished - started),
        "decode_tps": (len(output) - 1) / decode_seconds if decode_seconds else 0,
        "ttft_ms": (timestamps[0] - started) * 1000,
        "cleanup_ms": (finished - delivered) * 1000,
        **counters, "speculative_stats": dict(llm.last_speculative_stats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIO_BY_KEY), default=["jsonl", "reasoning"])
    parser.add_argument("--methods", nargs="+", choices=["ordinary", "mtp"], default=["ordinary", "mtp"])
    parser.add_argument("--cases", nargs="+", choices=["cold", "exact", "append", "branch", "shorter", "restore"], default=["cold", "exact", "append", "branch", "shorter"])
    parser.add_argument("--ctx-checkpoints", type=int, default=0)
    parser.add_argument("--checkpoint-on-device", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.max_tokens < 2 or args.ctx_checkpoints < 0:
        parser.error("Require repeats >= 2, max-tokens >= 2, ctx-checkpoints >= 0")
    if "restore" in args.cases and args.ctx_checkpoints == 0:
        parser.error("The restore case needs ctx-checkpoints > 0")
    methods = baseline_methods(args.baseline_dir) if args.baseline_dir else []
    versions = ["before", "after"] if methods else ["after"]
    rows, reference = [], {}
    report = {"settings": vars(args).copy(), "cache": None,
              "source_sha256": {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (llama_module, spec_module)},
              "rows": rows}
    report["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["settings"].items()}
    for method in args.methods:
        config = spec_module.SpecConfig(spec_type=spec_module.SpeculativeType.DRAFT_MTP, draft_n_max=3) if method == "mtp" else None
        with closing(Llama(
            model_path=str(args.model), n_ctx=args.n_ctx, n_batch=args.n_batch,
            n_ubatch=args.n_batch, n_gpu_layers=args.n_gpu_layers,
            ctx_checkpoints=args.ctx_checkpoints, checkpoint_on_device=args.checkpoint_on_device,
            load_mtp=method == "mtp", speculative=config, verbose=False,
        )) as llm:
            assert llm.cache is None and llm.is_hybrid
            report[method + "_native_rollback_slots"] = llm._ctx.n_rs_seq()
            for scenario in args.scenarios:
                tokens = llm.tokenize(SCENARIO_BY_KEY[scenario].prompt.encode())
                llm.reset()
                measure(llm, tokens, 8)  # Warm the loaded graphs and sampler.
                llm.reset()
                expected, _ = measure(llm, tokens, args.max_tokens)
                if scenario in reference:
                    assert expected == reference[scenario], (method, scenario, "ordinary/MTP mismatch")
                else:
                    reference[scenario] = expected
                cases = [case for case in args.cases if method == "ordinary" or case in ("cold", "append")]
                for case in cases:
                    for repeat in range(args.repeats):
                        for version in (versions if repeat % 2 == 0 else versions[::-1]):
                            with implementation(version, methods):
                                prepare(llm, tokens, case)
                                output, row = measure(llm, tokens, args.max_tokens)
                            assert output == expected, (method, scenario, case, version, repeat)
                            row.update(method=method, scenario=scenario, case=case, version=version,
                                       repeat=repeat, matches_cold_tokens=True)
                            rows.append(row)
                    selected = [r for r in rows if (r["method"], r["scenario"], r["case"]) == (method, scenario, case)]
                    summary = {version: {key: statistics.median(r[key] for r in selected if r["version"] == version)
                                         for key in ("request_tps", "decode_tps", "ttft_ms", "cleanup_ms", "evaluated_before_first_token", "checkpoint_captures", "checkpoint_bytes", "checkpoint_restores")}
                               for version in versions}
                    print(json.dumps({"method": method, "scenario": scenario, "case": case, **summary}), flush=True)
                    args.json_output.parent.mkdir(parents=True, exist_ok=True)
                    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
