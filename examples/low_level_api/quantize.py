"""Quantize a GGUF model through the low-level API."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

from common import HelpFormatter, add_logging_arguments, existing_file, run_cli

QUANTIZATION_TYPES = {
    "Q4_0": "LLAMA_FTYPE_MOSTLY_Q4_0",
    "Q4_1": "LLAMA_FTYPE_MOSTLY_Q4_1",
    "Q5_0": "LLAMA_FTYPE_MOSTLY_Q5_0",
    "Q5_1": "LLAMA_FTYPE_MOSTLY_Q5_1",
    "Q8_0": "LLAMA_FTYPE_MOSTLY_Q8_0",
    "Q2_K": "LLAMA_FTYPE_MOSTLY_Q2_K",
    "Q3_K_M": "LLAMA_FTYPE_MOSTLY_Q3_K_M",
    "Q4_K_M": "LLAMA_FTYPE_MOSTLY_Q4_K_M",
    "Q5_K_M": "LLAMA_FTYPE_MOSTLY_Q5_K_M",
    "Q6_K": "LLAMA_FTYPE_MOSTLY_Q6_K",
    "IQ4_XS": "LLAMA_FTYPE_MOSTLY_IQ4_XS",
}


def output_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise argparse.ArgumentTypeError(f"output already exists: {path}")
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError(
            f"output directory does not exist: {path.parent}"
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a GGUF model without overwriting existing files.",
        formatter_class=HelpFormatter,
        epilog="""Example:
  python examples/low_level_api/quantize.py model-f16.gguf model-q4_k_m.gguf Q4_K_M
""",
    )
    parser.add_argument("input", type=existing_file)
    parser.add_argument("output", type=output_file)
    parser.add_argument("type", choices=QUANTIZATION_TYPES)
    parser.add_argument(
        "--threads", type=int, default=0, help="Zero uses the backend default."
    )
    parser.add_argument("--allow-requantize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import llama_cpp
    from llama_cpp._logger import configure_logging

    configure_logging(verbose=args.verbose, verbosity=args.verbosity)

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp.llama_ftype, QUANTIZATION_TYPES[args.type])
    params.nthread = args.threads
    params.allow_requantize = args.allow_requantize
    params.dry_run = args.dry_run

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Type:   {args.type}")
    # The native quantizer reads this ctypes structure through a pointer.
    result = llama_cpp.llama_model_quantize(
        str(args.input).encode("utf-8"),
        str(args.output).encode("utf-8"),
        ctypes.byref(params),
    )
    if result != 0:
        raise RuntimeError(f"quantization failed with code {result}")
    print("Dry run completed." if args.dry_run else "Quantization completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
