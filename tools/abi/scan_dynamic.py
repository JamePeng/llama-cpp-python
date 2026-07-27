"""Inspect and compare exported symbols in PE, ELF, and Mach-O libraries.

This repository-only maintainer utility supports collection and verification
of cross-platform ctypes symbol candidates, with particular focus on optional
llama_ext APIs.

LIEF is imported lazily so that ``--help`` remains available when the optional
dependency is not installed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

LIBRARY_SUFFIXES = {".dll", ".dylib", ".so"}
__author__ = "JamePeng"


class ScanError(RuntimeError):
    """Raised when a shared library cannot be inspected."""


@dataclass(frozen=True)
class SymbolRecord:
    """One exported symbol and its cross-platform names."""

    raw_name: str
    lookup_name: str
    canonical_name: str
    abi: str
    address: str
    ordinal: int | None = None


@dataclass(frozen=True)
class BindingDeclaration:
    """One ctypes decorator declaration extracted without importing llama_cpp."""

    python_name: str
    candidates: tuple[str, ...]
    required: bool
    line: int


@dataclass(frozen=True)
class LibraryScan:
    """Metadata and exported symbols for one binary architecture."""

    library: str
    format: str
    platform: str
    architecture: str
    sha256: str
    symbols: tuple[SymbolRecord, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_timestamp() -> str:
    """Return a sortable, collision-resistant UTC generation timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _enum_name(value: Any) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1]


def get_format(binary: Any) -> str:
    value = str(binary.format).upper()
    if "MACHO" in value:
        return "Mach-O"
    if "ELF" in value:
        return "ELF"
    if "PE" in value:
        return "PE"
    return str(binary.format)


def get_platform(binary_format: str) -> str:
    return {
        "PE": "windows",
        "ELF": "linux",
        "Mach-O": "darwin",
    }.get(binary_format, "unknown")


def get_architecture(binary: Any, binary_format: str) -> str:
    header = binary.header
    if binary_format == "PE":
        return _enum_name(header.machine)
    if binary_format == "ELF":
        return _enum_name(header.machine_type)
    if binary_format == "Mach-O":
        return _enum_name(header.cpu_type)
    return "unknown"


def normalize_symbol_name(raw_name: str, binary_format: str) -> str:
    """Return the name used by ctypes/dlsym and cross-platform comparison.

    Mach-O symbol tables prefix external C names with an underscore. dlsym and
    ctypes callers use the source-level name without that platform prefix.
    """

    if binary_format == "Mach-O" and raw_name.startswith("_"):
        return raw_name[1:]
    return raw_name


def detect_abi(normalized_name: str) -> str:
    if normalized_name.startswith("?"):
        return "msvc-cxxabi"
    if normalized_name.startswith("_Z"):
        return "itanium-cxxabi"
    return "unmangled"


def canonicalize_symbol_name(normalized_name: str, abi: str) -> str:
    """Recover a source-level name from simple global C++ mangling.

    llama_ext functions are global functions, so their MSVC and Itanium
    spellings can be mapped without a full ABI demangler. Namespaced,
    overloaded, and templated symbols remain mangled to avoid false matches.
    """

    if abi == "msvc-cxxabi":
        match = re.match(r"^\?([^@?$]+)@@", normalized_name)
        if match:
            return match.group(1)

    if abi == "itanium-cxxabi":
        match = re.match(r"^_Z(\d+)", normalized_name)
        if match:
            length = int(match.group(1))
            start = match.end()
            candidate = normalized_name[start : start + length]
            if len(candidate) == length:
                return candidate

    return normalized_name


def _symbol_address(symbol: Any) -> str:
    value = getattr(symbol, "address", None)
    if value is None:
        value = getattr(symbol, "value", 0)
    return hex(int(value))


def _exported_symbols(binary: Any, binary_format: str) -> Iterable[Any]:
    if binary_format == "PE":
        if not binary.has_exports:
            return ()
        return binary.get_export().entries

    # LIEF's exported_symbols filters undefined ELF imports and non-exported
    # Mach-O symbols, unlike dynamic_symbols/symbols.
    return binary.exported_symbols


def _iter_binaries(parsed: Any) -> list[Any]:
    # A universal Mach-O may contain several architecture slices.
    if type(parsed).__name__ == "FatBinary":
        return list(parsed)
    return [parsed]


def scan_library(
    path: str | Path,
) -> list[LibraryScan]:
    """Inspect one library, returning one result per architecture slice."""

    try:
        import lief
    except ImportError as exc:
        raise ScanError(
            "LIEF is required for ABI inspection. Install it with: pip install lief"
        ) from exc

    library_path = Path(path).expanduser().resolve()
    if not library_path.is_file():
        raise ScanError(f"Not a file: {library_path.name}")

    try:
        parsed = lief.parse(str(library_path))
    except Exception as exc:
        detail = str(exc).replace(str(library_path), library_path.name)
        raise ScanError(f"Failed to parse {library_path.name}: {detail}") from exc

    if parsed is None:
        raise ScanError(f"LIEF did not recognize {library_path.name}")

    digest = sha256_file(library_path)
    results: list[LibraryScan] = []

    for binary in _iter_binaries(parsed):
        binary_format = get_format(binary)
        records: list[SymbolRecord] = []

        for symbol in _exported_symbols(binary, binary_format):
            raw_name = getattr(symbol, "name", None)
            if not raw_name:
                # PE supports ordinal-only exports. They cannot be matched to
                # Python bindings by name, so keep a stable synthetic label.
                ordinal = getattr(symbol, "ordinal", None)
                if ordinal is None:
                    continue
                raw_name = f"#{ordinal}"

            lookup_name = normalize_symbol_name(raw_name, binary_format)
            abi = detect_abi(lookup_name)
            canonical_name = canonicalize_symbol_name(lookup_name, abi)

            records.append(
                SymbolRecord(
                    raw_name=raw_name,
                    lookup_name=lookup_name,
                    canonical_name=canonical_name,
                    abi=abi,
                    address=_symbol_address(symbol),
                    ordinal=getattr(symbol, "ordinal", None),
                )
            )

        records.sort(key=lambda item: (item.canonical_name, item.raw_name))
        results.append(
            LibraryScan(
                library=library_path.name,
                format=binary_format,
                platform=get_platform(binary_format),
                architecture=get_architecture(binary, binary_format),
                sha256=digest,
                symbols=tuple(records),
            )
        )

    return results


def select_scans_by_symbols(
    scans: Sequence[LibraryScan],
    required_symbols: Sequence[str],
) -> list[LibraryScan]:
    """Select binaries by exported canonical names, independent of filenames."""

    if not required_symbols:
        return list(scans)
    selected = []
    for scan in scans:
        exported = {symbol.canonical_name for symbol in scan.symbols}
        if all(name in exported for name in required_symbols):
            selected.append(scan)
    return selected


def filter_scan_symbols(
    scans: Sequence[LibraryScan],
    prefixes: Sequence[str],
) -> list[LibraryScan]:
    if not prefixes:
        return list(scans)
    return [
        replace(
            scan,
            symbols=tuple(
                symbol
                for symbol in scan.symbols
                if any(symbol.canonical_name.startswith(prefix) for prefix in prefixes)
            ),
        )
        for scan in scans
    ]


def extract_ctypes_bindings(source: str | Path) -> list[BindingDeclaration]:
    """Extract literal ctypes decorator candidates without importing the module."""

    source_path = Path(source)
    try:
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
    except (OSError, SyntaxError) as exc:
        raise ScanError(f"Failed to parse binding source {source_path}: {exc}") from exc

    declarations: list[BindingDeclaration] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            decorator_name = ""
            if isinstance(decorator.func, ast.Name):
                decorator_name = decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                decorator_name = decorator.func.attr
            if not decorator_name.startswith("ctypes_function"):
                continue

            try:
                names = ast.literal_eval(decorator.args[0])
            except (ValueError, TypeError):
                continue
            if isinstance(names, str):
                candidates = (names,)
            elif isinstance(names, (list, tuple)) and all(
                isinstance(name, str) for name in names
            ):
                candidates = tuple(names)
            else:
                continue

            required = True
            for keyword in decorator.keywords:
                if keyword.arg == "required":
                    try:
                        required = bool(ast.literal_eval(keyword.value))
                    except (ValueError, TypeError):
                        pass

            declarations.append(
                BindingDeclaration(
                    python_name=node.name,
                    candidates=candidates,
                    required=required,
                    line=node.lineno,
                )
            )

    return sorted(declarations, key=lambda item: item.line)


def check_bindings(
    scan: LibraryScan,
    declarations: Sequence[BindingDeclaration],
) -> dict[str, Any]:
    """Check which ctypes candidate would be selected for one library."""

    exported = {symbol.lookup_name for symbol in scan.symbols}
    available = []
    missing_required = []
    missing_optional = []

    for declaration in declarations:
        selected = next(
            (name for name in declaration.candidates if name in exported),
            None,
        )
        item = {
            "python_name": declaration.python_name,
            "required": declaration.required,
            "line": declaration.line,
            "candidates": list(declaration.candidates),
            "selected": selected,
        }
        if selected is not None:
            available.append(item)
        elif declaration.required:
            missing_required.append(item)
        else:
            missing_optional.append(item)

    return {
        "library": scan.library,
        "platform": scan.platform,
        "architecture": scan.architecture,
        "declaration_count": len(declarations),
        "available_count": len(available),
        "available": available,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def compare_scans(scans: Sequence[LibraryScan]) -> dict[str, Any]:
    if len(scans) < 2:
        raise ValueError("At least two library scans are required for comparison")

    symbol_sets = [{symbol.canonical_name for symbol in scan.symbols} for scan in scans]
    common = set.intersection(*symbol_sets)
    libraries = []

    for index, scan in enumerate(scans):
        others = set.union(*(symbol_sets[i] for i in range(len(scans)) if i != index))
        libraries.append(
            {
                "library": scan.library,
                "platform": scan.platform,
                "architecture": scan.architecture,
                "symbol_count": len(symbol_sets[index]),
                "only_here": sorted(symbol_sets[index] - others),
                "missing_here": sorted(others - symbol_sets[index]),
            }
        )

    return {
        "common_count": len(common),
        "common": sorted(common),
        "libraries": libraries,
    }


def build_manifest(
    scans: Sequence[LibraryScan],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or generation_timestamp()
    symbols: dict[str, list[dict[str, Any]]] = defaultdict(list)
    libraries = []

    for scan in scans:
        libraries.append(_scan_metadata(scan))
        for symbol in scan.symbols:
            symbols[symbol.canonical_name].append(
                {
                    "library": scan.library,
                    "platform": scan.platform,
                    "architecture": scan.architecture,
                    "raw_name": symbol.raw_name,
                    "lookup_name": symbol.lookup_name,
                    "abi": symbol.abi,
                    "address": symbol.address,
                    "ordinal": symbol.ordinal,
                }
            )

    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "libraries": libraries,
        "symbols": dict(sorted(symbols.items())),
    }


def collect_library_paths(
    inputs: Sequence[str],
    *,
    recursive: bool = False,
) -> list[Path]:
    def is_shared_library(path: Path) -> bool:
        name = path.name.lower()
        return path.suffix.lower() in LIBRARY_SUFFIXES or ".so." in name

    paths: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            paths.extend(
                candidate
                for candidate in candidates
                if candidate.is_file() and is_shared_library(candidate)
            )
        else:
            paths.append(path)
    return sorted(set(paths), key=lambda item: str(item).lower())


def _scan_paths(
    paths: Sequence[Path],
) -> tuple[list[LibraryScan], list[str]]:
    scans: list[LibraryScan] = []
    errors: list[str] = []
    for path in paths:
        try:
            scans.extend(scan_library(path))
        except ScanError as exc:
            errors.append(str(exc))
        except Exception as exc:
            detail = str(exc).replace(str(path.resolve()), path.name)
            errors.append(f"{path.name}: {detail}")
    return scans, errors


def _timestamped_output_path(output: str | Path, timestamp: str) -> Path:
    path = Path(output)
    return path.with_name(f"{path.stem}.{timestamp}{path.suffix}")


def _write_output(
    text: str,
    output: str | None,
    *,
    timestamp: str,
) -> None:
    if output:
        output_path = _timestamped_output_path(output, timestamp)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {output_path}")
    else:
        print(text)


def _scan_metadata(scan: LibraryScan) -> dict[str, Any]:
    return {key: value for key, value in asdict(scan).items() if key != "symbols"}


def _jsonl_rows(scan: LibraryScan, generated_at: str) -> list[str]:
    metadata = _scan_metadata(scan)
    return [
        json.dumps(
            {
                "generated_at": generated_at,
                **metadata,
                **asdict(symbol),
            },
            ensure_ascii=False,
        )
        for symbol in scan.symbols
    ]


def write_library_jsonl(
    scans: Sequence[LibraryScan],
    output_dir: str | Path = "tools/abi/output",
    *,
    timestamp: str | None = None,
) -> list[Path]:
    """Write one same-named JSONL per library under a timestamped run directory."""

    timestamp = timestamp or generation_timestamp()
    destination = Path(output_dir) / timestamp
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[LibraryScan]] = defaultdict(list)
    for scan in scans:
        grouped[scan.library].append(scan)

    written = []
    for library, library_scans in sorted(grouped.items()):
        output_path = destination / f"{library}.jsonl"
        rows = [
            row
            for library_scan in library_scans
            for row in _jsonl_rows(library_scan, timestamp)
        ]
        output_path.write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        written.append(output_path)
    return written


def _scan_text(scans: Sequence[LibraryScan], errors: Sequence[str]) -> str:
    lines: list[str] = []
    for scan in scans:
        lines.append(
            f"{scan.library} [{scan.format}/{scan.architecture}]: "
            f"{len(scan.symbols)} exported symbol(s)"
        )
        for symbol in scan.symbols:
            raw_suffix = (
                f" (raw: {symbol.raw_name})"
                if symbol.raw_name != symbol.canonical_name
                else ""
            )
            lines.append(
                f"  {symbol.canonical_name} [{symbol.abi}]"
                f" @ {symbol.address}{raw_suffix}"
            )
    for error in errors:
        lines.append(f"ERROR: {error}")
    return "\n".join(lines)


def _compare_text(comparison: dict[str, Any]) -> str:
    lines = [f"Common canonical symbols: {comparison['common_count']}"]
    for library in comparison["libraries"]:
        lines.extend(
            [
                "",
                (
                    f"{library['library']} "
                    f"[{library['platform']}/{library['architecture']}]: "
                    f"{library['symbol_count']} symbol(s)"
                ),
                f"  Only here: {len(library['only_here'])}",
            ]
        )
        lines.extend(f"    {name}" for name in library["only_here"])
        lines.append(f"  Missing here: {len(library['missing_here'])}")
        lines.extend(f"    {name}" for name in library["missing_here"])
    return "\n".join(lines)


def _bindings_text(
    results: Sequence[dict[str, Any]],
    scope: str,
) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(
            f"{result['library']} "
            f"[{result['platform']}/{result['architecture']}]: "
            f"{result['available_count']}/{result['declaration_count']} "
            "binding(s) available in selected scope"
        )
        if scope in {"required", "all"}:
            lines.append(f"  Missing required: {len(result['missing_required'])}")
            lines.extend(
                f"    {item['python_name']} (line {item['line']})"
                for item in result["missing_required"]
            )
        if scope in {"optional", "all"}:
            lines.append(f"  Missing optional: {len(result['missing_optional'])}")
            lines.extend(
                f"    {item['python_name']} (line {item['line']})"
                for item in result["missing_optional"]
            )
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and compare PE, ELF, and Mach-O exported symbols."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("paths", nargs="+", help="Library files or directories")
        subparser.add_argument(
            "--prefix",
            action="append",
            default=[],
            help=(
                "Optional canonical-name filter; may be repeated "
                "(default: keep all exports)"
            ),
        )
        subparser.add_argument(
            "--select-symbol",
            action="append",
            default=[],
            help=(
                "Select libraries exporting this canonical symbol; may be "
                "repeated and does not depend on the library filename"
            ),
        )
        subparser.add_argument(
            "--recursive",
            action="store_true",
            help="Recursively search directory inputs",
        )
        subparser.add_argument("-o", "--output", help="Write output to this file")

    scan_parser = subparsers.add_parser("scan", help="List exported symbols")
    add_common(scan_parser)
    scan_parser.add_argument(
        "--format",
        choices=("text", "json", "jsonl"),
        default="jsonl",
        help="Output format",
    )
    scan_parser.add_argument(
        "--output-dir",
        default="tools/abi/output",
        help=(
            "Directory for default per-library JSONL files "
            "(default: tools/abi/output)"
        ),
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare canonical symbol names across libraries"
    )
    add_common(compare_parser)
    compare_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )

    manifest_parser = subparsers.add_parser(
        "manifest", help="Create a cross-platform symbol manifest"
    )
    add_common(manifest_parser)

    bindings_parser = subparsers.add_parser(
        "check-bindings",
        help="Check literal ctypes decorator candidates against libraries",
    )
    add_common(bindings_parser)
    bindings_parser.add_argument(
        "--source",
        default="llama_cpp/llama_cpp.py",
        help="Python binding source to inspect without importing it",
    )
    bindings_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    bindings_parser.add_argument(
        "--scope",
        choices=("optional", "required", "all"),
        default="optional",
        help=("Binding declarations to check " "(default: optional llama_ext APIs)"),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    paths = collect_library_paths(args.paths, recursive=args.recursive)
    if not paths:
        print("No shared libraries found.", file=sys.stderr)
        return 2

    # Scan all exports first. Selection must not depend on --prefix, because a
    # caller may use an anchor outside the displayed prefix set.
    scans, errors = _scan_paths(paths)
    if not scans:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.select_symbol:
        scans = select_scans_by_symbols(scans, args.select_symbol)
        if not scans:
            print(
                "No library exports all requested selection symbols: "
                + ", ".join(args.select_symbol),
                file=sys.stderr,
            )
            return 2

    scans = filter_scan_symbols(scans, args.prefix)
    if args.command in {"compare", "manifest"} and args.prefix:
        # A package lib directory normally contains ggml and accelerator
        # backends. Empty prefix matches are not comparison targets.
        scans = [scan for scan in scans if scan.symbols]
    timestamp = generation_timestamp()

    validation_failed = False

    if args.command == "scan":
        if args.format == "text":
            output = _scan_text(scans, errors)
        elif args.format == "json":
            output = json.dumps(
                {
                    "generated_at": timestamp,
                    "libraries": [asdict(scan) for scan in scans],
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        else:
            output = "\n".join(
                row for scan in scans for row in _jsonl_rows(scan, timestamp)
            )

        if args.format == "jsonl" and args.output is None:
            try:
                written = write_library_jsonl(
                    scans,
                    args.output_dir,
                    timestamp=timestamp,
                )
            except OSError as exc:
                print(f"ERROR: failed to write JSONL output: {exc}", file=sys.stderr)
                return 1
            for path in written:
                print(f"saved: {path}")
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1 if errors else 0
    elif args.command == "compare":
        if len(scans) < 2:
            print("Comparison requires at least two libraries.", file=sys.stderr)
            return 2
        comparison = compare_scans(scans)
        output = (
            _compare_text(comparison)
            if args.format == "text"
            else json.dumps(comparison, ensure_ascii=False, indent=2)
        )
    elif args.command == "manifest":
        output = json.dumps(
            build_manifest(scans, generated_at=timestamp),
            ensure_ascii=False,
            indent=2,
        )
    else:
        try:
            declarations = extract_ctypes_bindings(args.source)
        except ScanError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.scope == "optional":
            declarations = [
                declaration for declaration in declarations if not declaration.required
            ]
        elif args.scope == "required":
            declarations = [
                declaration for declaration in declarations if declaration.required
            ]
        if not declarations:
            print(
                f"No {args.scope} ctypes binding declarations found in "
                f"{args.source}.",
                file=sys.stderr,
            )
            return 2
        binding_results = [check_bindings(scan, declarations) for scan in scans]
        if not args.select_symbol and binding_results:
            # A package directory may contain arbitrarily named dependency and
            # backend libraries. The library ctypes would want is the one with
            # the greatest declaration coverage, regardless of filename.
            best_count = max(result["available_count"] for result in binding_results)
            binding_results = [
                result
                for result in binding_results
                if result["available_count"] == best_count
            ]
        output = (
            _bindings_text(binding_results, args.scope)
            if args.format == "text"
            else json.dumps(binding_results, ensure_ascii=False, indent=2)
        )
        validation_failed = any(
            result["missing_required"] or result["missing_optional"]
            for result in binding_results
        )

    try:
        _write_output(output, args.output, timestamp=timestamp)
    except OSError as exc:
        print(f"ERROR: failed to write output: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors or validation_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
