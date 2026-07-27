import json

import tools.abi.scan_dynamic as abi_tool

from tools.abi.scan_dynamic import (
    BindingDeclaration,
    SymbolRecord,
    LibraryScan,
    canonicalize_symbol_name,
    check_bindings,
    collect_library_paths,
    compare_scans,
    detect_abi,
    extract_ctypes_bindings,
    normalize_symbol_name,
    select_scans_by_symbols,
    write_library_jsonl,
)


def _scan(library: str, platform: str, names: list[str]) -> LibraryScan:
    records = tuple(
        SymbolRecord(
            raw_name=name,
            lookup_name=name,
            canonical_name=name,
            abi="unmangled",
            address="0x0",
        )
        for name in names
    )
    return LibraryScan(
        library=library,
        format="test",
        platform=platform,
        architecture="test",
        sha256="test",
        symbols=records,
    )


def test_normalizes_macho_external_prefix():
    assert normalize_symbol_name("_llama_decode", "Mach-O") == "llama_decode"
    assert normalize_symbol_name("__ZN5llama", "Mach-O") == "_ZN5llama"
    assert normalize_symbol_name("llama_decode", "ELF") == "llama_decode"
    assert normalize_symbol_name("llama_decode", "PE") == "llama_decode"


def test_detects_abi_after_platform_normalization():
    assert detect_abi("?function@@YAXXZ") == "msvc-cxxabi"
    assert detect_abi("_ZN5llama") == "itanium-cxxabi"
    assert detect_abi("llama_decode") == "unmangled"


def test_canonicalizes_simple_global_cpp_names():
    assert (
        canonicalize_symbol_name(
            "?llama_graph_reserve@@YAXXZ",
            "msvc-cxxabi",
        )
        == "llama_graph_reserve"
    )
    assert (
        canonicalize_symbol_name(
            "_Z19llama_graph_reserveP13llama_contextjjj",
            "itanium-cxxabi",
        )
        == "llama_graph_reserve"
    )
    nested = "_ZN5llama6detail3fooEv"
    assert canonicalize_symbol_name(nested, "itanium-cxxabi") == nested


def test_compares_canonical_names():
    comparison = compare_scans(
        [
            _scan("libllama.so", "linux", ["llama_decode"]),
            _scan(
                "llama.dll",
                "windows",
                ["llama_decode", "llama_windows_only"],
            ),
        ]
    )

    assert comparison["common"] == ["llama_decode"]
    assert comparison["libraries"][0]["missing_here"] == ["llama_windows_only"]
    assert comparison["libraries"][1]["only_here"] == ["llama_windows_only"]


def test_collects_versioned_elf_library(tmp_path):
    library = tmp_path / "libllama.so.1"
    library.touch()

    assert collect_library_paths([str(tmp_path)]) == [library]


def test_extracts_and_checks_literal_binding_aliases(tmp_path):
    source = tmp_path / "bindings.py"
    source.write_text(
        """
@ctypes_function(
    ["llama_ext", "?llama_ext@@YAXXZ", "_Z9llama_extv"],
    [],
    None,
    required=False,
)
def llama_ext():
    pass
""",
        encoding="utf-8",
    )
    declarations = extract_ctypes_bindings(source)
    scan = _scan("libllama.so", "linux", ["_Z9llama_extv"])
    result = check_bindings(scan, declarations)

    assert declarations == [
        BindingDeclaration(
            python_name="llama_ext",
            candidates=(
                "llama_ext",
                "?llama_ext@@YAXXZ",
                "_Z9llama_extv",
            ),
            required=False,
            line=8,
        )
    ]
    assert result["available"][0]["selected"] == "_Z9llama_extv"
    assert result["missing_optional"] == []


def test_selects_library_by_symbol_not_filename():
    scans = [
        _scan("custom-backend-name.dll", "windows", ["ggml_backend_init"]),
        _scan("renamed-native-output.bin", "windows", ["llama_decode"]),
    ]

    selected = select_scans_by_symbols(scans, ["llama_decode"])

    assert [scan.library for scan in selected] == ["renamed-native-output.bin"]


def test_writes_jsonl_named_after_dynamic_library(tmp_path):
    output_dir = tmp_path / "output"
    scans = [
        _scan("libllama.so", "linux", ["llama_decode"]),
        _scan("llama.dll", "windows", ["llama_decode"]),
    ]

    timestamp = "20260728T120000.123456Z"
    written = write_library_jsonl(
        scans,
        output_dir,
        timestamp=timestamp,
    )

    assert [path.name for path in written] == [
        "libllama.so.jsonl",
        "llama.dll.jsonl",
    ]
    run_dir = output_dir / timestamp
    row = json.loads((run_dir / "llama.dll.jsonl").read_text("utf-8"))
    assert row["library"] == "llama.dll"
    assert row["canonical_name"] == "llama_decode"
    assert row["generated_at"] == timestamp
    assert "path" not in row


def test_check_bindings_cli_fails_when_optional_api_is_missing(
    tmp_path,
    monkeypatch,
):
    library = tmp_path / "renamed.dll"
    library.touch()
    source = tmp_path / "bindings.py"
    source.write_text(
        """
@ctypes_function(["llama_ext", "_Z9llama_extv"], [], None, required=False)
def llama_ext():
    pass
""",
        encoding="utf-8",
    )
    scan = _scan("renamed.dll", "windows", ["llama_decode"])
    monkeypatch.setattr(
        abi_tool,
        "_scan_paths",
        lambda paths: ([scan], []),
    )

    exit_code = abi_tool.main(["check-bindings", str(library), "--source", str(source)])

    assert exit_code == 1
