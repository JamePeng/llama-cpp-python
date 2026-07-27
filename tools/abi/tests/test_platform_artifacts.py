"""Opt-in integration tests for real Windows, Linux, and macOS artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.abi.scan_dynamic import (
    check_bindings,
    collect_library_paths,
    extract_ctypes_bindings,
    scan_library,
    select_scans_by_symbols,
)

ARTIFACTS_ENV = "LLAMA_ABI_ARTIFACTS"
DEFAULT_ARTIFACTS = Path("tools/abi/artifacts")
REQUIRED_PLATFORMS = {"windows", "linux", "darwin"}
STABLE_LLAMA_SYMBOLS = {
    "llama_decode",
    "llama_model_load_from_file",
}
BINDING_SOURCE = Path("llama_cpp/llama_cpp.py")


@pytest.fixture(scope="module")
def platform_scans():
    configured = os.environ.get(ARTIFACTS_ENV)
    artifacts = Path(configured) if configured else DEFAULT_ARTIFACTS
    paths = collect_library_paths([str(artifacts)], recursive=True)

    if not paths and not configured:
        pytest.skip(
            "No ABI artifacts installed. Set LLAMA_ABI_ARTIFACTS to run "
            "the Windows/Linux/macOS integration test."
        )

    assert paths, f"No shared libraries found under {artifacts}"
    scans = [scan for path in paths for scan in scan_library(path)]
    scans = select_scans_by_symbols(scans, ["llama_decode"])
    by_platform = {scan.platform: scan for scan in scans}
    assert REQUIRED_PLATFORMS <= set(by_platform), (
        "The ABI artifact set must contain llama libraries for Windows, "
        f"Linux, and macOS. Found: {sorted(by_platform)}"
    )
    return by_platform


def test_windows_linux_and_macos_llama_exports(platform_scans):
    common = set.intersection(
        *(
            {symbol.canonical_name for symbol in platform_scans[platform].symbols}
            for platform in sorted(REQUIRED_PLATFORMS)
        )
    )
    assert STABLE_LLAMA_SYMBOLS <= common


def test_macos_macho_lookup_name_removes_symbol_table_prefix(platform_scans):
    decode = next(
        symbol
        for symbol in platform_scans["darwin"].symbols
        if symbol.canonical_name == "llama_decode"
    )
    assert decode.raw_name == "_llama_decode"
    assert decode.lookup_name == "llama_decode"


def test_optional_llama_ext_abi_aliases_on_all_platforms(platform_scans):
    optional = [
        declaration
        for declaration in extract_ctypes_bindings(BINDING_SOURCE)
        if not declaration.required
    ]
    assert optional, "No optional llama_ext ctypes bindings were found"

    for platform in sorted(REQUIRED_PLATFORMS):
        result = check_bindings(platform_scans[platform], optional)
        assert (
            result["missing_optional"] == []
        ), f"{platform} is missing optional llama_ext ABI aliases: " + ", ".join(
            item["python_name"] for item in result["missing_optional"]
        )
