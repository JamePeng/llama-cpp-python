# Cross-platform ABI inspection

Author: **JamePeng**

This repository-only tool inspects PE (`.dll`), ELF (`.so` and `.so.*`), and
Mach-O (`.dylib`) exports. Its primary purpose is to collect ctypes symbol
candidates and verify optional `llama_ext` bindings across MSVC, GCC/Clang,
and macOS builds.

## Boundary and safety

The tool is intentionally excluded from wheels:

```toml
wheel.packages = ["llama_cpp"]
```

It is not imported by `llama_cpp`, has no installed command, and keeps LIEF
out of project dependencies. Run it only from a trusted source checkout.
LIEF parses native binaries, so do not scan untrusted artifacts.

Install the maintainer-only dependency:

```bash
python -m pip install lief
```

The tool and its documentation use the same MIT License as this repository.

## Artifact layout

Run commands from the repository root. Put builds under
`tools/abi/artifacts`, or replace that argument with an external absolute
directory:

```text
tools/abi/artifacts/
├── windows-x86_64/
│   └── <Windows DLLs>
├── linux-x86_64/
│   └── <Linux shared libraries>
└── macos-arm64/
    └── <macOS dynamic libraries>
```

Names are not significant. `--select-symbol llama_decode` identifies the
llama library by content when dependency and backend libraries share the same
directory.

Artifacts may come from local builds, an installed or extracted wheel,
[project releases](https://github.com/JamePeng/llama-cpp-python/releases), or
[upstream releases](https://github.com/ggml-org/llama.cpp/releases). Record
the source revision, compiler, architecture, and build options. Upstream
artifacts may not contain fork-only `llama_ext` APIs.

## Scan exports

```bash
python -m tools.abi scan tools/abi/artifacts --recursive
```

The default output is one same-named JSONL file per library:

```text
tools/abi/output/
└── 20260728T153012.123456Z/
    ├── llama.dll.jsonl
    ├── libllama.so.jsonl
    └── libllama.dylib.jsonl
```

Useful options:

```bash
# Select only binaries that export llama_decode.
python -m tools.abi scan tools/abi/artifacts --recursive \
    --select-symbol llama_decode

# Print instead of writing per-library JSONL.
python -m tools.abi scan tools/abi/artifacts --recursive --format text

# Write one aggregate file; its filename receives a UTC timestamp.
python -m tools.abi scan tools/abi/artifacts --recursive \
    --format jsonl --output all-symbols.jsonl
```

`--prefix` is optional. By default all exports are retained:

```bash
python -m tools.abi scan tools/abi/artifacts --recursive \
    --prefix llama_ --prefix ggml_
```

## Check optional llama_ext bindings

This is the primary ABI validation command:

```bash
python -m tools.abi check-bindings tools/abi/artifacts \
    --recursive \
    --source llama_cpp/llama_cpp.py
```

It statically reads ctypes decorators without importing `llama_cpp`. The
default `--scope optional` checks declarations marked `required=False` and
returns exit code 1 if any candidate is missing. Other scopes are available:

```bash
python -m tools.abi check-bindings tools/abi/artifacts \
    --recursive --scope required
python -m tools.abi check-bindings tools/abi/artifacts \
    --recursive --scope all
```

## Compare and create a manifest

```bash
python -m tools.abi compare tools/abi/artifacts \
    --recursive --select-symbol llama_decode

python -m tools.abi manifest tools/abi/artifacts \
    --recursive --select-symbol llama_decode \
    --output llama-exports.json
```

Cross-platform comparison uses `canonical_name`:

```text
?llama_graph_reserve@@...       MSVC
_Z19llama_graph_reserve...      Linux Itanium ABI
__Z19llama_graph_reserve...     Mach-O symbol table
                    ↓
llama_graph_reserve             canonical name
```

Records retain `raw_name`, ctypes `lookup_name`, `canonical_name`, ABI,
address, ordinal, library filename, format, architecture, SHA-256, and UTC
generation time. They never contain the artifact's absolute source path.

Every run receives a timestamp, preventing normal output from overwriting
previous results. Generated artifacts and reports are ignored by Git.

## Verification

Unit tests are independent from the project's default test suite:

```bash
python -m pytest tools/abi/tests/test_scan_dynamic.py -q
```

The opt-in integration test requires Windows, Linux, and macOS artifacts:

```powershell
$env:LLAMA_ABI_ARTIFACTS = "tools/abi/artifacts"
python -m pytest tools/abi/tests/test_platform_artifacts.py -q
```

```bash
LLAMA_ABI_ARTIFACTS=tools/abi/artifacts \
python -m pytest tools/abi/tests/test_platform_artifacts.py -q
```

Without configured artifacts, integration tests skip. With
`LLAMA_ABI_ARTIFACTS` set, a missing platform or optional ABI alias fails.
