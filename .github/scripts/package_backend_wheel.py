"""Package backend wheels using the runtime libraries from one native build."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from email.parser import Parser
from zipfile import ZipFile


BACKENDS = {
    "cpu": "ggml-cpu",
    "blas": "ggml-blas",
    "cuda": "ggml-cuda",
    "hip": "ggml-hip",
    "metal": "ggml-metal",
    "musa": "ggml-musa",
    "vulkan": "ggml-vulkan",
    "sycl": "ggml-sycl",
    "cann": "ggml-cann",
    "opencl": "ggml-opencl",
    "openvino": "ggml-openvino",
    "webgpu": "ggml-webgpu",
    "hexagon": "ggml-hexagon",
    "zendnn": "ggml-zendnn",
    "zdnn": "ggml-zdnn",
    "rpc": "ggml-rpc",
    "virtgpu": "ggml-virtgpu",
    "et": "ggml-et",
}


def wheel_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_suffix(backend: str, version: str) -> str:
    if version and not re.fullmatch(r"\d+(?:\.\d+)*", version):
        raise RuntimeError("Backend version must be a dotted numeric version")
    if backend == "cuda":
        if len(version.split(".")) < 2:
            raise RuntimeError("CUDA requires a major.minor toolkit version")
        return "cu" + "".join(version.split(".")[:2])
    if backend in {"metal", "cpu"}:
        return ""
    return backend + version.replace(".", "")


def native_manifest(seed_dir: Path, backend: str, version: str) -> dict:
    version_suffix(backend, version)
    seeds = list(seed_dir.glob("*.whl"))
    if len(seeds) != 1:
        raise RuntimeError(f"Expected one native wheel in {seed_dir}, found {len(seeds)}")
    with ZipFile(seeds[0]) as wheel:
        names = wheel.namelist()
        if not any(n.startswith("llama_cpp/lib/") and BACKENDS[backend] in n.lower()
                   for n in names):
            raise RuntimeError(f"Native wheel has no {backend} backend library")
        wheel_info = next(n for n in names if n.endswith(".dist-info/WHEEL"))
        tags = Parser().parsestr(wheel.read(wheel_info).decode()).get_all("Tag", [])
    return {
        "backend": backend,
        "available_backends": sorted(
            key for key, library in BACKENDS.items()
            if any(n.startswith("llama_cpp/lib/") and library in n.lower() for n in names)
        ),
        "backend_version": version,
        "platform": sys.platform,
        "architecture": platform.machine().lower(),
        "wheel_platforms": sorted({tag.rsplit("-", 1)[1] for tag in tags}),
        "wheel_sha256": wheel_hash(seeds[0]),
    }


def verify_install() -> None:
    wheels = list(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one backend wheel to install")
    with ZipFile(wheels[0]) as wheel:
        name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        expected_version = Parser().parsestr(wheel.read(name).decode())["Version"]
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable,
         "--force-reinstall", str(wheels[0].resolve())],
        check=True,
    )
    code = """
import importlib.metadata
import platform
import sys
import llama_cpp

print("Python:", platform.python_version(), flush=True)
print("llama_cpp:", llama_cpp.__version__, flush=True)
print("Distribution:", importlib.metadata.version("llama_cpp_python"), flush=True)
print("Loaded from:", llama_cpp.__file__, flush=True)
assert llama_cpp.__version__ == sys.argv[1]
assert importlib.metadata.version("llama_cpp_python") == sys.argv[1]
print("Native system info:", llama_cpp.llama_print_system_info().decode(), flush=True)
"""
    # Isolated Python excludes the checkout and PYTHONPATH from module lookup.
    with tempfile.TemporaryDirectory(prefix="llama-wheel-check-") as directory:
        subprocess.run(
            [sys.executable, "-I", "-c", code, expected_version],
            cwd=directory,
            check=True,
        )


def package(seed_dir: Path, backend: str, version: str) -> None:
    recorded = json.loads((seed_dir / "native-build.json").read_text())
    actual = native_manifest(seed_dir, backend, version)
    if recorded != actual:
        differences = [
            f"{key}: native={recorded.get(key)!r}, requested={actual.get(key)!r}"
            for key in sorted(recorded.keys() | actual.keys())
            if recorded.get(key) != actual.get(key)
        ]
        raise RuntimeError("Native build manifest mismatch: " + "; ".join(differences))
    suffix = version_suffix(backend, version)
    seeds = list(seed_dir.glob("*.whl"))
    if len(seeds) != 1:
        raise RuntimeError(f"Expected one native wheel in {seed_dir}, found {len(seeds)}")

    staging = Path(".backend-native")
    if staging.exists() or list(Path("dist").glob("*.whl")):
        raise RuntimeError("Packaging requires a clean staging and output directory")

    libraries = {}
    with ZipFile(seeds[0]) as seed:
        metadata_names = [n for n in seed.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError("Expected one metadata record in the native wheel")
        metadata = Parser().parsestr(seed.read(metadata_names[0]).decode())
        base_version = metadata["Version"]
        for entry in seed.infolist():
            path = PurePosixPath(entry.filename)
            if entry.is_dir() or not entry.filename.startswith("llama_cpp/lib/"):
                continue
            if ".." in path.parts:
                raise RuntimeError(f"Invalid runtime path: {path}")
            data = seed.read(entry)
            target = staging.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            libraries[entry.filename] = hashlib.sha256(data).hexdigest()
    if not libraries:
        raise RuntimeError("The native wheel contains no runtime libraries")

    version_file = Path("llama_cpp/__init__.py")
    source = version_file.read_text()
    version_match = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)
    if not version_match or version_match[1] != base_version:
        raise RuntimeError("Source and native wheel versions do not match")
    if "+" in base_version:
        raise RuntimeError("Expected a base package version without a local suffix")

    project_file = Path("pyproject.toml")
    project = project_file.read_text()
    try:
        version_file.write_text(
            source[:version_match.start(1)] + base_version + ("+" + suffix if suffix else "")
            + source[version_match.end(1):]
        )
        project_file.write_text(
            project + '\n[tool.scikit-build.wheel.force-include]\n'
            + '".backend-native/llama_cpp/lib" = "llama_cpp/lib"\n'
        )
        # Explicit inclusion keeps ignored native files; platlib preserves platform tags.
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel",
             "-Cwheel.cmake=false", "-Cwheel.platlib=true"],
            check=True,
            env={**os.environ, "CMAKE_ARGS": ""},
        )
    finally:
        version_file.write_text(source)
        project_file.write_text(project)

    wheels = list(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one packaged backend wheel")
    with ZipFile(wheels[0]) as wheel:
        packaged = {
            name: hashlib.sha256(wheel.read(name)).hexdigest()
            for name in wheel.namelist()
            if name.startswith("llama_cpp/lib/") and not name.endswith("/")
        }
        if packaged != libraries:
            raise RuntimeError("Packaged runtime libraries differ from the native build")
        wheel_info = next(n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL"))
        tags = Parser().parsestr(wheel.read(wheel_info).decode()).get_all("Tag", [])
        if sorted({tag.rsplit("-", 1)[1] for tag in tags}) != recorded["wheel_platforms"]:
            raise RuntimeError("Packaged wheel platform differs from the native build")
    print(f"Packaged {wheels[0]} with {len(libraries)} unchanged runtime files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--seed-dir", type=Path, default=Path("native-wheel"))
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--backend-version", default="")
    args = parser.parse_args()
    if args.verify:
        verify_install()
    elif args.version:
        source = Path("llama_cpp/__init__.py").read_text()
        print("version=" + re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)[1])
    elif args.backend:
        if args.write_manifest:
            manifest = native_manifest(args.seed_dir, args.backend, args.backend_version)
            (args.seed_dir / "native-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        else:
            package(args.seed_dir, args.backend, args.backend_version)
    else:
        parser.error("--backend, --version, or --verify is required")
