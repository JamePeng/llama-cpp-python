---
title: Installation
page_type: guide
source_files:
  - README.md
  - vendor/llama.cpp/docs/build.md
  - vendor/llama.cpp/docs/backend/
last_updated: 2026-06-02
author: JamePeng
version_target: "latest"
---

# Installation

## Overview

This page explains how to install `llama-cpp-python` from source, with or
without hardware acceleration.

`llama-cpp-python` builds the native `llama.cpp` libraries during installation
and installs them inside the Python package. The exact build depends on your
Python version, compiler, CMake version, operating system, and selected
`llama.cpp` backend.

For most users, the safest installation path is:

1. Create a clean Python virtual environment.
2. Upgrade `pip`.
3. Install from the GitHub repository.
4. Pass `CMAKE_ARGS` only when you need a specific backend.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python | Python 3.9 or newer. The package metadata currently lists Python 3.9 through 3.14. |
| CMake | CMake 3.21 or newer. |
| C/C++ compiler | Required because the package builds `llama.cpp` native libraries. |
| Git | Required when installing from the GitHub repository or cloning recursively. |
| Backend SDKs | Required only for GPU or accelerator builds, such as CUDA, Vulkan, OpenVINO, ROCm/HIP, or SYCL. |

Platform compiler guidance:

| Platform | Typical compiler setup |
|---|---|
| Linux | `gcc` or `clang` plus Python development headers if required by your distribution. |
| Windows | Visual Studio 2022 Build Tools or MinGW. For most native builds, Visual Studio Build Tools is recommended. |
| macOS | Xcode Command Line Tools. Metal is enabled by default on supported macOS builds. |

---

## Use a Virtual Environment

Using a virtual environment avoids mixing build artifacts and dependencies from
different Python installations.

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

---

## Basic Installation

Install directly from the project repository:

```bash
python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

On Windows PowerShell:

```powershell
python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

This builds `llama.cpp` from source and installs the generated native runtime
libraries alongside the Python package.

Use verbose output when diagnosing build failures:

```bash
python -m pip install --verbose "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

---

## Install From a Local Clone

Clone recursively so the `vendor/llama.cpp` submodule is available:

```bash
git clone https://github.com/JamePeng/llama-cpp-python --recursive
cd llama-cpp-python
python -m pip install --upgrade pip
python -m pip install .
```

If you already cloned without `--recursive`, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

For editable development installs:

```bash
python -m pip install -e .
```

---

## Passing CMake Options

`llama.cpp` backend options are passed through CMake. There are two common
ways to pass those options during `pip install`.

### Environment Variable

Linux and macOS:

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Windows PowerShell:

```powershell
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Clear the variable after the build if you do not want it reused:

```powershell
Remove-Item Env:CMAKE_ARGS
```

### `pip --config-settings`

You can also pass CMake arguments through `pip`:

```bash
python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git" \
  -C cmake.args="-DGGML_BLAS=ON;-DGGML_BLAS_VENDOR=OpenBLAS"
```

Use semicolons inside `cmake.args` when passing multiple CMake definitions.

---

## Common CMake Options

The Python package forwards CMake options to the bundled `vendor/llama.cpp`
build. These options are useful across many backends.

| Option | Typical values | Use |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release`, `Debug` | Selects build type for single-config generators such as Ninja or Unix Makefiles. Release is the normal install choice. |
| `GGML_NATIVE` | `ON`, `OFF` | Controls whether ggml builds for the current host CPU/GPU. Use `OFF` for more portable wheels; use `ON` for local machine-specific optimization. |
| `BUILD_SHARED_LIBS` | `ON`, `OFF` | Controls shared versus static native libraries. The Python package normally installs shared runtime libraries. |
| `GGML_BACKEND_DL` | `ON`, `OFF` | Builds backend libraries so they can be loaded dynamically at runtime when supported by the build. |
| `GGML_CPU_ALL_VARIANTS` | `ON`, `OFF` | Builds multiple CPU backend variants for x86 feature sets when supported. Useful for portable x64 wheels. |
| `GGML_OPENMP` | `ON`, `OFF` | Enables OpenMP CPU parallelism. On Windows, OpenMP runtime DLLs may need to be packaged beside backend DLLs. |
| `CMAKE_PREFIX_PATH` | path list | Helps CMake find SDKs or libraries installed outside default locations. |
| `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` | compiler paths or names | Selects compilers, often needed for SYCL, HIP, or custom toolchains. |

Example portable CUDA build:

```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=OFF" \
  python -m pip install --force-reinstall --no-cache-dir \
  "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Example dynamic CPU backend build:

```bash
CMAKE_ARGS="-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF" \
  python -m pip install --force-reinstall --no-cache-dir \
  "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

---

## Backend Quick Reference

Choose one backend path that matches your hardware and installed SDKs.

| Backend | Typical CMake option | Notes |
|---|---|---|
| CPU only | none | Default portable path. Performance depends on CPU features and build options. |
| OpenBLAS | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` | CPU BLAS acceleration for prompt processing and larger batches. |
| BLIS | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME` | CPU BLAS route using BLIS. |
| Intel oneMKL | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp` | Intel CPU BLAS route. This is not the Intel GPU path. |
| CUDA | `-DGGML_CUDA=on` | Requires NVIDIA CUDA Toolkit matching your driver and GPU. |
| Metal | `-DGGML_METAL=on` | Enabled by default on supported macOS builds. Use `-DGGML_METAL=OFF` to disable. |
| Vulkan | `-DGGML_VULKAN=on` | Requires Vulkan SDK and platform-specific setup. |
| OpenVINO | `-DGGML_OPENVINO=ON` | Useful for Intel CPU, GPU, and NPU workflows after OpenVINO environment setup. |
| HIP / ROCm | `-DGGML_HIP=ON` | For supported AMD GPUs. May require `GPU_TARGETS`. |
| SYCL | `-DGGML_SYCL=on` | Usually used with Intel oneAPI compilers. |
| OpenCL | `-DGGML_OPENCL=ON` | Primarily documented for Qualcomm Adreno and Snapdragon workflows; can also apply to some other OpenCL devices. |
| CANN | `-DGGML_CANN=ON` | Ascend NPU backend. Requires Ascend drivers and CANN toolkit. |
| ZenDNN | `-DGGML_ZENDNN=ON` | AMD Zen CPU acceleration, mainly matrix multiplication paths. |
| zDNN | `-DGGML_ZDNN=ON -DZDNN_ROOT=/path/to/zdnn` | IBM Z / LinuxONE acceleration path. |

For the full list of backend options, check the upstream llama.cpp build
documentation and the current `vendor/llama.cpp` source.

---

## CUDA

CUDA builds require the NVIDIA CUDA Toolkit. Choose a toolkit version that is
compatible with your driver and GPU.

Linux:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Windows PowerShell:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

For newer NVIDIA GPUs with compute capability 90 or higher, the README notes
that Programmatic Dependent Launch can be enabled with:

```bash
-DGGML_CUDA_PDL=ON
```

Example:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_PDL=ON" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

If `nvcc` produces large volumes of non-blocking template warnings, the README
documents optional CUDA warning suppression:

```bash
-DCMAKE_CUDA_FLAGS="--diag-suppress=177 --diag-suppress=221 --diag-suppress=550"
```

### CUDA Portability and Architecture Selection

By default, llama.cpp may build for the GPU detected on the build machine. For
a wheel intended to run across multiple CUDA GPUs, disable native detection:

```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=OFF" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

If `nvcc` cannot detect your GPU, or if you want to control the generated
binary size, specify CUDA architectures explicitly:

```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Use NVIDIA's compute capability table to choose architecture numbers. For
example, RTX 30-series GPUs commonly use `86`, and RTX 4090 uses `89`.

If multiple CUDA toolkits are installed, point CMake at the intended compiler:

```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda-12.8/bin/nvcc" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Runtime variables that may matter after installation:

| Variable | Use |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Selects or hides CUDA devices for the current process. |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Enables unified-memory fallback on Linux when VRAM is exhausted. On Windows, similar behavior may be controlled by NVIDIA driver settings. |
| `GGML_CUDA_P2P` | Enables peer-to-peer access between GPUs when driver and hardware support it. |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` | Forces FP32 compute in selected cuBLAS paths, trading speed for numerical headroom. |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F` | Forces FP16 compute in selected cuBLAS paths when supported. |

---

## BLAS and CPU Acceleration

BLAS acceleration mainly improves prompt processing and larger batch prefill.
It generally does not improve single-token generation speed as much as GPU
offload.

### OpenBLAS

Use OpenBLAS when the OpenBLAS development package is available on your system.

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

On Linux, install the OpenBLAS development package with your system package
manager before building. Package names vary by distribution.

### BLIS

BLIS is selected through the `FLAME` BLAS vendor after BLIS is installed:

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

The upstream BLIS guide also notes that runtime variables such as
`BLIS_NUM_THREADS` and OpenMP affinity settings can affect CPU performance.

### Intel oneMKL for CPU

Intel oneMKL is a CPU BLAS path. It is different from Intel GPU acceleration,
which is usually handled through SYCL or OpenVINO.

```bash
source /opt/intel/oneapi/setvars.sh
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

---

## Metal on macOS

On macOS, Metal is enabled by default by this project when building on Apple
platforms. A normal install is usually enough:

```bash
python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

To disable Metal at build time:

```bash
CMAKE_ARGS="-DGGML_METAL=OFF" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

At runtime, use `n_gpu_layers=0` when you want CPU inference even though the
package was built with Metal support.

---

## Vulkan

Vulkan builds require the Vulkan SDK and any platform-specific environment
setup required by the SDK.

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

On Linux and macOS, make sure the Vulkan SDK setup script has been sourced in
the same shell session before running `pip install`.

On Windows, install the Vulkan SDK and make sure its environment variables are
available in the shell that runs the build.

On Linux, system packages can also provide the Vulkan loader and shader tools.
The upstream guide notes that SPIR-V headers may be required separately from
the Vulkan loader development package on some distributions.

For macOS Vulkan builds, Vulkan usually runs through a Metal translation layer.
The upstream guide builds Vulkan with Metal disabled:

```bash
CMAKE_ARGS="-DGGML_VULKAN=ON -DGGML_METAL=OFF" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

---

## OpenVINO

OpenVINO builds require the OpenVINO runtime and environment setup first.

Linux:

```bash
source /opt/intel/openvino/setupvars.sh
CMAKE_ARGS="-DGGML_OPENVINO=ON" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Windows:

```powershell
# Run this from a shell where OpenVINO setupvars.bat has been initialized,
# such as an OpenVINO command prompt, or initialize it through cmd first.
$env:CMAKE_ARGS = "-DGGML_OPENVINO=ON"
python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

The OpenVINO backend is intended for Intel CPU, GPU, and NPU workflows when the
OpenVINO runtime supports the target device.

Runtime variables:

| Variable | Use |
|---|---|
| `GGML_OPENVINO_DEVICE` | Selects `CPU`, `GPU`, `NPU`, or a specific GPU such as `GPU.0`. Defaults to CPU if unset or unavailable. |
| `GGML_OPENVINO_CACHE_DIR` | Enables OpenVINO model caching when set. Not supported on NPU devices according to upstream docs. |
| `GGML_OPENVINO_STATEFUL_EXECUTION` | Enables stateful KV-cache execution. Upstream docs recommend it for CPU/GPU performance and note it is not effective on NPU. |
| `GGML_OPENVINO_PREFILL_CHUNK_SIZE` | Controls NPU prefill chunk size. |
| `GGML_OPENVINO_PROFILING` | Enables OpenVINO profiling. |

Important limitations from the upstream OpenVINO backend docs:

- GPU stateless execution has known issues; use `GGML_OPENVINO_STATEFUL_EXECUTION=1` for GPU workflows.
- NPU runs may fail when context size is too large. Keep context size small for NPU workflows.
- Encoder models such as embedding and reranking models are not supported by the current OpenVINO backend implementation.
- Some benchmark workflows require Flash Attention enabled in the llama.cpp tool layer; in Python, verify behavior against your target model and backend.

---

## HIP / ROCm

HIP builds are for supported AMD GPUs.

Linux example:

```bash
CMAKE_ARGS="-DGGML_HIP=ON -DGPU_TARGETS=gfx1030" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

`GPU_TARGETS` is optional in some setups, but specifying your GPU architecture
can reduce build time and avoid unsupported target issues.

Windows ROCm builds are more environment-sensitive. The README currently
documents a TheRock ROCm workflow that sets `HIP_PATH`, `ROCM_PATH`,
`HIP_DEVICE_LIB_PATH`, compiler paths, `CMAKE_GENERATOR`, and `CMAKE_ARGS`
before running `pip install`.

For RDNA3 or CDNA hardware, upstream docs mention optional Flash Attention
acceleration through rocWMMA:

```bash
CMAKE_ARGS="-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DGGML_HIP_ROCWMMA_FATTN=ON" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Runtime variables that may matter:

| Variable | Use |
|---|---|
| `HIP_VISIBLE_DEVICES` | Selects visible HIP devices. |
| `HSA_OVERRIDE_GFX_VERSION` | Can help unsupported Linux GPUs use a nearby architecture value. Upstream docs note this is not supported on Windows. |
| `HIP_DEVICE_LIB_PATH` | Points to ROCm device bitcode libraries when clang cannot find them. |

---

## SYCL

SYCL builds are usually used with Intel oneAPI compilers.

```bash
source /opt/intel/oneapi/setvars.sh
CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

To request FP16 support:

```bash
CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Useful SYCL build options from the upstream backend docs:

| Option | Use |
|---|---|
| `GGML_SYCL_F16` | Enables FP16 build path. Test both FP32 and FP16 for your model and device. |
| `GGML_SYCL_TARGET` | Selects SYCL target type. Intel is the default target in upstream docs. |
| `GGML_SYCL_DEVICE_ARCH` | Selects device architecture when known. |
| `GGML_SYCL_GRAPH` | Enables the experimental SYCL graph extension. |
| `GGML_SYCL_DNN` | Enables oneDNN integration. |
| `GGML_SYCL_HOST_MEM_FALLBACK` | Allows host-memory fallback when device memory is full, at reduced speed. |
| `GGML_SYCL_SUPPORT_LEVEL_ZERO` | Enables Level Zero support for Intel GPU memory allocation. |

Useful SYCL runtime variables:

| Variable | Use |
|---|---|
| `ONEAPI_DEVICE_SELECTOR` | Selects a SYCL device, such as a specific Level Zero GPU. |
| `GGML_SYCL_ENABLE_FLASH_ATTN` | Enables or disables Flash Attention in the SYCL backend. |
| `GGML_SYCL_ENABLE_LEVEL_ZERO` | Uses Level Zero allocation when support was built in. |
| `GGML_SYCL_DISABLE_DNN` | Disables oneDNN path and uses oneMKL path. |
| `ZES_ENABLE_SYSMAN` | Helps query free GPU memory in some Intel GPU setups. |

---

## OpenCL

OpenCL support is documented upstream mainly for Qualcomm Adreno GPUs and
Snapdragon devices. It may also work on certain other OpenCL-capable GPUs, but
SYCL is usually preferred for modern Intel GPU workflows.

```bash
CMAKE_ARGS="-DGGML_OPENCL=ON" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

Useful OpenCL CMake options:

| Option | Default | Use |
|---|---|---|
| `GGML_OPENCL_EMBED_KERNELS` | `ON` | Embeds OpenCL kernels into the built binary or library. |
| `GGML_OPENCL_USE_ADRENO_KERNELS` | `ON` | Enables kernels optimized for Adreno. |

For Linux builds where OpenCL headers and ICD loader are installed in a custom
prefix, pass that location through `CMAKE_PREFIX_PATH`.

---

## CANN

CANN is the Ascend NPU backend. It requires Ascend drivers and the CANN toolkit
before building.

```bash
CMAKE_ARGS="-DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

The upstream CANN documentation focuses on Linux and Ascend devices such as
Atlas 300I A2 and Atlas 300I Duo. Supported model families and data types vary
by device generation.

---

## ZenDNN and zDNN

ZenDNN and zDNN are different backends.

| Backend | Hardware | CMake option |
|---|---|---|
| ZenDNN | AMD Zen CPUs, especially AMD EPYC | `-DGGML_ZENDNN=ON` |
| zDNN | IBM Z / LinuxONE with NNPA acceleration | `-DGGML_ZDNN=ON -DZDNN_ROOT=/path/to/zdnn` |

ZenDNN can be downloaded and built automatically by CMake:

```bash
CMAKE_ARGS="-DGGML_ZENDNN=ON -DCMAKE_BUILD_TYPE=Release" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

If you already have a ZenDNN installation:

```bash
CMAKE_ARGS="-DGGML_ZENDNN=ON -DZENDNN_ROOT=/path/to/ZenDNN/build/install" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

zDNN requires a zDNN library installation first:

```bash
CMAKE_ARGS="-DGGML_ZDNN=ON -DZDNN_ROOT=/opt/zdnn-libs" \
  python -m pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

ZenDNN currently accelerates matrix multiplication paths and may fall back to
the standard CPU backend for other operations.

---

## Dynamic Backend Wheels

The README notes that newer preview wheels may be built with:

```text
GGML_BACKEND_DL=ON
GGML_CPU_ALL_VARIANTS=ON
```

In that build mode, CPU backend variants are installed as separate runtime
libraries under:

```text
site-packages/llama_cpp/lib
```

Examples include:

```text
ggml-cpu-x64
ggml-cpu-sse42
ggml-cpu-haswell
ggml-cpu-skylakex
ggml-cpu-alderlake
ggml-cpu-zen4
```

On Windows, dynamic CPU backend DLLs may also need the LLVM OpenMP runtime
next to them:

```text
libomp140.x86_64.dll
```

Based on the current top-level `CMakeLists.txt`, this project installs many
`llama`, `ggml`, CPU-variant, accelerator backend, and `mtmd` targets into the
Python package runtime directory when those targets are available.

---

## Upgrading and Rebuilding

Use `--upgrade`, `--force-reinstall`, and `--no-cache-dir` when you need to
force a rebuild with new CMake options:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" \
  python -m pip install --upgrade --force-reinstall --no-cache-dir \
  "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

This is important because `pip` may otherwise reuse cached wheels or build
artifacts from a previous backend configuration.

For local editable builds, clean old native artifacts before rebuilding when
switching backends:

```bash
make clean
python -m pip install --verbose -e .
```

On Windows, if `make` is not available, remove `_skbuild` and old native
libraries under `llama_cpp/lib` manually before reinstalling.

---

## Verify Installation

Check that the package imports:

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

Check where the package was installed:

```bash
python -c "import llama_cpp, pathlib; print(pathlib.Path(llama_cpp.__file__).parent)"
```

Check the bundled native runtime libraries:

```bash
python -c "import llama_cpp, pathlib; print(list((pathlib.Path(llama_cpp.__file__).parent / 'lib').glob('*')))"
```

Run a minimal model load after downloading a GGUF model:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_gpu_layers=0,
    verbose=False,
)

output = llm("Hello,", max_tokens=8)
print(output["choices"][0]["text"])
```

For GPU builds, set `n_gpu_layers=-1` or another positive value to offload
layers:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_gpu_layers=-1,
)
```

---

## Development Workflow

Common local development commands:

```bash
git clone https://github.com/JamePeng/llama-cpp-python --recursive
cd llama-cpp-python
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest
```

The repository also includes a `Makefile` with useful targets:

| Target | Purpose |
|---|---|
| `make build` | Editable build with verbose output. |
| `make build.cuda` | Editable build with `GGML_CUDA=on`. |
| `make build.openblas` | Editable build with OpenBLAS. |
| `make build.openvino` | Editable build with OpenVINO. |
| `make build.vulkan` | Editable build with Vulkan. |
| `make build.sycl` | Editable build with SYCL. |
| `make test` | Run pytest with verbose tracing. |
| `make clean` | Remove local native build artifacts. |

When testing a different `llama.cpp` commit, update the `vendor/llama.cpp`
submodule, clean the local build, and reinstall. If the upstream C API changes,
the ctypes declarations in `llama_cpp/llama_cpp.py` may also need to be updated.

---

## Common Installation Pitfalls

| Symptom | Likely cause | What to try |
|---|---|---|
| CMake cannot find a compiler | Build tools are missing or not available in the current shell. | Install platform build tools and reopen the terminal. On Windows, use a Developer PowerShell or initialize Visual Studio build variables. |
| Build ignores new backend flags | `pip` reused a cached wheel or previous build. | Reinstall with `--force-reinstall --no-cache-dir`, and clean `_skbuild` for local builds. |
| CUDA backend does not build | CUDA Toolkit is missing, incompatible, or not on `PATH`. | Verify `nvcc --version`, CUDA driver compatibility, and `CUDA_PATH` on Windows. |
| CUDA build targets the wrong GPU generation | Native architecture detection picked the build machine GPU, or `nvcc` could not detect it. | Use `-DGGML_NATIVE=OFF` for portability or set `-DCMAKE_CUDA_ARCHITECTURES=...` explicitly. |
| Native library fails to load on Windows | Required DLLs are missing from `PATH` or `llama_cpp/lib`. | Check `llama_cpp/lib` for `llama.dll`, `ggml*.dll`, backend DLLs, and runtime DLLs such as OpenMP or CUDA dependencies. |
| GPU is not used at runtime | The package was built without that backend or `n_gpu_layers` is `0`. | Rebuild with the correct CMake backend flag and set `n_gpu_layers` to a positive value or `-1`. |
| OpenVINO GPU or NPU behaves unexpectedly | Runtime device selection or context size is unsuitable. | Set `GGML_OPENVINO_DEVICE`, enable `GGML_OPENVINO_STATEFUL_EXECUTION=1` for GPU, and keep context size smaller for NPU workflows. |
| SYCL device is not selected | oneAPI environment or device selector is missing. | Source oneAPI setup and set `ONEAPI_DEVICE_SELECTOR` for the intended device. |
| Submodule files are missing | Repository was cloned without `--recursive`. | Run `git submodule update --init --recursive`. |

For detailed diagnostics, see [[Troubleshooting]].

---

## Related Links

* [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
* [README Installation](https://github.com/JamePeng/llama-cpp-python/blob/main/README.md#installation)
* [llama.cpp build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
* [llama.cpp backend documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend)
