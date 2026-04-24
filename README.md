<p align="center">
  <img src="https://raw.githubusercontent.com/abetlen/llama-cpp-python/main/docs/icon.svg" style="height: 5rem; width: 5rem">
</p>

#  Python Bindings for [`llama.cpp`](https://github.com/ggml-org/llama.cpp)

[![Documentation Status](https://readthedocs.org/projects/llama-cpp-python/badge/?version=latest)](https://llama-cpp-python.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/JamePeng/llama-cpp-python/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/JamePeng/llama-cpp-python/actions/workflows/test.yaml)
![GitHub Tag](https://img.shields.io/github/v/tag/JamePeng/llama-cpp-python)
[![PyPI - License](https://img.shields.io/pypi/l/llama-cpp-python)](https://pypi.org/project/llama-cpp-python/)
[![PyPI - Downloads](https://static.pepy.tech/badge/llama-cpp-python/month)](https://pepy.tech/projects/llama-cpp-python)
[![Github All Releases](https://img.shields.io/github/downloads/abetlen/llama-cpp-python/total.svg?label=Github%20Downloads)]()

Simple Python bindings for **@ggerganov's** [`llama.cpp`](https://github.com/ggml-org/llama.cpp) library.
This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for text completion
    - OpenAI-like API
    - [LangChain compatibility](https://python.langchain.com/docs/integrations/llms/llamacpp)
    - [LlamaIndex compatibility](https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp.html)
- OpenAI compatible web server
    - [Local Copilot replacement](https://llama-cpp-python.readthedocs.io/en/latest/server/#code-completion)
    - [Function Calling support](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling)
    - [Vision API support](https://llama-cpp-python.readthedocs.io/en/latest/server/#multimodal-models)
    - [Multiple Models](https://llama-cpp-python.readthedocs.io/en/latest/server/#configuration-and-multi-model-support)

Documentation is available at [https://llama-cpp-python.readthedocs.io/en/latest](https://llama-cpp-python.readthedocs.io/en/latest).


## Discussions

Starting March 2026, I am excited to announce that we have officially enabled the **Discussions** tab for `llama-cpp-python`!

You can access it right here: [GitHub Discussions](https://github.com/JamePeng/llama-cpp-python/discussions).

**Why Discussions? & Updates on Documentation**
As the project has evolved, our existing documentation (`docs`) has unfortunately become a bit bloated and outdated. To provide you with more timely and clear information:

* **New Feature Releases:** Moving forward, whenever a new feature is rolled out, I will publish a dedicated standalone article in the Discussions section. These posts will include detailed explanations, usage guides, and important caveats.
* This approach will serve as a more agile and interactive "live documentation" while we figure out the best way to refactor the old docs.

**Join the Community**
I warmly welcome all of you to use this new space. Let's build together:

* 💬 **Discuss & Share:** Have a question, an idea, or a cool use case? Share it with the community!
* 🛠️ **Maintain & Test:** Help us test new features, troubleshoot issues, and collaboratively maintain the repository.
* 📚 **Learn & Grow:** I hope everyone can benefit from this project, learn from each other, and gain valuable insights.

Thank you for your continuous support!

## Installation

Requirements:

  - Python 3.9+
  - C compiler
      - Linux: gcc or clang
      - Windows: [`Visual Studio 2022 Build Tools`](https://download.visualstudio.microsoft.com/download/pr/6efb3484-905b-485c-8b5f-9d3a5f39e731/07908cd6d91e75b8ea4339d8f2cfa6e8d8bb4fd706af7b918ae391cd6fc2a066/vs_BuildTools.exe) or `MinGW`
      - MacOS: Xcode
  - CMake 3.21+
  - Git

To install the package, run:
- Method 1:
    ```bash
    pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
    ```
- Method 2:
    ```bash
    git clone https://github.com/JamePeng/llama-cpp-python --recursive
    cd llama-cpp-python
    python -m pip install -U pip
    pip install .
    ```

This will also build `llama.cpp` from source and install it alongside this python package.

If this fails, add `--verbose` to the `pip install` see the full cmake build log.

### Installation Configuration

`llama.cpp` supports a number of hardware acceleration backends to speed up inference as well as backend specific options. See the [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for a full list.

All `llama.cpp` cmake build options can be set via the `CMAKE_ARGS` environment variable or via the `--config-settings / -C` cli flag during installation.

<details open>
<summary>Environment Variables</summary>

```bash
# Linux and Mac
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

```powershell
# Windows
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```
</details>

<details>
<summary>CLI / requirements.txt</summary>

They can also be set via `pip install -C / --config-settings` command and saved to a `requirements.txt` file:

```bash
pip install --upgrade pip # ensure pip is up to date
pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git" \
  -C cmake.args="-DGGML_BLAS=ON;-DGGML_BLAS_VENDOR=OpenBLAS"
```

```txt
# requirements.txt

llama-cpp-python -C cmake.args="-DGGML_BLAS=ON;-DGGML_BLAS_VENDOR=OpenBLAS"
```

</details>

### Supported Backends

Below are some common backends, their build commands and any additional environment variables required.

<details open>
<summary>CUDA</summary>

Installing a CUDA-supported version requires the `CUDA Toolkit` environment to be installed first.

**Note: Please select and install according to your system environment and local graphics card model to ensure that the compilation is based on the optimal local environment.**

See here: https://developer.nvidia.com/cuda-toolkit-archive

More Information see: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda

Then, set the `GGML_CUDA=on` environment variable before installing:

```bash
# Linux
CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

```powershell
# Windows
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

**Pre-built Wheel (New)**

It is also possible to install a pre-built wheel with CUDA support. As long as your system meets some requirements:

- CUDA Version is 12.4, 12.6, 12.8 or 13.0
- Python Version is 3.10, 3.11, 3.12, 3.13 or 3.14
- Basic version(Default): A version compiled without using AVX instructions (for compatibility with CPU platforms lacking AVX instructions or with AVX instruction compatibility issues).
- AVX2 version: A version compiled using AVX2 instructions.

Check the releases page:
https://github.com/JamePeng/llama-cpp-python/releases

</details>

<details>
<summary>OpenBLAS (CPU)</summary>

To install with OpenBLAS, set the `GGML_BLAS` and `GGML_BLAS_VENDOR` environment variables before installing:

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```
</details>

<details>
<summary>OpenVINO</summary>

### Install OpenVINO Runtime

Follow the guide to install OpenVINO Runtime from an archive file: [Linux](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html) | [Windows](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-windows.html)

- **Linux:**

    <details>
    <summary>📦 Click to expand OpenVINO installation from an archive file on Ubuntu</summary>
    <br>

    ```bash
    wget https://raw.githubusercontent.com/ravi9/misc-scripts/main/openvino/ov-archive-install/install-openvino-from-archive.sh
    chmod +x install-openvino-from-archive.sh
    ./install-openvino-from-archive.sh
    ```

    Verify OpenVINO is initialized properly:
    ```bash
    echo $OpenVINO_DIR
    ```
    </details>

### Supported Devices

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel GPUs (integrated and discrete)
- Intel NPUs

Although OpenVINO supports a wide range of [Intel hardware](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html), the llama.cpp OpenVINO backend has been validated specifically on AI PCs such as the Intel® Core™ Ultra Series 1 and Series 2.

More Information see: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md

To install with OpenVINO, set the `GGML_OPENVINO=ON` environment variable before installing:

```bash
# Linux
source /opt/intel/openvino/setupvars.sh
# Windows
"C:\Program Files (x86)\Intel\openvino_2026.0\setupvars.bat"
# Build
CMAKE_ARGS="-DGGML_OPENVINO=ON" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```
</details>

<details>
<summary>Metal</summary>

On MacOS, Metal is enabled by default(`GGML_METAL=ON`). Using Metal makes the computation run on the GPU.

To disable the Metal build at compile time use the `CMAKE_ARGS="-DGGML_METAL=OFF"` cmake option.

```bash
pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

**Pre-built Wheel (New)**

It is also possible to install a pre-built wheel with Metal support. As long as your system meets some requirements:

- MacOS Version is 11.0 or later
- Python Version is 3.10, 3.11, 3.12, 3.13 or 3.14

Check the releases page:
https://github.com/JamePeng/llama-cpp-python/releases

</details>

<details>
<summary>HIP (ROCm)</summary>

This provides GPU acceleration on HIP-supported AMD GPUs. Make sure to have ROCm installed.

You can download it from your Linux distro's package manager or from here: [ROCm Quick Start (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick).

To install with HIP / ROCm support for AMD cards, set the `GGML_HIP=ON` environment variable before installing:

```bash
CMAKE_ARGS="-DGGML_HIP=ON -DGPU_TARGETS=gfx1030" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```
Note: `GPU_TARGETS` is optional, omitting it will build the code for all GPUs in the current system.

More details see here: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip

</details>

<details>
<summary>Vulkan</summary>

- For Windows User: Download and install the [`Vulkan SDK`](https://vulkan.lunarg.com/sdk/home#windows) with the default settings.

- For Linux User:
    * First, follow the official LunarG instructions for the installation and setup of the Vulkan SDK in the [Getting Started with the Linux Tarball Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html) guide.

    * After completing the first step, ensure that you have used the `source` command on the `setup_env.sh` file inside of the Vulkan SDK in your current terminal session. Otherwise, the build won't work. Additionally, if you close out of your terminal, you must perform this step again if you intend to perform a build. However, there are ways to make this persistent. Refer to the Vulkan SDK guide linked in the first step for more information about any of this.

- For Mac User:
    * Generally, follow LunarG's [Getting Started with the MacOS Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html) guide for installation and setup of the Vulkan SDK. There are two options of Vulkan drivers on macOS, both of which implement translation layers to map Vulkan to Metal. They can be hot-swapped by setting the `VK_ICD_FILENAMES` environment variable to point to the respective ICD JSON file. Check the box for "KosmicKrisp" during the LunarG Vulkan SDK installation.

    * Set environment variable for the LunarG Vulkan SDK after installation (and optionally add to your shell profile for persistence):
        ```bash
        source /path/to/vulkan-sdk/setup-env.sh
        ```

More Information see: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan

Then install with Vulkan support by set the `GGML_VULKAN=on` environment variable before installing:

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

</details>

<details>
<summary>SYCL</summary>

### Supported OS

| OS      | Status  | Verified                                       |
|---------|---------|------------------------------------------------|
| Linux   | Support | Ubuntu 22.04, Fedora Silverblue 39, Arch Linux |
| Windows | Support | Windows 11                                     |

### Intel GPU

SYCL backend supports Intel GPU Family:

- Intel Data Center Max Series
- Intel Flex Series, Arc Series
- Intel Built-in Arc GPU
- Intel iGPU in Core CPU (11th Generation Core CPU and newer, refer to [oneAPI supported GPU](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html#inpage-nav-1-1)).

On older Intel GPUs, you may try [OpenCL](/docs/backend/OPENCL.md) although the performance is not optimal, and some GPUs may not support OpenCL nor have any GPGPU capabilities.

More Information see here: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md

To install with SYCL support, set the `GGML_SYCL=on` environment variable before installing:

```bash
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh
# Option 1: Use FP32 (recommended for better performance in most cases)
CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
# Option 2: Use FP16
CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```
</details>

<details>
<summary>RPC</summary>

To install with RPC support, set the `GGML_RPC=on` environment variable before installing:

```bash
source /opt/intel/oneapi/setvars.sh   
CMAKE_ARGS="-DGGML_RPC=on" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```
</details>


### Windows Notes

<details>
<summary>Error: Can't find 'nmake' or 'CMAKE_C_COMPILER'</summary>

If you run into issues where it complains it can't find `'nmake'` `'?'` or CMAKE_C_COMPILER, you can extract w64devkit as [mentioned in llama.cpp repo](https://github.com/ggerganov/llama.cpp#openblas) and add those manually to CMAKE_ARGS before running `pip` install:

```ps
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DGGML_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe"
```

See the above instructions and set `CMAKE_ARGS` to the BLAS backend you want to use.
</details>

### MacOS Notes

Detailed MacOS Metal GPU install documentation is available at [docs/install/macos.md](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

<details>
<summary>M1 Mac Performance Issue</summary>

Note: If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64 architecture. For example:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

Otherwise, while installing it will build the llama.cpp x86 version which will be 10x slower on Apple Silicon (M1) Mac.
</details>

<details>
<summary>M Series Mac Error: `(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))`</summary>

Try installing with

```bash
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```
</details>

### Upgrading and Reinstalling

To upgrade and rebuild `llama-cpp-python` add `--upgrade --force-reinstall --no-cache-dir` flags to the `pip install` command to ensure the package is rebuilt from source.

## High-level API

[API Reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api)

The high-level API provides a simple managed interface through the [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) class.

Below is a short example demonstrating how to use the high-level API to for basic text completion:

```python
from llama_cpp import Llama

llm = Llama(
      model_path="./models/7B/llama-model.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)
```

By default `llama-cpp-python` generates completions in an OpenAI compatible format:

```python
{
  "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1679561337,
  "model": "./models/7B/llama-model.gguf",
  "choices": [
    {
      "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
      "index": 0,
      "logprobs": None,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 28,
    "total_tokens": 42
  }
}
```

Text completion is available through the [`__call__`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__) and [`create_completion`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion) methods of the [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) class.

### Pulling models from [Hugging Face Hub](https://huggingface.co/models)

You can download `Llama` models in `gguf` format directly from Hugging Face using the [`from_pretrained`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.from_pretrained) method.

You'll need to install the `huggingface_hub` package to use this feature (`pip install --upgrade huggingface_hub`).



```python
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    verbose=False
)
```

By default [`from_pretrained`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.from_pretrained) will download the model to the huggingface cache directory, you can then manage installed model files with the [`hf`](https://huggingface.co/docs/huggingface_hub/en/guides/cli) tool.

### Chat Completion

The high-level API also provides a simple interface for chat completion.

Chat completion requires that the model knows how to format the messages into a single prompt.
The `Llama` class does this using pre-registered chat formats (ie. `chatml`, `llama-2`, `gemma`, etc) or by providing a custom chat handler object.

The model will will format the messages into a single prompt using the following order of precedence:
  - Use the `chat_handler` if provided
  - Use the `chat_format` if provided
  - Use the `tokenizer.chat_template` from the `gguf` model's metadata (should work for most new models, older models may not have this)
  - else, fallback to the `llama-2` chat format

Set `verbose=True` to see the selected chat format.

```python
from llama_cpp import Llama
llm = Llama(
      model_path="path/to/llama-2/llama-model.gguf",
      chat_format="llama-2"
)
llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
              "content": "Describe this image in detail please."
          }
      ]
)
```

Chat completion is available through the [`create_chat_completion`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion) method of the [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) class.

For OpenAI API v1 compatibility, you use the [`create_chat_completion_openai_v1`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion_openai_v1) method which will return pydantic models instead of dicts.


### JSON and JSON Schema Mode

To constrain chat responses to only valid JSON or a specific JSON Schema use the `response_format` argument in [`create_chat_completion`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).

#### JSON Mode

The following example will constrain the response to valid JSON strings only.

```python
from llama_cpp import Llama
llm = Llama(model_path="path/to/model.gguf", chat_format="chatml")
llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
    },
    temperature=0.7,
)
```

#### JSON Schema Mode

To constrain the response further to a specific JSON Schema add the schema to the `schema` property of the `response_format` argument.

```python
from llama_cpp import Llama
llm = Llama(model_path="path/to/model.gguf", chat_format="chatml")
llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"team_name": {"type": "string"}},
            "required": ["team_name"],
        },
    },
    temperature=0.7,
)
```

### Function Calling

The high-level API supports OpenAI compatible function and tool calling. This is possible through the `functionary` pre-trained models chat format or through the generic `chatml-function-calling` chat format.

```python
from llama_cpp import Llama
llm = Llama(model_path="path/to/chatml/llama-model.gguf", chat_format="chatml-function-calling")
llm.create_chat_completion(
      messages = [
        {
          "role": "system",
          "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

        },
        {
          "role": "user",
          "content": "Extract Jason is 25 years old"
        }
      ],
      tools=[{
        "type": "function",
        "function": {
          "name": "UserDetail",
          "parameters": {
            "type": "object",
            "title": "UserDetail",
            "properties": {
              "name": {
                "title": "Name",
                "type": "string"
              },
              "age": {
                "title": "Age",
                "type": "integer"
              }
            },
            "required": [ "name", "age" ]
          }
        }
      }],
      tool_choice={
        "type": "function",
        "function": {
          "name": "UserDetail"
        }
      }
)
```

<details>
<summary>Functionary v2</summary>

The various gguf-converted files for this set of models can be found [here](https://huggingface.co/meetkai). Functionary is able to intelligently call functions and also analyze any provided function outputs to generate coherent responses. All v2 models of functionary supports **parallel function calling**. You can provide either `functionary-v1` or `functionary-v2` for the `chat_format` when initializing the Llama class.

Due to discrepancies between llama.cpp and HuggingFace's tokenizers, it is required to provide HF Tokenizer for functionary. The `LlamaHFTokenizer` class can be initialized and passed into the Llama class. This will override the default llama.cpp tokenizer used in Llama class. The tokenizer files are already included in the respective HF repositories hosting the gguf files.

```python
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
llm = Llama.from_pretrained(
  repo_id="meetkai/functionary-small-v2.2-GGUF",
  filename="functionary-small-v2.2.q4_0.gguf",
  chat_format="functionary-v2",
  tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.2-GGUF")
)
```

**NOTE**: There is no need to provide the default system messages used in Functionary as they are added automatically in the Functionary chat handler. Thus, the messages should contain just the chat messages and/or system messages that provide additional context for the model (e.g.: datetime, etc.).
</details>

---

## Continuing Assistant Responses (Prefill)

`llama-cpp-python` supports native **Assistant Prefill** for seamless message continuation. You can now simply use the `assistant_prefill=True` parameter in the `create_chat_completion` function.

This safely renders the `N-1` conversation history using standard Jinja templates (preserving exact control tokens) and flawlessly appends your partial text directly to the prompt.

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# An interrupted/partial conversation
messages = [
    {"role": "user", "content": "What are the first 5 planets in the solar system?"},
    {"role": "assistant", "content": "The first 5 planets in our solar system are:\n1. Mercury\n2."}
]

# Seamlessly continue the generation
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=50,
    assistant_prefill=True # <--- Enables seamless continuation
)

prefilled_text = messages[-1]["content"]
# The model will flawlessly continue from " Venus\n3. Earth..."
generated_text = response["choices"][0]["message"]["content"]

print(prefilled_text + generated_text)
```

---

## Dynamic LoRA Routing & Control Vectors (Multi-Tenant Serving)

Historically, `llama-cpp-python` only supported "static loading" where a LoRA was permanently baked into the context during initialization. Switching personas required reloading the entire model or duplicating it in VRAM.

`llama-cpp-python` now supports **Just-In-Time (JIT)** dynamic adapter routing. Instead of statically binding a single LoRA to a model during initialization (which locks the instance to a single task), you can now preload multiple adapters into VRAM and seamlessly apply them on-the-fly per request.

This architecture unlocks true **Multi-Tenant Serving**:
* **Zero-Latency Switching:** Compute graph weights are atomically modified in C++ memory instantly before evaluation.
* **VRAM Efficiency:** You only load the heavy base model once. Multiple LoRAs share the same base model memory.
* **Thread-Safe & Contamination-Free:** Strict internal state debouncing ensures that weights are perfectly cleaned between requests, guaranteeing zero persona contamination.

### Dynamic LoRA Example

```python
from llama_cpp import Llama

# 1. Load the pure base model once
llm = Llama(model_path="path/to/llama-3-8b.gguf")

# 2. Preload multiple LoRAs into VRAM
llm.load_lora("python_coder", "path/to/python-coder-lora.gguf")
llm.load_lora("translator", "path/to/spanish-translator-lora.gguf")

# 3. User A: Coding Task (Instantly applies the coder LoRA)
response_a = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Write a fast inverse square root in C."}],
    active_loras=[{"name": "python_coder", "scale": 1.0}]
)

# 4. User B: Translation Task (Zero-latency switch to the translator LoRA)
response_b = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Explain quantum physics in Spanish."}],
    active_loras=[{"name": "translator", "scale": 0.85}] # Apply at 85% strength
)

# 5. User C: General Query (Automatically wipes graph weights for a clean base model state)
response_c = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

# 6. Cleanup (Optional: manually free VRAM for specific LoRAs)
llm.unload_lora("python_coder")
```

### Control Vector Injection (Representation Engineering)

In addition to LoRA, the API supports dynamic injection of **Control Vectors (CVec)**. This allows you to steer the model's behavior, emotion, or alignment by directly modifying the activation values at specific hidden layers, without needing `.gguf` weight files.

```python
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Tell me a story about a futuristic city."}],
    control_vector={
        "data": [...],         # A flattened 1D list of floats representing the vector
        "layer_start": 15,     # Apply starting from this layer (inclusive)
        "layer_end": 32        # Apply up to this layer (inclusive)
    }
)
```
*Note(JamePeng): Ensure your `data` array length exactly matches `embedding_length * layer_end`. The C++ backend maps the buffer continuously starting from layer 1, so early skipped layers must be zero-padded in your array.*

---

##  Sampling Configuration & Usage (LlamaSamplingParams)

The `Llama` class provides extensive control over the `llama.cpp` sampling chain during text generation. You can configure state-of-the-art sampling algorithms, dynamic temperature, and advanced repetition penalties directly via the `generate`, `create_completion`, or `__call__` methods.

### Core Sampling Parameters

These are the most common parameters used to control the randomness and focus of the model's output.

* **`temperature`** (`float`, default: `0.80`): Controls the randomness of the generation. Higher values (e.g., `1.0`) make output more random, while lower values (e.g., `0.2`) make it more deterministic. Set to `<= 0.0` for greedy decoding.
* **`top_k`** (`int`, default: `40`): Limits the next token selection to the K most probable tokens. Set to `<= 0` to use the full vocabulary size.
* **`top_p`** (`float`, default: `0.95`): Nucleus sampling. Limits selection to a cumulative probability of P. Set to `1.0` to disable.
* **`min_p`** (`float`, default: `0.05`): Minimum P sampling. Drops tokens with a probability less than `min_p` relative to the most likely token. Set to `0.0` to disable.
* **`typical_p`** (`float`, default: `1.0`): Locally typical sampling. Adjusts probabilities based on the entropy of the distribution. Set to `1.0` to disable.


### Advanced & Experimental Samplers

* **XTC (Exclude Top Choice)**: Removes the most likely tokens to force the model to take creative alternative paths.
* **`xtc_probability`** (`float`, default: `0.0`): The chance for token removal. `0.0` disables XTC.
* **`xtc_threshold`** (`float`, default: `0.1`): The minimum probability threshold for a token to be considered for removal.


* **Dynamic Temperature**: Adjusts the temperature dynamically based on the entropy of the current token distribution.
* **`dynatemp_range`** (`float`, default: `0.0`): The range of the dynamic temperature. `0.0` disables it.
* **`dynatemp_exponent`** (`float`, default: `1.0`): Controls how entropy maps to temperature.


* **`top_n_sigma`** (`float`, default: `-1.0`): Limits selection to tokens with pre-softmax logits within $n * \sigma$ of the max logit. Set to `-1.0` to disable.
* **Adaptive-P**: Dynamically adjusts the target probability using an exponential moving average (EMA).
* **`adaptive_target`** (`float`, default: `-1.0`): The target probability (0.0 to 1.0). Negative values disable it.
* **`adaptive_decay`** (`float`, default: `0.9`): The EMA decay rate (0.0 to 0.99).


### Target Entropy (Mirostat)

Mirostat actively maintains a target entropy (`tau`) during generation to prevent text from becoming too boring or too chaotic.

* **`mirostat_mode`** (`int`, default: `0`): `0` = disabled, `1` = Mirostat 1.0, `2` = Mirostat 2.0.
* **`mirostat_tau`** (`float`, default: `5.0`): The target cross-entropy (surprisal) value.
* **`mirostat_eta`** (`float`, default: `0.1`): The learning rate used to update the algorithm's internal state.


### Repetition Penalties

* **Standard Penalties**:
* **`repeat_penalty`** (`float`, default: `1.0`): General penalty for repeated tokens. `1.0` = disabled.
* **`frequency_penalty`** (`float`, default: `0.0`): Penalty based on the absolute frequency of a token in the prompt.
* **`present_penalty`** (`float`, default: `0.0`): Flat penalty applied if a token is present anywhere in the context.
* **`penalty_last_n`** (`int`, default: `64`): The number of recent tokens to consider for standard penalties. `0` = disabled, `-1` = full context size.


* **DRY (Don't Repeat Yourself)**: An advanced exponential penalty specifically designed to break exact repeating sequences.
* **`dry_multiplier`** (`float`, default: `0.0`): The multiplier for the penalty. `0.0` disables DRY.
* **`dry_base`** (`float`, default: `1.75`): The base value for the exponential penalty.
* **`dry_allowed_length`** (`int`, default: `2`): Sequences extending beyond this length receive the penalty.
* **`dry_penalty_last_n`** (`int`, default: `0`): Tokens to scan for repetitions. `0` = disabled, `-1` = context size.
* **`dry_seq_breakers`** (`list[str]`, default: `["\n", ":", "\"", "*"]`): Tokens that reset the DRY sequence matching.


### Constraints & Callbacks

* **`logit_bias`** (`Dict[int, float]`, optional): Manually boost or penalize specific token IDs.
* **`grammar`** (`LlamaGrammar`, optional): Force the model to generate text matching a specific BNF-like grammar (e.g., valid JSON).
* **`logits_processor`** (`LogitsProcessorList`, optional): Custom Python callbacks to modify the logits tensor in-place before sampling.
* **`stopping_criteria`** (`StoppingCriteriaList`, optional): Custom Python callbacks to halt generation based on the current sequence or scores.

### 🛠️ Usage Example

You can pass these parameters directly when calling the model to generate text.

```python
from llama_cpp import Llama

# Load the model
model = Llama(model_path="path/to/your/model.gguf")

# Generate text with advanced sampling
response = model.create_completion(
    prompt="The secret to a happy life is",
    max_tokens=100,
    # Adjust core randomness
    temperature=0.85,
    top_p=0.90,
    min_p=0.05,
    # Prevent the model from repeating specific phrases
    dry_multiplier=0.8,
    dry_base=1.75,
    dry_allowed_length=3,
    # Standard repetition penalty
    repeat_penalty=1.1,
    penalty_last_n=256,
)

print(response["choices"][0]["text"])

```

---

## Multi-modal Models

`llama-cpp-python` supports such as llava1.5 which allow the language model to read information from both text and images.

Below are the supported multi-modal models and their respective chat handlers (Python API) and chat formats (Server API).

| Model | `LlamaChatHandler` | `chat_format` |
|:--- |:--- |:--- |
| [llava-v1.5-7b](https://huggingface.co/mys/ggml_llava-v1.5-7b) | `Llava15ChatHandler` | `llava-1-5` |
| [llava-v1.6-34b](https://huggingface.co/cjpais/llava-v1.6-34B-gguf) | `Llava16ChatHandler` | `llava-1-6` |
| [moondream2](https://huggingface.co/vikhyatk/moondream2) | `MoondreamChatHandler` | `moondream2` |
| [nanollava](https://huggingface.co/abetlen/nanollava-gguf) | `NanollavaChatHandler` | `nanollava` |
| [llama-3-vision-alpha](https://huggingface.co/abetlen/llama-3-vision-alpha-gguf) | `Llama3VisionAlphaChatHandler` | `llama-3-vision-alpha` |
| [minicpm-v-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) | `MiniCPMv26ChatHandler` | `minicpm-v-2.6`, `minicpm-v-4.0` |
| [minicpm-v-4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) | `MiniCPMv45ChatHandler` | `minicpm-v-4.5` |
| [gemma3](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF) | `Gemma3ChatHandler` | `gemma3` |
| [gemma4](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | `Gemma4ChatHandler` | `gemma4` |
| [glm4.1v](https://huggingface.co/unsloth/GLM-4.1V-9B-Thinking-GGUF) | `GLM41VChatHandler` | `glm4.1v` |
| [glm4.6v](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF) | `GLM46VChatHandler` | `glm4.6v` |
| [granite-docling](https://huggingface.co/ibm-granite/granite-docling-258M-GGUF) | `GraniteDoclingChatHandler` | `granite-docling` |
| [lfm2-vl](https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF) | `LFM2VLChatHandler` | `lfm2-vl` |
| [lfm2.5-vl](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF) | `LFM25VLChatHandler` | `lfm2.5-vl` |
| [paddleocr-vl-1.5](https://huggingface.co/JamePeng2023/PaddleOCR-VL-1.5-GGUF) | `PaddleOCRChatHandler` | `paddleocr` |
| [qwen2.5-vl](https://huggingface.co/unsloth/Qwen2.5-VL-3B-Instruct-GGUF) | `Qwen25VLChatHandler` | `qwen2.5-vl` |
| [qwen3-vl](https://huggingface.co/unsloth/Qwen3-VL-8B-Thinking-GGUF) | `Qwen3VLChatHandler` | `qwen3-vl` |
| [qwen3.5](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | `Qwen35ChatHandler` | `qwen3.5` |
| [qwen3.6](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | `Qwen35ChatHandler` | `qwen3.6` |
| [step3-vl](https://huggingface.co/JamePeng2023/Step3-VL-10B-GGUF) | `Step3VLChatHandler` | `step3-vl` |

Then you'll need to use a custom chat handler to load the clip model and process the chat messages and images.

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

model_path="path/to/llava/ggml-model-f16.gguf"
mmproj_path="path/to/llava/mmproj-model-f16.gguf"

llm = Llama(
  model_path=model_path,
  chat_handler=Llava15ChatHandler(clip_model_path=mmproj_path),
  n_ctx=2048,
)

llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type" : "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg" } }
            ]
        }
    ]
)
```

You can also pull the model from the Hugging Face Hub using the `from_pretrained` method.

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

chat_handler = MoondreamChatHandler.from_pretrained(
  repo_id="vikhyatk/moondream2",
  filename="*mmproj*",
)

llm = Llama.from_pretrained(
  repo_id="vikhyatk/moondream2",
  filename="*text-model*",
  chat_handler=chat_handler,
  n_ctx=2048, # n_ctx should be increased to accommodate the image embedding
)

response = llm.create_chat_completion(
    messages = [
        {
            "role": "user",
            "content": [
                {"type" : "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg" } }

            ]
        }
    ]
)
print(response["choices"][0]["text"])
```

**Note**: Multi-modal models also support tool calling and JSON mode.


## Loading a Local Image With Qwen3VL(Thinking/Instruct)

<summary>This script demonstrates how to load a local image, encode it as a base64 Data URI, and pass it to a local Qwen3-VL model (with the 'force_reasoning' parameter enabled for thinking model, disabled for instruct model) for processing using the llama-cpp-python library.</summary><br>


**Example Code**: <details>

```python
# Import necessary libraries
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler
import base64
import os

# --- Model Configuration ---
# Define the path to the main model file
MODEL_PATH = r"./Qwen3-VL-8B-Thinking-F16.gguf"
# Define the path to the multi-modal projector file
MMPROJ_PATH = r"./mmproj-Qwen3-VL-8b-Thinking-F16.gguf"

# --- Initialize the Llama Model ---
llm = Llama(
    model_path=MODEL_PATH,
    # Set up the chat handler for Qwen3-VL, specifying the projector path
    chat_handler=Qwen3VLChatHandler(
      clip_model_path=MMPROJ_PATH,
      force_reasoning=True,  # Note: Some models use `enable_thinking` as a switch variable. See the comments in the corresponding model's chathandler for details.
      image_min_tokens=1024, # Note: Qwen3-VL models require at minimum 1024 image tokens to function correctly on bbox grounding tasks
    ),
    n_gpu_layers=-1,  # Offload all layers to the GPU
    n_ctx=10240,      # Set the context window size
    swa_full=True,
)

# Comprehensive MIME type mapping (updated as of 2025)
# Based on Pillow 10.x+ "Fully Supported" (Read & Write) formats
# Reference: IANA official media types + common real-world usage
# See: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
_IMAGE_MIME_TYPES = {
    # Most common formats
    '.png':  'image/png',
    '.jpg':  'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif':  'image/gif',
    '.webp': 'image/webp',

    # Next-generation formats
    '.avif': 'image/avif',
    '.jp2':  'image/jp2',
    '.j2k':  'image/jp2',
    '.jpx':  'image/jp2',

    # Legacy / Windows formats
    '.bmp':  'image/bmp',
    '.ico':  'image/x-icon',
    '.pcx':  'image/x-pcx',
    '.tga':  'image/x-tga',
    '.icns': 'image/icns',

    # Professional / Scientific imaging
    '.tif':  'image/tiff',
    '.tiff': 'image/tiff',
    '.eps':  'application/postscript',
    '.dds':  'image/vnd-ms.dds',
    '.dib':  'image/dib',
    '.sgi':  'image/sgi',

    # Portable Map formats (PPM/PGM/PBM)
    '.pbm':  'image/x-portable-bitmap',
    '.pgm':  'image/x-portable-graymap',
    '.ppm':  'image/x-portable-pixmap',

    # Miscellaneous / Older formats
    '.xbm':  'image/x-xbitmap',
    '.mpo':  'image/mpo',
    '.msp':  'image/msp',
    '.im':   'image/x-pillow-im',
    '.qoi':  'image/qoi',
}

def image_to_base64_data_uri(
    file_path: str,
    *,
    fallback_mime: str = "application/octet-stream"
) -> str:
    """
    Convert a local image file to a base64-encoded data URI with the correct MIME type.

    Supports 20+ image formats (PNG, JPEG, WebP, AVIF, BMP, ICO, TIFF, etc.).

    Args:
        file_path: Path to the image file on disk.
        fallback_mime: MIME type used when the file extension is unknown.

    Returns:
        A valid data URI string (e.g., data:image/webp;base64,...).

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If reading the file fails.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    mime_type = _IMAGE_MIME_TYPES.get(extension, fallback_mime)

    if mime_type == fallback_mime:
        print(f"Warning: Unknown extension '{extension}' for '{file_path}'. "
              f"Using fallback MIME type: {fallback_mime}")

    try:
        with open(file_path, "rb") as img_file:
            encoded_data = base64.b64encode(img_file.read()).decode("utf-8")
    except OSError as e:
        raise OSError(f"Failed to read image file '{file_path}': {e}") from e

    return f"data:{mime_type};base64,{encoded_data}"

# --- Main Logic for Image Processing ---

# 1. Create a list containing all image paths
image_paths = [
    r'./scene.jpeg',
    r'./cat.png',
    r'./network.webp',
    # Add more image paths here if needed
]

# 2. Create an empty list to store the message objects (images and text)
images_messages = []

# 3. Loop through the image path list, convert each image to a Data URI,
#    and add it to the message list as an image_url object.
for path in image_paths:
    data_uri = image_to_base64_data_uri(path)
    images_messages.append({"type": "image_url", "image_url": {"url": data_uri}})

# 4. Add the final text prompt at the end of the list
images_messages.append({"type": "text", "text": "Describes the images."})

# 5. Use this list to build the chat_completion request
res = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a highly accurate vision-language assistant. Provide detailed, precise, and well-structured image descriptions."},
        # The user's content is the list containing both images and text
        {"role": "user", "content": images_messages}
    ]
)

# Print the assistant's response
print(res["choices"][0]["message"]["content"])

```

</details>

## Comprehensive Omni MultiModal Example: Gemma-4 (Vision + Audio + Text)

Below is a complete, production-ready example demonstrating how to dynamically route and process both image and audio files. It includes a universal media processor that automatically converts local files into the correct payload structure (Data URIs for images, and `input_audio` for audio files).

> **⚠️ IMPORTANT: GEMMA-4 MODEL CAPABILITIES & LIMITATIONS**
> * **Gemma4 E2B / E4B:** Supports Full Multimodal (Vision + Audio + Text). `enable_thinking` **MUST** be `True`(default).
> * **Gemma4 31B / 26BA4B:** Supports Vision + Text ONLY (Audio is NOT supported). `enable_thinking` can be toggled (`True` or `False`).

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler
import base64
import os

# Model and multimodal projection paths
MODEL_PATH = r"/path/to/Gemma-4-E4B-It-BF16.gguf"
# BF16 mmproj is required for audio. Other quantizations are known to have degraded performance.
MMPROJ_PATH = r"/path/to/mmproj-Gemma-4-E4B-It-BF16.gguf"

# Initialize the Llama model with multimodal support
# Note: Since we are using E4B here, enable_thinking MUST be True, and audio is supported.
llm = Llama(
    model_path=MODEL_PATH,
    chat_handler=Gemma4ChatHandler(
        clip_model_path=MMPROJ_PATH,
        enable_thinking=True,  # MUST be True for E2B/E4B models
        verbose=True,          # Enable Debug Info
    ),
    n_gpu_layers=-1,
    n_ctx=10240,
    verbose=True,              # Enable Debug Info
)

# 1. Extend the MIME dictionary to support audio formats
_MEDIA_MIME_TYPES = {
    # ------ Image formats ------
    '.png':  ('image', 'image/png'),
    '.jpg':  ('image', 'image/jpeg'),
    '.jpeg': ('image', 'image/jpeg'),
    '.gif':  ('image', 'image/gif'),
    '.webp': ('image', 'image/webp'),
    '.bmp':  ('image', 'image/bmp'),

    # ------ Audio formats ------
    '.wav':  ('audio', 'wav'),    # OpenAI standard usually uses raw format names for audio
    '.mp3':  ('audio', 'mp3'),
    # '.flac': ('audio', 'flac'),
}

def build_media_payload(file_path: str) -> dict:
    """
    Read a local media file (image or audio) and convert it into a valid input payload for the LLM.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Media file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    media_category, mime_or_format = _MEDIA_MIME_TYPES.get(extension, ('unknown', 'application/octet-stream'))

    if media_category == 'unknown':
        print(f"Warning: Unknown extension '{extension}'. It might not be processed correctly.")

    # Read and Base64 encode the file
    with open(file_path, "rb") as f:
        encoded_data = base64.b64encode(f.read()).decode("utf-8")

    # 2. Return the appropriate dictionary structure based on the media type
    if media_category == 'image':
        # Image format: Data URI (OpenAI compatible)
        data_uri = f"data:{mime_or_format};base64,{encoded_data}"
        return {
            "type": "image_url",
            "image_url": {"url": data_uri}
        }

    elif media_category == 'audio':
        # Audio format: input_audio (OpenAI compatible)
        return {
            "type": "input_audio",
            "input_audio": {
                "data": encoded_data,
                "format": mime_or_format
            }
        }
    else:
        # Fallback for unsupported formats
        return {"type": "text", "text": f"[Attached unsupported file: {file_path}]"}


def run_inference(media_paths: list, text_prompt: str):
    """
    Helper function to dynamically build the payload and run inference.
    """
    # 3. Build the user_content list
    user_content = []

    # Automatically parse each file and append to the payload
    for path in media_paths:
        payload = build_media_payload(path)
        user_content.append(payload)

    # Append the final text instruction
    user_content.append({
        "type": "text",
        "text": text_prompt
    })

    print(f"\n--- Running Inference with {len(media_paths)} media file(s) ---")

    # 4. Send to the model for inference
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": """
            You are a highly capable multimodal assistant that can process both text, vision and audio.

            """}, # Note: Supported ONLY by Gemma4 E2B / E4B.
            {"role": "user", "content": user_content}
        ],
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=8192,
    )

    print("\n[Model Response]:")
    print(response["choices"][0]["message"]["content"])
    print("-" * 60)


# ==============================================================================
# Main Inference Examples
# Uncomment the example block you wish to execute.
# ==============================================================================

# --- Example A: Image + Audio (Full Multimodal) ---
# Note: Supported ONLY by Gemma4 E2B / E4B.
run_inference(
    media_paths=[r"/path/to/test.png", r"/path/to/test.wav"],
    text_prompt="Introduce the content by combining the images and converting the audio to text."
)

# --- Example B: Image Only (Vision + Text) ---
# Note: Supported by all Gemma4 variants (E2B, E4B, 31B, 26BA4B).
# run_inference(
#     media_paths=[r"/path/to/test.png"],
#     text_prompt="Describe the contents of this image in detail."
# )

# --- Example C: Audio Only (Audio + Text) ---
# Note: Supported ONLY by Gemma4 E2B / E4B.
# run_inference(
#     media_paths=[r"/path/to/test.wav"],
#     text_prompt="Transcribe this audio and summarize the main points."
# )
```


---

## Embeddings & Reranking (GGUF)

`llama-cpp-python` provides a high-performance, memory-efficient specialized class `LlamaEmbedding` for generating text embeddings and calculating reranking scores.

### Key Features:
* **Streaming Batch Processing:** Process massive datasets (e.g., Hundreds of documents) without running out of memory (OOM).
* **Native Reranking:** Built-in support for Cross-Encoder models (outputting relevance scores instead of vectors).
* **Optimized Performance:** Utilizes Unified KV Cache for parallel encoding of multiple documents.
* **Chat Template Support:** Support for rerank templates has been introduced (via `llama_model_chat_template(model, b"rerank")`), which can automatically populate the query and document into a specific format.

### Support Embeddings & Rerank Model:


|  Model             |  Type     |  Link                                                  |  Status      |
|--------------------|-----------|--------------------------------------------------------|--------------|
|      `bge-m3`      | Embedding |[bge-m3-GGUF](https://huggingface.co/gpustack/bge-m3-GGUF)             |  Useful ✅  |
|`bge-reranker-v2-m3`|   Rerank  |[bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF) |  Useful ✅  |
|`qwen3-reranker`|   Rerank  |[Qwen3-Reranker-GGUF](https://huggingface.co/JamePeng2023/Qwen3-Reranker-GGUF) |  Useful ✅  |

#### TODO(JamePeng): Needs more extensive testing with various embedding and rerank models. :)

### 1. Text Embeddings (Vector Search)

To generate embeddings, use the `LlamaEmbedding` class. It automatically configures the model for vector generation.

```python
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(model_path="path/to/bge-m3.gguf", n_gpu_layers=-1, pooling_type=LLAMA_POOLING_TYPE_NONE)

# 1. Simple usage (OpenAI-compatible format)
response = llm.create_embedding("Hello, world!")
print(response['data'][0]['embedding'])

# 2. Batch processing (High Performance)
# You can pass a large list of strings; the streaming batcher handles memory automatically.
documents = ["Hello, world!", "Goodbye, world!", "Llama is cute."] * 100
embeddings = llm.embed(documents) # Returns a list of lists (vectors)

print(f"Generated {len(embeddings)} vectors.")
```

**Advanced Output Formats:**
You can request raw arrays or cosine similarity matrices directly:

```python
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(model_path="path/to/bge-m3.gguf", n_gpu_layers=-1, pooling_type=LLAMA_POOLING_TYPE_NONE)

# Returns raw List[float] instead of a dictionary wrapper
vector = llm.create_embedding("Text", output_format="array")

# Returns a similarity matrix (A @ A.T) in the response
# Note: Requires numpy installed
response = llm.create_embedding(
    ["apple", "fruit", "car"],
    output_format="json+"
)
print(response["cosineSimilarity"])
```

### 2. Reranking (Cross-Encoder Scoring)

Reranking models (like `bge-reranker`) take a **Query** and a list of **Documents** as input and output a relevance score (scalar) for each document.

> **Important:** You must explicitly set `pooling_type` to `LLAMA_POOLING_TYPE_RANK` (4) when initializing the model.

```python
import llama_cpp
from llama_cpp.llama_embedding import LlamaEmbedding

# Initialize a Reranking model
ranker = LlamaEmbedding(
    model_path="path/to/qwen3-reranker-0.6b-q8_0.gguf",
    pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,  # Crucial for Rerankers!
    n_gpu_layers=-1,
    n_ctx=0
)

query = "What causes Rain?"
docs = [
    "Clouds are made of water droplets...", # Relevant
    "To bake a cake you need flour...",     # Irrelevant
    "Rain is liquid water in the form of droplets..." # Highly Relevant
]

# Calculate relevance scores
# Logic: Constructs inputs like "[BOS] query [SEP] doc [EOS]" automatically
scores = ranker.rank(query, docs)

# Result: List of floats (higher means more relevant)
print(scores) 
# e.g., [0.0011407170677557588, 5.614783731289208e-05, 0.7173627614974976] -> The 3rd doc is the best match
```

### 3. Normalization

The `embed` method supports various mathematical normalization strategies via the `normalize` parameter.

| Normalization modes | $Integer$ | Description         | Formula |
|---------------------|-----------|---------------------|---------|
| NORM_MODE_NONE | $-1$      | none                |
| NORM_MODE_MAX_INT16 | $0$       | max absolute int16  | $\Large{{32760 * x_i} \over\max \lvert x_i\rvert}$
| NORM_MODE_TAXICAB | $1$       | taxicab             | $\Large{x_i \over\sum \lvert x_i\rvert}$
| NORM_MODE_EUCLIDEAN | $2$       | euclidean (default) | $\Large{x_i \over\sqrt{\sum x_i^2}}$
| NORM_MODE_PNORM | $>2$      | p-norm              | $\Large{x_i \over\sqrt[p]{\sum \lvert x_i\rvert^p}}$

This is useful for optimizing storage or preparing vectors for cosine similarity search (which requires L2 normalization).

```python
from llama_cpp.llama_embedding import (
  LLAMA_POOLING_TYPE_NONE,
  NORM_MODE_MAX_INT16,
  NORM_MODE_TAXICAB,
  NORM_MODE_EUCLIDEAN
)

# Initialize the model (automatically sets embeddings=True)
llm = LlamaEmbedding(model_path="path/to/bge-m3.gguf", n_gpu_layers=-1, pooling_type=LLAMA_POOLING_TYPE_NONE)

# Taxicab (L1)
vec_l1 = llm.embed("text", normalize=NORM_MODE_TAXICAB)

# Default is Euclidean (L2) - Standard for vector databases
vec_l2 = llm.embed("text", normalize=NORM_MODE_EUCLIDEAN)

# Max Absolute Int16 - Useful for quantization/compression
vec_int16 = llm.embed("text", normalize=NORM_MODE_MAX_INT16)

# Raw Output (No Normalization) - Get the raw floating point values from the model
embeddings_raw = llm.embed(["search query", "document text"], normalize=NORM_MODE_NONE)
```

### Legacy Usage (Deprecated)

The standard `Llama` class still supports basic embedding generation, but it lacks the memory optimizations and reranking capabilities of `LlamaEmbedding`.

```python
# Old method - Not recommended for large batches or reranking
llm = llama_cpp.Llama(model_path="...", embeddings=True)
emb = llm.create_embedding("text")
```

---

### Speculative Decoding

`llama-cpp-python` supports speculative decoding which allows the model to generate completions based on a draft model.

The fastest way to use speculative decoding is through the `LlamaNGramMapDecoding`(**Recommend**) or `LlamaPromptLookupDecoding` class.

Just pass this as a draft model to the `Llama` class during initialization.

```python
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaNGramMapDecoding

llama = Llama(
    model_path="path/to/qwen-3.6-27b.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    draft_model=LlamaNGramMapDecoding(
        ngram_size=3,
        num_pred_tokens=10
    )
)

response = llama.create_chat_completion(
    messages=[{"role": "user", "content": "Write a python script..."}]
)
```
Note: `LlamaPromptLookupDecoding.num_pred_tokens` is the number of tokens to predict 10 is the default and generally good for gpu, 2 performs better for cpu-only machines. Now, `LlamaNGramMapDecoding` with the new Hash Map algorithm, draft generation becomes instantaneous $O(1)$, and the time consumption is almost 0 regardless of whether you set the prediction to 2 or 10 words.

### Adjusting the Context Window

The context window of the Llama models determines the maximum number of tokens that can be processed at once. By default, this is set to 512 tokens, but can be adjusted based on your requirements.

For instance, if you want to work with larger contexts, you can expand the context window by setting the n_ctx parameter when initializing the Llama object:

```python
llm = Llama(model_path="./models/llama-model.gguf", n_ctx=2048)
```

## Docker image

See here: https://github.com/JamePeng/llama-cpp-python/tree/main/docker#cuda_simple

## OpenAI Compatible Web Server (Deprecated)

`llama-cpp-python` offers a web server which aims to act as a drop-in replacement for the OpenAI API.
This allows you to use llama.cpp compatible models with any OpenAI compatible client (language libraries, services, etc).

To install the server package and get started:

```bash
pip install 'llama-cpp-python[server]'
python3 -m llama_cpp.server --model models/7B/llama-model.gguf
```

Similar to Hardware Acceleration section above, you can also install with GPU (cuBLAS) support like this:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install 'llama-cpp-python[server]'
python3 -m llama_cpp.server --model models/7B/llama-model.gguf --n_gpu_layers 35
```

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to see the OpenAPI documentation.

To bind to `0.0.0.0` to enable remote connections, use `python3 -m llama_cpp.server --host 0.0.0.0`.
Similarly, to change the port (default is 8000), use `--port`.

You probably also want to set the prompt format. For chatml, use

```bash
python3 -m llama_cpp.server --model models/7B/llama-model.gguf --chat_format chatml
```

That will format the prompt according to how model expects it. You can find the prompt format in the model card.
For possible options, see [llama_cpp/llama_chat_format.py](llama_cpp/llama_chat_format.py) and look for lines starting with "@register_chat_format".

If you have `huggingface-hub` installed, you can also use the `--hf_model_repo_id` flag to load a model from the Hugging Face Hub.

```bash
python3 -m llama_cpp.server --hf_model_repo_id Qwen/Qwen2-0.5B-Instruct-GGUF --model '*q8_0.gguf'
```

### Web Server Features

- [Local Copilot replacement](https://llama-cpp-python.readthedocs.io/en/latest/server/#code-completion)
- [Function Calling support](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling)
- [Vision API support](https://llama-cpp-python.readthedocs.io/en/latest/server/#multimodal-models)
- [Multiple Models](https://llama-cpp-python.readthedocs.io/en/latest/server/#configuration-and-multi-model-support)

## Low-level API

[API Reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#low-level-api)

The low-level API is a direct [`ctypes`](https://docs.python.org/3/library/ctypes.html) binding to the C API provided by `llama.cpp`.
The entire low-level API can be found in [llama_cpp/llama_cpp.py](https://github.com/abetlen/llama-cpp-python/blob/master/llama_cpp/llama_cpp.py) and directly mirrors the C API in [llama.h](https://github.com/ggerganov/llama.cpp/blob/master/llama.h).

Below is a short example demonstrating how to use the low-level API to tokenize a prompt:

```python
import llama_cpp
import ctypes
llama_cpp.llama_backend_init(False) # Must be called once at the start of each program
params = llama_cpp.llama_context_default_params()
# use bytes for char * params
model = llama_cpp.llama_load_model_from_file(b"./models/7b/llama-model.gguf", params)
ctx = llama_cpp.llama_new_context_with_model(model, params)
max_tokens = params.n_ctx
# use ctypes arrays for array params
tokens = (llama_cpp.llama_token * int(max_tokens))()
n_tokens = llama_cpp.llama_tokenize(ctx, b"Q: Name the planets in the solar system? A: ", tokens, max_tokens, llama_cpp.c_bool(True))
llama_cpp.llama_free(ctx)
```

Check out the [examples folder](examples/low_level_api) for more examples of using the low-level API.

## Documentation

Documentation is available via [https://llama-cpp-python.readthedocs.io/](https://llama-cpp-python.readthedocs.io/).
If you find any issues with the documentation, please open an issue or submit a PR.

## Development

This package is under active development and I welcome any contributions.

To get started, clone the repository and install the package in editable / development mode:

```bash
git clone https://github.com/JamePeng/llama-cpp-python --recursive
cd llama-cpp-python

# Upgrade pip (required for editable mode)
pip install --upgrade pip

# Install with pip
pip install -e .

# if you want to use the fastapi / openapi server
pip install -e '.[server]'

# to install all optional dependencies
pip install -e '.[all]'

# to clear the local build cache
make clean
```

Now try running the tests

```bash
pytest
```

There's a `Makefile` available with useful targets.
A typical workflow would look like this:

```bash
make build
make test
```

You can also test out specific commits of `llama.cpp` by checking out the desired commit in the `vendor/llama.cpp` submodule and then running `make clean` and `pip install -e .` again. Any changes in the `llama.h` API will require
changes to the `llama_cpp/llama_cpp.py` file to match the new API (additional changes may be required elsewhere).

## FAQ

### Are there pre-built binaries / binary wheels available?

The recommended installation method is to install from source as described above.
The reason for this is that `llama.cpp` is built with compiler optimizations that are specific to your system.
Using pre-built binaries would require disabling these optimizations or supporting a large number of pre-built binaries for each platform.

That being said there are some pre-built binaries available through the Releases as well as some community provided wheels.

In the future, I would like to provide pre-built binaries and wheels for common platforms and I'm happy to accept any useful contributions in this area.

### How does this compare to other Python bindings of `llama.cpp`?

I originally wrote this package for my own use with two goals in mind:

- Provide a simple process to install `llama.cpp` and access the full C API in `llama.h`and `mtmd.h` from Python

- Provide a high-level Python API that can be used as a drop-in replacement for the OpenAI API so existing apps can be easily ported to use `llama.cpp`

- Provide a high-throughput, relatively low-latency Python library by continuously optimizing (reducing unnecessary CPU processing or algorithm tuning) and accepting feedback (issues or pull requests), making loading and running GGUF files via Python simpler and more controllable.

- Provides clearer code comments and error code analysis feedback in llama.cpp, based on common usage feedback and code execution flow, to help more users who are learning LLM through this project understand the project's operation and subsequent feedback optimization.

### OSError: libcudart.so.XX/cudart64_XX.dll: cannot open shared object file: No such file or directory
This error is primarily caused by the following reasons:

- Missing Installation or Configuration: The CUDA Toolkit is either not installed, or the environment variables were not correctly configured after installation, preventing the system from locating the required dynamic link libraries. You can try running `nvidia-smi` or `nvcc` in your terminal to check if they output results correctly.

- Version Mismatch: The CUDA Toolkit environment is installed and configured, but it does not match the CUDA version of the pre-compiled llama-cpp-python wheel you are using. For example, your local environment might be running CUDA 12.1, but you installed a version compiled for CUDA 12.6.

- Recommendation (Build from Source): It is recommended to fully configure your local CUDA Toolkit environment (ensuring the PATH for dynamic libraries is set and the nvcc compiler is recognized). Then, clone the code and compile it locally. Remember to enable the -DGGML_CUDA=on CMake option during compilation. This ensures the installation achieves maximum compatibility with your local system.

### FileNotFoundError: Could not find module (like ggml.dll, ggml-cpu.dll, ggml-cuda.dll)

**Step 1:** Locate the `lib` folder of the `llama-cpp-python` library within your current Python runtime environment: `Python3XX\Lib\site-packages\llama_cpp\lib\`

**Step 2:** Verify that the missing DLL mentioned in the error is correctly present in this directory. Developers often have multiple Python environments locally, or projects like ComfyUI may use embedded virtual environments. Please ensure that you are installing the library and running the code in the exact same environment.

This error is primarily caused by the following reasons:

1. **Environment Mismatch:** The Python environment used for installation is different from the one being used for execution.

2. **Instruction Set Incompatibility:** Regarding `ggml.dll` and `ggml-cpu.dll`, the instruction sets (such as AVX) supported by the pre-compiled version may be incompatible with your local processor. (This typically manifests as `OSError: [WinError -1073741795] Windows Error 0xc000001d` after execution).

3. **CUDA Version Mismatch:** Regarding `ggml-cuda.dll`, the CUDA version of the pre-compiled library does not match your local CUDA Toolkit version (e.g., a mismatch between CUDA 12.X and CUDA 13.X). It is recommended to fully configure your local CUDA Toolkit environment (ensuring the PATH for dynamic libraries is set and the nvcc compiler is recognized). Then, clone the code and compile it locally.

### Why are libraries compiled by other authors only around 100MB, while your pre-compiled versions range from 300MB to 900MB?

My GitHub Actions script is configured to compile against **all supported CUDA compute architectures** for each specific CUDA version I maintain.

For example:

* **CUDA 13.0.2:** Currently supports architectures from SM75 (Turing) up to SM120a (Blackwell).
* **CUDA 12.4.1 and 12.6.3:** Support older architectures as well, such as SM70.
* *(Note: The Windows versions are built to support every architecture compatible with the respective CUDA version).*

The reason libraries from other authors are smaller is that they often **only compile for a single architecture** (e.g., targeting only the RTX 30 series [SM86] or the RTX 40 series [SM89]). To maximize convenience, I provide an **integrated compilation** covering a wide range of hardware; you simply need to select the CUDA version that matches your environment to load and run it.

### Quick tips for develop/user (continuously updated):

* 1. I've determined that `llama_cpp.server` is currently in a semi-deprecated state (meaning it won't be maintained unless absolutely necessary, and I might even consider deleting or separating it to reduce the library size). I highly recommend using the `llama-server` program maintained by the upstream `llama.cpp` project, which offers a lower-level implementation, more frequent maintenance and optimization, and more reliable API calls.

* 2. Regarding AMD and Intel graphics cards, AMD can certainly use ROCm as the primary backend (but the drawback is that it's basically only stable on Linux platforms), and Intel's Sycl will also encounter some compilation difficulties. I consistently recommend using the Vulkan backend for these two types of graphics cards for greater efficiency and stability, because the upstream `llama.cpp` Vulkan backend is actively maintained by many developers, generally allowing you to enjoy new feature optimizations and bug fixes earlier and faster.

* 3. If you are using hybrid multimodal model for building ComfyUI nodes or running single-turn API wrappers where you do not need multi-turn state rollbacks, simply initialize your Llama instance with `ctx_checkpoints=0`:

        ```python
        llm = Llama(
            model_path="./Qwen3.5-VL-9B.gguf",
            chat_handler=MTMDChatHandler(clip_model_path="./mmproj.gguf"),
            n_ctx=4096,
            ctx_checkpoints=0  # <-- SET THIS TO 0 TO ENABLE ZERO-LATENCY FAST PATH
        )
        ```


### Any suggestions, contributions, and modifications to this package will be directed toward building a user-friendly, efficient, and secure Python library.

## License

This project is licensed under the terms of the MIT license.
