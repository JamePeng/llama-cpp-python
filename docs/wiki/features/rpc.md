---
title: RPC Model Offloading
feature_name: RPC Model Offloading
source_files:
  - llama_cpp/llama.py
  - llama_cpp/_rpc.py
  - llama_cpp/llama_multimodal.py
  - vendor/llama.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp
last_updated: 2026-09-30
version_target: "latest"
---

# RPC Model Offloading

Use `Llama(rpc_servers=...)` to place model layers on devices exposed by a
compatible `ggml-rpc-server`. The GGUF file is read on the Python host; the
RPC server supplies compute and device memory. Both the Python build and the
server need RPC support from compatible llama.cpp revisions.

## Start the server

Build the Python package with `GGML_RPC=ON` as described in the
[installation guide](../install.md#rpc-build-option). Download a prebuilt
server from the official [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases),
choosing an archive for the server host's operating system, architecture, and
backend (such as CUDA or Vulkan). Keep `ggml-rpc-server` with the matching
ggml backend libraries and required runtime libraries from that release.
Choose a release compatible with the llama.cpp revision in the Python package.
On Windows PowerShell, run this from the extracted binary directory:

```powershell
.\ggml-rpc-server.exe --host 127.0.0.1 --port 50052 --device CUDA0
```

The server lists available device names at startup. Omit `--device` to let it
select accelerators, or choose one or more names it recognizes. When Python
runs on another machine, bind the server to an address reachable from that
machine and use that address in `rpc_servers`. The RPC protocol has no
authentication or encryption; use it only on a trusted network.

## Load a model on remote devices

Place a compatible GGUF on the Python host and adjust its path and the server
address below:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    rpc_servers=["127.0.0.1:50052"],
    rpc_local_devices=[],
    n_gpu_layers=-1,
    n_ctx=2048,
)
try:
    response = llm("The capital of France is", max_tokens=16)
    print(response["choices"][0]["text"])
finally:
    llm.close()
```

`rpc_local_devices=[]` excludes local GPUs from model-layer selection. Omit
this argument to include the default local GPUs, or pass names such as
`["CUDA0"]` to select and order them explicitly. The selected device order is
all devices from each RPC endpoint, in `rpc_servers` order, followed by the
selected local GPUs. `main_gpu` uses this order for `LLAMA_SPLIT_MODE_NONE`;
when `tensor_split` is supplied, provide one value per selected device.
`rpc_servers` also accepts a comma-separated string. Row split mode is not
supported with RPC.

## Multimodal input

The language model and multimodal projector have separate device selection.
For a Qwen3.5 GGUF and its matching `mmproj`, the generic MTMD handler can
process an image while the language model uses RPC:

```python
import base64
from pathlib import Path

from llama_cpp import Llama

image = base64.b64encode(Path("./image.png").read_bytes()).decode("ascii")
image_url = f"data:image/png;base64,{image}"

llm = Llama(
    model_path="./Qwen3.5-9B-MTP-Q8_0.gguf",
    mmproj_path="./mmproj-Qwen3.5-9B-MTP-BF16.gguf",
    rpc_servers=["127.0.0.1:50052"],
    rpc_local_devices=[],
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=256,
    chat_handler_kwargs={
        "extra_template_arguments": {"enable_thinking": False},
    },
)
try:
    response = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "What is in this image?"},
            ],
        }],
        max_tokens=64,
    )
    print(response["choices"][0]["message"]["content"])
finally:
    llm.close()
```

The projector initializes on the first multimodal request, so constructing
`Llama` alone does not verify that the `mmproj` loaded. The current generic
handler leaves its MTMD device unspecified. With `use_gpu=True` (the default),
MTMD selects the first GPU in the process-wide ggml registry, which may be a
local GPU even when `rpc_local_devices=[]`. Set
`chat_handler_kwargs={"use_gpu": False}` to run the projector on the local CPU.
The high-level handler does not currently expose explicit RPC device
selection for the projector. See the [Qwen image guide](../examples/vision/vision-qwen.md)
for a dedicated handler and more image-input details.

## Registration and failures

`Llama` loads dynamic ggml backends before registering RPC endpoints. It
probes each server's protocol and device count before native registration. An
unavailable or incompatible server fails model construction; Python does not
silently switch the model to local GPUs. A disconnect between the probe and
native registration can still make ggml terminate the process.

Registrations belong to ggml's process-wide backend registry and are reused
for repeated endpoints. They remain until Python exits because models may
hold pointers to their devices. Each `Llama` instance passes an explicit
device list so previously registered RPC servers do not become part of a
different model's device selection. The adapter allows at most 16 distinct
RPC endpoints per process; restart Python to use more distinct endpoints.
