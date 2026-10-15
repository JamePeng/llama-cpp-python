# Qwen3.5 image chat

Use `Qwen35ChatHandler` with a Qwen3.5 text GGUF and its matching vision projector.
The example uses `Qwen3.5-0.8B-Q4_K_M.gguf` and
`mmproj-Qwen3.5-0.8b-Q8_0.gguf` from
[JamePeng2023/Qwen3.5-0.8B-GGUF](https://huggingface.co/JamePeng2023/Qwen3.5-0.8B-GGUF).
Download both files into the working directory and provide a local `picture.png`.
The native build must include MTMD support for this model.

## Generate an answer

```python
import base64
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_multimodal import Qwen35ChatHandler

image = base64.b64encode(Path("picture.png").read_bytes()).decode("ascii")
handler = Qwen35ChatHandler(
    mmproj_path="./mmproj-Qwen3.5-0.8b-Q8_0.gguf",
    image_max_tokens=256,
    enable_thinking=False,
    use_gpu=False,
    verbose=False,
)
llm = None
try:
    llm = Llama(
        model_path="./Qwen3.5-0.8B-Q4_K_M.gguf",
        chat_handler=handler,
        n_ctx=2048,
        n_batch=256,
        n_gpu_layers=0,
        ctx_checkpoints=4,
        verbose=False,
    )
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + image,
            }},
            {"type": "text", "text": "Describe this image briefly."},
        ]}],
        max_tokens=64,
        temperature=0,
    )
    print(response["choices"][0]["message"]["content"])
finally:
    if llm is not None:
        llm.close()
    handler.close()
```

The example uses CPU execution. For an appropriate GPU build, configure target
layer offload and the handler's `use_gpu` separately. Output depends on the image
and model; generation is not a validation of image recognition accuracy.

## Use an RPC-backed language model

Start a compatible RPC server as shown in the
[RPC model offloading guide](../../features/rpc.md), then add
`rpc_servers=["127.0.0.1:50052"]`, `rpc_local_devices=[]`, and
`n_gpu_layers=-1` to the `Llama(...)` call above. Keep `use_gpu=False` on this
handler if the projector should run on the local CPU. The RPC device list
selects model-layer placement only; it does not select the projector backend.
The [RPC guide's multimodal example](../../features/rpc.md#multimodal-input)
also shows a matching Qwen3.5 model and projector with the generic MTMD handler.

## N-gram speculation

To enable n-gram proposals, add these imports and pass the configuration to the
`Llama` constructor above:

```python
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

speculative = SpecConfig(spec_type=SpeculativeType.NGRAM_MAP_K)
# Add speculative=speculative to Llama(...); keep ctx_checkpoints positive.
```

`NGRAM_MAP_K4V` is also supported by the prefilled-media path. MTMD first evaluates
the media; n-gram proposals never emit its negative ledger IDs as decoder tokens.
DFlash-family engines are rejected by this chat handler before prefill.

## MTP image chat

For a Qwen3.5 GGUF containing MTP heads, pass the following configuration to
`Llama(...)` in the image-chat example above:

```python
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

speculative = SpecConfig(spec_type=SpeculativeType.DRAFT_MTP, draft_n_max=3)
# Add speculative=speculative to Llama(...).
```

The runnable [MTP image-chat example](../../../../examples/high_level_api/high_level_api_mtp_image_chat.py)
accepts local model, projector and image paths. For the Qwen3.5-9B-MTP Q8 target
and BF16 projector used in integration testing, run from the repository root:

```powershell
python -m examples.high_level_api.high_level_api_mtp_image_chat `
  --model "D:\git_project\Qwen3.5-9B-MTP\Qwen3.5-9B-MTP-Q8_0.gguf" `
  --mmproj "D:\git_project\Qwen3.5-9B-MTP\mmproj-Qwen3.5-9B-MTP-BF16.gguf" `
  --image ".\picture.png" `
  --prompt "Describe this image briefly."
```

The script defaults to GPU execution for both the target and projector, with
`n_ctx=8192` and `max_tokens=4096`. Use `--n-gpu-layers 0 --no-projector-gpu`
for CPU execution. Use `--draft-tokens`, `--draft-min` and `--draft-p-min` to
adjust proposal limits and the confidence threshold. The script prints MTP and
runtime parameters, the answer, draft acceptance, verification steps and rollbacks.

MTP is enabled by default. Repeat the command with `--no-mtp` for ordinary
decoding, keeping the image, prompt, GPU and token-budget arguments identical.
Use `--mtp` to enable it explicitly. Both modes report decode speed, sustained
speed, decode TTFT and request wall time; MTP also reports proposal statistics.

Both modes time active calls to the token generator with the same clock,
excluding media prefill, model loading, caller processing and final cleanup.
This includes sampler setup after prefill, and the token count includes yielded
end-of-generation tokens. Sustained decode tokens/s also excludes the first
token; decode TTFT starts after prefill. MTP phase timings cover drafting, target
decode/synchronization, hidden-state processing and acceptance. Request wall time
includes media prefill and response assembly, but excludes model loading.

MTMD forwards each decoded media sub-batch to MTP, preserving image positions
and pairing embeddings with right-shifted target hidden states. The handler
rebuilds target and draft state together for each request. Target-only prefix
checkpoints are not reused. The complete prompt must fit `--n-ctx` (default
8192), and media batches must fit without context shifting; reduce
`--image-max-tokens` or increase `--n-ctx` if necessary. High-level image MTP was
validated with Qwen3.5-9B-MTP and its matching projector.

## Reuse and failure recovery

The handler checks native sequence removal before changing its Python ledger,
uses the media helper's returned native position, and passes the final valid
output to generation without replaying media IDs as text tokens. Media token
counts must not be substituted for native positions in custom integrations.

If evaluation fails after prefill starts, the handler resets uncertain state,
releases temporary media resources, and propagates the exception. Submit the full
messages again on retry. This includes `KeyboardInterrupt` during media prefill;
interrupts later in the text generation loop follow the normal
[abort contract](../../core/Llama.md#abort).

Use only one active request per instance. Close a streaming response before
reusing the instance. See [caching](../../features/caching.md) for checkpoint
lifetimes and [speculative limits](../../modules/LlamaSpeculative.md#native-draft-positions-and-image-batches)
for position constraints.
