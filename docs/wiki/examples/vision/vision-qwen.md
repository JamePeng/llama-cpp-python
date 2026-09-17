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
MTP and DFlash-family engines are rejected by this chat handler before prefill,
even if the GGUF contains MTP weights or a lower-level draft API accepts embeddings.

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
