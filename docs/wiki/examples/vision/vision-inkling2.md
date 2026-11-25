# Inkling 2 Mini image chat with DSpark

Use a native build with Inkling 2 MTMD and DSpark support. Put the three matching
GGUF files in `Inkling-2-Mini/` and provide a local `picture.png`.

| File | Role |
|---|---|
| `Inkling-2-Mini-AD-Q4_K.gguf` | Target model that reads the prompt and verifies draft tokens |
| `mmproj-Inkling-2-Mini-BF16.gguf` | Vision projector that encodes the image |
| `dspark-Inkling-2-Mini-Q8_0.gguf` | Draft model with the Inkling 2 Markov head |

## Generate an answer

`GenericMTMDChatHandler` reads the GGUF chat template and passes image batches
to the target model. DSpark uses the resulting layer features to prepare its
draft context.

```python
import base64
from contextlib import closing
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_multimodal import GenericMTMDChatHandler
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

model_dir = Path("Inkling-2-Mini")
image = base64.b64encode(Path("picture.png").read_bytes()).decode("ascii")
speculative = SpecConfig(
    spec_type=SpeculativeType.DRAFT_DSPARK,
    draft_model_path=str(model_dir / "dspark-Inkling-2-Mini-Q8_0.gguf"),
    draft_n_max=4,
    draft_n_gpu_layers="all",
)

with GenericMTMDChatHandler(
    chat_format=None,
    mmproj_path=str(model_dir / "mmproj-Inkling-2-Mini-BF16.gguf"),
    use_gpu=True,
    extra_template_arguments={"reasoning_effort": "none"},
    verbose=False,
) as handler, closing(Llama(
    model_path=str(model_dir / "Inkling-2-Mini-AD-Q4_K.gguf"),
    chat_handler=handler,
    speculative=speculative,
    n_ctx=2048,
    n_batch=32,
    n_ubatch=32,
    n_gpu_layers=-1,
    verbose=False,
)) as llm:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + image,
            }},
            {"type": "text", "text": (
                "What is the dominant color of the image? "
                "Describe it in one short sentence."
            )},
        ]}],
        max_tokens=128,
        temperature=0,
        seed=42,
    )
    print(response["choices"][0]["message"]["content"])
    stats = llm.last_speculative_stats
    print(
        f"Drafted {stats['drafted']}, verified {stats['verified']}, "
        f"accepted {stats['accepted_draft_tokens']} draft tokens"
    )
```

The example configures GPU placement for the target, draft, and projector.
The complete image and text prompt must fit within `n_ctx`.
`message.content` can include model reasoning followed by the final answer.

## Verify draft activity

Check that `drafted` and `verified` are positive. To compare feature transfer
paths, set `LLAMA_DFLASH_DEV=0` before creating `Llama` to use host copies.
Set it to `1` to allow device transfer on a compatible backend.

In a CUDA test with separate 256 × 256 red and blue images, `n_batch=32` split
each image across multiple media batches. Both transfer paths returned the same
answers and counts. The target verified 96 draft tokens per image and accepted
49 for red and 51 for blue. These counts describe that test run.

For details on the media callback and draft positions, see
[Speculative Decoding](../../modules/LlamaSpeculative.md#native-draft-positions-and-image-batches).
