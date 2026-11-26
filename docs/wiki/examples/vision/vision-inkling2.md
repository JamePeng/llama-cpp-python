# Inkling 2 Mini image chat with DFlash2 or DSpark

Place the target model and vision projector in `Inkling-2-Mini/` and provide a
local `picture.png`. For speculative decoding, add the draft GGUF for your chosen
mode and use a native build that supports that Inkling 2 draft model.

| File | Role |
|---|---|
| `Inkling-2-Mini-AD-Q4_K.gguf` | Target model that reads the prompt and verifies draft tokens |
| `mmproj-Inkling-2-Mini-BF16.gguf` | Vision projector that encodes the image |
| `Inkling-2-Mini-DFlash2-Q8_0.gguf` | DFlash2 draft model with a token selector |
| `dspark-Inkling-2-Mini-Q8_0.gguf` | Draft model with the Inkling 2 Markov head |

## Generate an answer

`GenericMTMDChatHandler` reads the GGUF chat template and passes image batches
to the target model. Both draft models use the resulting layer features. Set
`mode` to `"dflash2"`, `"dspark"`, or `"off"` to run the same image prompt.

```python
import base64
from contextlib import closing
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_multimodal import GenericMTMDChatHandler
from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

model_dir = Path("Inkling-2-Mini")
image = base64.b64encode(Path("picture.png").read_bytes()).decode("ascii")
mode = "dflash2"
drafts = {
    "dflash2": (SpeculativeType.DRAFT_DFLASH2, "Inkling-2-Mini-DFlash2-Q8_0.gguf"),
    "dspark": (SpeculativeType.DRAFT_DSPARK, "dspark-Inkling-2-Mini-Q8_0.gguf"),
}
speculative = None
if mode != "off":
    draft_type, draft_file = drafts[mode]
    speculative = SpecConfig(
        spec_type=draft_type,
        draft_model_path=str(model_dir / draft_file),
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
    if speculative is not None:
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

In a DSpark CUDA test with separate 256 × 256 red and blue images, `n_batch=32` split
each image across multiple media batches. Both transfer paths returned the same
answers and counts. The target verified 96 draft tokens per image and accepted
49 for red and 51 for blue. These counts describe that test run.

For details on the media callback and draft positions, see
[Speculative Decoding](../../modules/LlamaSpeculative.md#native-draft-positions-and-image-batches).

## End-to-end latency on an RTX 5090

Across one text completion and two image requests, DFlash2 reduced elapsed time
by **38%** and DSpark by **32%**. All three modes returned identical text for
each input. Times are in seconds. Speedups are relative to speculative decoding off.

| Request | Output tokens | Off | DFlash2 | DSpark |
|---|---:|---:|---:|---:|
| Number sequence | 48 | 0.555 s | 0.244 s (2.28×) | 0.251 s (2.22×) |
| Red image | 71 | 1.075 s | 0.720 s (1.49×) | 0.798 s (1.35×) |
| Blue image | 81 | 1.166 s | 0.793 s (1.47×) | 0.850 s (1.37×) |
| **Three requests combined** | **200** | **2.782 s** | **1.726 s (1.61×)** | **1.885 s (1.48×)** |

With the same image prompts capped at **32 output tokens**, the red image took
0.693 s with speculation off, 0.542 s with DFlash2 (1.28×), and 0.530 s with
DSpark (1.31×). The blue image took 0.683 s, 0.552 s (1.24×), and 0.518 s
(1.32×), respectively.

### End-to-end output throughput

Throughput is the number of completion tokens divided by the full request time,
including prompt prefill and, for images, image decoding and vision encoding.
The table gives the median rate over three measured rounds. The combined rate
uses the 200 output tokens from the number sequence and the two full image
answers divided by their total elapsed time in each round.

| Request | Off (tok/s) | DFlash2 (tok/s) | DSpark (tok/s) |
|---|---:|---:|---:|
| Number sequence | 86.5 | 196.8 | 191.5 |
| Red image | 66.0 | 98.6 | 89.0 |
| Blue image | 69.5 | 102.1 | 95.2 |
| **Three requests combined** | **71.9** | **115.9** | **106.1** |
| Red image (32-token limit) | 46.2 | 59.0 | 60.3 |
| Blue image (32-token limit) | 46.9 | 58.0 | 61.8 |

### Benchmark conditions

- **Setup.** Windows, one RTX 5090, llama.cpp revision `98cc730cc`, and the GGUFs
  listed above. Both draft modes used `LLAMA_DFLASH_DEV=1` and `draft_n_max=4`.
  GPU placement, context size, batch sizes, temperature, and seed matched the example.
- **Inputs.** The text case used `create_completion` with
  `Count from one to twenty: 1, 2, 3,` and a 48-token limit. The image cases used
  separate solid red and blue 256 × 256 PNGs with the example's prompt and
  128-token limit. Output counts include any reasoning returned by the model.
- **Timing.** Each mode ran a warm-up before three measured rounds. The model
  context was reset before every request. Timing started with the non-streaming
  Python API call and ended when the full response returned, including image
  decoding and vision encoding for image requests. Model loading and PNG/base64
  preparation took place before timing. Each row reports the median of three
  rounds. For the combined row, the three request times were summed within each
  round before taking the median.
