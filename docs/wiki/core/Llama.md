```yaml
---
title: Llama Class
module_name: llama_cpp.llama
source_file: llama_cpp/llama.py
class_name: Llama
last_updated: 2026-05-06
version_target: "latest"
---
```

## Overview
The `Llama` class is the core, high-level Python wrapper for a `llama.cpp` model. It handles model loading, memory management (KV cache), tokenization, and generation (both base text completion and chat formatting). It includes advanced features like dynamic LoRA routing, dual-mode hybrid/recurrent checkpointing, speculative decoding, and context shifting.

## Constructor (`__init__`)

Initialize the model and context. Note that model loading will immediately allocate RAM/VRAM based on the selected offloading parameters.

### Core Model & Hardware Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str` | **Required** | Model file path (GGUF format) |
| `n_gpu_layers` | `Union[int, Literal["auto", "all"]]` | `"auto"` | Number of model layers stored in VRAM:<br>• `auto`/`-1`: auto-selected by llama.cpp<br>• `all`/`-2`: all layers<br>• integer N: first N layers<br>• `0`: disable layer offload |
| `cpu_moe` | `bool` | `False` | Whether to keep all MoE weights on CPU |
| `n_cpu_moe` | `int` | `0` | Number of first N MoE layers to keep on CPU (compatible with `cpu_moe`) |
| `split_mode` | `int` | `LLAMA_SPLIT_MODE_LAYER` | Model GPU split mode:<br>• `LLAMA_SPLIT_MODE_NONE`: single GPU<br>• `LLAMA_SPLIT_MODE_ROW`: row-level split<br>• `LLAMA_SPLIT_MODE_LAYER`: layer-level split |
| `main_gpu` | `int` | `0` | The primary GPU to use for intermediate results or the entire model. |
| `tensor_split` | `List[float]` | `None` | Proportional split of tensors across GPUs (max `LLAMA_MAX_DEVICES`). |
| `use_mmap` | `bool` | `True` | Whether to use memory mapping (mmap) if possible. |
| `use_mlock` | `bool` | `False` | Force the system to keep the model in RAM, preventing swapping. |
| `kv_overrides` | `Dict` | `None` | Key-value overrides for the model metadata (supports bool, int, float, str). |
| `numa` | `Union[bool, int]`| `False` | NUMA strategy (e.g., `GGML_NUMA_STRATEGY_DISTRIBUTE`). |

### Context & Performance Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_ctx` | `int` | `512` | Text context size. Set to `0` to load from model metadata. |
| `n_batch` | `int` | `2048` | Maximum batch size for prompt processing. |
| `n_ubatch` | `int` | `512` | Physical batch size. |
| `n_threads` | `int` | `None` | Number of threads for generation (defaults to CPU count // 2). |
| `n_threads_batch`| `int` | `None` | Number of threads for batch processing (defaults to CPU count). |
| `flash_attn_type`| `int` | `AUTO` | Controls Flash Attention activation (`LLAMA_FLASH_ATTN_TYPE_AUTO`). |
| `swa_full` | `bool` | `None` | Whether to use full-size SWA cache |
| `kv_unified` | `bool` | `None` | Use single unified KV buffer for the KV cache of all sequences |
| `type_k` / `type_v`| `int` | `None` | KV cache data type for K and V (defaults to `f16`). |
| `offload_kqv` | `bool` | `True` | Whether to offload K, Q, V tensors to GPU. |

### Advanced & Chat Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `chat_format` | `str` | `None` | String specifying the chat template (e.g., `"llama-2"`, `"chatml"`). Guessed from GGUF if None. |
| `chat_handler` | `LlamaChatCompletionHandler` | `None` | Optional custom handler. See [[ChatHandlers]]. |
| `draft_model` | `LlamaDraftModel` | `None` | Optional draft model for speculative decoding. |
| `ctx_checkpoints` | `int` | `16` | Max hybrid/recurrent context checkpoints to keep. Set to `0` to disable checkpointing for single-turn fast paths. |
| `checkpoint_interval` | `int` | `4096` | Token interval for saving periodic Hybrid/Recurrent checkpoints during long prompt evaluation. |
| `checkpoint_on_device` | `bool` | `False` | Store Hybrid/Recurrent checkpoint tensor payloads in `llama_context`-owned device buffers via `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. Reduces device-to-host copy overhead, but only one active checkpoint per `seq_id` is safe. |

*(Note: There are numerous additional RoPE/YaRN scaling parameters available for specialized context extension. Refer to the source code for the full list).*

---

## Core Methods

### `create_chat_completion`
Generates a chat response using the configured `chat_format` or `chat_handler`.
```python
import llama_cpp

model = llama_cpp.Llama(model_path="models/qwen2.5-7b-instruct.gguf", n_gpu_layers=-1)

response = model.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain KV caching."}
    ],
    temperature=0.7,
    max_tokens=2048
)
print(response["choices"][0]["message"]["content"])
```

### `create_completion` / `__call__`
Generates standard text completion from a raw string prompt.
```python
import llama_cpp

model = llama_cpp.Llama(model_path="models/llama-3-8b.gguf")
output = model("The capital of Japan is", max_tokens=10, stop=["\n"])
print(output["choices"][0]["text"])
```

### `generate`
A low-level generator yielding token IDs one by one. Highly customizable with sampling parameters, dynamic LoRA mounting, and control vectors.
```python
import llama_cpp

model = llama_cpp.Llama(model_path="models/llama-3-8b.gguf")
tokens = model.tokenize(b"def fibonacci(n):")

for token in model.generate(tokens, top_k=40, top_p=0.95, temp=0.2):
    print(model.detokenize([token]).decode('utf-8'), end="", flush=True)
```

### `eval`
Low-level method to ingest and evaluate a sequence of tokens. Used internally to update the KV cache and logits. Handles **Context Shifting** automatically to prevent OOM when the token count exceeds `n_ctx`.
```python
# Evaluates a chunk of tokens and updates internal state
model.eval(tokens=[1, 453, 234, 987], active_loras=[{"name": "coding_adapter", "scale": 1.0}])
```

### `abort`
Immediately halts an active generation loop safely.
* **Usage**: Typically called from a separate monitoring thread (like a timer). When triggered, the running stream will exit and the final chunk will contain `"finish_reason": "abort"`.

### Dynamic LoRA Management
The `Llama` class allows you to load multiple LoRAs into VRAM and apply them dynamically per-generation or per-eval.
* `load_lora(name: str, path: str)`: Loads an adapter into VRAM (does not apply it yet).
* `unload_lora(name: str)`: Releases the specific LoRA from VRAM.
* `list_loras() -> List[str]`: Returns names of all registered LoRAs.
* `unload_all_loras()`: Forces VRAM release for all loaded adapters.

---

## Best Practices & Common Patterns

1. **Context Shifting & Prompt Caching**:

   By default, when calling `.generate()` or `.create_completion(reset=True)`, the engine checks for the longest matching prefix in the existing KV cache. To maximize speed, keep system prompts static and only append new dialogue to avoid re-evaluating the entire history. If the context limit is reached during `eval`, the model will automatically trigger a Context Shift (discarding older tokens while attempting to keep `n_keep` tokens, usually the system prompt).

2. **Basic Chat with JSON Mode**:
    Forces the model to output valid JSON by using the `response_format` parameter.
    ```python
    from llama_cpp import Llama

    llm = Llama(model_path="path/to/model.gguf", n_gpu_layers=-1)

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Extract name and age from: John is 30."}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    print(response["choices"][0]["message"]["content"])
    ```

3. **Speculative Decoding**:

    Accelerates generation by using a small "draft" model to predict tokens, which the larger model then validates in parallel.
    The fastest way to use speculative decoding is through the `LlamaNGramMapDecoding`(**Recommend**) or `LlamaPromptLookupDecoding` class.
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

    for chunk in main_llm.create_completion("Explain quantum physics", stream=True):
        print(chunk["choices"][0]["text"], end="")
    ```
    Note: `LlamaPromptLookupDecoding.num_pred_tokens` is the number of tokens to predict 10 is the default and generally good for gpu, 2 performs better for cpu-only machines. Now, `LlamaNGramMapDecoding` with the new Hash Map algorithm, draft generation becomes instantaneous $O(1)$, and the time consumption is almost 0 regardless of whether you set the prediction to 2 or 10 words.

4. **Dynamic LoRA Routing**:

   You can load multiple LoRAs using `load_lora()` at startup. Then, pass the `active_loras` parameter to `.generate()`, `.create_completion()`, or `.create_chat_completion()` to dynamically apply them to specific queries without reloading the base model.

   Multi-LoRA Dynamic Switching Example:<br>

    Load multiple adapters and apply them selectively without reloading the base model.
    ```python
    llm = Llama(model_path="base_model.gguf")
    llm.load_lora("coding", "codellama_adapter.gguf")
    llm.load_lora("story", "storywriter_adapter.gguf")
    llm.load_lora("sql_expert", "adapters/sql_lora.gguf")

    # Use coding adapter
    llm.create_completion("def sort:", active_loras=[{"name": "coding", "scale": 1.0}])

    # Use story adapter
    llm.create_completion("Once upon a time", active_loras=[{"name": "story", "scale": 0.9}])

    # Use sql adapter
    llm.create_completion("SELECT *", active_loras=[{"name": "sql_expert", "scale": 0.8}])v
    ```

5. **Hybrid & Recurrent Architectures**:

   The class natively detects Hybrid/Recurrent models (for example LFM2VL/LFM2.5VL, Qwen3.5/3.6, Mamba, RWKV, or specialized SWA models such as Gemma3/4) and automatically enables the `HybridCheckpointCache`.

   Unlike regular Transformer KV caches, Hybrid/Recurrent model memory cannot always be safely truncated token-by-token. The wrapper therefore saves periodic sequence-state checkpoints during long context prefill, allowing rollback to a verified prefix without corrupting recurrent state.

   `HybridCheckpointCache` supports two checkpoint storage modes:

   - **Host checkpoint mode** (`checkpoint_on_device=False`, default): checkpoint payloads are serialized into Python-owned bytes. This supports multiple historical checkpoints per `seq_id`, which is useful for multi-turn reuse and deeper rollback history.
   - **Device checkpoint mode** (`checkpoint_on_device=True`): checkpoint tensor payloads are stored in `llama_context`-owned device buffers via `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. Python only keeps the host-visible serialized portion. This reduces device-to-host tensor copy overhead, but only one active checkpoint per `seq_id` is safe because device payloads are keyed by `seq_id`.

   *Tips*: If you are using a hybrid multimodal model for ComfyUI nodes or single-turn API wrappers where you do not need multi-turn state rollback, initialize your `Llama` instance with `ctx_checkpoints=0`:

   ```python
   llm = Llama(
       model_path="./Qwen3.5-VL-9B.gguf",
       chat_handler=MTMDChatHandler(clip_model_path="./mmproj.gguf"),
       n_ctx=4096,
       ctx_checkpoints=0  # Disable checkpoints for zero-latency single-turn fast paths
   )
   ```

    For long prompts on GPU-backed Hybrid/Recurrent models, you can enable device-backed checkpoints to reduce device-to-host copy overhead:

    ```python
    llm = Llama(
        model_path="./Qwen3.6-27B.gguf",
        n_ctx=32768,
        n_gpu_layers=-1,
        ctx_checkpoints=16,
        checkpoint_interval=4096,
        checkpoint_on_device=True
    )
    ```

    Use `checkpoint_on_device=False` if you need multiple historical checkpoints for the same `seq_id`. Use `checkpoint_on_device=True` when fast rollback/checkpointing is more important than keeping many historical checkpoint payloads.

6.  **Assistant Prefill**:

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

7. **Interrupting Reasoning & Assistant Prefill (Time-boxing)**:

    Use the `abort()` method alongside `assistant_prefill=True` to forcefully stop a reasoning model (like Qwen or DeepSeek) if it thinks for too long, inject a bridge text, and force it to output the final answer.
    ```python
    import threading
    from llama_cpp import Llama

    llm = Llama(model_path="Qwen3.6-27B.gguf", n_ctx=4096, n_gpu_layers=-1)

    def run_controlled_generation(prompt: str, timeout_seconds: int = 10):
        messages = [{"role": "user", "content": prompt}]

        # 1. Set a time bomb to interrupt long <think> phases
        def timeout_handler():
            llm.abort()

        timer = threading.Timer(timeout_seconds, timeout_handler)
        timer.start()

        stream = llm.create_chat_completion(
            messages=messages, max_tokens=2048, stream=True
        )

        partial_response = ""
        finish_reason = None

        for chunk in stream:
            finish_reason = chunk["choices"][0].get("finish_reason")

            if finish_reason is not None and finish_reason != "abort":
                timer.cancel()
                break

            if finish_reason == "abort":
                break

            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                partial_response += delta
                print(delta, end="", flush=True)

        # 2. Forced Intervention and Prefill Continuation
        if finish_reason == "abort":
            # Inject bridge text to forcefully close the reasoning tag
            bridge_text = "\n...Wait, I have thought long enough, let's start answering the user.\n</think>\n\n"
            print(bridge_text, end="", flush=True)

            prefilled_content = partial_response + bridge_text
            messages.append({"role": "assistant", "content": prefilled_content})

            # Use assistant_prefill=True to seamlessly continue the text block
            stream_part2 = llm.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                stream=True,
                assistant_prefill=True
            )

            for chunk in stream_part2:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)

    run_controlled_generation("Explain quantum mechanics in a way that relates to bugs in code.", timeout_seconds=8)
    ```

---

## Deprecated / Changed APIs

> ⚠️ **Warning:** The internal embedding methods on the `Llama` class are deprecated and will be removed.

* `embed()` ➔ **Deprecated.**
* `create_embedding()` ➔ **Deprecated.**

**Migration Note:** Do not use `Llama(..., embeddings=True)` combined with `model.create_embedding(...)`. Instead, use the dedicated `LlamaEmbedding` class, which offers optimized batching and reranking support.
*See: [[LlamaEmbedding]]*

---

## Related Links

* [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
* [[Llama Cache](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaCache.md)] - Implementing disk or RAM-based prompt caching (LlamaRAMCache, **TrieCache**, **HybridCheckpointCache**).
* [[Llama Embedding](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaEmbedding.md)] - Dedicated class for text embeddings and reranking.
* [[Llama Speculative Decoding](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaSpeculative.md)] - Provides draft model interfaces and prompt-based speculative decoding helpers.
* [[ChatHandlers]] - Customizing `LlamaChatCompletionHandler` for function calling and vision/omni models (e.g., `[[Gemma4ChatHandler]]`, `[[Qwen35ChatHandler]]`).
