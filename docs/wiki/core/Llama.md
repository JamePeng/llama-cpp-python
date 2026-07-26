---
title: Llama Class
module_name: llama_cpp.llama
source_file: llama_cpp/llama.py
class_name: Llama
last_updated: 2026-07-26
version_target: "latest"
---

## Overview

The `Llama` class is the core, high-level Python wrapper for a `llama.cpp` model. It handles model loading, memory management (KV cache), tokenization, and generation (both base text completion and chat formatting). It includes advanced features like dynamic LoRA routing, dual-mode hybrid/recurrent checkpointing, speculative decoding, and context shifting.

## Role in the Library

`Llama` is the main user-facing entry point for loading a GGUF model and
creating a native `llama.cpp` context. It exposes completion, chat, tokenization,
embedding, state, sampling, and runtime configuration APIs through one managed
object.

Use `Llama` when one application needs a general-purpose model interface.
For embedding-only applications, `LlamaEmbedding` provides embedding-oriented
defaults and additional reranking helpers while inheriting the same model and
context lifecycle.

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
| `numa` | `Union[bool, int]` | `False` | NUMA strategy (e.g., `GGML_NUMA_STRATEGY_DISTRIBUTE`). |

### Context & Batch Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_ctx` | `int` | `512` | Text context size. Set to `0` to load from model metadata. |
| `n_keep` | `int` | `256` | Preferred number of leading tokens to preserve during automatic context shifting. |
| `n_batch` | `int` | `2048` | Maximum number of tokens in a logical prompt-processing batch. The effective value cannot exceed `n_ctx`. |
| `n_ubatch` | `int` | `512` | Maximum number of tokens in a physical micro-batch processed by llama.cpp. |
| `n_seq_max` | `int` | `1` | Maximum independent sequence states in one decode batch. Embedding calls split automatically at this limit; larger values enable more parallel sequences. |
| `n_rs_seq` | `int` | `0` | Experimental recurrent-state snapshots retained per sequence for rollback. `0` disables rollback snapshots. |
| `n_outputs_max` | `int` | `0` | Maximum outputs in a physical batch. `0` is converted to the effective `n_batch`. |
| `n_threads` | `int` | `None` | Number of threads for generation (defaults to CPU count // 2). |
| `n_threads_batch` | `int` | `None` | Number of threads for batch processing (defaults to CPU count). |
| `ctx_type` | `int` | `LLAMA_CONTEXT_TYPE_DEFAULT` | Context implementation selected by llama.cpp. Keep the default unless a model or backend requires another context type. |

### Embedding, Attention & KV Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `embeddings` | `bool` | `False` | Enable embedding extraction alongside logits. Must be `True` before calling `embed()` or `create_embedding()`. |
| `pooling_type` | `int` | `LLAMA_POOLING_TYPE_UNSPECIFIED` | Pooling strategy for embedding output. `UNSPECIFIED` follows model metadata, `NONE` returns token-level vectors, and `RANK` returns classifier or reranking output. |
| `attention_type` | `int` | `LLAMA_ATTENTION_TYPE_UNSPECIFIED` | Attention mode used by the context. `UNSPECIFIED` lets llama.cpp select the model-compatible behavior. |
| `logits_all` | `bool` | `False` | Retain logits for every evaluated token instead of only requested outputs. Completion log probabilities require this mode. |
| `flash_attn_type` | `int` | `LLAMA_FLASH_ATTN_TYPE_AUTO` | Controls when Flash Attention is enabled. |
| `offload_kqv` | `bool` | `True` | Offload K, Q, and V tensor operations to the selected device when supported. |
| `swa_full` | `Optional[bool]` | `None` | Use a full-size sliding-window-attention cache. `None` keeps llama.cpp's default. |
| `kv_unified` | `Optional[bool]` | `None` | Use a unified KV buffer for all sequences. `LlamaEmbedding` enables this automatically. |
| `type_k` / `type_v` | `Optional[int]` | `None` | KV cache data types for keys and values. `None` uses llama.cpp defaults. |

### Advanced & Chat Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `chat_format` | `str` | `None` | String specifying the chat template (e.g., `"llama-2"`, `"chatml"`). Guessed from GGUF if None. |
| `chat_handler` | `LlamaChatCompletionHandler` | `None` | Optional custom handler. See [[ChatHandlers]]. |
| `draft_model` | `LlamaDraftModel` | `None` | Optional draft model for speculative decoding. |
| `ctx_checkpoints` | `int` | `16` | Max hybrid/recurrent context checkpoints to keep. Set to `0` to disable checkpointing for single-turn fast paths. |
| `checkpoint_interval` | `int` | `4096` | Token interval for saving periodic Hybrid/Recurrent checkpoints during long prompt evaluation. |
| `checkpoint_on_device` | `bool` | `False` | Store Hybrid/Recurrent checkpoint tensor payloads in `llama_context`-owned device buffers via `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. Reduces device-to-host copy overhead, but only one active checkpoint per `seq_id` is safe. |

### Runtime Logging Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `verbose` | `bool` | `True` | Backward-compatible boolean native logging switch. `False` keeps only error-level llama.cpp / ggml logs; `True` enables debug-level native logs. If `verbosity` is provided, `verbosity` takes precedence over `verbose`. |
| `verbosity` | `Optional[Union[int, str, bool]]` | `None` | Fine-grained llama.cpp-style native runtime log verbosity. Numeric levels: `0=output`, `1=error`, `2=warning`, `3=info`, `4=trace`, `5=debug`. Use `verbosity=3` for llama.cpp-style default info logs. String aliases such as `"silent"`, `"quiet"`, `"info"`, `"trace"`, and `"debug"` are also accepted. |
| `log_filters` | `Optional[Sequence[str]]` | `None` | Optional substring filters for native runtime logs. If any provided substring appears in a decoded backend log message, that message is suppressed. The default logger may include built-in filters for noisy low-level logs such as `CUDA Graph id %d reuse` messages. Pass an empty list `[]` to disable default substring filtering. |
| `log_filters_case_sensitive` | `bool` | `True` | Whether `log_filters` should match case-sensitively. Defaults to `True` for predictable low-level backend log filtering. |

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

### Runtime Logging Control

The `Llama` class exposes lightweight runtime helpers for adjusting native llama.cpp / ggml logging after initialization.

> **Note:** Native backend logging is process-global because llama.cpp / ggml use a global log callback. Changing verbosity or log filters affects all `Llama` instances in the current Python process.

* `set_verbosity(verbosity: Union[int, str, bool, None])`: Set native runtime log verbosity.
* `get_verbosity() -> int`: Return the current native runtime log verbosity.
* `set_log_filters(filters: Sequence[str], case_sensitive: bool = True)`: Replace substring filters for native runtime logs.
* `add_log_filters(filters: Sequence[str])`: Append substring filters.
* `get_log_filters() -> List[str]`: Return the current substring filters.
* `clear_log_filters()`: Clear all substring filters, including default filters.
* `reset_log_filters()`: Restore default substring filters.

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/qwen3.gguf",
    verbosity=3,  # llama.cpp-style info logs
)

# Temporarily enable debug-level native logs.
llm.set_verbosity(5)

# Suppress noisy backend messages by substring.
llm.add_log_filters([
    "CUDA Graph",
    "CUDA graph",
    "clip_model_loader: tensor",
])

# Return to quiet error-only logging.
llm.set_verbosity(1)
```

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
    llm.create_completion("SELECT *", active_loras=[{"name": "sql_expert", "scale": 0.8}])
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
8. **Runtime Logging & Backend Noise Filtering**:

   `Llama` supports fine-grained native llama.cpp / ggml logging through `verbosity`. This is more precise than the legacy `verbose` boolean flag.

   ```python
   from llama_cpp import Llama

   # Legacy behavior:
   # verbose=False -> error-only logs
   llm_quiet = Llama(
       model_path="models/qwen3.gguf",
       verbose=False,
   )

   # Recommended precise logging:
   # 0 = output, 1 = error, 2 = warning, 3 = info, 4 = trace, 5 = debug
   llm = Llama(
       model_path="models/qwen3.gguf",
       verbosity=3,  # llama.cpp-style default info logs
   )
    ```

    For low-level debugging, use `verbosity=5`. By default, the logger may suppress known noisy backend messages such as CUDA Graph reuse logs. Pass `log_filters=[]` to disable all substring filtering.

    ```python
    llm = Llama(
        model_path="models/qwen3.gguf",
        verbosity=5,
        log_filters=[],  # show all debug logs, including normally filtered ones
    )
    ```

    To suppress additional noisy messages, pass substring filters:

    ```python
    llm = Llama(
        model_path="models/qwen3.gguf",
        verbosity=5,
        log_filters=[
            "CUDA Graph id",
            "clip_model_loader: tensor",
            "ggml_cuda_graph_update_required",
        ],
    )
    ```

    You can also adjust logging at runtime:

    ```python
    llm.set_verbosity(5)
    llm.add_log_filters(["llama_perf_context_print"])

    # Later, return to warning-level logs.
    llm.set_verbosity(2)
    ```

    **Important:** native backend logging is process-global. Runtime changes affect all `Llama` instances in the same Python process.

    **verbose=False** vs. **verbosity=0**: These have distinct behaviors.
    - `verbose=False` silences Python wrapper prints but not backend diagnostics; like `if self.verbose: print()`
    - `verbosity=0` silences all backend non-error output.

---

## Embeddings

The `Llama` embedding methods are maintained and use streaming batches. Create
the model with `embeddings=True` before calling them.

```python
from llama_cpp import Llama, LLAMA_POOLING_TYPE_UNSPECIFIED

llm = Llama(
    model_path="path/to/embedding-model.gguf",
    embeddings=True,
    pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED,
    n_batch=512,
    n_ubatch=512,
    n_seq_max=8,
    kv_unified=True,
)

try:
    # Raw sequence embeddings with explicit L2 normalization.
    vectors = llm.embed(["query", "document"], normalize=2)

    # OpenAI-compatible response.
    response = llm.create_embedding(
        ["query", "document"],
        normalize=True,
    )
finally:
    llm.close()
```

### `embed(input, normalize=False, truncate=True, separator=None, return_count=False)`

Generate raw embedding values for strings or pre-tokenized inputs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | `Union[str, List[str], List[List[int]]]` | Required | A single string, a list of strings, or a list containing pre-tokenized token-ID lists. |
| `normalize` | `Union[bool, int]` | `False` | `False` returns raw values, while `True` applies L2 normalization. Integer modes are listed below. Rank outputs are not normalized. |
| `truncate` | `bool` | `True` | Truncate each input to the smaller of the context capacity and logical batch capacity. If disabled, an input longer than `n_batch` raises `ValueError`. |
| `separator` | `Optional[str]` | `None` | Split a single string into multiple independent inputs. When set, the result uses the batch return shape. |
| `return_count` | `bool` | `False` | Return `(result, total_token_count)` instead of only the embedding result. |

Normalization modes follow the llama.cpp embedding example:

| Value | Behavior |
|---|---|
| `False` or `-1` | No normalization |
| `True` or `2` | Euclidean/L2 normalization |
| `0` | Scale by the maximum absolute value to a maximum magnitude of `32760` |
| `1` | Taxicab/L1 normalization |
| Integer greater than `2` | Corresponding p-norm normalization |

Unlike `LlamaEmbedding.embed()`, the standard `Llama.embed()` method defaults to
raw, unnormalized output for backward compatibility.

The return shape depends on the input and pooling type:

| Input / pooling mode | Return shape |
|---|---|
| Single string with sequence pooling | `List[float]` |
| String list or separator-split string with sequence pooling | `List[List[float]]` |
| `LLAMA_POOLING_TYPE_NONE` | One token embedding matrix per input: `List[List[float]]` for a single string or `List[List[List[float]]]` for a batch |
| `LLAMA_POOLING_TYPE_RANK` with one classifier output | A scalar for a single string or a list of scalars for a batch |
| `LLAMA_POOLING_TYPE_RANK` with multiple classifier outputs | A classifier vector for each input |
| Any mode with `return_count=True` | `(result, total_token_count)` |

Use `LLAMA_POOLING_TYPE_UNSPECIFIED` for ordinary sentence embeddings unless
the model documentation requires a specific sequence pooling strategy.
`LLAMA_POOLING_TYPE_NONE` is token-level output and should not be used when one
vector per input document is expected.

### `create_embedding(input, model=None, normalize=False, truncate=True)`

Wrap sequence or token-level embedding output in an OpenAI-compatible response:

```python
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [...],
            "index": 0,
        }
    ],
    "model": "path/to/embedding-model.gguf",
    "usage": {
        "prompt_tokens": 12,
        "total_tokens": 12,
    },
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | `Union[str, List[str]]` | Required | One string or a list of strings. |
| `model` | `Optional[str]` | `None` | Model name placed in the response. Defaults to `model_path`. |
| `normalize` | `Union[bool, int]` | `False` | Passed directly to `embed()`. |
| `truncate` | `bool` | `True` | Passed directly to `embed()`. |

For parallel batches, `n_seq_max` must cover every sequence ID active in a
single decode batch. The default `n_seq_max=1` is valid and processes multiple
inputs sequentially. Increasing it allows more inputs to be decoded in
parallel; for example, `n_seq_max=8` permits IDs `0` through `7` in one batch.
`n_batch` limits logical input tokens, `n_ubatch` controls the physical token
batch, and `n_seq_max` limits independent sequences.

`LlamaEmbedding` remains available as the specialized convenience class. It
automatically enables embedding-oriented context options, defaults to L2
normalization, provides additional output formats, and adds the `rank()` helper
for formatting query/document pairs.

> **OpenAI compatibility:** use sequence pooling when calling
> `create_embedding()` through an OpenAI-compatible client. Token-level pooling
> (`LLAMA_POOLING_TYPE_NONE`) produces nested token vectors rather than the
> single flat vector normally expected for each input.

---

## Related Links

* [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
* [[Llama Cache](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaCache.md)] - Implementing disk or RAM-based prompt caching (LlamaRAMCache, **TrieCache**, **HybridCheckpointCache**).
* [[Llama Embedding](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaEmbedding.md)] - Dedicated class for text embeddings and reranking.
* [[Llama Speculative Decoding](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaSpeculative.md)] - Provides draft model interfaces and prompt-based speculative decoding helpers.
* [[ChatHandlers]] - Customizing `LlamaChatCompletionHandler` for function calling and vision/omni models (e.g., `[[Gemma4ChatHandler]]`, `[[Qwen35ChatHandler]]`).
