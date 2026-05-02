# llama-cpp-python Wiki

Welcome to the `llama-cpp-python` wiki :)

This wiki provides structured, source-code-aligned documentation for the public APIs, core classes, modules, examples, and development notes of `llama-cpp-python`.

The documentation is maintained with the help of LLMs, but the source of truth is always the latest code in `llama_cpp/`.

---

## Quick Navigation

### Core API

Start here if you are using `llama-cpp-python` directly.

| Page | Description |
|---|---|
| [core/Llama\|Llama] | Main high-level interface for loading GGUF models, running completions, chat completions, tokenization, embeddings, and model configuration. |

---

### Modules

These pages document major source modules and related classes.

| Page | Description |
|---|---|
| [modules/LlamaCache\|Llama Cache] | Cache interfaces and implementations for reusing model state across repeated prompts. |
| [modules/LlamaEmbedding\|Llama Embedding] | Embedding-related APIs and usage patterns. |
| [modules/LlamaSpeculative\|Llama Speculative Decoding] | Draft model interfaces and prompt-based speculative decoding helpers. |

---

### Wiki Maintenance

These pages define how the wiki should be written, updated, and reviewed.

| Page | Description |
|---|---|
| [SCHEMA\|Wiki Schema] | Documentation schema and rules for LLM-maintained wiki pages. |
| [contributing-to-wiki\|Contributing to the Wiki] | Contribution guide for writing and updating wiki documentation. |

---

## Recommended Reading Order

If you are new to this wiki, read the pages in this order:

1. [[core/Llama|Llama](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
2. [[modules/LlamaEmbedding|Llama Embedding](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaEmbedding.md)]
3. [[modules/LlamaCache|Llama Cache](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaCache.md)]
4. [[modules/LlamaSpeculative|Llama Speculative Decoding](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/modules/LlamaSpeculative.md)]

If you are contributing documentation, start with:

1. [[SCHEMA|Wiki Schema](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/SCHEMA.md)]
2. [[contributing-to-wiki|Contributing to the Wiki](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/contributing-to-wiki.md)]

---

## Documentation Status

The wiki is still being expanded.

Currently available pages:

- `core/Llama.md`
- `modules/LlamaCache.md`
- `modules/LlamaEmbedding.md`
- `modules/LlamaSpeculative.md`
- `SCHEMA.md`
- `contributing-to-wiki.md`

Some planned pages may already exist as empty placeholder files. Empty pages are intentionally not linked from this index until they are completed.

---

## Planned Areas

Future documentation may cover:

- Installation and build options
- Chat formats and chat handlers
- Low-level ctypes bindings
- Multimodal APIs
- Type definitions and structured return values
- Troubleshooting
- Runnable examples
- Development notes

---

## Documentation Principles

This wiki follows a few core rules:

- Source code is the source of truth.
- Parameters, defaults, and behavior must match the latest implementation.
- Examples should be complete and runnable.
- Deprecated or legacy APIs should be clearly marked.
- Internal implementation details should not be presented as stable public APIs.
- Pages should be concise, practical, and easy to navigate.

---

## Project Links

- GitHub: [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)
- Wiki schema: [SCHEMA](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/SCHEMA.md)
- Contribution guide: [contributing-to-wiki](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/contributing-to-wiki.md)