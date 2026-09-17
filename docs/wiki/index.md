# llama-cpp-python Wiki

Welcome to the `llama-cpp-python` wiki :)

This wiki provides source-aligned documentation for the public APIs, core
classes, feature workflows, examples, and maintainer tooling in
`llama-cpp-python`. The latest code in `llama_cpp/` and the corresponding
vendored `llama.cpp` APIs remain the source of truth.

---

## Quick Navigation

### Getting Started

| Page | Description |
|---|---|
| [Installation](install.md) | Build and source-installation guide covering Python setup, CMake options, native backends, hardware acceleration, rebuilds, and verification. |
| [Runtime Troubleshooting](troubleshooting.md) | Cache misses, stale snapshots, cancellation, media failures, and embedding state changes. |

### Core API

| Page | Description |
|---|---|
| [Llama](core/Llama.md) | Main high-level interface for GGUF loading, text and chat completion, tokenization, embeddings, sampling, speculative decoding, caching, and lifecycle management. |

### Modules

| Page | Description |
|---|---|
| [Llama Cache](modules/LlamaCache.md) | Cache interfaces and implementations for reusing model state across repeated prompts. |
| [Llama Embedding](modules/LlamaEmbedding.md) | Dedicated embedding APIs, configuration, output formats, and batching behavior. |
| [Llama Grammar](modules/LlamaGrammar.md) | Grammar definitions, custom roots, lazy triggers, JSON Schema conversion, errors, and sampler ownership. |
| [Llama Speculative Decoding](modules/LlamaSpeculative.md) | Stateful MTP, DFlash, DFlash2, DSpark, and n-gram engines; configuration, lifecycle, rollback, statistics, and benchmarks. |
| [Logger](modules/Logger.md) | Python and native logging configuration, callbacks, levels, filtering, and output routing. |
| [llama.cpp ctypes Bindings](modules/LlamaCppBindings.md) | Source-oriented reference for the low-level llama.cpp and ggml ctypes bindings. |
| [MTMD ctypes Bindings](modules/MTMDCppBindings.md) | Source-oriented reference for the low-level multimodal ctypes bindings. |

### Feature Guides

| Page | Description |
|---|---|
| [Caching and State Reuse](features/caching.md) | Live prefixes, full snapshots, partial checkpoints, ownership, and cache misses. |
| [Embeddings and Reranking](features/embeddings-rerank.md) | End-to-end sentence embeddings, token-level vectors, normalization, streaming batches, similarity output, and cross-encoder reranking. |
| [Grammar and Constrained Generation](features/grammar.md) | GBNF and JSON Schema usage, reusable definitions, lazy sampling, conversion optimizations, and supported behavior and limitations. |

### Examples

| Page | Description |
|---|---|
| [Qwen3.5 Image Chat](examples/vision/vision-qwen.md) | Matching model/projector setup, image requests, n-gram speculation, and prefill recovery. |
| [DFlash2 Speculative Decoding](examples/dflash2-speculative-decoding.md) | Configure a DFlash2 sidecar, validate selector execution, compare ordinary and speculative output, and tune draft length. |
| [MTMD Speech Synthesis](examples/audio/audio-tts.md) | Generate speech with Qwen3-TTS Base or Pocket TTS; configure reference audio and FA, and explore CLI and Streamlit examples. |

### Development

| Page | Description |
|---|---|
| [Git Commit Generation Agent](development/git-commit-generation-agent.md) | Maintainer workflow for producing clear, structured, and source-aware Git commit messages. |

### Wiki Maintenance

| Page | Description |
|---|---|
| [Wiki Schema](SCHEMA.md) | Documentation structure, page templates, source requirements, and maintenance rules. |
| [Contributing to the Wiki](contributing-to-wiki.md) | Contribution workflow for creating and updating documentation. |

---

## Recommended Reading Order

For general model loading and generation:

1. [Installation](install.md)
2. [Llama](core/Llama.md)
3. [Llama Cache](modules/LlamaCache.md)
4. [Llama Grammar](modules/LlamaGrammar.md)
5. [Logger](modules/Logger.md)

For embeddings and reranking:

1. [Llama Embedding](modules/LlamaEmbedding.md)
2. [Embeddings and Reranking](features/embeddings-rerank.md)

For grammar-constrained generation:

1. [Grammar and Constrained Generation](features/grammar.md)
2. [Llama Grammar](modules/LlamaGrammar.md)
3. [Llama](core/Llama.md)

For speculative decoding:

1. [Llama](core/Llama.md)
2. [Llama Speculative Decoding](modules/LlamaSpeculative.md)
3. [DFlash2 Speculative Decoding](examples/dflash2-speculative-decoding.md)

For text-to-speech:

1. [MTMD Speech Synthesis](examples/audio/audio-tts.md) — Qwen3-TTS Base and Pocket TTS, reference audio, Flash Attention, output formats, and current cloning limitations.
2. [CLI TTS Example](../../examples/high_level_api/mtmd_tts.py) — single requests, multilingual scenarios, and batch synthesis.
3. [Streamlit TTS Playground](../../examples/streamlit_tts/README.md) — reference upload/recording, playback, and downloads.

For documentation contributors:

1. [Wiki Schema](SCHEMA.md)
2. [Contributing to the Wiki](contributing-to-wiki.md)
3. [Git Commit Generation Agent](development/git-commit-generation-agent.md)

---

## Documentation Status

Completed pages currently linked from this index:

- `install.md`
- `core/Llama.md`
- `modules/LlamaCache.md`
- `modules/LlamaEmbedding.md`
- `modules/LlamaGrammar.md`
- `modules/LlamaSpeculative.md`
- `modules/Logger.md`
- `modules/LlamaCppBindings.md`
- `modules/MTMDCppBindings.md`
- `features/embeddings-rerank.md`
- `features/grammar.md`
- `features/caching.md`
- `examples/vision/vision-qwen.md`
- `troubleshooting.md`
- `examples/dflash2-speculative-decoding.md`
- `examples/audio/audio-tts.md`
- `development/git-commit-generation-agent.md`
- `SCHEMA.md`
- `contributing-to-wiki.md`

The repository also contains empty placeholder files for planned documentation.
They are intentionally not linked as usable pages until content has been added
and checked against the implementation.

### Planned areas

- Basic and chat-completion examples
- Additional vision models and audio-input examples
- Multi-model and tool-call feature guides
- Common and MCP type references
- Additional backend diagnostics

---

## Documentation Principles

- Treat source code as the source of truth.
- Keep parameters, defaults, version availability, and behavior aligned with
  the latest implementation.
- Prefer complete, runnable examples without local machine-specific paths.
- Clearly mark deprecated APIs, preview features, and current limitations.
- Distinguish stable public interfaces from private implementation helpers.
- Do not link empty placeholder pages as finished documentation.
- Keep pages concise, practical, and easy to navigate.

---

## Project Links

- [llama-cpp-python on GitHub](https://github.com/JamePeng/llama-cpp-python)
- [Installation guide](install.md)
- [Wiki schema](SCHEMA.md)
- [Contribution guide](contributing-to-wiki.md)
