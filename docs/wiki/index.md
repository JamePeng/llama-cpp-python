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

### Core API

| Page | Description |
|---|---|
| [Llama](core/Llama.md) | Main high-level interface for GGUF loading, text and chat completion, tokenization, embeddings, sampling, speculative decoding, caching, and lifecycle management. |

### Modules

| Page | Description |
|---|---|
| [Llama Cache](modules/LlamaCache.md) | Cache interfaces and implementations for reusing model state across repeated prompts. |
| [Llama Embedding](modules/LlamaEmbedding.md) | Dedicated embedding APIs, configuration, output formats, and batching behavior. |
| [Llama Grammar](modules/LlamaGrammar.md) | Grammar parsing and constrained-generation utilities. |
| [Llama Speculative Decoding](modules/LlamaSpeculative.md) | Stateful MTP, DFlash, DSpark, and n-gram engines; configuration, lifecycle, rollback, statistics, and benchmarks. |
| [Logger](modules/Logger.md) | Python and native logging configuration, callbacks, levels, filtering, and output routing. |

### Feature Guides

| Page | Description |
|---|---|
| [Embeddings and Reranking](features/embeddings-rerank.md) | End-to-end sentence embeddings, token-level vectors, normalization, streaming batches, similarity output, and cross-encoder reranking. |

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

For speculative decoding:

1. [Llama](core/Llama.md)
2. [Llama Speculative Decoding](modules/LlamaSpeculative.md)

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
- `features/embeddings-rerank.md`
- `development/git-commit-generation-agent.md`
- `SCHEMA.md`
- `contributing-to-wiki.md`

The repository also contains empty placeholder files for planned documentation.
They are intentionally not linked as usable pages until content has been added
and checked against the implementation.

### Planned areas

- Basic and chat-completion examples
- Speculative-decoding examples
- Vision and audio examples
- Caching, grammar, multi-model, and tool-call feature guides
- Low-level llama.cpp and MTMD ctypes bindings
- Common and MCP type references
- Troubleshooting and backend diagnostics

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
