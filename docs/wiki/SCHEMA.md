# LLM Wiki Schema – llama-cpp-python

**Schema Metadata**:
- **Author**: JamePeng
- **Maintainer**: LLM-assisted documentation workflow
- **Project**: [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) wiki
- **Last Modified**: 2026-05-02
- **Version Target**: latest source code
- **Schema Version**: 0.3

**Purpose**:
- Maintain a living, always-up-to-date, structured documentation wiki for the `llama-cpp-python` library, with LLMs acting as the primary documentation maintainer.
- The wiki must help users understand the latest public API, core classes, modules, configuration options, examples, and migration paths based on the current source code.
- The wiki should explain not only *how to call an API*, but also *what role the class/module plays in the library*, *how its state is configured*, and *how users should choose between related APIs*.

**Core Principles**:
- The source of truth is the latest code in `llama_cpp/`, especially:
  - `llama.py`
  - `_internals.py`
  - `llama_chat_format.py`
  - `llama_cache.py`
  - `llama_embedding.py`
  - `llama_types.py`
  - `llama_cpp.py`
  - `mtmd_cpp.py`
  - `_ggml.py`
- Never invent parameters or behavior. Always read the current source code before writing/updating a page.
- Prefer documenting public and user-facing APIs first. Internal implementation details may be documented only when they help users understand behavior, extension points, debugging, or advanced usage.
- All examples must be complete, runnable with the latest API, and include necessary imports.
- Clearly mark deprecated, legacy, or changed usage with a warning and show the modern replacement.
- Use internal wiki links (e.g. [[Llama]], [[Qwen35ChatHandler]]) for cross-referencing.
- Keep pages concise, professional, and user-friendly.

**Documentation Language**:
- The default documentation language is **English**.
- All generated wiki pages, examples, explanations, titles, tables, and warnings should be written in English unless the user explicitly requests another language.
- Code comments inside examples should also be in English by default.
- If the source code contains Chinese comments or non-English notes, translate them into clear English while preserving the original meaning.

**Page Types and Templates**:

1. **Class / Module Page** (e.g. core/Llama.md, modules/LlamaEmbedding.md)
   - Frontmatter (YAML):
     ```yaml
     ---
     title: Llama Class
     class_name: Llama
     source_file: llama_cpp/llama.py
     last_updated: YYYY-MM-DD
     version_target: "latest"
     ---
     ```
   - Sections (in order):
     - Overview
     - Role in the Library
     - Constructor (`__init__`) – full parameter table with types, defaults, and explanations
     - Important Attributes / State
     - Core Methods (with signatures and usage examples)
     - Best Practices & Common Patterns
     - Deprecated / Changed APIs (with migration notes)
     - Related Links

   - The **Overview** should briefly explain:
     - What the class or module is.
     - What problem it solves.
     - Whether it is a high-level public API, extension point, helper, or internal implementation detail.
     - When users should use it.

   - The **Role in the Library** should explain how the class or module relates to nearby APIs. For example, whether it wraps low-level bindings, handles chat formatting, manages cache state, provides embeddings, or connects to multimodal behavior.

   - Constructor parameter tables should use:

     | Parameter | Type | Default | Description |
     |---|---|---|---|

   - Important attributes or state should use:

     | Attribute | Type | Source | Description |
     |---|---|---|---|

   - Only document attributes that affect user understanding, configuration, lifecycle, inference behavior, caching, chat formatting, embeddings, or debugging. Do not document every trivial private variable.

2. **Feature Page** (features/xxx.md)
   - Overview, When to use, Related APIs, Code examples, Configuration Notes, Limitations, Related features
   - Feature pages should explain workflows across multiple classes or modules.

3. **Example Page** (examples/xxx.md)
   - Goal, Prerequisites, Complete runnable code block, Expected output, Tips
   - Rules:
      * Use the latest API.
      * Include all imports as need.
      * Avoid pseudo-code.
      * Keep examples focused.
      * Mention required model assumptions when needed, such as GGUF file path or chat format.

**Update Rules**:
- Before updating any page, the LLM must read the relevant source files.
- Update the `last_updated` date.
- If a new feature appears, such as a new chat handler, sampler, cache type, embedding API, multimodal API, or backend option, create or expand the corresponding page.
- If behavior is inferred from implementation rather than explicitly documented in code, mark the explanation as implementation-based.
- Maintain a high standard of readability and accuracy.

This schema is the contract. All generated content must follow it.