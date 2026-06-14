# LLM Wiki Schema – llama-cpp-python

**Schema Metadata**:
- **Author**: JamePeng
- **Maintainer**: LLM-assisted documentation workflow
- **Project**: [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) wiki
- **Last Modified**: 2026-06-02
- **Version Target**: latest source code
- **Schema Version**: 0.4

**Purpose**:
- Maintain a living, always-up-to-date, structured documentation wiki for the `llama-cpp-python` library, with LLMs acting as the primary documentation maintainer.
- The wiki must help users understand the latest public API, core classes, modules, configuration options, examples, and migration paths based on the current source code.
- The wiki should explain not only *how to call an API*, but also *what role the class/module plays in the library*, *how its state is configured*, and *how users should choose between related APIs*.
- The schema also defines the expected wiki directory layout, page ownership, and update rules so new pages can be generated consistently.

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
  - `_logger.py`
- Never invent parameters or behavior. Always read the current source code before writing/updating a page.
- Prefer documenting public and user-facing APIs first. Internal implementation details may be documented only when they help users understand behavior, extension points, debugging, or advanced usage.
- All examples must be complete, runnable with the latest API, and include necessary imports.
- Clearly mark deprecated, legacy, or changed usage with a warning and show the modern replacement.
- Use internal wiki links, such as `[[Llama]]`, `[[LlamaCache]]`, `[[LlamaSpeculative]]`, or `[[Qwen35ChatHandler]]`, for cross-referencing.
- Keep pages concise, professional, and user-friendly.

**Documentation Language**:
- The default documentation language is **English**.
- All generated wiki pages, examples, explanations, titles, tables, and warnings should be written in English unless the user explicitly requests another language.
- Code comments inside examples should also be in English by default.
- If the source code contains Chinese comments or non-English notes, translate them into clear English while preserving the original meaning.

**Wiki Directory Layout**:

The wiki should be organized by documentation purpose rather than by source-file location alone.

```text
docs/wiki/
├─ core/               # Core classes and modules (e.g., Llama, main API objects)
├─ development/        # Developer-focused pages, tools, agents, CI/CD workflows
├─ examples/           # Complete runnable examples for users
├─ features/           # High-level features spanning multiple classes/modules
├─ modules/            # Specialized modules (cache, embeddings, logging, speculative decoding, bindings)
├─ types/              # Type definitions and data structures used across the library
├─ .gitkeep            # Placeholder for Git to track empty directories
├─ contributing-to-wiki.md  # Guidelines for contributing to the wiki
├─ index.md            # Entry point and table of contents
├─ install.md          # Installation instructions
├─ SCHEMA.md           # Documentation schema and style guide (this file)
├─ troubleshooting.md  # Known issues, debugging tips, FAQ
```

### Top-Level Files

| Path | Purpose | Update Guidance |
|---|---|---|
| `docs/wiki/SCHEMA.md` | Defines the documentation contract, directory structure, page templates, and LLM update rules. | Update when adding a new page type, directory, documentation standard, or structural convention. |
| `docs/wiki/index.md` | Main wiki landing page and navigation entry. | Update when important pages are added, renamed, reorganized, or promoted. |
| `docs/wiki/contributing-to-wiki.md` | Human and LLM contribution guide for maintaining the wiki. | Keep aligned with this schema, especially source-reading and accuracy rules. |
| `docs/wiki/install.md` | Installation guide placeholder or final installation documentation. | Convert from placeholder to complete page when installation docs are ready. |
| `docs/wiki/troubleshooting.md` | Troubleshooting guide placeholder or final diagnostics documentation. | Expand with common runtime, build, backend, model loading, and environment issues. |
| `docs/wiki/.gitkeep` | Keeps the wiki directory tracked when needed. | No documentation content is required. |

### Directory Ownership

| Directory | Purpose | Typical Content | Primary Audience |
|---|---|---|---|
| `core/` | High-level public entry points and central user APIs. | `Llama`, model lifecycle, generation APIs, chat/completion interfaces. | General users and advanced users. |
| `modules/` | Focused subsystem pages, user-facing modules, low-level bindings, helpers, and advanced API areas. | Cache, embeddings, grammar, speculative decoding, logging, llama.cpp bindings, MTMD bindings. | Advanced users, extension authors, maintainers. |
| `features/` | Workflow-oriented guides that span multiple APIs or modules. | Chat formatting, structured output, multimodal usage, backend loading, caching workflows, speculative decoding workflows. | Users solving a specific task. |
| `examples/` | Complete runnable examples. | Minimal inference, chat completion, embeddings, grammar-constrained generation, speculative decoding, multimodal usage. | Users who want copy-paste starting points. |
| `types/` | Type and schema documentation. | Request/response structures, typed dictionaries, protocol-style types, OpenAI-compatible payloads. | Users integrating with typed code or API-compatible workflows. |
| `development/` | Maintainer-facing documentation and contribution workflows. | Build notes, CI notes, release notes, commit generation workflow, documentation maintenance rules. | Maintainers and contributors. |

**Page Types and Templates**:

1. **Class / Module Page**
   Examples: `core/Llama.md`, `modules/LlamaEmbedding.md`, `modules/LlamaCache.md`

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

   - Sections, in order:
     - Overview
     - Role in the Library
     - Constructor (`__init__`) – full parameter table with types, defaults, and explanations
     - Important Attributes / State
     - Core Methods, with signatures and usage examples
     - Best Practices & Common Patterns
     - Deprecated / Changed APIs, with migration notes
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

2. **Feature Page**
   Example: `features/speculative-decoding.md`, `features/embeddings-rerank.md`

   Feature pages should explain workflows across multiple classes or modules.

   Required sections:
   - Overview
   - When to Use
   - Related APIs
   - Code Examples
   - Configuration Notes
   - Limitations
   - Related Features

3. **Example Page**
   Example: `examples/chat-completion.md`

   Required sections:
   - Goal
   - Prerequisites
   - Complete Runnable Code
   - Expected Output
   - Tips

   Rules:
   - Use the latest API.
   - Include all required imports.
   - Avoid pseudo-code.
   - Keep examples focused.
   - Mention required model assumptions when needed, such as GGUF file path, embedding mode, grammar file, chat format, or multimodal assets.

4. **Development Page**
   Example: `development/GitCommitGenerationAgent.md`

   Development pages are maintainer-facing and may document repository workflows, CI, release notes, build matrix decisions, or documentation maintenance conventions.

   Required sections:
   - Overview
   - Scope
   - Workflow
   - Inputs / Outputs
   - Rules and Constraints
   - Examples
   - Related Links

**Cross-Linking Rules**:

- Use wiki-style internal links for pages that exist or should exist, such as `[[Llama]]`, `[[LlamaCache]]`, `[[LlamaSpeculative]]`, and `[[Logger]]`.
- Link from high-level pages to lower-level module pages when the module explains advanced details.
- Link from feature pages back to the relevant class/module pages.
- Avoid circular explanations. A page may link to another page for details instead of repeating the same explanation.

**Update Rules**:

- Before updating any page, the LLM must read the relevant source files.
- Update the `last_updated` date.
- If a new feature appears, such as a new chat handler, sampler, cache type, embedding API, multimodal API, backend option, or binding wrapper, create or expand the corresponding page.
- If behavior is inferred from implementation rather than explicitly documented in code, mark the explanation as implementation-based.
- Empty files should be converted into explicit placeholder pages instead of being left blank.
- Maintain a high standard of readability and accuracy.

**Quality Checklist**:

Before finalizing a wiki page, verify:

- The page reflects the latest source code.
- All parameters, defaults, and return values are accurate.
- Examples are runnable and include necessary imports.
- Internal links point to the correct wiki page names.
- Advanced or low-level APIs are clearly labeled.
- Deprecated behavior is clearly separated from current usage.
- The page avoids undocumented claims, speculative behavior, or outdated assumptions.

This schema is the contract. All generated content must follow it.
