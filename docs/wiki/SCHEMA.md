# LLM Wiki Schema – llama-cpp-python

**Purpose**: Maintain a living, always-up-to-date, structured documentation wiki for the llama-cpp-python library using LLMs as the primary maintainer.

**Core Principles**:
- The source of truth is the latest code in `llama_cpp/` (especially `llama.py`, `llama_chat_format.py`, `llama_cpp.py`, `llama_types.py`, `mtmd_cpp.py`, `_internals.py`, `_ggml.py`).
- Never invent parameters or behavior. Always read the current source code before writing/updating a page.
- All examples must be complete, runnable with the latest API, and include necessary imports.
- Clearly mark any deprecated/old usage with a warning and show the modern replacement.
- Use internal wiki links (e.g. [[Llama]], [[Qwen35ChatHandler]]) for cross-referencing.
- Keep pages concise, professional, and user-friendly.

**Page Types and Templates**:

1. **Class / Module Page** (e.g. core/Llama.md)
   - Frontmatter (YAML):
     ```yaml
     ---
     title: Llama Class
     class_name: Llama
     last_updated: YYYY-MM-DD
     version_target: "latest"
     ---
     ```
   - Sections (in order):
     - Overview
     - Constructor (`__init__`) – full parameter table with types, defaults, and explanations
     - Core Methods (with signatures and examples)
     - Best Practices & Common Patterns
     - Deprecated / Changed APIs (with migration notes)
     - Related Links

2. **Feature Page** (features/xxx.md)
   - Overview, When to use, Code examples, Limitations, Related features

3. **Example Page** (examples/xxx.md)
   - Goal, Prerequisites, Complete runnable code block, Expected output, Tips

**Update Rules**:
- Before updating any page, the LLM must read the relevant source files.
- Update the `last_updated` date.
- If a new feature (e.g. new ChatHandler, new sampler) appears in code, create or expand the corresponding page.
- Maintain a high standard of readability and accuracy.

This schema is the contract. All generated content must follow it.