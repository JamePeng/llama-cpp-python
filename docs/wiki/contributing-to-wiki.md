# Contributing to the LLM Wiki

Thank you for helping improve the `llama-cpp-python` LLM Wiki.

This wiki is maintained with the help of LLMs, but all documentation must stay grounded in the latest source code. The goal is to keep the wiki accurate, practical, and easy to use for both humans and LLM-based documentation workflows.

## Documentation Source of Truth

The source of truth is always the current code in `llama_cpp/`.

Before creating or updating a wiki page, read the relevant source files first. Do not rely only on memory, old examples, outdated documentation, or external summaries.

Important source files include:

- `llama.py`
- `_internals.py`
- `llama_chat_format.py`
- `llama_cache.py`
- `llama_embedding.py`
- `llama_types.py`
- `llama_cpp.py`
- `llama_speculative.py`
- `mtmd_cpp.py`
- `_ggml.py`

## General Rules

When contributing documentation:

- Use English by default.
- Keep pages concise, clear, and practical.
- Do not invent parameters, defaults, return values, or behavior.
- Include complete runnable examples when adding code samples.
- Prefer modern APIs over deprecated or legacy usage.
- Clearly mark deprecated or changed APIs with migration notes.
- Use internal wiki links such as `[[Llama]]`, `[[Chat Completion]]`, or `[[LlamaNGramMapDecoding]]`.
- Update the `last_updated` field in page frontmatter.

## Page Structure

Follow the project wiki schema defined in [`SCHEMA.md`](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/SCHEMA.md).

Most pages should be one of the following types:

### Class / Module Page

Use this for important classes or source modules.

Typical sections:

- Overview
- Role in the Library
- Constructor (`__init__`)
- Important Attributes / State
- Core Methods
- Best Practices & Common Patterns
- Deprecated / Changed APIs
- Related Links

### Feature Page

Use this for workflows that involve multiple APIs.

Examples:

- Chat completion
- Text completion
- Embeddings
- Caching
- Speculative decoding
- Multimodal usage

Typical sections:

- Overview
- When to use
- Related APIs
- Code examples
- Configuration notes
- Limitations
- Related features

### Example Page

Use this for runnable examples.

Typical sections:

- Goal
- Prerequisites
- Complete code
- Expected output
- Tips
- Related links

## Handling Files with Multiple Classes

If a source file contains multiple related classes, create one module overview page first.

Create separate class pages only when a class is:

- Public or commonly imported by users
- Configuration-heavy
- Behavior-heavy
- A major extension point
- Likely to be searched by name

Small helper classes, internal classes, and simple data containers can usually stay documented only on the module page.

Avoid duplicating full documentation across module pages and class pages.

## Documenting Parameters and Attributes

Constructor parameters should use this format:

| Parameter | Type | Default | Description |
|---|---|---|---|

Important attributes or state should use this format:

| Attribute | Type | Source | Description |
|---|---|---|---|

Only document attributes that help users understand configuration, lifecycle, inference behavior, caching, chat formatting, embeddings, debugging, or extension points.

Do not document every private variable. Private attributes may be mentioned only when they are useful for explaining behavior or debugging, and they should be marked as internal.

## Code Examples

All code examples should:

- Include required imports.
- Be runnable with the latest API.
- Avoid pseudo-code unless explicitly marked as conceptual.
- Use clear model path placeholders such as `./model.gguf`.
- Mention assumptions such as chat format, embedding mode, GPU configuration, or required model type when relevant.

Example:

```python
from llama_cpp import Llama

llm = Llama(model_path="./model.gguf")

output = llm.create_completion("Hello,")
print(output["choices"][0]["text"])
````

## Accuracy Requirements

Do not guess.

If behavior is not clearly documented but can be inferred from implementation, say:

> Based on the current implementation, ...

If an API appears internal, say:

> This appears to be an internal implementation detail and should not be treated as a stable public API.

If you cannot verify something from the current source code, do not include it as fact.

## Pull Request Checklist

Before submitting a documentation change, check that:

* [ ] The relevant source files were reviewed.
* [ ] The page follows `SCHEMA.md`.
* [ ] Frontmatter is present and `last_updated` is updated.
* [ ] Parameters, defaults, and signatures match the source code.
* [ ] Examples are complete and runnable.
* [ ] Deprecated or legacy APIs are clearly marked.
* [ ] Internal APIs are not presented as stable public APIs.
* [ ] Related pages are linked with internal wiki links.
* [ ] The page is concise and avoids unnecessary duplication.

## Commit Message Style

Use simple documentation-focused commit messages.

Examples:

```bash
docs: add speculative decoding wiki page
docs: update Llama constructor parameters
docs: expand chat handler documentation
docs: clarify cache API usage
docs: update wiki schema to v0.3
```

## Final Note

The wiki should help users understand the latest `llama-cpp-python` API from the source code itself.

Accuracy is more important than completeness. When in doubt, verify the code first.

