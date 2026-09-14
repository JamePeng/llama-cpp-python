# Contributing to the LLM Wiki

Help readers choose the right API, use it correctly, and understand its limits.
Human and LLM contributions follow the same source verification standards. The
[wiki schema](SCHEMA.md) defines coverage, metadata, and writing conventions;
this guide describes how to apply them.

## 1. Establish scope and evidence

Read the page, its related guide or API reference, and the relevant checked-out
source before editing. Follow calls into helpers or the vendored native code
when they determine documented behavior.

Identify the changed user-visible contracts: signatures, defaults, results,
validation stages, ownership, supported modes, or fallbacks. Distinguish Python
implementation from native availability and model requirements. An upstream
patch, enum declaration, or accepted parameter does not alone prove usable
support in this checkout or an installed package.

For a correction, verify the disputed behavior directly. Do not preserve an old
claim merely because other pages repeat it. Label implementation inferences and
their basis; omit unsupported claims.

## 2. Choose the page and structure

Use class or module pages for API contracts, feature guides for workflows, and
example pages for runnable starting points. Keep a module overview for related
classes; split out substantial public APIs or extension points when useful.
Link to shared explanations instead of duplicating them.

Lead with the task, recommended API, and important prerequisites. Choose headings
that fit the content; the schema requires relevant coverage, not identical
sections on every page. Omit empty deprecation or best-practice sections.

Use the [reference patterns](SCHEMA.md#reference-patterns-from-completed-pages)
for examples of clear structure, then verify behavioral details against current
source. Write in English unless requested otherwise.

## 3. Document the contract precisely

Keep defaults and positional-only or keyword-only markers in signatures. Use
parameter tables for types, defaults, units, sentinel values, and interactions.
Distinguish writable settings from read-only properties. Document private state
only when it explains user-visible behavior, and label it internal.

Explain what a fallback loses, when validation happens, and which layer imposes
a restriction. Avoid turning partial keyword support into a claim of complete
standards compliance. State whether a performance change affects setup,
conversion, or generation, and how long any cache lives.

Use frontmatter for new class and module references. Preserve existing metadata
conventions and update `last_updated` on reviewed, changed pages that have it.
Feature, example, and index pages do not need metadata added solely for uniformity.
See the [metadata policy](SCHEMA.md#metadata) for field and date semantics.

## 4. Make examples usable

Runnable examples need imports, setup, use, cleanup, and stated prerequisites.
Focused snippets may assume existing objects, but name those assumptions before
the code. Label signatures and conceptual sketches separately.

This standalone completion script requires an installed package with a working
native library and a compatible text-generation GGUF model at `./model.gguf`:

```python
from contextlib import closing

from llama_cpp import Llama

with closing(Llama(model_path="./model.gguf")) as llm:
    output = llm.create_completion(prompt="Hello,", max_tokens=32)
    print(output["choices"][0]["text"])
```

Generated text varies with the model and sampling settings. The example uses
`closing` to call `Llama.close()` when the block exits.

Use portable model paths and explain additional assets or settings where needed.
Use the lifecycle supported by the API; do not assume every object is a context
manager. Label platform-specific shell snippets accurately.

## 5. Update links and navigation

Use relative Markdown links, such as `[Llama Grammar](modules/LlamaGrammar.md)`
from this directory or `[Grammar guide](../features/grammar.md)` from a module
page. Check target files and anchors; avoid double-bracket wiki syntax and local
filesystem URLs.

Maintain the directory, page inventory, and completion status only in the
[index](index.md). Individual pages should link to relevant topics without
repeating the wiki directory or page list.

When a page becomes complete, update the index: add or revise its
navigation entry, adjust reading order when useful, update documentation status,
and remove the matching planned entry if that scope is complete. Do not advertise
empty placeholders as finished pages or add filler solely to make them nonempty.

## 6. Review and report

- [ ] Changed claims match the relevant Python and native sources.
- [ ] API signatures, defaults, returns, and examples agree.
- [ ] Ownership, prerequisites, errors, fallbacks, and limits are clear where relevant.
- [ ] Structure and metadata follow the schema without empty boilerplate.
- [ ] Runnable examples include setup and cleanup; partial snippets state assumptions.
- [ ] Markdown fences, relative links, anchors, and whitespace have been checked.
- [ ] Navigation reflects completed content, and unrelated edits are preserved.
- [ ] Validation reporting distinguishes static checks from executed examples.

Keep test links, local test paths, execution records, and machine-specific
benchmark tables out of wiki pages. Put relevant evidence in the review summary
or commit description. Describe performance scope without importing local
measurement logs. Do not claim model execution when only static checks ran.

## Commit messages

Use a concise English subject describing the documentation outcome. Add a body
when useful to explain scope, a corrected misconception, or validation.

```text
docs(wiki): clarify grammar ownership and conversion limitations
docs(wiki): refresh navigation for completed grammar guides
docs(wiki): align schema and contribution guidance with current conventions
```

Drafting a commit message does not require staging or committing files. Perform
repository actions only within the requested task scope.
