# LLM Wiki Schema – llama-cpp-python

**Schema Metadata**:

- **Author**: JamePeng
- **Maintainer**: Human and LLM-assisted documentation workflow
- **Project**: [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) wiki
- **Last Modified**: 2026-09-15
- **Version Target**: current checked-out source
- **Schema Version**: 0.5

## Purpose and scope

The wiki helps readers choose APIs, configure them correctly, understand their
behavior, and handle their limits. This schema defines page ownership, coverage,
and writing conventions. The [contribution guide](contributing-to-wiki.md)
describes the editing workflow; the [index](index.md) is the single place to
maintain the wiki directory, page inventory, navigation, and completion status.

Write in English, including example comments, unless requested otherwise. Lead
with what the reader can do and which API to use. Explain internal details when
they clarify configuration, ownership, performance, extension points, or debugging.

## Evidence and accuracy

Read the relevant checked-out source before changing behavioral claims. Start
with the public API and follow calls into helpers and native code as needed.

- Check the corresponding vendored implementation for native behavior. An
  upstream change does not establish that the checked-out vendor includes it.
- Distinguish Python support, native support, the loaded library build, and
  model or hardware requirements. A declared enum or accepted parameter alone
  does not establish that a working implementation is available.
- Verify signatures, defaults, return values, validation, and fallback paths.
  Explain when errors occur: construction, conversion, model attachment, or
  generation, as applicable.
- Bound support claims. Supporting some `allOf` combinations does not imply
  complete JSON Schema intersection or satisfiability checking.
- Explain what a fallback preserves or discards. A generic JSON grammar fallback
  must not be described as preserving the original schema constraints.
- Separate facts established by code from inference. Label inference and its
  basis; omit unsupported claims.
- Claim release availability only with evidence from the relevant release or
  tag. Source support does not guarantee support in an installed wheel.

For each limitation, explain the trigger, consequence, and supported alternative
when one exists. Distinguish unsupported behavior from deliberate design choices.

## Page ownership

Organize pages by the reader's task rather than mirroring source files. Consult
the index for page locations; do not duplicate directory trees or page inventories
in individual guides. Keep only cross-references relevant to the page's subject.

For a module with multiple classes, start with an overview. Split out a class
when it is a substantial public API, configuration surface, or extension point.
Keep small helpers on the module page. API references own signatures and
contracts; feature guides own workflows and cross-API choices. Link between them
instead of duplicating whole explanations.

## Metadata

Use YAML frontmatter for new class and module references. Preserve existing
metadata conventions; feature guides, examples, and navigation pages do not need
frontmatter added solely for uniformity.

```yaml
---
title: Llama Grammar
module_name: llama_cpp.llama_grammar
source_file: llama_cpp/llama_grammar.py
last_updated: YYYY-MM-DD
version_target: "latest"
---
```

Use `class_name` for a class or `module_name` for a module. Use `source_file` for
one primary source or `source_files` as a YAML list for multiple sources. These
are repository-relative metadata paths, not links. Update `last_updated` on
reviewed, changed pages that have it; do not refresh untouched pages. `latest`
means the checked-out source reviewed, not every released package. Use one clear
H1 for page content.

## Page coverage

These are coverage requirements where applicable, not mandatory heading names or
a fixed section order. Combine short sections and omit empty boilerplate. Add
deprecation and migration notes only when relevant.

### Class and module references

Open with purpose, public or internal status, the recommended entry point, and
its relationship to nearby APIs. Cover construction, important public state,
methods, lifecycle, errors, and related guides.

Preserve positional-only `/`, keyword-only `*`, and defaults in signatures.
Distinguish writable configuration, read-only properties, and per-request state.
Explain who owns native resources, whether definitions can be reused, and how
resources are closed where relevant. Do not infer thread safety from reusability.

| Table | Suggested columns |
|---|---|
| Parameters | Parameter, Type, Default, Description |
| Public attributes | Attribute, Type, Access or Source, Description |
| Errors | Condition, Exception or Outcome, Stage |
| Support | Option, Implementation, Status, Requirements |

Explain units, sentinel values, interacting options, ignored arguments, and
precedence. Do not list every private variable.

### Feature guides

Start with the task and a useful entry point or minimal example. Explain API
choices, prerequisites, configuration semantics, workflow, results, and limits.
Use a decision table when different APIs or modes solve different needs.

Attribute restrictions to the responsible layer: converter, sampler, chat
handler, native build, or model. For performance changes, identify the affected
phase and cache lifetime. Faster schema conversion does not establish faster
per-token inference.

### Runnable examples

State the goal and prerequisites before complete code. Include imports,
configuration, resource cleanup, and how to inspect results. Explain model
requirements, output formats, and relevant failure or cancellation behavior.
Label sample output as illustrative when generation can vary.

### Installation, diagnostics, types, and development

- Installation: identify the platform and shell, prerequisites, options, and
  verification steps. Distinguish updating source from rebuilding the native
  library actually loaded by Python.
- Diagnostics: organize by symptom with concrete checks and remedies; explain
  the effects of destructive recovery steps.
- Types: describe required fields, optional fields, variants, and where each
  structure is produced or consumed.
- Development: define scope, inputs, outputs, workflow, and action boundaries.
  The [commit generation guide](development/git-commit-generation-agent.md)
  illustrates separating drafting from repository actions.

## Reference patterns from completed pages

Borrow these structures, then recheck any behavioral details against the current
source. These examples are not permanent guarantees that every claim stays current.

| Page | Pattern to reuse |
|---|---|
| [Llama Grammar](modules/LlamaGrammar.md) | Separate reusable configuration from sampler state; explain validation stages and errors. |
| [Grammar guide](features/grammar.md) | Move from usage to conversion semantics, performance scope, and precise vendor differences. |
| [Speculative decoding](modules/LlamaSpeculative.md) | Introduce the public entry point, compare availability, then explain lifecycle and rollback. |
| [Embeddings and reranking](features/embeddings-rerank.md) | Map goals to APIs and pooling modes; state model requirements alongside recommendations. |
| [Audio TTS](examples/audio/audio-tts.md) | Put experimental status and prerequisites early; connect cleanup and output interpretation to complete code. |
| [Installation](install.md) | Separate platform commands and backend options; explain native rebuild requirements. |
| [Wiki index](index.md) | Offer task-based reading paths and separate completed pages from planned areas. |

## Examples and prose

Distinguish three kinds of code blocks:

- **Runnable example**: includes imports, setup, use, and cleanup; the user
  supplies only stated dependencies and model or input paths.
- **Focused snippet**: demonstrates one operation and explicitly names required
  existing objects, such as a configured `llm`; it is not a standalone script.
- **Signature or conceptual sketch**: clearly labeled reference or pseudocode,
  never described as executable validation.

Use portable placeholders such as `./model.gguf`, not contributor-specific
absolute paths. State model, tokenizer, chat format, multimodal asset, or backend
requirements where relevant. Use the cleanup mechanism actually supported by the
API. Label shell blocks appropriately for PowerShell, Bash, or another shell.

Write direct explanations: the action, its result, and the condition that changes
that result. Use tables for comparisons and lists for parallel choices or steps.
Avoid repeated warnings, empty headings, and broad claims such as “all models”
when only particular configurations are established.

## Links and navigation

Use standard relative Markdown links, resolved from the containing page:
`[Grammar guide](features/grammar.md)` at the wiki root, or
`[Llama Grammar](../modules/LlamaGrammar.md)` from a feature page. Avoid
double-bracket wiki links and contributor filesystem URLs. Link to source files
with repository-relative Markdown paths when useful.

Verify that linked pages and anchors exist. Do not link empty or planned files
as completed documentation. Empty files may remain placeholders until useful
content is ready; filler content is not a completion requirement.

When adding, renaming, or promoting a page, update the index's navigation and
status, and its reading order when helpful. Remove the matching planned entry
when that scope is complete. Align descriptions with actual coverage. Do not
duplicate directory listings or completion inventories in other pages.

## Validation and maintenance

Before finalizing a change:

1. Verify changed behavior against the relevant Python and native sources.
2. Check signatures, examples, ownership, fallbacks, and limitations within scope.
3. Check Markdown fences, relative links, anchors, metadata, and whitespace.
4. Check runnable snippets as appropriate. Syntax compilation, mocked checks,
   and real model execution establish different things; report what was done.
   A documentation-only edit does not automatically require model inference.
5. Synchronize affected navigation without rewriting unrelated pages or dates.

Keep test links, local test paths, execution records, and machine-specific
benchmark tables out of wiki pages. Describe user-facing behavior and performance
scope there; keep validation evidence and measured results in review notes or
commit descriptions when relevant. Never claim checks or execution that were
not performed.
