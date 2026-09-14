# Grammar definitions and sampling

`LlamaGrammar` stores a reusable GBNF definition. It does not own a native
sampler or token history. Each sampling context creates and releases its own
native grammar sampler, so the same definition can be reused for independent
generations. Native GBNF parsing happens when the sampler is initialized with
the model; construction only checks text inputs and options.

```python
from llama_cpp import LlamaGrammar

grammar = LlamaGrammar.from_string('answer ::= "yes" | "no"', root="answer")
result = llm.create_completion("Answer yes or no:", grammar=grammar)
```

`from_file(path, root=...)` also supports a custom start rule. Schema-generated
grammars use `root`. The `grammar` and `root` properties expose the definition.
Empty text and embedded NUL characters are rejected. The factory methods retain
`verbose` for compatibility; it does not enable native validation.

## JSON Schema conversion

```python
grammar = LlamaGrammar.from_json_schema({
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
})
```

Conversion uses the Python `SchemaConverter`, not the C++ common library.
Input dictionaries are copied recursively before resolving references.
Only a subset of JSON Schema is implemented; conversion is not a complete
JSON Schema validator.

An empty schema `{}` accepts any JSON value, including arrays and scalars.
Use `{"type": "object"}` to require an object. This also applies to nested
schemas such as `items: {}` and `properties: {"answer": {}}`.
The chat `json_object` response-format helper normalizes an empty schema to an
object. Missing, null, or empty tool parameters normalize to an object with no
declared properties, matching the vendor's empty-parameter handling.

## Lazy grammar

Pass trigger patterns or token IDs in the grammar definition, then enable lazy
sampling with the existing `grammar_lazy=True` completion option:

```python
grammar = LlamaGrammar.from_string(
    'answer ::= "yes" | "no"',
    root="answer",
    triggers=[r"[\s\S]*?(yes|no)"],
)
result = llm.create_completion(prompt, grammar=grammar, grammar_lazy=True)
```

String triggers are native regex patterns, not literal words. The native API
matches from the start of generated output and feeds the grammar from the
first capture group. Integer triggers include the triggering token in grammar
processing. Choose a trigger that captures the beginning of the GBNF content.
`from_file` and `from_json_schema` accept `triggers` as well. Trigger options
are stored as a tuple and are ignored when lazy sampling is disabled.
Lazy sampling without triggers raises `ValueError`, preventing a grammar that
never activates. Previously the main Python sampling context ignored lazy
options and initialized an eager sampler.

The internal `GrammarSampler` supports a context manager, idempotent `close`,
and rejects `apply`, `accept`, and `reset` after closure. `LlamaGrammar` itself
does not need `close` because it owns no native resources.

## Vendor compatibility audit

Compared against vendor commit `acecd56032ddc34bada14a2d978f110d9c987095`.
The former upstream Python conversion example was deleted in that commit;
ongoing references are `common/json-schema.cpp`,
`common/json-schema-to-grammar.cpp`, and their tests.

| Area | Python status |
| --- | --- |
| Empty Schema, including nested occurrences | Aligned: any JSON value |
| `minLength` / `maxLength` without type | Aligned: infer string |
| Explicit array type without `items` | Applies item-count constraints using any-value items |
| Explicit integer with fractional bounds | Rounds inward, including exclusive bounds |
| Numeric bounds without a type | Aligned: do not infer integer |
| Empty unions/enums and invalid counts | Rejected before emitting GBNF |
| Regex non-capturing groups | Supported, including nested groups |
| Regex character escapes | Supports digit, word, whitespace and complement classes, control and Unicode escapes |
| Reachable references | Schema-aware traversal, escaped JSON Pointers, root references and reference chains |
| Enum intersections | Compares JSON values structurally; supports objects, arrays and mixed types |
| Additional property names | Excludes decoded names, including equivalent JSON escapes and overlapping prefixes |
| Native grammar root and lazy initialization | Passed through the Python sampling context |
| Common C++ intermediate representation | Not ported; no public C binding is required |

Remaining differences and limitations to consider separately:

- Unsupported patterns: Python rejects unanchored patterns and unsupported
  constructs; the vendor converter can warn and fall back to unrestricted
  strings. Python keeps explicit failure rather than silently discarding the
  requested constraint. Some chat helpers still have their existing generic
  JSON fallback on conversion failure.
- `allOf`: Python supports enum/const intersections, optional type filtering,
  reference chains, nested intersections and compatible object-property merges.
  Booleans are distinct from numbers; numerically equal integers and floats
  compare equally. Empty intersections and unsupported combinations raise an
  error instead of producing an unrelated object grammar. General intersections
  of string, numeric and conflicting property constraints remain unsupported.
  Where possible, express these constraints in a single schema.
- References: Python retains opt-in HTTPS fetching (`allow_fetch=True`), while
  the vendor builder supports document-local references. Python resolves only
  reachable schema nodes and leaves const/enum payloads untouched. Recursive
  object schemas are supported; pure reference cycles and recursive `allOf`
  combinations are rejected. Named anchors and `$id` scope changes are not
  implemented.
- Regex patterns and property-key exclusions operate on Unicode scalar values
  and accept equivalent raw, short-escape and Unicode-escape representations.
  Supplementary characters support surrogate-pair escapes; unpaired surrogates
  are outside this subset. Backreferences, lookarounds and lazy/possessive regex
  quantifiers are rejected. Object property generation still follows the
  converter's declared ordering and additional-property policy.
- Grammar names and formatting can differ even when the accepted language is
  equivalent. Exact GBNF text equality is not a general compatibility test.
