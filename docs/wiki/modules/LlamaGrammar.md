---
title: Llama Grammar
module_name: llama_cpp.llama_grammar
source_file: llama_cpp/llama_grammar.py
last_updated: 2026-09-15
version_target: "latest"
---

# Llama Grammar

## Overview

`llama_cpp.llama_grammar` provides reusable grammar definitions, built-in GBNF
strings, and Python JSON Schema to GBNF conversion. For generation examples and
conversion behavior, see the [grammar guide](../features/grammar.md).

`LlamaGrammar` stores text and configuration. Each sampling context owns its own
native grammar sampler and token history. Constructing a definition validates
its inputs; the native sampler checks GBNF syntax and the existence of the
selected root when the grammar is attached to a model.

## Classes and constants

| Name | Role |
| --- | --- |
| `LlamaGrammar` | Public grammar definition created from text, a file, or JSON Schema. |
| `SchemaConverter` | Internal converter used by `json_schema_to_gbnf`; its methods are not a stable public API. |
| `BuiltinRule` | Internal container for primitive grammar text and dependencies. |
| `LLAMA_GRAMMAR_DEFAULT_ROOT` | Default start rule, `"root"`. |

Built-in grammar strings:

| Constant | Description |
| --- | --- |
| `JSON_GBNF` | JSON with an object at the root; nested values may have other JSON types. |
| `JSON_ARR_GBNF` | JSON arrays. |
| `ARITHMETIC_GBNF` | Simple arithmetic expressions. |
| `C_GBNF` | Example subset of C-like declarations and statements. |
| `LIST_GBNF` | Newline-separated Markdown-style list items. |
| `ENGLISH_GBNF` | Example English-character grammar; incomplete. |
| `CHESS_GBNF`, `JAPANESE_GBNF` | Currently JSON-like example grammars. |

Internal rule constants include `SPACE_RULE`, `PRIMITIVE_RULES`,
`STRING_FORMAT_RULES`, `RESERVED_NAMES`, `DOT`, and `DOTALL`.

## `LlamaGrammar`

### Constructor

Prefer the factory methods below for application code.

```python
def __init__(
    self,
    *args,
    _grammar: str,
    root: str = LLAMA_GRAMMAR_DEFAULT_ROOT,
    triggers: Optional[List[Union[str, int]]] = None,
    **kwargs,
)
```

`_grammar` is required. `root` and `triggers` are keyword-only. Legacy `*args`
and `**kwargs` are accepted but unused.

### Properties

| Property | Type | Meaning |
| --- | --- | --- |
| `grammar` | `str` | GBNF source text. |
| `root` | `str` | Native sampler start-rule name. |
| `triggers` | `Tuple[Union[str, int], ...]` | Snapshot of regex patterns and token IDs for lazy sampling. |

These properties have no setters. The trigger list is copied into a tuple, so
subsequent changes to the caller's list do not affect the definition. The
instance owns no native resources and requires no `close()` call.

### `from_string`

```python
@classmethod
def from_string(
    cls,
    grammar: str,
    verbose: bool = True,
    *,
    root: str = LLAMA_GRAMMAR_DEFAULT_ROOT,
    triggers: Optional[List[Union[str, int]]] = None,
) -> "LlamaGrammar"
```

Wraps GBNF text. `root` selects the start rule; its default is `"root"`.
`verbose` is retained for compatibility and currently has no effect.

```python
from llama_cpp import LlamaGrammar

grammar = LlamaGrammar.from_string('answer ::= "yes" | "no"', root="answer")
assert grammar.root == "answer"
```

### `from_file`

```python
@classmethod
def from_file(
    cls,
    file: Union[str, Path],
    verbose: bool = True,
    *,
    root: str = LLAMA_GRAMMAR_DEFAULT_ROOT,
    triggers: Optional[List[Union[str, int]]] = None,
) -> "LlamaGrammar"
```

Reads a non-empty UTF-8 GBNF file and applies the same input checks as
`from_string`. `root` and `triggers` have the same meaning in both methods.

```python
grammar = LlamaGrammar.from_file("./answer.gbnf", root="answer")
```

### `from_json_schema`

```python
@classmethod
def from_json_schema(
    cls,
    json_schema: Union[str, dict],
    prop_order: Optional[List[str]] = None,
    allow_fetch: bool = False,
    dotall: bool = False,
    raw_pattern: bool = False,
    verbose: bool = True,
    *,
    triggers: Optional[List[Union[str, int]]] = None,
) -> "LlamaGrammar"
```

Converts a JSON string or dictionary to GBNF. Generated grammars always use
`root`; this factory does not accept a custom root name. Input dictionaries
are copied recursively before references are resolved.

| Parameter | Meaning |
| --- | --- |
| `json_schema` | JSON Schema string or dictionary. |
| `prop_order` | Preferred property order. Required properties always precede optional properties; unspecified properties retain their input order. |
| `allow_fetch` | Enables HTTPS reference fetching; disabled by default. |
| `dotall` | Lets pattern dots match line terminators; disabled by default. |
| `raw_pattern` | Emits pattern content without JSON string quoting and escaping; intended for raw output. |
| `verbose` | Retained for compatibility; currently has no effect. |
| `triggers` | Lazy-sampling regex patterns or token IDs. |

Conversion and definition-validation failures are wrapped in `ValueError`.

```python
from llama_cpp import LlamaGrammar

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
}
grammar = LlamaGrammar.from_json_schema(schema, prop_order=["answer", "confidence"])
```

### Lazy sampling

Set `triggers` on a definition and pass `grammar_lazy=True` to generation:

```python
from llama_cpp import LlamaGrammar

# llm is an existing Llama model instance.
grammar = LlamaGrammar.from_string(
    'answer ::= "yes" | "no"',
    root="answer",
    triggers=[r"[\s\S]*?(yes|no)[\s\S]*"],
)
result = llm.create_completion(
    "Answer yes or no: is water wet?",
    grammar=grammar,
    grammar_lazy=True,
)
```

String triggers are native regex patterns, not literal words. Patterns match
the entire accumulated output; grammar processing starts at the first capture
group and includes the remaining buffered output. A token trigger includes the
triggering token in grammar processing. Select triggers whose captured content
can begin the grammar. Patterns must be valid for the native regex engine;
construction only checks their text, not regex syntax.

Triggers are ignored when lazy sampling is disabled. Lazy sampling without
triggers raises `ValueError` during sampler initialization.

## `json_schema_to_gbnf`

```python
def json_schema_to_gbnf(
    schema: Union[str, dict],
    prop_order: Optional[List[str]] = None,
    allow_fetch: bool = False,
    dotall: bool = False,
    raw_pattern: bool = False,
)
```

Returns the generated GBNF as a `str`. Conversion options match
`from_json_schema`; this function does not configure roots or lazy triggers.
A non-string, non-dictionary input raises `TypeError`. Other conversion errors
are propagated without the factory method's `ValueError` wrapper.

```python
from llama_cpp.llama_grammar import json_schema_to_gbnf

gbnf = json_schema_to_gbnf({
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 3,
})
```

## Conversion behavior

The converter implements a subset of JSON Schema, not a complete validator.

| Area | Behavior |
| --- | --- |
| Empty schema | `{}` accepts any JSON value, including nested occurrences. Use `{"type": "object"}` to require an object. |
| Objects | Supports properties, required fields, and additional properties. When properties are declared, explicitly set `additionalProperties: true` to allow undeclared keys. |
| Property names | Additional keys cannot reuse declared names, including equivalent JSON escape spellings. |
| Arrays | Supports item schemas, tuple-like `prefixItems` or array-valued `items`, and item-count bounds for homogeneous arrays. `items` takes precedence over `prefixItems`. |
| Strings | Supports length bounds, anchored regex patterns, and selected formats: date, time, date-time, and uuid/uuid1 through uuid5. |
| Type inference | String length keywords can imply string type. Numeric bounds alone do not imply integer type. |
| Integer bounds | Supports inclusive/exclusive bounds and rounds fractional bounds inward. |
| Alternatives | `anyOf` and `oneOf` generate alternatives; `oneOf` does not enforce exclusive matching. |
| Enum/const | Supports JSON values; `allOf` intersections compare structures and distinguish booleans from numbers. |
| References | Traverses reachable schemas, supports reference chains, escaped JSON Pointers and root references, and preserves literal const/enum data. |

Compatible object merges and enum/const intersections are supported in `allOf`,
including nested intersections and optional type filtering. Empty intersections,
conflicting property constraints, and unsupported general intersections raise
errors. Pure reference cycles and recursive `allOf` are rejected; recursive
object schemas can be represented.

HTTPS references require `allow_fetch=True`. Named anchors and `$id` scope
changes are not implemented. Regex backreferences, lookarounds, and lazy or
possessive quantifiers are rejected. Regex character handling and additional-key
exclusion use Unicode scalar values, including escaped surrogate pairs, but not
unpaired surrogates. Unsupported formats such as `uri` and `email` do not add
format-specific validation.

## Internal conversion and reuse

Each `json_schema_to_gbnf` call creates a new `SchemaConverter`. Its internal
state includes generated rules (`_rules`), resolved references (`_refs`), unique
reference rule names (`_ref_rule_names`), and character-rule caches.

Optional property suffixes are built once from right to left, avoiding repeated
recursive traversal and list slicing. Rule registration preserves existing
names and empty reference placeholders while using single dictionary lookups.

`_character_input_rules` bypasses normalization for repeated character ranges;
`_character_rules` reuses rules for equivalent normalized ranges; `_hex_intervals`
reuses hexadecimal grammar fragments. These caches belong to one converter and
are discarded with it. Native samplers and token histories are never cached
here. Reusing a `LlamaGrammar` definition across requests avoids repeating Schema
to GBNF conversion; it does not reuse a previous request's native sampler state.

These optimizations reduce conversion work without changing property order,
accepted output, or the public API. They do not change per-token sampling.

## Errors and lifecycle

| Entry point | Error | Condition |
| --- | --- | --- |
| Text construction | `TypeError` | Grammar is not a string. |
| Text construction | `ValueError` | Empty/blank grammar, embedded NUL, or invalid root/trigger inputs. |
| `from_file` | `FileNotFoundError` | File does not exist. |
| `from_file` | `IOError` | File cannot be read as UTF-8. |
| `from_json_schema` | `ValueError` | Conversion or definition validation fails. |
| `json_schema_to_gbnf` | `TypeError` | Input is neither a string nor a dictionary. |
| Native sampler initialization | `RuntimeError` | Native GBNF parsing fails or the start rule is missing. |

Root names must match `[A-Za-z0-9-]+`. Trigger strings must be non-empty and
contain no NUL. Token IDs must be integers in `[0, 2**31)`; booleans are rejected.
The model vocabulary determines whether a token ID is meaningful.

The internal `GrammarSampler` supports a context manager and idempotent `close`.
Its `apply`, `accept`, and `reset` methods reject use after closure. Applications
normally let `Llama` manage these resources.

## Related links

- [Grammar guide](../features/grammar.md)
- [Llama core](../core/Llama.md)
- [Wiki index](../index.md)
