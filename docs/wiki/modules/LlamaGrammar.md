---
title: Llama Grammar
module_name: llama_cpp.llama_grammar
source_file: llama_cpp/llama_grammar.py
last_updated: 2026-05-03
version_target: "latest"
---

# Llama Grammar

## Overview

`llama_grammar.py` provides grammar utilities for constrained generation in `llama-cpp-python`.

The module defines the `LlamaGrammar` class, a collection of built-in GBNF grammar strings, and a JSON Schema to GBNF converter based on the upstream `llama.cpp` grammar tooling.

Use this module when you need to guide model output toward a specific grammar, such as JSON, JSON arrays, lists, arithmetic expressions, or custom GBNF rules.

## Role in the Library

`LlamaGrammar` acts as a lightweight wrapper around a GBNF grammar string.

The module also includes helper logic for converting JSON Schema definitions into GBNF grammar text. This allows users to define structured output constraints using JSON Schema-like input and convert it into a grammar format usable by llama.cpp-style constrained generation.

## Important Classes

| Class | Status | Description |
|---|---|---|
| `LlamaGrammar` | public | Main wrapper class for grammar strings. Supports creation from raw strings, files, and JSON Schema. |
| `BuiltinRule` | internal helper | Small container used by the JSON Schema converter to store built-in grammar rule content and dependencies. |
| `SchemaConverter` | internal implementation | Converts JSON Schema structures into GBNF grammar rules. Used by `json_schema_to_gbnf`. |

## Constants

### Default Root

| Constant | Type | Value | Description |
|---|---|---|---|
| `LLAMA_GRAMMAR_DEFAULT_ROOT` | `str` | `"root"` | Default root rule name used by `LlamaGrammar`. |

### Built-in GBNF Grammars

The module includes several built-in GBNF grammar strings.

| Constant | Description |
|---|---|
| `ARITHMETIC_GBNF` | Grammar for simple arithmetic-like expressions. |
| `C_GBNF` | Example grammar for a subset of C-like declarations and statements. |
| `CHESS_GBNF` | JSON-like grammar currently defined similarly to object/array/value grammar. |
| `ENGLISH_GBNF` | Simple English-character grammar. The source notes that it may be incomplete and mostly serves as an example. |
| `JAPANESE_GBNF` | JSON-like grammar currently defined similarly to object/array/value grammar. |
| `JSON_ARR_GBNF` | Grammar for generating JSON arrays. |
| `JSON_GBNF` | Grammar for JSON objects and values. |
| `LIST_GBNF` | Grammar for newline-separated Markdown-style list items. |

### JSON Schema Conversion Rules

The module also defines internal constants used by `SchemaConverter`.

| Constant | Description |
|---|---|
| `SPACE_RULE` | Shared grammar rule for constrained whitespace. |
| `PRIMITIVE_RULES` | Built-in grammar rules for primitive schema types such as boolean, number, integer, object, array, string, and null. |
| `STRING_FORMAT_RULES` | Built-in grammar rules for selected string formats such as date, time, and date-time. |
| `RESERVED_NAMES` | Rule names reserved by the converter. |
| `DOTALL` | Pattern rule matching any Unicode code point. |
| `DOT` | Pattern rule matching any character except line breaks. |

## `LlamaGrammar`

```python
class LlamaGrammar
````

Main wrapper for GBNF grammar text.

### Constructor

```python
def __init__(self, *args, _grammar: str, **kwargs)
```

| Parameter  | Type     | Default  | Description                                                                      |
| ---------- | -------- | -------- | -------------------------------------------------------------------------------- |
| `*args`    | variadic | none     | Accepted by the constructor but not used directly in the current implementation. |
| `_grammar` | `str`    | required | Grammar string stored by the instance.                                           |
| `**kwargs` | variadic | none     | Accepted by the constructor but not used directly in the current implementation. |

### Important Attributes / State

| Attribute  | Type           | Source                          | Description                                             |
| ---------- | -------------- | ------------------------------- | ------------------------------------------------------- |
| `_grammar` | `str`          | `_grammar` constructor argument | Internal grammar string stored by the instance.         |
| `_root`    | `str`          | `LLAMA_GRAMMAR_DEFAULT_ROOT`    | Internal root rule name. Defaults to `"root"`.          |
| `grammar`  | `str` property | `_grammar`                      | Read-only property returning the stored grammar string. |

## Class Methods

### `from_string`

```python
@classmethod
def from_string(
    cls,
    grammar: str,
    verbose: bool = True,
) -> "LlamaGrammar"
```

Creates a `LlamaGrammar` instance from a raw GBNF grammar string.

| Parameter | Type   | Default  | Description                                                                                       |
| --------- | ------ | -------- | ------------------------------------------------------------------------------------------------- |
| `grammar` | `str`  | required | Raw GBNF grammar string.                                                                          |
| `verbose` | `bool` | `True`   | Accepted by the method. The current implementation forwards no logging behavior from this method. |

Returns:

| Type           | Description                                              |
| -------------- | -------------------------------------------------------- |
| `LlamaGrammar` | Grammar instance containing the provided grammar string. |

#### Example

```python
from llama_cpp.llama_grammar import LlamaGrammar, JSON_GBNF

grammar = LlamaGrammar.from_string(JSON_GBNF)

print(grammar.grammar)
```

### `from_file`

```python
@classmethod
def from_file(
    cls,
    file: Union[str, Path],
    verbose: bool = True,
) -> "LlamaGrammar"
```

Creates a `LlamaGrammar` instance from a UTF-8 grammar file.

| Parameter | Type               | Default  | Description              |
| --------- | ------------------ | -------- | ------------------------ |
| `file`    | `Union[str, Path]` | required | Path to a grammar file.  |
| `verbose` | `bool`             | `True`   | Passed to `from_string`. |

Behavior based on the current implementation:

* Raises `FileNotFoundError` if the file does not exist.
* Raises `IOError` if reading the file fails.
* Raises `ValueError` if the grammar file is empty.
* Reads the file using UTF-8 encoding.

#### Example

```python
from llama_cpp.llama_grammar import LlamaGrammar

grammar = LlamaGrammar.from_file("./json.gbnf")

print(grammar.grammar)
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
) -> "LlamaGrammar"
```

Creates a `LlamaGrammar` instance by converting a JSON Schema string or dictionary into GBNF grammar.

| Parameter     | Type                  | Default  | Description                                                                                              |
| ------------- | --------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `json_schema` | `Union[str, dict]`    | required | JSON Schema input as a JSON string or Python dictionary.                                                 |
| `prop_order`  | `Optional[List[str]]` | `None`   | Optional property order. The source comment notes this can help improve stability for small models.      |
| `allow_fetch` | `bool`                | `False`  | Allows remote schema fetching for HTTPS `$ref` values when enabled.                                      |
| `dotall`      | `bool`                | `False`  | Controls whether pattern `.` should match all Unicode code points during regex-to-grammar conversion.    |
| `raw_pattern` | `bool`                | `False`  | Controls whether regex patterns are converted as raw grammar patterns instead of quoted string patterns. |
| `verbose`     | `bool`                | `True`   | Passed to `from_string`.                                                                                 |

Returns:

| Type           | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `LlamaGrammar` | Grammar instance containing the generated GBNF grammar string. |

If conversion fails, the method raises `ValueError`.

#### Example

```python
from llama_cpp.llama_grammar import LlamaGrammar

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}

grammar = LlamaGrammar.from_json_schema(schema)

print(grammar.grammar)
```

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

Converts a JSON Schema string or dictionary into a GBNF grammar string.

| Parameter     | Type                  | Default  | Description                                                                                         |
| ------------- | --------------------- | -------- | --------------------------------------------------------------------------------------------------- |
| `schema`      | `Union[str, dict]`    | required | JSON Schema input. Strings are parsed with `json.loads`; dictionaries are copied before conversion. |
| `prop_order`  | `Optional[List[str]]` | `None`   | Optional property ordering used by object rule generation.                                          |
| `allow_fetch` | `bool`                | `False`  | Allows remote HTTPS `$ref` fetching when enabled.                                                   |
| `dotall`      | `bool`                | `False`  | Controls regex dot behavior during pattern conversion.                                              |
| `raw_pattern` | `bool`                | `False`  | Controls how regex pattern rules are emitted.                                                       |

Returns:

| Type  | Description                    |
| ----- | ------------------------------ |
| `str` | Generated GBNF grammar string. |

The function raises `TypeError` if `schema` is neither a JSON string nor a dictionary.

### Example

```python
from llama_cpp.llama_grammar import json_schema_to_gbnf

schema = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 3,
}

gbnf = json_schema_to_gbnf(schema)

print(gbnf)
```

## `SchemaConverter`

```python
class SchemaConverter
```

Internal implementation class used by `json_schema_to_gbnf`.

`SchemaConverter` walks a JSON Schema dictionary, resolves references, builds grammar rules, and formats them into GBNF.

This class is useful for understanding how conversion works, but most users should use `LlamaGrammar.from_json_schema` or `json_schema_to_gbnf` instead.

> Warning: `SchemaConverter` appears to be an implementation detail. It should not be treated as the primary public API unless the project explicitly documents it as stable.

### Constructor

```python
def __init__(
    self,
    *,
    prop_order,
    allow_fetch,
    dotall,
    raw_pattern,
)
```

| Parameter     | Type         | Description                                                             |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| `prop_order`  | mapping-like | Property ordering map used when generating object rules.                |
| `allow_fetch` | `bool`       | Enables or disables remote schema fetching for supported `$ref` values. |
| `dotall`      | `bool`       | Controls regex dot behavior.                                            |
| `raw_pattern` | `bool`       | Controls raw pattern handling.                                          |

### Important Internal State

| Attribute              | Type         | Description                                      |
| ---------------------- | ------------ | ------------------------------------------------ |
| `_prop_order`          | mapping-like | Stores property ordering preferences.            |
| `_allow_fetch`         | `bool`       | Stores whether remote references may be fetched. |
| `_dotall`              | `bool`       | Stores regex dot behavior.                       |
| `_raw_pattern`         | `bool`       | Stores raw pattern handling behavior.            |
| `_rules`               | `dict`       | Accumulates generated grammar rules.             |
| `_refs`                | `dict`       | Stores resolved JSON Schema references.          |
| `_refs_being_resolved` | `set`        | Tracks references currently being resolved.      |

### Key Methods

| Method               | Description                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `resolve_refs`       | Resolves local and supported HTTPS `$ref` references in a schema.                        |
| `visit`              | Main schema visitor that generates grammar rules based on schema structure.              |
| `format_grammar`     | Formats generated rules into a GBNF grammar string.                                      |
| `_build_object_rule` | Builds object grammar rules from properties, required fields, and additional properties. |
| `_visit_pattern`     | Converts supported regex patterns into GBNF rules.                                       |
| `_add_rule`          | Adds or reuses a grammar rule name.                                                      |
| `_add_primitive`     | Adds primitive rules and their dependencies.                                             |

## Supported JSON Schema Features

Based on the current implementation, the converter includes handling for:

* `type`
* `properties`
* `required`
* `additionalProperties`
* `$ref`
* `oneOf`
* `anyOf`
* `allOf`
* `const`
* `enum`
* `items`
* `prefixItems`
* `minItems`
* `maxItems`
* `pattern`
* `format`
* `minLength`
* `maxLength`
* integer bounds:

  * `minimum`
  * `exclusiveMinimum`
  * `maximum`
  * `exclusiveMaximum`

String formats handled by built-in rules include:

* `date`
* `time`
* `date-time`
* UUID-like formats matching `uuid`, `uuid1`, `uuid2`, `uuid3`, `uuid4`, or `uuid5`

The source includes a TODO comment for unsupported string formats such as `uri` and `email`.

## Error Handling

| API                             | Error               | Condition                                 |
| ------------------------------- | ------------------- | ----------------------------------------- |
| `LlamaGrammar.from_file`        | `FileNotFoundError` | Grammar file path does not exist.         |
| `LlamaGrammar.from_file`        | `IOError`           | Grammar file cannot be read.              |
| `LlamaGrammar.from_file`        | `ValueError`        | Grammar file is empty.                    |
| `LlamaGrammar.from_json_schema` | `ValueError`        | JSON Schema to GBNF conversion fails.     |
| `json_schema_to_gbnf`           | `TypeError`         | Schema input is neither `str` nor `dict`. |

## Common Usage

### Use a Built-in Grammar

```python
from llama_cpp.llama_grammar import LlamaGrammar, JSON_GBNF

grammar = LlamaGrammar.from_string(JSON_GBNF)

print(grammar.grammar)
```

### Load Grammar from a File

```python
from llama_cpp.llama_grammar import LlamaGrammar

grammar = LlamaGrammar.from_file("./grammar.gbnf")

print(grammar.grammar)
```

### Convert JSON Schema to Grammar

```python
from llama_cpp.llama_grammar import LlamaGrammar

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
}

grammar = LlamaGrammar.from_json_schema(
    schema,
    prop_order=["answer", "confidence"],
)

print(grammar.grammar)
```

### Convert JSON Schema Directly to GBNF

```python
from llama_cpp.llama_grammar import json_schema_to_gbnf

schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
}

gbnf = json_schema_to_gbnf(schema)

print(gbnf)
```

## Best Practices & Common Patterns

* Use `LlamaGrammar.from_string` when you already have a GBNF grammar string.
* Use `LlamaGrammar.from_file` when storing grammar definitions in `.gbnf` files.
* Use `LlamaGrammar.from_json_schema` when generating grammars from JSON Schema input.
* Use `json_schema_to_gbnf` directly when you only need the generated grammar string.
* Keep JSON Schemas small and explicit when targeting constrained generation.
* Use `prop_order` when output field order matters for stability.
* Keep `allow_fetch=False` unless remote `$ref` fetching is explicitly needed.
* Prefer public helpers over using `SchemaConverter` directly.
* Do not rely on internal converter methods as stable public APIs.

## Limitations

* `SchemaConverter` is implementation-oriented and may change.
* Remote `$ref` fetching is only attempted for HTTPS references and requires `allow_fetch=True`.
* The source includes TODO notes for unsupported string formats such as `uri` and `email`.
* Regex pattern conversion explicitly rejects unsupported pattern syntax such as lookaheads and non-greedy modifiers.
* The exact runtime integration between `LlamaGrammar` and model generation should be verified from the relevant generation APIs before documenting end-to-end constrained generation behavior.

## Related Links

* [[Index-Home](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)]
* [[Llama Core](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)]
