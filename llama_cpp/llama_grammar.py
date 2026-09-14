"""GBNF definitions and JSON Schema conversion.

Reference: vendor/llama.cpp/common/json-schema{,-to-grammar}.cpp.
Native grammar parsing and sampling are owned by the sampling context.
"""

# flake8: noqa
from pathlib import Path
import copy
import math
import json
import re
import sys
from typing import (
    Any,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

LLAMA_GRAMMAR_DEFAULT_ROOT = "root"


class LlamaGrammar:
    """Reusable grammar definition; each sampling context owns its native state.

    Construction checks text inputs, not GBNF syntax. The native sampler parses
    the grammar when it is attached to a model. ``verbose`` is retained for API
    compatibility on the factory methods.
    """

    def __init__(self, *args, _grammar: str, root: str = LLAMA_GRAMMAR_DEFAULT_ROOT, triggers: Optional[List[Union[str, int]]] = None, **kwargs):
        if not isinstance(_grammar, str):
            raise TypeError("grammar must be a string")
        # Native APIs receive NUL-terminated strings.
        if not _grammar.strip() or '\x00' in _grammar:
            raise ValueError("grammar must be non-empty and contain no NUL characters")
        if not isinstance(root, str) or not re.fullmatch(r'[A-Za-z0-9-]+', root):
            raise ValueError("root must be a non-empty GBNF rule name")
        self._grammar = _grammar
        self._root = root
        # Snapshot triggers so caller edits cannot change this definition.
        self._triggers = tuple(triggers or ())
        for trigger in self._triggers:
            if isinstance(trigger, str):
                if not trigger or '\x00' in trigger:
                    raise ValueError("trigger patterns must be non-empty and contain no NUL characters")
            elif type(trigger) is not int or not 0 <= trigger < 2**31:
                raise ValueError("triggers must be regex strings or non-negative int32 token IDs")

    @property
    def triggers(self) -> Tuple[Union[str, int], ...]:
        """Regex patterns and token IDs used when sampling with grammar_lazy=True."""
        return self._triggers

    @property
    def root(self) -> str:
        """Start rule passed to the native grammar sampler."""
        return self._root

    @property
    def grammar(self) -> str:
        """GBNF source text, without model-specific sampling state."""
        return self._grammar

    @classmethod
    def from_string(cls, grammar: str, verbose: bool = True, *, root: str = LLAMA_GRAMMAR_DEFAULT_ROOT, triggers: Optional[List[Union[str, int]]] = None) -> "LlamaGrammar":
        """Wrap GBNF text; syntax is checked during native sampler initialization."""
        return cls(_grammar=grammar, root=root, triggers=triggers)

    @classmethod
    def from_file(cls, file: Union[str, Path], verbose: bool = True, *, root: str = LLAMA_GRAMMAR_DEFAULT_ROOT, triggers: Optional[List[Union[str, int]]] = None) -> "LlamaGrammar":
        """Load a non-empty UTF-8 GBNF file."""
        file_path = Path(file)

        if not file_path.exists():
            raise FileNotFoundError(f"{cls.__name__}.from_file: file not found: {file_path}")

        try:
            grammar_content = file_path.read_text(encoding='utf-8')
        except Exception as err:
            raise IOError(f"{cls.__name__}.from_file: error reading grammar file: {err}")

        if not grammar_content.strip():
            raise ValueError(f"{cls.__name__}.from_file: grammar file is empty")

        return cls.from_string(grammar_content, verbose=verbose, root=root, triggers=triggers)

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
    ) -> "LlamaGrammar":
        """
        Convert a JSON Schema to GBNF with the default root rule.

        json_schema: A JSON Schema string or dictionary.
        prop_order: Preferred property order; required properties come first.
        allow_fetch: Allow HTTPS references during schema conversion.
        dotall: Let regex dots match line breaks.
        raw_pattern: Emit patterns without JSON string quoting.
        verbose: Retained for API compatibility; currently has no effect.
        triggers: Regex patterns or token IDs for lazy sampling.
        """
        try:
            gbnf_grammar_str = json_schema_to_gbnf(
                json_schema,
                prop_order=prop_order,
                allow_fetch=allow_fetch,
                dotall=dotall,
                raw_pattern=raw_pattern,
            )
            return cls.from_string(gbnf_grammar_str, verbose=verbose, triggers=triggers)
        except Exception as e:
            raise ValueError(f"{cls.__name__}.from_json_schema: conversion failed: {e}")


"""llama.cpp gbnf rules from vendor/llama.cpp/grammars"""

ARITHMETIC_GBNF = r"""
root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
"""

C_GBNF = r"""
root ::= (declaration)*

declaration ::= dataType identifier "(" parameter? ")" "{" statement* "}"

dataType  ::= "int" ws | "float" ws | "char" ws
identifier ::= [a-zA-Z_] [a-zA-Z_0-9]*

parameter ::= dataType identifier

statement ::=
    ( dataType identifier ws "=" ws expression ";" ) |
    ( identifier ws "=" ws expression ";" ) |
    ( identifier ws "(" argList? ")" ";" ) |
    ( "return" ws expression ";" ) |
    ( "while" "(" condition ")" "{" statement* "}" ) |
    ( "for" "(" forInit ";" ws condition ";" ws forUpdate ")" "{" statement* "}" ) |
    ( "if" "(" condition ")" "{" statement* "}" ("else" "{" statement* "}")? ) |
    ( singleLineComment ) |
    ( multiLineComment )

forInit ::= dataType identifier ws "=" ws expression | identifier ws "=" ws expression
forUpdate ::= identifier ws "=" ws expression

condition ::= expression relationOperator expression
relationOperator ::= ("<=" | "<" | "==" | "!=" | ">=" | ">")

expression ::= term (("+" | "-") term)*
term ::= factor(("*" | "/") factor)*

factor ::= identifier | number | unaryTerm | funcCall | parenExpression
unaryTerm ::= "-" factor
funcCall ::= identifier "(" argList? ")"
parenExpression ::= "(" ws expression ws ")"

argList ::= expression ("," ws expression)*

number ::= [0-9]+

singleLineComment ::= "//" [^\n]* "\n"
multiLineComment ::= "/*" ( [^*] | ("*" [^/]) )* "*/"

ws ::= ([ \t\n]+)
"""

CHESS_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""

ENGLISH_GBNF = r"""
# note: this might be incomplete, mostly an example
root        ::= en-char+ ([ \t\n] en-char+)*
en-char     ::= letter | digit | punctuation
letter      ::= [a-zA-Z]
digit       ::= [0-9]
punctuation ::= [!"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~]
"""

JAPANESE_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""

JSON_ARR_GBNF = r"""
# This is the same as json.gbnf but we restrict whitespaces at the end of the root array
# Useful for generating JSON arrays

root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws

arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""


JSON_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
"""

LIST_GBNF = r"""
root ::= item+

# Excludes various line break characters
item ::= "- " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"
"""

"""llama.cpp json-schema to grammar converter from vendor/llama.cpp/examples/json-schema-to-grammar.py"""

def _build_repetition(item_rule, min_items, max_items, separator_rule=None):

    if max_items == 0:
        return ""

    if min_items == 0 and max_items == 1:
        return f'{item_rule}?'

    if not separator_rule:
        if min_items == 1 and max_items is None:
            return f'{item_rule}+'
        elif min_items == 0 and max_items is None:
            return f'{item_rule}*'
        else:
            return f'{item_rule}{{{min_items},{max_items if max_items is not None else ""}}}'

    result = item_rule + ' ' + _build_repetition(f'({separator_rule} {item_rule})', min_items - 1 if min_items > 0 else 0, max_items - 1 if max_items is not None else None)
    return f'({result})?' if min_items == 0 else result

def _generate_min_max_int(min_value: Optional[int], max_value: Optional[int], out: list, decimals_left: int = 16, top_level: bool = True):
    def digit_range(from_char: str, to_char: str):
        out.append("[")
        if from_char == to_char:
            out.append(from_char)
        else:
            out.append(from_char)
            out.append("-")
            out.append(to_char)
        out.append("]")

    def more_digits(min_digits: int, max_digits: int):
        out.append("[0-9]")
        if min_digits == max_digits and min_digits == 1:
            return
        out.append("{")
        out.append(str(min_digits))
        if max_digits != min_digits:
            out.append(",")
            if max_digits != sys.maxsize:
                out.append(str(max_digits))
        out.append("}")

    def uniform_range(from_str: str, to_str: str):
        i = 0
        while i < len(from_str) and from_str[i] == to_str[i]:
            i += 1
        if i > 0:
            out.append("\"")
            out.append(from_str[:i])
            out.append("\"")
        if i < len(from_str):
            if i > 0:
                out.append(" ")
            sub_len = len(from_str) - i - 1
            if sub_len > 0:
                from_sub = from_str[i+1:]
                to_sub = to_str[i+1:]
                sub_zeros = "0" * sub_len
                sub_nines = "9" * sub_len

                to_reached = False
                out.append("(")
                if from_sub == sub_zeros:
                    digit_range(from_str[i], chr(ord(to_str[i]) - 1))
                    out.append(" ")
                    more_digits(sub_len, sub_len)
                else:
                    out.append("[")
                    out.append(from_str[i])
                    out.append("] ")
                    out.append("(")
                    uniform_range(from_sub, sub_nines)
                    out.append(")")
                    if ord(from_str[i]) < ord(to_str[i]) - 1:
                        out.append(" | ")
                        if to_sub == sub_nines:
                            digit_range(chr(ord(from_str[i]) + 1), to_str[i])
                            to_reached = True
                        else:
                            digit_range(chr(ord(from_str[i]) + 1), chr(ord(to_str[i]) - 1))
                        out.append(" ")
                        more_digits(sub_len, sub_len)
                if not to_reached:
                    out.append(" | ")
                    digit_range(to_str[i], to_str[i])
                    out.append(" ")
                    uniform_range(sub_zeros, to_sub)
                out.append(")")
            else:
                out.append("[")
                out.append(from_str[i])
                out.append("-")
                out.append(to_str[i])
                out.append("]")

    if min_value is not None and max_value is not None:
        if min_value < 0 and max_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(-max_value, -min_value, out, decimals_left, top_level=True)
            out.append(")")
            return

        if min_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(0, -min_value, out, decimals_left, top_level=True)
            out.append(") | ")
            min_value = 0

        min_s = str(min_value)
        max_s = str(max_value)
        min_digits = len(min_s)
        max_digits = len(max_s)

        for digits in range(min_digits, max_digits):
            uniform_range(min_s, "9" * digits)
            min_s = "1" + "0" * digits
            out.append(" | ")
        uniform_range(min_s, max_s)
        return

    less_decimals = max(decimals_left - 1, 1)

    if min_value is not None:
        if min_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(None, -min_value, out, decimals_left, top_level=False)
            out.append(") | [0] | [1-9] ")
            more_digits(0, decimals_left - 1)
        elif min_value == 0:
            if top_level:
                out.append("[0] | [1-9] ")
                more_digits(0, less_decimals)
            else:
                more_digits(1, decimals_left)
        elif min_value <= 9:
            c = str(min_value)
            range_start = '1' if top_level else '0'
            if c > range_start:
                digit_range(range_start, chr(ord(c) - 1))
                out.append(" ")
                more_digits(1, less_decimals)
                out.append(" | ")
            digit_range(c, "9")
            out.append(" ")
            more_digits(0, less_decimals)
        else:
            min_s = str(min_value)
            length = len(min_s)
            c = min_s[0]

            if c > "1":
                digit_range("1" if top_level else "0", chr(ord(c) - 1))
                out.append(" ")
                more_digits(length, less_decimals)
                out.append(" | ")
            digit_range(c, c)
            out.append(" (")
            _generate_min_max_int(int(min_s[1:]), None, out, less_decimals, top_level=False)
            out.append(")")
            if c < "9":
                out.append(" | ")
                digit_range(chr(ord(c) + 1), "9")
                out.append(" ")
                more_digits(length - 1, less_decimals)
        return

    if max_value is not None:
        if max_value >= 0:
            if top_level:
                out.append("\"-\" [1-9] ")
                more_digits(0, less_decimals)
                out.append(" | ")
            _generate_min_max_int(0, max_value, out, decimals_left, top_level=True)
        else:
            out.append("\"-\" (")
            _generate_min_max_int(-max_value, None, out, decimals_left, top_level=False)
            out.append(")")
        return

    raise RuntimeError("At least one of min_value or max_value must be set")


class BuiltinRule:
    def __init__(self, content: str, deps: list = None):
        self.content = content
        self.deps = deps or []

# Constraining spaces to prevent model "running away".
SPACE_RULE = '| " " | "\\n"{1,2} [ \\t]{0,20}'

PRIMITIVE_RULES = {
    'boolean'      : BuiltinRule('("true" | "false")', []),
    'decimal-part' : BuiltinRule('[0-9]{1,16}', []),
    'integral-part': BuiltinRule('[0] | [1-9] [0-9]{0,15}', []),
    'number'       : BuiltinRule('("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?', ['integral-part', 'decimal-part']),
    'integer'      : BuiltinRule('("-"? integral-part)', ['integral-part']),
    'value'        : BuiltinRule('object | array | string | number | boolean | null', ['object', 'array', 'string', 'number', 'boolean', 'null']),
    'object'       : BuiltinRule('"{" space ( string ":" space value ("," space string ":" space value)* )? space "}"', ['string', 'value']),
    'array'        : BuiltinRule('"[" space ( value ("," space value)* )? space "]"', ['value']),
    'uuid'         : BuiltinRule(r'"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""', []),
    'char'         : BuiltinRule(r'[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})', []),
    'string'       : BuiltinRule(r'"\"" char* "\""', ['char']),
    'null'         : BuiltinRule('"null"', []),
}

# TODO: support "uri", "email" string formats
STRING_FORMAT_RULES = {
    'date'            : BuiltinRule('[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( \"0\" [1-9] | [1-2] [0-9] | "3" [0-1] )', []),
    'time'            : BuiltinRule('([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )', []),
    'date-time'       : BuiltinRule('date "T" time', ['date', 'time']),
    'date-string'     : BuiltinRule('"\\"" date "\\""', ['date']),
    'time-string'     : BuiltinRule('"\\"" time "\\""', ['time']),
    'date-time-string': BuiltinRule('"\\"" date-time "\\""', ['date-time']),
}

DOTALL = '[\\U00000000-\\U0010FFFF]'
DOT = '[^\\x0A\\x0D]'

RESERVED_NAMES = set(["root", "dot", *PRIMITIVE_RULES.keys(), *STRING_FORMAT_RULES.keys()])

INVALID_RULE_CHARS_RE = re.compile(r'[^a-zA-Z0-9-]+')
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"\\]')
GRAMMAR_RANGE_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"\]\-\\]')
GRAMMAR_LITERAL_ESCAPES = {'\r': '\\r', '\n': '\\n', '"': '\\"', '-': '\\-', ']': '\\]', '\\': '\\\\'}

_MISSING = object()

class SchemaConverter:
    def __init__(self, *, prop_order, allow_fetch, dotall, raw_pattern):
        self._prop_order = prop_order
        self._allow_fetch = allow_fetch
        self._dotall = dotall
        self._raw_pattern = raw_pattern
        self._rules = {
            'space': SPACE_RULE,
        }
        self._refs = {}
        self._ref_rule_names = {}
        self._character_rules = {}
        # Caches live only as long as this schema conversion.
        self._character_input_rules = {}
        self._hex_intervals = {}

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)) or m.group(0), literal
        )
        return f'"{escaped}"'

    def not_literal(self, literal: str, dotall: bool = True, maybe_escaped_underscores = False) -> str:
        '''
            not_literal('a') -> '[^a]'
            not_literal('abc') -> '([^a] | "a" ([^b] | "b" ([^c])?)?)?'
        '''
        assert len(literal) > 0, 'Empty literal not supported'
        def recurse(i: int):
            c = literal[i]
            if maybe_escaped_underscores and c == '_':
                yield f'[^{c}\\\\]'
                yield ' | '
                yield f'"\\\\"? "{c}"'
            else:
                yield f'[^{c}]'
            if i < len(literal) - 1:
                yield ' | '
                yield self._format_literal(c)
                yield ' ('
                yield from recurse(i + 1)
                yield ')?'

        return ''.join(('(', *recurse(0), ')'))

    @staticmethod
    def _ranges(ranges):
        merged = []
        for lo, hi in sorted(ranges):
            if lo > hi:
                continue
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        return merged

    @classmethod
    def _subtract_ranges(cls, ranges, excluded):
        result = cls._ranges(ranges)
        for start, end in cls._ranges(excluded):
            parts = []
            for lo, hi in result:
                if hi < start or lo > end:
                    parts.append((lo, hi))
                else:
                    if lo < start:
                        parts.append((lo, start - 1))
                    if hi > end:
                        parts.append((end + 1, hi))
            result = parts
        return result

    def _hex_interval(self, lo, hi):
        """Four hex digits with case-insensitive A-F, compressed by prefix."""
        key = (lo, hi)
        cached = self._hex_intervals.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        def build(low, high, digits):
            if digits == 0:
                return '""'
            if low == 0 and high == 16 ** digits - 1:
                return '[0-9a-fA-F]' + (f'{{{digits}}}' if digits > 1 else '')
            block = 16 ** (digits - 1)
            options = []
            for first in range(low // block, high // block + 1):
                ch = format(first, 'x')
                head = f'[{ch}{ch.upper()}]' if first > 9 else f'"{ch}"'
                tail = build(max(0, low - first * block), min(block - 1, high - first * block), digits - 1)
                options.append(head + ' ' + tail)
            return '(' + ' | '.join(options) + ')'
        result = build(lo, hi, 4)
        self._hex_intervals[key] = result
        return result

    def _character_rule(self, ranges, raw=False):
        # Compare decoded characters, including equivalent JSON escape spellings.
        input_key = (raw, tuple(ranges))
        cached = self._character_input_rules.get(input_key, _MISSING)
        if cached is not _MISSING:
            return cached
        ranges = self._subtract_ranges(input_key[1], [(0xD800, 0xDFFF)])
        if not ranges:
            raise ValueError('Character class matches no Unicode scalar values')
        cache_key = (raw, tuple(ranges))
        cached = self._character_rules.get(cache_key, _MISSING)
        if cached is not _MISSING:
            self._character_input_rules[input_key] = cached
            return cached
        options = []
        direct = ranges if raw else self._subtract_ranges(ranges, [(0, 31), (34, 34), (92, 92)])
        if direct:
            def point(value):
                return chr(92) + ('u%04X' % value if value <= 0xFFFF else 'U%08X' % value)
            options.append('[' + ''.join(point(lo) + ('-' + point(hi) if hi != lo else '') for lo, hi in direct) + ']')
        if not raw:
            short = {'"': chr(92) + '"', chr(92): chr(92) * 2, '/': chr(92) + '/',
                     chr(8): chr(92) + 'b', chr(12): chr(92) + 'f', chr(10): chr(92) + 'n',
                     chr(13): chr(92) + 'r', chr(9): chr(92) + 't'}
            for ch, encoded in short.items():
                if any(lo <= ord(ch) <= hi for lo, hi in ranges):
                    options.append(self._format_literal(encoded))
            marker = self._format_literal(chr(92) + 'u')
            for lo, hi in ranges:
                if lo <= 0xFFFF:
                    options.append(marker + ' ' + self._hex_interval(lo, min(hi, 0xFFFF)))
                if hi >= 0x10000:
                    low, high = max(lo, 0x10000) - 0x10000, hi - 0x10000
                    first, last = low >> 10, high >> 10
                    blocks = [(first, first, low & 1023, (high & 1023) if first == last else 1023)]
                    if last > first:
                        if last > first + 1:
                            blocks.append((first + 1, last - 1, 0, 1023))
                        blocks.append((last, last, 0, high & 1023))
                    for h0, h1, l0, l1 in blocks:
                        options.append(marker + ' ' + self._hex_interval(0xD800 + h0, 0xD800 + h1) + ' ' +
                                       marker + ' ' + self._hex_interval(0xDC00 + l0, 0xDC00 + l1))
        name = self._add_rule(f'char-range-{len(self._character_rules)}', ' | '.join(options))
        self._character_rules[cache_key] = name
        self._character_input_rules[input_key] = name
        return name

    def _not_strings(self, strings):
        """JSON string keys excluding exact decoded property names."""
        trie = {}
        for string in strings:
            node = trie
            for ch in string:
                node = node.setdefault(ch, {})
            node[None] = True
        any_char = self._character_rule([(0, 0x10FFFF)])

        def build(node):
            children = sorted(ch for ch in node if ch is not None)
            options = ['""'] if None not in node else []
            remaining = self._subtract_ranges([(0, 0x10FFFF)], [(ord(ch), ord(ch)) for ch in children])
            options.append(self._character_rule(remaining) + ' ' + any_char + '*')
            for ch in children:
                options.append(self._character_rule([(ord(ch), ord(ch))]) + ' (' + build(node[ch]) + ')')
            return ' | '.join(options)

        quote = self._format_literal('"')
        return quote + ' (' + build(trie) + ') ' + quote

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub('-', name)
        existing = self._rules.get(esc_name, _MISSING)
        if existing is _MISSING or existing == rule:
            key = esc_name
        else:
            i = 0
            while True:
                key = f'{esc_name}{i}'
                existing = self._rules.get(key, _MISSING)
                if existing is _MISSING or existing == rule:
                    break
                i += 1
        self._rules[key] = rule
        return key

    def resolve_refs(self, schema: dict, url: str):
        """Resolve reachable schema nodes without traversing literal JSON data."""
        from urllib.parse import unquote, urljoin

        documents = {url: schema}
        visited = {}

        def walk(node, base):
            if not isinstance(node, dict) or id(node) in visited:
                return
            visited[id(node)] = node
            if '$ref' in node:
                ref = node['$ref']
                if not isinstance(ref, str):
                    raise ValueError('$ref must be a string')
                if ref.startswith('#'):
                    document_url, fragment = base, ref[1:]
                else:
                    absolute = urljoin(base, ref)
                    document_url, _, fragment = absolute.partition('#')
                    if not document_url.startswith('https://'):
                        raise ValueError(f'Unsupported ref {ref}')
                if document_url not in documents:
                    if not self._allow_fetch:
                        raise ValueError('Fetching remote schemas is not allowed')
                    import requests
                    response = requests.get(document_url, timeout=30)
                    response.raise_for_status()
                    documents[document_url] = response.json()
                fragment = unquote(fragment)
                if fragment and not fragment.startswith('/'):
                    raise ValueError(f'Unsupported reference anchor: {ref}')
                canonical = document_url + '#' + fragment
                node['$ref'] = canonical
                if canonical not in self._refs:
                    target = documents[document_url]
                    for part in fragment.split('/')[1:]:
                        if re.search(r'~(?![01])', part):
                            raise ValueError(f'Invalid JSON Pointer in {ref}')
                        key = part.replace('~1', '/').replace('~0', '~')
                        if isinstance(target, dict) and key in target:
                            target = target[key]
                        elif isinstance(target, list) and re.fullmatch(r'0|[1-9][0-9]*', key) and int(key) < len(target):
                            target = target[int(key)]
                        else:
                            raise ValueError(f'Cannot resolve {ref}: {key!r} not found')
                    if not isinstance(target, dict):
                        raise ValueError(f'Reference {ref} must target a schema object')
                    # Register before descending so recursive objects can refer back.
                    self._refs[canonical] = target
                    walk(target, document_url)
                return
            # Match converter precedence; const/enum payloads are data, not schemas.
            if 'allOf' in node:
                children = list(node['allOf']) if isinstance(node['allOf'], list) else []
                siblings = {k: v for k, v in node.items() if k != 'allOf'}
                walk(siblings, base)
            elif 'oneOf' in node or 'anyOf' in node:
                children = node.get('oneOf', node.get('anyOf'))
            elif 'const' in node or 'enum' in node:
                return
            else:
                children = []
                properties = node.get('properties', {})
                if isinstance(properties, dict):
                    children.extend(properties.values())
                for key in ('additionalProperties', 'items', 'prefixItems', 'allOf'):
                    value = node.get(key)
                    children.extend(value if isinstance(value, list) else [value])
            if isinstance(children, list):
                for child in children:
                    walk(child, base)

        walk(schema, url)
        return schema

    def _dereference(self, schema):
        seen = set()
        while isinstance(schema, dict) and '$ref' in schema:
            ref = schema['$ref']
            if ref in seen:
                raise ValueError('Reference cycle contains no concrete schema')
            seen.add(ref)
            schema = self._refs[ref]
        return schema

    def _generate_union_rule(self, name, alt_schemas):
        return ' | '.join((
            self.visit(alt_schema, f'{name}{"-" if name else "alternative-"}{i}')
            for i, alt_schema in enumerate(alt_schemas)
        ))

    def _visit_pattern(self, pattern, name):
        """Translate anchored regex atoms to GBNF over decoded JSON characters."""
        if not isinstance(pattern, str):
            raise ValueError('pattern must be a string')
        trailing_slashes = len(pattern[:-1]) - len(pattern[:-1].rstrip('\\'))
        if not pattern.startswith('^') or not pattern.endswith('$') or trailing_slashes % 2:
            raise ValueError('Pattern must start with "^" and end with "$"')
        source, pos = pattern[1:-1], 0
        universe = [(0, 0x10FFFF)]
        classes = {
            'd': [(48, 57)],
            'w': [(48, 57), (65, 90), (95, 95), (97, 122)],
            's': [(9, 13), (32, 32), (160, 160), (0x1680, 0x1680),
                  (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F),
                  (0x205F, 0x205F), (0x3000, 0x3000), (0xFEFF, 0xFEFF)],
        }

        def escaped(in_class=False):
            nonlocal pos
            pos += 1
            if pos >= len(source):
                raise ValueError('Trailing escape in pattern')
            ch = source[pos]
            pos += 1
            if ch.lower() in classes:
                ranges = classes[ch.lower()]
                return self._subtract_ranges(universe, ranges) if ch.isupper() else ranges
            controls = {'n': 10, 'r': 13, 't': 9, 'f': 12, 'v': 11}
            if in_class:
                controls['b'] = 8
            if ch in controls:
                return [(controls[ch], controls[ch])]
            if ch in ('u', 'x'):
                count = 4 if ch == 'u' else 2
                digits = source[pos:pos+count]
                if len(digits) != count or not re.fullmatch(r'[0-9a-fA-F]+', digits):
                    raise ValueError('Invalid hexadecimal escape in pattern')
                pos += count
                value = int(digits, 16)
                if 0xD800 <= value <= 0xDBFF and source[pos:pos+2] == '\\u':
                    low_digits = source[pos+2:pos+6]
                    if re.fullmatch(r'[0-9a-fA-F]{4}', low_digits):
                        low = int(low_digits, 16)
                        if 0xDC00 <= low <= 0xDFFF:
                            value = 0x10000 + ((value - 0xD800) << 10) + low - 0xDC00
                            pos += 6
                if 0xD800 <= value <= 0xDFFF:
                    raise ValueError('Unpaired surrogate in pattern')
                return [(value, value)]
            if ch in r'\^$.|?*+()[]{}-/"':
                return [(ord(ch), ord(ch))]
            raise ValueError(f'Unsupported regex escape: {chr(92)}{ch}')

        def class_atom():
            nonlocal pos
            if pos >= len(source):
                raise ValueError('Unbalanced character class')
            if source[pos] == '\\':
                return escaped(in_class=True)
            value = ord(source[pos])
            pos += 1
            return [(value, value)]

        def expression(grouped=False):
            nonlocal pos
            alternatives, sequence = [], []
            while pos < len(source):
                ch = source[pos]
                if ch == ')':
                    if not grouped:
                        raise ValueError('Unbalanced parentheses in pattern')
                    break
                if ch == '|':
                    alternatives.append(' '.join(sequence) or '""')
                    sequence = []
                    pos += 1
                    continue
                if ch == '(':
                    pos += 1
                    if source[pos:pos+2] == '?:':
                        pos += 2
                    elif source[pos:pos+1] == '?':
                        raise ValueError('Unsupported regex group syntax')
                    atom = '(' + expression(grouped=True) + ')'
                    if pos >= len(source) or source[pos] != ')':
                        raise ValueError('Unbalanced parentheses in pattern')
                    pos += 1
                else:
                    if ch == '[':
                        pos += 1
                        negated = source[pos:pos+1] == '^'
                        if negated:
                            pos += 1
                        ranges = []
                        while pos < len(source) and source[pos] != ']':
                            left = class_atom()
                            if source[pos:pos+1] == '-' and source[pos+1:pos+2] not in ('', ']'):
                                pos += 1
                                right = class_atom()
                                if len(left) != 1 or len(right) != 1 or left[0][0] != left[0][1] or right[0][0] != right[0][1] or left[0][0] > right[0][0]:
                                    raise ValueError('Invalid character class range')
                                left = [(left[0][0], right[0][0])]
                            ranges.extend(left)
                        if pos >= len(source):
                            raise ValueError('Unbalanced character class')
                        pos += 1
                        if negated:
                            ranges = self._subtract_ranges(universe, ranges)
                    elif ch == '\\':
                        ranges = escaped()
                    elif ch == '.':
                        ranges = universe if self._dotall else self._subtract_ranges(universe, [(10, 10), (13, 13), (0x2028, 0x2029)])
                        pos += 1
                    elif ch in '^$*+?{}':
                        raise ValueError(f'Unsupported or misplaced regex operator: {ch}')
                    else:
                        ranges = [(ord(ch), ord(ch))]
                        pos += 1
                    atom = self._character_rule(ranges, raw=self._raw_pattern)
                if pos < len(source) and source[pos] in '*+?':
                    atom += source[pos]
                    pos += 1
                elif source[pos:pos+1] == '{':
                    match = re.match(r'\{(\d+)(,([0-9]*))?\}', source[pos:])
                    if not match:
                        raise ValueError('Invalid regex quantifier')
                    minimum = int(match[1])
                    maximum = (int(match[3]) if match[3] else None) if match[2] else minimum
                    if maximum is not None and maximum < minimum:
                        raise ValueError('Invalid regex quantifier range')
                    atom = _build_repetition(atom, minimum, maximum) or '""'
                    pos += len(match[0])
                sequence.append(atom)
            alternatives.append(' '.join(sequence) or '""')
            return ' | '.join(alternatives)

        body = expression()
        if not self._raw_pattern:
            quote = self._format_literal('"')
            body = quote + ' (' + body + ') ' + quote
        return self._add_rule(name, body)

    def _resolve_ref(self, ref):
        if ref not in self._ref_rule_names:
            # Allocate by identity, not a lossy transformation of the pointer text.
            name = f'schema-ref-{len(self._ref_rule_names)}'
            while name in self._rules:
                name += '-ref'
            self._ref_rule_names[ref] = name
            self._rules[name] = ''
            resolved = self._dereference(self._refs[ref])
            actual = self.visit(resolved, name)
            if actual != name:
                self._rules[name] = actual
        return self._ref_rule_names[ref]

    def _generate_constant_rule(self, value):
        return self._format_literal(json.dumps(value))

    @staticmethod
    def _json_value_key(value):
        if value is None:
            return ('null',)
        if isinstance(value, bool):
            return ('boolean', value)
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('JSON numbers must be finite')
            return ('number', value)
        if isinstance(value, str):
            return ('string', value)
        if isinstance(value, list):
            return ('array', tuple(SchemaConverter._json_value_key(v) for v in value))
        if isinstance(value, dict):
            return ('object', tuple(sorted((k, SchemaConverter._json_value_key(v)) for k, v in value.items())))
        raise ValueError('Invalid JSON value in enum')

    def _visit_all_of(self, schema, name, rule_name):
        components = []
        active = set()

        def collect(node):
            node = self._dereference(node)
            if not isinstance(node, dict):
                raise ValueError('allOf components must be schema objects')
            if id(node) in active:
                raise ValueError('Recursive allOf is not supported')
            if 'allOf' in node:
                active.add(id(node))
                if not isinstance(node['allOf'], list) or not node['allOf']:
                    raise ValueError('allOf must be a non-empty array')
                siblings = {k: v for k, v in node.items() if k != 'allOf'}
                if siblings:
                    components.append(siblings)
                for child in node['allOf']:
                    collect(child)
                active.remove(id(node))
            else:
                components.append(node)

        collect(schema)
        metadata = {'title', 'description', 'default', 'examples', '$defs', 'definitions', '$schema', '$id', '$comment'}
        components = [{k: v for k, v in node.items() if k not in metadata} for node in components]
        components = [node for node in components if node]
        if not components:
            return self.visit({}, name)
        if all(set(node) <= {'enum', 'const', 'type'} for node in components) and any('enum' in node or 'const' in node for node in components):
            alternatives = []
            for node in components:
                if 'enum' not in node and 'const' not in node:
                    continue
                values = node.get('enum', [node.get('const')])
                if not isinstance(values, list) or not values:
                    raise ValueError('enum must be a non-empty array')
                keyed = {self._json_value_key(v): v for v in values}
                if 'const' in node:
                    keyed = {k: v for k, v in keyed.items() if k == self._json_value_key(node['const'])}
                alternatives.append(keyed)
            keys = set(alternatives[0]).intersection(*(set(v) for v in alternatives[1:]))
            def matches_type(value, kind):
                if isinstance(kind, list):
                    return any(matches_type(value, item) for item in kind)
                if kind == 'integer':
                    return type(value) is int or (type(value) is float and value.is_integer())
                expected = {'null': type(None), 'boolean': bool, 'string': str, 'array': list, 'object': dict}
                if kind == 'number':
                    return type(value) in (int, float)
                if kind not in expected:
                    raise ValueError(f'Unsupported allOf type {kind!r}')
                return type(value) is expected[kind]
            keys = {key for key in keys if all('type' not in node or matches_type(alternatives[0][key], node['type']) for node in components)}
            if not keys:
                raise ValueError('allOf has an empty enum intersection')
            values = [value for key, value in alternatives[0].items() if key in keys]
            return self._add_rule(rule_name, '(' + ' | '.join(self._generate_constant_rule(v) for v in values) + ')')
        if len(components) == 1:
            return self.visit(components[0], name)
        allowed = {'type', 'properties', 'required', 'additionalProperties'}
        if not all(set(node) <= allowed and node.get('type', 'object') == 'object' for node in components):
            raise ValueError('Unsupported allOf intersection; combine constraints in a single schema')
        properties, required = {}, set()
        for node in components:
            for key, value in node.get('properties', {}).items():
                if key in properties and self._json_value_key(properties[key]) != self._json_value_key(value):
                    raise ValueError(f'Unsupported allOf intersection for property {key!r}')
                properties[key] = value
            required.update(node.get('required', []))
        for node in components:
            additional = node.get('additionalProperties')
            if additional is not None and additional is not True:
                if additional is not False or set(node.get('properties', {})) != set(properties):
                    raise ValueError('Unsupported allOf additionalProperties intersection')
        if not required <= properties.keys():
            raise ValueError('allOf requires properties without declared schemas')
        allow_additional = all(node.get('additionalProperties') is True or
                               ('properties' not in node and 'additionalProperties' not in node)
                               for node in components)
        return self._add_rule(rule_name, self._build_object_rule(list(properties.items()), required, name, True if allow_additional else None))

    def visit(self, schema, name):
        if not isinstance(schema, dict):
            raise ValueError(f'Schema at {name or "root"} must be an object')
        for key in ('oneOf', 'anyOf', 'allOf', 'enum'):
            if key in schema and (not isinstance(schema[key], list) or not schema[key]):
                raise ValueError(f'{key} must be a non-empty array')
        if schema.get('type') == []:
            raise ValueError('type must not be empty')
        for key in ('minLength', 'maxLength', 'minItems', 'maxItems'):
            if key in schema and (type(schema[key]) is not int or schema[key] < 0):
                raise ValueError(f'{key} must be a non-negative integer')
        schema_type = schema.get('type')
        schema_format = schema.get('format')
        rule_name = name + '-' if name in RESERVED_NAMES else name or 'root'

        if (ref := schema.get('$ref')) is not None:
            return self._add_rule(rule_name, self._resolve_ref(ref))

        elif 'allOf' in schema:
            return self._visit_all_of(schema, name, rule_name)

        elif 'oneOf' in schema or 'anyOf' in schema:
            return self._add_rule(rule_name, self._generate_union_rule(name, schema.get('oneOf') or schema['anyOf']))

        elif isinstance(schema_type, list):
            return self._add_rule(rule_name, self._generate_union_rule(name, [{**schema, 'type': t} for t in schema_type]))

        elif 'const' in schema:
            return self._add_rule(rule_name, self._generate_constant_rule(schema['const']))

        elif 'enum' in schema:
            rule = '(' + ' | '.join((self._generate_constant_rule(v) for v in schema['enum'])) + ')'
            return self._add_rule(rule_name, rule)

        elif schema_type in (None, 'object') and \
             ('properties' in schema or \
              ('additionalProperties' in schema and schema['additionalProperties'] is not True)):
            required = set(schema.get('required', []))
            properties = list(schema.get('properties', {}).items())
            return self._add_rule(rule_name, self._build_object_rule(properties, required, name, schema.get('additionalProperties')))

        elif schema_type == 'array' or (schema_type is None and ('items' in schema or 'prefixItems' in schema)):
            items = schema.get('items', schema.get('prefixItems', {}))
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space ' +
                    ' "," space '.join(
                        self.visit(item, f'{name}{"-" if name else ""}tuple-{i}')
                        for i, item in enumerate(items)) +
                    ' space "]"')
            else:
                item_rule_name = self.visit(items, f'{name}{"-" if name else ""}item')
                min_items = schema.get("minItems", 0)
                max_items = schema.get("maxItems")
                return self._add_rule(rule_name, '"[" space ' + _build_repetition(item_rule_name, min_items, max_items, separator_rule='"," space') + ' space "]"')

        elif schema_type in (None, 'string') and 'pattern' in schema:
            return self._visit_pattern(schema['pattern'], rule_name)

        elif schema_type in (None, 'string') and re.match(r'^uuid[1-5]?$', schema_format or ''):
            return self._add_primitive(
                'root' if rule_name == 'root' else schema_format,
                PRIMITIVE_RULES['uuid']
            )

        elif schema_type in (None, 'string') and f'{schema_format}-string' in STRING_FORMAT_RULES:
            prim_name = f'{schema_format}-string'
            return self._add_rule(rule_name, self._add_primitive(prim_name, STRING_FORMAT_RULES[prim_name]))

        elif schema_type in (None, 'string') and ('minLength' in schema or 'maxLength' in schema):
            char_rule = self._add_primitive('char', PRIMITIVE_RULES['char'])
            min_len = schema.get('minLength', 0)
            max_len = schema.get('maxLength')

            return self._add_rule(rule_name, r'"\"" ' + _build_repetition(char_rule, min_len, max_len) + r' "\""')

        elif schema_type == 'integer' and \
                ('minimum' in schema or 'exclusiveMinimum' in schema or 'maximum' in schema or 'exclusiveMaximum' in schema):
            min_value = None
            max_value = None
            if 'minimum' in schema:
                min_value = math.ceil(schema['minimum'])
            elif 'exclusiveMinimum' in schema:
                min_value = math.floor(schema['exclusiveMinimum']) + 1
            if 'maximum' in schema:
                max_value = math.floor(schema['maximum'])
            elif 'exclusiveMaximum' in schema:
                max_value = math.ceil(schema['exclusiveMaximum']) - 1

            out = ["("]
            _generate_min_max_int(min_value, max_value, out)
            out.append(")")
            return self._add_rule(rule_name, ''.join(out))

        elif schema_type == 'object':
            return self._add_rule(rule_name, self._add_primitive('object', PRIMITIVE_RULES['object']))

        elif schema_type is None and isinstance(schema, dict):
            # No type constraint and no recognized structural keywords (e.g. {"description": "..."}).
            # Per JSON Schema semantics this is equivalent to {} and accepts any value.
            return self._add_rule(rule_name, self._add_primitive('value', PRIMITIVE_RULES['value']))

        else:
            assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
            # TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
            return self._add_primitive('root' if rule_name == 'root' else schema_type, PRIMITIVE_RULES[schema_type])

    def _add_primitive(self, name: str, rule: BuiltinRule):
        n = self._add_rule(name, rule.content)

        for dep in rule.deps:
            dep_rule = PRIMITIVE_RULES.get(dep) or STRING_FORMAT_RULES.get(dep)
            assert dep_rule, f'Rule {dep} not known'
            if dep not in self._rules:
                self._add_primitive(dep, dep_rule)
        return n

    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], name: str, additional_properties: Optional[Union[bool, Any]]):
        prop_order = self._prop_order
        # sort by position in prop_order (if specified) then by original order
        sorted_props = [kv[0] for _, kv in sorted(enumerate(properties), key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]))]

        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                fr'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}'
            )
        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]
        additional_key = object()

        if additional_properties is not None and additional_properties != False:
            sub_name = f'{name}{"-" if name else ""}additional'
            value_rule = self.visit(additional_properties, f'{sub_name}-value') if isinstance(additional_properties, dict) else \
                self._add_primitive('value', PRIMITIVE_RULES['value'])
            key_rule = self._add_primitive('string', PRIMITIVE_RULES['string']) if not sorted_props \
                else self._add_rule(f'{sub_name}-k', self._not_strings(sorted_props))

            prop_kv_rule_names[additional_key] = self._add_rule(
                f'{sub_name}-kv',
                f'{key_rule} ":" space {value_rule}'
            )
            optional_props.append(additional_key)

        if not required_props and not optional_props:
            return '"{" space "}"'

        rule = '"{" space '
        rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)

        if optional_props:
            rule += ' ('
            if required_props:
                rule += ' "," space ( '

            # Build each optional suffix once, preserving rule allocation order.
            alternatives = [''] * len(optional_props)
            optional_suffix = None
            for i in range(len(optional_props) - 1, -1, -1):
                k = optional_props[i]
                kv_rule_name = prop_kv_rule_names[k]
                comma_ref = f'( "," space {kv_rule_name} )'
                suffix_ref = ''
                if optional_suffix is not None:
                    suffix_ref = ' ' + self._add_rule(
                        f'{name}{"-" if name else ""}{"additional" if k is additional_key else k}-rest',
                        optional_suffix,
                    )
                alternatives[i] = kv_rule_name + (' ' + comma_ref + '*' if k is additional_key else '') + suffix_ref
                optional_suffix = comma_ref + ('*' if k is additional_key else '?') + suffix_ref

            rule += ' | '.join(alternatives)
            if required_props:
                rule += ' )'
            rule += ' )?'

        rule += ' space "}"'

        return rule

    def format_grammar(self):
        return '\n'.join(
            f'{name} ::= {rule}'
            for name, rule in sorted(self._rules.items(), key=lambda kv: kv[0])
        )


def json_schema_to_gbnf(
    schema: Union[str, dict],
    prop_order: Optional[List[str]] = None,
    allow_fetch: bool = False,
    dotall: bool = False,
    raw_pattern: bool = False,
):
    prop_order = prop_order or []

    if isinstance(schema, str):
        schema = json.loads(schema)
    elif isinstance(schema, dict):
        schema = copy.deepcopy(schema)
    else:
        raise TypeError("schema must be a JSON string or dictionary")

    converter = SchemaConverter(
        prop_order={name: idx for idx, name in enumerate(prop_order)},
        allow_fetch=allow_fetch,
        dotall=dotall,
        raw_pattern=raw_pattern,
    )
    schema = converter.resolve_refs(schema, "stdin")
    converter.visit(schema, "")
    return converter.format_grammar()
