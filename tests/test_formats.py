import llama_cpp
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
import json
import copy
import os
import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from llama_cpp import llama_grammar
from llama_cpp import _internals


    # assert grammar._n_rules == 3
    # assert grammar._start_rule_index == 2
    # assert grammar.grammar is not None


    # assert grammar.grammar is not None


    # assert grammar.grammar is not None


@pytest.mark.parametrize('schema, expected', [
    ({}, 'root ::= value'),
    ({'type': 'array', 'items': {}}, 'item ::= value'),
    ({'type': 'object', 'properties': {'x': {}}}, 'x ::= value'),
    ({'minimum': 2}, 'root ::= value'),
])
def test_unconstrained_schema(schema, expected):
    assert expected in llama_grammar.json_schema_to_gbnf(schema).splitlines()


def test_implicit_string_and_array_bounds():
    convert = llama_grammar.json_schema_to_gbnf
    assert convert({'minLength': 2}) == convert({'type': 'string', 'minLength': 2})
    assert convert({'type': 'array', 'minItems': 2}) == convert(
        {'type': 'array', 'items': {}, 'minItems': 2})


@pytest.mark.parametrize('bounds, integer_bounds', [
    ({'minimum': 1.2, 'maximum': 4.8}, {'minimum': 2, 'maximum': 4}),
    ({'exclusiveMinimum': -2.5, 'exclusiveMaximum': 2.5}, {'minimum': -2, 'maximum': 2}),
])
def test_integer_fractional_bounds(bounds, integer_bounds):
    convert = llama_grammar.json_schema_to_gbnf
    assert convert({'type': 'integer', **bounds}) == convert({'type': 'integer', **integer_bounds})


@pytest.mark.parametrize('schema', [
    {'anyOf': []}, {'oneOf': []}, {'allOf': []}, {'enum': []}, {'type': []},
    {'type': 'array', 'minItems': -1}, {'minLength': True},
    {'type': 'array', 'items': False},
])
def test_invalid_schema_is_rejected(schema):
    with pytest.raises(ValueError):
        llama_cpp.LlamaGrammar.from_json_schema(schema)


def test_schema_conversion_does_not_mutate_input():
    schema = {'$defs': {'x': {'type': 'string'}},
              'properties': {'x': {'$ref': '#/$defs/x'}}}
    original = copy.deepcopy(schema)
    first = llama_cpp.LlamaGrammar.from_json_schema(schema)
    second = llama_cpp.LlamaGrammar.from_json_schema(schema)
    assert schema == original
    assert first.grammar == second.grammar


def test_grammar_definition_options(tmp_path):
    path = tmp_path / 'custom.gbnf'
    path.write_text('answer ::= "yes"', encoding='utf-8')
    triggers = ['(yes)', 42]
    grammar = llama_cpp.LlamaGrammar.from_file(path, root='answer', triggers=triggers)
    triggers.clear()
    assert grammar.root == 'answer'
    assert grammar.triggers == ('(yes)', 42)
    assert grammar.grammar == 'answer ::= "yes"'


@pytest.mark.parametrize('text', ['', '   ', 'root ::= "x"\x00'])
def test_invalid_grammar_text(text):
    with pytest.raises(ValueError):
        llama_cpp.LlamaGrammar.from_string(text)


def test_native_sampler_lifecycle_and_root(monkeypatch):
    init = Mock(return_value=object())
    free = Mock()
    monkeypatch.setattr(_internals.llama_cpp, 'llama_sampler_init_grammar', init)
    monkeypatch.setattr(_internals.llama_cpp, 'llama_sampler_free', free)
    model = SimpleNamespace(vocab=object())
    with _internals.GrammarSampler(model, 'answer ::= "yes"', root='answer') as sampler:
        init.assert_called_once_with(model.vocab, b'answer ::= "yes"', b'answer')
    sampler.close()
    free.assert_called_once()
    for operation in (lambda: sampler.apply(None), lambda: sampler.accept(1), sampler.reset):
        with pytest.raises(RuntimeError, match='closed'):
            operation()


def test_lazy_sampler_passes_patterns_and_tokens(monkeypatch):
    def initialize(vocab, text, root, patterns, n_patterns, tokens, n_tokens):
        assert root == b'answer'
        assert list(patterns) == [b'(yes)'] and n_patterns == 1
        assert list(tokens) == [42] and n_tokens == 1
        return object()
    init = Mock(side_effect=initialize)
    monkeypatch.setattr(_internals.llama_cpp, 'llama_sampler_init_grammar_lazy_patterns', init)
    monkeypatch.setattr(_internals.llama_cpp, 'llama_sampler_free', Mock())
    with _internals.GrammarSampler(SimpleNamespace(vocab=None), 'answer ::= "yes"',
                                   lazy=True, triggers=['(yes)', 42], root='answer'):
        init.assert_called_once()
    with pytest.raises(ValueError, match='trigger'):
        _internals.GrammarSampler(SimpleNamespace(vocab=None), 'root ::= "x"', lazy=True)


@pytest.mark.parametrize('schema', [
    {}, {'type': 'array', 'items': {}}, {'minLength': 2},
    {'type': 'integer', 'minimum': 1.2, 'maximum': 4.8},
])
def test_converted_grammar_parses_in_native_library(schema):
    # Standard GBNF without token literals can be parsed without a vocabulary.
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema)
    with _internals.GrammarSampler(SimpleNamespace(vocab=None), grammar.grammar):
        pass


@pytest.mark.parametrize('pattern', ['^(?:foo|bar)baz$', '^(?:(?:ab)+c)?d$'])
def test_non_capturing_pattern_groups(pattern):
    convert = llama_grammar.json_schema_to_gbnf
    grammar = convert({'type': 'string', 'pattern': pattern})
    assert grammar == convert({'type': 'string', 'pattern': pattern.replace('(?:', '(')})
    with _internals.GrammarSampler(SimpleNamespace(vocab=None), grammar):
        pass


@pytest.mark.parametrize('pattern', ['^(?:ab$', '^ab)$'])
def test_unbalanced_pattern_groups(pattern):
    with pytest.raises(ValueError, match='parentheses'):
        llama_cpp.LlamaGrammar.from_json_schema({'type': 'string', 'pattern': pattern})


@pytest.mark.parametrize('schema', [
    {'const': {'$ref': 'literal'}},
    {'enum': [{'$ref': 'literal'}]},
    {'type': 'string', '$defs': {'unused': {'$ref': '#/missing'}}},
])
def test_reference_resolution_ignores_data_and_unused_definitions(schema):
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema)
    with _internals.GrammarSampler(SimpleNamespace(vocab=None), grammar.grammar):
        pass


def test_reference_pointer_escaping_and_chains():
    schema = {'$defs': {'a/b~c': {'enum': ['x']}, 'alias': {'$ref': '#/$defs/a~1b~0c'}},
              'allOf': [{'$ref': '#/$defs/alias'}, {'enum': ['x', 'y']}]}
    grammar = llama_grammar.json_schema_to_gbnf(schema)
    assert 'root ::= ("\\\"x\\\"")' in grammar


def test_recursive_root_reference_parses():
    schema = {'type': 'object', 'properties': {'next': {'anyOf': [{'$ref': '#'}, {'type': 'null'}]}},
              'required': ['next']}
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema)
    with _internals.GrammarSampler(SimpleNamespace(vocab=None), grammar.grammar):
        pass


@pytest.mark.parametrize('schema', [
    {'$ref': '#'},
    {'$defs': {'a': {'$ref': '#/$defs/b'}, 'b': {'$ref': '#/$defs/a'}}, '$ref': '#/$defs/a'},
    {'$ref': '#/$defs/x~2y', '$defs': {}},
])
def test_invalid_reference_cycles_and_pointers(schema):
    with pytest.raises(ValueError):
        llama_cpp.LlamaGrammar.from_json_schema(schema)


def test_remote_reference_uses_its_own_document(monkeypatch):
    response = Mock()
    response.json.return_value = {'$defs': {'x': {'type': 'string'}}, '$ref': '#/$defs/x'}
    get = Mock(return_value=response)
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=get))
    schema = {'type': 'array', 'items': {'$ref': 'https://example.invalid/schema.json'}}
    with pytest.raises(ValueError, match='Fetching'):
        llama_cpp.LlamaGrammar.from_json_schema(schema)
    get.assert_not_called()
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema, allow_fetch=True)
    get.assert_called_once_with('https://example.invalid/schema.json', timeout=30)
    response.raise_for_status.assert_called_once()
    with _internals.GrammarSampler(SimpleNamespace(vocab=None), grammar.grammar):
        pass


@pytest.mark.parametrize('values', [[{'x': 1}], [[1, 'x']], [1, 'x', True], [{'b': 2, 'a': 1}]])
def test_allof_json_enum_values(values):
    schema = {'allOf': [{'enum': values}, {'enum': list(reversed(values))}]}
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema)
    with _internals.GrammarSampler(SimpleNamespace(vocab=None), grammar.grammar):
        pass


@pytest.mark.parametrize('schema', [
    {'allOf': [{'enum': [True]}, {'enum': [1]}]},
    {'allOf': [{'enum': ['a']}, {'enum': ['b']}]},
    {'allOf': [{'type': 'string'}, {'minLength': 2}]},
])
def test_allof_impossible_or_unsupported_intersections_fail(schema):
    with pytest.raises(ValueError, match='allOf'):
        llama_cpp.LlamaGrammar.from_json_schema(schema)


@pytest.mark.parametrize('pattern', [r'^\bword$', r'^(a)\1$', r'^a{3,2}$', r'^[z-a]$', r'^a*?$', r'^[\d-a]$'])
def test_unsupported_or_invalid_regex_is_rejected(pattern):
    with pytest.raises(ValueError):
        llama_cpp.LlamaGrammar.from_json_schema({'type': 'string', 'pattern': pattern})


def _native_accepts(model, grammar, text):
    with _internals.GrammarSampler(model._model, grammar) as sampler:
        tokens = model.tokenize(text.encode('utf-8'), add_bos=False)
        for token in [*tokens, model.token_eos()]:
            candidate = _internals.llama_cpp.llama_token_data(id=token, logit=0, p=0)
            data = _internals.llama_cpp.llama_token_data_array(
                data=ctypes.pointer(candidate), size=1, selected=-1, sorted=False)
            sampler.apply(ctypes.byref(data))
            if not np.isfinite(candidate.logit):
                return False
            sampler.accept(token)
        return True


@pytest.mark.parametrize('pattern, good, bad', [
    (r'^\d+$', '"123"', '"abc"'),
    (r'^[\dA-F]{2}$', '"A7"', '"G7"'),
    (r'^\w+\s\D$', '"A_1\\n!"', '"A_1 7"'),
    (r'^a\nb$', '"a\\nb"', '"anb"'),
    (r'^\\$', '"\\\\"', '"x"'),
    (r'^\u4e2d$', '"\\u4E2D"', '"x"'),
    (r'^\uD83D\uDE00$', '"😀"', '"x"'),
    (r'^[^a]+$', '"b"', '"\\u0061"'),
    (r'^.$', '"😀"', '"\\n"'),
    (r'^(ab|c){2}$', '"abc"', '"ab"'),
])
def test_model_regex_character_semantics(grammar_model, pattern, good, bad):
    grammar = llama_grammar.json_schema_to_gbnf({'type': 'string', 'pattern': pattern})
    assert _native_accepts(grammar_model, grammar, good)
    assert not _native_accepts(grammar_model, grammar, bad)


@pytest.mark.parametrize('names, allowed, excluded', [
    (['abc'], '"a"', '"abc"'),
    (['a', 'abc'], '"ab"', '"abc"'),
    (['a'], '"ab"', '"\\u0061"'),
    (['a]'], '"a"', '"a]"'),
    (['a-b'], '"a+b"', '"a-b"'),
    (['a\\b'], '"ab"', '"a\\\\b"'),
    (['a"b'], '"ab"', '"a\\"b"'),
    (['中'], '"文"', '"\\u4E2D"'),
    (['😀'], '"😁"', '"\\uD83D\\uDE00"'),
    ([''], '"a"', '""'),
    (['\n'], '"n"', '"\\n"'),
])
def test_model_property_name_exclusions(grammar_model, names, allowed, excluded):
    converter = llama_grammar.SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
    converter._add_rule('root', converter._not_strings(names))
    grammar = converter.format_grammar()
    assert _native_accepts(grammar_model, grammar, allowed)
    assert not _native_accepts(grammar_model, grammar, excluded)


def test_model_reference_names_do_not_collide(grammar_model):
    schema = {'$defs': {'a_b': {'const': 1}, 'a-b': {'const': 2}},
              'type': 'array', 'prefixItems': [{'$ref': '#/$defs/a_b'}, {'$ref': '#/$defs/a-b'}]}
    grammar = llama_grammar.json_schema_to_gbnf(schema)
    assert _native_accepts(grammar_model, grammar, '[1,2]')
    assert not _native_accepts(grammar_model, grammar, '[1,1]')


@pytest.mark.parametrize('schema, expected', [
    ({'type': 'string', 'pattern': r'^\d{3}$'}, '123'),
    ({'allOf': [{'enum': [{'ok': True}, {'ok': False}]}, {'enum': [{'ok': True}]}]}, {'ok': True}),
    ({'$defs': {'a/b': {'const': {'ok': True}}, 'alias': {'$ref': '#/$defs/a~1b'}},
      '$ref': '#/$defs/alias'}, {'ok': True}),
])
def test_model_upgraded_schema_generation(grammar_model, schema, expected):
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema)
    text = _grammar_completion(grammar_model, 'Return only this JSON: ' + json.dumps(expected), grammar)
    assert json.loads(text) == expected


def test_model_additional_properties_rejects_escaped_duplicate(grammar_model):
    schema = {'type': 'object', 'properties': {'a': {'const': True}}, 'required': ['a'],
              'additionalProperties': {'type': 'integer'}}
    grammar = llama_grammar.json_schema_to_gbnf(schema)
    assert _native_accepts(grammar_model, grammar, '{"a":true,"ab":2}')
    assert not _native_accepts(grammar_model, grammar, '{"a":true,"\\u0061":2}')


def test_model_star_property_is_not_additional_properties_marker(grammar_model):
    schema = {'type': 'object', 'properties': {'*': {'const': True}}, 'required': ['*'],
              'additionalProperties': {'type': 'integer'}}
    grammar = llama_grammar.json_schema_to_gbnf(schema)
    assert _native_accepts(grammar_model, grammar, '{"*":true,"x":2}')
    assert not _native_accepts(grammar_model, grammar, '{"*":2}')


def test_typed_allof_enum_intersection():
    grammar = llama_grammar.json_schema_to_gbnf({'allOf': [
        {'type': 'integer', 'enum': [1, 2]}, {'enum': [True, 1.0, 'x']},
    ]})
    assert 'root ::= ("1")' in grammar


def test_allof_checks_const_sibling():
    with pytest.raises(ValueError, match='empty enum intersection'):
        llama_cpp.LlamaGrammar.from_json_schema({'const': 1, 'allOf': [{'enum': [2]}]})


def test_allof_conflicting_boolean_numeric_property_is_rejected():
    with pytest.raises(ValueError, match='intersection for property'):
        llama_cpp.LlamaGrammar.from_json_schema({'allOf': [
            {'properties': {'x': {'const': True}}}, {'properties': {'x': {'const': 1}}},
        ]})


def test_large_optional_object_does_not_require_python_recursion():
    schema = {'type': 'object', 'properties': {
        f'field{i}': {'type': 'integer'} for i in range(1200)
    }}
    grammar = llama_grammar.json_schema_to_gbnf(schema)
    assert 'field1199-kv ::=' in grammar
    assert 'root ::= "{" space' in grammar


def test_rule_lookup_preserves_empty_placeholders_and_name_collisions():
    converter = llama_grammar.SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
    assert converter._add_rule('reference', '') == 'reference'
    assert converter._add_rule('reference', '"a"') == 'reference0'
    assert converter._add_rule('reference', '"a"') == 'reference0'
    assert converter._add_rule('reference', '') == 'reference'


def test_character_cache_keeps_raw_json_and_converter_namespaces_separate():
    def converter():
        return llama_grammar.SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
    first, second = converter(), converter()
    chars = [(50, 57), (48, 51)]
    original = first._character_rule(chars)
    snapshot = first.format_grammar()
    assert first._character_rule(chars) == original
    assert first._character_rule([(48, 57)]) == original
    assert first.format_grammar() == snapshot
    assert first._character_rule(chars, raw=True) != original
    fresh = second._character_rule([(65, 90)])
    assert fresh == original
    assert second.format_grammar() != snapshot
    assert chars == [(50, 57), (48, 51)]


@pytest.mark.parametrize('required, additional, order, good, bad', [
    ([], False, None, ['{}', '{"a":1}', '{"c":3}', '{"a":1,"c":3}'],
     ['{"c":3,"a":1}', '{"a":1,"a":2}', '{"x":1}']),
    (['b'], False, ['c', 'a'], ['{"b":1}', '{"b":1,"a":2}', '{"b":1,"c":2,"a":3}'],
     ['{}', '{"a":2,"b":1}', '{"b":1,"a":2,"c":3}']),
    ([], True, None, ['{}', '{"a":1,"x":2,"y":3}', '{"x":2}'],
     ['{"x":2,"a":1}', '{"a":1,"\\u0061":2}']),
])
def test_model_optional_suffix_order_and_cardinality(grammar_model, required, additional, order, good, bad):
    schema = {'type': 'object', 'properties': {k: {'type': 'integer'} for k in ['a', 'b', 'c']},
              'required': required, 'additionalProperties': additional}
    grammar = llama_grammar.json_schema_to_gbnf(schema, prop_order=order)
    for text in good:
        assert _native_accepts(grammar_model, grammar, text), text
    for text in bad:
        assert not _native_accepts(grammar_model, grammar, text), text


@pytest.fixture(scope='module')
def grammar_model():
    model_path = os.environ.get("LLAMA_TEST_TRANSFORMER_MODEL")
    if not model_path:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("LLAMA_TEST_TRANSFORMER_MODEL is required in Actions")
        pytest.skip("Set LLAMA_TEST_TRANSFORMER_MODEL to run real-model tests")
    assert os.path.isfile(model_path), model_path
    model = llama_cpp.Llama(model_path=model_path, n_gpu_layers=0,
                            n_ctx=512, n_batch=128, verbose=False)
    try:
        yield model
    finally:
        model.close()


def _grammar_completion(model, instruction, grammar, **kwargs):
    prompt = (f'<|im_start|>user\n{instruction}<|im_end|>\n'
              '<|im_start|>assistant\n')
    output = model.create_completion(prompt, grammar=grammar, max_tokens=96,
                                     temperature=0, seed=42, **kwargs)
    text = output['choices'][0]['text']
    print(f'grammar model output: {text!r}')
    assert output['choices'][0]['finish_reason'] == 'stop'
    return text


def test_model_custom_grammar_root(grammar_model):
    grammar = llama_cpp.LlamaGrammar.from_string('answer ::= "GRAMMAR_OK"', root='answer')
    for _ in range(2):
        assert _grammar_completion(grammar_model, 'Reply with HELLO.', grammar) == 'GRAMMAR_OK'


@pytest.mark.parametrize('schema, instruction, expected_type', [
    ({'minLength': 2, 'maxLength': 4}, 'Return the JSON string "a".', str),
    ({'type': 'integer', 'minimum': 1.2, 'maximum': 4.8}, 'Return 100.', int),
    ({'type': 'array', 'minItems': 2, 'maxItems': 2,
      'items': {'type': 'integer', 'minimum': 1, 'maximum': 3}}, 'Return [].', list),
])
def test_model_schema_constraints(grammar_model, schema, instruction, expected_type):
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema)
    value = json.loads(_grammar_completion(grammar_model, instruction, grammar))
    assert type(value) is expected_type
    if expected_type is str:
        assert 2 <= len(value) <= 4
    elif expected_type is int:
        assert 2 <= value <= 4
    else:
        assert len(value) == 2 and all(type(v) is int and 1 <= v <= 3 for v in value)


def test_model_lazy_grammar_completion(grammar_model):
    grammar = llama_cpp.LlamaGrammar.from_string(
        'answer ::= "{\\"ok\\":true}"', root='answer',
        triggers=[r'[\s\S]*?(\{)[\s\S]*'],
    )
    text = _grammar_completion(grammar_model, 'Return only this JSON: {"ok":false}',
                               grammar, grammar_lazy=True)
    assert json.loads(text[text.index('{'):]) == {'ok': True}


@pytest.mark.parametrize('trigger_kind', ['pattern', 'token'])
def test_model_lazy_activation_and_reset(grammar_model, trigger_kind):
    model = grammar_model
    opening = model.tokenize(b'{', add_bos=False)
    assert len(opening) == 1
    triggers = [r'[\s\S]*?(\{)[\s\S]*'] if trigger_kind == 'pattern' else opening
    grammar = 'answer ::= "{\\"ok\\":true}"'
    data = _internals.LlamaTokenDataArray(n_vocab=model.n_vocab())
    logits = np.zeros(model.n_vocab(), dtype=np.float32)
    invalid = model.tokenize(b'X', add_bos=False)[0]
    try:
        with _internals.GrammarSampler(model._model, grammar, lazy=True,
                                       triggers=triggers, root='answer') as sampler:
            def candidate_logit():
                data.copy_logits(logits)
                sampler.apply(ctypes.byref(data.candidates))
                return data._logit_view[invalid]
            assert np.isfinite(candidate_logit())
            for token in model.tokenize(b'prefix ', add_bos=False):
                sampler.accept(token)
            assert np.isfinite(candidate_logit())
            sampler.accept(opening[0])
            assert np.isneginf(candidate_logit())
            sampler.reset()
            assert np.isfinite(candidate_logit())
    finally:
        data.close()


@pytest.mark.parametrize('schema, accepted', [
    ({}, b'7'), ({'type': 'array', 'items': {}}, b'[7,"x",null]'),
])
def test_model_empty_schema_accepts_scalar_values(grammar_model, schema, accepted):
    model = grammar_model
    grammar = llama_cpp.LlamaGrammar.from_json_schema(schema)
    data = _internals.LlamaTokenDataArray(n_vocab=model.n_vocab())
    logits = np.zeros(model.n_vocab(), dtype=np.float32)
    try:
        with _internals.GrammarSampler(model._model, grammar.grammar) as sampler:
            for token in model.tokenize(accepted, add_bos=False):
                data.copy_logits(logits)
                sampler.apply(ctypes.byref(data.candidates))
                assert np.isfinite(data._logit_view[token])
                sampler.accept(token)
            data.copy_logits(logits)
            sampler.apply(ctypes.byref(data.candidates))
            assert np.isfinite(data._logit_view[model.token_eos()])
    finally:
        data.close()


def test_empty_response_schema_remains_object():
    from llama_cpp.llama_chat_format import _grammar_for_response_format
    grammar = _grammar_for_response_format({'type': 'json_object', 'schema': {}})
    assert 'root ::= object' in grammar.grammar.splitlines()


@pytest.mark.parametrize('function', [{}, {'parameters': None}, {'parameters': {}}])
def test_empty_tool_parameters_remain_empty_object(function):
    from llama_cpp.llama_chat_format import _tool_parameter_schema
    from llama_cpp.llama_grammar import json_schema_to_gbnf
    schema = _tool_parameter_schema(function)
    assert schema == {'type': 'object', 'properties': {}}
    assert 'root ::= "{" space "}"' in json_schema_to_gbnf(schema).splitlines()


def test_formatter_stop_token_boundary():
    # Verify that model-specific stop token IDs terminate generation.
    formatter = Jinja2ChatFormatter(
        template="{{ messages[0].content }}",
        eos_token="</s>",
        bos_token="",
        stop_token_ids=[248044],
    )
    response = formatter(messages=[{"role": "user", "content": "Hello"}])

    assert response.stopping_criteria is not None
    criterion = response.stopping_criteria[0]
    logits = np.empty(0, dtype=np.single)

    assert criterion(np.array([], dtype=np.intc), logits) is False
    assert criterion(np.array([1, 248044], dtype=np.intc), logits) is True
    assert criterion(np.array([1, 2], dtype=np.intc), logits) is False


def test_formatter_preserves_inputs_and_exposes_hf_template_context():
    messages = [{"role": "user", "content": [{"type": "text", "text": "<&中文"}]}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    original = copy.deepcopy(messages)
    formatter = Jinja2ChatFormatter(
        template=(
            "{{ bos_token }}{{ pad_token }}{% generation %}"
            "{{ [messages, tools, documents, add_generation_prompt] | tojson }}"
            "{% endgeneration %}"
        ),
        bos_token="<s>", eos_token="</s>",
        special_tokens_map={"pad_token": "<pad>", "bos_token": "incorrect"},
    )
    response = formatter(messages=messages, tools=tools, documents=[{"text": "source"}],
                         add_generation_prompt=False)
    assert response.prompt.startswith("<s><pad>")
    assert "<&中文" in response.prompt
    assert json.loads(response.prompt[len("<s><pad>"):]) == [messages, tools, [{"text": "source"}], False]
    assert response.stop == ["</s>"] and response.added_special
    assert messages == original
