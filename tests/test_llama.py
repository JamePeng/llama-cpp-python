import ctypes
import multiprocessing
import os
import threading
from types import SimpleNamespace
import pytest
import numpy as np
from scipy.special import log_softmax
from huggingface_hub import hf_hub_download

import llama_cpp
import llama_cpp._internals as internals
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE
from llama_cpp.llama_speculative import _speculative_generation_timing_stats

from typing import (
    List,
    Dict,
)


MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def test_speculative_generation_timing_reports_sustained_throughput():
    stats = _speculative_generation_timing_stats(
        generated_tokens=384,
        time_to_first_token_seconds=0.043,
        time_to_last_token_seconds=4.081,
    )

    assert stats["generation_tokens"] == 384
    assert stats["generation_seconds"] == pytest.approx(4.081)
    assert stats["generation_tokens_per_second"] == pytest.approx(384 / 4.081)
    assert stats["time_to_first_token_seconds"] == pytest.approx(0.043)
    assert stats["sustained_tokens"] == 383
    assert stats["sustained_seconds"] == pytest.approx(4.038)
    assert stats["sustained_tokens_per_second"] == pytest.approx(383 / 4.038)


def test_completion_public_apis_forward_ignore_eos():
    llm = object.__new__(llama_cpp.Llama)
    private_calls = []

    def fake_private(**kwargs):
        private_calls.append(kwargs)
        yield {"choices": [{"text": "", "finish_reason": "length"}]}

    llm._create_completion = fake_private
    llama_cpp.Llama.create_completion(llm, [1], ignore_eos=True)
    assert private_calls[0]["ignore_eos"] is True

    public_calls = []

    def fake_public(**kwargs):
        public_calls.append(kwargs)
        return {"choices": [{"text": "", "finish_reason": "length"}]}

    llm.create_completion = fake_public
    llama_cpp.Llama.__call__(llm, "prompt", ignore_eos=True)
    assert public_calls[0]["ignore_eos"] is True


@pytest.mark.parametrize(
    ("ignore_eos", "expected_text", "expected_finish_reason"),
    [(False, "", "stop"), (True, "<eog>", "length")],
)
def test_private_completion_respects_ignore_eos_at_eog_boundary(
    monkeypatch, ignore_eos, expected_text, expected_finish_reason
):
    llm = object.__new__(llama_cpp.Llama)
    forwarded = []

    def fake_generate(tokens, **kwargs):
        assert tokens == [1]
        forwarded.append(kwargs["ignore_eos"])
        yield 2

    llm._model = SimpleNamespace(
        vocab=object(),
        token_bos=lambda: 1,
        token_eos=lambda: 2,
        token_sep=lambda: -1,
        token_fim_pre=lambda: -1,
        token_fim_mid=lambda: -1,
        token_fim_suf=lambda: -1,
        get_add_sep=lambda: False,
    )
    llm._abort_event = threading.Event()
    llm.metadata = {}
    llm.spm_infill = False
    llm.verbose = False
    llm._n_ctx = 8
    llm._logits_all = False
    llm._seed = 0
    llm.cache = None
    llm.model_path = "fake.gguf"
    llm.input_ids = np.zeros(8, dtype=np.intc)
    llm.n_tokens = 1
    llm.scores = np.zeros((1, 4), dtype=np.float32)
    llm.generate = fake_generate
    llm.detokenize = (
        lambda tokens, prev_tokens=None, **kwargs: b"".join(
            b"<eog>" if token == 2 else b"x" for token in tokens
        )
    )
    monkeypatch.setattr(
        "llama_cpp.llama.llama_cpp_lib.llama_token_is_eog",
        lambda vocab, token: token == 2,
    )

    result = next(
        llm._create_completion(
            prompt=[1], max_tokens=1, ignore_eos=ignore_eos
        )
    )

    assert forwarded == [ignore_eos]
    assert result["choices"][0]["text"] == expected_text
    assert result["choices"][0]["finish_reason"] == expected_finish_reason


@pytest.mark.parametrize("generated_tokens", [0, 1])
def test_speculative_generation_timing_has_no_sustained_rate_for_short_output(
    generated_tokens,
):
    stats = _speculative_generation_timing_stats(
        generated_tokens=generated_tokens,
        time_to_first_token_seconds=0.025,
        time_to_last_token_seconds=0.025,
    )

    assert stats["sustained_tokens"] == 0
    assert stats["sustained_seconds"] == 0.0
    assert stats["sustained_tokens_per_second"] == 0.0


def test_model_init_frees_native_model_when_vocab_lookup_fails(monkeypatch):
    native_model_handle = object()
    freed_model_handles = []

    def model_path_exists(_path):
        return True

    def load_native_model(_path, _params):
        return native_model_handle

    def fail_to_get_model_vocab(_model_handle):
        return None

    def record_model_free(model_handle):
        freed_model_handles.append(model_handle)

    monkeypatch.setattr(internals.os.path, "exists", model_path_exists)
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_load_from_file",
        load_native_model,
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_get_vocab",
        fail_to_get_model_vocab,
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_free",
        record_model_free,
    )

    with pytest.raises(ValueError, match="Failed to get vocab"):
        internals.LlamaModel(
            path_model="model.gguf",
            params=object(),
            verbose=False,
        )

    assert freed_model_handles == [native_model_handle]


def test_batch_init_frees_native_batch_when_validation_fails(monkeypatch):
    class InvalidMixedNativeBatch:
        token = object()
        embd = object()

    invalid_mixed_batch = InvalidMixedNativeBatch()
    freed_batch_handles = []

    def allocate_invalid_mixed_batch(_n_tokens, _embd, _n_seq_max):
        return invalid_mixed_batch

    def record_batch_free(batch_handle):
        freed_batch_handles.append(batch_handle)

    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_batch_init",
        allocate_invalid_mixed_batch,
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_batch_free",
        record_batch_free,
    )

    with pytest.raises(RuntimeError, match="expected batch.token to be NULL"):
        internals.LlamaBatch(
            n_tokens=1,
            embd=1,
            n_seq_max=1,
            mixed=True,
            verbose=False,
        )

    assert freed_batch_handles == [invalid_mixed_batch]


def test_context_close_releases_parent_references():
    context = internals.LlamaContext.__new__(internals.LlamaContext)
    context.ctx = None
    context.model = object()
    context.params = object()
    context._exit_stack = None

    context.close()
    context.close() # Closing an already closed context must be a no-op.

    assert context.model is None
    assert context.params is None


def test_sampling_context_partial_init_can_close_idempotently(monkeypatch):
    closed_resources = []

    class MinimalModelForSampling:
        model = object()
        verbose = False

        def n_vocab(self):
            return 8

    class TrackedTokenDataArray:
        def __init__(self, *, n_vocab):
            assert n_vocab == 8

        def close(self):
            closed_resources.append("token-data")

    class TrackedSamplerChain:
        def close(self):
            closed_resources.append("sampler-chain")

    def get_sampling_vocab(_model_handle):
        return object()

    def fail_sampler_chain_build(_sampling_context):
        raise RuntimeError("sampler chain build failed")

    monkeypatch.setattr(internals, "LlamaTokenDataArray", TrackedTokenDataArray)
    monkeypatch.setattr(internals, "LlamaSampler", TrackedSamplerChain)
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_model_get_vocab",
        get_sampling_vocab,
    )
    monkeypatch.setattr(
        internals.LlamaSamplingContext,
        "_build_sampler_chain",
        fail_sampler_chain_build,
    )

    sampling_context = internals.LlamaSamplingContext.__new__(
        internals.LlamaSamplingContext
    )
    with pytest.raises(RuntimeError, match="sampler chain build failed"):
        sampling_context.__init__(
            params=internals.LlamaSamplingParams(),
            model=MinimalModelForSampling(),
        )

    sampling_context.close()
    sampling_context.close() # Closing an already closed context must be a no-op.

    # Full-vocabulary candidate storage is lazy and was never needed because
    # sampler-chain construction failed first.
    assert closed_resources == ["sampler-chain"]
    assert sampling_context.model is None
    assert sampling_context.params is None
    assert sampling_context.vocab is None


def test_llama_cpp_version():
    assert llama_cpp.__version__


def test_llama_cpp_tokenization():
    """
    Test the tokenizer API (Llama.tokenize and Llama.detokenize).
    Verifies handling of BOS (Begin of Sentence), EOS (End of Sentence), and special tokens.
    """
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)

    assert llama
    assert llama._ctx.ctx is not None

    text = b"Hello World"

    tokens = llama.tokenize(text)
    assert tokens[0] == llama.token_bos()
    assert tokens == [1, 15043, 2787]
    detokenized = llama.detokenize(tokens)
    assert detokenized[1:] == text

    tokens = llama.tokenize(text, add_bos=False)
    assert tokens[0] != llama.token_bos()
    assert tokens == [15043, 2787]

    detokenized = llama.detokenize(tokens)
    assert detokenized == text

    text = b"Hello World</s>"
    tokens = llama.tokenize(text)
    assert tokens[-1] != llama.token_eos()
    assert tokens == [1, 15043, 2787, 829, 29879, 29958]

    tokens = llama.tokenize(text, special=True)
    assert tokens[-1] == llama.token_eos()
    assert tokens == [1, 15043, 2787, 2]

    text = b""
    tokens = llama.tokenize(text, add_bos=True, special=True)
    assert tokens[-1] != llama.token_eos()
    assert tokens == [llama.token_bos()]
    assert text == llama.detokenize(tokens)


def test_llama_batch_seq_id_error_guidance():
    """Sequence-capacity errors should explain how to fix parallel batching."""
    batch = internals.LlamaBatch(
        n_tokens=2,
        embd=0,
        n_seq_max=1,
        verbose=False,
    )
    try:
        with pytest.raises(ValueError) as exc_info:
            batch.add_sequence(
                token_array=[1],
                pos_array=[0],
                seq_ids=[1],
                logits_array=[True],
            )

        message = str(exc_info.value)
        assert "n_seq_max=1" in message
        assert "valid IDs are 0 through 0" in message
        assert "n_seq_max>=2" in message
        assert "LlamaEmbedding" in message
    finally:
        batch.close()


def test_llama_batch_mixed_embeddings_are_copied_contiguously():
    batch = internals.LlamaBatch(
        n_tokens=2,
        embd=4,
        n_seq_max=1,
        mixed=True,
        verbose=False,
    )
    try:
        first = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        second = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        batch.add_token_embedding(10, first, 0, [0], False)
        batch.add_token_embedding(11, second, 1, [0], True)

        actual = np.ctypeslib.as_array(batch.batch.embd, shape=(8,)).copy()
        np.testing.assert_array_equal(actual, np.concatenate((first, second)))
        assert batch.batch.n_tokens == 2
        assert [batch.batch.token[i] for i in range(2)] == [10, 11]
    finally:
        batch.close()


def test_llama_batch_mrope_embeddings_use_four_position_planes():
    batch = internals.LlamaBatch(
        n_tokens=3,
        embd=2,
        n_seq_max=1,
        verbose=False,
    )
    try:
        original_pos = ctypes.cast(batch.batch.pos, ctypes.c_void_p).value
        batch.enable_mrope_positions()
        expanded_pos = ctypes.cast(batch.batch.pos, ctypes.c_void_p).value
        batch.add_embeddings_mrope(
            [1.0, 2.0, 3.0, 4.0],
            pos_array=[[4, 5], [4, 5], [4, 5], [0, 0]],
            seq_ids=[0],
            logits_array=[False, True],
        )

        assert expanded_pos != original_pos
        assert [batch.batch.pos[i] for i in range(8)] == [
            4, 5, 4, 5, 4, 5, 0, 0
        ]
        assert [batch.batch.logits[i] for i in range(2)] == [0, 1]
        assert batch.batch.n_tokens == 2
    finally:
        batch.close()


@pytest.fixture
def llama_cpp_model_path():
    """Fixture to download a real GGUF model for integration tests."""
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    model_path = hf_hub_download(repo_id, filename)
    return model_path


def test_real_model(llama_cpp_model_path):
    """
    Test the Low-Level API (internals.*).
    This manually constructs the Model, Context, Batch, and Sampler Chain.
    """
    assert os.path.exists(llama_cpp_model_path)

    # 1. Setup Model Parameters
    params = llama_cpp.llama_model_default_params()
    params.check_tensors = False

    # 2. Load the Model
    model = internals.LlamaModel(path_model=llama_cpp_model_path, params=params)

    # 3. Setup Context Parameters
    cparams = llama_cpp.llama_context_default_params()
    cparams.n_ctx = 32
    cparams.n_batch = 16
    cparams.n_ubatch = 16
    cparams.n_threads = multiprocessing.cpu_count()
    cparams.n_threads_batch = multiprocessing.cpu_count()
    cparams.swa_full = True
    cparams.kv_unified = True

    # 4. Create the Context
    context = internals.LlamaContext(model=model, params=cparams)
    tokens = model.tokenize(b"Hello, world!", add_bos=True, special=True)

    assert tokens == [9707, 11, 1879, 0]

    # New prompt for generation test
    tokens = model.tokenize(b"The quick brown fox jumps", add_bos=True, special=True)

    batch = internals.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

    seed = 1337
    sampler = internals.LlamaSampler()
    sampler.add_top_k(50)
    sampler.add_top_p(0.9, 1)
    sampler.add_temp(0.8)
    sampler.add_dist(seed)

    result = list(tokens)
    n_eval = len(tokens)
    batch.reset()
    pos_array = list(range(n_eval))
    logits_array = [False] * (n_eval - 1) + [True]

    batch.add_sequence(
        token_array=tokens,
        pos_array=pos_array,
        seq_ids=[0],
        logits_array=logits_array
    )
    context.decode(batch)

    for _ in range(4):
        token_id = sampler.sample(context, -1)
        sampler.accept(token_id)
        result.append(token_id)

        batch.reset()

        batch.add_token(
            token=token_id,
            pos=n_eval,
            seq_ids=[0],
            logits=True
        )

        context.decode(batch)
        n_eval += 1

    output = result[len(tokens):]
    output_text = model.detokenize(output, special=True)
    print(output_text)
    assert b"over" in output_text or b"lazy dog" in output_text

def test_real_llama(llama_cpp_model_path):
    model = llama_cpp.Llama(
        llama_cpp_model_path,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        logits_all=False,
        swa_full=True,
        kv_unified=True,
    )

    # 1. Basic Completion Test
    output = model.create_completion(
        "The quick brown fox jumps",
        max_tokens=4,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        seed=1337
    )
    text = output["choices"][0]["text"]
    assert "over" in text or "lazy dog" in text

    # 2. Grammar Constraint Test (Updated: Coin Flip)
    # We verify that the model ONLY outputs "heads" or "tails".
    # This tests the sampler mechanism, not the model's intelligence.
    output = model.create_completion(
        "Flip a coin: heads or tails? Result:",
        max_tokens=4,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        seed=1337,
        grammar=llama_cpp.LlamaGrammar.from_string("""
            root ::= "heads" | "tails"
        """)
    )

    generated_text = output["choices"][0]["text"]
    print(f"\n[Grammar Coin Flip] Output: {generated_text}")

    # Assert that the output is strictly one of the allowed grammar options
    assert generated_text in ["heads", "tails"], \
        f"Grammar failed! Expected 'heads' or 'tails', got: '{generated_text}'"

    # 3. Logit Bias Test
    suffix = b"rot"
    tokens = model.tokenize(suffix, add_bos=True, special=True)
    logit_bias: Dict[int, float] = {}

    for token_id in tokens:
        logit_bias[token_id] = 1000

    output = model.create_completion(
        "The capital of france is par",
        max_tokens=4,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        seed=1337,
        logit_bias=logit_bias
    )

    assert output["choices"][0]["text"].lower().startswith("rot")

def test_grammar_sampling_safety(llama_cpp_model_path):
    """
    Test 2: Grammar-constrained sampling (safety / stability check)
    This test forces very strict JSON-like output using a minimal grammar.
    """
    # Very restrictive grammar — only allows simple { "key": number }
    # (intentionally limited to trigger potential accept-stage bugs)
    model = llama_cpp.Llama(
        llama_cpp_model_path,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        logits_all=False,
        swa_full=True,
        kv_unified=True,
    )
    grammar_text = r'''
        root   ::= object
        object ::= "{" space pair "}"
        pair   ::= string ":" space value
        string ::= "\"" [a-z]+ "\""
        value  ::= number
        number ::= [0-9]+
        space  ::= [ ]?
    '''

    # Create grammar object from string definition
    grammar = llama_cpp.LlamaGrammar.from_string(grammar_text)

    # Prompt that naturally wants to produce something JSON-like
    prompt = "Generate a JSON with age:"

    # Generate with grammar constraint + near-greedy sampling
    output = model.create_completion(
        prompt,
        max_tokens=20,
        grammar=grammar,
        temperature=0.1
    )

    generated_text = output["choices"][0]["text"]
    print(f"\n[Grammar] Output: {generated_text}")

    # Basic structural validation (we don't parse full JSON here — just checking survival + minimal shape)
    assert "{" in generated_text and "}" in generated_text, \
        "Generated text is missing JSON object braces"
    assert ":" in generated_text, \
        "Generated text is missing key-value separator (:)"

def test_logit_bias(llama_cpp_model_path):
    """
    Test 3: Logit Bias
    Verifies that specific tokens can be forced using logit bias.
    """
    # Load model with minimal context to save memory (just for tokenization & small generation)
    model = llama_cpp.Llama(
        llama_cpp_model_path,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        logits_all=False,
        swa_full=True,
        kv_unified=True,
    )

    # Target token we want to force the model to generate
    target_word = " banana"           # Note the leading space — important for most tokenizers
    # Get the token ID corresponding to " banana" (Qwen-style tokenizer expected)
    target_token = model.tokenize(target_word.encode("utf-8"), add_bos=False)[0]

    # Apply very strong positive bias to make this token extremely likely
    bias = {target_token: 100.0}

    # Generate a very short continuation with temperature=0 (greedy) + strong bias
    output = model.create_completion(
        "I like to eat",
        max_tokens=3,
        logit_bias=bias,
        temperature=0.0
    )

    # Extract generated text
    generated_text = output["choices"][0]["text"]
    print(f"\n[Bias] Output: {generated_text}")

    # Verify that our forced token actually appeared in the output
    assert "banana" in generated_text, f"Expected 'banana' in output, got: '{generated_text}'"


def test_custom_logits_processor(llama_cpp_model_path):
    """
    Test 4: Custom Logits Processor (Pure Python Implementation).

    Verifies that we can manipulate logits in Python before sampling.
    In this test, we suppress any token containing the letter 'e'.
    """
    # Load model with minimal context to save memory (just for tokenization & small generation)
    model = llama_cpp.Llama(
        llama_cpp_model_path,
        n_ctx=64,
        n_batch=32,
        n_ubatch=32,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        logits_all=False,
        swa_full=True,
        kv_unified=True,
    )

    def no_e_processor(input_ids, scores):
        """
        Filters out tokens containing 'e'.
        """
        for token_id in range(len(scores)):
            # Decode single token → get its string representation
            token_str = model.detokenize([token_id]).decode("utf-8", errors="ignore")

            # Ban tokens that contain 'e' anywhere in their decoded form
            if "e" in token_str:
                scores[token_id] = -float("inf")

        return scores

    # Generate with greedy sampling (temperature=0) + our custom processor
    output = model.create_completion(
        "The alphabet starts with",
        max_tokens=10,
        logits_processor=llama_cpp.LogitsProcessorList([no_e_processor]),
        temperature=0.0
    )

    generated_text = output["choices"][0]["text"]
    print(f"\n[Custom] Output (No 'e'): {generated_text}")

    # Basic validation: make sure no 'e' appears in the generated text
    assert "e" not in generated_text, \
        f"Expected no letter 'e' in output, but found one:\n  Output was: '{generated_text}'"

def test_real_llama_embeddings(llama_cpp_model_path):
    """
    Test embedding generation through the specialized LlamaEmbedding class.
    """
    model = LlamaEmbedding(
        model_path=llama_cpp_model_path,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
    )
    try:
        # The inherited n_seq_max=1 processes this list as three streaming
        # decode batches instead of assigning an invalid seq_id.
        embeddings = model.embed(["Hello", "world", "embedding"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(embedding) > 0 for embedding in embeddings)
    finally:
        model.close()


def test_real_llama_base_embedding_api(llama_cpp_model_path):
    """
    Test the maintained embedding API on the standard Llama class.

    Covers pre-tokenized batching, normalization, separator-based string
    batching, token counts, and the OpenAI-compatible response wrapper.
    """
    model = llama_cpp.Llama(
        model_path=llama_cpp_model_path,
        embeddings=True,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        n_seq_max=2,
        kv_unified=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
        verbose=False,
    )

    try:
        token_inputs = [
            model.tokenize(b"Hello"),
            model.tokenize(b"world"),
        ]
        embeddings, token_count = model.embed(
            token_inputs,
            normalize=True,
            return_count=True,
        )

        assert len(embeddings) == len(token_inputs)
        assert token_count == sum(map(len, token_inputs))
        assert len(embeddings[0]) == len(token_inputs[0])
        assert np.linalg.norm(embeddings[0][0]) == pytest.approx(1.0)

        split_embeddings = model.embed(
            "Hello\nworld",
            separator="\n",
            normalize=False,
        )
        assert len(split_embeddings) == 2

        response = model.create_embedding(
            ["Hello", "world"],
            normalize=2,
        )
        assert response["object"] == "list"
        assert len(response["data"]) == 2
        assert response["usage"]["prompt_tokens"] > 0
        assert response["usage"]["total_tokens"] == response["usage"]["prompt_tokens"]
    finally:
        model.close()
