import ctypes
import json
import multiprocessing
import threading
from types import SimpleNamespace

import pytest
import numpy as np
from huggingface_hub import hf_hub_download

import llama_cpp
import llama_cpp._internals as internals
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE
from llama_cpp.llama_speculative import _speculative_generation_timing_stats

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
    ("present_penalty", "presence_penalty", "expected"),
    [(0.0, 1.5, 1.5), (0.7, 1.5, 0.7)],
)
def test_completion_presence_penalty_alias(
    present_penalty, presence_penalty, expected
):
    llm = object.__new__(llama_cpp.Llama)
    private_calls = []

    def fake_private(**kwargs):
        private_calls.append(kwargs)
        yield {"choices": [{"text": "", "finish_reason": "length"}]}

    llm._create_completion = fake_private
    llama_cpp.Llama.create_completion(
        llm,
        [1],
        present_penalty=present_penalty,
        presence_penalty=presence_penalty,
    )

    assert private_calls[0]["present_penalty"] == expected

    # __call__ is a distinct public entry point but can share this forwarding
    # test instead of maintaining a second near-identical mock setup.
    public_calls = []

    def fake_public(**kwargs):
        public_calls.append(kwargs)
        return {"choices": [{"text": "", "finish_reason": "length"}]}

    llm.create_completion = fake_public
    llama_cpp.Llama.__call__(
        llm,
        "prompt",
        present_penalty=present_penalty,
        presence_penalty=presence_penalty,
    )

    assert public_calls[0]["presence_penalty"] == presence_penalty
    assert public_calls[0]["present_penalty"] == present_penalty


@pytest.mark.parametrize(
    ("present_penalty", "presence_penalty", "expected"),
    [(0.0, 1.5, 1.5), (0.7, 1.5, 0.7)],
)
def test_chat_completion_presence_penalty_alias(
    present_penalty, presence_penalty, expected
):
    llm = object.__new__(llama_cpp.Llama)
    handler_calls = []

    def fake_handler(**kwargs):
        handler_calls.append(kwargs)
        return {"choices": [{"message": {"content": ""}}]}

    llm.chat_handler = fake_handler
    llm._chat_handlers = {}
    llm.chat_format = None
    llama_cpp.Llama.create_chat_completion(
        llm,
        messages=[{"role": "user", "content": "Hello"}],
        present_penalty=present_penalty,
        presence_penalty=presence_penalty,
    )

    assert handler_calls[0]["present_penalty"] == expected
    assert "presence_penalty" not in handler_calls[0]


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


def test_context_manages_borrowed_threadpool_references(monkeypatch):
    context = internals.LlamaContext.__new__(internals.LlamaContext)
    context.ctx = object()
    context._threadpool_refs = None
    calls = []

    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_synchronize",
        lambda ctx: calls.append(("synchronize", ctx)),
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_attach_threadpool",
        lambda ctx, pool, batch_pool: calls.append(
            ("attach", ctx, pool, batch_pool)
        ),
    )
    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_detach_threadpool",
        lambda ctx: calls.append(("detach", ctx)),
    )

    pool = ctypes.c_void_p(1)
    context.attach_threadpool(pool)
    assert context._threadpool_refs == (pool, pool)
    assert calls[-2:] == [
        ("synchronize", context.ctx),
        ("attach", context.ctx, pool, None),
    ]

    context.detach_threadpool()
    assert context._threadpool_refs is None
    assert calls[-2:] == [
        ("synchronize", context.ctx),
        ("detach", context.ctx),
    ]


def test_context_manages_native_abort_callback_lifetime(monkeypatch):
    context = internals.LlamaContext.__new__(internals.LlamaContext)
    context.ctx = object()
    context._abort_callback_ref = None
    context._abort_callback_data_ref = None
    calls = []

    monkeypatch.setattr(
        internals.llama_cpp,
        "llama_set_abort_callback",
        lambda ctx, callback, data: calls.append((ctx, callback, data)),
    )

    data = ctypes.c_void_p(42)
    context.set_abort_callback(lambda user_data: user_data == data.value, data)

    callback = context._abort_callback_ref
    assert callback(data) is True
    assert context._abort_callback_data_ref is data
    assert calls[-1] == (context.ctx, callback, data)

    context.set_abort_callback(None)
    assert context._abort_callback_ref is None
    assert context._abort_callback_data_ref is None
    assert bool(calls[-1][1]) is False


def test_high_level_abort_updates_python_and_native_flags():
    llm = object.__new__(llama_cpp.Llama)
    llm.verbose = False
    llm._abort_event = threading.Event()
    llm._native_abort_flag = ctypes.c_bool(False)
    data = ctypes.cast(ctypes.pointer(llm._native_abort_flag), ctypes.c_void_p)

    assert llama_cpp.llama._llama_native_abort_callback(data) is False
    llm.abort()
    assert llm._abort_event.is_set()
    assert llama_cpp.llama._llama_native_abort_callback(data) is True


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


def test_llama_cpp_tokenization():
    """
    Test the tokenizer API (Llama.tokenize and Llama.detokenize).
    Verifies handling of BOS (Begin of Sentence), EOS (End of Sentence), and special tokens.
    """
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)

    try:
        text = b"Hello World"

        tokens = llama.tokenize(text)
        assert tokens == [llama.token_bos(), 15043, 2787]
        assert llama.detokenize(tokens)[1:] == text

        tokens = llama.tokenize(text, add_bos=False)
        assert tokens == [15043, 2787]
        assert llama.detokenize(tokens) == text

        text = b"Hello World</s>"
        assert llama.tokenize(text) == [1, 15043, 2787, 829, 29879, 29958]
        assert llama.tokenize(text, special=True) == [1, 15043, 2787, llama.token_eos()]

        tokens = llama.tokenize(b"", add_bos=True, special=True)
        assert tokens == [llama.token_bos()]
        assert llama.detokenize(tokens) == b""
    finally:
        llama.close()


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


@pytest.fixture(scope="module")
def llama_cpp_model_path():
    """Fixture to download a real GGUF model for integration tests."""
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    model_path = hf_hub_download(repo_id, filename)
    return model_path


@pytest.fixture(scope="module")
def shared_completion_model(llama_cpp_model_path):
    """Load the common completion model once for independent feature tests."""
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
    try:
        yield model
    finally:
        model.close()


@pytest.fixture
def completion_model(shared_completion_model):
    """Give each feature test a clean context without reloading model weights."""
    shared_completion_model.reset()
    try:
        yield shared_completion_model
    finally:
        shared_completion_model.reset()


def test_real_model(llama_cpp_model_path):
    """
    Test the Low-Level API (internals.*).
    This manually constructs the Model, Context, Batch, and Sampler Chain.
    """
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
    assert b"over" in output_text or b"lazy dog" in output_text


def test_real_llama_completion(completion_model):
    output = completion_model.create_completion(
        "The quick brown fox jumps",
        max_tokens=4,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        seed=1337
    )
    text = output["choices"][0]["text"]
    assert "over" in text or "lazy dog" in text


def test_grammar_sampling_safety(completion_model):
    """A strict grammar must produce a complete, parseable JSON object."""
    grammar_text = r'''
        root   ::= object
        object ::= "{" space pair "}"
        pair   ::= string ":" space value
        string ::= "\"" [a-z]+ "\""
        value  ::= number
        number ::= [0-9]+
        space  ::= [ ]?
    '''

    grammar = llama_cpp.LlamaGrammar.from_string(grammar_text)
    output = completion_model.create_completion(
        "Generate a JSON with age:",
        max_tokens=20,
        grammar=grammar,
        temperature=0.1
    )

    generated_text = output["choices"][0]["text"]
    parsed = json.loads(generated_text)
    assert len(parsed) == 1
    assert isinstance(next(iter(parsed.values())), int)

def test_logit_bias(completion_model):
    """A strong positive bias must force the selected token."""
    # Target token we want to force the model to generate
    target_word = " banana"           # Note the leading space — important for most tokenizers
    # Get the token ID corresponding to " banana" (Qwen-style tokenizer expected)
    target_token = completion_model.tokenize(target_word.encode("utf-8"), add_bos=False)[0]

    # Apply very strong positive bias to make this token extremely likely
    bias = {target_token: 100.0}

    # Generate a very short continuation with temperature=0 (greedy) + strong bias
    output = completion_model.create_completion(
        "I like to eat",
        max_tokens=3,
        logit_bias=bias,
        temperature=0.0
    )

    generated_text = output["choices"][0]["text"]
    assert "banana" in generated_text, f"Expected 'banana' in output, got: '{generated_text}'"


def test_custom_logits_processor(completion_model):
    """
    Test 4: Custom Logits Processor (Pure Python Implementation).

    Verifies that we can manipulate logits in Python before sampling.
    In this test, we suppress any token containing the letter 'e'.
    """
    def no_e_processor(input_ids, scores):
        """
        Filters out tokens containing 'e'.
        """
        for token_id in range(len(scores)):
            # Decode single token → get its string representation
            token_str = completion_model.detokenize([token_id]).decode("utf-8", errors="ignore")

            # Ban tokens that contain 'e' anywhere in their decoded form
            if "e" in token_str:
                scores[token_id] = -float("inf")

        return scores

    # Generate with greedy sampling (temperature=0) + our custom processor
    output = completion_model.create_completion(
        "The alphabet starts with",
        max_tokens=10,
        logits_processor=llama_cpp.LogitsProcessorList([no_e_processor]),
        temperature=0.0
    )

    generated_text = output["choices"][0]["text"]
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
