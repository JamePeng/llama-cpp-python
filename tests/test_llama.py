import ctypes
import multiprocessing
import os
import pytest
import numpy as np
from scipy.special import log_softmax
from huggingface_hub import hf_hub_download

import llama_cpp
import llama_cpp._internals as internals
from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

from typing import (
    List,
    Dict,
)


MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


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
    params.use_mmap = llama_cpp.llama_supports_mmap()
    params.use_direct_io = False
    params.use_mlock = llama_cpp.llama_supports_mlock()
    params.check_tensors = False

    # 2. Load the Model
    model = internals.LlamaModel(path_model=llama_cpp_model_path, params=params)

    # 3. Setup Context Parameters
    cparams = llama_cpp.llama_context_default_params()
    cparams.n_ctx = 16
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
    n_eval = 0
    curr_tokens = tokens

    for _ in range(4):
        # Prepare batch with current tokens
        batch.set_batch(curr_tokens, n_past=n_eval, logits_all=False)

        # Decode (run inference)
        context.decode(batch)
        n_eval += len(curr_tokens)

        # Sample the next token (index -1 means the last token in the batch)
        token_id = sampler.sample(context, -1)

        # Accept the token to update internal sampler state
        sampler.accept(token_id)

        # Update loop variables
        curr_tokens = [token_id]
        result.append(token_id)

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


    output = model.create_completion(
        "Is the Louvre Museum in Paris, France?, 'yes' or 'no'?:\n",
        max_tokens=4,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        seed=1337,
        grammar=llama_cpp.LlamaGrammar.from_string("""
root ::= "yes" | "no"
""")
    )
    assert output["choices"][0]["text"] == "yes"

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
    Test Embedding Generation.
    Verifies that the model can produce vector embeddings.
    """
    model = LlamaEmbedding(
         model_path=llama_cpp_model_path,
         n_ctx=32,
         n_batch=32,
         n_ubatch=32,
         pooling_type=LLAMA_POOLING_TYPE_NONE)
    # Smoke test for now
    embeddings = model.embed("Hello, world!")
    assert isinstance(embeddings, list)
    assert len(embeddings) > 0
