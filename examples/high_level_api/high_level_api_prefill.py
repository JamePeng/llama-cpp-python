"""Decision one token without sampling
"""
import argparse
import numpy as np
from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="../models/7B/ggml-model.bin")
args = parser.parse_args()

llm = Llama(
    model_path=args.model,
    n_ctx=2048,
)

prompt = """\
Question: Is the following statement true?

2 + 2 = 4

Answer with exactly Y or N.
Answer:
"""

result = llm.prefill(prompt)

y_tokens = llm.tokenize(b"Y", add_bos=False, special=False)
n_tokens = llm.tokenize(b"N", add_bos=False, special=False)

assert len(y_tokens) == 1
assert len(n_tokens) == 1

y_token = y_tokens[0]
n_token = n_tokens[0]

# pull Y/N logit from vocab
y_logit = float(result.logits[y_token])
n_logit = float(result.logits[n_token])

print("Y logit:", y_logit)
print("N logit:", n_logit)

# softmax from Y/N only
candidate_logits = np.array([y_logit, n_logit])
candidate_probs = np.exp(candidate_logits - candidate_logits.max())
candidate_probs /= candidate_probs.sum()

print("Y:", candidate_probs[0])
print("N:", candidate_probs[1])