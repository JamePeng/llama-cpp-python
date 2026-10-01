"""Decision one token without sampling
"""
import argparse
import numpy as np
from llama_cpp import Llama, LlamaGrammar

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

Y: Yes
N; No
"""

result = llm.create_chat_prefill(
    messages=[{"role": "user", "content": prompt}],
    grammar=LlamaGrammar.from_string('root ::= "Y" | "N"'),
)

answer_probs = {}
for token_id, prob in result.probabilities.items():
    text = llm.detokenize([token_id]).decode("utf-8")
    answer_probs[text] = answer_probs.get(text, 0.0) + prob

print("Y:", answer_probs["Y"])
print("N:", answer_probs["N"])
