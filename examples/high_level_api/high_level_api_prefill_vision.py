"""Decision one token without sampling (with vision).
Gemma4 model only.
"""
import numpy as np
import argparse
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "media" / "apple.jpg"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-mm", "--mmproj", type=str)
parser.add_argument("-i", "--image", type=str, default=IMAGE_PATH)
args = parser.parse_args()

# Use gemma4 handler
chat_handler = Gemma4ChatHandler(
    clip_model_path=args.mmproj,
    enable_thinking=False,
)

llm = Llama(
    model_path=args.model,
    chat_handler=chat_handler,
    n_ctx=4096,
)

messages = [
    {
        "role": "system",
        "content": "Answer with exactly Y or N."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": args.image.as_uri(),
                },
            },
            {
                "type": "text",
                "text": "Is there any visible written word in this image? "
                        "Answer with exactly one character: Y or N."
            },
        ],
    },
]

result = llm.create_chat_prefill(
  messages=messages,
)

result_chat = llm.create_chat_completion(
  messages=messages,
)

print(result_chat)

print("Top 20 tokens: ")
top_k = 20
top_ids = np.argsort(result.logits)[-top_k:][::-1]

for token_id in top_ids:
    token = llm.detokenize([int(token_id)])
    print(
        repr(token),
        float(result.logits[token_id]),
    )

y_tokens = llm.tokenize(b"Y", add_bos=False, special=False)
n_tokens = llm.tokenize(b"N", add_bos=False, special=False)

assert len(y_tokens) == 1
assert len(n_tokens) == 1

y_token = y_tokens[0]
n_token = n_tokens[0]

y_logit = float(result.logits[y_token])
n_logit = float(result.logits[n_token])

print("Y logit:", y_logit)
print("N logit:", n_logit)

# Softmax in Y/N
candidate_logits = np.array([y_logit, n_logit], dtype=np.float32)
candidate_probs = np.exp(candidate_logits - candidate_logits.max())
candidate_probs /= candidate_probs.sum()

print("P(Y | {Y,N}) =", float(candidate_probs[0]))
print("P(N | {Y,N}) =", float(candidate_probs[1]))
