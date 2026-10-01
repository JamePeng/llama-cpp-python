"""Decision one token without sampling (with vision).
Gemma4 model only.
"""
import argparse
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Gemma4ChatHandler

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "media" / "apple.jpg"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-mm", "--mmproj", type=str)
parser.add_argument("-i", "--image", type=Path, default=IMAGE_PATH)
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
    grammar=LlamaGrammar.from_string('root ::= "Y" | "N"'),
)

answer_probs = {}
for token_id, prob in result.probabilities.items():
    text = llm.detokenize([token_id]).decode("utf-8")
    answer_probs[text] = answer_probs.get(text, 0.0) + prob

print("Y:", answer_probs["Y"])
print("N:", answer_probs["N"])
