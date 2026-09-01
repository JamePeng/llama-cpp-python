# Low-level API examples

These examples use the raw `llama_cpp` bindings. For normal applications,
prefer the high-level `Llama` class; use this directory when you need direct
control over tokenization, batches, context memory, or sampler chains.

The generation examples intentionally target decoder-only text models. They do
not implement encoder-decoder, diffusion, embedding, or multimodal pipelines.

## What is included

| Example | Purpose |
| --- | --- |
| `generate.py` | Tokenize, decode, sample, and stream one completion |
| `chat.py` | Apply the GGUF chat template and keep conversation history |
| `reason_act.py` | Run a small model/tool loop with a safe calculator |
| `quantize.py` | Quantize a GGUF model with named quantization types |
| `runtime.py` | Shared resource and decoding helper used by the examples |

Run any command with `-h` to see all options. Paths containing spaces work
when quoted.

## Text generation

```bash
python examples/low_level_api/generate.py -m path/to/model.gguf -p "Explain why the sky is blue:" --n-ctx 4096 --max-tokens 256 --n-gpu-layers all
```

Use `--temperature 0` for greedy output. Add `--show-timings` to print the
native context performance report.

`--n-ctx` is the total context capacity shared by the prompt and generated
tokens. `--max-tokens` (short form `-n`) limits only the new tokens generated
for the current response. Their defaults in `generate.py` are `4096` and `256`.

`--n-gpu-layers auto` is the default. Use `all` to request full offload or an
integer to set an exact maximum.

## Interactive chat

Use an instruct/chat GGUF containing a supported chat template:

```bash
python examples/low_level_api/chat.py -m path/to/instruct-model.gguf --n-ctx 4096 --max-tokens 512 --system "You are a patient programming tutor."
```

Enter `/reset` to clear history or `/exit` to quit. If the conversation fills
the context, reset it or increase `--n-ctx`.

Chat templates may contain control-token text. The chat examples therefore
parse special tokens, while `generate.py` treats the same text literally.

## Reason and action

```bash
python examples/low_level_api/reason_act.py -m path/to/instruct-model.gguf --n-ctx 4096 --max-tokens 256 "What is (17 * 23) / 4?"
```

The calculator accepts numeric arithmetic only; it never evaluates arbitrary
Python code.

## Quantization

The output file must not already exist:

```bash
python examples/low_level_api/quantize.py model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

Use `--dry-run` to estimate the operation without writing the quantized model.

The commands are shown on one line so they work unchanged in PowerShell and
POSIX shells. In PowerShell, use the backtick character instead of `\` if you
choose to split a command across multiple lines.

## Native logging

All four commands support the same native runtime logging options. With neither
option, logging keeps the llama.cpp-style default verbosity of `3` (`info`).

- `--verbose` is the compatibility switch and enables verbosity `5` (`debug`).
- `--verbosity LEVEL` provides fine-grained control and takes precedence over
  `--verbose` when both are present.

`LEVEL` accepts either a number or a name: `0`/`output`, `1`/`error`,
`2`/`warning`, `3`/`info`, `4`/`trace`, or `5`/`debug`.

```bash
# Keep warnings and errors only.
python examples/low_level_api/generate.py -m model.gguf --verbosity warning

# Show detailed native backend and model-loading logs.
python examples/low_level_api/generate.py -m model.gguf --verbose

# --verbosity wins, so this uses info rather than debug.
python examples/low_level_api/generate.py -m model.gguf --verbose --verbosity info
```

## API flow

The generation examples follow the current low-level lifecycle:

1. Initialize the process-global backend and discover packaged backend plugins.
2. Obtain its vocabulary and create a context.
3. Tokenize input and submit `llama_batch` objects with `llama_decode`.
4. Select tokens through a `llama_sampler_chain`.
5. Free the sampler, batch, context, and model in reverse order.

The backend remains initialized for the life of the process, matching the
high-level `Llama` implementation. This keeps multiple model instances from
invalidating one another.

## Batch sizing

`--n-batch` limits the logical prompt chunk submitted to `llama_decode`.
`--n-ubatch auto` limits the physical graph to `min(n_batch, 512)`. For an
explicit value, keep `n_ubatch <= n_batch <= n_ctx`; validation enforces this.

If context memory cannot fit a logical chunk, `runtime.py` halves that chunk
and retries. Other native decode status codes are reported without retrying.
