# Speech synthesis with MTMD

`MTMDAudioGenerator` provides experimental, non-streaming speech synthesis using
the vendored `mtmd_helper_gen_audio_*` API. It supports Qwen3-TTS and Pocket TTS
backbones with matching audio generation mmproj files. Audio-input models such
as Qwen3-ASR are not TTS models.

For Qwen3-TTS, use the Base variant supported by the vendor example.
CustomVoice is not currently supported end to end: its checkpoint lacks the
speaker encoder expected by the mmproj converter, and the generation helper
does not expose preset speaker IDs. Loading its backbone alone is insufficient.

## Usage

For an interactive UI with reference upload/recording, playback, downloads and
GPU/FA controls, see the [Streamlit TTS playground](../../../../examples/streamlit_tts/README.md).
Its model paths are placeholders; configure matching GGUF files before use.

```python
from contextlib import closing
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
from llama_cpp.llama_multimodal import MTMDAudioGenerator

with closing(Llama(
    model_path="/path/to/model/qwen3-tts-base.gguf",
    embeddings=True,
    pooling_type=LLAMA_POOLING_TYPE_NONE,
    n_ctx=4096,
)) as llama:
    with MTMDAudioGenerator(mmproj_path="/path/to/model/mmproj-qwen3-tts-base.gguf", use_gpu=True) as generator:
        audio = generator.create_speech(
            llama=llama,
            text="Hello, welcome to speech synthesis.",
            language="en",
            speaker_reference="speaker.wav",
            seed=42,
        )
        audio.save("output.wav")
        print(audio.duration, audio.finish_reason)
```

Use `embeddings=True` (plural). The context must expose token embeddings with
pooling set to `LLAMA_POOLING_TYPE_NONE`. Use a dedicated `Llama`: every request
clears its KV and Python token state before decoding and again on completion or
failure. A generator binds to the first Llama it uses and cannot be rebound.
Close the generator before closing that Llama, as in the nested contexts above.
The generator does not own or close the Llama.

`speaker_reference` accepts a path, HTTP(S) URL, data URI, or encoded WAV/MP3/FLAC
bytes. Raw sample arrays are not accepted. `from_pretrained()` is inherited from
`MTMDBaseHandler` and downloads the mmproj; load the backbone separately.

## Model options

| Option | Qwen3-TTS | Pocket TTS |
| --- | --- | --- |
| Reference audio | Optional | Required |
| Language | `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru`, or names accepted by the helper | Selected by weights; omit `language` |
| Token sampling | Uses temperature, top-k, top-p, min-p and repetition penalty | Skipped |
| Flow/decoder settings | Managed by vendor | Managed by vendor, including variant settings |

Defaults are `temperature=0.8`, `top_k=40`, `top_p=0.95`, `min_p=0.05`, and
`repeat_penalty=1.05`. The repetition window uses the active context size.
Qwen's `top_k` and `top_p` also configure the helper's code predictor. A
nonpositive `top_k` disables the backbone top-k filter but selects the helper's
default for the code predictor. Temperature only controls backbone sampling.
Pocket ignores these sampling options; `seed` still reaches its helper.
`seed=None` uses the native random-seed sentinel. Identical seeds do not promise
bitwise identical audio across hardware or backend versions.

Pocket weights must be converted from the appropriate `languages/<name>`
directory, following the [vendor TTS documentation](../../../../vendor/llama.cpp/tools/tts/README.md).

### Qwen voice cloning limitations

The current vendor pipeline uses reference audio only to extract a speaker
embedding, corresponding to the speaker-only approach of upstream
`x_vector_only_mode=True`. It does **not** support `ref_text` or full voice
cloning conditioned on both reference audio and its transcript. Neither
`ref_text` nor `x_vector_only_mode` is exposed by `create_speech`; the cloning
mode cannot currently be switched. Voice similarity may be lower than with
the upstream transcript-conditioned mode.

For reference input, use an encoded audio file path, URL, data URI, or audio
bytes. Bare base64 strings and `(numpy_array, sample_rate)` tuples are not
accepted directly.

## Results and limits

`GeneratedAudio` owns its `data` bytes after the request returns:

- `format="wav"`: complete mono PCM16 little-endian WAV, the default.
- `format="pcm_f32"`: raw native-endian float32 mono samples, without a header.
- `sample_rate` and `n_samples`: native output metadata; `duration` is seconds.
- `finish_reason="stop"`: the helper signalled end-of-speech.
- `finish_reason="length"`: the generation-step budget was exhausted; speech
  may be cut off. Increase `max_frames` and provide sufficient context capacity.

`save()` writes these bytes unchanged. A `.wav` extension does not convert raw
PCM to WAV. `max_frames=512` bounds helper steps, not text tokens or seconds;
Pocket's chunk transitions also consume helper steps. There is no automatic
context shifting in this wrapper. Native decode failures raise `RuntimeError`.

Calls on one generator cannot overlap, and it cannot be closed while generating.
Do not use or close the bound Llama concurrently. `llama.abort()` can interrupt
an active request from another thread; synthesis raises `InterruptedError` and
cleans up. Cancellation is checked between native calls and does not guarantee
immediate interruption inside the mmproj encoder or vocoder.

## Implementation and validation

The wrapper reuses `MTMDBaseHandler` for media and context management. It drives
`set_input`, batched `step_prompt`, hidden-state feedback through `step_gen`, and
one final `get_output`. Qwen uses a request-local `LlamaSamplingContext`; Pocket
passes `LLAMA_TOKEN_NULL`. Prompt construction, codebooks, text segmentation and
vocoder state remain in the vendor helper. No text completion loop is involved.

The helper currently returns accumulated audio and flushes pending features in
`get_output`. This wrapper therefore returns complete audio and does not expose
streaming or incremental text input.

Run the CLI-style Python example:

```sh
python -m examples.high_level_api.mtmd_tts --model /path/to/model/backbone.gguf --mmproj /path/to/model/mmproj.gguf --text "Hello world" --speaker speaker.wav --output output.wav
```

The example also accepts repeated `--text` arguments and `--demo daily|game|houston`
for sequential requests using the same loaded model. `--demo multilingual` supplies
Chinese, English and Japanese requests for Qwen. Multiple outputs receive numbered
suffixes. Use `--speaker-bytes` with a local `--speaker` file to demonstrate encoded
reference bytes, or `--format pcm_f32 --output speech.pcm` for raw audio with a JSON
metadata sidecar. Run `--help` for usage examples.

For Qwen, optionally add `--language en`. For Pocket, omit it. Compare the output
with `vendor/llama.cpp/tools/tts/tts.cpp` using matching weights, input and sampling
settings. Check audible content, WAV metadata, repeated requests and truncated
outputs. Mock pipeline tests cover control flow and ownership, but do not verify
model quality or native model compatibility; real-weight validation is required.

The generator validates WAV metadata and sample lengths, and rejects empty,
non-finite, out-of-range, constant, or >=99% full-scale clipped audio with
`RuntimeError`. These checks detect broken output, not pronunciation or voice
quality. Quiet but varying audio is accepted. Rejected requests still clean up
their generation state and can be followed by a new request.

### Flash Attention

`MTMDAudioGenerator` supports mmproj Flash Attention through `flash_attn`:

- `None` (default): AUTO.
- `True`: enable Flash Attention.
- `False`: disable Flash Attention.

This setting is independent of the `Llama` backbone's Flash Attention setting.
`use_gpu=True` runs mmproj on GPU with either FA setting. In the CLI example,
use `--mmproj-fa auto`, `--mmproj-fa on`, or `--mmproj-fa off`.

Audio validation is enabled for all settings. If a backend produces invalid
audio, check the model files and native build; FA OFF or CPU mmproj can help
isolate the issue.

The Pocket `english_2026-04_24l` pack also reports that the vendor has no tuned
settings for that exact variant name. It uses fallback settings; Python does
not override the model's variant or assume another pack's tuning applies.
