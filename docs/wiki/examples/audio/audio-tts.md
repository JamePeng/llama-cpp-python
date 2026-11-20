# Speech synthesis with MTMD

`MTMDAudioGenerator` provides experimental, non-streaming speech synthesis using
the vendored `mtmd_helper_gen_audio_*` API. It supports Qwen3-TTS, Pocket TTS,
Soprano, Chatterbox, Drama 3, Zonos3, Kokoro, tiny-voca and VoxCPM2, plus YuE2 song
generation. Each backbone requires a matching audio generation mmproj. Audio-input models such
as Qwen3-ASR are not TTS models.

For Qwen3-TTS, use the Base variant supported by the vendor example.
The high-level API supports Base reference audio. CustomVoice preset speaker IDs
are outside the current generation helper's interface.

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

Drama 3 uses 44.1 kHz output and native helper sampling. Pass raw text, optionally
with `<|speaker:0|>` through `<|speaker:15|>` tags; the helper adds speaker 0 when
needed. Omit `language`, `voice` and `speaker_reference`. Only `seed` affects its
sampling. Each frame contains 2048 samples; `max_frames` is also bounded by the
native tower's combined prompt/frame limit of 4096 positions.

Chatterbox Turbo (English) and Multilingual V3 also use `MTMDAudioGenerator`.
Pass `speaker_reference` or `voice="default"`; omitting both selects the first
voice included in the weights. Multilingual V3 requires
`Llama(n_seq_max=2, kv_unified=True)` and accepts language tags such as `fr`.
The backbone GGUF must also contain `<architecture>.classifier_free_guidance=true`.
Python queries this flag before initializing the helper; a missing or false
flag raises `ValueError` with conversion guidance. Reconvert older backbones
using the current vendor Chatterbox converter and use matching projector weights.
The loaded llama library must expose `llama_model_classifier_free_guidance`;
rebuild the native library if this optional extension is unavailable.

`n_seq_max` is the physical sequence capacity: `2` already reserves the
conditioned and unconditional branches. The helper pairs sequence 0 with
`n_seq_max // 2`; Python validates the capacity and unified KV setting without
doubling or otherwise modifying them. Turbo does not require the metadata
flag or paired sequences. These checks do not apply to other TTS pipelines.
The helper manages both CFG branches and decodes the complete audio at the end.
See the [conversion guide](../../../../vendor/llama.cpp/docs/multimodal/tts.md#chatterbox).

| Option | Qwen3-TTS | Pocket TTS | Soprano |
| --- | --- | --- | --- |
| Reference audio | Optional | Required unless using a predefined voice | Unsupported; fixed voice |
| Language | `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru`, or names accepted by the helper | Selected by weights; omit `language` | English; omit `language` |
| Token sampling | Uses temperature, top-k, top-p, min-p and repetition penalty | Skipped | Uses the same token sampling options as Qwen |
| Flow/decoder settings | Managed by vendor | `temperature` overrides flow noise; other settings follow vendor | Managed by vendor |

`temperature=None` (the default) uses `0.8` for Qwen3-TTS, Soprano and
Chatterbox backbone sampling, and the native model default for Pocket.
An explicit value controls the corresponding stage; for example,
`temperature=0.7` (CLI: `--temperature 0.7`) sets Pocket flow noise to `0.7`.
`0` selects greedy backbone sampling or disables Pocket flow noise. Explicit
values must be finite, nonnegative float32 values. Drama 3 ignores temperature.
Pocket previously ignored explicit temperature values; they now take effect.

Other defaults are `top_k=40`, `top_p=0.95`, `min_p=0.05`, and
`repeat_penalty=1.05`. The repetition window uses the active context size.
Qwen's `top_k` and `top_p` also configure the helper's code predictor. A
nonpositive `top_k` disables the backbone top-k filter but selects the helper's
default for the code predictor. Pocket ignores these token sampling options;
`seed` still reaches its helper.
This requires native libraries built from the matching vendor revision.
`seed=None` uses the native random-seed sentinel. Identical seeds do not promise
bitwise identical audio across hardware or backend versions.

Pocket weights must be converted from the appropriate `languages/<name>`
directory, following the [vendor TTS documentation](../../../../vendor/llama.cpp/docs/multimodal/tts.md).
For weights containing predefined voices, pass `voice="alba"` instead of
`speaker_reference`; the two options cannot be combined. Available names depend
on the weights. The low-level `mtmd_gen_audio_voice_name` API enumerates them.

### Zonos3

Use a ZONOS3-Mini backbone and matching F32 mmproj. Output is mono 44.1 kHz.
Omitting `voice` and `speaker_reference` selects the first voice in the weights
(`Savannah_en-American_F_1` in the tested release). Set `voice` to another
predefined name, or pass reference audio; these options are mutually exclusive.
References use the first 60 seconds and do not require a transcript.

```python
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
from llama_cpp.llama_multimodal import MTMDAudioGenerator

llama = Llama(
    model_path="/path/to/model/zonos3-mini.gguf",
    embeddings=True, pooling_type=LLAMA_POOLING_TYPE_NONE,
    n_ctx=4096, n_batch=128, n_ubatch=128, n_gpu_layers=-1,
)
generator = MTMDAudioGenerator(mmproj_path="/path/to/model/mmproj-zonos3-mini-f32.gguf")
try:
    audio = generator.create_speech(
        llama=llama, text="Hello, your coffee is ready.",
        voice="Imogen_en-British_F_1", speed=1.2, seed=42,
    )
    audio.save("voice.wav")
    audio = generator.create_speech(
        llama=llama, text="It costs $3.50.", language="en",
        speaker_reference="/path/to/reference.wav", speed=0.8, loudness=-25,
        response_format="pcm_f32", seed=42,
    )
    audio.save("cloned.pcm")
finally:
    generator.close()
    llama.close()
```

| Option | Zonos3 behavior |
| --- | --- |
| `speed` | Positive relative speaking rate; `1.0` leaves the native rate condition at its default. |
| `loudness` | Negative target LUFS; `0.0` leaves the native loudness condition at its default. |
| `temperature` | Native frame sampling; `None` uses the model default (`0.75` in the current vendor). |
| `language` | Selects text normalization, not the spoken language; the language follows the text. |
| `top_k`, `top_p`, `min_p`, `repeat_penalty` | Zonos3 uses its native helper's sampling settings. |

Delivery values select model conditioning buckets, rather than applying a
waveform speed/volume filter; exact duration or measured LUFS is not guaranteed.
Loudness has a stronger effect with cloned references; predefined voices tend
to retain their reference level.

The backbone's named `tts` template is required and rendered before generation.
It normalizes supported numbers, money and times using the packaged vendor
Python pattern matcher. Use `en`, `fr`, `de`, `es`, `it`, `pt`, `ja` or `zh`/`cmn`
for the corresponding normalization (regional tags are accepted); unrecognized
languages pass through. Empty language uses English. Supply already normalized
text for forms outside the template's coverage.

For the CLI example, select the Zonos3 files and add
`--voice Imogen_en-British_F_1 --speed 1.2`, or
`--speaker /path/to/reference.wav --loudness -25`. See the
[vendor Zonos3 guide](../../../../vendor/llama.cpp/docs/multimodal/tts.md#zonos3).

### YuE2

YuE2-3B generates 48 kHz stereo songs from style tags and lyrics. Use matching
backbone and mmproj GGUF files; the projector includes the separate YuE2-Vae
decoder. Follow the [vendor conversion guide](../../../../vendor/llama.cpp/docs/multimodal/tts.md#yue2).

The shared projector loader introduced for tiny-voca rejected YuE2 projectors
that omitted `cls.output.weight`, an optional Qwen3 reranking head that YuE2
does not use. JamePeng's fix allows the tensor to be absent and was merged
into the llama.cpp mainline as `53ecea05a`. Use a native build that includes
this commit. [llama.cpp #32427](https://github.com/ggml-org/llama.cpp/pull/32427)
· [llama.cpp #32606](https://github.com/ggml-org/llama.cpp/pull/32606)

```python
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
from llama_cpp.llama_multimodal import MTMDAudioGenerator

llama = Llama(
    model_path="/path/to/model/YuE2-3B-BF16.gguf",
    n_ctx=0, n_seq_max=2, kv_unified=True, n_batch=128, n_ubatch=128,
    n_gpu_layers=-1, embeddings=True, pooling_type=LLAMA_POOLING_TYPE_NONE,
)
generator = MTMDAudioGenerator(mmproj_path="/path/to/model/mmproj-YuE2-3B-BF16.gguf")
try:
    audio = generator.create_speech(
        llama=llama,
        text="[Tags]\nGentle acoustic pop, warm vocals\n[Lyrics]\n[Verse]\nMorning light across the bay\nWe begin another day\n",
        voice="full", max_frames=None, seed=42,
    )
    audio.save("song.wav")
    print(audio.channels, audio.sample_rate, audio.duration)
finally:
    generator.close()
    llama.close()
```

`voice` selects score planning: `full` (default), `melody`, or `off`; it is not
a predefined speaker. Omit `language` and `speaker_reference`. An optional
`\n[ABC]\n<score>` suffix supplies a score in full/melody modes.

Use a UTF-8 song file with style tags and lyrics, for example:

```text
[Tags]
Gentle acoustic pop, warm vocals, soft guitar, relaxed tempo
[Lyrics]
[Verse]
Morning light across the bay
We begin another day
[Chorus]
Carry every hope along
Let the morning be our song
```

- Keep the model's training context (24576 positions per sequence) and F16 KV
  for both keys and values. `n_ctx=0` selects the model context.
- `cfg_scale=0` uses native defaults: `1.01` in off mode, `1.0` otherwise.
  `1.0` disables guidance; other values require a CFG-enabled backbone and
  `n_seq_max >= 2, kv_unified=True`. Python preserves the supplied sequence count.
- `flow_steps=0` uses the native 32-step NAR solver. Smaller positive values
  reduce computation and may reduce quality. The accepted range is 1–1024.
- `temperature` controls code sampling; other token-sampling options follow
  the native pipeline.
- Prefer `max_frames=None` for a complete song. A finite budget counts score
  tokens and code-generation steps together. Stopping before the first audio
  code cannot produce audio; shorter planning via `voice="off"` helps bounded tests.

**Duration and generation limits**

The native pipeline allows up to 4096 score tokens and about 9000 audio codes.
Each audio code represents 40 ms, giving an upper limit of **about 6 minutes**.
Model EOS or context pressure can end a song earlier; increasing `max_frames`
or context capacity does not remove the native audio-code limit.

Without a caller limit, native EOS sampling starts after about 200 audio codes
(roughly 8 seconds). This is not a guaranteed output range: an explicit step
budget or context limit can produce a shorter clip.

| Use | Python options | Approximate audio limit |
| --- | --- | --- |
| Short preview | `voice="off", max_frames=64` | 2.56 seconds |
| Default step budget, without score planning | `voice="off", max_frames=512` | 20.48 seconds |
| Up to half a minute | `voice="off", max_frames=750` | 30 seconds |
| Up to one minute | `voice="off", max_frames=1500` | 60 seconds |
| Let the model finish | `max_frames=None` | Model EOS/context limit, at most about 6 minutes |

These are caps, not requested durations; the decoder also trims a small number
of boundary samples. In `full`/`melody` modes, score planning consumes part of
the same budget, so `max_frames * 0.04` does not predict the audio duration.
There is no exact-duration argument. A 26-second song ending with
`finish_reason="stop"` is a normal model ending, not the maximum supported length.

`finish_reason="length"` means the Python step budget was exhausted. `"stop"`
means the native pipeline stopped, which can include EOS or a native
code/context limit. Reuse the loaded model and generator for sequential calls;
each request resets generation state. `seed` controls sampling; repeatability
also depends on using the same backend and configuration.

CLI input can come from a UTF-8 file:

```sh
python -m examples.high_level_api.mtmd_tts --model /path/to/model/YuE2-3B-BF16.gguf --mmproj /path/to/model/mmproj-YuE2-3B-BF16.gguf --text-file /path/to/song.txt --voice off --ctx-size 0 --cfg --max-frames 0 --gpu-layers -1 --mmproj-gpu --output song.wav
```

CLI `--max-frames 0` corresponds to Python `max_frames=None`. For a bounded
preview, use `--voice off --max-frames 750` instead; this caps audio near 30 seconds,
but the model may finish earlier. Use `--voice full` or `--voice melody` for
score planning. The high-level API returns audio after generation completes;
it does not expose the native streaming mode.

### Soprano

Soprano-80M and Soprano-1.1-80M require matching backbone and mmproj GGUF files
converted from the same checkpoint, including its `decoder.pth`. See the
[vendor conversion instructions](../../../../vendor/llama.cpp/docs/multimodal/tts.md#soprano).
Use a native build with Soprano support and supply normalized English text;
the wrapper does not normalize numbers or abbreviations.

```sh
python -m examples.high_level_api.mtmd_tts --model /path/to/model/soprano.gguf --mmproj /path/to/model/mmproj-soprano.gguf --text "Hello, welcome back." --temperature 0 --output soprano.wav
```

Omit `speaker_reference` and `language`; the API rejects both for this fixed
English voice. The helper constructs the prompt and decodes the complete
utterance into 32 kHz audio. `max_frames` must be at least 2; early termination
before two frames still raises a native decoding error. Both WAV and float32
PCM are supported, with no streaming output.

### Kokoro and tiny-voca

Both models use the first voice in the mmproj by default. Pass `voice` to
select another preset and `speed` to change the speaking rate (tiny-voca accepts
`0.25` through `4.0`). The helper generates speech without token sampling.

```python
audio = generator.create_speech(llama=llama, text="The meeting is at noon.", voice="F2", speed=1.2)
```

This example uses tiny-voca. For Kokoro, try `voice="af_heart"` with matching weights.
Both models convert English text with the GGUF's named `tts` template.
Use `raw_prompt=True` to skip template rendering and pass phonemes directly
to the helper. Each model has its own phoneme format.

```python
# Use the matching backbone and mmproj for each model.
kokoro_audio = kokoro_generator.create_speech(
    llama=kokoro_llama, text="həlˈO wˈɜɹld.", voice="af_heart", raw_prompt=True,
)
tiny_voca_audio = tiny_voca_generator.create_speech(
    llama=tiny_voca_llama, text="h ə l ˈoʊ <sp> w ˈɜː l d .", voice="F2", raw_prompt=True,
)
```

Kokoro uses its own phoneme symbols. tiny-voca expects space-separated symbols,
`<sp>` between words, and `ˈ` or `ˌ` before a stressed vowel.
The CLI equivalent is `--raw-prompt`, which works with `--text` or a UTF-8
`--text-file`.

```sh
python -m examples.high_level_api.mtmd_tts --model /path/to/model/tiny-voca.gguf --mmproj /path/to/model/mmproj-tiny-voca.gguf --text "h ə l ˈoʊ <sp> w ˈɜː l d ." --voice F2 --raw-prompt --output hello.wav
```

GGUF files converted without a `tts` template already accept phonemes directly.
See the vendor guides for
[Kokoro](../../../../vendor/llama.cpp/docs/multimodal/tts.md#kokoro) and
[tiny-voca](../../../../vendor/llama.cpp/docs/multimodal/tts.md#tiny-voca).

### VoxCPM2

Use a matching VoxCPM2 backbone and audio mmproj for 48 kHz mono speech. The
model follows the language of the text. Put a voice description in parentheses
at the start of the text, or pass `speaker_reference` to clone an existing voice.
The helper uses up to the first 30 seconds of the reference. Omit `language`,
`voice`, and `speaker_ref_text`.

```python
audio = generator.create_speech(
    llama=llama, text="Your coffee is ready.",
    speaker_reference="/path/to/reference.wav", max_frames=None,
    cfg_scale=2.0, flow_steps=10,
)
```

`temperature` scales the flow noise (default `1.0`). Set `flow_steps` to
`2–32`, or leave it at `0` for the model default of 10 steps. `cfg_scale=0`
selects the native guidance scale of 2.0. A single sequence is sufficient
for guidance. `max_frames=None` uses the native stop condition and frame limit.

```sh
python -m examples.high_level_api.mtmd_tts --model /path/to/model/VoxCPM2.gguf --mmproj /path/to/model/mmproj-VoxCPM2.gguf --text "(A calm narrator) Welcome back." --cfg-scale 2 --flow-steps 10 --max-frames 0 --output speech.wav
```

See the [vendor guide](../../../../vendor/llama.cpp/docs/multimodal/tts.md#voxcpm2) for conversion instructions.

### Qwen voice cloning

Pass the words spoken in the reference clip as `speaker_ref_text` (CLI
`--speaker-text`) to enable in-context voice cloning. Use a clip of at most
30 seconds and an mmproj converted with the speech tokenizer encoder. With an
older mmproj, a longer clip or no transcript, the helper uses the speaker embedding.

```python
audio = generator.create_speech(
    llama=llama, text="Your coffee is ready.", language="en",
    speaker_reference="/path/to/reference.wav", speaker_ref_text="Good morning.",
)
```

For reference input, use an encoded audio file path, URL, data URI, or audio
bytes. Bare base64 strings and `(numpy_array, sample_rate)` tuples are not
accepted directly.

## Results and limits

`GeneratedAudio` owns its `data` bytes after the request returns:

- `format="wav"`: complete PCM16 little-endian WAV, the default.
- `format="pcm_f32"`: interleaved native-endian float32 samples, without a header.
- `channels`: native channel count; YuE2 uses stereo, the supported speech models use mono.
- `sample_rate` and `n_samples`: output rate and samples **per channel**;
  `duration = n_samples / sample_rate` is seconds.
- `finish_reason="stop"`: the native pipeline ended (EOS or a native model/context limit).
- `finish_reason="length"`: the generation-step budget was exhausted; speech
  may be cut off. Increase `max_frames` and provide sufficient context capacity.

`save()` writes these bytes unchanged. A `.wav` extension does not convert raw
PCM to WAV. `max_frames=512` bounds helper steps, not text tokens or seconds;
Pocket's chunk transitions also consume helper steps. Pass `max_frames=None`
to use the native model limit; a native limit of `0` runs until model EOS.
There is no automatic
context shifting in this wrapper. Native decode failures raise `RuntimeError`.

Calls on one generator cannot overlap, and it cannot be closed while generating.
Do not use or close the bound Llama concurrently. `llama.abort()` can interrupt
an active request from another thread; synthesis raises `InterruptedError` and
cleans up. Cancellation is checked between native calls and does not guarantee
immediate interruption inside the mmproj encoder or vocoder.

The delivery defaults are `speed=1.0` and `loudness=0.0`. Zonos3, Kokoro and
tiny-voca accept speed changes. Zonos3 also accepts loudness changes. Native `set_input`
code `2` raises `ValueError` for invalid
model input; other failures raise `RuntimeError`, and cancellation takes
precedence as `InterruptedError`. Failed requests release temporary resources
so a later request can retry.

The low-level bindings also include MiniMax-Music3 and Fish S2. Their
model-specific generation/conditioning is not enabled by `create_speech` yet.

## Implementation and validation

The wrapper reuses `MTMDBaseHandler` for media and context management. It drives
`set_input`, batched `step_prompt`, hidden-state feedback through `step_gen`, and
one final `get_output`. Qwen and Soprano use a request-local `LlamaSamplingContext`; Pocket
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

For Qwen, optionally add `--language en`. For Pocket and Soprano, omit it. Compare the output
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
