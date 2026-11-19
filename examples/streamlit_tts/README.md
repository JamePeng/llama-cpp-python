# Streamlit TTS playground

An English-language UI for local, non-streaming speech synthesis with
`MTMDAudioGenerator`. Supports audio upload, microphone recording, reference
paths/URLs, multiple languages, sequential requests, WAV playback/download,
waveforms, preset voice selection, Zonos3 speed/loudness controls, GPU settings
and mmproj Flash Attention AUTO/ON/OFF.

## Supported models

| Model | Requirements | Notes |
| --- | --- | --- |
| Zonos3 Mini | Backbone GGUF and matching audio mmproj | Predefined voices or reference audio; language normalization, speed and loudness conditioning. |
| Soprano 1.1 80M | Backbone GGUF and matching audio mmproj | Fixed English voice; no reference audio. |
| Qwen3-TTS Base | Backbone GGUF and matching audio-generation mmproj | Tested with the 12Hz 1.7B Base BF16 pair. Reference audio is optional; language is selectable. |
| Pocket TTS | Backbone GGUF and matching language-pack mmproj | Tested with the English `english_2026-04_24l` BF16 pair. Use reference audio or a packaged voice if available; language comes from the weights. |

Qwen CustomVoice and audio-input-only models such as
Qwen3-ASR are not supported. Other checkpoint sizes are not validated by this
example. This demo uses the Qwen reference speaker embedding. For cloning with
a transcript, use `--speaker-text` in the [CLI example](../high_level_api/mtmd_tts.py).

## Run

Use a Python environment with a build of this repository that includes
`MTMDAudioGenerator` and the native audio-generation helpers. For GPU synthesis,
use a CUDA-enabled build. Installing the UI dependencies does not replace your
existing llama-cpp-python build.

From the repository root:

```sh
python -m pip install -r examples/streamlit_tts/requirements.txt
python -m streamlit run examples/streamlit_tts/app.py --server.address 127.0.0.1
```

Open http://localhost:8501. The command binds to localhost.
If the port is occupied, append `--server.port 8502`.

All model paths are placeholders under `/path/to/model/`. Replace both paths
in the sidebar with your actual GGUF files before generating. No model weights
or reference audio are bundled or downloaded automatically. Upload a reference,
record one, enter a path/URL (for example `/path/to/reference.wav`), or select
**Preset / default**. Click **Load model / list voices** to load the pair and
populate the preset selector from the native model. Generating directly also
loads the model and uses its default voice when available.

The interface is English; Chinese and Japanese example sentences are retained
to demonstrate multilingual synthesis with Qwen. English Pocket weights should
be used with English text. Custom path fields do not validate model architecture;
choose a matching preset and model pair.

Zonos3 language controls text normalization for English, Chinese, Japanese,
German, French, Spanish, Italian and Portuguese; it does not guarantee a voice
is equally fluent in each language. Reference audio uses at most its first
60 seconds. Speed and loudness are model conditioning, not exact duration or
output LUFS guarantees; preset voices may limit the loudness effect.
Temperature uses each model's default unless **Override model temperature** is checked.

## Behavior

- Models load on the first request and are reused. Changing model/runtime
  settings replaces the loaded pair on the next request.
- One shared engine serializes access across browser sessions. Concurrent
  generation is rejected. Use **Unload model / free memory** to release resources.
- Each non-empty line can be a separate request, up to 10 per batch. Audio is
  available after a complete request; there is no streaming playback.
- WAV and generation settings are saved under this example's `outputs/` folder.
  The current session retains the last 10 results. Clearing history does not
  delete saved files. Output metadata includes text and model configuration,
  but does not contain reference audio or its URL.
- The first request includes model loading time. Reaching the generation-step
  limit can truncate speech; increase the limit or split the text.

See the [TTS wiki](../../docs/wiki/examples/audio/audio-tts.md) for API details.
