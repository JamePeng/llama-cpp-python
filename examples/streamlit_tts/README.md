# Streamlit TTS playground

An English-language UI for local, non-streaming speech synthesis with
`MTMDAudioGenerator`. Supports audio upload, microphone recording, reference
paths/URLs, multiple languages, sequential requests, WAV playback/download,
waveforms, GPU settings and mmproj Flash Attention AUTO/ON/OFF.

## Supported models

| Model | Requirements | Notes |
| --- | --- | --- |
| Qwen3-TTS Base | Backbone GGUF and matching audio-generation mmproj | Tested with the 12Hz 1.7B Base BF16 pair. Reference audio is optional; language is selectable. |
| Pocket TTS | Backbone GGUF and matching language-pack mmproj | Tested with the English `english_2026-04_24l` BF16 pair. Reference audio is required; language comes from the weights. |

Qwen CustomVoice, preset speaker IDs and audio-input-only models such as
Qwen3-ASR are not supported. Other checkpoint sizes are not validated by this
example. Qwen reference conditioning uses speaker embeddings only: neither
`ref_text`-conditioned full cloning nor an `x_vector_only_mode` switch is exposed.

## Run

Use a Python environment with a build of this repository that includes
`MTMDAudioGenerator` and the native audio-generation helpers. For GPU synthesis,
use a CUDA-enabled build. Installing the UI dependencies does not replace your
existing llama-cpp-python build.

From the repository root:

```sh
python -m pip install -r examples/streamlit_tts/requirements.txt
cd examples/streamlit_tts
python -m streamlit run app.py
```

Open http://localhost:8501. The included configuration binds to localhost.
If the port is occupied, append `--server.port 8502`.

All model paths are placeholders under `/path/to/model/`. Replace both paths
in the sidebar with your actual GGUF files before generating. No model weights
or reference audio are bundled or downloaded automatically. Upload a reference,
record one, enter a path/URL, or select **None** for Qwen without a reference.

The interface is English; Chinese and Japanese example sentences are retained
to demonstrate multilingual synthesis with Qwen. English Pocket weights should
be used with English text. Custom path fields do not validate model architecture;
choose a matching preset and model pair.

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

## Test without model weights

From this directory:

```sh
python -m unittest discover -s tests -v
```
