"""Non-streaming Qwen3-TTS / Pocket TTS examples.

Run from the repository root with matching backbone and mmproj GGUF files:

  python -m examples.high_level_api.mtmd_tts --model /path/to/model/backbone.gguf --mmproj /path/to/model/mmproj.gguf --text "Hello world"

Use the same --model and --mmproj arguments with any of these examples:

  --text "Hello world"                         Basic Qwen synthesis
  --text "Welcome back" --speaker voice.wav    Reference voice (required for Pocket)
  --text "Hello" --speaker voice.wav --speaker-bytes
  --text "First message" --text "Next message" --output messages.wav
  --demo daily --speaker voice.wav             Reuse one generator for several lines
  --demo game --gpu-layers -1 --mmproj-gpu --mmproj-fa on
  --demo houston --speaker voice.wav
  --demo multilingual --speaker voice.wav      Qwen only: Chinese, English, Japanese
  --text "Hello" --format pcm_f32 --output speech.pcm

Pocket selects its language from the weights; omit --language and use an
appropriate language pack. Raw PCM output includes a JSON metadata sidecar.
"""

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
from llama_cpp.llama_multimodal import MTMDAudioGenerator


DEMOS = {
    "daily": [
        ("Could I have a coffee and a sandwich, please?", None),
        ("Would you like a bag? You can tap your card here.", None),
    ],
    "game": [
        ("Alert. Unidentified contacts detected. Radar tracking online.", None),
        ("Defensive systems ready. Awaiting your orders, Commander.", None),
    ],
    "houston": [
        ("Odyssey, this is Houston. We have you loud and clear. Report your status.", None),
        ("Houston, Odyssey here. All systems are stable. The crew is ready.", None),
    ],
    "multilingual": [
        ("您好，请问可以打包吗？谢谢。", "zh"),
        ("Hello, could I get this to go? Thank you.", "en"),
        ("こんにちは。持ち帰りでお願いします。", "ja"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Backbone GGUF")
    parser.add_argument("--mmproj", required=True, help="Matching audio mmproj GGUF")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Repeat for multiple sequential requests")
    source.add_argument("--demo", choices=DEMOS, help="Built-in example sentences")
    parser.add_argument("--speaker", help="Reference audio path, URL or data URI; required for Pocket")
    parser.add_argument("--speaker-bytes", action="store_true", help="Read a local --speaker file as encoded audio bytes")
    parser.add_argument("--language", help="Qwen language code or name; omit for Pocket")
    parser.add_argument("--format", choices=("wav", "pcm_f32"), default="wav")
    parser.add_argument("--output", help="Output path; multiple requests add a numbered suffix")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8, help="Qwen backbone sampling temperature")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--mmproj-gpu", action="store_true")
    parser.add_argument("--mmproj-fa", choices=("auto", "on", "off"), default="auto",
                        help="mmproj Flash Attention (default: auto)")
    args = parser.parse_args()
    if args.speaker_bytes and not args.speaker:
        parser.error("--speaker-bytes requires a local --speaker file")
    if args.demo == "multilingual" and args.language:
        parser.error("--demo multilingual supplies a language for each request")

    requests = DEMOS[args.demo] if args.demo else [(text, None) for text in args.text]
    speaker = Path(args.speaker).read_bytes() if args.speaker_bytes else args.speaker
    output = Path(args.output or ("output.wav" if args.format == "wav" else "output.pcm"))
    if args.format == "pcm_f32" and output.suffix.lower() == ".wav":
        parser.error("pcm_f32 has no WAV header; use a .pcm output path")
    output.parent.mkdir(parents=True, exist_ok=True)

    with closing(Llama(
        model_path=args.model,
        embeddings=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
    )) as llama:
        with MTMDAudioGenerator(
            mmproj_path=args.mmproj, use_gpu=args.mmproj_gpu,
            flash_attn={"auto": None, "on": True, "off": False}[args.mmproj_fa],
        ) as generator:
            for index, (text, language) in enumerate(requests, 1):
                audio = generator.create_speech(
                    llama=llama, text=text, language=language or args.language,
                    speaker_reference=speaker, max_frames=args.max_frames,
                    seed=args.seed, temperature=args.temperature, response_format=args.format,
                )
                path = output if len(requests) == 1 else output.with_name(f"{output.stem}-{index:02d}{output.suffix}")
                audio.save(path)
                if audio.format == "pcm_f32":
                    path.with_suffix(path.suffix + ".json").write_text(json.dumps({
                        "sample_rate": audio.sample_rate, "channels": audio.channels,
                        "n_samples": audio.n_samples, "dtype": "float32", "byte_order": sys.byteorder,
                        "finish_reason": audio.finish_reason,
                    }, indent=2), encoding="utf-8")
                print(f"Saved {audio.duration:.2f}s at {audio.sample_rate} Hz to {path} ({audio.finish_reason})")
                if audio.finish_reason == "length":
                    print("Frame limit reached; increase --max-frames if the speech is incomplete.")


if __name__ == "__main__":
    main()
