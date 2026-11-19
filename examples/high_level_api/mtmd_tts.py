"""Non-streaming MTMD speech and YuE2 song synthesis examples.

Run from the repository root with matching backbone and mmproj GGUF files:

  python -m examples.high_level_api.mtmd_tts --model /path/to/model/backbone.gguf --mmproj /path/to/model/mmproj.gguf --text "Hello world"

Use the same --model and --mmproj arguments with any of these examples:

  --text "Hello world"                         Basic Qwen or Soprano synthesis
  --text "Welcome back" --speaker /path/to/reference.wav    Reference voice
  --text "Welcome back" --speaker /path/to/reference.wav --speaker-text "Good morning."    Qwen ICL
  --text "Welcome back" --voice alba           Pocket weights with predefined voices
  --text "<|speaker:0|> Hello. <|speaker:1|> Welcome back."    Drama 3 speakers
  --text "Hello" --speaker voice.wav --speaker-bytes
  --text "First message" --text "Next message" --output messages.wav
  --demo daily --speaker voice.wav             Reuse one generator for several lines
  --demo game --gpu-layers -1 --mmproj-gpu --mmproj-fa on
  --demo houston --speaker voice.wav
  --demo multilingual --speaker voice.wav      Qwen only: Chinese, English, Japanese
  --text "Hello" --voice Imogen_en-British_F_1 --speed 1.2    Zonos3
  --text "Hello" --speaker /path/to/reference.wav --loudness -25    Zonos3 clone
  --text "h ə l ˈoʊ <sp> w ˈɜː l d ." --voice F2 --raw-prompt    tiny-voca phonemes
  --text "(A calm narrator) Hello world." --cfg-scale 2 --flow-steps 10    VoxCPM2 voice design
  --text "Hello" --format pcm_f32 --output speech.pcm
  --text-file /path/to/song.txt --voice off --cfg --ctx-size 0 --max-frames 0    YuE2
  --text-file /path/to/song.txt --voice off --cfg --ctx-size 0 --max-frames 750  YuE2: up to ~30 s

Pocket selects its language from the weights; omit --language and use an
appropriate language pack. Soprano uses a fixed English voice: omit --language
and --speaker, and use --max-frames >= 2. Raw PCM includes a JSON metadata sidecar.
YuE2 allows about 6 minutes; --max-frames 0 uses native limits. Finite budgets
count score tokens and audio codes together; full/melody planning consumes steps.
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
    source.add_argument("--text-file", type=Path, help="UTF-8 text, e.g. YuE2 [Tags]/[Lyrics] song input")
    source.add_argument("--demo", choices=DEMOS, help="Built-in example sentences")
    parser.add_argument("--raw-prompt", action="store_true", help="Skip the TTS template and pass model-ready text or phonemes")
    voices = parser.add_mutually_exclusive_group()
    voices.add_argument("--speaker", help="Reference audio path, URL or data URI")
    voices.add_argument("--voice", help="Predefined voice name, or YuE2 full/melody/off planning mode")
    parser.add_argument("--speaker-text", help="Transcript of --speaker for Qwen3-TTS Base cloning")
    parser.add_argument("--speaker-bytes", action="store_true", help="Read a local --speaker file as encoded audio bytes")
    parser.add_argument("--language", help="Qwen/Chatterbox language or Zonos3/Kokoro template language")
    parser.add_argument("--cfg", action="store_true", help="Reserve two total sequences with unified KV for Chatterbox Multilingual or YuE2 guidance")
    parser.add_argument("--format", choices=("wav", "pcm_f32"), default="wav")
    parser.add_argument("--output", help="Output path; multiple requests add a numbered suffix")
    parser.add_argument("--max-frames", type=int, default=512, help="Helper-step limit; 0 uses native limits (YuE2: up to ~6 min); not seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float,
                        help="Backbone temperature (default: 0.8), or Pocket/Zonos3/YuE2/VoxCPM2 native temperature")
    parser.add_argument("--speed", type=float, default=1.0, help="Relative speaking speed for Zonos3, Kokoro or tiny-voca")
    parser.add_argument("--loudness", type=float, default=0.0, help="Zonos3 target LUFS (negative; 0 uses model default)")
    parser.add_argument("--cfg-scale", type=float, default=0.0, help="YuE2 or VoxCPM2 guidance; 0 uses model defaults")
    parser.add_argument("--flow-steps", type=int, default=0, help="YuE2 or VoxCPM2 flow steps; 0 uses model defaults (VoxCPM2: 2–32 otherwise)")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--mmproj-gpu", action="store_true")
    parser.add_argument("--mmproj-fa", choices=("auto", "on", "off"), default="auto",
                        help="mmproj Flash Attention (default: auto)")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    if args.speaker_bytes and not args.speaker:
        parser.error("--speaker-bytes requires a local --speaker file")
    if args.speaker_text is not None and not args.speaker:
        parser.error("--speaker-text requires --speaker")
    if args.demo == "multilingual" and args.language:
        parser.error("--demo multilingual supplies a language for each request")

    texts = [args.text_file.read_text(encoding="utf-8")] if args.text_file else args.text
    requests = DEMOS[args.demo] if args.demo else [(text, None) for text in texts]
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
        n_seq_max=2 if args.cfg else 1,
        kv_unified=args.cfg,
        n_gpu_layers=args.gpu_layers,
    )) as llama:
        with MTMDAudioGenerator(
            mmproj_path=args.mmproj, use_gpu=args.mmproj_gpu,
            flash_attn={"auto": None, "on": True, "off": False}[args.mmproj_fa],
        ) as generator:
            for index, (text, language) in enumerate(requests, 1):
                audio = generator.create_speech(
                    llama=llama, text=text, language=language or args.language,
                    speaker_reference=speaker, voice=args.voice, max_frames=args.max_frames or None,
                    speaker_ref_text=args.speaker_text, raw_prompt=args.raw_prompt,
                    seed=args.seed, temperature=args.temperature, response_format=args.format,
                    speed=args.speed, loudness=args.loudness, cfg_scale=args.cfg_scale, flow_steps=args.flow_steps,
                )
                path = output if len(requests) == 1 else output.with_name(f"{output.stem}-{index:02d}{output.suffix}")
                audio.save(path)
                if audio.format == "pcm_f32":
                    path.with_suffix(path.suffix + ".json").write_text(json.dumps({
                        "sample_rate": audio.sample_rate, "channels": audio.channels,
                        "n_samples": audio.n_samples, "dtype": "float32", "byte_order": sys.byteorder,
                        "finish_reason": audio.finish_reason,
                    }, indent=2), encoding="utf-8")
                print(f"Saved {audio.duration:.2f}s at {audio.sample_rate} Hz, {audio.channels} channel(s), to {path} ({audio.finish_reason})")
                if audio.finish_reason == "length":
                    print("Frame limit reached; increase --max-frames if the speech is incomplete.")


if __name__ == "__main__":
    main()
