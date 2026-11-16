"""Local MTMD TTS playground. Run: python -m streamlit run app.py"""

import atexit
from dataclasses import asdict
from datetime import datetime
import io
import json
from pathlib import Path
import time
from typing import cast
import uuid
import wave

import numpy as np
import streamlit as st

from tts_engine import ModelConfig, TTSEngine

ROOT = Path(__file__).resolve().parent
MODELS = {
    "Zonos3 · Mini": (
        "/path/to/model/ZONOS3-Mini-Q8_0.gguf",
        "/path/to/model/mmproj-ZONOS3-Mini-F32.gguf",
    ),
    "Qwen3-TTS Base · 1.7B": (
        "/path/to/model/qwen3-tts-base.gguf",
        "/path/to/model/mmproj-qwen3-tts-base.gguf",
    ),
    "Soprano · 1.1 80M": (
        "/path/to/model/Soprano-1.1-80M.gguf",
        "/path/to/model/mmproj-Soprano-1.1-80M.gguf",
    ),
    "Pocket TTS · English": (
        "/path/to/model/pocket-tts.gguf",
        "/path/to/model/mmproj-pocket-tts.gguf",
    ),
}
SCENARIOS = {
    "Custom": ("Hello, welcome to speech synthesis. How can I help you today?", "en"),
    "Restaurant · Chinese": ("您好，请给我一杯拿铁和一份三明治。可以打包吗？谢谢。", "zh"),
    "Convenience store · English": ("Would you like a bag? You can tap your card here. Thank you, and have a nice day.", "en"),
    "Game adjutant · Chinese": ("指挥官，雷达发现未知目标。防御系统已就绪。增援部队正在等待部署。", "zh"),
    "Game adjutant · English": ("Commander, unidentified contacts detected. Defensive systems online. Awaiting your orders.", "en"),
    "Houston · English": ("Houston, Odyssey here. All crew members are feeling good. Systems are stable. We are ready for the next update.", "en"),
    "Missile defense · English": ("Alert. Incoming missile detected. Tracking confirmed. Interceptors away. Threat neutralized.", "en"),
    "Numbers and prices · English": ("Your total is $12.50 for 2 sandwiches. Order 42 will be ready in 5 minutes.", "en"),
    "Restaurant · Japanese": ("こんにちは。コーヒーを一つお願いします。持ち帰りにできますか。", "ja"),
}
LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "de": "German", "fr": "French", "ko": "Korean", "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian"}


@st.cache_resource(max_entries=1, on_release=lambda engine: engine.close())
def get_engine():
    engine = TTSEngine()
    atexit.register(engine.close)
    return engine


def use_scenario():
    text, language = SCENARIOS[st.session_state.scenario]
    st.session_state.text = text
    st.session_state.language = language


def use_model():
    english_only = st.session_state.model_name.startswith(("Pocket", "Soprano"))
    st.session_state.scenario = "Convenience store · English" if english_only else "Custom"
    use_scenario()


def save_audio(audio, metadata):
    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True)
    filename = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.wav"
    path = folder / filename
    audio.save(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return filename


st.set_page_config(page_title="TTS playground", page_icon=":material/graphic_eq:", layout="wide")
st.title("TTS playground")
st.caption("Local speech synthesis with Zonos3, Qwen3-TTS Base, Pocket TTS and Soprano.")
st.session_state.setdefault("history", [])
st.session_state.setdefault("text", SCENARIOS["Custom"][0])
st.session_state.setdefault("language", "en")
st.session_state.setdefault("voice_info", None)

with st.sidebar:
    st.subheader("Model and runtime")
    model_name = cast(str, st.selectbox("Model", list(MODELS), key="model_name", on_change=use_model))
    pocket = model_name.startswith("Pocket")
    zonos = model_name.startswith("Zonos3")
    soprano = model_name.startswith("Soprano")
    english_only = pocket or soprano
    defaults = MODELS[model_name]
    with st.expander("Model file paths", expanded=True):
        model_path = st.text_input("Backbone GGUF", defaults[0], key=f"model_{model_name}")
        mmproj_path = st.text_input("Audio mmproj GGUF", defaults[1], key=f"mmproj_{model_name}")
    gpu = st.toggle("Run backbone on GPU", value=True, key="gpu")
    mmproj_gpu = st.toggle("Run audio model on GPU", value=True, key="mmproj_gpu")
    fa = cast(str, st.selectbox("Audio model Flash Attention", ["AUTO", "ON", "OFF"], key="fa"))
    n_ctx = st.select_slider("Context size", [1024, 2048, 4096, 8192], value=4096, key="n_ctx")
    config = ModelConfig(model_path.strip(), mmproj_path.strip(), -1 if gpu else 0, mmproj_gpu, {"AUTO": None, "ON": True, "OFF": False}[fa], n_ctx)
    if st.button("Load model / list voices", key="load_model", icon=":material/record_voice_over:"):
        try:
            with st.spinner("Loading model and available voices..."):
                st.session_state.voice_info = (asdict(config), get_engine().describe(config))
            st.success("Model ready.")
        except Exception as exc:
            st.session_state.voice_info = None
            st.error(f"Loading failed: {exc}")
    st.caption("Runtime changes take effect on the next generation.")
    if st.button("Unload model / free memory", icon=":material/memory:", key="unload"):
        try:
            get_engine().unload()
            st.session_state.voice_info = None
            st.success("Model unloaded. It will reload on the next generation.")
        except RuntimeError as exc:
            st.error(str(exc))

editor, reference = st.columns([1.6, 1], gap="large")
with editor:
    st.subheader("01 · Write your script")
    options = [name for name, (_, lang) in SCENARIOS.items() if not english_only or lang == "en"]
    st.selectbox("Example scenario", options, key="scenario", on_change=use_scenario)
    if english_only:
        st.caption("This preset uses English weights. Language is determined by the model.")
    else:
        languages = [lang for lang in LANGUAGES if not zonos or lang not in ("ko", "ru")]
        st.selectbox("Language", languages, format_func=LANGUAGES.__getitem__, key="language")
        if zonos:
            st.caption("Language selects text normalization. Voice names identify the preset's language and accent.")

with reference:
    st.subheader("02 · Choose a voice")
    speaker: str | bytes | None = None
    preview: bytes | None = None
    voice: str | None = None
    source = "Preset / default"
    if soprano:
        st.info("Soprano uses a fixed English voice. Reference audio is not supported.")
    else:
        source = cast(str, st.selectbox("Voice source", ["Preset / default", "Upload audio", "Record audio", "File path / URL"], key="ref_source"))
        if source == "Preset / default" and (zonos or pocket):
            loaded = st.session_state.voice_info
            voices = loaded[1]["voices"] if loaded and loaded[0] == asdict(config) else []
            voice = st.selectbox("Preset voice", [None, *voices], format_func=lambda v: v or "Model default", key=f"voice_{model_name}")
            if not voices:
                st.caption("Use Load model / list voices to choose a named voice, or generate with the model default.")
        elif source == "Upload audio":
            uploaded = st.file_uploader("Upload reference audio", type=["wav", "mp3", "flac", "ogg"], max_upload_size=30, key="reference_upload")
            if uploaded is not None:
                speaker = preview = uploaded.getvalue()
        elif source == "Record audio":
            recorded = st.audio_input("Record a reference voice", sample_rate=24000, key="reference_recording")
            if recorded is not None:
                speaker = preview = recorded.getvalue()
        elif source == "File path / URL":
            speaker = st.text_input("Audio path, URL or data URI", key="ref_path", placeholder="/path/to/reference.wav").strip() or None
        if preview:
            st.audio(preview, alt="Reference voice preview")
        if zonos:
            st.caption("Use a preset or reference audio. Zonos3 uses up to the first 60 seconds of a reference.")
        elif pocket:
            st.caption("Use a packaged voice if available, or provide reference audio.")
        else:
            st.caption("Qwen3-TTS Base uses speaker embeddings only; ref_text-based cloning and CustomVoice are not supported.")

with editor:
    with st.form("synthesis"):
        text = st.text_area("Text to synthesize", key="text", height=175, max_chars=4000)
        batch = st.checkbox("Generate each line separately", key="batch", help="Up to 10 non-empty lines, using the same reference voice.")
        with st.expander("Generation settings"):
            seed = st.number_input("Seed", min_value=0, max_value=4294967294, value=42, step=1, key="seed")
            frames = st.slider("Maximum generation steps", 32, 1024, 512, step=32, key="frames")
            override_temperature = st.checkbox("Override model temperature", key="override_temperature", disabled=pocket)
            selected_temperature = st.slider("Temperature", 0.0, 1.5, 0.8, step=0.05, disabled=pocket, key="temperature")
            temperature = selected_temperature if override_temperature and not pocket else None
            speed, loudness = 1.0, 0.0
            if zonos:
                speed = st.slider("Speech speed", 0.5, 2.0, 1.0, step=0.05, key="speed")
                loudness = st.select_slider("Loudness (LUFS)", [0, -32, -28, -24, -20, -16, -12], value=0, key="loudness", format_func=lambda v: "Model default" if v == 0 else str(v))
                st.caption("Speed and loudness condition the model; exact duration and output LUFS are not guaranteed. Preset voices can limit the loudness effect.")
            st.caption("Steps are not seconds. Split long text into separate requests.")
        submitted = st.form_submit_button("Generate speech", type="primary", icon=":material/play_arrow:", width="stretch")

st.divider()
st.subheader("03 · Listen and download")
if submitted:
    lines = [line.strip() for line in text.splitlines() if line.strip()] if batch else [text.strip()]
    if not lines or not all(lines):
        st.error("Enter text to synthesize.")
    elif len(lines) > 10:
        st.error("Use no more than 10 lines per batch.")
    elif source != "Preset / default" and not speaker:
        st.error("Provide reference audio or select Preset / default.")
    else:
        progress = st.progress(0, text="Preparing model. The first request loads the weights...")
        try:
            for index, sentence in enumerate(lines, 1):
                started = time.perf_counter()
                audio = get_engine().synthesize(
                    config, text=sentence, language=None if english_only else st.session_state.language,
                    speaker_reference=speaker, seed=int(seed), max_frames=frames,
                    temperature=temperature, voice=voice, speed=speed, loudness=float(loudness), response_format="wav",
                )
                metadata = dict(text=sentence, model=model_name, language=None if english_only else st.session_state.language,
                                duration=audio.duration, sample_rate=audio.sample_rate, n_samples=audio.n_samples,
                                elapsed=round(time.perf_counter() - started, 3), finish_reason=audio.finish_reason,
                                seed=int(seed), max_frames=frames, temperature=temperature, voice=voice,
                                speed=speed, loudness=loudness, config=asdict(config))
                filename = save_audio(audio, metadata)
                st.session_state.history.insert(0, dict(filename=filename, data=audio.data, **metadata))
                del st.session_state.history[10:]
                progress.progress(index / len(lines), text=f"Completed {index} / {len(lines)}")
            st.success("Done. Audio and settings were saved to the outputs folder.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.caption("Check that the backbone and mmproj match. For memory issues, disable audio GPU or reduce context size. Completed audio remains below.")
        finally:
            progress.empty()

history = st.session_state.history
if not history:
    st.info("Your audio player, waveform and download links will appear here.")
else:
    if st.button("Clear listening history", key="clear_history"):
        st.session_state.history = []
        st.rerun()
    st.caption("The last 10 results are kept in this session. Clearing history does not delete saved files.")
    for index, item in enumerate(history):
        with st.container(border=True):
            st.markdown(f"**{item['model']}** · {item['text']}")
            st.audio(item["data"], format="audio/wav", alt="Synthesized speech")
            st.caption(f"{item['duration']:.2f} s · {item['sample_rate']} Hz · elapsed {item['elapsed']:.2f} s (including initial loading)")
            if item["finish_reason"] == "length":
                st.warning("Generation reached the step limit. Increase the limit or split the text if speech is incomplete.")
            st.download_button("Download WAV", data=item["data"], file_name=item["filename"], mime="audio/wav", key=f"download_{item['filename']}", on_click="ignore")
            if st.toggle("Show waveform and settings", key=f"details_{item['filename']}"):
                with wave.open(io.BytesIO(item["data"]), "rb") as wav:
                    signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
                step = max(1, len(signal) // 1500)
                st.line_chart({"Amplitude": signal[::step]}, height=130, alt="Speech waveform")
                st.json({k: v for k, v in item.items() if k != "data"})
