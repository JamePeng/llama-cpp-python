"""Local Qwen3-TTS / Pocket TTS playground. Run: python -m streamlit run app.py"""

import atexit
from dataclasses import asdict
from datetime import datetime
import io
import json
from pathlib import Path
import time
import uuid
import wave

import numpy as np
import streamlit as st

from tts_engine import ModelConfig, TTSEngine

ROOT = Path(__file__).resolve().parent
MODELS = {
    "Qwen3-TTS Base · 1.7B": (
        "/path/to/model/qwen3-tts-base.gguf",
        "/path/to/model/mmproj-qwen3-tts-base.gguf",
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
    pocket = st.session_state.model_name.startswith("Pocket")
    st.session_state.scenario = "Convenience store · English" if pocket else "Custom"
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
st.caption("Write a line. Try a voice. Local synthesis with Qwen3-TTS and Pocket TTS.")
st.info("Supported: Qwen3-TTS Base with a matching audio mmproj, and Pocket TTS with a matching language-pack mmproj. Tested presets: Qwen 1.7B Base and Pocket English. CustomVoice / preset speaker IDs are not supported.")
st.session_state.setdefault("history", [])
st.session_state.setdefault("text", SCENARIOS["Custom"][0])
st.session_state.setdefault("language", "en")

with st.sidebar:
    st.subheader("Model and runtime")
    model_name = st.selectbox("Model", list(MODELS), key="model_name", on_change=use_model)
    pocket = model_name.startswith("Pocket")
    defaults = MODELS[model_name]
    with st.expander("Model file paths", expanded=True):
        model_path = st.text_input("Backbone GGUF", str(defaults[0]), key=f"model_{pocket}")
        mmproj_path = st.text_input("Audio mmproj GGUF", str(defaults[1]), key=f"mmproj_{pocket}")
    gpu = st.toggle("Run backbone on GPU", value=True)
    mmproj_gpu = st.toggle("Run audio model on GPU", value=True)
    fa = st.selectbox("Audio model Flash Attention", ["AUTO", "ON", "OFF"], key="fa")
    n_ctx = st.select_slider("Context size", [1024, 2048, 4096, 8192], value=4096)
    st.caption("Runtime changes take effect on the next generation.")
    if st.button("Unload model / free memory", icon=":material/memory:", key="unload"):
        try:
            get_engine().unload()
            st.success("Model unloaded. It will reload on the next generation.")
        except RuntimeError as exc:
            st.error(str(exc))

editor, reference = st.columns([1.6, 1], gap="large")
with editor:
    st.subheader("01 · Write your script")
    options = [name for name, (_, lang) in SCENARIOS.items() if not pocket or lang == "en"]
    st.selectbox("Example scenario", options, key="scenario", on_change=use_scenario)
    if pocket:
        st.caption("This Pocket preset uses English weights. Language is determined by the weights.")
    else:
        st.selectbox("Language", list(LANGUAGES), format_func=LANGUAGES.get, key="language")

with reference:
    st.subheader("02 · Choose a voice")
    source = st.radio("Reference audio source", ["Upload audio", "Record audio", "File path / URL", "None"], key="ref_source")
    speaker = None
    preview = None
    if source == "Upload audio":
        uploaded = st.file_uploader("Upload reference audio", type=["wav", "mp3", "flac", "ogg"], max_upload_size=30)
        if uploaded is not None:
            speaker = preview = uploaded.getvalue()
    elif source == "Record audio":
        recorded = st.audio_input("Record a reference voice", sample_rate=24000)
        if recorded is not None:
            speaker = preview = recorded.getvalue()
    elif source == "File path / URL":
        speaker = st.text_input("Audio path, URL or data URI", key="ref_path", placeholder="/path/to/model/reference.wav").strip() or None
    if preview:
        st.audio(preview)
    st.caption("Qwen uses speaker embeddings only; ref_text-based cloning is not supported. Pocket requires reference audio.")

with editor:
    with st.form("synthesis"):
        text = st.text_area("Text to synthesize", key="text", height=175, max_chars=4000)
        batch = st.checkbox("Generate each line separately", help="Up to 10 non-empty lines, using the same reference voice.")
        with st.expander("Generation settings"):
            seed = st.number_input("Seed", min_value=0, max_value=4294967294, value=42, step=1)
            frames = st.slider("Maximum generation steps", 32, 1024, 512, step=32)
            temperature = st.slider("Temperature", 0.0, 1.5, 0.8, step=0.05, disabled=pocket)
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
    elif source != "None" and not speaker:
        st.error("Provide reference audio or select None.")
    elif pocket and not speaker:
        st.error("Pocket TTS requires reference audio.")
    else:
        config = ModelConfig(model_path.strip(), mmproj_path.strip(), -1 if gpu else 0, mmproj_gpu, {"AUTO": None, "ON": True, "OFF": False}[fa], n_ctx)
        progress = st.progress(0, text="Preparing model. The first request loads the weights...")
        try:
            for index, sentence in enumerate(lines, 1):
                started = time.perf_counter()
                audio = get_engine().synthesize(
                    config, text=sentence, language=None if pocket else st.session_state.language,
                    speaker_reference=speaker, seed=int(seed), max_frames=frames,
                    temperature=temperature, response_format="wav",
                )
                metadata = dict(text=sentence, model=model_name, language=None if pocket else st.session_state.language,
                                duration=audio.duration, sample_rate=audio.sample_rate, n_samples=audio.n_samples,
                                elapsed=round(time.perf_counter() - started, 3), finish_reason=audio.finish_reason,
                                seed=int(seed), max_frames=frames, temperature=temperature, config=asdict(config))
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
            st.audio(item["data"], format="audio/wav")
            st.caption(f"{item['duration']:.2f} s · {item['sample_rate']} Hz · elapsed {item['elapsed']:.2f} s (including initial loading)")
            if item["finish_reason"] == "length":
                st.warning("Generation reached the step limit. Increase the limit or split the text if speech is incomplete.")
            st.download_button("Download WAV", data=item["data"], file_name=item["filename"], mime="audio/wav", key=f"download_{item['filename']}", on_click="ignore")
            with st.expander("Waveform and settings", expanded=False):
                with wave.open(io.BytesIO(item["data"]), "rb") as wav:
                    signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
                step = max(1, len(signal) // 1500)
                st.line_chart({"Amplitude": signal[::step]}, height=130)
                st.json({k: v for k, v in item.items() if k != "data"})
