"""One reusable, serialized MTMD model pair for the local demo."""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    model: str
    mmproj: str
    gpu_layers: int = -1
    mmproj_gpu: bool = True
    flash_attn: Optional[bool] = None
    n_ctx: int = 4096


class TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._resources = ExitStack()
        self._config = None
        self._llama = None
        self._generator = None

    def _unload(self):
        try:
            self._resources.close()
        finally:
            self._resources = ExitStack()
            self._config = self._llama = self._generator = None

    def close(self):
        with self._lock:
            self._unload()

    def unload(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("The model is generating audio. Wait before unloading.")
        try:
            self._unload()
        finally:
            self._lock.release()

    def _load(self, config):
        if self._config == config:
            return
        for label, filename in (("Backbone", config.model), ("mmproj", config.mmproj)):
            if not Path(filename).is_file():
                raise ValueError(f"{label} file does not exist: {filename}")
        from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
        from llama_cpp.llama_multimodal import MTMDAudioGenerator

        self._unload()
        with ExitStack() as resources:
            llama = Llama(
                model_path=config.model, embeddings=True,
                pooling_type=LLAMA_POOLING_TYPE_NONE, n_ctx=config.n_ctx,
                n_batch=min(512, config.n_ctx), n_gpu_layers=config.gpu_layers,
                verbose=False,
            )
            resources.callback(llama.close)
            generator = MTMDAudioGenerator(
                mmproj_path=config.mmproj, use_gpu=config.mmproj_gpu,
                flash_attn=config.flash_attn, verbose=False,
            )
            resources.callback(generator.close)
            self._resources = resources.pop_all()
            self._llama, self._generator, self._config = llama, generator, config

    def synthesize(self, config, **request):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Another request is using the model. Please try again shortly.")
        try:
            self._load(config)
            assert self._generator is not None and self._llama is not None
            return self._generator.create_speech(llama=self._llama, **request)
        finally:
            self._lock.release()

    def describe(self, config):
        """Load the pair and copy voice names owned by the native context."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Another request is using the model. Please try again shortly.")
        try:
            self._load(config)
            assert self._generator is not None and self._llama is not None
            self._generator._init_mtmd_context(self._llama)
            mtmd, ctx = self._generator._mtmd_cpp, self._generator.mtmd_ctx
            info = mtmd.mtmd_gen_audio_get_info(ctx)
            voices = []
            for index in range(info.n_voices):
                name = mtmd.mtmd_gen_audio_voice_name(ctx, index)
                if name:
                    voices.append(name.decode("utf-8"))
            return dict(sample_rate=info.sample_rate, voices=voices)
        except Exception:
            # A failed projector initialization must not retain a partial pair.
            self._unload()
            raise
        finally:
            self._lock.release()
