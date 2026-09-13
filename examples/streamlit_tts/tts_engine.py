"""One reusable, serialized MTMD model pair for the local demo."""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import threading


@dataclass(frozen=True)
class ModelConfig:
    model: str
    mmproj: str
    gpu_layers: int = -1
    mmproj_gpu: bool = True
    flash_attn: bool | None = None
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
            return self._generator.create_speech(llama=self._llama, **request)
        finally:
            self._lock.release()
