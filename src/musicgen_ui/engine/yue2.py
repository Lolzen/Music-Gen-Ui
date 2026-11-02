"""Adapter for the official ``yue2`` package (YuE2Pipeline).

``load()`` downloads (first run) and verifies the weights via
``YuE2Pipeline.from_pretrained``; YuE2 moves them to the GPU on the first
generation. The pipeline stays resident until ``unload()``. YuE2 runs one
request at a time, which the engine manager guarantees.
"""
from __future__ import annotations

import gc
from pathlib import Path

from ..config import Settings
from ..download import DownloadMonitor
from ..prompting import CompiledRequest
from .base import Cancelled, CancelledFn, EngineResult, ProgressFn

INSTALL_HINT = "Paket 'yue2' fehlt – ./install.sh ausführen (siehe README)"
SEMANTIC_HZ = 25


class YuE2Engine:
    name = "yue2"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def info(self) -> dict:
        s = self.settings
        return {"engine": self.name, "model": s.model, "vae": s.vae, "device": s.device,
                "low_vram": s.low_vram, "quantization": s.quantization, "backend": s.backend,
                "loaded": self.loaded}

    def load(self, progress: ProgressFn) -> None:
        if self._pipe is not None:
            return
        try:
            from yue2 import YuE2Pipeline
        except ImportError as exc:
            raise RuntimeError(INSTALL_HINT) from exc
        s = self.settings
        progress("download", 0, 0)
        with DownloadMonitor([s.model, s.vae], lambda size: progress("download", size, 0)):
            self._pipe = YuE2Pipeline.from_pretrained(
                s.model, vae=s.vae, device=s.device, low_vram=s.low_vram,
                quantization=s.quantization, backend=s.backend,
                memory_budget_gib=s.memory_budget_gib, progress=False)

    def unload(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            pipe.close()  # drops model/VAE and empties the CUDA cache
            del pipe
            gc.collect()

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult:
        self.load(progress)
        names = {"abc": "score", "semantic": "audio"}
        counts: dict[str, int] = {}

        def on_token(phase, _token):
            counts[phase] = counts.get(phase, 0) + 1
            if counts[phase] % SEMANTIC_HZ == 0:  # ~ once per second of music
                progress(names.get(phase, phase), counts[phase], 0)

        try:
            song = self._pipe(**request.engine_kwargs(), cancelled=cancelled, on_token=on_token,
                              on_progress=lambda stage, done, total: progress(stage, done, total))
        except InterruptedError as exc:
            raise Cancelled() from exc
        result = song.save_artifacts(out_dir)
        score = out_dir / "score.abc"
        return EngineResult(
            audio_path=out_dir / "audio.flac", sample_rate=song.sample_rate,
            duration_s=float(result["audio_seconds"]), truncated=bool(result["truncated"]),
            score_abc=score.read_text(encoding="utf-8") if score.exists() else None,
            extra={"timing": result.get("timing"), "weights": result.get("weights")})
