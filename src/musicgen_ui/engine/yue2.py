"""Adapter for the official ``yue2`` package (YuE2Pipeline).

The pipeline is loaded lazily on the first job and kept resident. YuE2 runs one
request at a time, which the job queue guarantees.
"""
from __future__ import annotations

import threading
from pathlib import Path

from ..config import Settings
from ..prompting import CompiledRequest
from .base import Cancelled, CancelledFn, EngineResult, ProgressFn


class YuE2Engine:
    name = "yue2"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._lock = threading.Lock()

    def info(self) -> dict:
        s = self.settings
        return {"engine": self.name, "model": s.model, "vae": s.vae, "device": s.device,
                "low_vram": s.low_vram, "quantization": s.quantization, "backend": s.backend,
                "loaded": self._pipe is not None}

    def _pipeline(self, progress: ProgressFn):
        with self._lock:
            if self._pipe is None:
                try:
                    from yue2 import YuE2Pipeline
                except ImportError as exc:  # pragma: no cover - depends on install
                    raise RuntimeError("Paket 'yue2' fehlt: pip install '.[yue2]' (siehe README)") from exc
                progress("load", 0, 0)
                s = self.settings
                self._pipe = YuE2Pipeline.from_pretrained(
                    s.model, vae=s.vae, device=s.device, low_vram=s.low_vram,
                    quantization=s.quantization, backend=s.backend,
                    memory_budget_gib=s.memory_budget_gib, progress=False)
            return self._pipe

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult:
        pipe = self._pipeline(progress)
        counts = {"abc": 0, "semantic": 0}
        limits = {"abc": (request.abc_sampling or {}).get("max_tokens", 4096),
                  "semantic": (request.semantic_sampling or {}).get("max_tokens", 9000)}
        names = {"abc": "score", "semantic": "audio"}

        def on_token(phase, _token):
            counts[phase] = counts.get(phase, 0) + 1
            if counts[phase] % 25 == 0:
                progress(names.get(phase, phase), counts[phase], limits.get(phase, 0))

        try:
            song = pipe(**request.engine_kwargs(), cancelled=cancelled, on_token=on_token,
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
