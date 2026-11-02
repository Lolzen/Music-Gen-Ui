"""Adapter for the official ``yue2`` package (YuE2Pipeline).

``load()`` downloads (first run) and verifies the weights via
``YuE2Pipeline.from_pretrained``; YuE2 moves them to the GPU on the first
generation. The pipeline stays resident until ``unload()``. YuE2 runs one
request at a time, which the engine manager guarantees.

Compatibility: the ``yue2`` API differs between the release tag
``yue2-v0.1.6`` and later main-branch commits (both report version 0.1.6).
The release has ``offload_ar`` but no ``low_vram`` and no ``on_progress``;
newer commits have both. :func:`detect_api` inspects the installed
package and only passes what it accepts.
"""
from __future__ import annotations

import gc
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Settings
from ..download import DownloadMonitor
from ..prompting import CompiledRequest
from .base import Cancelled, CancelledFn, EngineResult, ProgressFn

INSTALL_HINT = "Paket 'yue2' fehlt – ./install.sh ausführen (siehe README)"
SEMANTIC_HZ = 25


@dataclass(frozen=True)
class YuE2Api:
    version: str
    low_vram: bool        # YuE2Pipeline(low_vram=...) – newer commits
    offload_ar: bool      # YuE2Pipeline(offload_ar=...) – release 0.1.6
    stage_progress: bool  # pipe(..., on_progress=...) – newer commits
    init_params: frozenset


def _named_params(fn) -> frozenset:
    return frozenset(name for name, p in inspect.signature(fn).parameters.items()
                     if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL))


def detect_api(pipeline_cls) -> YuE2Api:
    try:
        from importlib.metadata import version
        ver = version("yue2-infer")
    except Exception:  # noqa: BLE001 - fake/dev installs
        ver = "unbekannt"
    init = _named_params(pipeline_cls.__init__)
    call = _named_params(pipeline_cls.__call__)
    return YuE2Api(ver, "low_vram" in init, "offload_ar" in init, "on_progress" in call, init)


class YuE2Engine:
    name = "yue2"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self.api: YuE2Api | None = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def info(self) -> dict:
        s = self.settings
        return {"engine": self.name, "model": s.model, "vae": s.vae, "device": s.device,
                "low_vram": s.low_vram, "quantization": s.quantization, "backend": s.backend,
                "loaded": self.loaded, "low_vram_mode": self._low_vram_mode(),
                "yue2": None if self.api is None else
                {k: v for k, v in asdict(self.api).items() if k != "init_params"}}

    def _low_vram_mode(self) -> str | None:
        """How the low-VRAM setting is realised by the installed YuE2."""
        if not self.settings.low_vram:
            return None
        if self.api is None or self.api.low_vram:
            return "low_vram"
        return "offload_ar" if self.api.offload_ar else "unsupported"

    def _pipeline_kwargs(self) -> dict:
        s = self.settings
        kwargs = {"device": s.device, "quantization": s.quantization, "backend": s.backend,
                  "memory_budget_gib": s.memory_budget_gib}
        mode = self._low_vram_mode()
        if mode in ("low_vram", "offload_ar"):
            kwargs[mode] = True
        return {k: v for k, v in kwargs.items() if k in self.api.init_params}

    def load(self, progress: ProgressFn) -> None:
        if self._pipe is not None:
            return
        try:
            from yue2 import YuE2Pipeline
        except ImportError as exc:
            raise RuntimeError(INSTALL_HINT) from exc
        s = self.settings
        self.api = detect_api(YuE2Pipeline)
        progress("download", 0, 0)
        with DownloadMonitor([s.model, s.vae], lambda size: progress("download", size, 0)):
            self._pipe = YuE2Pipeline.from_pretrained(s.model, vae=s.vae, progress=False,
                                                      **self._pipeline_kwargs())

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

        call = {**request.engine_kwargs(), "cancelled": cancelled, "on_token": on_token}
        if self.api.stage_progress:
            call["on_progress"] = lambda stage, done, total: progress(stage, done, total)
        try:
            song = self._pipe(**call)
        except InterruptedError as exc:
            raise Cancelled() from exc
        result = song.save_artifacts(out_dir)
        score = out_dir / "score.abc"
        return EngineResult(
            audio_path=out_dir / "audio.flac", sample_rate=song.sample_rate,
            duration_s=float(result["audio_seconds"]), truncated=bool(result["truncated"]),
            score_abc=score.read_text(encoding="utf-8") if score.exists() else None,
            extra={"timing": result.get("timing"), "weights": result.get("weights")})
