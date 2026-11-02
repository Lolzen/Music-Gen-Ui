"""Owns the active engine: loading, unloading, switching settings, generating.

One lock serialises everything that touches the model. Generation holds it for
the whole song; UI actions that would interfere (switch settings, unload)
fail fast with :class:`Busy` instead of blocking the HTTP request.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path

from .config import Settings, vllm_installed, yue2_installed
from .engine import Engine, EngineResult, create_engine
from .engine.base import CancelledFn, ProgressFn
from .prompting import CompiledRequest

log = logging.getLogger(__name__)


class Busy(RuntimeError):
    pass


class EngineManager:
    def __init__(self, settings: Settings, engine: Engine | None = None):
        self.settings = settings
        self._fixed_engine = engine is not None  # tests inject an engine
        self.engine: Engine = engine or create_engine(settings)
        self._lock = threading.Lock()
        self._state = "ready" if self.engine.loaded else "unloaded"
        self._message: str | None = None
        self._progress = {"stage": None, "done": 0, "total": 0}
        self._since = time.time()

    # -- status --------------------------------------------------------------
    def _set(self, state: str, message: str | None = None) -> None:
        self._state, self._message, self._since = state, message, time.time()

    def _report(self, stage: str, done: int, total: int) -> None:
        self._progress = {"stage": stage, "done": done, "total": total}

    def status(self) -> dict:
        state = self._state
        if state == "ready" and not self.engine.loaded:  # e.g. unloaded by an error path
            state = "unloaded"
        return {
            "engine": self.engine.name,
            "configured": self.settings.engine,
            "state": state,  # unloaded | loading | ready | busy | error
            "message": self._message,
            "progress": self._progress if state == "loading" else None,
            "since": self._since,
            "info": self.engine.info(),
            "installed": {"yue2": yue2_installed(), "vllm": vllm_installed()},
        }

    # -- actions -------------------------------------------------------------
    def _acquire(self, action: str) -> None:
        if not self._lock.acquire(blocking=False):
            raise Busy(f"{action} geht nicht, während ein Song generiert oder das Modell geladen wird")

    def _load_locked(self, progress: ProgressFn) -> None:
        if self.engine.loaded:
            return
        self._set("loading")

        def both(stage, done, total):
            self._report(stage, done, total)
            progress(stage, done, total)

        try:
            self.engine.load(both)
        except Exception as exc:
            self._set("error", f"{type(exc).__name__}: {exc}")
            raise
        self._set("ready")

    def load_async(self) -> None:
        """Start loading in the background (returns immediately)."""
        self._acquire("Laden")

        def run():
            try:
                self._load_locked(lambda *_: None)
            except Exception:  # noqa: BLE001 - state carries the error
                log.exception("Model load failed")
            finally:
                self._lock.release()

        threading.Thread(target=run, name="engine-load", daemon=True).start()

    def unload(self) -> None:
        self._acquire("Entladen")
        try:
            self.engine.unload()
            self._set("ready" if self.engine.loaded else "unloaded")
        finally:
            self._lock.release()

    def configure(self, values: dict, *, persist: bool = True) -> None:
        """Apply new settings; recreates the engine if needed (raises ValueError/Busy)."""
        candidate = dataclasses.replace(self.settings)
        candidate.update(values)  # validate before touching anything
        self._acquire("Einstellungen ändern")
        try:
            changed = candidate.editable() != self.settings.editable()
            self.settings.update(values)
            if persist:
                self.settings.save()
            if changed and not self._fixed_engine:
                self.engine.unload()
                self.engine = create_engine(self.settings)
                self._set("ready" if self.engine.loaded else "unloaded")
        finally:
            self._lock.release()

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult:
        """Blocking; waits for a running load to finish."""
        with self._lock:
            self._load_locked(progress)
            self._set("busy")
            try:
                return self.engine.generate(request, out_dir, progress=progress, cancelled=cancelled)
            finally:
                if self._state == "busy":
                    self._set("ready" if self.engine.loaded else "unloaded")

    def shutdown(self) -> None:
        try:
            self.engine.unload()
        except Exception:  # noqa: BLE001
            log.exception("Unload on shutdown failed")
