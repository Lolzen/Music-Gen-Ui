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
from .engine import Cancelled, Engine, EngineResult, create_engine
from .engine.base import CancelledFn, ProgressFn
from .memory import InsufficientMemory, MemoryLog, RamGuard, available_gib, is_oom, read_peaks, reset_peaks
from .predict import ProfileStore, profile_key
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
        self.profiles = ProfileStore(settings.data_dir / "stats" / "memory-profiles.json")

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

    def check_ram(self) -> None:
        """Refuse to start a song that would push host RAM below the guard limit."""
        limit = self.settings.ram_guard_gib
        free = available_gib()
        if limit <= 0 or free is None:
            return
        needed = self.engine.ram_needed_gib() + limit
        if free < needed:
            raise InsufficientMemory(
                f"Zu wenig freier Arbeitsspeicher: {free:.1f} GB frei, für diesen Song werden ca. "
                f"{needed:.1f} GB gebraucht (Modell + {limit:.1f} GB Sicherheitsreserve). "
                "Andere Programme schließen (z.B. andere WebUIs) oder die RAM-Notbremse in den Einstellungen anpassen.")

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult:
        """Blocking; waits for a running load to finish.

        Memory protection: RAM pre-check, RAM watchdog (emergency stop), cleanup
        after CUDA OOM, and moving the model to its idle placement afterwards.
        """
        with self._lock, MemoryLog(self.settings.data_dir / "logs", request.id) as mlog:
            s = self.settings
            mlog.event(f"Song startet (Engine {self.engine.name}, Offload {'an' if s.offload else 'aus'}, "
                       f"zwischen Songs: {s.idle_placement}, VRAM-Reserve {s.vram_reserve_gib} GB, "
                       f"RAM-Notbremse {s.ram_guard_gib} GB)")

            def tracked(stage: str, done: int, total: int) -> None:
                if stage != mlog.stage:
                    mlog.stage = stage
                    mlog.event("Phase beginnt")
                progress(stage, done, total)

            try:
                self.check_ram()
                self._load_locked(tracked)
            except Exception as exc:
                mlog.event(f"FEHLER vor dem Start: {exc}")
                raise
            self._set("busy")
            guard = RamGuard(s.ram_guard_gib)
            try:
                reset_peaks()
                with guard:
                    try:
                        result = self.engine.generate(request, out_dir, progress=tracked,
                                                      cancelled=lambda: cancelled() or guard.tripped)
                    except Cancelled:
                        if guard.tripped:
                            raise InsufficientMemory(guard.message) from None
                        raise
                mlog.stage = "fertig"
                peaks = read_peaks()
                mlog.event(f"Song fertig ({result.duration_s:.0f} s Musik), Spitzen: "
                           + ", ".join(f"{k}={v} GB" for k, v in peaks.items()))
                if self.engine.name == "yue2":
                    self.profiles.record(profile_key(s), peaks, result.duration_s)
                return result
            except Exception as exc:
                mlog.event(f"ABBRUCH: {type(exc).__name__}: {exc}")
                if guard.tripped or is_oom(exc):
                    log.warning("Freeing the model after memory error: %s", exc)
                    self.engine.unload()
                    mlog.event("Modell entladen")
                raise
            finally:
                try:
                    if self.engine.loaded:
                        self.engine.after_job(s.idle_placement)
                        mlog.stage = "danach"
                        mlog.event(f"Modell abgelegt: {s.idle_placement}")
                except Exception:  # noqa: BLE001 - never mask the job result
                    log.exception("Idle placement failed")
                if self._state == "busy":
                    self._set("ready" if self.engine.loaded else "unloaded")

    def shutdown(self) -> None:
        try:
            self.engine.unload()
        except Exception:  # noqa: BLE001
            log.exception("Unload on shutdown failed")
