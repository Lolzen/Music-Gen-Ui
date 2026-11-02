"""Single-worker job queue with on-disk library.

YuE2 processes one request at a time per pipeline, so all generations go
through one worker thread. Each job lives in ``<library>/<job id>/`` with
``job.json`` (our metadata) next to the engine artifacts.
"""
from __future__ import annotations

import json
import logging
import queue
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .engine import Cancelled
from .models import SongSpec
from .plugins import PluginManager
from .prompting import compile_spec, make_id
from .runtime import EngineManager

log = logging.getLogger(__name__)
ACTIVE = ("queued", "running")


@dataclass
class Job:
    id: str
    group: str
    title: str
    spec: dict
    request: dict
    warnings: list[str] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict = field(default_factory=lambda: {"stage": None, "done": 0, "total": 0})
    error: str | None = None
    result: dict | None = None

    def public(self) -> dict:
        return asdict(self)


class JobQueue:
    def __init__(self, manager: EngineManager, library_dir: Path, plugins: PluginManager | None = None):
        self.manager = manager
        self.library_dir = library_dir
        self.plugins = plugins or PluginManager()
        self._jobs: dict[str, Job] = {}
        self._cancel: set[str] = set()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._load_library()

    # -- persistence ---------------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        return self.library_dir / job_id

    def _save(self, job: Job) -> None:
        directory = self._job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "job.json.tmp"
        tmp.write_text(json.dumps(job.public(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(directory / "job.json")

    def _load_library(self) -> None:
        for path in sorted(self.library_dir.glob("*/job.json")):
            try:
                job = Job(**json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping unreadable %s: %s", path, exc)
                continue
            if job.status in ACTIVE:  # server stopped mid-job
                job.status, job.error = "failed", "Server wurde während der Generierung beendet"
                self._save(job)
            self._jobs[job.id] = job

    # -- public API ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="generation-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout)
            self._thread = None

    def prepare(self, spec: SongSpec) -> tuple[SongSpec, list[str]]:
        """Run spec plugins; returns the (possibly modified) spec and warnings."""
        warnings: list[str] = []
        data = self.plugins.transform_spec(spec.model_dump(), warnings)
        return SongSpec.model_validate(data), warnings

    def preview(self, spec: SongSpec) -> dict:
        spec, warnings = self.prepare(spec)
        compiled = compile_spec(spec, seed=spec.seed if spec.seed is not None else 0, request_id="preview")
        native = self.plugins.transform_request({"style": compiled.style, "lyrics": compiled.lyrics}, warnings)
        compiled.style, compiled.lyrics = native["style"], native["lyrics"]
        compiled.warnings = warnings + compiled.warnings
        return compiled.to_dict()

    def submit(self, spec: SongSpec) -> list[Job]:
        spec, plugin_warnings = self.prepare(spec)
        base_seed = spec.seed if spec.seed is not None else secrets.randbelow(2**31)
        group = make_id(spec.title)
        jobs = []
        for i in range(spec.candidates):
            warnings = list(plugin_warnings)
            job_id = group if spec.candidates == 1 else f"{group}-t{i + 1}"
            compiled = compile_spec(spec, seed=base_seed + i, request_id=job_id)
            native = self.plugins.transform_request({"style": compiled.style, "lyrics": compiled.lyrics}, warnings)
            compiled.style, compiled.lyrics = native["style"], native["lyrics"]
            compiled.warnings = warnings + compiled.warnings
            title = spec.title if spec.candidates == 1 else f"{spec.title} (Take {i + 1})"
            job = Job(id=job_id, group=group, title=title, spec=spec.model_dump(),
                      request=compiled.to_dict(), warnings=compiled.warnings)
            with self._lock:
                self._jobs[job.id] = job
            self._save(job)
            self._queue.put(job.id)
            jobs.append(job)
        return jobs

    @property
    def running(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.status == "running"), None)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        with self._lock:
            if job.status == "queued":
                self._finish(job, "cancelled")
            elif job.status == "running":
                self._cancel.add(job_id)
        return job

    def delete(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status in ACTIVE:
            return False
        import shutil
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)
        with self._lock:
            self._jobs.pop(job_id, None)
        return True

    def artifact(self, job_id: str, name: str) -> Path | None:
        if job_id not in self._jobs or name not in {"audio.flac", "score.abc"}:
            return None
        path = self._job_dir(job_id) / name
        return path if path.is_file() else None

    # -- worker ------------------------------------------------------------------
    def _finish(self, job: Job, status: str, error: str | None = None) -> None:
        job.status, job.error, job.finished = status, error, time.time()
        self._save(job)
        self.plugins.on_complete(job.public())

    def _worker(self) -> None:
        from .prompting import CompiledRequest, Section

        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            job.status, job.started = "running", time.time()
            self._save(job)

            def progress(stage: str, done: int, total: int, job=job) -> None:
                job.progress = {"stage": stage, "done": done, "total": total}

            data = dict(job.request)
            data["sections"] = [Section(**s) for s in data.get("sections", [])]
            request = CompiledRequest(**data)
            try:
                result = self.manager.generate(request, self._job_dir(job.id), progress=progress,
                                               cancelled=lambda job_id=job.id: job_id in self._cancel)
            except Cancelled:
                self._finish(job, "cancelled")
            except Exception as exc:  # noqa: BLE001
                log.exception("Job %s failed", job.id)
                self._finish(job, "failed", describe_error(exc))
            else:
                job.result = {"duration_s": round(result.duration_s, 2), "sample_rate": result.sample_rate,
                              "truncated": result.truncated, "has_score": result.score_abc is not None,
                              "extra": _jsonable(result.extra)}
                self._finish(job, "done")
            finally:
                self._cancel.discard(job.id)


def describe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower():
        text += (" → Grafikspeicher voll. In den Einstellungen VRAM-Modus 'FP8' oder 'Low-VRAM' wählen, "
                 "andere GPU-Programme schließen oder max_tokens reduzieren.")
    return text


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
