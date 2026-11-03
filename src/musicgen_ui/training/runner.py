"""Starts/stops the training worker process and holds the GPU while it runs."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..config import Settings, yue2_installed
from ..runtime import Busy, EngineManager
from .project import ProjectStore


class TrainingRunner:
    def __init__(self, store: ProjectStore, manager: EngineManager, settings: Settings):
        self.store, self.manager, self.settings = store, manager, settings
        self.proc: subprocess.Popen | None = None
        self.project_id: str | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> str | None:
        return self.project_id if self.proc is not None and self.proc.poll() is None else None

    def _device(self) -> str:
        if self.settings.device not in ("auto", ""):
            return self.settings.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def start(self, project_id: str, *, fake: bool = False, steps: int = 0) -> None:
        with self._lock:
            if self.active:
                raise Busy(f"Es läuft bereits ein Training ({self.active})")
            project = self.store.load(project_id)
            problems = self.store.validate(project)
            if problems:
                raise ValueError("; ".join(problems))
            if not fake and not yue2_installed():
                raise ValueError("Training braucht YuE2 – ./install.sh (ohne --mock) ausführen")
            directory = self.store.dir(project_id)
            self.manager.reserve("training", f"LoRA-Training: {project.name}")
            try:
                (directory / "STOP").unlink(missing_ok=True)
                (directory / "status.json").unlink(missing_ok=True)
                log = open(directory / "train.log", "a", encoding="utf-8")
                log.write(f"\n===== Training {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                log.flush()
                cmd = [sys.executable, "-m", "musicgen_ui.training.worker", str(directory),
                       "--model", self.settings.model, "--device", self._device()]
                if fake:
                    cmd.append("--fake")
                if steps:
                    cmd += ["--steps", str(steps)]
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                self.project_id = project_id
            except Exception:
                self.manager.release_reservation()
                raise
            threading.Thread(target=self._watch, args=(self.proc, log, directory), daemon=True).start()

    def _watch(self, proc: subprocess.Popen, log, directory: Path) -> None:
        code = proc.wait()
        log.close()
        status = self.store.status(directory.name)
        if code != 0 and status.get("phase") not in ("error", "stopped"):
            import json
            status.update(phase="error", message=f"Training-Prozess beendet mit Code {code} – siehe Protokoll")
            (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
        self.manager.release_reservation()

    def stop(self, project_id: str) -> None:
        if self.active != project_id:
            return
        (self.store.dir(project_id) / "STOP").touch()
        proc = self.proc

        def force():
            time.sleep(120)
            if proc.poll() is None:
                proc.terminate()
        threading.Thread(target=force, daemon=True).start()

    def wait(self, timeout: float | None = None) -> int | None:
        return None if self.proc is None else self.proc.wait(timeout)
