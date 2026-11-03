"""Host-RAM and VRAM helpers: status, pre-flight check, emergency brake.

Running out of *host* RAM makes Linux swap until the desktop freezes; a CUDA
OOM inside PyTorch only raises an exception. The RAM guard therefore checks
before each song and watches during generation, stopping the job (and
freeing the model) before the system starts thrashing.
"""
from __future__ import annotations

import gc
import threading
from pathlib import Path

GIB = 2**30


def ram_status() -> dict | None:
    """Physical RAM from /proc/meminfo (Linux); swap is deliberately not counted."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    total, available = values["MemTotal"], values["MemAvailable"]
    return {"total_gib": round(total / GIB, 2), "available_gib": round(available / GIB, 2),
            "used_gib": round((total - available) / GIB, 2),
            "swap_used_gib": round((values.get("SwapTotal", 0) - values.get("SwapFree", 0)) / GIB, 2)}


def available_gib() -> float | None:
    status = ram_status()
    return None if status is None else status["available_gib"]


def free_cuda_cache() -> None:
    """Return cached-but-unused VRAM to the driver (desktop, other programs)."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    return "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()


class InsufficientMemory(RuntimeError):
    pass


class RamGuard:
    """Background watchdog. ``tripped`` turns True when free RAM drops below the limit."""

    def __init__(self, limit_gib: float, interval: float = 0.5, reader=None):
        self.limit_gib, self.interval = limit_gib, interval
        self.reader = reader or available_gib
        self.tripped = False
        self.lowest_gib: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ram-guard", daemon=True)

    @property
    def enabled(self) -> bool:
        return self.limit_gib > 0 and self.reader() is not None

    def _check(self) -> None:
        free = self.reader()
        if free is None:
            return
        self.lowest_gib = free if self.lowest_gib is None else min(self.lowest_gib, free)
        if free < self.limit_gib:
            self.tripped = True

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._check()

    @property
    def message(self) -> str:
        return (f"RAM-Notbremse: nur noch {self.lowest_gib:.1f} GB Arbeitsspeicher frei "
                f"(Grenze {self.limit_gib:.1f} GB). Der Song wurde abgebrochen und das Modell entladen, "
                "damit das System nicht einfriert. Andere Programme schließen oder unter ⚙ Einstellungen "
                "'Modell zwischen Songs: entladen' wählen.")

    def __enter__(self):
        if self.enabled:
            self._check()
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


class MemoryLog:
    """Writes RAM/swap/VRAM every few seconds while a song is generated.

    Every line is flushed and fsync'ed, so the last readings survive a frozen
    or force-restarted system. Works without systemd/journald (runit, etc.).
    Files: ``<data>/logs/memory-YYYY-MM-DD.log``, the newest ``keep`` are kept.
    """

    def __init__(self, directory: Path, job: str, interval: float = 2.0, keep: int = 7,
                 ram_reader=None, gpu_reader=None):
        import time
        self.directory, self.job, self.interval, self.keep = Path(directory), job, interval, keep
        self.ram_reader = ram_reader or ram_status
        self.gpu_reader = gpu_reader or _gpu_status
        self.stage = "start"
        self.path = self.directory / f"memory-{time.strftime('%Y-%m-%d')}.log"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-log", daemon=True)
        self._lock = threading.Lock()

    def _snapshot(self) -> str:
        parts = []
        ram = self.ram_reader()
        if ram:
            parts.append(f"RAM frei {ram['available_gib']:.1f}/{ram['total_gib']:.1f} GB")
            parts.append(f"Swap {ram['swap_used_gib']:.1f} GB")
        try:
            gpu = self.gpu_reader()
        except Exception:  # noqa: BLE001 - logging must never break a song
            gpu = None
        if gpu and gpu.get("cuda"):
            parts.append(f"VRAM {gpu['used_gib']:.1f}/{gpu['total_gib']:.1f} GB "
                         f"(PyTorch reserviert {gpu.get('process_reserved_gib', 0):.1f})")
        return "  ".join(parts)

    def write(self, text: str) -> None:
        import os
        import time
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  job={self.job}  phase={self.stage:<9} {text}\n"
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            pass

    def event(self, text: str) -> None:
        self.write(f"{text}  |  {self._snapshot()}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.write(self._snapshot())

    def _prune(self) -> None:
        files = sorted(self.directory.glob("memory-*.log"))
        for old in files[:-self.keep]:
            old.unlink(missing_ok=True)

    def __enter__(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._prune()
        except OSError:
            pass
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def _gpu_status() -> dict:
    from .system import gpu_status
    return gpu_status()


def latest_log(directory: Path, lines: int = 300) -> tuple[str | None, str]:
    files = sorted(Path(directory).glob("memory-*.log"))
    if not files:
        return None, ""
    text = files[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    return files[-1].name, "\n".join(text[-lines:])
