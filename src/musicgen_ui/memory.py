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
