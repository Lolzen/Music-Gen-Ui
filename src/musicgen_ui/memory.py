"""Host-RAM and VRAM helpers: status, pre-flight check, emergency brake.

Running out of *host* RAM makes Linux swap until the desktop freezes; a CUDA
OOM inside PyTorch only raises an exception. The RAM guard therefore checks
before each song and watches during generation, stopping the job (and
freeing the model) before the system starts thrashing.
"""
from __future__ import annotations

import gc
import re
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

    def __init__(self, directory: Path, job: str, interval: float = 1.0, keep: int = 7,
                 ram_reader=None, gpu_reader=None, telemetry_reader=None):
        import time
        self.directory, self.job, self.interval, self.keep = Path(directory), job, interval, keep
        self.ram_reader = ram_reader or ram_status
        self.gpu_reader = gpu_reader or _gpu_status
        self.telemetry_reader = telemetry_reader or gpu_telemetry
        self.stage = "start"
        self.path = self.directory / f"memory-{time.strftime('%Y-%m-%d')}.log"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-log", daemon=True)
        self._gpu_thread = threading.Thread(target=self._sample_gpu, name="memory-log-gpu", daemon=True)
        self._gpu_started: float | None = None
        self._gpu_last: tuple[float, str] | None = None
        self._lock = threading.Lock()

    def _gpu_text(self) -> str:
        parts = []
        try:
            gpu = self.gpu_reader()
        except Exception:  # noqa: BLE001 - logging must never break a song
            gpu = None
        if gpu and gpu.get("cuda"):
            parts.append(f"VRAM {gpu['used_gib']:.1f}/{gpu['total_gib']:.1f} GB "
                         f"(PyTorch reserviert {gpu.get('process_reserved_gib', 0):.1f})")
        telemetry = self.telemetry_reader()
        if telemetry:
            parts.append(telemetry)
        return "  ".join(parts)

    def _sample_gpu(self) -> None:
        """Own thread: a GPU busy with long kernels can block driver queries (nvidia-smi,
        mem_get_info); RAM lines must keep coming and the stall itself is reported."""
        import time
        while not self._stop.is_set():
            self._gpu_started = time.monotonic()
            text = self._gpu_text()
            self._gpu_last = (time.monotonic(), text)
            self._gpu_started = None
            self._stop.wait(self.interval)

    def _snapshot(self) -> str:
        import time
        parts = []
        ram = self.ram_reader()
        if ram:
            parts.append(f"RAM frei {ram['available_gib']:.1f}/{ram['total_gib']:.1f} GB")
            parts.append(f"Swap {ram['swap_used_gib']:.1f} GB")
        started, last = self._gpu_started, self._gpu_last
        if started is not None and time.monotonic() - started > max(3.0, 3 * self.interval):
            parts.append(f"GPU antwortet seit {time.monotonic() - started:.0f} s nicht (voll ausgelastet?)")
        elif last is not None and last[1]:
            parts.append(last[1])
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
        self._gpu_thread.start()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


_SMI_FIELDS = "power.draw,power.limit,temperature.gpu,clocks.sm,utilization.gpu,pstate"
_SMI_PCIE = ",pcie.link.gen.current,pcie.link.width.current"
_smi_available: bool | None = None
_smi_pcie = True                      # dropped if this nvidia-smi does not know the PCIe fields
_replays = {"value": None, "at": 0.0}
REPLAY_INTERVAL_S = 15.0


def _run_nowait(cmd: list[str], timeout: float):
    """subprocess.run(), but a child stuck in the driver cannot block us after the timeout."""
    import subprocess
    from types import SimpleNamespace
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()  # no second communicate(): it would wait for an uninterruptible child
        raise
    return SimpleNamespace(returncode=proc.returncode, stdout=out)


def parse_replays(text: str) -> int | None:
    match = re.search(r"Replays Since Reset\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _pcie_replays(now: float) -> int | None:
    """PCIe replay counter (link errors); read every 15 s because `nvidia-smi -q` is slower."""
    import subprocess
    if now - _replays["at"] >= REPLAY_INTERVAL_S:
        _replays["at"] = now
        try:
            out = _run_nowait(["nvidia-smi", "-q", "-i", "0"], 3).stdout
            _replays["value"] = parse_replays(out)
        except Exception:  # noqa: BLE001
            _replays["value"] = None
    return _replays["value"]


def gpu_telemetry() -> str | None:
    """Power, temperature, clock, load and PCIe link via nvidia-smi (no root needed).

    Shows whether a freeze coincides with a power/heat spike or PCIe link errors."""
    global _smi_available, _smi_pcie
    if _smi_available is False:
        return None
    import shutil
    import subprocess
    import time
    if _smi_available is None:
        _smi_available = shutil.which("nvidia-smi") is not None
        if not _smi_available:
            return None
    try:
        fields = _SMI_FIELDS + (_SMI_PCIE if _smi_pcie else "")
        result = _run_nowait(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"], 2)
        if result.returncode != 0 and _smi_pcie:
            _smi_pcie = False  # older/newer nvidia-smi without these fields: keep the rest working
            return gpu_telemetry()
        values = [v.strip() for v in result.stdout.strip().split(",")]
        power, limit, temp, clock, util, pstate = values[:6]
        text = f"GPU {float(power):.0f}/{float(limit):.0f} W  {temp} °C  {clock} MHz  Last {util} %  {pstate}"
        if _smi_pcie and len(values) >= 8:
            text += f"  PCIe Gen{values[6]} x{values[7]}"
        replays = _pcie_replays(time.monotonic())
        if replays is not None:
            text += f"  Replays {replays}"
        return text
    except Exception:  # noqa: BLE001 - telemetry is best effort
        return None


def last_log_line(directory: Path, job: str) -> str | None:
    """Last memory-log line of a job (used to explain a song interrupted by a crash)."""
    marker = f"job={job} "
    for file in sorted(Path(directory).glob("memory-*.log"), reverse=True)[:2]:
        try:
            lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if marker in line:
                return line
    return None


def _gpu_status() -> dict:
    from .system import gpu_status
    return gpu_status()


def latest_log(directory: Path, lines: int = 300) -> tuple[str | None, str]:
    files = sorted(Path(directory).glob("memory-*.log"))
    if not files:
        return None, ""
    text = files[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    return files[-1].name, "\n".join(text[-lines:])


# ---------------------------------------------------------------- per-process peaks
def _status_kib(field: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_rss_gib() -> float | None:
    kib = _status_kib("VmRSS")
    return None if kib is None else kib / 2**20


def _torch_cuda():
    import sys
    torch = sys.modules.get("torch")  # never import torch just for statistics
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
    except AttributeError:  # torch is still being imported by another thread
        return None
    return torch


def reset_peaks() -> None:
    """Start a new measurement window for peak RSS and peak PyTorch VRAM."""
    try:
        Path("/proc/self/clear_refs").write_text("5")  # resets VmHWM (Linux)
    except OSError:
        pass
    torch = _torch_cuda()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def read_peaks() -> dict:
    peaks = {}
    hwm = _status_kib("VmHWM")
    if hwm is not None:
        peaks["ram_peak_gib"] = round(hwm / 2**20, 2)
    torch = _torch_cuda()
    if torch is not None:
        peaks["vram_peak_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 2)
    return peaks


def process_vram_gib() -> float:
    torch = _torch_cuda()
    return 0.0 if torch is None else torch.cuda.memory_reserved() / GIB
