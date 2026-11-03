"""Engine interface. One engine instance owns the GPU; calls are serialised by the job queue."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ..prompting import CompiledRequest

# stage name, completed, total. total == 0 means "unknown" (e.g. token count while sampling,
# download bytes), the UI then shows the raw count instead of a percentage.
ProgressFn = Callable[[str, int, int], None]
CancelledFn = Callable[[], bool]


class Cancelled(Exception):
    pass


@dataclass
class EngineResult:
    audio_path: Path
    sample_rate: int
    duration_s: float
    truncated: bool = False
    score_abc: str | None = None
    extra: dict = field(default_factory=dict)


class Engine(Protocol):
    name: str

    @property
    def loaded(self) -> bool: ...

    def load(self, progress: ProgressFn) -> None:
        """Download/verify/load the model. Must be idempotent."""

    def unload(self) -> None:
        """Release the model and free (V)RAM."""

    def after_job(self, placement: str) -> None:
        """Move the model where it should rest between songs ("ram" | "gpu" | "unload")."""

    def ram_needed_gib(self) -> float:
        """Extra host RAM the next song may need (e.g. loading or parking the model)."""

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult: ...

    def info(self) -> dict: ...
