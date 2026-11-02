"""Engine interface. One engine instance owns the GPU; calls are serialised by the job queue."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ..prompting import CompiledRequest

# stage name, completed, total (total may be 0 when unknown)
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

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult: ...

    def info(self) -> dict: ...
