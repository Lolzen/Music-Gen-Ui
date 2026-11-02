"""GPU-free stand-in that renders a simple seeded melody. For UI development and tests."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..prompting import CompiledRequest
from .base import Cancelled, CancelledFn, EngineResult, ProgressFn

SR = 48000
_SCALE = [0, 2, 4, 5, 7, 9, 11, 12]


class MockEngine:
    name = "mock"

    def __init__(self, delay: float = 0.4):
        self.delay = delay

    def info(self) -> dict:
        return {"engine": self.name, "note": "Platzhalter-Audio, kein Modell geladen"}

    def _stage(self, name: str, steps: int, progress: ProgressFn, cancelled: CancelledFn) -> None:
        for i in range(steps):
            if cancelled():
                raise Cancelled()
            if self.delay:
                time.sleep(self.delay / steps)
            progress(name, i + 1, steps)

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult:
        rng = np.random.default_rng(request.seed)
        lines = [l for s in request.sections for l in s.lines if l.strip()]
        n_notes = int(np.clip(len(lines) * 4, 16, 120))
        if request.cot != "off":
            self._stage("score", 8, progress, cancelled)
        self._stage("audio", 8, progress, cancelled)
        self._stage("nar", 4, progress, cancelled)
        self._stage("vae", 2, progress, cancelled)

        note_len = 0.25
        degrees = np.cumsum(rng.integers(-2, 3, n_notes)).clip(-3, 10)
        t = np.arange(int(SR * note_len)) / SR
        env = np.minimum(1, t * 40) * np.exp(-3 * t)
        notes = []
        for d in degrees:
            octave, step = divmod(int(d), len(_SCALE))
            freq = 261.63 * 2 ** ((_SCALE[step] + 12 * octave) / 12)
            notes.append(0.2 * env * np.sin(2 * np.pi * freq * t))
        mono = np.concatenate(notes).astype(np.float32)
        audio = np.stack([mono, np.roll(mono, 240)], axis=1)

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "audio.flac"
        sf.write(path, audio, SR)
        abc = None
        if request.cot != "off":
            abc = request.abc or ("X:1\nT:mock\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\n"
                                  "V:Vocal\n" + " ".join("CDEFGAB"[int(d) % 7] for d in degrees[:32]) + " |]\n")
            (out_dir / "score.abc").write_text(abc, encoding="utf-8")
        return EngineResult(path, SR, len(audio) / SR, False, abc)
