"""Runtime settings, read from environment variables (prefix ``MGU_``)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # "mock" runs without a GPU and produces placeholder audio (for UI work).
    engine: str = "mock"
    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 7860

    # YuE2 options (passed to YuE2Pipeline.from_pretrained)
    model: str = "m-a-p/YuE2-3B"
    vae: str = "m-a-p/YuE2-Vae"
    device: str = "auto"
    low_vram: bool = False
    quantization: str = "none"  # "none" | "fp8"
    backend: str = "torch"  # "torch" | "torch-eager" | "vllm"
    memory_budget_gib: float = 24.0

    # Mock engine: seconds of simulated work per stage (0 for tests).
    mock_delay: float = 0.4

    plugin_dirs: list[Path] = field(default_factory=lambda: [Path("plugins")])

    @property
    def library_dir(self) -> Path:
        return self.data_dir / "library"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        s = cls()
        s.engine = env("MGU_ENGINE", s.engine)
        s.data_dir = Path(env("MGU_DATA_DIR", str(s.data_dir)))
        s.host = env("MGU_HOST", s.host)
        s.port = int(env("MGU_PORT", str(s.port)))
        s.model = env("MGU_MODEL", s.model)
        s.vae = env("MGU_VAE", s.vae)
        s.device = env("MGU_DEVICE", s.device)
        s.low_vram = _bool(env("MGU_LOW_VRAM"), s.low_vram)
        s.quantization = env("MGU_QUANTIZATION", s.quantization)
        s.backend = env("MGU_BACKEND", s.backend)
        s.memory_budget_gib = float(env("MGU_MEMORY_BUDGET_GIB", str(s.memory_budget_gib)))
        s.mock_delay = float(env("MGU_MOCK_DELAY", str(s.mock_delay)))
        dirs = env("MGU_PLUGIN_DIRS")
        if dirs:
            s.plugin_dirs = [Path(p) for p in dirs.split(os.pathsep) if p]
        return s
