"""Runtime settings.

Precedence (lowest to highest): defaults → ``<data_dir>/settings.json``
(saved from the UI) → environment variables (``MGU_*``) → command line.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

ENGINES = ("auto", "yue2", "mock")
QUANTIZATIONS = ("none", "fp8")
BACKENDS = ("torch", "torch-eager", "vllm")
DEVICES = ("auto", "cuda", "cpu", "mps", "xpu")

# Settings the UI may change and persist.
EDITABLE = ("engine", "model", "vae", "device", "low_vram", "quantization", "backend", "memory_budget_gib")

_ENV = {
    "engine": "MGU_ENGINE", "model": "MGU_MODEL", "vae": "MGU_VAE", "device": "MGU_DEVICE",
    "low_vram": "MGU_LOW_VRAM", "quantization": "MGU_QUANTIZATION", "backend": "MGU_BACKEND",
    "memory_budget_gib": "MGU_MEMORY_BUDGET_GIB", "host": "MGU_HOST", "port": "MGU_PORT",
    "mock_delay": "MGU_MOCK_DELAY",
}


def _installed(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def yue2_installed() -> bool:
    return _installed("yue2")


def vllm_installed() -> bool:
    return _installed("vllm")


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # "auto" = YuE2 if installed, otherwise the mock engine.
    engine: str = "auto"
    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 7860

    # YuE2 options (passed to YuE2Pipeline.from_pretrained)
    model: str = "m-a-p/YuE2-3B"
    vae: str = "m-a-p/YuE2-Vae"
    device: str = "auto"
    low_vram: bool = False
    quantization: str = "none"
    backend: str = "torch"
    memory_budget_gib: float = 24.0

    # Mock engine: seconds of simulated work per job (0 for tests).
    mock_delay: float = 2.0

    plugin_dirs: list[Path] = field(default_factory=lambda: [Path("plugins")])

    @property
    def library_dir(self) -> Path:
        return self.data_dir / "library"

    @property
    def settings_file(self) -> Path:
        return self.data_dir / "settings.json"

    def resolved_engine(self) -> str:
        if self.engine == "auto":
            return "yue2" if yue2_installed() else "mock"
        return self.engine

    # -- editable subset ---------------------------------------------------
    def editable(self) -> dict:
        return {k: getattr(self, k) for k in EDITABLE}

    def update(self, values: dict) -> None:
        """Validate and apply editable values (raises ValueError)."""
        values = {k: v for k, v in values.items() if k in EDITABLE}
        checks = {"engine": ENGINES, "quantization": QUANTIZATIONS, "backend": BACKENDS, "device": DEVICES}
        for key, allowed in checks.items():
            if key in values and values[key] not in allowed:
                raise ValueError(f"{key} muss eines von {', '.join(allowed)} sein")
        for key in ("model", "vae"):
            if key in values and (not isinstance(values[key], str) or not values[key].strip()):
                raise ValueError(f"{key} darf nicht leer sein")
        if "low_vram" in values:
            values["low_vram"] = bool(values["low_vram"])
        if "memory_budget_gib" in values:
            budget = float(values["memory_budget_gib"])
            if not 4 <= budget <= 1024:
                raise ValueError("memory_budget_gib muss zwischen 4 und 1024 liegen")
            values["memory_budget_gib"] = budget
        merged = {**self.editable(), **values}
        if merged["low_vram"] and merged["backend"] == "vllm":
            raise ValueError("low_vram funktioniert nicht mit dem vllm-Backend")
        for key, value in values.items():
            setattr(self, key, value.strip() if isinstance(value, str) else value)

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(json.dumps(self.editable(), indent=2), encoding="utf-8")

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, data_dir: Path | None = None, *, env: dict | None = None) -> "Settings":
        env = os.environ if env is None else env
        s = cls()
        s.data_dir = Path(data_dir or env.get("MGU_DATA_DIR") or s.data_dir)
        if s.settings_file.is_file():
            try:
                s.update(json.loads(s.settings_file.read_text(encoding="utf-8")))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                log.warning("Ignoring invalid %s: %s", s.settings_file, exc)
        types = {f.name: f.type for f in fields(cls)}
        for key, var in _ENV.items():
            raw = env.get(var)
            if raw is None:
                continue
            kind = types[key]  # string annotations (from __future__ import annotations)
            value = {"bool": _bool, "int": int, "float": float}.get(kind, str)(raw)
            if key in EDITABLE:
                s.update({key: value})
            else:
                setattr(s, key, value)
        dirs = env.get("MGU_PLUGIN_DIRS")
        if dirs:
            s.plugin_dirs = [Path(p) for p in dirs.split(os.pathsep) if p]
        return s

    # Backwards-compatible alias
    from_env = load

    def to_dict(self) -> dict:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["plugin_dirs"] = [str(p) for p in self.plugin_dirs]
        return data
