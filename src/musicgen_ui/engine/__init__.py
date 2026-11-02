from __future__ import annotations

from ..config import Settings
from .base import Cancelled, Engine, EngineResult


def create_engine(settings: Settings) -> Engine:
    if settings.engine == "mock":
        from .mock import MockEngine
        return MockEngine(delay=settings.mock_delay)
    if settings.engine == "yue2":
        from .yue2 import YuE2Engine
        return YuE2Engine(settings)
    raise ValueError(f"Unknown engine {settings.engine!r} (expected 'mock' or 'yue2')")


__all__ = ["Cancelled", "Engine", "EngineResult", "create_engine"]
