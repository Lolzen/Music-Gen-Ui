from __future__ import annotations

from ..config import Settings
from .base import Cancelled, Engine, EngineResult


def create_engine(settings: Settings) -> Engine:
    name = settings.resolved_engine()
    if name == "mock":
        from .mock import MockEngine
        return MockEngine(delay=settings.mock_delay)
    if name == "yue2":
        from .yue2 import YuE2Engine
        return YuE2Engine(settings)
    raise ValueError(f"Unknown engine {name!r} (expected 'auto', 'yue2' or 'mock')")


__all__ = ["Cancelled", "Engine", "EngineResult", "create_engine"]
