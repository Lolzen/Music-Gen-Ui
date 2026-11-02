"""Plugin loading and hook dispatch.

Plugins are trusted local code. They are loaded from the configured plugin
directories:

* ``*.py``  – must define ``register(api)``
* ``*.lua`` – must ``return`` a table (see ``plugins/example_presets.lua``)

Hooks
-----
``transform_spec(spec: dict) -> dict | None``
    Edit the structured song spec before it is compiled (e.g. style macros).
``transform_request(req: dict) -> dict | None``
    Edit the final ``{"style", "lyrics"}`` sent to the model.
``on_complete(job: dict) -> None``
    Called after a job finished (any status).

A failing hook never breaks a generation; the error is reported as a warning.
"""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)
HOOKS = ("transform_spec", "transform_request", "on_complete")
REQUEST_KEYS = ("style", "lyrics")


@dataclass
class PluginInfo:
    name: str
    path: str
    kind: str  # "python" | "lua"
    hooks: list[str] = field(default_factory=list)
    presets: int = 0
    error: str | None = None


class PluginAPI:
    """Object handed to ``register(api)`` of Python plugins."""

    def __init__(self, manager: "PluginManager", info: PluginInfo):
        self._manager, self._info = manager, info

    def on(self, hook: str, fn: Callable) -> None:
        if hook not in HOOKS:
            raise ValueError(f"Unknown hook {hook!r}; available: {', '.join(HOOKS)}")
        self._manager._hooks[hook].append((self._info.name, fn))
        self._info.hooks.append(hook)

    def add_preset(self, preset: dict) -> None:
        if not isinstance(preset, dict) or not preset.get("name"):
            raise ValueError("A preset needs at least a 'name'")
        self._manager.presets.append({**preset, "source": self._info.name})
        self._info.presets += 1


class PluginManager:
    def __init__(self) -> None:
        self.plugins: list[PluginInfo] = []
        self.presets: list[dict] = []
        self._hooks: dict[str, list[tuple[str, Callable]]] = {h: [] for h in HOOKS}

    # -- loading -----------------------------------------------------------
    def load_dirs(self, dirs: list[Path]) -> None:
        for directory in dirs:
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.name.startswith(("_", ".")):
                    continue
                if path.suffix == ".py":
                    self._load(path, "python", self._load_python)
                elif path.suffix == ".lua":
                    self._load(path, "lua", self._load_lua)

    def _load(self, path: Path, kind: str, loader: Callable[[Path, PluginAPI], None]) -> None:
        info = PluginInfo(name=path.stem, path=str(path), kind=kind)
        self.plugins.append(info)
        try:
            loader(path, PluginAPI(self, info))
        except Exception as exc:  # noqa: BLE001 - report any plugin failure
            info.error = f"{type(exc).__name__}: {exc}"
            log.warning("Plugin %s failed to load: %s", path, info.error)

    @staticmethod
    def _load_python(path: Path, api: PluginAPI) -> None:
        spec = importlib.util.spec_from_file_location(f"mgu_plugin_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        register = getattr(module, "register", None)
        if not callable(register):
            raise TypeError("Python plugin must define register(api)")
        register(api)

    @staticmethod
    def _load_lua(path: Path, api: PluginAPI) -> None:
        from .lua import load_lua_plugin
        load_lua_plugin(path, api)

    # -- dispatch ----------------------------------------------------------
    def _transform(self, hook: str, data: dict, warnings: list[str],
                   allowed: tuple[str, ...] | None = None) -> dict:
        for name, fn in self._hooks[hook]:
            try:
                result = fn(dict(data))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Plugin {name}.{hook} fehlgeschlagen: {exc}")
                continue
            if result is None:
                continue
            if not isinstance(result, dict):
                warnings.append(f"Plugin {name}.{hook} muss ein dict/table liefern")
                continue
            if allowed is not None:
                result = {k: v for k, v in result.items() if k in allowed and isinstance(v, str)}
            data = {**data, **result}
        return data

    def transform_spec(self, spec: dict, warnings: list[str]) -> dict:
        return self._transform("transform_spec", spec, warnings)

    def transform_request(self, request: dict, warnings: list[str]) -> dict:
        return self._transform("transform_request", request, warnings, REQUEST_KEYS)

    def on_complete(self, job: dict) -> None:
        for name, fn in self._hooks["on_complete"]:
            try:
                fn(dict(job))
            except Exception as exc:  # noqa: BLE001
                log.warning("Plugin %s.on_complete failed: %s", name, exc)

    def summary(self) -> list[dict[str, Any]]:
        return [vars(p) for p in self.plugins]
