"""Lua plugin runtime (optional, needs ``pip install lupa``).

Each plugin file gets its own Lua state with a reduced standard library
(no ``io``, ``os.execute``, ``require``, ``load`` ...). This prevents
accidents, it is not a security boundary: only install plugins you trust.
Data crosses the boundary as plain tables, never as Python objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .manager import HOOKS

_REMOVE = ("io", "require", "dofile", "loadfile", "load", "loadstring", "package",
           "debug", "collectgarbage", "python", "module")
_OS_KEEP = ("time", "clock", "date", "difftime")


def _runtime():
    try:
        from lupa import LuaRuntime
    except ImportError as exc:
        raise RuntimeError("Lua-Plugins brauchen das Paket 'lupa' (pip install '.[lua]')") from exc
    lua = LuaRuntime(register_eval=False, register_builtins=False, unpack_returned_tuples=True)
    g = lua.globals()
    os_table = g.os
    safe_os = lua.table_from({k: os_table[k] for k in _OS_KEEP if os_table[k] is not None})
    for name in _REMOVE:
        g[name] = None
    g.os = safe_os
    return lua


def to_python(value: Any, like: Any = None) -> Any:
    """Convert Lua tables to lists/dicts.

    An empty Lua table is ambiguous; ``like`` (the data originally passed in)
    decides whether it becomes ``[]`` or ``{}``.
    """
    from lupa import lua_type
    if lua_type(value) != "table":
        return value
    items = list(value.items())
    keys = [k for k, _ in items]
    if not keys:
        return [] if isinstance(like, list) else {}
    if all(isinstance(k, int) for k in keys) and sorted(keys) == list(range(1, len(keys) + 1)):
        ref = like[0] if isinstance(like, list) and like else None
        return [to_python(v, ref) for _, v in sorted(items)]
    ref = like if isinstance(like, dict) else {}
    return {str(k): to_python(v, ref.get(str(k))) for k, v in items}


def to_lua(lua, value: Any) -> Any:
    if isinstance(value, dict):
        return lua.table_from({k: to_lua(lua, v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return lua.table_from([to_lua(lua, v) for v in value])
    return value


def load_lua_plugin(path: Path, api) -> None:
    from lupa import lua_type
    lua = _runtime()
    plugin = lua.execute(path.read_text(encoding="utf-8"))
    if lua_type(plugin) != "table":
        raise TypeError("Lua plugin must `return { ... }` a table")
    presets = plugin["presets"]
    if presets is not None:
        for preset in to_python(presets):
            api.add_preset(preset)
    for hook in HOOKS:
        fn = plugin[hook]
        if fn is None:
            continue
        if lua_type(fn) != "function":
            raise TypeError(f"{hook} must be a function")
        api.on(hook, _wrap(lua, fn))


def _wrap(lua, fn):
    def call(data: dict):
        return to_python(fn(to_lua(lua, data)), data)
    return call
