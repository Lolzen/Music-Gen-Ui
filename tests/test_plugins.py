from pathlib import Path

import pytest

from musicgen_ui.plugins import PluginManager

ROOT = Path(__file__).resolve().parents[1]


def test_python_plugin(tmp_path):
    (tmp_path / "upper.py").write_text(
        "def register(api):\n"
        "    api.add_preset({'name': 'P'})\n"
        "    api.on('transform_request', lambda r: {'style': r['style'].upper(), 'seed': 1})\n")
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n")
    (tmp_path / "_ignored.py").write_text("raise RuntimeError('never loaded')\n")
    pm = PluginManager()
    pm.load_dirs([tmp_path])
    assert {p.name: p.error is None for p in pm.plugins} == {"broken": False, "upper": True}
    warnings = []
    out = pm.transform_request({"style": "pop", "lyrics": "x"}, warnings)
    assert out == {"style": "POP", "lyrics": "x"}  # non-whitelisted key dropped
    assert pm.presets == [{"name": "P", "source": "upper"}]


def test_failing_hook_becomes_warning(tmp_path):
    (tmp_path / "bad.py").write_text(
        "def register(api):\n    api.on('transform_spec', lambda s: 1/0)\n")
    pm = PluginManager()
    pm.load_dirs([tmp_path])
    warnings = []
    assert pm.transform_spec({"style": "a"}, warnings) == {"style": "a"}
    assert warnings and "bad.transform_spec" in warnings[0]


def test_lua_example_plugin():
    pytest.importorskip("lupa")
    pm = PluginManager()
    pm.load_dirs([ROOT / "plugins"])
    lua = next(p for p in pm.plugins if p.name == "example_presets")
    assert lua.error is None and lua.kind == "lua"
    names = [p["name"] for p in pm.presets]
    assert "Synthwave Duett" in names
    duet = next(p for p in pm.presets if p["name"] == "Synthwave Duett")
    assert duet["voices"][1] == {"id": "duo", "gender": "female", "descriptors": ["airy"]}
    out = pm.transform_spec({"style": "{lofi} beat {nope}"}, [])
    assert out["style"] == "lo-fi, tape saturation, vinyl crackle beat {nope}"


def test_lua_sandbox_removes_io(tmp_path):
    pytest.importorskip("lupa")
    (tmp_path / "evil.lua").write_text("return { transform_spec = function(s) io.open('x','w') return s end }")
    (tmp_path / "osexec.lua").write_text("os.execute('true') return {}")
    pm = PluginManager()
    pm.load_dirs([tmp_path])
    assert next(p for p in pm.plugins if p.name == "osexec").error  # os.execute is gone
    warnings = []
    pm.transform_spec({"style": "a"}, warnings)
    assert warnings  # io is nil -> hook error becomes a warning


def test_lua_empty_tables_keep_their_shape(tmp_path):
    pytest.importorskip("lupa")
    (tmp_path / "noop.lua").write_text("return { transform_spec = function(s) return s end }")
    pm = PluginManager()
    pm.load_dirs([tmp_path])
    spec = {"voices": [{"id": "lead", "descriptors": []}], "audio_sampling": {}, "tags": ["a"]}
    assert pm.transform_spec(spec, []) == spec
