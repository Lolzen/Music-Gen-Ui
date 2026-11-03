import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from musicgen_ui.config import Settings
from musicgen_ui.engine.base import Cancelled
from musicgen_ui.models import SongSpec
from musicgen_ui.prompting import compile_spec
from musicgen_ui.server import create_app


# ---------------------------------------------------------------- settings
def test_settings_precedence(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"low_vram": True, "quantization": "fp8", "engine": "mock"}))
    s = Settings.load(tmp_path, env={"MGU_QUANTIZATION": "none", "MGU_PORT": "9000"})
    assert s.low_vram is True          # from file
    assert s.quantization == "none"    # env beats file
    assert s.port == 9000 and s.engine == "mock"


def test_settings_validation():
    s = Settings()
    with pytest.raises(ValueError):
        s.update({"quantization": "int4"})
    with pytest.raises(ValueError):
        s.update({"backend": "vllm", "low_vram": True})
    s.update({"memory_budget_gib": "20", "unknown": 1})
    assert s.memory_budget_gib == 20.0


def test_invalid_settings_file_is_ignored(tmp_path):
    (tmp_path / "settings.json").write_text("{broken")
    assert Settings.load(tmp_path, env={}).engine == "auto"


# ---------------------------------------------------------------- API
def make_client(tmp_path, delay=0.0):
    return TestClient(create_app(Settings(engine="mock", data_dir=tmp_path, mock_delay=delay, plugin_dirs=[])))


def test_settings_api_persists_and_switches_engine(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/api/settings").json()["values"]["engine"] == "mock"
        assert c.put("/api/settings", json={"quantization": "bogus"}).status_code == 422
        res = c.put("/api/settings", json={"engine": "yue2", "quantization": "fp8"}).json()
        assert res["engine"]["engine"] == "yue2" and res["engine"]["state"] == "unloaded"
        assert json.loads((tmp_path / "settings.json").read_text())["quantization"] == "fp8"
        status = c.get("/api/status").json()
        assert status["engine"]["info"]["quantization"] == "fp8"
        assert "gpu" in status
        if not status["engine"]["installed"]["yue2"]:
            c.post("/api/engine/load")
            for _ in range(50):
                engine = c.get("/api/status").json()["engine"]
                if engine["state"] != "loading":
                    break
                time.sleep(0.05)
            assert engine["state"] == "error" and "install.sh" in engine["message"]


def test_settings_blocked_while_generating(tmp_path):
    with make_client(tmp_path, delay=5) as c:
        c.get("/api/status")  # warm-up: the first call may import torch (slow)
        job = c.post("/api/jobs", json={"lyrics": "[Verse]\nla", "seed": 1}).json()[0]
        for _ in range(50):
            if c.get(f"/api/jobs/{job['id']}").json()["status"] == "running":
                break
            time.sleep(0.02)
        for _ in range(250):  # the job turns "running" just before the engine reports "busy"
            if c.get("/api/status").json()["engine"]["state"] == "busy":
                break
            time.sleep(0.02)
        assert c.get("/api/status").json()["engine"]["state"] == "busy"
        assert c.put("/api/settings", json={"low_vram": True}).status_code == 409
        c.post(f"/api/jobs/{job['id']}/cancel")


# ---------------------------------------------------------------- YuE2 adapter against a fake package
class FakeSong:
    sample_rate = 48000

    def save_artifacts(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "audio.flac").write_bytes(b"fLaC")
        (directory / "score.abc").write_text("X:1\nK:C\n")
        return {"status": "complete", "truncated": False, "audio_seconds": 12.5,
                "timing": {"e2e_seconds": 3.0}, "weights": {"mot": "x"}}


class _FakeBase:
    instances = []

    def _record(self, kwargs):
        self.init_kwargs, self.calls, self.closed = kwargs, [], False
        _FakeBase.instances.append(self)

    @classmethod
    def from_pretrained(cls, model="m-a-p/YuE2-3B", *, vae="m-a-p/YuE2-Vae", revision=None,
                        vae_revision=None, local_files_only=False, token=None, cache_dir=None,
                        progress=True, **kwargs):
        return cls(model, vae, progress=progress, **kwargs)

    def _run(self, kwargs):
        self.calls.append(kwargs)
        if kwargs["cancelled"]():
            raise InterruptedError("Cancelled during semantic")
        for _ in range(50):
            kwargs["on_token"]("semantic", 1)
        if "on_progress" in kwargs:
            kwargs["on_progress"]("nar", 32, 32)
        return FakeSong()

    def close(self):
        self.closed = True


class FakePipelineMain(_FakeBase):
    """Signatures of yue2 main branch (low_vram, on_progress)."""

    def __init__(self, model_dir, vae_dir, *, device="auto", memory_budget_gib=24,
                 backend="torch", generation_config=None, verify_hashes=True,
                 vae_core_frames=None, quantization="none", offload_ar=False, low_vram=False, progress=True):
        self._record(dict(model=model_dir, device=device, memory_budget_gib=memory_budget_gib, backend=backend,
                          quantization=quantization, offload_ar=offload_ar, low_vram=low_vram, progress=progress,
                          vae_core_frames=vae_core_frames))

    def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
                 semantic_sampling=None, cancelled=None, on_token=None, on_progress=None, **kwargs):
        return self._run(dict(kwargs, style=style, lyrics=lyrics, semantic_sampling=semantic_sampling,
                              cancelled=cancelled, on_token=on_token, on_progress=on_progress))


class FakePipelineRelease(_FakeBase):
    """Signatures of the yue2-v0.1.6 release tag (offload_ar, no low_vram/on_progress)."""

    def __init__(self, model_dir, vae_dir, *, device="auto", memory_budget_gib=24,
                 backend="torch", generation_config=None, verify_hashes=True,
                 vae_core_frames=None, quantization="none", offload_ar=False, progress=True):
        self._record(dict(model=model_dir, device=device, memory_budget_gib=memory_budget_gib, backend=backend,
                          quantization=quantization, offload_ar=offload_ar, progress=progress,
                          vae_core_frames=vae_core_frames))

    def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
                 semantic_sampling=None, cancelled=None, on_token=None, **kwargs):
        # Like the release: unknown kwargs end up in the request and fail there.
        unknown = set(kwargs) - {"id", "cot", "seed", "abc", "cfg_scale"}
        if unknown:
            raise TypeError(f"SongRequest.__init__() got an unexpected keyword argument {unknown.pop()!r}")
        return self._run(dict(kwargs, style=style, lyrics=lyrics, semantic_sampling=semantic_sampling,
                              cancelled=cancelled, on_token=on_token))


FakePipeline = FakePipelineMain


def install_fake(monkeypatch, cls):
    module = types.ModuleType("yue2")
    module.YuE2Pipeline = cls
    monkeypatch.setitem(sys.modules, "yue2", module)
    _FakeBase.instances.clear()
    return module


@pytest.fixture
def fake_yue2(monkeypatch):
    return install_fake(monkeypatch, FakePipelineMain)


def test_yue2_engine_contract(fake_yue2, tmp_path):
    from musicgen_ui.engine.yue2 import YuE2Engine
    settings = Settings(engine="yue2", quantization="fp8", offload=True, memory_budget_gib=22)
    engine = YuE2Engine(settings)
    spec = SongSpec(style="pop", lyrics="[Verse]\nhi", seed=3, cfg_scale=1.5, audio_sampling={"temperature": 1.1})
    events = []
    result = engine.generate(compile_spec(spec), tmp_path / "out",
                             progress=lambda *e: events.append(e), cancelled=lambda: False)
    pipe = _FakeBase.instances[0]
    assert pipe.init_kwargs["model"] == "m-a-p/YuE2-3B"
    assert pipe.init_kwargs["quantization"] == "fp8" and pipe.init_kwargs["low_vram"] is True
    assert pipe.init_kwargs["memory_budget_gib"] == 22 and pipe.init_kwargs["progress"] is False
    call = pipe.calls[0]
    assert call["seed"] == 3 and call["cfg_scale"] == 1.5
    assert call["semantic_sampling"] == {"temperature": 1.1}
    assert ("audio", 25, 0) in events and ("audio", 50, 0) in events and ("nar", 32, 32) in events
    assert result.duration_s == 12.5 and result.score_abc.startswith("X:1")

    with pytest.raises(Cancelled):
        engine.generate(compile_spec(spec), tmp_path / "out2", progress=lambda *e: None, cancelled=lambda: True)

    engine.unload()
    assert pipe.closed and not engine.loaded


def test_yue2_end_to_end_through_api(fake_yue2, tmp_path):
    settings = Settings(engine="yue2", data_dir=tmp_path, plugin_dirs=[])
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/status").json()["engine"]["state"] == "unloaded"
        job = c.post("/api/jobs", json={"lyrics": "[Verse]\nla", "seed": 1}).json()[0]
        for _ in range(100):
            job = c.get(f"/api/jobs/{job['id']}").json()
            if job["status"] == "done":
                break
            time.sleep(0.02)
        assert job["status"] == "done" and job["result"]["duration_s"] == 12.5
        assert c.get("/api/status").json()["engine"]["state"] == "ready"
        assert c.post("/api/engine/unload").json()["state"] == "unloaded"
        profiles = json.loads((tmp_path / "stats" / "memory-profiles.json").read_text())
        assert any(v.get("ram_peak_gib") for v in profiles.values())  # peak measured and stored
        assert _FakeBase.instances[0].closed


def test_yue2_release_api_compat(monkeypatch, tmp_path):
    """yue2-v0.1.6 release: low_vram → offload_ar, no on_progress (the reported TypeError)."""
    install_fake(monkeypatch, FakePipelineRelease)
    from musicgen_ui.engine.yue2 import YuE2Engine
    engine = YuE2Engine(Settings(engine="yue2", offload=True))
    events = []
    result = engine.generate(compile_spec(SongSpec(lyrics="[Verse]\nhi", seed=1)), tmp_path / "out",
                             progress=lambda *e: events.append(e), cancelled=lambda: False)
    pipe = _FakeBase.instances[0]
    assert pipe.init_kwargs["offload_ar"] is True and pipe.init_kwargs["vae_core_frames"] == 512
    assert "on_progress" not in pipe.calls[0]
    assert ("audio", 50, 0) in events and result.duration_s == 12.5
    assert "synthesize" not in engine._pipe.__dict__  # temporary phase markers removed again
    info = engine.info()
    assert info["low_vram_mode"] == "offload_ar" and info["yue2"]["stage_progress"] is False


def test_yue2_main_api_uses_low_vram(monkeypatch, tmp_path):
    install_fake(monkeypatch, FakePipelineMain)
    from musicgen_ui.engine.yue2 import YuE2Engine
    engine = YuE2Engine(Settings(engine="yue2", offload=True))
    engine.load(lambda *e: None)
    pipe = _FakeBase.instances[0]
    assert pipe.init_kwargs["low_vram"] is True and pipe.init_kwargs["offload_ar"] is False
    assert pipe.init_kwargs["vae_core_frames"] == 512
    assert engine.info()["low_vram_mode"] == "low_vram"
