import json
import time

import pytest
from fastapi.testclient import TestClient

from musicgen_ui import memory, runtime
from musicgen_ui.config import Settings
from musicgen_ui.engine.mock import MockEngine
from musicgen_ui.server import create_app


# ---------------------------------------------------------------- settings
def test_offload_settings_and_alias(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"low_vram": True}))  # old name
    s = Settings.load(tmp_path, env={})
    assert s.offload is True and s.low_vram is True
    assert Settings.load(tmp_path, env={"MGU_LOW_VRAM": "0"}).offload is False
    assert Settings.load(tmp_path, env={"MGU_OFFLOAD": "1", "MGU_IDLE": "unload"}).idle_placement == "unload"
    with pytest.raises(ValueError):
        Settings().update({"offload": True, "idle_placement": "gpu"})
    with pytest.raises(ValueError):
        Settings().update({"offload": True, "backend": "vllm"})
    with pytest.raises(ValueError):
        Settings().update({"idle_placement": "swap"})
    with pytest.raises(ValueError):
        Settings().update({"ram_guard_gib": -1})


def test_ram_status_reads_proc_meminfo():
    status = memory.ram_status()
    if status is None:
        pytest.skip("no /proc/meminfo")
    assert status["total_gib"] > 0 and 0 <= status["available_gib"] <= status["total_gib"]


def test_ram_guard_trips():
    values = iter([10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
    guard = memory.RamGuard(2.0, interval=0.01, reader=lambda: next(values, 1.0))
    with guard:
        for _ in range(100):
            if guard.tripped:
                break
            time.sleep(0.01)
    assert guard.tripped and guard.lowest_gib == 1.0 and "RAM-Notbremse" in guard.message
    assert memory.RamGuard(0, reader=lambda: 1.0).enabled is False


# ---------------------------------------------------------------- manager behaviour
class TrackingMock(MockEngine):
    def __init__(self, delay=0.0, need=0.0):
        super().__init__(delay)
        self.need, self.placements, self.unloads = need, [], 0

    def ram_needed_gib(self):
        return self.need

    def after_job(self, placement):
        self.placements.append(placement)

    def unload(self):
        self.unloads += 1


def wait_job(c, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


SPEC = {"lyrics": "[Verse]\nla la\n[Chorus]\noh", "seed": 1}


def make(tmp_path, engine, **settings):
    return TestClient(create_app(Settings(engine="mock", data_dir=tmp_path, plugin_dirs=[], **settings),
                                 engine=engine))


def test_idle_placement_after_each_song(tmp_path):
    engine = TrackingMock()
    with make(tmp_path, engine, idle_placement="unload") as c:
        job = c.post("/api/jobs", json=SPEC).json()[0]
        assert wait_job(c, job["id"])["status"] == "done"
    assert engine.placements == ["unload"]


def test_ram_precheck_refuses_song(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "available_gib", lambda: 6.0)
    engine = TrackingMock(need=7.5)
    with make(tmp_path, engine, ram_guard_gib=2.0) as c:
        job = wait_job(c, c.post("/api/jobs", json=SPEC).json()[0]["id"])
    assert job["status"] == "failed" and "Zu wenig freier Arbeitsspeicher" in job["error"]
    assert engine.placements == []  # never started


def test_ram_guard_stops_running_song_and_unloads(tmp_path, monkeypatch):
    readings = {"free": 20.0}
    monkeypatch.setattr(runtime, "available_gib", lambda: readings["free"])
    monkeypatch.setattr(memory, "available_gib", lambda: readings["free"])
    engine = TrackingMock(delay=3.0)
    with make(tmp_path, engine, ram_guard_gib=2.0) as c:
        job_id = c.post("/api/jobs", json=SPEC).json()[0]["id"]
        for _ in range(100):
            if c.get(f"/api/jobs/{job_id}").json()["status"] == "running":
                break
            time.sleep(0.02)
        readings["free"] = 0.5  # RAM runs out mid-song
        job = wait_job(c, job_id)
    assert job["status"] == "failed" and "RAM-Notbremse" in job["error"]
    assert engine.unloads >= 1


def test_status_reports_ram(tmp_path):
    with make(tmp_path, TrackingMock()) as c:
        status = c.get("/api/status").json()
        assert "ram" in status
        values = c.get("/api/settings").json()["values"]
        assert values["offload"] is False and values["idle_placement"] == "ram"
        assert c.put("/api/settings", json={"offload": True, "vram_reserve_gib": 4}).status_code == 200
        assert c.put("/api/settings", json={"idle_placement": "gpu"}).status_code == 422  # conflicts with offload


# ---------------------------------------------------------------- YuE2 adapter specifics
def test_yue2_budget_from_vram_reserve(monkeypatch):
    from musicgen_ui.engine import yue2
    monkeypatch.setattr(yue2, "_cuda_total_gib", lambda: 23.99)
    engine = yue2.YuE2Engine(Settings(vram_reserve_gib=3.0))
    assert engine._memory_budget() == 22.99  # YuE2 subtracts 2 GiB → 20.99 GiB cap = total − 3
    engine = yue2.YuE2Engine(Settings(vram_reserve_gib=0.5))
    assert engine._memory_budget() == 23.99  # never below YuE2's own 2 GiB reserve
    engine = yue2.YuE2Engine(Settings(vram_reserve_gib=3.0, memory_budget_gib=16))
    assert engine._memory_budget() == 16


class _Param:
    def __init__(self, device):
        self.device = type("D", (), {"type": device})()


class _Module:
    def __init__(self, device="cuda"):
        self.device, self.moves = device, []

    def to(self, target):
        self.device = str(target)
        self.moves.append(str(target))
        return self

    def parameters(self):
        return iter([_Param(self.device)])


def test_yue2_after_job_placements(monkeypatch):
    from musicgen_ui.engine import yue2
    engine = yue2.YuE2Engine(Settings())
    pipe = type("P", (), {})()
    pipe._model, pipe._vae, pipe.device, pipe.closed = _Module("cuda"), _Module("cuda"), "cuda", False
    pipe.close = lambda: setattr(pipe, "closed", True)
    engine._pipe = pipe

    assert engine.ram_needed_gib() == yue2.MODEL_RAM_FALLBACK_GIB  # model on GPU → decode will park it in RAM
    engine.after_job("ram")
    assert pipe._model.device == "cpu" and pipe._vae.device == "cpu"
    assert engine.ram_needed_gib() == 0.0
    engine.after_job("gpu")
    assert pipe._model.device == "cuda" and pipe._vae.device == "cpu"
    engine.after_job("unload")
    assert pipe.closed and not engine.loaded
