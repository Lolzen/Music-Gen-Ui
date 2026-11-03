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
        for _ in range(1500):  # wait until the song really generates (first status call may import torch)
            if c.get(f"/api/jobs/{job_id}").json()["progress"]["stage"]:
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


def test_memory_log_survives_and_is_served(tmp_path):
    engine = TrackingMock(delay=0.3)
    with make(tmp_path, engine) as c:
        job = wait_job(c, c.post("/api/jobs", json=SPEC).json()[0]["id"])
        assert job["status"] == "done"
        text = c.get("/api/logs/memory").text
    logs = list((tmp_path / "logs").glob("memory-*.log"))
    assert len(logs) == 1
    assert "Song startet" in text and "Song fertig" in text and "Phase beginnt" in text
    assert f"job={job['id']}" in text


def test_memory_log_sampling_and_prune(tmp_path):
    for day in range(1, 10):
        (tmp_path / f"memory-2026-01-0{day}.log").write_text("old\n")
    ram = {"available_gib": 5.0, "total_gib": 32.0, "used_gib": 27.0, "swap_used_gib": 1.5}
    gpu = {"cuda": True, "used_gib": 20.0, "total_gib": 24.0, "process_reserved_gib": 18.0}
    with memory.MemoryLog(tmp_path, "song-1", interval=0.02, keep=3,
                          ram_reader=lambda: ram, gpu_reader=lambda: gpu) as mlog:
        mlog.stage = "audio"
        time.sleep(0.15)
    text = mlog.path.read_text()
    assert "phase=audio" in text and "RAM frei 5.0/32.0 GB" in text and "VRAM 20.0/24.0 GB" in text
    assert len(list(tmp_path.glob("memory-*.log"))) <= 4  # 3 kept + today's


def test_interrupted_job_reports_last_phase(tmp_path):
    from musicgen_ui.jobs import JobQueue
    from musicgen_ui.runtime import EngineManager
    settings = Settings(engine="mock", data_dir=tmp_path, plugin_dirs=[])
    lib = tmp_path / "library" / "song-1"
    lib.mkdir(parents=True)
    (lib / "job.json").write_text(json.dumps({"id": "song-1", "group": "song-1", "title": "x", "spec": {},
                                              "request": {}, "status": "running"}))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "memory-2026-11-03.log").write_text(
        "2026-11-03 15:48:23  job=song-1  phase=audio     RAM frei 23.7/31.2 GB\n"
        "2026-11-03 15:48:25  job=song-1  phase=nar       RAM frei 21.6/31.2 GB  GPU 438/450 W  71 °C\n"
        "2026-11-03 15:49:00  job=other  phase=vae       RAM frei 1.0/31.2 GB\n")
    queue = JobQueue(EngineManager(settings, MockEngine(0)), tmp_path / "library")
    error = queue.get("song-1").error
    assert "Phase „nar“ um 15:48:25" in error and "438/450 W" in error


def test_telemetry_line_in_memory_log(tmp_path):
    with memory.MemoryLog(tmp_path, "s", interval=0.02, ram_reader=lambda: None, gpu_reader=lambda: None,
                          telemetry_reader=lambda: "GPU 312/450 W  61 °C") as mlog:
        time.sleep(0.08)
    assert "GPU 312/450 W" in mlog.path.read_text()


def test_parse_pcie_replays():
    sample = """        Replays Since Reset                            : 7
        Replay Number Rollovers                        : 0"""
    assert memory.parse_replays(sample) == 7
    assert memory.parse_replays("no pcie section") is None


def test_stress_refuses_without_cuda():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("real GPU present")
    from musicgen_ui import stress
    with pytest.raises(SystemExit, match="CUDA"):
        stress.main(["--seconds", "1"])
