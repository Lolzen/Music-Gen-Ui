import json
import logging

import pytest
from fastapi.testclient import TestClient

from musicgen_ui import predict
from musicgen_ui.__main__ import QuietPolling
from musicgen_ui.config import Settings
from musicgen_ui.engine.mock import MockEngine
from musicgen_ui.server import create_app


def test_hf_weights_match_yue2_3b():
    groups = predict.model_groups("m-a-p/YuE2-3B")
    params = sum(g["elements"] for g in groups.values())
    assert 3.55e9 < params < 3.65e9  # "approximately 3.58B parameters"
    assert groups["ar"]["elements"] == groups["nar"]["elements"]


def test_modes_and_options_change_prediction():
    base = predict.predict(Settings())
    assert base["vram"]["basis"] == "geschätzt" and base["assumptions"]["cuda_graphs"]
    assert 8 < base["vram"]["peak_gib"] < 24           # plausible range; real runs replace the estimate
    assert 9 < base["ram"]["peak_gib"] < 13
    offload = predict.predict(Settings(offload=True))
    assert offload["vram"]["phases"]["Klangsynthese"] < base["vram"]["phases"]["Klangsynthese"]
    fp8 = predict.predict(Settings(quantization="fp8"))
    assert fp8["vram"]["phases"]["Komponieren"] < base["vram"]["phases"]["Komponieren"]
    cfg = predict.predict(Settings(), {"cfg_scale": 2.0})
    assert cfg["assumptions"]["cfg_branches"] == 2 and cfg["vram"]["phases"]["Komponieren"] > base["vram"]["phases"]["Komponieren"]
    short = predict.predict(Settings(), {"audio_max_tokens": 1500})
    assert short["vram"]["peak_gib"] < base["vram"]["peak_gib"]
    unload = predict.predict(Settings(idle_placement="unload"))
    assert unload["ram"]["idle_gib"] < base["ram"]["idle_gib"]


def test_gguf_groups_and_modes(tmp_path):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    from gguf import GGMLQuantizationType as Q
    path = tmp_path / "yue2-lm-q8_0.gguf"
    w = gguf.GGUFWriter(str(path), "yue2")
    rng = np.random.default_rng(0)
    q8 = lambda shape: gguf.quants.quantize(rng.standard_normal(shape).astype(np.float32), Q.Q8_0)  # noqa: E731
    w.add_tensor("token_embd.weight", q8((512, 256)), raw_dtype=Q.Q8_0)
    w.add_tensor("output.weight", q8((512, 256)), raw_dtype=Q.Q8_0)
    w.add_tensor("blk.0.attn_q.weight", q8((256, 256)), raw_dtype=Q.Q8_0)
    w.add_tensor("blk.0.nar_ffn_up.weight", q8((1024, 256)), raw_dtype=Q.Q8_0)
    w.add_tensor("output_norm.weight", rng.standard_normal(256).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    groups = predict.model_groups(str(path))
    assert groups["nar"]["elements"] == 1024 * 256 and groups["nar"]["bytes"] == 1024 * 256 // 32 * 34
    assert groups["embed"]["elements"] == 512 * 256 and groups["extra"]["elements"] == 256
    weights = {m: predict.predict(Settings(model=str(path), gguf_mode=m))["weights_bytes"] for m in ("bf16", "balanced", "max")}
    assert weights["max"] < weights["balanced"] < weights["bf16"]
    assert not predict.predict(Settings(model=str(path), gguf_mode="max"))["assumptions"]["cuda_graphs"]


def test_measured_profile_replaces_estimate(tmp_path):
    store = predict.ProfileStore(tmp_path / "p.json")
    key = predict.profile_key(Settings())
    store.record(key, {"vram_peak_gib": 17.0, "ram_peak_gib": 9.0}, 120)
    store.record(key, {"vram_peak_gib": 15.0, "ram_peak_gib": 9.5}, 60)  # keeps the maximum
    entry = store.get(key)
    assert entry["vram_peak_gib"] == 17.0 and entry["ram_peak_gib"] == 9.5 and entry["song_seconds"] == 120
    result = predict.predict(Settings(), measured=entry)
    assert result["vram"]["basis"] == "gemessen" and result["vram"]["peak_gib"] == 17.5  # + CUDA context
    assert result["vram"]["estimate_gib"] > 0
    assert predict.profile_key(Settings(offload=True)) != key


def test_system_view_adds_other_programs():
    pred = {"vram": {"peak_gib": 18.0}, "ram": {"peak_gib": 10.0, "idle_gib": 9.0}}
    view = predict.system_view(pred, {"cuda": True, "total_gib": 24.0, "used_gib": 9.5},
                               {"total_gib": 32.0, "used_gib": 14.0}, process_rss_gib=10.0,
                               process_vram_gib=7.0, cuda_active=True)
    assert view["vram"]["predicted_gib"] == 20.0  # 9.5 used − (7 + 0.5 ours) + 18
    assert view["ram"]["predicted_gib"] == 14.0   # 14 used − 10 ours + 10


def test_predict_api(tmp_path):
    settings = Settings(engine="mock", data_dir=tmp_path, plugin_dirs=[], models_dir=tmp_path / "models")
    with TestClient(create_app(settings, engine=MockEngine(0))) as c:
        r = c.post("/api/predict", json={"spec": {"cfg_scale": 2}}).json()
        assert r["engine"] == "mock" and "vram" in r and "system" in r
        assert r["assumptions"]["cfg_branches"] == 2
        candidate = c.post("/api/predict", json={"settings": {"offload": True}}).json()
        assert candidate["assumptions"]["offload"] and not settings.offload  # nothing saved
        assert c.post("/api/predict", json={"settings": {"gguf_mode": "nope"}}).status_code == 422
        assert "process" in c.get("/api/status").json()
    from musicgen_ui import server
    fake_gpu = {"cuda": True, "total_gib": 24.0, "used_gib": 2.0}
    original = server.gpu_status
    server.gpu_status = lambda: fake_gpu
    try:
        with TestClient(create_app(settings, engine=MockEngine(0))) as c:
            r = c.post("/api/predict", json={"settings": {"vram_reserve_gib": 4}}).json()
            assert r["vram"]["cap_gib"] == 20.0 and r["system"]["vram"]["total_gib"] == 24.0
    finally:
        server.gpu_status = original


def test_quiet_polling_filter():
    f = QuietPolling()
    record = lambda method, path, status: logging.LogRecord(  # noqa: E731
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:1", method, path, "1.1", status), None)
    assert not f.filter(record("GET", "/api/jobs", 200))
    assert not f.filter(record("GET", "/api/status", 200))
    assert not f.filter(record("POST", "/api/predict", 200))
    assert f.filter(record("POST", "/api/jobs", 200))      # starting a song stays visible
    assert f.filter(record("GET", "/api/status", 500))     # errors stay visible
    assert f.filter(record("GET", "/api/jobs/abc/audio", 200))


# ---------------------------------------------------------------- calibration
from musicgen_ui import calibrate  # noqa: E402


def test_factors_and_running_mean(tmp_path):
    breakdown = {"Komponieren": {"weights_gib": 6.8, "overhead_gib": 3.0},
                 "Klangsynthese": {"weights_gib": 6.8, "overhead_gib": 2.0},
                 "Dekodieren": {"weights_gib": 0.27, "overhead_gib": 0.01}}
    assert calibrate.factors_from(breakdown, {"Komponieren": 8.3, "Klangsynthese": 30.0, "Dekodieren": 2.0}) == \
        {"Komponieren": 0.5, "Klangsynthese": 5.0}  # clamped; tiny overhead ignored
    path = tmp_path / "cal.json"
    calibrate.update(path, breakdown, {"Komponieren": 8.3}, ram_peak_gib=12.0, ram_model_gib=6.8)
    data = calibrate.update(path, breakdown, {"Komponieren": 9.8})
    assert data["factors"]["Komponieren"] == 0.75 and data["counts"]["Komponieren"] == 2
    assert data["ram_base_gib"] == round(12.0 - 6.8 - 0.27 - 0.3, 2) and data["samples"] == 2


def test_prediction_uses_calibration():
    plain = predict.predict(Settings())
    cal = predict.predict(Settings(), calibration={"factors": {"Komponieren": 0.2, "Klangsynthese": 2.0},
                                                   "ram_base_gib": 1.5, "samples": 3})
    b = plain["breakdown"]["Komponieren"]
    assert cal["vram"]["basis"] == "kalibriert" and cal["ram"]["basis"] == "kalibriert"
    assert cal["vram"]["phases"]["Komponieren"] == round(b["weights_gib"] + 0.2 * b["overhead_gib"] + 0.5, 2)
    assert cal["vram"]["phases"]["Klangsynthese"] > plain["vram"]["phases"]["Klangsynthese"]
    assert cal["ram"]["peak_gib"] < plain["ram"]["peak_gib"]


USER_LOG = """2026-11-03 16:30:00  job=Free-Palestine-c9187a  phase=start     Song startet (Engine yue2, Offload aus, zwischen Songs: ram, VRAM-Reserve 3.0 GB, RAM-Notbremse 2.0 GB)  |  RAM frei 26.1/31.2 GB
2026-11-03 16:30:10  job=Free-Palestine-c9187a  phase=audio     RAM frei 26.1/31.2 GB  Swap 0.0 GB  VRAM 8.2/22.1 GB (PyTorch reserviert 7.4)  GPU 115/500 W
2026-11-03 16:31:30  job=Free-Palestine-c9187a  phase=nar       RAM frei 26.1/31.2 GB  Swap 0.0 GB  VRAM 10.3/22.1 GB (PyTorch reserviert 9.5)  GPU 177/500 W
2026-11-03 16:32:10  job=Free-Palestine-c9187a  phase=vae       RAM frei 24.0/31.2 GB  Swap 0.0 GB  VRAM 3.1/22.1 GB (PyTorch reserviert 2.2)
2026-11-03 16:32:24  job=Free-Palestine-c9187a  phase=fertig    Song fertig (81 s Musik), Spitzen:   |  RAM frei 24.0/31.2 GB
2026-11-03 16:40:00  job=crashed-1  phase=nar       RAM frei 26.1/31.2 GB  VRAM 10.3/22.1 GB (PyTorch reserviert 9.5)
"""


def test_calibrate_cli_from_user_log(tmp_path, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "memory-2026-11-03.log").write_text(USER_LOG)
    songs = calibrate.songs_from_logs(logs)
    assert [s["job"] for s in songs] == ["Free-Palestine-c9187a"]  # unfinished song ignored
    assert songs[0]["phases"] == {"Komponieren": 7.4, "Klangsynthese": 9.5, "Dekodieren": 2.2}
    assert songs[0]["offload"] is False and songs[0]["seconds"] == 81
    assert calibrate.main(["--data-dir", str(tmp_path)]) == 0
    report = capsys.readouterr().out
    assert "Komponieren" in report and "gemessen   7.4 GB" in report and "1 neue Song" in report
    data = calibrate.load(tmp_path / "stats" / "calibration.json")
    assert data["samples"] == 1 and data["factors"]["Komponieren"] < 1  # composition overhead was overestimated
    calibrate.main(["--data-dir", str(tmp_path)])  # second run must not count the song twice
    assert calibrate.load(tmp_path / "stats" / "calibration.json")["samples"] == 1
    calibrate.main(["--data-dir", str(tmp_path), "--reset"])
    assert not (tmp_path / "stats" / "calibration.json").exists()


def test_song_records_phase_peaks_and_calibrates(tmp_path, monkeypatch):
    """Through the real runtime path (fake yue2): per-phase peaks stored, RAM base calibrated."""
    import sys
    import types
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_settings_engine import FakePipelineRelease, install_fake
    install_fake(monkeypatch, FakePipelineRelease)
    import time
    settings = Settings(engine="yue2", data_dir=tmp_path, plugin_dirs=[], models_dir=tmp_path / "models")
    with TestClient(create_app(settings)) as c:
        job = c.post("/api/jobs", json={"lyrics": "[Verse]\nla", "seed": 1}).json()[0]
        for _ in range(250):
            if c.get(f"/api/jobs/{job['id']}").json()["status"] not in ("queued", "running"):
                break
            time.sleep(0.02)
        profile = next(iter(json.loads((tmp_path / "stats" / "memory-profiles.json").read_text()).values()))
        assert "Komponieren" in profile["phases"] and "ram_peak_gib" in profile["phases"]["Komponieren"]
        cal = calibrate.load(tmp_path / "stats" / "calibration.json")
        assert cal["samples"] == 1 and "ram_base_gib" in cal
        r = c.post("/api/predict", json={}).json()
        assert r["ram"]["basis"] in ("kalibriert", "gemessen")
