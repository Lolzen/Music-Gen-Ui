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
