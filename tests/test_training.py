import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")
pytest.importorskip("yue2")
pytest.importorskip("scipy")

from musicgen_ui.config import Settings  # noqa: E402
from musicgen_ui.engine.mock import MockEngine  # noqa: E402
from musicgen_ui.server import create_app  # noqa: E402
from musicgen_ui.training import pipeline as P  # noqa: E402


def wav_bytes(seconds=2.0, rate=44100, freq=220.0):
    import soundfile as sf
    t = np.arange(int(seconds * rate)) / rate
    audio = np.stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * 1.5 * t)], 1).astype(np.float32) * 0.2
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    settings = Settings(engine="mock", data_dir=tmp_path / "data", plugin_dirs=[], models_dir=tmp_path / "models")
    with TestClient(create_app(settings, engine=MockEngine(0))) as c:
        yield c


def wait_training(c, pid, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        view = c.get(f"/api/training/{pid}").json()
        if not view["active"] and view["status"].get("phase") not in ("start", "prep", "tokenize", "dataset", "load", "train"):
            return view
        time.sleep(0.3)
    raise AssertionError("training did not finish")


def make_project(c, songs=3):
    view = c.post("/api/training", json={"name": "Mein Artist", "trigger": "xyzq", "description": "dark synth pop"}).json()
    pid = view["project"]["id"]
    for i in range(songs):
        view = c.post(f"/api/training/{pid}/songs", params={"filename": f"song {i}.wav"},
                      content=wav_bytes(freq=200 + 40 * i)).json()
    return pid, view


def test_project_validation_and_upload(client):
    assert client.post("/api/training", json={"name": "x", "trigger": "has space"}).status_code == 422
    pid, view = make_project(client)
    assert [s["file"] for s in view["project"]["songs"]] == ["song-0.wav", "song-1.wav", "song-2.wav"]
    assert any("Rechte" in p for p in view["problems"]) and any("Lyrics" in p for p in view["problems"])
    assert client.post(f"/api/training/{pid}/start", json={"fake": True}).status_code == 422
    bad = client.post(f"/api/training/{pid}/songs", params={"filename": "evil.exe"}, content=b"x")
    assert bad.status_code == 422


def test_training_run_checkpoints_and_export(client, tmp_path):
    pid, view = make_project(client)
    songs = [{"file": s["file"], "lyrics": "[Verse]\nla la la\n[Chorus]\noh"} for s in view["project"]["songs"]]
    settings = {"steps": 4, "checkpoint_from": 2, "checkpoint_every": 2, "eval_every": 2, "rank": 4, "max_len": 600}
    view = client.put(f"/api/training/{pid}", json={"consent": True, "songs": songs, "settings": settings}).json()
    assert view["problems"] == []
    assert client.post(f"/api/training/{pid}/start", json={"fake": True}).status_code == 200
    assert client.get("/api/status").json()["engine"]["state"] == "training"  # generation is blocked
    assert client.post(f"/api/training/{pid}/start", json={"fake": True}).status_code == 409
    view = wait_training(client, pid)
    assert view["status"]["phase"] == "done", view["log"]
    names = {c["name"] for c in view["checkpoints"]}
    assert {"step-2", "step-4", "last"} <= names
    assert view["status"]["evals"][0]["step"] == 0 and "minted_val" in view["status"]["evals"][0]
    assert client.get("/api/status").json()["engine"]["state"] != "training"  # GPU released again
    checkpoint = next(c for c in view["checkpoints"] if c["name"] == "step-4")
    exported = client.post(f"/api/training/{pid}/export", json={"file": checkpoint["file"]}).json()
    assert exported["sidecar"]["trigger"] == "xyzq" and exported["sidecar"]["recommended"]["plan_mode"] == "off"
    from musicgen_ui.lora import read_adapter
    adapter = read_adapter(exported["file"], {"q": 64, "kv": 32, "ff": 128})
    assert adapter.format == "mothersuperior" and adapter.halves == ["ar"] and adapter.rank == 4
    assert adapter.meta["trigger"] == "xyzq"
    lora = client.get("/api/models").json()["lora"]
    assert any(e["value"] == exported["file"] and e["meta"]["halves"] == ["ar"] for e in lora)
    # caches are reused on a second run
    cache = tmp_path / "data" / "training" / pid / "cache"
    assert len(list(cache.glob("*/mert.npy"))) == 3 and len(list(cache.glob("*/codes-v9.npy"))) == 3


def test_training_stop(client):
    pid, view = make_project(client)
    songs = [{"file": s["file"], "lyrics": "[Verse]\nla"} for s in view["project"]["songs"]]
    client.put(f"/api/training/{pid}", json={"consent": True, "songs": songs,
                                             "settings": {"steps": 5000, "rank": 4, "max_len": 600, "eval_every": 5000}})
    client.post(f"/api/training/{pid}/start", json={"fake": True})
    end = time.time() + 120
    while client.get(f"/api/training/{pid}").json()["status"].get("phase") != "train" and time.time() < end:
        time.sleep(0.3)
    client.post(f"/api/training/{pid}/stop")
    view = wait_training(client, pid)
    assert view["status"]["phase"] == "stopped"
    assert any(c["name"] == "last" for c in view["checkpoints"])


def test_predict_codes_covers_long_input():
    head = P.TokenizerHead().eval()
    codes = P.predict_codes(head, np.random.default_rng(0).standard_normal((1300, 1024)).astype(np.float16), "cpu")
    assert codes.shape == (1300,) and codes.min() >= 0 and codes.max() < P.VOCAB


def test_safe_torch_load(tmp_path):
    good = tmp_path / "pack.pt"
    torch.save([{"name": "a", "src": "minted", "style": "pop", "lyrics": "x",
                 "codec": np.arange(5, dtype=np.int32)}], good)
    data = P.safe_torch_load(good)
    assert data[0]["codec"].tolist() == [0, 1, 2, 3, 4]

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    bad = tmp_path / "evil.pt"
    torch.save({"x": Evil()}, bad)
    with pytest.raises(Exception):
        P.safe_torch_load(bad)


def test_parallel_updates_do_not_corrupt_project(client):
    from concurrent.futures import ThreadPoolExecutor
    pid, view = make_project(client)
    files = [s["file"] for s in view["project"]["songs"]]

    def put(i):
        return client.put(f"/api/training/{pid}", json={"songs": [{"file": files[i % 3], "lyrics": f"[Verse]\nzeile {i}"}],
                                                        "settings": {"steps": 100 + i}}).status_code
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(put, range(40)))
    assert set(codes) == {200}
    project = client.get(f"/api/training/{pid}").json()["project"]
    assert all(s["lyrics"].startswith("[Verse]") for s in project["songs"])  # every song kept its update
