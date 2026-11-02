import time

import pytest
from fastapi.testclient import TestClient

from musicgen_ui.config import Settings
from musicgen_ui.server import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(engine="mock", data_dir=tmp_path, mock_delay=0, plugin_dirs=[])
    with TestClient(create_app(settings)) as c:
        yield c


def wait(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


SPEC = {"title": "Test", "style": "pop", "lyrics": "[Verse]\nhello world\n[Chorus]\noh oh", "seed": 7}


def test_info_and_static(client):
    info = client.get("/api/info").json()
    assert info["engine"]["engine"] == "mock"
    assert "chorus" in info["sections"]
    assert client.get("/").status_code == 200


def test_preview(client):
    req = client.post("/api/preview", json=SPEC).json()
    assert req["style"] == "English, pop, female vocal"
    assert req["lyrics"].startswith("[Verse]\nhello world")


def test_generate_candidates_and_artifacts(client):
    jobs = client.post("/api/jobs", json={**SPEC, "candidates": 2}).json()
    assert [j["request"]["seed"] for j in jobs] == [7, 8]
    assert jobs[0]["group"] == jobs[1]["group"]
    for job in jobs:
        done = wait(client, job["id"])
        assert done["status"] == "done", done
        assert done["result"]["has_score"]
        audio = client.get(f"/api/jobs/{job['id']}/audio")
        assert audio.status_code == 200 and audio.content[:4] == b"fLaC"
        assert client.get(f"/api/jobs/{job['id']}/score").text.startswith("X:1")
    assert len(client.get("/api/jobs").json()) == 2
    assert client.delete(f"/api/jobs/{jobs[0]['id']}").status_code == 200
    assert len(client.get("/api/jobs").json()) == 1


def test_library_survives_restart(tmp_path):
    settings = Settings(engine="mock", data_dir=tmp_path, mock_delay=0, plugin_dirs=[])
    with TestClient(create_app(settings)) as c:
        job_id = c.post("/api/jobs", json=SPEC).json()[0]["id"]
        wait(c, job_id)
    with TestClient(create_app(settings)) as c:
        assert c.get(f"/api/jobs/{job_id}").json()["status"] == "done"


def test_validation_errors(client):
    assert client.post("/api/jobs", json={**SPEC, "cfg_scale": 99}).status_code == 422
    r = client.post("/api/jobs", json={**SPEC, "abc": "X:1\nK:C\nC|", "plan_mode": "off"})
    assert r.status_code == 422


def test_cancel_running(tmp_path):
    settings = Settings(engine="mock", data_dir=tmp_path, mock_delay=2, plugin_dirs=[])
    with TestClient(create_app(settings)) as c:
        first, second = (c.post("/api/jobs", json=SPEC).json()[0] for _ in range(2))
        assert c.post(f"/api/jobs/{second['id']}/cancel").json()["status"] == "cancelled"
        time.sleep(0.3)
        c.post(f"/api/jobs/{first['id']}/cancel")
        assert wait(c, first["id"])["status"] == "cancelled"
