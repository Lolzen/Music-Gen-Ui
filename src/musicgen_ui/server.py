"""FastAPI app: JSON API under /api, static single-page UI at /."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import BACKENDS, DEVICES, ENGINES, IDLE_PLACEMENTS, QUANTIZATIONS, Settings
from .jobs import JobQueue
from .models import YUE2_DEFAULTS, SongSpec
from .plugins import PluginManager
from .prompting import KNOWN_SECTIONS
from .runtime import Busy, EngineManager
from .memory import ram_status
from .system import gpu_status

WEB_DIR = Path(__file__).parent / "web"
log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, engine=None) -> FastAPI:
    settings = settings or Settings.load()
    plugins = PluginManager()
    plugins.load_dirs(settings.plugin_dirs)
    manager = EngineManager(settings, engine)
    queue = JobQueue(manager, settings.library_dir, plugins)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        queue.start()
        yield
        queue.stop()
        manager.shutdown()

    app = FastAPI(title="Music-Gen-UI", version="0.2.0", lifespan=lifespan)
    app.state.queue, app.state.manager, app.state.plugins = queue, manager, plugins

    def job_or_404(job_id: str):
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(404, "Job nicht gefunden")
        return job

    def guarded(action):
        try:
            return action()
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # -- info / engine ---------------------------------------------------------
    @app.get("/api/info")
    def info():
        return {"engine": manager.status(), "defaults": YUE2_DEFAULTS,
                "sections": KNOWN_SECTIONS, "presets": plugins.presets,
                "plugins": plugins.summary(), "spec_schema": SongSpec.model_json_schema()}

    @app.get("/api/status")
    def status():
        running = queue.running
        return {"engine": manager.status(), "gpu": gpu_status(), "ram": ram_status(),
                "running_job": running.id if running else None}

    @app.get("/api/settings")
    def get_settings():
        return {"values": settings.editable(),
                "choices": {"engine": ENGINES, "quantization": QUANTIZATIONS,
                            "backend": BACKENDS, "device": DEVICES, "idle_placement": IDLE_PLACEMENTS,
                            "vae": ["m-a-p/YuE2-Vae", "m-a-p/YuE2-Vae-legacy"]}}

    @app.put("/api/settings")
    def put_settings(values: dict = Body(...)):
        guarded(lambda: manager.configure(values))
        return {"values": settings.editable(), "engine": manager.status()}

    @app.post("/api/engine/load")
    def load():
        guarded(manager.load_async)
        return manager.status()

    @app.post("/api/engine/unload")
    def unload():
        guarded(manager.unload)
        return manager.status()

    # -- generation --------------------------------------------------------------
    @app.post("/api/preview")
    def preview(spec: SongSpec):
        return guarded(lambda: queue.preview(spec))

    @app.post("/api/jobs")
    def submit(spec: SongSpec):
        return guarded(lambda: [job.public() for job in queue.submit(spec)])

    @app.get("/api/jobs")
    def jobs():
        return [job.public() for job in queue.all()]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return job_or_404(job_id).public()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job_or_404(job_id)
        return queue.cancel(job_id).public()

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        job_or_404(job_id)
        if not queue.delete(job_id):
            raise HTTPException(409, "Laufende Jobs zuerst abbrechen")
        return {"deleted": job_id}

    @app.get("/api/jobs/{job_id}/audio")
    def audio(job_id: str):
        path = queue.artifact(job_or_404(job_id).id, "audio.flac")
        if path is None:
            raise HTTPException(404, "Noch kein Audio")
        return FileResponse(path, media_type="audio/flac", filename=f"{job_id}.flac")

    @app.get("/api/jobs/{job_id}/score")
    def score(job_id: str):
        path = queue.artifact(job_or_404(job_id).id, "score.abc")
        if path is None:
            raise HTTPException(404, "Keine Partitur")
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
