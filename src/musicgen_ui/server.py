"""FastAPI app: JSON API under /api, static single-page UI at /."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .engine import create_engine
from .jobs import JobQueue
from .models import YUE2_DEFAULTS, SongSpec
from .plugins import PluginManager
from .prompting import KNOWN_SECTIONS

WEB_DIR = Path(__file__).parent / "web"
log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, engine=None) -> FastAPI:
    settings = settings or Settings.from_env()
    plugins = PluginManager()
    plugins.load_dirs(settings.plugin_dirs)
    queue = JobQueue(engine or create_engine(settings), settings.library_dir, plugins)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        queue.start()
        yield
        queue.stop()

    app = FastAPI(title="Music-Gen-UI", version="0.1.0", lifespan=lifespan)
    app.state.queue, app.state.settings, app.state.plugins = queue, settings, plugins

    def job_or_404(job_id: str):
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(404, "Job nicht gefunden")
        return job

    @app.get("/api/info")
    def info():
        return {"engine": queue.engine.info(), "defaults": YUE2_DEFAULTS,
                "sections": KNOWN_SECTIONS, "presets": plugins.presets,
                "plugins": plugins.summary(), "spec_schema": SongSpec.model_json_schema()}

    @app.post("/api/preview")
    def preview(spec: SongSpec):
        try:
            return queue.preview(spec)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/jobs")
    def submit(spec: SongSpec):
        try:
            return [job.public() for job in queue.submit(spec)]
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

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
