"""FastAPI app: JSON API under /api, static single-page UI at /."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import json

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import catalog
from .config import BACKENDS, DEVICES, ENGINES, GGUF_MODES, GPU_SHARE_LEVELS, IDLE_PLACEMENTS, QUANTIZATIONS, Settings
from .jobs import JobQueue
from .models import YUE2_DEFAULTS, SongSpec
from .plugins import PluginManager
from .prompting import KNOWN_SECTIONS
from .runtime import Busy, EngineManager
from . import predict as predict_mod
from .memory import latest_log, process_rss_gib, process_vram_gib, ram_status
from .system import gpu_status
from .training.config import COMPANION_NAR, HEADS, TOKENIZER_REPO, TrainConfig
from .training.project import ProjectStore, slug
from .training.runner import TrainingRunner

WEB_DIR = Path(__file__).parent / "web"
log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, engine=None) -> FastAPI:
    settings = settings or Settings.load()
    plugins = PluginManager()
    plugins.load_dirs(settings.plugin_dirs)
    catalog.ensure_dirs(settings.models_dir)
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
                "process": {"rss_gib": process_rss_gib(), "vram_gib": process_vram_gib(),
                            "cuda_active": manager.engine.loaded and manager.engine.name == "yue2"},
                "running_job": running.id if running else None}

    @app.get("/api/settings")
    def get_settings():
        return {"values": settings.editable(),
                "choices": {"engine": ENGINES, "quantization": QUANTIZATIONS,
                            "backend": BACKENDS, "device": DEVICES, "idle_placement": IDLE_PLACEMENTS, "gguf_mode": GGUF_MODES, "gpu_share": GPU_SHARE_LEVELS,
                            "vae": ["m-a-p/YuE2-Vae", "m-a-p/YuE2-Vae-legacy"]}}

    @app.put("/api/settings")
    def put_settings(values: dict = Body(...)):
        guarded(lambda: manager.configure(values))
        return {"values": settings.editable(), "engine": manager.status()}

    @app.get("/api/models")
    def models():
        return {**catalog.scan(settings.models_dir),
                "selected": {"checkpoint": settings.model, "vae": settings.vae},
                "gguf_mode": settings.gguf_mode}

    @app.post("/api/predict")
    def predict_memory(body: dict = Body(default={})):
        """Memory prediction for the current settings, optionally with unsaved UI values."""
        try:
            candidate = predict_mod.with_overrides(settings, body.get("settings"))
            result = predict_mod.predict(candidate, body.get("spec") or {},
                                         manager.profiles.get(predict_mod.profile_key(candidate)))
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        result["engine"] = candidate.resolved_engine()
        gpu = gpu_status()
        result["system"] = predict_mod.system_view(
            result, gpu, ram_status(), process_rss_gib(), process_vram_gib(),
            cuda_active=manager.engine.loaded and manager.engine.name == "yue2")
        if gpu.get("cuda"):  # YuE2 caps PyTorch at min(budget − 2, total − reserve)
            result["vram"]["cap_gib"] = round(min(candidate.memory_budget_gib - 2,
                                                  gpu["total_gib"] - max(candidate.vram_reserve_gib, 2.0)), 2)
        return result

    # -- LoRA training ----------------------------------------------------------
    projects = ProjectStore(settings.data_dir / "training")
    trainer = TrainingRunner(projects, manager, settings)
    app.state.trainer = trainer
    lora_dir = settings.models_dir / "lora"

    def project_or_404(project_id: str):
        try:
            return projects.load(project_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(404, "Projekt nicht gefunden") from exc

    def not_while_training(project_id: str):
        if trainer.active == project_id:
            raise HTTPException(409, "Während des Trainings nicht möglich")

    def project_view(project):
        status = projects.status(project.id)
        return {"project": project.to_dict(), "status": status, "active": trainer.active == project.id,
                "checkpoints": projects.checkpoints(project.id), "log": projects.log_tail(project.id),
                "problems": projects.validate(project),
                "companion": projects.find_companion(lora_dir, project.head)}

    @app.get("/api/training")
    def training_list():
        return {"active": trainer.active, "heads": list(HEADS), "projects": [
            {**p.to_dict(), "status": projects.status(p.id).get("phase", "idle"), "songs": len(p.songs)}
            for p in projects.list()]}

    @app.post("/api/training")
    def training_create(body: dict = Body(...)):
        return project_view(guarded(lambda: projects.create(body.get("name", ""), body.get("trigger", ""),
                                                            body.get("description", ""))))

    @app.get("/api/training/{project_id}")
    def training_get(project_id: str):
        return project_view(project_or_404(project_id))

    @app.put("/api/training/{project_id}")
    def training_update(project_id: str, body: dict = Body(...)):
        not_while_training(project_id)
        with projects.lock(project_id):
            project = project_or_404(project_id)
            update_project(project, body)
            projects.save(project)
        return project_view(project)

    def update_project(project, body: dict) -> None:
        for key in ("name", "description", "use_regularizer", "consent"):
            if key in body:
                setattr(project, key, body[key])
        if body.get("head") in HEADS:
            project.head = body["head"]
        if isinstance(body.get("settings"), dict):
            allowed = TrainConfig().to_dict()
            for key, value in body["settings"].items():
                if key in allowed and isinstance(value, (int, float)) and value >= 0:
                    project.settings[key] = type(allowed[key])(value)
        if isinstance(body.get("songs"), list):
            by_file = {s.file: s for s in project.songs}
            for update in body["songs"]:
                song = by_file.get(update.get("file"))
                if song:
                    for key in ("title", "lyrics", "description"):
                        if isinstance(update.get(key), str):
                            setattr(song, key, update[key][:20000])

    @app.delete("/api/training/{project_id}")
    def training_delete(project_id: str):
        project_or_404(project_id)
        not_while_training(project_id)
        import shutil
        shutil.rmtree(projects.dir(project_id), ignore_errors=True)
        return {"deleted": project_id}

    @app.post("/api/training/{project_id}/songs")
    async def training_upload(project_id: str, request: Request, filename: str):
        project = project_or_404(project_id)
        not_while_training(project_id)
        try:
            target = projects.song_target(project, filename)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        size, limit = 0, 600 * 2**20
        with open(target, "wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, "Datei zu groß (max. 600 MB)")
                fh.write(chunk)
        projects.register_song(project, target, filename)
        return project_view(projects.load(project_id))

    @app.delete("/api/training/{project_id}/songs/{file}")
    def training_remove_song(project_id: str, file: str):
        project = project_or_404(project_id)
        not_while_training(project_id)
        projects.remove_song(project, file)
        return project_view(projects.load(project_id))

    @app.post("/api/training/{project_id}/start")
    def training_start(project_id: str, body: dict = Body(default={})):
        project_or_404(project_id)
        guarded(lambda: trainer.start(project_id, fake=bool(body.get("fake")), steps=int(body.get("steps") or 0)))
        return project_view(projects.load(project_id))

    @app.post("/api/training/{project_id}/stop")
    def training_stop(project_id: str):
        trainer.stop(project_id)
        return project_view(project_or_404(project_id))

    @app.post("/api/training/{project_id}/export")
    def training_export(project_id: str, body: dict = Body(...)):
        project = project_or_404(project_id)
        source = Path(body.get("file", ""))
        if source.parent.resolve() != (projects.dir(project_id) / "checkpoints").resolve() or not source.is_file():
            raise HTTPException(422, "Checkpoint nicht gefunden")
        target_dir = lora_dir / project.id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{slug(project.name)}-{source.stem}.safetensors"
        import shutil
        shutil.copyfile(source, target)
        sidecar = {"name": f"{project.name} ({source.stem})", "trigger": project.trigger, "kind": "artist",
                   "recommended": {"plan_mode": "off"}, "companion": projects.find_companion(lora_dir, project.head),
                   "head": project.head, "trained_on": len(project.songs)}
        target.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"file": str(target.resolve()), "sidecar": sidecar}

    @app.post("/api/training/companion")
    def training_companion(body: dict = Body(default={})):
        version = body.get("version", "v9")
        if version not in COMPANION_NAR:
            raise HTTPException(422, "Unbekannte Version")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(TOKENIZER_REPO, COMPANION_NAR[version], local_dir=str(lora_dir / "mothersuperior"))
        except Exception as exc:  # noqa: BLE001 - network/hub errors go to the UI
            raise HTTPException(502, f"Download fehlgeschlagen: {exc}") from exc
        return {"file": str(Path(path).resolve())}

    @app.get("/api/logs/memory")
    def memory_log():
        name, text = latest_log(settings.data_dir / "logs")
        header = f"# {name}\n" if name else "# Noch kein Speicher-Protokoll vorhanden\n"
        return PlainTextResponse(header + text)

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
