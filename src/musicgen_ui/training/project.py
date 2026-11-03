"""Training projects on disk: ``<data>/training/<id>/``.

    project.json          name, trigger, description, settings, songs (file, title, lyrics, description)
    songs/                uploaded audio
    cache/<song>/         mert.npy (features), codes.npy (semantic tokens)
    checkpoints/          step-N / best / last .safetensors (Mothersuperior LoRA layout)
    status.json           written by the worker (phase, step, losses, message)
    train.log             worker output
    STOP                  created to ask the worker to stop
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import TrainConfig

AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".aiff", ".aif"}
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(text: str, fallback: str = "projekt") -> str:
    value = _SAFE.sub("-", text.strip()).strip("-.")[:60]
    return value or fallback


@dataclass
class Song:
    file: str
    title: str
    lyrics: str = ""
    description: str = ""


@dataclass
class Project:
    id: str
    name: str
    trigger: str
    description: str = ""          # default sound description for every song
    head: str = "v9"               # tokenizer head / companion NAR LoRA version
    use_regularizer: bool = True
    consent: bool = False          # rights to the recordings / consent of the performers
    settings: dict = field(default_factory=lambda: TrainConfig().to_dict())
    songs: list[Song] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def lock(self, project_id: str) -> threading.Lock:
        """Serialises read-modify-write of one project (the web server handles requests in parallel)."""
        with self._guard:
            return self._locks.setdefault(project_id, threading.Lock())

    def dir(self, project_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", project_id):
            raise ValueError("ungültige Projekt-ID")
        return self.root / project_id

    def list(self) -> list[Project]:
        projects = []
        for file in sorted(self.root.glob("*/project.json")):
            try:
                projects.append(self.load(file.parent.name))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(projects, key=lambda p: p.created, reverse=True)

    def load(self, project_id: str) -> Project:
        data = json.loads((self.dir(project_id) / "project.json").read_text(encoding="utf-8"))
        data["songs"] = [Song(**s) for s in data.get("songs", [])]
        settings = TrainConfig().to_dict()
        settings.update({k: v for k, v in data.get("settings", {}).items() if k in settings})
        data["settings"] = settings
        return Project(**data)

    def save(self, project: Project) -> None:
        directory = self.dir(project.id)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f"project.json.{secrets.token_hex(4)}.tmp"
        tmp.write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(directory / "project.json")

    def create(self, name: str, trigger: str, description: str = "") -> Project:
        trigger = trigger.strip()
        if not re.fullmatch(r"[\w-]{2,40}", trigger):
            raise ValueError("Trigger-Wort: 2–40 Buchstaben/Ziffern, ohne Leerzeichen (z.B. 'xyzq')")
        project = Project(id=f"{slug(name)}-{secrets.token_hex(2)}", name=name.strip() or trigger, trigger=trigger,
                          description=description.strip())
        self.save(project)
        return project

    def song_target(self, project: Project, filename: str) -> Path:
        suffix = Path(filename).suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            raise ValueError(f"Format {suffix or '?'} nicht unterstützt ({', '.join(sorted(AUDIO_SUFFIXES))})")
        stem = slug(Path(filename).stem, "song")
        target = self.dir(project.id) / "songs" / f"{stem}{suffix}"
        n = 2
        while target.exists():
            target = target.with_name(f"{stem}-{n}{suffix}")
            n += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def register_song(self, project: Project, target: Path, filename: str) -> Song:
        song = Song(file=target.name, title=Path(filename).stem)
        with self.lock(project.id):
            fresh = self.load(project.id)
            fresh.songs.append(song)
            self.save(fresh)
        return song

    def add_song(self, project: Project, filename: str, chunks) -> Song:
        target = self.song_target(project, filename)
        with open(target, "wb") as fh:
            for chunk in chunks:
                fh.write(chunk)
        return self.register_song(project, target, filename)

    def find_companion(self, lora_dir: Path, version: str) -> str | None:
        """Matching sound (NAR) LoRA in models/lora; the plain file is preferred over *_comfyui."""
        found = sorted(Path(lora_dir).rglob(f"nar_lora_joint_{version}*.safetensors"),
                       key=lambda f: ("comfyui" in f.name, f.name))
        return str(found[0].resolve()) if found else None

    def remove_song(self, project: Project, file: str) -> None:
        with self.lock(project.id):
            project = self.load(project.id)
            project.songs = [s for s in project.songs if s.file != file]
            self.save(project)
        (self.dir(project.id) / "songs" / file).unlink(missing_ok=True)
        cache = self.dir(project.id) / "cache" / Path(file).stem
        if cache.is_dir():
            for item in cache.iterdir():
                item.unlink()
            cache.rmdir()

    def status(self, project_id: str) -> dict:
        try:
            return json.loads((self.dir(project_id) / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"phase": "idle"}

    def checkpoints(self, project_id: str) -> list[dict]:
        folder = self.dir(project_id) / "checkpoints"
        found = []
        for file in sorted(folder.glob("*.safetensors")):
            match = re.fullmatch(r"step-(\d+)", file.stem)
            found.append({"name": file.stem, "file": str(file.resolve()), "step": int(match[1]) if match else None,
                          "size_mb": round(file.stat().st_size / 2**20, 1), "modified": file.stat().st_mtime})
        return sorted(found, key=lambda c: (c["step"] is None, c["step"] or 0))

    def log_tail(self, project_id: str, lines: int = 60) -> str:
        try:
            text = (self.dir(project_id) / "train.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def validate(self, project: Project) -> list[str]:
        problems = []
        if not project.consent:
            problems.append("Bestätigung fehlt: Rechte an den Aufnahmen bzw. Einwilligung der Künstler")
        if len(project.songs) < 3:
            problems.append("Mindestens 3 Songs nötig (empfohlen: 10–30)")
        missing = [s.title for s in project.songs if not s.lyrics.strip()]
        if missing:
            problems.append(f"Vollständige Lyrics fehlen bei: {', '.join(missing[:5])}"
                            + (" …" if len(missing) > 5 else "") + " (für Instrumentals '[instrumental]' eintragen)")
        return problems
