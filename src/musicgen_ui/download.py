"""Model download helpers, shared by the UI (with progress) and the CLI.

CLI:  ``music-gen-ui-download [--model ID] [--vae ID]``
Files go to the normal Hugging Face cache (``HF_HOME``), exactly where
``YuE2Pipeline.from_pretrained`` looks for them.
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Callable


def is_hub_id(model: str) -> bool:
    path = Path(model).expanduser()
    return not path.is_dir() and not path.is_absolute() and not model.startswith(".")


def cache_bytes(repo_ids: list[str]) -> int:
    """Bytes already present (complete or partial) in the HF cache for these repos."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return 0
    total = 0
    for repo in repo_ids:
        blobs = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "blobs"
        if blobs.is_dir():
            total += sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())
    return total


class DownloadMonitor:
    """Reports the growing cache size every second while a download runs."""

    def __init__(self, repo_ids: list[str], report: Callable[[int], None], interval: float = 1.0):
        self.repo_ids = [r for r in repo_ids if is_hub_id(r)]
        self.report, self.interval = report, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.report(cache_bytes(self.repo_ids))

    def __enter__(self):
        if self.repo_ids:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def download(model: str, vae: str) -> tuple[Path, Path]:
    from yue2.storage import resolve_model  # same file selection as the pipeline
    return resolve_model(model), resolve_model(vae)


def main() -> None:
    from .config import Settings
    settings = Settings.load()
    parser = argparse.ArgumentParser(description="YuE2-Modelle vorab herunterladen")
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--vae", default=settings.vae)
    args = parser.parse_args()
    try:
        model, vae = download(args.model, args.vae)
    except ImportError:
        raise SystemExit("Paket 'yue2' fehlt – zuerst ./install.sh ausführen")
    print(f"Modell: {model}\nVAE:    {vae}")


if __name__ == "__main__":
    main()
