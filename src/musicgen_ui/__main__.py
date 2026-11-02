"""``python -m musicgen_ui`` / ``music-gen-ui``: start the web server."""
from __future__ import annotations

import argparse
import logging

import uvicorn

from .config import Settings
from .server import create_app


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Music-Gen-UI web server")
    parser.add_argument("--engine", choices=["mock", "yue2"], default=settings.engine)
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--low-vram", action="store_true", default=settings.low_vram)
    args = parser.parse_args()
    settings.engine, settings.host, settings.port = args.engine, args.host, args.port
    settings.low_vram = args.low_vram
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
