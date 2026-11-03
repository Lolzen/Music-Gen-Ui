"""``python -m musicgen_ui`` / ``music-gen-ui``: start the web server."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .config import ENGINES, Settings
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Music-Gen-UI web server")
    parser.add_argument("--data-dir", help="Ordner für Songs und Einstellungen (Standard: ./data)")
    parser.add_argument("--models-dir", help="Modell-Ordner (Standard: ./models)")
    parser.add_argument("--engine", choices=ENGINES, help="auto = YuE2 falls installiert, sonst Mock")
    parser.add_argument("--host", help="z.B. 0.0.0.0 für Zugriff aus dem Netzwerk")
    parser.add_argument("--port", type=int)
    parser.add_argument("--offload", "--low-vram", dest="offload", action="store_true", default=None,
                        help="Offload: ungenutzte Modellteile in den RAM auslagern (weniger VRAM, langsamer)")
    parser.add_argument("--fp8", action="store_true", default=None, help="experimentelles FP8 (ab RTX 40xx)")
    parser.add_argument("--preload", action="store_true", help="Modell direkt beim Start laden")
    args = parser.parse_args()

    settings = Settings.load(args.data_dir)
    overrides = {"engine": args.engine, "offload": args.offload,
                 "quantization": "fp8" if args.fp8 else None}
    settings.update({k: v for k, v in overrides.items() if v is not None})
    if args.models_dir:
        settings.models_dir = Path(args.models_dir)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = create_app(settings)
    if args.preload:
        app.state.manager.load_async()
    print(f"\n  Music-Gen-UI: http://{settings.host}:{settings.port}  (Engine: {settings.resolved_engine()})\n")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
