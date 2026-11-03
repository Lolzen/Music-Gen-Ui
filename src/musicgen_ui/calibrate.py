"""Calibration of the memory prediction from measured songs.

Each phase is predicted as *exact weights + estimated overhead*. After a real
song the measured peak per phase gives ``factor = (measured - weights) / overhead``;
factors are averaged over songs and stored in ``<data>/stats/calibration.json``.
The RAM base (Python + PyTorch + CUDA libraries) is calibrated the same way.

    python -m musicgen_ui.calibrate            report + calibrate from logs/profiles
    python -m musicgen_ui.calibrate --reset    forget the calibration
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

PHASE_OF_STAGE = {"score": "Komponieren", "audio": "Komponieren", "nar": "Klangsynthese", "vae": "Dekodieren"}
FACTOR_RANGE = (0.2, 5.0)
MIN_OVERHEAD_GIB = 0.05


def load(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def factors_from(breakdown: dict, measured_phases: dict) -> dict:
    """Overhead factor per phase from one song (measured in GiB, PyTorch-reserved)."""
    factors = {}
    for phase, gib in measured_phases.items():
        part = breakdown.get(phase)
        if not part or part["overhead_gib"] < MIN_OVERHEAD_GIB:
            continue
        factor = (gib - part["weights_gib"]) / part["overhead_gib"]
        factors[phase] = round(min(max(factor, FACTOR_RANGE[0]), FACTOR_RANGE[1]), 3)
    return factors


def update(path: Path, breakdown: dict, measured_phases: dict, ram_peak_gib: float | None = None,
           ram_model_gib: float | None = None, source: str = "song") -> dict:
    """Fold one measured song into the stored calibration (running mean per phase)."""
    data = load(path)
    counts = data.setdefault("counts", {})
    factors = data.setdefault("factors", {})
    for phase, factor in factors_from(breakdown, measured_phases).items():
        n = counts.get(phase, 0)
        factors[phase] = round((factors.get(phase, factor) * n + factor) / (n + 1), 3)
        counts[phase] = n + 1
    if ram_peak_gib and ram_model_gib is not None:
        from .predict import VAE_GIB
        base = min(max(ram_peak_gib - ram_model_gib - VAE_GIB - 0.3, 0.5), 10.0)
        n = counts.get("ram", 0)
        data["ram_base_gib"] = round((data.get("ram_base_gib", base) * n + base) / (n + 1), 2)
        counts["ram"] = n + 1
    data["samples"] = data.get("samples", 0) + 1
    data["updated"] = time.time()
    data.setdefault("sources", []).append(source)
    data["sources"] = data["sources"][-20:]
    save(path, data)
    return data


# ---------------------------------------------------------------- memory-log parsing (older songs)
_LINE = re.compile(r"^(\S+ \S+)\s+job=(\S+)\s+phase=(\S+)\s+(.*)$")
_RESERVED = re.compile(r"PyTorch reserviert ([\d.]+)")
_DONE = re.compile(r"Song fertig \((\d+) s Musik\)")
_OFFLOAD = re.compile(r"Offload (an|aus)")


def songs_from_logs(log_dir: Path) -> list[dict]:
    """Per finished song: max PyTorch-reserved VRAM per phase (1-second samples)."""
    songs: dict[str, dict] = {}
    for file in sorted(Path(log_dir).glob("memory-*.log")):
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _LINE.match(line)
            if not match:
                continue
            _, job, stage, rest = match.groups()
            song = songs.setdefault(job, {"job": job, "phases": {}, "seconds": None, "offload": None})
            if (m := _OFFLOAD.search(rest)) and song["offload"] is None:
                song["offload"] = m[1] == "an"
            if m := _DONE.search(rest):
                song["seconds"] = int(m[1])
            phase = PHASE_OF_STAGE.get(stage)
            if phase and (m := _RESERVED.search(rest)):
                song["phases"][phase] = max(song["phases"].get(phase, 0.0), float(m[1]))
    return [s for s in songs.values() if s["seconds"] and s["phases"]]


def main(argv=None) -> int:
    from .config import Settings
    from dataclasses import replace
    from .predict import predict, profile_key
    parser = argparse.ArgumentParser(description="Speicher-Prognose kalibrieren")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reset", action="store_true", help="Kalibrierung löschen")
    args = parser.parse_args(argv)
    settings = Settings.load(args.data_dir)
    stats = settings.data_dir / "stats"
    cal_path = stats / "calibration.json"
    if args.reset:
        cal_path.unlink(missing_ok=True)
        print("Kalibrierung gelöscht.")
        return 0

    profiles = load(stats / "memory-profiles.json")
    current = profiles.get(profile_key(settings), {})
    already = set(load(cal_path).get("sources", []))
    print("== Music-Gen-UI Kalibrierung ==")
    print(f"Modell {settings.model} | Offload {'an' if settings.offload else 'aus'} | "
          f"FP8 {'an' if settings.quantization == 'fp8' else 'aus'} | Backend {settings.backend}")
    used = 0
    for song in songs_from_logs(settings.data_dir / "logs"):
        s = replace(settings)
        if song["offload"] is not None:
            s.offload = song["offload"]
        hints = {"audio_tokens_actual": song["seconds"] * 25, "plan_mode": "full"}
        estimate = predict(s, hints)
        if song["job"] not in already:
            update(cal_path, estimate["breakdown"], song["phases"], source=song["job"])
            used += 1
        print(f"\nSong {song['job']} ({song['seconds']} s Musik, Offload {'an' if s.offload else 'aus'}):")
        for phase, gib in song["phases"].items():
            part = estimate["breakdown"][phase]
            est = part["weights_gib"] + part["overhead_gib"]
            print(f"  {phase:14s} gemessen {gib:5.1f} GB  geschätzt {est:5.1f} GB "
                  f"(Gewichte {part['weights_gib']:.1f} + Rest {part['overhead_gib']:.1f})")
    if current:
        print(f"\nGemessene Spitzen dieser Konfiguration: VRAM {current.get('vram_peak_gib', '?')} GB, "
              f"RAM {current.get('ram_peak_gib', '?')} GB (längster Song {current.get('song_seconds', '?')} s)")
        for phase, values in (current.get("phases") or {}).items():
            print(f"  {phase:14s} VRAM {values.get('vram_peak_gib', '?')} GB  RAM {values.get('ram_peak_gib', '?')} GB")
    cal = load(cal_path)
    print(f"\nKalibrierung: {used} neue Song(s) aus dem Protokoll übernommen, insgesamt {cal.get('samples', 0)}.")
    print("Faktoren für den geschätzten Rest (1.0 = Schätzung passte):",
          json.dumps(cal.get("factors", {}), ensure_ascii=False), "| RAM-Basis:", cal.get("ram_base_gib", "–"))
    full = predict(settings, {}, None, cal)
    print(f"Prognose für den längsten Song (6 min): VRAM ~{full['vram']['peak_gib']} GB "
          f"({', '.join(f'{k} {v}' for k, v in full['vram']['phases'].items())}), RAM ~{full['ram']['peak_gib']} GB")
    print("\nHinweis: Werte aus dem Protokoll sind Sekunden-Stichproben; ab jetzt misst jeder Song exakte Spitzen je Phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
