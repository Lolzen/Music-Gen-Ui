"""GPU stress test that mimics YuE2's sound synthesis (sudden BF16 tensor-core load).

    python -m musicgen_ui.stress --mode instant --seconds 60
    python -m musicgen_ui.stress --mode ramp --seconds 90

Separates hardware/power problems from software: if this also freezes the
system, YuE2 and Music-Gen-UI are not the cause. Every second power, temperature,
clock and PCIe state are printed and appended (fsync'ed) to
``data/logs/stress-<date>.log`` so the last values survive a forced restart.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .memory import gpu_telemetry


def _log(path: Path, text: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GPU-Stresstest (wie YuE2-Klangsynthese)")
    parser.add_argument("--mode", choices=("instant", "ramp"), default="instant",
                        help="instant: sofort Volllast (wie YuE2); ramp: Last in 10 Stufen steigern")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--size", type=int, default=8192, help="Matrixgröße (8192 ≈ YuE2-artige Last)")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Keine CUDA-GPU gefunden")
    log = Path(args.data_dir) / "logs" / f"stress-{time.strftime('%Y-%m-%d')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _log(log, f"Start: Modus {args.mode}, {args.seconds} s, Matrix {args.size}, {torch.cuda.get_device_name(0)}")
    _log(log, f"Leerlauf: {gpu_telemetry() or 'keine Telemetrie (nvidia-smi fehlt?)'}")

    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = last = time.monotonic()
    done = 0
    try:
        while (now := time.monotonic()) - start < args.seconds:
            if args.mode == "ramp":
                level = min(1.0, 0.1 + (now - start) / args.seconds * 1.2)  # 10 % … 100 %
                busy = 0.1 * level
                t0 = time.monotonic()
                while time.monotonic() - t0 < busy:
                    for _ in range(4):
                        a @ b
                    torch.cuda.synchronize()
                    done += 4
                time.sleep(0.1 * (1 - level))
            else:
                for _ in range(8):
                    a @ b
                torch.cuda.synchronize()
                done += 8
            if now - last >= 1.0:
                last = now
                tflops = done * 2 * args.size ** 3 / (now - start) / 1e12
                _log(log, f"t={now - start:5.1f}s  {tflops:6.1f} TFLOPS  {gpu_telemetry() or ''}")
    except KeyboardInterrupt:
        _log(log, "Abgebrochen (Strg+C)")
        return 1
    _log(log, f"Fertig ohne Absturz: {gpu_telemetry() or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
