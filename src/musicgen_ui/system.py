"""GPU / runtime status for the UI header."""
from __future__ import annotations

import importlib.util
import sys


def gpu_status() -> dict:
    # Only import torch if it is installed; it is already loaded once YuE2 runs.
    if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
        return {"torch": False, "cuda": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - broken installs should not break the UI
        return {"torch": False, "cuda": False, "error": str(exc)}
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if info["cuda"]:
        free, total = torch.cuda.mem_get_info()
        info.update(
            name=torch.cuda.get_device_name(0),
            capability=".".join(map(str, torch.cuda.get_device_capability(0))),
            total_gib=round(total / 2**30, 2),
            used_gib=round((total - free) / 2**30, 2),
            process_reserved_gib=round(torch.cuda.memory_reserved(0) / 2**30, 2),
        )
    return info
