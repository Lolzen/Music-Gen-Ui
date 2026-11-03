"""Memory prediction for a model/settings combination (shown on the VRAM/RAM bars).

Weights are exact: GGUF from the tensor directory, Hugging-Face checkpoints from
their config. Caches and activations are estimates derived from the YuE2 code
(KV cache size, CUDA-graph weight copies, NAR context, VAE tiles). After every
real song the measured peaks are stored per configuration and then replace the
estimate ("gemessen" instead of "geschätzt").
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .catalog import model_format, read_gguf_tensors
from .config import Settings

GIB = 2**30
DEFAULT_ARCH = dict(hidden=2048, layers=28, heads=16, kv_heads=8, head_dim=128, ff=6144, vocab=184704,
                    latent=64, frames=24576, context=24576)
CUDA_CONTEXT_GIB = 0.5        # CUDA context + library workspaces outside PyTorch's allocator
PROCESS_BASE_GIB = 2.5        # Python + PyTorch/CUDA libraries in host RAM
VAE_GIB = 0.27                # YuE2-Vae decoder, FP32
VAE_ACT_GIB_PER_1K = 1.6      # decode activations per 1024-frame tile (estimate)
NAR_TRANSIENT_GIB = 1.2       # hidden states / MLP intermediates over the NAR context (estimate)
PREFIX_TOKENS = {"off": 400, "melody": 3000, "full": 3500}  # style+lyrics(+score) prompt length
GROUPS = ("embed", "head", "ar", "nar", "extra")


# ---------------------------------------------------------------- weights per group
def _classify(name: str) -> str:
    if name.startswith("token_embd"):
        return "embed"
    if name.startswith("output.weight"):
        return "head"
    if name.startswith("blk."):
        return "nar" if ".nar_" in name else "ar"
    return "extra"


@lru_cache(maxsize=32)
def _gguf_groups(path: str, size: int, mtime: float) -> dict:
    groups = {g: {"elements": 0, "bytes": 0} for g in GROUPS}
    for tensor in read_gguf_tensors(Path(path)):
        group = groups[_classify(tensor["name"])]
        group["elements"] += tensor["elements"]
        group["bytes"] += tensor["bytes"]
    return groups


def _arch_from(model: str) -> dict:
    arch = dict(DEFAULT_ARCH)
    config = Path(model) / "config.json"
    if config.is_file():
        try:
            c = json.loads(config.read_text(encoding="utf-8"))
            arch.update(hidden=c.get("hidden_size", arch["hidden"]), layers=c.get("num_hidden_layers", arch["layers"]),
                        heads=c.get("num_attention_heads", arch["heads"]),
                        kv_heads=c.get("num_key_value_heads", arch["kv_heads"]), head_dim=c.get("head_dim", arch["head_dim"]),
                        ff=c.get("intermediate_size", arch["ff"]), vocab=c.get("vocab_size", arch["vocab"]),
                        frames=c.get("max_latent_frames", arch["frames"]))
        except (OSError, ValueError):
            pass
    return arch


def _hf_groups(arch: dict) -> dict:
    h, f, hd = arch["hidden"], arch["ff"], arch["head_dim"]
    attn = h * arch["heads"] * hd * 2 + h * arch["kv_heads"] * hd * 2 + 2 * hd
    layer = attn + 3 * h * f + 2 * h
    extra = (h * arch["latent"] + h) + (arch["latent"] * h + arch["latent"]) + (256 * h + h) + (h * h + h) \
        + arch["frames"] * h + h
    elements = {"embed": arch["vocab"] * h, "head": arch["vocab"] * h, "ar": layer * arch["layers"],
                "nar": layer * arch["layers"], "extra": extra}
    return {g: {"elements": e, "bytes": e * 2} for g, e in elements.items()}


def model_groups(model: str) -> dict:
    if model_format(model) == "gguf":
        path = Path(model)
        stat = path.stat()
        return _gguf_groups(str(path.resolve()), stat.st_size, stat.st_mtime)
    return _hf_groups(_arch_from(model))


# ---------------------------------------------------------------- measured profiles
def profile_key(s: Settings) -> str:
    gguf = model_format(s.model) == "gguf"
    return "|".join([s.model, s.vae, f"gguf={s.gguf_mode if gguf else '-'}", f"q={s.quantization}",
                     f"offload={int(s.offload)}", f"backend={s.backend}", f"device={s.device}"])


class ProfileStore:
    """Measured peaks per configuration (keeps the largest song seen)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> dict | None:
        return self._read().get(key)

    def record(self, key: str, peaks: dict, song_seconds: float, phases: dict | None = None) -> None:
        if not peaks:
            return
        data = self._read()
        old = data.get(key, {})
        entry = {**old, "song_seconds": max(song_seconds, old.get("song_seconds", 0)), "updated": time.time()}
        for name, value in peaks.items():
            entry[name] = max(value, old.get(name, 0))
        if phases:
            merged = dict(old.get("phases") or {})
            for phase, values in phases.items():
                merged[phase] = {k: max(v, (merged.get(phase) or {}).get(k, 0)) for k, v in values.items()}
            entry["phases"] = merged
        data[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------- prediction
def predict(settings: Settings, hints: dict | None = None, measured: dict | None = None,
            calibration: dict | None = None) -> dict:
    """Peak VRAM / host RAM of *this process* for one song with these settings.

    Each phase = exact weights + estimated overhead (caches, activations). A
    calibration (factors learned from measured songs) scales the overhead."""
    hints = hints or {}
    s = settings
    gguf = model_format(s.model) == "gguf"
    groups = model_groups(s.model)
    arch = DEFAULT_ARCH if gguf else _arch_from(s.model)
    mode = s.gguf_mode if gguf else None

    packed = {"embed": False, "extra": False, "head": mode == "max", "ar": mode == "max",
              "nar": mode in ("balanced", "max")}

    def gpu(group: str) -> int:
        g = groups[group]
        return g["bytes"] if packed[group] else g["elements"] * 2  # unpacked weights live as BF16

    fp8 = s.quantization == "fp8" and mode != "max"
    eager = s.backend == "torch-eager" or mode == "max"
    graphs = not eager and not fp8 and s.backend == "torch"
    ar_phase_ar = groups["ar"]["elements"] if fp8 else gpu("ar")  # FP8 AR: 1 byte per weight

    plan = hints.get("plan_mode") or "full"
    cfg = hints.get("cfg_scale")
    branches = 2 if (cfg is not None and cfg != 1) or (cfg is None and plan == "off") else 1
    audio_tokens = int(hints.get("audio_max_tokens") or 9000)
    tokens = PREFIX_TOKENS.get(plan, 3500) + audio_tokens
    # the AR KV cache is allocated for the full budget; the NAR works on what was generated
    nar_tokens = PREFIX_TOKENS.get(plan, 3500) + int(hints.get("audio_tokens_actual") or audio_tokens)
    kv_per_token = arch["layers"] * 2 * arch["kv_heads"] * arch["head_dim"] * 2
    kv = tokens * kv_per_token * branches
    fused = (arch["layers"] * (arch["heads"] * arch["head_dim"] + 2 * arch["kv_heads"] * arch["head_dim"] + 2 * arch["ff"])
             * arch["hidden"] * 2) if graphs else 0

    others = gpu("embed") + gpu("head") + gpu("extra")
    tile = 512 if s.offload else 1024
    parts = {  # phase -> (weights on the GPU, overhead) in bytes
        "Komponieren": (others + ar_phase_ar + gpu("nar"), kv + fused),
        "Klangsynthese": (gpu("nar") + gpu("extra") + (0 if s.offload else gpu("embed") + gpu("head") + gpu("ar")),
                          nar_tokens * kv_per_token + NAR_TRANSIENT_GIB * GIB),
        "Dekodieren": (VAE_GIB * GIB, VAE_ACT_GIB_PER_1K * tile / 1024 * GIB),
    }
    factors = (calibration or {}).get("factors", {})
    phases = {name: w + o * factors.get(name, 1.0) for name, (w, o) in parts.items()}
    vram_peak = max(phases.values()) / GIB + CUDA_CONTEXT_GIB
    calibrated = bool(factors)

    model_bytes = sum(gpu(g) for g in GROUPS)
    ram_model = model_bytes  # loading and decoding both hold the whole model in host RAM once
    base = (calibration or {}).get("ram_base_gib", PROCESS_BASE_GIB)
    ram_peak = base + ram_model / GIB + VAE_GIB + 0.3
    idle = {"ram": base + model_bytes / GIB, "gpu": base, "unload": base}[s.idle_placement]

    result = {
        "model": s.model, "gguf_mode": mode, "key": profile_key(s),
        "assumptions": {"audio_tokens": audio_tokens, "song_seconds": round(audio_tokens / 25),
                        "cfg_branches": branches, "cuda_graphs": graphs, "fp8": fp8, "offload": s.offload},
        "weights_gib": round(model_bytes / GIB, 2), "weights_bytes": model_bytes,
        "file_gib": round(sum(g["bytes"] for g in groups.values()) / GIB, 2) if gguf else None,
        "vram": {"peak_gib": round(vram_peak, 2), "basis": "kalibriert" if calibrated else "geschätzt",
                 "phases": {k: round(v / GIB + CUDA_CONTEXT_GIB, 2) for k, v in phases.items()}},
        "ram": {"peak_gib": round(ram_peak, 2), "idle_gib": round(idle, 2),
                "basis": "kalibriert" if "ram_base_gib" in (calibration or {}) else "geschätzt"},
        "breakdown": {k: {"weights_gib": round(w / GIB, 3), "overhead_gib": round(o / GIB, 3)} for k, (w, o) in parts.items()},
        "calibration": {"factors": factors, "samples": (calibration or {}).get("samples", 0)} if calibration else None,
    }
    if measured:
        if "vram_peak_gib" in measured:
            result["vram"].update(peak_gib=round(measured["vram_peak_gib"] + CUDA_CONTEXT_GIB, 2), basis="gemessen",
                                  estimate_gib=round(vram_peak, 2))
        if "ram_peak_gib" in measured:
            result["ram"].update(peak_gib=measured["ram_peak_gib"], basis="gemessen", estimate_gib=round(ram_peak, 2))
        result["measured_song_seconds"] = measured.get("song_seconds")
    return result


def with_overrides(settings: Settings, values: dict | None) -> Settings:
    """Validated copy of the settings with UI values applied (raises ValueError)."""
    candidate = replace(settings)
    if values:
        candidate.update(values)
    return candidate


def system_view(prediction: dict, gpu: dict | None, ram: dict | None,
                process_rss_gib: float | None, process_vram_gib: float, cuda_active: bool) -> dict:
    """Translate process peaks into whole-device numbers for the bars (others + our peak)."""
    view = {}
    if gpu and gpu.get("cuda"):
        ours_now = process_vram_gib + (CUDA_CONTEXT_GIB if cuda_active else 0)
        others = max(gpu["used_gib"] - ours_now, 0)
        view["vram"] = {"total_gib": gpu["total_gib"], "used_gib": gpu["used_gib"],
                        "predicted_gib": round(others + prediction["vram"]["peak_gib"], 2)}
    if ram:
        ours_now = process_rss_gib or 0
        others = max(ram["used_gib"] - ours_now, 0)
        view["ram"] = {"total_gib": ram["total_gib"], "used_gib": ram["used_gib"],
                       "predicted_gib": round(others + prediction["ram"]["peak_gib"], 2),
                       "idle_gib": round(others + prediction["ram"]["idle_gib"], 2)}
    return view
