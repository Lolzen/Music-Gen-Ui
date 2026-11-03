"""LoRA adapters for YuE2: read the common formats, apply to / remove from a loaded model.

Supported files (see docs/LORA.md):

* **Mothersuperior / Music-Gen-UI** (``.safetensors``, also ``.pt`` via safe loading):
  ``layers.N.[nar_]self_attn.{q,k,v,o}_proj.lora_A|lora_B`` and ``layers.N.[nar_]mlp.*``;
  ``W += B @ A`` (scale 1, or ``alpha/rank`` from metadata). NAR files may carry *full*
  ``vae2llm``/``llm2vae`` replacements. ``.pt`` artist checkpoints: ``{"lora": [A0,B0,...]}``.
* **ComfyUI native** (Starnodes, lym00, ``*_comfyui``): ``diffusion_model.model.layers.N.*`` = NAR,
  ``text_encoders.model.layers.N.*`` = AR; fused ``qkv_proj`` / ``gate_up_proj`` with
  ``lora_down``/``lora_up`` (+ optional ``.alpha``); ``.diff`` / ``.diff_b`` full deltas.
* **Starnodes internal** ``yue2-lora-v1``: ``model.layers.N.nar_self_attn.q_proj.lora_down.weight``.

Deltas are merged in place (keeps YuE2's CUDA graphs working); removing an adapter
restores the touched tensors exactly from the model source (no RAM copies kept).
Packed GGUF layers (``QuantLinear``) get the LoRA as a side path instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
# YuE2-3B row split of ComfyUI's fused matrices
DEFAULT_DIMS = {"q": 2048, "kv": 1024, "ff": 6144}


@dataclass
class LowRank:
    A: torch.Tensor          # [r, in]
    B: torch.Tensor          # [out, r]
    scale: float = 1.0


@dataclass
class Adapter:
    path: str
    name: str
    format: str
    deltas: dict[str, LowRank] = field(default_factory=dict)          # module path -> low-rank delta
    replacements: dict[str, torch.Tensor] = field(default_factory=dict)  # param path -> new value
    diffs: dict[str, torch.Tensor] = field(default_factory=dict)       # param path -> additive delta
    meta: dict = field(default_factory=dict)

    @property
    def halves(self) -> list[str]:
        modules = list(self.deltas) + [k.rsplit(".", 1)[0] for k in (*self.replacements, *self.diffs)]
        found = set()
        for name in modules:
            found.add("nar" if (".nar_" in name or name.startswith(("vae2llm", "llm2vae", "time_embedder"))) else "ar")
        return sorted(found)

    @property
    def rank(self) -> int | None:
        ranks = {d.A.shape[0] for d in self.deltas.values()}
        return min(ranks) if ranks else None

    def summary(self) -> dict:
        return {"name": self.name, "format": self.format, "halves": self.halves, "rank": self.rank,
                "modules": len(self.deltas), "replacements": len(self.replacements) + len(self.diffs),
                "trigger": self.meta.get("trigger"), "meta": {k: v for k, v in self.meta.items() if isinstance(v, (str, int, float, bool))}}


# ---------------------------------------------------------------- reading
def _load_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), "pt") as handle:
            meta = dict(handle.metadata() or {})
        return load_file(str(path)), meta
    if path.suffix in (".pt", ".pth", ".ckpt"):
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=True)  # never unpickle code
        except Exception as exc:
            raise ValueError(f"{path.name}: .pt-Datei lässt sich nicht sicher laden ({exc}). "
                             "Bitte die .safetensors-Version der LoRA verwenden.") from exc
        return _from_pt(data), {"rank": str(data.get("rank", "")), "targets": str(data.get("targets", ""))}
    raise ValueError(f"{path.name}: unbekanntes Dateiformat")


def _from_pt(data: dict) -> dict[str, torch.Tensor]:
    """Mothersuperior .pt: {"lora": [A0,B0,...] layer-major, "io": {...}} → named tensors."""
    if not isinstance(data, dict) or "lora" not in data:
        raise ValueError("unbekanntes .pt-Format (erwartet {'lora': [...]})")
    nar = "io" in data or "nar" in str(data.get("targets", "")).lower().split(" ")[0]
    prefix = "nar_" if nar else ""
    names = [(f"{prefix}self_attn", p) for p in ATTN] + [(f"{prefix}mlp", p) for p in MLP]
    tensors, it = {}, iter(data["lora"])
    layer = 0
    while True:
        try:
            for block, proj in names:
                tensors[f"layers.{layer}.{block}.{proj}.lora_A"] = next(it)
                tensors[f"layers.{layer}.{block}.{proj}.lora_B"] = next(it)
        except StopIteration:
            break
        layer += 1
    for module, state in (data.get("io") or {}).items():
        for key, value in state.items():
            tensors[f"{module}.{key}"] = value
    return tensors


def _meta_scale(meta: dict) -> float:
    try:
        alpha, rank = float(meta.get("alpha", "")), float(meta.get("rank", ""))
        return alpha / rank if rank else 1.0
    except ValueError:
        return 1.0


_MS = re.compile(r"^layers\.(\d+)\.((?:nar_)?(?:self_attn|mlp))\.(\w+_proj)\.lora_([AB])$")
_COMFY = re.compile(r"^(diffusion_model|text_encoders)\.model\.layers\.(\d+)\.(self_attn|mlp)\."
                    r"(qkv_proj|o_proj|gate_up_proj|down_proj)\.(lora_down|lora_up)\.weight$")
_COMFY_ALPHA = re.compile(r"^(diffusion_model|text_encoders)\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.alpha$")
_STARNODES = re.compile(r"^(model\.layers\.\d+\.(?:nar_)?(?:self_attn|mlp)\.\w+_proj)\.lora_(down|up)\.weight$")
_COMFY_DIFF = re.compile(r"^diffusion_model\.(vae2llm|llm2vae|time_embedder\.mlp\.\d|latent_pos_embed)\.(diff|diff_b)$")


def read_adapter(path: Path, dims: dict | None = None) -> Adapter:
    path = Path(path)
    tensors, meta = _load_tensors(path)
    dims = {**DEFAULT_DIMS, **(dims or {})}
    sidecar = path.with_suffix(".json") if path.with_suffix(".json").is_file() else path.with_name("lora.json")
    if sidecar.is_file():
        try:
            meta = {**meta, **json.loads(sidecar.read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            pass
    adapter = Adapter(str(path), meta.get("name") or path.stem, "unknown", meta=meta)
    keys = list(tensors)
    if any(_MS.match(k) for k in keys):
        adapter.format = "mothersuperior"
        scale = _meta_scale(meta)
        pairs: dict[str, dict] = {}
        for key in keys:
            m = _MS.match(key)
            if m:
                pairs.setdefault(f"model.layers.{m[1]}.{m[2]}.{m[3]}", {})[m[4]] = tensors[key]
            elif re.match(r"^(vae2llm|llm2vae)\.(weight|bias)$", key):
                adapter.replacements[key] = tensors[key]
        for module, ab in pairs.items():
            adapter.deltas[module] = LowRank(ab["A"].float(), ab["B"].float(), scale)
    elif any(_COMFY.match(k) for k in keys):
        adapter.format = "comfyui"
        alphas = {}
        for key in keys:
            m = _COMFY_ALPHA.match(key)
            if m:
                alphas[(m[1], m[2], m[3], m[4])] = float(tensors[key].float().reshape(-1)[0])
        fused: dict[tuple, dict] = {}
        for key in keys:
            m = _COMFY.match(key)
            if m:
                fused.setdefault((m[1], m[2], m[3], m[4]), {})[m[5]] = tensors[key].float()
            elif _COMFY_DIFF.match(key):
                d = _COMFY_DIFF.match(key)
                adapter.diffs[f"{d[1]}.{'weight' if d[2] == 'diff' else 'bias'}"] = tensors[key].float()
        q, kv, ff = dims["q"], dims["kv"], dims["ff"]
        for (root, layer, block, proj), wts in fused.items():
            down, up = wts["lora_down"], wts["lora_up"]
            alpha = alphas.get((root, layer, block, proj))
            scale = alpha / down.shape[0] if alpha else 1.0
            hf_block = ("nar_" if root == "diffusion_model" else "") + block
            base = f"model.layers.{layer}.{hf_block}"
            if proj == "qkv_proj":
                for name, (lo, hi) in (("q_proj", (0, q)), ("k_proj", (q, q + kv)), ("v_proj", (q + kv, q + 2 * kv))):
                    adapter.deltas[f"{base}.{name}"] = LowRank(down, up[lo:hi], scale)
            elif proj == "gate_up_proj":
                adapter.deltas[f"{base}.gate_proj"] = LowRank(down, up[:ff], scale)
                adapter.deltas[f"{base}.up_proj"] = LowRank(down, up[ff:2 * ff], scale)
            else:
                adapter.deltas[f"{base}.{proj}"] = LowRank(down, up, scale)
    elif any(_STARNODES.match(k) for k in keys):
        adapter.format = "starnodes"
        scale = _meta_scale({"alpha": meta.get("alpha", meta.get("source_alpha", "")),
                             "rank": meta.get("rank", meta.get("source_rank", ""))})
        pairs = {}
        for key in keys:
            m = _STARNODES.match(key)
            if m:
                pairs.setdefault(m[1], {})[m[2]] = tensors[key].float()
        for module, du in pairs.items():
            adapter.deltas[module] = LowRank(du["down"], du["up"], scale)
    else:
        raise ValueError(f"{path.name}: unbekanntes LoRA-Layout, z.B. Schlüssel {keys[:3]}")
    adapter.meta.setdefault("trigger", meta.get("trigger_word") or meta.get("modelspec.trigger_phrase") or meta.get("trigger"))
    return adapter


def inspect_adapter(path: Path) -> dict:
    """Cheap summary for the model catalog (reads only the safetensors header)."""
    path = Path(path)
    if path.suffix != ".safetensors":
        return {"format": "pt", "halves": [], "note": ".pt-Datei – Details beim Laden"}
    import struct
    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    meta = header.pop("__metadata__", {}) or {}
    keys = list(header)
    if any(_MS.match(k) for k in keys):
        fmt = "mothersuperior"
        halves = sorted({"nar" if ".nar_" in k else "ar" for k in keys if _MS.match(k)})
    elif any(_COMFY.match(k) for k in keys):
        fmt = "comfyui"
        halves = sorted({"nar" if k.startswith("diffusion_model") else "ar" for k in keys if _COMFY.match(k)})
    elif any(_STARNODES.match(k) for k in keys):
        fmt, halves = "starnodes", sorted({"nar" if ".nar_" in k else "ar" for k in keys if _STARNODES.match(k)})
    else:
        return {"format": None, "halves": [], "note": "kein bekanntes YuE2-LoRA-Layout"}
    return {"format": fmt, "halves": halves, "meta": meta,
            "trigger": meta.get("trigger") or meta.get("trigger_word") or meta.get("modelspec.trigger_phrase")}


# ---------------------------------------------------------------- applying
SourceFn = Callable[[str], torch.Tensor]  # HF param path -> original tensor (CPU)


class LoraState:
    """Tracks which adapters are merged into a model and undoes them exactly."""

    def __init__(self):
        self.signature: tuple = ()
        self.touched: set[str] = set()      # param paths changed in place
        self.side: list = []                # QuantLinear modules with side adapters

    def clear(self, model: torch.nn.Module, source: SourceFn) -> None:
        with torch.no_grad():
            for param_path in sorted(self.touched):
                param = model.get_parameter(param_path)
                param.copy_(source(param_path).to(device=param.device, dtype=param.dtype))
            for module in self.side:
                module.adapters.clear()
        self.touched, self.side, self.signature = set(), [], ()

    def apply(self, model: torch.nn.Module, adapters: list[tuple[Adapter, float]], source: SourceFn) -> dict:
        signature = tuple((a.path, round(s, 4)) for a, s in adapters)
        if signature == self.signature:
            return {"changed": False}
        self.clear(model, source)
        from .gguf_support.loader import QuantLinear  # local: needs torch only
        merged = side = 0
        with torch.no_grad():
            for adapter, strength in adapters:
                if strength == 0:
                    continue
                for module_path, delta in adapter.deltas.items():
                    module = model.get_submodule(module_path)
                    if isinstance(module, QuantLinear):
                        module.adapters.append((delta.A, delta.B, delta.scale * strength))
                        self.side.append(module)
                        side += 1
                        continue
                    weight = module.weight
                    update = (delta.B.to(weight.device, torch.float32) @ delta.A.to(weight.device, torch.float32))
                    weight.add_((update * (delta.scale * strength)).to(weight.dtype))
                    self.touched.add(f"{module_path}.weight")
                    merged += 1
                for param_path, value in adapter.replacements.items():
                    param = model.get_parameter(param_path)
                    target = value.to(param.device, torch.float32)
                    param.copy_((param.float() * (1 - strength) + target * strength).to(param.dtype))
                    self.touched.add(param_path)
                for param_path, value in adapter.diffs.items():
                    param = model.get_parameter(param_path)
                    param.add_((value.to(param.device, torch.float32) * strength).to(param.dtype))
                    self.touched.add(param_path)
        self.signature = signature
        return {"changed": True, "merged": merged, "side": side, "touched": len(self.touched)}


def hf_source(model_dir: Path) -> SourceFn:
    """Original tensors from a Hugging-Face YuE2 folder (memory-mapped, read on demand)."""
    from safetensors import safe_open
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        files = sorted(model_dir.glob("*.safetensors"))
        mapping = {}
        for file in files:
            with safe_open(str(file), "pt") as handle:
                mapping.update({k: file.name for k in handle.keys()})

    def get(name: str) -> torch.Tensor:
        with safe_open(str(model_dir / mapping[name]), "pt") as handle:
            return handle.get_tensor(name)
    return get


def gguf_source(path: Path) -> SourceFn:
    """Original (dequantized) tensors from a YuE2 GGUF file."""
    import gguf
    from .gguf_support.loader import lm_hf_name, tensor_to_float
    reader = gguf.GGUFReader(str(path))
    by_hf = {lm_hf_name(t.name): t for t in reader.tensors}

    def get(name: str) -> torch.Tensor:
        return tensor_to_float(by_hf[name], torch.float32)
    return get
