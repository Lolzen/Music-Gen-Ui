"""Model catalog: scans the ``models/`` folder (like image-generation UIs).

    models/
      yue2/   checkpoints: Hugging-Face folders (config.json + *.safetensors) or *.gguf
      vae/    audio decoders: Hugging-Face folders or *.gguf
      lora/   LoRA adapters (*.safetensors), see docs/LORA.md

Hugging Face IDs (downloaded into the HF cache on first use) are always offered
as well. Every entry says which engine can run it and why it might not fit.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .download import cache_bytes

SUBDIRS = {"checkpoint": "yue2", "vae": "vae", "lora": "lora"}
HUB_CHECKPOINTS = ["m-a-p/YuE2-3B"]
HUB_VAES = ["m-a-p/YuE2-Vae", "m-a-p/YuE2-Vae-legacy"]
HF_MODEL_TYPES = {"checkpoint": "yue2", "vae": "yue2_vae"}

README = """Modell-Ordner von Music-Gen-UI
==============================

yue2/  Checkpoints: Hugging-Face-Ordner (config.json + *.safetensors) oder *.gguf
vae/   Audio-Decoder (VAE): Hugging-Face-Ordner oder *.gguf
lora/  LoRA-Adapter (*.safetensors)

Neue Dateien erscheinen nach "Aktualisieren" (🔄) im Modell-Picker der Oberfläche.
"""

# llama.cpp ``general.file_type`` values
GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}
_QUANT_IN_NAME = re.compile(r"(?i)(?<![A-Z0-9])(I?Q\d(?:_[0-9A-Z]+)*|BF16|F16|F32|NVFP4|MXFP4)(?![A-Z0-9])")


@dataclass
class ModelEntry:
    id: str                 # stable key for the UI
    kind: str               # checkpoint | vae | lora
    label: str
    value: str              # what goes into the settings (hub id or path)
    source: str             # hub | local
    format: str             # hf | gguf | safetensors
    engine: str | None      # engine that can run it: "yue2" (PyTorch) | "gguf" | None (unusable)
    size_gib: float | None = None
    quant: str | None = None
    architecture: str | None = None
    downloaded: bool | None = None
    note: str | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- GGUF header
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf_metadata(path: Path, max_array_items: int = 16) -> dict:
    """Read the key/value header of a GGUF file (no tensors, no dependencies)."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("keine GGUF-Datei")
        (version,) = struct.unpack("<I", fh.read(4))
        if version < 2:
            raise ValueError(f"GGUF-Version {version} wird nicht unterstützt")
        tensor_count, kv_count = struct.unpack("<QQ", fh.read(16))

        def read_string() -> str:
            (length,) = struct.unpack("<Q", fh.read(8))
            return fh.read(length).decode("utf-8", errors="replace")

        def read_value(vtype: int):
            if vtype in _GGUF_SCALARS:
                fmt = _GGUF_SCALARS[vtype]
                return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
            if vtype == 8:
                return read_string()
            if vtype == 9:
                item_type, count = struct.unpack("<IQ", fh.read(12))
                if count > max_array_items and item_type in _GGUF_SCALARS:
                    fh.seek(count * struct.calcsize(_GGUF_SCALARS[item_type]), 1)  # skip big arrays fast
                    return f"<{count} Werte>"
                if count > max_array_items and item_type == 8:  # e.g. tokenizer vocab
                    for _ in range(count):
                        (length,) = struct.unpack("<Q", fh.read(8))
                        fh.seek(length, 1)
                    return f"<{count} Werte>"
                items = [read_value(item_type) for _ in range(count)]
                return items if count <= max_array_items else f"<{count} Werte>"
            raise ValueError(f"unbekannter GGUF-Werttyp {vtype}")

        meta = {"gguf.version": version, "gguf.tensor_count": tensor_count}
        for _ in range(kv_count):
            key = read_string()
            (vtype,) = struct.unpack("<I", fh.read(4))
            meta[key] = read_value(vtype)
        return meta


def _quant_from(meta: dict, name: str) -> str | None:
    file_type = meta.get("general.file_type")
    if isinstance(file_type, int) and file_type in GGUF_FILE_TYPES:
        return GGUF_FILE_TYPES[file_type]
    match = _QUANT_IN_NAME.search(name)
    return match.group(1).upper() if match else None


def _gguf_entry(path: Path, kind: str, root: Path) -> ModelEntry:
    entry = ModelEntry(id=f"local:{path.relative_to(root)}", kind=kind, label=path.stem, value=str(path.resolve()),
                       source="local", format="gguf", engine="gguf", size_gib=round(path.stat().st_size / 2**30, 2))
    try:
        meta = read_gguf_metadata(path)
    except (OSError, ValueError, struct.error) as exc:
        entry.engine, entry.note = None, f"GGUF-Kopf nicht lesbar: {exc}"
        return entry
    entry.architecture = meta.get("general.architecture")
    entry.quant = _quant_from(meta, path.name)
    entry.meta = {k: v for k, v in meta.items() if k.startswith("general.") and not isinstance(v, list)}
    name = f"{meta.get('general.name', '')} {path.name}".lower()
    if entry.architecture == "llama" and "yue" in name and ("yue2" not in name):
        entry.engine = None
        entry.note = ("YuE v1 (Vorgänger), z.B. Stage-2-Modell – passt nicht zu YuE2 und erzeugt allein keine Songs")
    elif entry.quant and entry.quant.startswith(("Q2", "IQ1", "IQ2")):
        entry.note = "sehr starke Quantisierung – deutlicher Qualitätsverlust zu erwarten"
    return entry


def _hf_entry(path: Path, kind: str, root: Path) -> ModelEntry | None:
    config_file = path / "config.json"
    weights = list(path.glob("*.safetensors"))
    if not config_file.is_file() or not weights:
        return None
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}
    model_type = config.get("model_type")
    entry = ModelEntry(id=f"local:{path.relative_to(root)}", kind=kind, label=path.name, value=str(path.resolve()),
                       source="local", format="hf", engine="yue2", architecture=model_type,
                       size_gib=round(sum(f.stat().st_size for f in weights) / 2**30, 2))
    expected = HF_MODEL_TYPES.get(kind)
    if model_type and expected and model_type != expected:
        entry.engine = None
        entry.note = f"model_type '{model_type}' passt nicht in diesen Ordner (erwartet '{expected}')"
    return entry


def _lora_entry(path: Path, root: Path) -> ModelEntry:
    sidecar = path.with_name("lora.json")
    meta = {}
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    return ModelEntry(id=f"local:{path.relative_to(root)}", kind="lora", label=meta.get("name") or path.stem,
                      value=str(path.resolve()), source="local", format="safetensors", engine=None,
                      size_gib=round(path.stat().st_size / 2**30, 3), meta=meta,
                      note="LoRA-Loader folgt (Roadmap Phase 2)")


def _hub_entry(repo: str, kind: str) -> ModelEntry:
    size = cache_bytes([repo])
    return ModelEntry(id=f"hub:{repo}", kind=kind, label=f"{repo} (Hugging Face)", value=repo, source="hub",
                      format="hf", engine="yue2", downloaded=size > 0,
                      size_gib=round(size / 2**30, 2) if size else None,
                      note=None if size else "wird beim ersten Laden heruntergeladen")


def ensure_dirs(root: Path) -> None:
    try:
        for sub in SUBDIRS.values():
            (root / sub).mkdir(parents=True, exist_ok=True)
        readme = root / "README.txt"
        if not readme.exists():
            readme.write_text(README, encoding="utf-8")
    except OSError:
        pass


def _scan(root: Path, kind: str) -> list[ModelEntry]:
    folder = root / SUBDIRS[kind]
    if not folder.is_dir():
        return []
    entries = []
    for path in sorted(folder.rglob("*")):
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        if kind == "lora":
            if path.suffix == ".safetensors" and path.is_file():
                entries.append(_lora_entry(path, root))
        elif path.suffix.lower() == ".gguf" and path.is_file():
            entries.append(_gguf_entry(path, kind, root))
        elif path.is_dir():
            entry = _hf_entry(path, kind, root)
            if entry is not None:
                entries.append(entry)
    return entries


def scan(root: Path) -> dict:
    root = Path(root)
    ensure_dirs(root)
    return {
        "root": str(root.resolve()),
        "dirs": {kind: str((root / sub).resolve()) for kind, sub in SUBDIRS.items()},
        "checkpoint": [asdict(e) for e in [*_scan(root, "checkpoint"), *(_hub_entry(r, "checkpoint") for r in HUB_CHECKPOINTS)]],
        "vae": [asdict(e) for e in [*_scan(root, "vae"), *(_hub_entry(r, "vae") for r in HUB_VAES)]],
        "lora": [asdict(e) for e in _scan(root, "lora")],
    }


def model_format(value: str) -> str:
    """Format of a configured model/VAE value: "gguf" for .gguf files, else "hf"."""
    return "gguf" if str(value).lower().endswith(".gguf") else "hf"
