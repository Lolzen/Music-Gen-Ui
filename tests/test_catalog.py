import json
import struct
from pathlib import Path

from fastapi.testclient import TestClient

from musicgen_ui import catalog
from musicgen_ui.config import Settings
from musicgen_ui.engine.mock import MockEngine
from musicgen_ui.server import create_app


def _s(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path: Path, kv: dict) -> None:
    """Minimal GGUF v3 header with string/uint32/string-array values and no tensors."""
    body = b""
    for key, value in kv.items():
        body += _s(key)
        if isinstance(value, str):
            body += struct.pack("<I", 8) + _s(value)
        elif isinstance(value, int):
            body += struct.pack("<I", 4) + struct.pack("<I", value)
        elif isinstance(value, list):
            body += struct.pack("<I", 9) + struct.pack("<IQ", 8, len(value)) + b"".join(_s(v) for v in value)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kv)) + body)


def make_models(root: Path) -> None:
    catalog.ensure_dirs(root)
    write_gguf(root / "yue2" / "yue2-lm-Q4_K_M-imat.gguf",
               {"general.architecture": "yue2", "general.name": "YuE2 3B", "general.file_type": 15,
                "tokenizer.ggml.tokens": [f"t{i}" for i in range(500)]})
    write_gguf(root / "yue2" / "YuE-s2-1B-general-Q4_K_M.gguf",
               {"general.architecture": "llama", "general.name": "YuE-s2-1B-general", "general.file_type": 15})
    (root / "yue2" / "broken.gguf").write_bytes(b"nope")
    hf = root / "yue2" / "YuE2-3B"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "yue2"}))
    (hf / "model.safetensors").write_bytes(b"x" * 2048)
    vae = root / "vae" / "YuE2-Vae"
    vae.mkdir()
    (vae / "config.json").write_text(json.dumps({"model_type": "yue2_vae"}))
    (vae / "model.safetensors").write_bytes(b"x")
    wrong = root / "vae" / "oops"
    wrong.mkdir()
    (wrong / "config.json").write_text(json.dumps({"model_type": "yue2"}))
    (wrong / "model.safetensors").write_bytes(b"x")
    write_gguf(root / "vae" / "yue2-vae-standard-f32.gguf", {"general.architecture": "yue2-vae", "general.file_type": 0})
    (root / "lora").mkdir(exist_ok=True)
    (root / "lora" / "dreampop.safetensors").write_bytes(b"x")
    (root / "lora" / "lora.json").write_text(json.dumps({"name": "DreamPop v2", "trigger": "sv_dreampop"}))


def test_gguf_header_reader(tmp_path):
    write_gguf(tmp_path / "m.gguf", {"general.architecture": "yue2", "general.file_type": 7,
                                     "tokenizer.ggml.tokens": [str(i) for i in range(1000)], "x.after": "ok"})
    meta = catalog.read_gguf_metadata(tmp_path / "m.gguf")
    assert meta["general.architecture"] == "yue2" and meta["general.file_type"] == 7
    assert meta["tokenizer.ggml.tokens"] == "<1000 Werte>" and meta["x.after"] == "ok"


def test_scan_classifies_models(tmp_path):
    make_models(tmp_path)
    result = catalog.scan(tmp_path)
    ckpt = {e["label"]: e for e in result["checkpoint"]}
    q4 = ckpt["yue2-lm-Q4_K_M-imat"]
    assert q4["format"] == "gguf" and q4["engine"] == "yue2" and q4["quant"] == "Q4_K_M"
    assert q4["architecture"] == "yue2"
    v1 = ckpt["YuE-s2-1B-general-Q4_K_M"]
    assert v1["engine"] is None and "YuE v1" in v1["note"]
    assert ckpt["broken"]["engine"] is None
    assert ckpt["YuE2-3B"]["format"] == "hf" and ckpt["YuE2-3B"]["engine"] == "yue2"
    assert ckpt["m-a-p/YuE2-3B (Hugging Face)"]["source"] == "hub"
    vae = {e["label"]: e for e in result["vae"]}
    assert vae["YuE2-Vae"]["engine"] == "yue2" and vae["oops"]["engine"] is None
    assert vae["yue2-vae-standard-f32"]["quant"] == "F32"
    assert result["lora"][0]["label"] == "DreamPop v2"
    assert (tmp_path / "README.txt").is_file()


def test_models_api_and_gguf_selection_message(tmp_path):
    make_models(tmp_path / "models")
    settings = Settings(engine="mock", data_dir=tmp_path / "data", plugin_dirs=[], models_dir=tmp_path / "models")
    with TestClient(create_app(settings, engine=MockEngine(0))) as c:
        data = c.get("/api/models").json()
        assert data["selected"]["checkpoint"] == "m-a-p/YuE2-3B"
        gguf = next(e for e in data["checkpoint"] if e["format"] == "gguf" and e["engine"])
        assert c.put("/api/settings", json={"model": gguf["value"]}).status_code == 200
        assert c.get("/api/models").json()["selected"]["checkpoint"] == gguf["value"]


def test_gguf_in_wrong_folder_and_unknown_layout(tmp_path):
    catalog.ensure_dirs(tmp_path)
    write_gguf(tmp_path / "yue2" / "yue2-vae-standard-f32.gguf", {"general.architecture": "yue2-vae"})
    write_gguf(tmp_path / "vae" / "yue2-lm-q8_0.gguf", {"general.architecture": "yue2"})
    write_gguf(tmp_path / "yue2" / "other.gguf", {"general.architecture": "edgespeak-yue2"})
    result = catalog.scan(tmp_path)
    notes = {e["label"]: e["note"] for e in result["checkpoint"] + result["vae"] if e["source"] == "local"}
    assert "models/vae/" in notes["yue2-vae-standard-f32"]
    assert "models/yue2/" in notes["yue2-lm-q8_0"]
    assert "Unbekanntes GGUF-Layout" in notes["other"]
