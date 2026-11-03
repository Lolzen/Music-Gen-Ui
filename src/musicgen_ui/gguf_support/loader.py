"""Load YuE2 GGUF files into the official PyTorch YuE2 pipeline.

Supported layout: the one written by HOT-Step-CPP's ``engine/tools/convert-yue2.py``
(e.g. https://huggingface.co/scragnog/YuE2-GGUF):

* ``yue2-lm-*.gguf``  – architecture ``yue2``: AR + NAR backbone, heads, tokenizer
* ``yue2-vae-*.gguf`` – architecture ``yue2-vae``: decoder (+ encoder), weight norm folded

Names map 1:1 onto ``YuE2ForCausalLM`` / ``YuE2VAE`` (no fused tensors).

Modes for the LM (``gguf_mode``):

* ``bf16``     – everything dequantized once while loading → full speed (CUDA graphs),
                 same VRAM as the original
* ``balanced`` – AR (composition) dequantized, NAR (sound) stays quantized and is
                 dequantized per layer call → CUDA graphs still work, less VRAM
* ``max``      – all transformer linears + lm_head stay quantized → least VRAM, but the AR
                 loop must run eagerly (no CUDA graphs) and is clearly slower
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGMLQuantizationType as Q
from torch import nn

from .dequant import dequantize, dequantize_functions

LM_ARCH, VAE_ARCH = "yue2", "yue2-vae"
GGUF_MODES = ("bf16", "balanced", "max")
ORDINARY_TOKENS = 151643  # qwen.tiktoken size expected by YuE2TextTokenizer
FLOAT_TYPES = (Q.F32, Q.F16)

# ---------------------------------------------------------------- name mapping
_LM_FIXED = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "vae2llm.weight": "vae2llm.weight", "vae2llm.bias": "vae2llm.bias",
    "llm2vae.weight": "llm2vae.weight", "llm2vae.bias": "llm2vae.bias",
    "time_embd.0.weight": "time_embedder.mlp.0.weight", "time_embd.0.bias": "time_embedder.mlp.0.bias",
    "time_embd.1.weight": "time_embedder.mlp.2.weight", "time_embd.1.bias": "time_embedder.mlp.2.bias",
    "latent_pos_embed.weight": "latent_pos_embed.pe",
}
_LM_BLOCK = re.compile(r"^blk\.(\d+)\.(nar_)?(attn|ffn)_(q_norm|k_norm|norm|q|k|v|output|gate|up|down)\.weight$")
_ATTN = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "output": "o_proj", "q_norm": "q_norm", "k_norm": "k_norm"}
_FFN = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def lm_hf_name(name: str) -> str | None:
    """GGUF tensor name → ``YuE2ForCausalLM`` state-dict key (None if unknown)."""
    if name in _LM_FIXED:
        return _LM_FIXED[name]
    match = _LM_BLOCK.match(name)
    if not match:
        return None
    index, nar, kind, part = match.groups()
    prefix = f"model.layers.{index}."
    if kind == "attn":
        if part == "norm":
            return prefix + ("nar_input_layernorm" if nar else "input_layernorm") + ".weight"
        sub = _ATTN.get(part)
        return f"{prefix}{'nar_self_attn' if nar else 'self_attn'}.{sub}.weight" if sub else None
    if part == "norm":
        return prefix + ("nar_pre_mlp_layernorm" if nar else "post_attention_layernorm") + ".weight"
    sub = _FFN.get(part)
    return f"{prefix}{'nar_mlp' if nar else 'mlp'}.{sub}.weight" if sub else None


_VAE_PATTERNS = [
    (re.compile(r"^dec\.conv_in$"), lambda m: "decoder.layers.0"),
    (re.compile(r"^dec\.blk\.(\d+)\.snake_pre$"), lambda m: f"decoder.layers.{m[1]}.layers.0"),
    (re.compile(r"^dec\.blk\.(\d+)\.upsample$"), lambda m: f"decoder.layers.{m[1]}.layers.1"),
    (re.compile(r"^dec\.blk\.(\d+)\.res\.(\d+)\.(snake1|conv1|snake2|conv2)$"),
     lambda m: f"decoder.layers.{m[1]}.layers.{int(m[2]) + 2}.layers."
               f"{['snake1', 'conv1', 'snake2', 'conv2'].index(m[3])}"),
    (re.compile(r"^dec\.snake_out$"), lambda m: "decoder.layers.7"),
    (re.compile(r"^dec\.conv_out$"), lambda m: "decoder.layers.8"),
]


def vae_hf_name(name: str) -> str | None:
    """GGUF VAE tensor → decoder module path + suffix (``enc.*`` returns None: not needed)."""
    base, _, suffix = name.rpartition(".")
    for pattern, build in _VAE_PATTERNS:
        match = pattern.match(base)
        if match:
            return f"{build(match)}.{suffix}"
    return None


# ---------------------------------------------------------------- tensor helpers
def torch_shape(tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in reversed(tensor.shape.tolist()))


def tensor_to_float(tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize one GGUF tensor on the CPU."""
    qtype, shape = Q(tensor.tensor_type), torch_shape(tensor)
    if qtype == Q.F32:
        return torch.from_numpy(np.array(tensor.data, dtype=np.float32)).reshape(shape).to(dtype)
    if qtype == Q.F16:
        return torch.from_numpy(np.array(tensor.data, dtype=np.float16)).reshape(shape).to(dtype)
    raw = torch.from_numpy(np.array(tensor.data)).view(torch.uint8)
    if qtype in dequantize_functions:
        return dequantize(raw, qtype, shape, dtype=torch.float32).to(dtype)
    values = gguf.quants.dequantize(np.array(tensor.data), qtype)  # slow numpy path (rare types)
    return torch.from_numpy(np.ascontiguousarray(values)).reshape(shape).to(dtype)


class QuantLinear(nn.Module):
    """Linear layer whose weight stays GGUF-quantized and is dequantized per call.

    The packed blocks are a non-trainable uint8 ``Parameter`` so ``.to()``,
    YuE2's AR offload and low-VRAM placement move them like any weight.
    """

    def __init__(self, packed: torch.Tensor, qtype: Q, in_features: int, out_features: int,
                 bias: torch.Tensor | None = None):
        super().__init__()
        self.in_features, self.out_features, self.qtype = in_features, out_features, qtype
        self.qweight = nn.Parameter(packed.view(torch.uint8).reshape(out_features, -1), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.adapters: list = []  # LoRA side paths (A [r,in], B [out,r], scale), see musicgen_ui.lora

    def dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        # fp16 matches the format of the block scales and is fast on GPUs; fp32 on CPU.
        compute = torch.float16 if self.qweight.device.type == "cuda" else torch.float32
        return dequantize(self.qweight, self.qtype, (self.out_features, self.in_features), dtype=compute).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.dequantized(x.dtype), self.bias)
        for A, B, scale in self.adapters:  # packed weights cannot absorb a LoRA, so add it on the side
            low = F.linear(x.float(), A.to(x.device, torch.float32))
            out = out + (F.linear(low, B.to(x.device, torch.float32)) * scale).to(out.dtype)
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, gguf={self.qtype.name}"


# ---------------------------------------------------------------- metadata / tokenizer
def _field_value(field_):
    if not field_.types:
        return None
    kind = field_.types[0]
    if kind == gguf.GGUFValueType.STRING:
        return bytes(field_.parts[field_.data[0]]).decode("utf-8", errors="replace")
    if kind == gguf.GGUFValueType.ARRAY:
        return None  # arrays (tokenizer etc.) are read explicitly
    return field_.parts[field_.data[0]].tolist()[0]


def scalar_metadata(reader: gguf.GGUFReader) -> dict:
    return {key: value for key, field_ in reader.fields.items()
            if (value := _field_value(field_)) is not None}


def _string_array(reader: gguf.GGUFReader, key: str) -> list[str]:
    field_ = reader.fields.get(key)
    if field_ is None:
        raise ValueError(f"GGUF enthält kein '{key}'")
    return [bytes(field_.parts[i]).decode("utf-8") for i in field_.data]


def _gpt2_byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def write_tiktoken(reader: gguf.GGUFReader, target: Path) -> None:
    """Rebuild ``qwen.tiktoken`` (``<base64 bytes> <rank>`` lines) from the GGUF vocabulary."""
    tokens = _string_array(reader, "tokenizer.ggml.tokens")
    if len(tokens) < ORDINARY_TOKENS + 1 or tokens[ORDINARY_TOKENS] != "<|endoftext|>":
        raise ValueError("GGUF-Tokenizer passt nicht zu YuE2 (erwartet 151643 Tokens + <|endoftext|>)")
    decoder = _gpt2_byte_decoder()
    lines = []
    for rank, token in enumerate(tokens[:ORDINARY_TOKENS]):
        try:
            data = bytes(decoder[ch] for ch in token)
        except KeyError as exc:
            raise ValueError(f"Unbekanntes Tokenizer-Format in Token {rank}") from exc
        lines.append(f"{base64.b64encode(data).decode()} {rank}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def lm_config(meta: dict):
    from yue2.modeling_yue2 import YuE2Config

    def get(key, default=None):
        value = meta.get(f"yue2.{key}", default)
        if value is None:
            raise ValueError(f"GGUF-Metadaten unvollständig: yue2.{key} fehlt")
        return value

    return YuE2Config(
        hidden_size=int(get("embedding_length")), num_hidden_layers=int(get("block_count")),
        num_attention_heads=int(get("attention.head_count")),
        num_key_value_heads=int(get("attention.head_count_kv")),
        head_dim=int(get("attention.key_length")), intermediate_size=int(get("feed_forward_length")),
        vocab_size=int(get("vocab_size")),
        rms_norm_eps=round(float(get("attention.layer_norm_rms_epsilon")), 12),
        rope_theta=float(get("rope.freq_base")), max_position_embeddings=int(get("context_length")),
        latent_dim=int(get("latent_dim")), max_latent_frames=int(get("max_latent_frames")),
        timestep_shift=float(get("timestep_shift", 1.0)), torch_dtype="bfloat16")


# ---------------------------------------------------------------- loading
@dataclass
class LoadReport:
    path: str
    mode: str
    quantized: list[str] = field(default_factory=list)          # modules kept packed
    dequantized_types: dict[str, int] = field(default_factory=dict)
    packed_types: dict[str, int] = field(default_factory=dict)
    bytes: int = 0                                                 # size of the built model
    eager_required: bool = False

    def summary(self) -> dict:
        return {"path": self.path, "mode": self.mode, "quantized_modules": len(self.quantized),
                "dequantized_types": self.dequantized_types, "packed_types": self.packed_types,
                "model_gib": round(self.bytes / 2**30, 2), "eager_required": self.eager_required}


def _open(path: Path, arch: str) -> tuple[gguf.GGUFReader, dict]:
    reader = gguf.GGUFReader(str(path))
    meta = scalar_metadata(reader)
    found = meta.get("general.architecture")
    if found != arch:
        raise ValueError(f"{path.name}: GGUF-Architektur '{found}', erwartet '{arch}'. "
                         "Unterstützt wird das Layout von HOT-Step-CPP (z.B. scragnog/YuE2-GGUF).")
    return reader, meta


def _keep_packed(module: str, mode: str) -> bool:
    if mode == "max":
        return module == "lm_head" or re.search(r"\.(self_attn|nar_self_attn|mlp|nar_mlp)\.\w+_proj$", module) is not None
    if mode == "balanced":
        return re.search(r"\.(nar_self_attn|nar_mlp)\.\w+_proj$", module) is not None
    return False


def _replace(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent_name) if parent_name else model, child, module)


def _count(counter: dict, qtype: Q) -> None:
    counter[qtype.name] = counter.get(qtype.name, 0) + 1


def _model_bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))


def load_lm(path: Path, mode: str = "balanced"):
    """Build a ``YuE2ForCausalLM`` (CPU) from a GGUF file. Returns (model, meta, report)."""
    from yue2.modeling_yue2 import YuE2ForCausalLM
    if mode not in GGUF_MODES:
        raise ValueError(f"gguf_mode muss eines von {GGUF_MODES} sein")
    path = Path(path)
    reader, meta = _open(path, LM_ARCH)
    with torch.device("meta"):
        model = YuE2ForCausalLM(lm_config(meta))
    report = LoadReport(str(path), mode, eager_required=(mode == "max"))
    state, unknown = {}, []
    for tensor in reader.tensors:
        key = lm_hf_name(tensor.name)
        if key is None:
            unknown.append(tensor.name)
            continue
        qtype = Q(tensor.tensor_type)
        module_name = key.rsplit(".", 1)[0]
        if (key.endswith(".weight") and _keep_packed(module_name, mode)
                and qtype not in FLOAT_TYPES and qtype in dequantize_functions):
            out_features, in_features = torch_shape(tensor)
            packed = torch.from_numpy(np.array(tensor.data))
            _replace(model, module_name, QuantLinear(packed, qtype, in_features, out_features))
            report.quantized.append(module_name)
            _count(report.packed_types, qtype)
        else:
            state[key] = tensor_to_float(tensor, torch.bfloat16)
            _count(report.dequantized_types, qtype)
    if unknown:
        raise ValueError(f"{path.name}: {len(unknown)} unbekannte Tensoren, z.B. {unknown[:3]} – anderes GGUF-Layout?")
    expected = {k for k in model.state_dict() if not k.endswith(".qweight")}
    missing = expected - set(state)
    if missing:
        raise ValueError(f"{path.name}: {len(missing)} Tensoren fehlen, z.B. {sorted(missing)[:3]}")
    model.load_state_dict(state, strict=False, assign=True)
    leftovers = [n for n, t in (*model.named_parameters(), *model.named_buffers()) if t.is_meta]
    if leftovers:
        raise ValueError(f"{path.name}: nicht initialisierte Tensoren {leftovers[:3]}")
    model.eval().requires_grad_(False)
    report.bytes = _model_bytes(model)
    return model, meta, report


def load_vae(path: Path):
    """Build a decoder-only ``YuE2VAE`` (FP32, CPU) from a GGUF file. Returns (model, meta)."""
    from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
    path = Path(path)
    reader, meta = _open(path, VAE_ARCH)
    config = YuE2VAEConfig(release_variant=meta.get("yue2vae.variant", "standard"))
    model = YuE2VAE(config, decoder_only=True)
    state = {}
    for tensor in reader.tensors:
        key = vae_hf_name(tensor.name)
        if key is None:
            continue  # encoder tensors are not needed for decoding
        value = tensor_to_float(tensor, torch.float32)
        if key.endswith(".weight"):  # folded weight norm → weight_g / weight_v (exact inverse)
            base = key[: -len(".weight")]
            norm = value.flatten(1).norm(dim=1).reshape(-1, *([1] * (value.ndim - 1)))
            state[f"{base}.weight_g"], state[f"{base}.weight_v"] = norm, value
        else:
            state[key] = value
    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError(f"{path.name}: VAE-Tensoren passen nicht (fehlend {sorted(expected - set(state))[:3]}, "
                         f"unbekannt {sorted(set(state) - expected)[:3]})")
    model.load_state_dict(state, strict=True)
    model.to(dtype=torch.float32).eval().requires_grad_(False)
    return model, meta


# ---------------------------------------------------------------- pipeline skeleton
def _placeholder_safetensors(target: Path) -> None:
    from safetensors.torch import save_file
    save_file({"gguf_placeholder": torch.zeros(1)}, str(target),
              metadata={"note": "Weights are loaded from a GGUF file by Music-Gen-UI"})


def make_lm_skeleton(gguf_path: Path, directory: Path) -> Path:
    """Directory YuE2Pipeline accepts as ``model_dir``: config, tokenizer, placeholder weights.

    The real weights are injected afterwards (``pipe._model``)."""
    directory.mkdir(parents=True, exist_ok=True)
    reader, meta = _open(Path(gguf_path), LM_ARCH)
    config = lm_config(meta)
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    write_tiktoken(reader, directory / "qwen.tiktoken")
    _placeholder_safetensors(directory / "placeholder.safetensors")
    return directory


def make_vae_skeleton(gguf_path: Path, directory: Path) -> Path:
    from yue2.modeling_vae import YuE2VAEConfig
    directory.mkdir(parents=True, exist_ok=True)
    _, meta = _open(Path(gguf_path), VAE_ARCH)
    config = YuE2VAEConfig(release_variant=meta.get("yue2vae.variant", "standard"))
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    _placeholder_safetensors(directory / "placeholder.safetensors")
    return directory
