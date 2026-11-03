"""Artist-LoRA training for YuE2's composition (AR) half.

Re-implementation of the workflow published with
https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4
(CC BY-NC 4.0; the tokenizer head and the regularizer pack are downloaded from
there at runtime, nothing of it is shipped here):

1. audio → MERT-v2-FullSong layer-20 features at 25 Hz
2. features → YuE2 semantic codes with the community "tokenizer head"
3. dataset: artist songs (style caption with trigger + full lyrics, score-free
   ``cot=off`` prefix) mixed 50/50 with YuE2's own songs (regularizer pack)
4. rank-64 LoRA on the AR attention/MLP projections, cross-entropy on the codes

The resulting files use the Mothersuperior safetensors layout
(``layers.N.self_attn.q_proj.lora_A`` …), so ``musicgen_ui.lora`` loads them.
"""
from __future__ import annotations

import importlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (COMPANION_NAR, FRAME_HZ, HEADS, MERT_REPO, REGULARIZER,  # noqa: F401 (re-export)
                     TOKENIZER_REPO, VOCAB, TrainConfig)

AR_TARGETS = (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("mlp", ("gate_proj", "up_proj", "down_proj")))


# ---------------------------------------------------------------- audio + features
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, rate


def resample(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return audio.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(rate, target)
    return resample_poly(audio, target // g, rate // g, axis=0).astype(np.float32)


class MertFeatures:
    """MERT-v2-FullSong hidden layer 20 (1024-dim), interpolated to 25 Hz."""

    def __init__(self, device: str):
        from transformers import AutoFeatureExtractor, AutoModel
        self.device = device
        self.proc = AutoFeatureExtractor.from_pretrained(MERT_REPO, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MERT_REPO, trust_remote_code=True).to(device).eval()

    @torch.inference_mode()
    def __call__(self, mono24: np.ndarray) -> np.ndarray:
        chunk = 24000 * 30
        pieces = [mono24[s:s + chunk] for s in range(0, len(mono24), chunk)]
        pieces = [p for p in pieces if len(p) >= 24000]
        feats = []
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else _null()
        with autocast:
            for piece in pieces:
                inputs = {k: v.to(self.device) for k, v in
                          self.proc([piece], sampling_rate=24000, return_tensors="pt").items()}
                hidden = self.model(**inputs, output_hidden_states=True).hidden_states[20]
                feats.append(hidden.reshape(-1, hidden.shape[-1]).float())
        features = torch.cat(feats, 0)
        frames = int(round(len(mono24) / 24000 * FRAME_HZ))
        return F.interpolate(features.T[None], size=frames, mode="linear", align_corners=False)[0].T.half().cpu().numpy()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------- tokenizer head
class TokenizerHead(nn.Module):
    """8-layer transformer: MERT features → YuE2 semantic codes (layout of the published head)."""

    WINDOW, DIM, LAYERS, HEADS = 512, 512, 8, 8

    def __init__(self, din: int = 1024, vocab: int = VOCAB):
        super().__init__()
        self.inp = nn.Linear(din, self.DIM)
        self.pos = nn.Parameter(torch.zeros(1, self.WINDOW, self.DIM))
        layer = nn.TransformerEncoderLayer(self.DIM, self.HEADS, 4 * self.DIM, dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.DIM)
        self.head = nn.Linear(self.DIM, vocab)

    def forward(self, x):
        return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]])))


def load_head(path: Path, device: str) -> TokenizerHead:
    from safetensors.torch import load_file
    state = load_file(str(path))
    head = TokenizerHead(state["inp.weight"].shape[1], state["head.weight"].shape[0])
    head.load_state_dict({k: v.float() for k, v in state.items()})
    return head.to(device).eval()


def instance_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean(0)) / (x.std(0) + 1e-5)


@torch.no_grad()
def predict_codes(head: TokenizerHead, features: np.ndarray, device: str) -> np.ndarray:
    """Overlapping 512-frame windows, keeping the central half of each (as published)."""
    x, win = instance_norm(features), head.WINDOW
    total = len(x)
    out = np.zeros(total, dtype=np.int64)
    starts = list(range(0, max(1, total - win + 1), win // 2))
    if starts[-1] + win < total:
        starts.append(max(0, total - win))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else _null()
    for s0 in starts:
        xw = x[s0:s0 + win]
        n = len(xw)
        if n < win:
            xw = np.pad(xw, ((0, win - n), (0, 0)))
        with autocast:
            pred = head(torch.tensor(xw[None], device=device))[0, :n].float().argmax(-1).cpu().numpy()
        lo = s0 + (0 if s0 == 0 else win // 4)
        hi = s0 + n - (0 if s0 + n >= total else win // 4)
        out[lo:hi] = pred[lo - s0:hi - s0]
    return out


# ---------------------------------------------------------------- data
def artist_style(trigger: str, description: str) -> str:
    text = f"{trigger}, in the style of {trigger}. {description}".strip()
    return " ".join(text.split())[:1500]


def build_prefix(style: str, lyrics: str, tokenizer) -> list[int]:
    from yue2.protocol import SongRequest, token_prefixes
    return token_prefixes(SongRequest(style=style, lyrics=lyrics or "[instrumental]", cot="off", seed=1, id="train"),
                          tokenizer)


def safe_torch_load(path: Path):
    """torch.load without arbitrary unpickling: only numpy arrays/dtypes are allowed besides plain data."""
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
    allowed = []
    for name in unsafe:
        if not name.startswith("numpy."):
            raise ValueError(f"{Path(path).name} enthält nicht erlaubte Objekte ({name}) – Laden abgebrochen")
        module, _, attr = name.rpartition(".")
        allowed.append(getattr(importlib.import_module(module), attr))
    # numpy arrays also reference their dtype classes, which are not always listed above
    allowed += [np.ndarray, np.dtype] + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(str(path), map_location="cpu", weights_only=True)


@dataclass
class Item:
    name: str
    src: str              # artist | minted | minted_val
    prefix: list[int]
    codec: np.ndarray


def regularizer_items(pack: list[dict], tokenizer, limit: int | None = None) -> list[Item]:
    items = []
    for record in pack[:limit] if limit else pack:
        items.append(Item(record["name"], record["src"], build_prefix(record["style"], record["lyrics"], tokenizer),
                          np.asarray(record["codec"], dtype=np.int64)))
    return items


# ---------------------------------------------------------------- LoRA training
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        device = base.weight.device
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=device))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


@dataclass
class TrainHooks:
    on_step: Callable[[int, float, int], None] = lambda step, loss, length: None
    on_eval: Callable[[int, dict], None] = lambda step, losses: None
    on_checkpoint: Callable[[int, Path, str], None] = lambda step, path, kind: None
    should_stop: Callable[[], bool] = lambda: False
    log: Callable[[str], None] = print


def wrap_lora(model, rank: int) -> list[tuple[str, LoRALinear]]:
    wrapped = []
    for i, layer in enumerate(model.model.layers):
        for block_name, names in AR_TARGETS:
            block = getattr(layer, block_name)
            for name in names:
                module = LoRALinear(getattr(block, name), rank)
                setattr(block, name, module)
                wrapped.append((f"layers.{i}.{block_name}.{name}", module))
    return wrapped


def save_lora(wrapped, path: Path, meta: dict) -> None:
    from safetensors.torch import save_file
    tensors = {}
    for name, module in wrapped:
        tensors[f"{name}.lora_A"] = module.A.detach().float().cpu().contiguous()
        tensors[f"{name}.lora_B"] = module.B.detach().float().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={k: str(v) for k, v in meta.items()})
    tmp.replace(path)


def _sequence(item: Item, max_len: int, device) -> tuple[torch.Tensor, int]:
    from yue2.protocol import CODEC_OFFSET, MUSIC_END
    prefix = list(item.prefix)
    codes = [int(c) + CODEC_OFFSET for c in item.codec]
    room = max_len - len(prefix) - 1
    body = codes[:room] + ([MUSIC_END] if len(codes) <= room else [])
    return torch.tensor([prefix + body], device=device), len(prefix)


def _ar_layer(layer, x, cos, sin):
    from yue2.nar import attention
    q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
    h = attention(q[0], k[0], v[0], causal=True)
    x = x + layer.self_attn.o_proj(h.flatten(1)[None])
    return x + layer.mlp(layer.post_attention_layernorm(x))


def lm_loss(model, ids: torch.Tensor, prefix_len: int, chunk: int, grad: bool) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint
    backbone = model.model
    x = backbone.embed_tokens(ids)
    cos, sin = backbone.rotary_emb(torch.arange(ids.shape[1], device=ids.device)[None])
    for layer in backbone.layers:
        x = checkpoint(_ar_layer, layer, x, cos, sin, use_reentrant=False) if grad else _ar_layer(layer, x, cos, sin)
    hidden = backbone.norm(x[0])[prefix_len - 1:-1]
    target = ids[0, prefix_len:]
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, hidden.shape[0], chunk):
        logits = model.lm_head(hidden[s:s + chunk]).float()
        total = total + F.cross_entropy(logits, target[s:s + chunk], reduction="sum")
    return total / hidden.shape[0]


def train_lora(model, artist: list[Item], minted: list[Item], minted_val: list[Item], config: TrainConfig,
               out_dir: Path, meta: dict, hooks: TrainHooks | None = None) -> dict:
    """Train the AR LoRA in place on ``model`` (YuE2ForCausalLM, frozen BF16)."""
    hooks = hooks or TrainHooks()
    if not artist:
        raise ValueError("Keine Songs im Datensatz")
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    device = next(model.parameters()).device
    model.requires_grad_(False)
    wrapped = wrap_lora(model, config.rank)
    params = [p for _, m in wrapped for p in (m.A, m.B)]
    optimizer = torch.optim.AdamW(params, lr=config.lr, betas=(0.9, 0.95), weight_decay=0.0)
    hooks.log(f"LoRA: {sum(p.numel() for p in params) / 1e6:.1f} M Parameter, Rang {config.rank}, "
              f"{len(artist)} eigene Songs, {len(minted)} Regularisierungs-Songs")
    pool = minted or artist  # without a regularizer every step uses artist songs

    def evaluate() -> dict:
        result = {}
        for tag, items in (("minted_val", minted_val[:6]), ("artist", artist[:6])):
            if items:
                with torch.no_grad():
                    losses = [lm_loss(model, *_sequence(it, config.max_len, device), config.chunk, grad=False).item()
                              for it in items]
                result[tag] = sum(losses) / len(losses)
        return result

    losses = evaluate()
    hooks.on_eval(0, losses)
    best = losses.get("artist", float("inf"))
    for step in range(1, config.steps + 1):
        if hooks.should_stop():
            hooks.log(f"Gestoppt bei Schritt {step - 1}")
            save_lora(wrapped, out_dir / "last.safetensors", {**meta, "step": step - 1})
            hooks.on_checkpoint(step - 1, out_dir / "last.safetensors", "last")
            return {"stopped": True, "step": step - 1, "best_artist_loss": best}
        warm = min(1.0, step / 50)
        cosine = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(step, config.schedule_steps) / config.schedule_steps))
        for group in optimizer.param_groups:
            group["lr"] = config.lr * warm * cosine
        total, length = 0.0, 0
        for _ in range(config.accumulate):
            item = rng.choice(artist) if rng.random() < config.artist_fraction else rng.choice(pool)
            ids, prefix_len = _sequence(item, config.max_len, device)
            loss = lm_loss(model, ids, prefix_len, config.chunk, grad=True) / config.accumulate
            loss.backward()
            total += loss.item()
            length = ids.shape[1]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        hooks.on_step(step, total, length)
        if step % config.eval_every == 0 or step == config.steps:
            losses = evaluate()
            hooks.on_eval(step, losses)
            if losses.get("artist", float("inf")) < best:
                best = losses["artist"]
                save_lora(wrapped, out_dir / "best.safetensors", {**meta, "step": step})
                hooks.on_checkpoint(step, out_dir / "best.safetensors", "best")
            save_lora(wrapped, out_dir / "last.safetensors", {**meta, "step": step})
        if step >= config.checkpoint_from and (step - config.checkpoint_from) % config.checkpoint_every == 0:
            path = out_dir / f"step-{step}.safetensors"
            save_lora(wrapped, path, {**meta, "step": step})
            hooks.on_checkpoint(step, path, "step")
    return {"stopped": False, "step": config.steps, "best_artist_loss": best}
