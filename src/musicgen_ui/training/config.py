"""Training constants and settings (no torch import, usable by the web server)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

TOKENIZER_REPO = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
HEADS = {"v9": "tokenizer_head_joint_v9.safetensors", "v8": "tokenizer_head_joint_v8.safetensors",
         "v5": "tokenizer_head_joint_v5.safetensors", "v4": "tokenizer_head_joint_v4.safetensors"}
COMPANION_NAR = {"v9": "nar_lora_joint_v9.bf16.safetensors", "v8": "nar_lora_joint_v8.bf16.safetensors",
                 "v5": "nar_lora_joint_v5.bf16.safetensors", "v4": "nar_lora_joint_v4.bf16.safetensors"}
REGULARIZER = ("Mothersuperior/yue2-minted-corpus", "regularizer/minted_regularizer_pack.pt")
MERT_REPO = "m-a-p/MERT-v2-FullSong"
FRAME_HZ = 25
VOCAB = 32768


@dataclass
class TrainConfig:
    steps: int = 1600              # author's advice: past ~1500 steps the model memorises the songs
    rank: int = 64
    lr: float = 1e-4
    artist_fraction: float = 0.5   # share of own songs per step, rest from the regularizer pack
    schedule_steps: int = 3000
    checkpoint_from: int = 600
    checkpoint_every: int = 200
    eval_every: int = 100
    accumulate: int = 2
    max_len: int = 12288
    chunk: int = 1024
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
