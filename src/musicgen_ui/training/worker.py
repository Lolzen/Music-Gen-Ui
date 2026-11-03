"""Training worker (own process, so the GPU is fully released afterwards).

    python -m musicgen_ui.training.worker <project_dir> --model <id|dir|gguf> --device cuda [--fake]

Writes ``status.json`` after every step and appends to ``train.log``. Creating
the file ``STOP`` in the project directory ends the run after the current step
(the last state is saved as ``last.safetensors``).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import pipeline as P
from .project import ProjectStore


class Status:
    def __init__(self, directory: Path):
        self.path = directory / "status.json"
        self.data = {"phase": "start", "started": time.time(), "evals": [], "losses": [], "checkpoints": []}

    def update(self, **values) -> None:
        self.data.update(values, updated=time.time())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {message}"
        print(line, flush=True)


# ---------------------------------------------------------------- components
class RealComponents:
    def __init__(self, args, project_dir: Path):
        self.args, self.dir = args, project_dir

    def _model_dir(self) -> Path | None:
        from yue2.storage import resolve_model
        model = self.args.model
        if model.lower().endswith(".gguf"):
            return None
        return Path(resolve_model(model))

    def tokenizer(self):
        from yue2.tokenization_yue2 import YuE2TextTokenizer
        model_dir = self._model_dir()
        if model_dir is not None:
            return YuE2TextTokenizer(model_dir / "qwen.tiktoken")
        import gguf
        from ..gguf_support.loader import write_tiktoken
        target = self.dir / "cache" / "qwen.tiktoken"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            write_tiktoken(gguf.GGUFReader(self.args.model), target)
        return YuE2TextTokenizer(target)

    def features(self):
        return P.MertFeatures(self.args.device)

    def head(self, version: str):
        from huggingface_hub import hf_hub_download
        return P.load_head(Path(hf_hub_download(P.TOKENIZER_REPO, P.HEADS[version])), self.args.device)

    def regularizer(self) -> list[dict]:
        from huggingface_hub import hf_hub_download
        repo, filename = P.REGULARIZER
        return P.safe_torch_load(Path(hf_hub_download(repo, filename, repo_type="dataset")))

    def base_model(self):
        import torch
        model_dir = self._model_dir()
        if model_dir is None:
            from ..gguf_support.loader import load_lm
            model = load_lm(Path(self.args.model), "bf16")[0]
        else:
            from yue2.modeling_yue2 import YuE2ForCausalLM
            model = YuE2ForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True)
        return model.eval().to(self.args.device)


class FakeComponents:
    """Tiny random stand-ins so the whole flow can be tested on a CPU."""

    def __init__(self, args, project_dir: Path):
        self.args = args

    def tokenizer(self):
        class Tok:
            def encode(self, text):
                return [10 + b % 1000 for b in text.encode("utf-8")][:200]
        return Tok()

    def features(self):
        rng = np.random.default_rng(0)
        return lambda mono24: rng.standard_normal((int(round(len(mono24) / 24000 * P.FRAME_HZ)), 1024)).astype(np.float16)

    def head(self, version: str):
        import torch
        torch.manual_seed(0)
        return P.TokenizerHead().eval()

    def regularizer(self) -> list[dict]:
        rng = np.random.default_rng(1)
        return [{"name": f"m{i}", "src": "minted_val" if i < 2 else "minted", "style": "pop", "lyrics": "[Verse]\nla",
                 "codec": rng.integers(0, P.VOCAB, 40).astype(np.int32)} for i in range(6)]

    def base_model(self):
        import torch
        from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
        torch.manual_seed(0)
        return YuE2ForCausalLM(YuE2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                                          num_key_value_heads=2, head_dim=16, intermediate_size=128,
                                          max_latent_frames=64)).to(torch.bfloat16).eval()


# ---------------------------------------------------------------- run
def run(project_dir: Path, args) -> int:
    store = ProjectStore(project_dir.parent)
    project = store.load(project_dir.name)
    status = Status(project_dir)
    stop_file = project_dir / "STOP"
    stop_file.unlink(missing_ok=True)
    comps = (FakeComponents if args.fake else RealComponents)(args, project_dir)
    config = P.TrainConfig(**project.settings)
    if args.steps:
        config.steps = args.steps
    status.update(phase="prep", message="Songs werden analysiert (MERT)", songs=len(project.songs), song=0,
                  steps=config.steps, step=0, project=project.id, settings=config.to_dict())
    try:
        tokenizer = comps.tokenizer()
        cache = project_dir / "cache"
        extractor = None
        for index, song in enumerate(project.songs, 1):
            target = cache / Path(song.file).stem / "mert.npy"
            if not target.exists():
                extractor = extractor or comps.features()
                audio, rate = P.load_audio(project_dir / "songs" / song.file)
                mono = P.resample(audio.mean(1), rate, 24000)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, extractor(mono))
                status.log(f"MERT: {song.title} ({len(mono) / 24000:.0f} s)")
            status.update(song=index, message=f"Analysiert: {song.title}")
            if stop_file.exists():
                raise KeyboardInterrupt
        del extractor
        _free()

        status.update(phase="tokenize", message=f"Tokens mit Encoder {project.head}", song=0)
        head = None
        artist = []
        for index, song in enumerate(project.songs, 1):
            codes_file = cache / Path(song.file).stem / f"codes-{project.head}.npy"
            if not codes_file.exists():
                head = head or comps.head(project.head)
                features = np.load(cache / Path(song.file).stem / "mert.npy")
                np.save(codes_file, P.predict_codes(head, features, args.device if not args.fake else "cpu"))
            codes = np.load(codes_file)
            style = P.artist_style(project.trigger, song.description or project.description)
            artist.append(P.Item(song.file, "artist", P.build_prefix(style, song.lyrics.strip(), tokenizer), codes))
            status.update(song=index, message=f"Tokens: {song.title} ({len(codes)} Frames)")
        del head
        _free()

        minted, minted_val = [], []
        if project.use_regularizer:
            status.update(phase="dataset", message="Regularisierungs-Paket wird geladen (~100 MB beim ersten Mal)")
            items = P.regularizer_items(comps.regularizer(), tokenizer)
            minted = [i for i in items if i.src == "minted"]
            minted_val = [i for i in items if i.src == "minted_val"]
            status.log(f"Regularisierung: {len(minted)} Songs (+{len(minted_val)} zur Kontrolle)")

        status.update(phase="load", message="Basismodell wird geladen")
        model = comps.base_model()
        started = time.time()
        meta = {"name": project.name, "trigger": project.trigger, "rank": config.rank, "head": project.head,
                "base_model": args.model, "format": "musicgen-ui-artist-lora-v1", "trained_halves": "ar",
                "recommended_plan_mode": "off"}

        def on_step(step, loss, length):
            losses = status.data["losses"][-400:] + [[step, round(loss, 4)]]
            per_step = (time.time() - started) / step
            status.update(phase="train", step=step, loss=round(loss, 4), losses=losses, seq_len=length,
                          eta_s=round(per_step * (config.steps - step)),
                          message=f"Schritt {step}/{config.steps}")
            if step <= 3 or step % 20 == 0:
                status.log(f"Schritt {step}: Loss {loss:.3f}, Länge {length}")

        def on_eval(step, losses):
            status.update(evals=status.data["evals"] + [{"step": step, **{k: round(v, 4) for k, v in losses.items()}}])
            status.log(f"Auswertung Schritt {step}: " + ", ".join(f"{k} {v:.3f}" for k, v in losses.items()))

        def on_checkpoint(step, path, kind):
            entries = [c for c in status.data["checkpoints"] if c["file"] != str(path)]
            status.update(checkpoints=entries + [{"step": step, "file": str(path), "kind": kind}])

        hooks = P.TrainHooks(on_step=on_step, on_eval=on_eval, on_checkpoint=on_checkpoint,
                             should_stop=stop_file.exists, log=status.log)
        status.update(phase="train", message="Training startet")
        result = P.train_lora(model, artist, minted, minted_val, config, project_dir / "checkpoints", meta, hooks)
        status.update(phase="stopped" if result["stopped"] else "done", result=result,
                      message="Gestoppt" if result["stopped"] else "Fertig – Checkpoints anhören und den besten übernehmen")
        return 0
    except KeyboardInterrupt:
        status.update(phase="stopped", message="Gestoppt")
        return 0
    except Exception as exc:  # noqa: BLE001 - report every failure to the UI
        status.log(traceback.format_exc())
        hint = " → Grafikspeicher voll: Rang oder Songlänge reduzieren, andere GPU-Programme schließen." \
            if "out of memory" in str(exc).lower() else ""
        status.update(phase="error", message=f"{type(exc).__name__}: {exc}{hint}")
        return 1


def _free() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Music-Gen-UI LoRA-Training")
    parser.add_argument("project_dir")
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=0, help="Schritte überschreiben (Tests)")
    parser.add_argument("--fake", action="store_true", help="winzige Zufallsmodelle (Tests ohne GPU)")
    args = parser.parse_args(argv)
    return run(Path(args.project_dir), args)


if __name__ == "__main__":
    sys.exit(main())
