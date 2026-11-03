"""Adapter for the official ``yue2`` package (YuE2Pipeline).

``load()`` downloads (first run) and verifies the weights via
``YuE2Pipeline.from_pretrained``; YuE2 moves them to the GPU on the first
generation. The pipeline stays resident until ``unload()``. YuE2 runs one
request at a time, which the engine manager guarantees.

Compatibility: the ``yue2`` API differs between the release tag
``yue2-v0.1.6`` and later main-branch commits (both report version 0.1.6).
The release has ``offload_ar`` but no ``low_vram`` and no ``on_progress``;
newer commits have both. :func:`detect_api` inspects the installed
package and only passes what it accepts.
"""
from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from ..catalog import model_format
from ..config import Settings
from ..download import DownloadMonitor
from ..memory import GIB, free_cuda_cache
from ..prompting import CompiledRequest
from .base import Cancelled, CancelledFn, EngineResult, ProgressFn

INSTALL_HINT = "Paket 'yue2' fehlt – ./install.sh ausführen (siehe README)"
SEMANTIC_HZ = 25
MODEL_RAM_FALLBACK_GIB = 7.5   # YuE2-3B in BF16, used until the real file size is known
# Rough host-RAM need of a GGUF model before it is loaded (exact value afterwards).
GGUF_RAM_ESTIMATE_GIB = {"bf16": 7.5, "balanced": 6.0, "max": 3.5}
OFFLOAD_VAE_FRAMES = 512        # YuE2's own default for small memory budgets
# GPU sharing: (attention query block, pause every n NAR layers, pause in ms)
GPU_SHARE = {"balanced": (4096, 4, 1.0), "smooth": (2048, 2, 3.0)}


@dataclass(frozen=True)
class YuE2Api:
    version: str
    low_vram: bool        # YuE2Pipeline(low_vram=...) – newer commits
    offload_ar: bool      # YuE2Pipeline(offload_ar=...) – release 0.1.6
    stage_progress: bool  # pipe(..., on_progress=...) – newer commits
    init_params: frozenset


def _named_params(fn) -> frozenset:
    return frozenset(name for name, p in inspect.signature(fn).parameters.items()
                     if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL))


def detect_api(pipeline_cls) -> YuE2Api:
    try:
        from importlib.metadata import version
        ver = version("yue2-infer")
    except Exception:  # noqa: BLE001 - fake/dev installs
        ver = "unbekannt"
    init = _named_params(pipeline_cls.__init__)
    call = _named_params(pipeline_cls.__call__)
    return YuE2Api(ver, "low_vram" in init, "offload_ar" in init, "on_progress" in call, init)


class YuE2Engine:
    name = "yue2"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self.api: YuE2Api | None = None
        self.gguf_report: dict | None = None
        self.lora_state = None          # musicgen_ui.lora.LoraState once a LoRA was used
        self._adapters: dict = {}       # path -> (mtime, Adapter)
        self._source = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def info(self) -> dict:
        s = self.settings
        return {"engine": self.name, "model": s.model, "vae": s.vae, "device": s.device,
                "low_vram": s.low_vram, "quantization": s.quantization, "backend": s.backend,
                "loaded": self.loaded, "low_vram_mode": self._low_vram_mode(),
                "offload": s.offload, "idle_placement": s.idle_placement,
                "gpu_share": s.gpu_share,
                "gguf": self.uses_gguf, "gguf_mode": s.gguf_mode if self.uses_gguf else None,
                "gguf_report": self.gguf_report,
                "loras": list(self.lora_state.signature) if self.lora_state else [],
                "memory_budget_gib": self._memory_budget(),
                "yue2": None if self.api is None else
                {k: v for k, v in asdict(self.api).items() if k != "init_params"}}

    def _low_vram_mode(self) -> str | None:
        """How the offload setting is realised by the installed YuE2."""
        if not self.settings.offload:
            return None
        if self.api is None or self.api.low_vram:
            return "low_vram"
        return "offload_ar" if self.api.offload_ar else "unsupported"

    def _pipeline_kwargs(self) -> dict:
        s = self.settings
        kwargs = {"device": s.device, "quantization": s.quantization, "backend": s.backend,
                  "memory_budget_gib": self._memory_budget()}
        mode = self._low_vram_mode()
        if mode in ("low_vram", "offload_ar"):
            kwargs[mode] = True
            kwargs["vae_core_frames"] = OFFLOAD_VAE_FRAMES  # decode in smaller pieces
        return {k: v for k, v in kwargs.items() if k in self.api.init_params}

    def _memory_budget(self) -> float:
        """YuE2 caps PyTorch at ``budget - 2 GiB``; derive the budget from our VRAM reserve."""
        s = self.settings
        budget = s.memory_budget_gib
        total = _cuda_total_gib() if s.device in ("auto", "cuda") else None
        if total:
            budget = min(budget, total - max(s.vram_reserve_gib, 2.0) + 2.0)
        return round(budget, 2)

    def _model_gib(self) -> float:
        if self.gguf_report is not None and self._pipe is not None:
            return self.gguf_report["model_gib"]
        if model_format(self.settings.model) == "gguf":
            return GGUF_RAM_ESTIMATE_GIB.get(self.settings.gguf_mode, MODEL_RAM_FALLBACK_GIB)
        directory = getattr(self._pipe, "model_dir", None)
        if directory is not None:
            size = sum(f.stat().st_size for f in Path(directory).glob("*.safetensors"))
            if size:
                return size / GIB
        return MODEL_RAM_FALLBACK_GIB

    def _model_device(self) -> str | None:
        model = getattr(self._pipe, "_model", None)
        if model is None:
            return None
        try:
            return next(model.parameters()).device.type
        except (StopIteration, AttributeError):
            return None

    def ram_needed_gib(self) -> float:
        # Loading goes through host RAM, and YuE2 parks the model in RAM for decoding.
        # If it already rests in RAM, moving it to the GPU frees that space again.
        return 0.0 if self._model_device() == "cpu" else self._model_gib()

    def after_job(self, placement: str) -> None:
        if self._pipe is None:
            return
        if placement == "unload":
            self.unload()
            return
        target = "cpu" if placement == "ram" else getattr(self._pipe, "device", None)
        for name in ("_model", "_vae"):
            module = getattr(self._pipe, name, None)
            if module is not None and target is not None and (name == "_model" or target == "cpu"):
                module.to(target)  # the VAE always rests in RAM; it is small and only used briefly
        free_cuda_cache()

    def load(self, progress: ProgressFn) -> None:
        if self._pipe is not None:
            return
        try:
            from yue2 import YuE2Pipeline
        except ImportError as exc:
            raise RuntimeError(INSTALL_HINT) from exc
        s = self.settings
        self.api = detect_api(YuE2Pipeline)
        if self.uses_gguf:
            self._load_gguf(YuE2Pipeline, progress)
            return
        progress("download", 0, 0)
        with DownloadMonitor([s.model, s.vae], lambda size: progress("download", size, 0)):
            self._pipe = YuE2Pipeline.from_pretrained(s.model, vae=s.vae, progress=False,
                                                      **self._pipeline_kwargs())

    # -- GGUF ------------------------------------------------------------------
    @property
    def uses_gguf(self) -> bool:
        return model_format(self.settings.model) == "gguf" or model_format(self.settings.vae) == "gguf"

    def _skeleton_dir(self, path: Path, kind: str) -> Path:
        stat = path.stat()
        return self.settings.data_dir / "cache" / "gguf" / f"{kind}-{path.stem}-{stat.st_size}-{int(stat.st_mtime)}"

    def _load_gguf(self, pipeline_cls, progress: ProgressFn) -> None:
        """Official pipeline from small skeleton dirs, real weights injected from GGUF."""
        try:
            from ..gguf_support import loader
            from yue2.storage import resolve_model
        except ImportError as exc:
            raise RuntimeError("GGUF braucht die Pakete 'gguf' und 'yue2' – ./install.sh erneut ausführen") from exc
        s = self.settings
        if s.backend == "vllm":
            raise RuntimeError("GGUF-Modelle laufen nicht mit dem vllm-Backend – unter ⚙ Einstellungen 'torch' wählen")
        lm_gguf, vae_gguf = model_format(s.model) == "gguf", model_format(s.vae) == "gguf"
        if lm_gguf and s.gguf_mode == "max" and s.quantization == "fp8":
            raise RuntimeError("GGUF-Modus 'Maximal' und FP8 schließen sich aus – einen davon abschalten")
        for value in (s.model, s.vae):
            if model_format(value) == "gguf" and not Path(value).is_file():
                raise RuntimeError(f"GGUF-Datei nicht gefunden: {value}")

        progress("download", 0, 0)
        with DownloadMonitor([v for v in (s.model, s.vae) if model_format(v) != "gguf"],
                             lambda size: progress("download", size, 0)):
            model_dir = (loader.make_lm_skeleton(Path(s.model), self._skeleton_dir(Path(s.model), "lm"))
                         if lm_gguf else resolve_model(s.model))
            vae_dir = (loader.make_vae_skeleton(Path(s.vae), self._skeleton_dir(Path(s.vae), "vae"))
                       if vae_gguf else resolve_model(s.vae))
        kwargs = {k: v for k, v in self._pipeline_kwargs().items() if k in self.api.init_params}
        if lm_gguf and s.gguf_mode == "max" and kwargs.get("backend", "torch") == "torch":
            kwargs["backend"] = "torch-eager"  # CUDA graphs need plain BF16 AR layers
        pipe = pipeline_cls(model_dir, vae_dir, progress=False, **kwargs)
        try:
            if lm_gguf:
                progress("gguf", 0, 0)
                model, _meta, report = loader.load_lm(Path(s.model), s.gguf_mode)
                pipe._model = model  # YuE2 skips its own loader when the model is already set
                self.gguf_report = report.summary()
            if vae_gguf:
                progress("gguf", 1, 0)
                pipe._vae, _meta = loader.load_vae(Path(s.vae))
        except Exception:
            pipe.close()
            raise
        self._pipe = pipe

    @contextmanager
    def _gpu_share(self):
        """Give the desktop/browser GPU time during sound synthesis.

        YuE2 runs the NAR attention over the whole ~24k-token sequence in one
        kernel per layer; on a GPU that also drives the display, the desktop
        starves and browsers' GPU watchdogs kill them. We split the attention
        into query blocks (YuE2's own ``query_chunk_size``) and pause briefly
        every few NAR layers. The AR phase (CUDA graphs) is not touched."""
        level = GPU_SHARE.get(self.settings.gpu_share)
        if level is None or self.settings.device == "cpu" or self.settings.backend == "vllm":
            yield
            return
        model = getattr(self._pipe, "_model", None)
        if model is None and hasattr(self._pipe, "_load_model"):
            model = self._pipe._load_model(for_nar=True)  # YuE2 would load it lazily inside the song
        import time
        import torch
        try:
            import yue2.nar as nar
        except ImportError:  # other yue2 versions/stand-ins: only the pauses apply
            nar = None
        block, every, pause_ms = level
        original = getattr(nar, "attention", None)

        def chunked(q, k, v, *args, query_chunk_size=None, **kwargs):
            return original(q, k, v, *args, query_chunk_size=query_chunk_size or block, **kwargs)

        counter = {"n": 0}

        def breather(_module, _inputs):
            counter["n"] += 1
            if counter["n"] % every == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                time.sleep(pause_ms / 1000)

        handles = []
        if original is not None:
            nar.attention = chunked
        try:
            if model is not None:
                handles = [layer.nar_mlp.register_forward_pre_hook(breather)
                           for layer in model.model.layers if hasattr(layer, "nar_mlp")]
            yield
        finally:
            if original is not None:
                nar.attention = original
            for handle in handles:
                handle.remove()

    def _mark_phases(self, progress: ProgressFn) -> dict:
        pipe, wrapped = self._pipe, {}
        for name, stage in (("synthesize", "nar"), ("decode", "vae")):
            original = getattr(pipe, name, None)
            if original is None:
                continue

            def marked(*args, _original=original, _stage=stage, **kwargs):
                progress(_stage, 0, 0)
                return _original(*args, **kwargs)
            setattr(pipe, name, marked)  # instance attribute, removed after the song
            wrapped[name] = original
        return wrapped

    # -- LoRA --------------------------------------------------------------------
    def _adapter(self, path: str, dims: dict):
        from ..lora import read_adapter
        file = Path(path)
        if not file.is_file():
            raise RuntimeError(f"LoRA nicht gefunden: {path}")
        key = (file.stat().st_mtime, tuple(sorted(dims.items())))
        cached = self._adapters.get(path)
        if cached is None or cached[0] != key:
            cached = (key, read_adapter(file, dims))
            self._adapters[path] = cached
        return cached[1]

    def _apply_loras(self, refs: list[dict], progress: ProgressFn) -> None:
        if not refs and (self.lora_state is None or not self.lora_state.signature):
            return
        if self.settings.backend == "vllm":
            raise RuntimeError("LoRAs funktionieren nicht mit dem vllm-Backend – unter ⚙ Einstellungen 'torch' wählen")
        from ..lora import LoraState, gguf_source, hf_source
        pipe = self._pipe
        model = pipe._model if getattr(pipe, "_model", None) is not None else pipe._load_model(for_nar=True)
        if self._source is None:
            self._source = (gguf_source(Path(self.settings.model)) if model_format(self.settings.model) == "gguf"
                            else hf_source(Path(pipe.model_dir)))
        self.lora_state = self.lora_state or LoraState()
        c = model.config
        dims = {"q": c.num_attention_heads * c.head_dim, "kv": c.num_key_value_heads * c.head_dim,
                "ff": c.intermediate_size}
        adapters = [(self._adapter(ref["path"], dims), float(ref.get("strength", 1.0))) for ref in refs]
        progress("lora", 0, 0)
        self.lora_state.apply(model, adapters, self._source)

    def unload(self) -> None:
        pipe, self._pipe = self._pipe, None
        self.gguf_report = None
        self.lora_state, self._source = None, None
        if pipe is not None:
            pipe.close()  # drops model/VAE and empties the CUDA cache
            del pipe
            free_cuda_cache()

    def generate(self, request: CompiledRequest, out_dir: Path, *,
                 progress: ProgressFn, cancelled: CancelledFn) -> EngineResult:
        self.load(progress)
        names = {"abc": "score", "semantic": "audio"}
        counts: dict[str, int] = {}

        def on_token(phase, _token):
            counts[phase] = counts.get(phase, 0) + 1
            if counts[phase] % SEMANTIC_HZ == 0:  # ~ once per second of music
                progress(names.get(phase, phase), counts[phase], 0)

        self._apply_loras(request.loras or [], progress)
        call = {**request.engine_kwargs(), "cancelled": cancelled, "on_token": on_token}
        markers = {}
        if self.api.stage_progress:
            call["on_progress"] = lambda stage, done, total: progress(stage, done, total)
        else:
            # The release reports no synthesis/decoding progress: mark the phase changes
            # (the memory log then shows exactly where a crash happened).
            markers = self._mark_phases(progress)
        try:
            with self._gpu_share():
                song = self._pipe(**call)
        except InterruptedError as exc:
            raise Cancelled() from exc
        finally:
            for name in markers:
                self._pipe.__dict__.pop(name, None)
        result = song.save_artifacts(out_dir)
        score = out_dir / "score.abc"
        return EngineResult(
            audio_path=out_dir / "audio.flac", sample_rate=song.sample_rate,
            duration_s=float(result["audio_seconds"]), truncated=bool(result["truncated"]),
            score_abc=score.read_text(encoding="utf-8") if score.exists() else None,
            extra={"timing": result.get("timing"), "weights": result.get("weights")})


def _cuda_total_gib() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / GIB
