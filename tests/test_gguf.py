"""GGUF support against the real yue2 release code (CPU, tiny random models)."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
pytest.importorskip("yue2")

from gguf import GGMLQuantizationType as Q  # noqa: E402

from musicgen_ui.gguf_support import loader  # noqa: E402
from musicgen_ui.gguf_support.dequant import dequantize, dequantize_functions  # noqa: E402

# byte offsets of the fp16 scale fields per block (must hold finite values)
FP16_FIELDS = {Q.Q8_0: [0], Q.Q4_0: [0], Q.Q4_1: [0, 2], Q.Q5_0: [0], Q.Q5_1: [0, 2], Q.Q2_K: [80, 82],
               Q.Q3_K: [108], Q.Q4_K: [0, 2], Q.Q5_K: [0, 2], Q.Q6_K: [208], Q.IQ4_NL: [0], Q.IQ4_XS: [0]}


def random_blocks(qtype, n_blocks, rng):
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, size=(n_blocks, size), dtype=np.uint8)
    for offset in FP16_FIELDS[qtype]:
        scales = rng.uniform(0.001, 0.05, n_blocks).astype(np.float16)
        raw[:, offset:offset + 2] = scales.view(np.uint8).reshape(n_blocks, 2)
    return raw, block


@pytest.mark.parametrize("qtype", sorted(FP16_FIELDS, key=lambda q: q.name), ids=lambda q: q.name)
def test_torch_kernels_match_gguf_reference(qtype):
    rng = np.random.default_rng(int(qtype))
    raw, block = random_blocks(qtype, 8, rng)
    rows = raw.reshape(2, -1)  # 2 rows of 4 blocks
    reference = gguf.quants.dequantize(rows, qtype)
    ours = dequantize(torch.from_numpy(rows), qtype, reference.shape, dtype=torch.float32).numpy()
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-6)


def test_all_kernels_are_tested():
    assert set(dequantize_functions) - {Q.BF16} == set(FP16_FIELDS)


# ---------------------------------------------------------------- tiny model in HOT-Step layout
TINY = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            intermediate_size=128, max_position_embeddings=24576, max_latent_frames=512)


def gguf_lm_name(key: str) -> str:
    """Inverse of loader.lm_hf_name, written independently like convert-yue2.py does."""
    fixed = {v: k for k, v in loader._LM_FIXED.items()}
    if key in fixed:
        return fixed[key]
    parts = key.split(".")
    i, module = parts[2], parts[3]
    names = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
             "nar_input_layernorm": "nar_attn_norm", "nar_pre_mlp_layernorm": "nar_ffn_norm"}
    if module in names:
        return f"blk.{i}.{names[module]}.weight"
    sub = parts[4]
    prefix = {"self_attn": "attn", "nar_self_attn": "nar_attn", "mlp": "ffn", "nar_mlp": "nar_ffn"}[module]
    short = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "output", "q_norm": "q_norm",
             "k_norm": "k_norm", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}[sub]
    return f"blk.{i}.{prefix}_{short}.weight"


def gpt2_encoder():
    return {b: c for c, b in loader._gpt2_byte_decoder().items()}


def synthetic_vocab():
    enc = gpt2_encoder()
    tokens = [enc[b] for b in range(256)]
    i = 0
    while len(tokens) < loader.ORDINARY_TOKENS:
        data = bytes([65 + (i % 26), 97 + (i // 26) % 26, 48 + (i // 676) % 10]) + i.to_bytes(3, "big")
        tokens.append("".join(enc[b] for b in data))
        i += 1
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [f"<extra_{k}>" for k in range(205)]
    return tokens + specials


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
    torch.manual_seed(0)
    model = YuE2ForCausalLM(YuE2Config(**TINY)).to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in model.parameters():  # realistic small weights
            p.copy_(torch.randn_like(p, dtype=torch.float32).mul_(0.05).to(p.dtype))
    path = tmp_path_factory.mktemp("lm") / "yue2-lm-q8_0.gguf"
    writer = gguf.GGUFWriter(str(path), "yue2")
    c = model.config
    for key, value in {"context_length": c.max_position_embeddings, "embedding_length": c.hidden_size,
                       "block_count": c.num_hidden_layers, "feed_forward_length": c.intermediate_size,
                       "attention.head_count": c.num_attention_heads,
                       "attention.head_count_kv": c.num_key_value_heads, "attention.key_length": c.head_dim,
                       "vocab_size": c.vocab_size, "latent_dim": c.latent_dim,
                       "max_latent_frames": c.max_latent_frames}.items():
        writer.add_uint32(f"yue2.{key}", value)
    writer.add_float32("yue2.attention.layer_norm_rms_epsilon", c.rms_norm_eps)
    writer.add_float32("yue2.rope.freq_base", c.rope_theta)
    writer.add_float32("yue2.timestep_shift", c.timestep_shift)
    writer.add_file_type(7)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(synthetic_vocab())
    for key, tensor in model.state_dict().items():
        value = tensor.float().numpy()
        name = gguf_lm_name(key)
        if key == "latent_pos_embed.pe":
            writer.add_tensor(name, gguf.quants.quantize(value, Q.BF16), raw_dtype=Q.BF16)
        elif value.ndim == 2 and value.shape[1] % 32 == 0 and key.endswith(".weight"):
            writer.add_tensor(name, gguf.quants.quantize(value, Q.Q8_0), raw_dtype=Q.Q8_0)
        else:
            writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return model, path


def test_lm_name_mapping_covers_model(tiny_lm):
    model, _ = tiny_lm
    for key in model.state_dict():
        assert loader.lm_hf_name(gguf_lm_name(key)) == key


@pytest.mark.parametrize("mode", ["bf16", "balanced", "max"])
def test_load_lm_modes(tiny_lm, mode):
    original, path = tiny_lm
    model, meta, report = loader.load_lm(path, mode)
    assert meta["general.architecture"] == "yue2"
    packed = {n for n, m in model.named_modules() if isinstance(m, loader.QuantLinear)}
    if mode == "bf16":
        assert not packed
    elif mode == "balanced":
        assert packed and all(".nar_" in n for n in packed)
        assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)  # AR keeps CUDA graphs
    else:
        assert "lm_head" in packed and "model.layers.0.self_attn.q_proj" in packed
    assert report.eager_required == (mode == "max")
    # every weight close to the original (Q8_0 error), quantized ones via their forward
    torch.manual_seed(2)
    for name, module in original.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "vae2llm" and module.in_features % 32 == 0:
            x = torch.randn(3, module.in_features, dtype=torch.bfloat16)
            loaded = model.get_submodule(name)
            torch.testing.assert_close(loaded(x).float(), module(x).float(), atol=0.05, rtol=0.05)
    torch.testing.assert_close(model.model.norm.weight, original.model.norm.weight)
    torch.testing.assert_close(model.latent_pos_embed.pe, original.latent_pos_embed.pe)
    if mode != "bf16":
        assert report.bytes < loader._model_bytes(original)


def test_quantlinear_moves_with_to(tiny_lm):
    _, path = tiny_lm
    model, _, _ = loader.load_lm(path, "max")
    q = model.model.layers[0].mlp.gate_proj
    assert next(q.parameters()).dtype == torch.uint8  # YuE2's offload uses next(module.parameters())
    model.to("meta")
    assert q.qweight.is_meta


def test_tokenizer_roundtrip(tiny_lm, tmp_path):
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    _, path = tiny_lm
    reader = gguf.GGUFReader(str(path))
    loader.write_tiktoken(reader, tmp_path / "qwen.tiktoken")
    tok = YuE2TextTokenizer(tmp_path / "qwen.tiktoken")
    ids = tok.encode("Hello, world! Ünïcode")
    assert tok.decode(ids) == "Hello, world! Ünïcode"


# ---------------------------------------------------------------- VAE
@pytest.fixture(scope="module")
def tiny_vae(tmp_path_factory):
    from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
    torch.manual_seed(1)
    vae = YuE2VAE(YuE2VAEConfig(), decoder_only=True).float().eval()
    path = tmp_path_factory.mktemp("vae") / "yue2-vae-standard-f32.gguf"
    writer = gguf.GGUFWriter(str(path), "yue2-vae")
    writer.add_string("yue2vae.variant", "standard")
    inverse = {}
    state = vae.state_dict()
    bases = sorted({k.rsplit(".", 1)[0] for k in state})
    for base in bases:
        gguf_base = _vae_gguf_base(base)
        if f"{base}.weight_v" in state:
            g, v = state[f"{base}.weight_g"], state[f"{base}.weight_v"]
            folded = (g * v / v.flatten(1).norm(dim=1).reshape(-1, 1, 1)).numpy()
            writer.add_tensor(f"{gguf_base}.weight", folded)
        for suffix in ("bias", "alpha", "beta"):
            if f"{base}.{suffix}" in state:
                writer.add_tensor(f"{gguf_base}.{suffix}", state[f"{base}.{suffix}"].numpy())
        inverse[gguf_base] = base
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return vae, path


def _vae_gguf_base(base: str) -> str:
    """Inverse of loader.vae_hf_name (decoder.layers.* → dec.*), mirroring convert-yue2.py."""
    p = base.split(".")[2:]  # after "decoder.layers"
    top = int(p[0])
    if len(p) == 1:
        return {0: "dec.conv_in", 7: "dec.snake_out", 8: "dec.conv_out"}[top]
    inner = int(p[2])
    if len(p) == 3:
        return f"dec.blk.{top}." + {0: "snake_pre", 1: "upsample"}[inner]
    return f"dec.blk.{top}.res.{inner - 2}." + ["snake1", "conv1", "snake2", "conv2"][int(p[4])]


def test_load_vae_restores_weight_norm(tiny_vae):
    original, path = tiny_vae
    vae, meta = loader.load_vae(path)
    assert meta["yue2vae.variant"] == "standard"
    z = torch.randn(1, 64, 4)
    with torch.inference_mode():
        torch.testing.assert_close(vae.decode(z), original.decode(z), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- full song through the official pipeline
@pytest.mark.parametrize("mode", ["balanced", "max"])
def test_engine_generates_song_from_gguf(tiny_lm, tiny_vae, tmp_path, mode):
    from musicgen_ui.config import Settings
    from musicgen_ui.engine.yue2 import YuE2Engine
    from musicgen_ui.models import SongSpec
    from musicgen_ui.prompting import compile_spec

    _, lm_path = tiny_lm
    _, vae_path = tiny_vae
    settings = Settings(model=str(lm_path), vae=str(vae_path), device="cpu", gguf_mode=mode,
                        data_dir=tmp_path / "data")
    engine = YuE2Engine(settings)
    spec = SongSpec(style="pop", lyrics="[Verse]\nla", seed=1,
                    score_sampling={"max_tokens": 4}, audio_sampling={"max_tokens": 12})
    request = compile_spec(spec)
    request.abc_sampling["min_tokens"] = 0
    request.semantic_sampling["min_tokens"] = 4
    events = []
    result = engine.generate(request, tmp_path / "out", progress=lambda *e: events.append(e), cancelled=lambda: False)
    assert result.audio_path.is_file() and result.sample_rate == 48000 and result.duration_s > 0
    assert isinstance(result.truncated, bool) and isinstance(result.extra["truncated_parts"], list)
    assert any(e[0] == "gguf" for e in events)
    stages = [e[0] for e in events]
    assert "nar" in stages and "vae" in stages and stages.index("nar") < stages.index("vae")  # phase markers
    info = engine.info()
    assert info["gguf"] and info["gguf_report"]["quantized_modules"] > 0
    engine.unload()
    assert not engine.loaded


# ---------------------------------------------------------------- LoRA (formats, merge/restore, side path)
def _random_lowrank(module, rank, gen):
    A = torch.randn(rank, module.in_features, generator=gen) * 0.05
    B = torch.randn(module.out_features, rank, generator=gen) * 0.05
    return A, B


def _write_ms(path, model, prefix, rank=4, extra=None):
    from safetensors.torch import save_file
    gen = torch.Generator().manual_seed(7)
    tensors, deltas = {}, {}
    for i, layer in enumerate(model.model.layers):
        for block, names in ((f"{prefix}self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                             (f"{prefix}mlp", ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                mod = layer.get_submodule(f"{block}.{name}")
                A, B = _random_lowrank(mod, rank, gen)
                tensors[f"layers.{i}.{block}.{name}.lora_A"], tensors[f"layers.{i}.{block}.{name}.lora_B"] = A, B
                deltas[f"model.layers.{i}.{block}.{name}"] = B @ A
    tensors.update(extra or {})
    save_file(tensors, str(path), metadata={"rank": str(rank)})
    return deltas


@pytest.fixture
def hf_tiny(tiny_lm, tmp_path):
    """Tiny model saved like an HF YuE2 folder (source for exact restore)."""
    from safetensors.torch import save_file
    from musicgen_ui.lora import hf_source
    model, _ = tiny_lm
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    import copy
    return copy.deepcopy(model), hf_source(tmp_path)


def test_lora_merge_and_exact_restore(hf_tiny, tmp_path):
    from musicgen_ui.lora import LoraState, read_adapter
    model, source = hf_tiny
    before = {k: v.clone() for k, v in model.state_dict().items()}
    deltas = _write_ms(tmp_path / "artist.safetensors", model, "")
    adapter = read_adapter(tmp_path / "artist.safetensors", {"q": 64, "kv": 32, "ff": 128})
    assert adapter.format == "mothersuperior" and adapter.halves == ["ar"] and adapter.rank == 4
    state = LoraState()
    state.apply(model, [(adapter, 0.5)], source)
    key = "model.layers.1.mlp.up_proj"
    expected = (before[f"{key}.weight"].float() + 0.5 * deltas[key]).to(torch.bfloat16)
    torch.testing.assert_close(model.get_submodule(key).weight, expected)
    assert not state.apply(model, [(adapter, 0.5)], source)["changed"]  # unchanged selection is free
    state.apply(model, [], source)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k  # bit-exact restore


def test_lora_formats_are_equivalent(hf_tiny, tmp_path):
    """Mothersuperior, ComfyUI (fused, alpha) and .pt describe the same delta."""
    from safetensors.torch import save_file
    from musicgen_ui.lora import read_adapter
    model, _ = hf_tiny
    dims = {"q": 64, "kv": 32, "ff": 128}
    deltas = _write_ms(tmp_path / "ms.safetensors", model, "nar_",
                       extra={"vae2llm.weight": torch.zeros_like(model.vae2llm.weight).float(),
                              "vae2llm.bias": torch.zeros_like(model.vae2llm.bias).float()})
    ms = read_adapter(tmp_path / "ms.safetensors", dims)
    assert ms.halves == ["nar"] and "vae2llm.weight" in ms.replacements
    # ComfyUI: fuse q|k|v and gate|up block-diagonally, store alpha = rank_total (scale 1.0)
    comfy = {}
    for i in range(len(model.model.layers)):
        for fused, parts in (("self_attn.qkv_proj", ("nar_self_attn.q_proj", "nar_self_attn.k_proj", "nar_self_attn.v_proj")),
                             ("mlp.gate_up_proj", ("nar_mlp.gate_proj", "nar_mlp.up_proj"))):
            downs = [ms.deltas[f"model.layers.{i}.{p}"].A for p in parts]
            ups = [ms.deltas[f"model.layers.{i}.{p}"].B for p in parts]
            r = sum(d.shape[0] for d in downs)
            up = torch.zeros(sum(u.shape[0] for u in ups), r)
            row = col = 0
            for d, u in zip(downs, ups):
                up[row:row + u.shape[0], col:col + d.shape[0]] = u
                row, col = row + u.shape[0], col + d.shape[0]
            base = f"diffusion_model.model.layers.{i}.{fused}"
            comfy[f"{base}.lora_down.weight"], comfy[f"{base}.lora_up.weight"] = torch.cat(downs), up
            comfy[f"{base}.alpha"] = torch.tensor(float(r))
        for proj in ("self_attn.o_proj", "mlp.down_proj"):
            d = ms.deltas[f"model.layers.{i}.nar_{proj}"]
            base = f"diffusion_model.model.layers.{i}.{proj}"
            comfy[f"{base}.lora_down.weight"], comfy[f"{base}.lora_up.weight"] = d.A, d.B * 2
            comfy[f"{base}.alpha"] = torch.tensor(d.A.shape[0] / 2)  # alpha/rank = 0.5 compensates the ×2
    save_file(comfy, str(tmp_path / "x_comfyui.safetensors"))
    cf = read_adapter(tmp_path / "x_comfyui.safetensors", dims)
    assert cf.format == "comfyui" and cf.halves == ["nar"]
    for module, delta in deltas.items():
        got = cf.deltas[module]
        torch.testing.assert_close((got.B @ got.A) * got.scale, delta, atol=1e-5, rtol=1e-5)
    # Mothersuperior .pt checkpoint (layer-major list) loads safely and identically
    order = []
    for i in range(len(model.model.layers)):
        for block, names in (("nar_self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                             ("nar_mlp", ("gate_proj", "up_proj", "down_proj"))):
            order += [ms.deltas[f"model.layers.{i}.{block}.{n}"] for n in names]
    torch.save({"lora": [t for d in order for t in (d.A, d.B)], "rank": 4,
                "io": {"vae2llm": {"weight": torch.zeros(1), "bias": torch.zeros(1)}}}, tmp_path / "nar.pt")
    pt = read_adapter(tmp_path / "nar.pt", dims)
    assert pt.halves == ["nar"] and len(pt.deltas) == len(ms.deltas)


def test_lora_side_path_on_packed_layers(tiny_lm, tmp_path):
    from musicgen_ui.lora import LoraState, gguf_source, read_adapter
    original, path = tiny_lm
    model, _, _ = loader.load_lm(path, "balanced")
    _write_ms(tmp_path / "nar.safetensors", original, "nar_")
    adapter = read_adapter(tmp_path / "nar.safetensors", {"q": 64, "kv": 32, "ff": 128})
    q = model.model.layers[0].nar_mlp.up_proj
    x = torch.randn(2, q.in_features, dtype=torch.bfloat16)
    plain = q(x)
    state = LoraState()
    state.apply(model, [(adapter, 1.0)], gguf_source(path))
    d = adapter.deltas["model.layers.0.nar_mlp.up_proj"]
    torch.testing.assert_close(q(x).float(), plain.float() + (x.float() @ (d.B @ d.A).T), atol=0.05, rtol=0.05)
    state.apply(model, [], gguf_source(path))
    assert not q.adapters and torch.equal(q(x), plain)


def test_engine_applies_lora_per_song(tiny_lm, tiny_vae, tmp_path):
    from musicgen_ui.config import Settings
    from musicgen_ui.engine.yue2 import YuE2Engine
    from musicgen_ui.models import SongSpec
    from musicgen_ui.prompting import compile_spec
    original, lm_path = tiny_lm
    _, vae_path = tiny_vae
    _write_ms(tmp_path / "artist.safetensors", original, "")
    engine = YuE2Engine(Settings(model=str(lm_path), vae=str(vae_path), device="cpu", gguf_mode="balanced",
                                 data_dir=tmp_path / "data"))
    spec = SongSpec(style="pop", lyrics="[Verse]\nla", seed=1, plan_mode="off", audio_sampling={"max_tokens": 8},
                    loras=[{"path": str(tmp_path / "artist.safetensors"), "strength": 0.8}])
    request = compile_spec(spec)
    request.semantic_sampling["min_tokens"] = 4
    engine.generate(request, tmp_path / "a", progress=lambda *e: None, cancelled=lambda: False)
    assert engine.info()["loras"] == [(str(tmp_path / "artist.safetensors"), 0.8)]
    weight = engine._pipe._model.model.layers[0].self_attn.q_proj.weight.clone()
    request.loras = []
    engine.generate(request, tmp_path / "b", progress=lambda *e: None, cancelled=lambda: False)
    assert engine.info()["loras"] == []
    assert not torch.equal(engine._pipe._model.model.layers[0].self_attn.q_proj.weight, weight)  # LoRA removed


# ---------------------------------------------------------------- yue2.cpp layout (Serveurperso/YuE2-GGUF)
@pytest.fixture(scope="module")
def tiny_lm_cpp(tiny_lm, tmp_path_factory):
    """Same tiny model in the yue2.cpp layout: original names, config as JSON, file_type as string."""
    import json
    model, _ = tiny_lm
    path = tmp_path_factory.mktemp("cpp") / "YuE2-3B-Q8_0.gguf"
    writer = gguf.GGUFWriter(str(path), "yue2")
    writer.add_string("yue2.config_json", json.dumps({**model.config.to_dict(), "dtype": "bfloat16"}))
    writer.add_uint32("yue2.block_count", model.config.num_hidden_layers)
    writer.add_string("general.file_type", "Q8_0")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(synthetic_vocab())
    for key, tensor in model.state_dict().items():
        value = tensor.float().numpy()
        if value.ndim == 2 and value.shape[1] % 32 == 0 and key.endswith(".weight") and key != "latent_pos_embed.pe":
            writer.add_tensor(key, gguf.quants.quantize(value, Q.Q8_0), raw_dtype=Q.Q8_0)
        else:
            writer.add_tensor(key, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_yue2cpp_layout_loads(tiny_lm, tiny_lm_cpp, tmp_path):
    from musicgen_ui import catalog
    original, _ = tiny_lm
    model, _, report = loader.load_lm(tiny_lm_cpp, "balanced")
    assert report.quantized and model.config.hidden_size == TINY["hidden_size"]
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(model.model.layers[1].self_attn.q_proj(x).float(),
                               original.model.layers[1].self_attn.q_proj(x).float(), atol=0.05, rtol=0.05)
    skeleton = loader.make_lm_skeleton(tiny_lm_cpp, tmp_path / "skel")
    assert (skeleton / "qwen.tiktoken").is_file()
    root = tmp_path / "models"
    catalog.ensure_dirs(root)
    (root / "yue2" / tiny_lm_cpp.name).write_bytes(tiny_lm_cpp.read_bytes())
    entry = next(e for e in catalog.scan(root)["checkpoint"] if e["format"] == "gguf")
    assert entry["engine"] == "yue2" and entry["quant"] == "Q8_0" and "yue2.cpp" in entry["note"]


def test_gpu_share_chunks_attention_and_cleans_up(tiny_lm, tiny_vae, tmp_path, monkeypatch):
    import yue2.nar as nar
    from musicgen_ui.config import Settings
    from musicgen_ui.engine.yue2 import YuE2Engine
    from musicgen_ui.models import SongSpec
    from musicgen_ui.prompting import compile_spec
    _, lm_path = tiny_lm
    _, vae_path = tiny_vae
    seen = []
    original = nar.attention

    def spy(q, k, v, *args, query_chunk_size=None, **kwargs):
        seen.append(query_chunk_size)
        return original(q, k, v, *args, query_chunk_size=query_chunk_size, **kwargs)
    monkeypatch.setattr(nar, "attention", spy)
    engine = YuE2Engine(Settings(model=str(lm_path), vae=str(vae_path), device="auto", gpu_share="smooth",
                                 data_dir=tmp_path / "data"))
    spec = SongSpec(style="pop", lyrics="[Verse]\nla", seed=1, plan_mode="off", audio_sampling={"max_tokens": 8})
    request = compile_spec(spec)
    request.semantic_sampling["min_tokens"] = 4
    engine.generate(request, tmp_path / "out", progress=lambda *e: None, cancelled=lambda: False)
    assert seen and set(seen) == {2048}                 # synthesis attention ran in query blocks
    assert nar.attention is spy                          # our patch was undone
    hooks = [len(layer.nar_mlp._forward_pre_hooks) for layer in engine._pipe._model.model.layers]
    assert hooks == [0] * len(hooks)                     # pause hooks removed after the song
