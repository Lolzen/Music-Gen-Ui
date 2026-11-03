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
    assert any(e[0] == "gguf" for e in events)
    info = engine.info()
    assert info["gguf"] and info["gguf_report"]["quantized_modules"] > 0
    engine.unload()
    assert not engine.loaded
