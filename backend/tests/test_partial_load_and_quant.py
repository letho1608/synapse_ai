"""
Behavioral tests: a node holds only the layers it serves, and weight-only
quantization keeps the model's answers.

Found 2026-09-30: ensure_shard() called AutoModelForCausalLM.from_pretrained
on every node, so each node held the WHOLE model and "run models too big for
one machine" was not true. Layer-range loading materializes only
[start, end] (+ embedding on the first stage, norm + head on the last).
"""
import sys

import numpy as np
import pytest
import torch

TINY = ("peft-internal-testing/tiny-random-GPT2LMHeadModel", 5)
SMOL = ("HuggingFaceTB/SmolLM2-135M-Instruct", 30)


def _cached(model_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


needs_smol = pytest.mark.skipif(not _cached(SMOL[0]), reason="SmolLM2 not cached")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SYNAPSE_PARTIAL_LOAD", "1")
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "none")


def _engine():
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    return PyTorchHFInferenceEngine()


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", [pytest.param(TINY, id="tiny-gpt2"),
                                  pytest.param(SMOL, id="smollm2", marks=needs_smol)])
async def test_middle_stage_holds_only_its_layers(spec):
    from synapse.inference.shard import Shard
    model_id, n = spec
    lo, hi = 1, n - 2
    eng = _engine()
    await eng.ensure_shard(Shard(model_id, lo, hi, n))
    assert eng.loaded_layers() == list(range(lo, hi + 1))
    pm = eng._pm
    assert pm.embed is None and pm.lm_head is None and pm.final_norm is None
    per_layer = pm.layer_param_count() * 4  # fp32 on CPU
    assert pm.loaded_bytes() <= per_layer * (hi - lo + 1) + 1_000_000
    # A layer outside the range is not silently available.
    with pytest.raises(RuntimeError, match="not loaded"):
        eng._get_transformer_components()["layers"][0]


@pytest.mark.asyncio
@needs_smol
async def test_two_half_shards_split_the_memory():
    """Two nodes serving 0-14 and 15-29 each hold about half the layers."""
    from synapse.inference.shard import Shard
    model_id, n = SMOL
    a, b = _engine(), _engine()
    await a.ensure_shard(Shard(model_id, 0, 14, n))
    await b.ensure_shard(Shard(model_id, 15, 29, n))
    full = _engine()
    await full.ensure_shard(Shard(model_id, 0, 29, n))
    layer_bytes = full._pm.layer_param_count() * 4 * n
    assert a.loaded_layers() == list(range(0, 15)) and b.loaded_layers() == list(range(15, 30))
    assert a._pm.embed is not None and a._pm.lm_head is None
    assert b._pm.embed is None and b._pm.lm_head is not None
    # Each half carries ~half the decoder weights (+ its end: embed or head).
    for eng in (a, b):
        assert eng._pm.loaded_bytes() < 0.5 * layer_bytes + 120_000_000
    assert a._pm.loaded_bytes() + b._pm.loaded_bytes() < full._pm.loaded_bytes() + 120_000_000


@pytest.mark.asyncio
async def test_layers_load_on_demand_and_release():
    from synapse.inference.shard import Shard
    model_id, n = TINY
    eng = _engine()
    await eng.ensure_shard(Shard(model_id, 0, 1, n))
    assert eng.loaded_layers() == [0, 1]
    await eng.ensure_shard(Shard(model_id, 2, 4, n))  # new range after a re-plan
    assert eng.loaded_layers() == [0, 1, 2, 3, 4]
    assert eng.release_layers([2, 3, 4]) == [0, 1]
    assert eng.loaded_layers() == [2, 3, 4]


def test_quant_linear_round_trip_and_conv1d():
    from synapse.inference.weight_quant import QuantLinear, quantize_module
    torch.manual_seed(0)
    w = torch.randn(48, 100)
    for mode, tol in (("int8", 0.01), ("int4", 0.12)):
        q = QuantLinear(w, torch.zeros(48), mode)
        rel = float((q.dequantize() - w).norm() / w.norm())
        assert rel < tol, (mode, rel)
    from transformers.pytorch_utils import Conv1D
    conv = Conv1D(24, 40)  # GPT-2 layout: weight [in, out]
    x = torch.randn(3, 40)
    ref = conv(x)
    holder = torch.nn.Sequential(conv)
    assert quantize_module(holder, "int8") == 1
    assert torch.allclose(holder(x), ref, atol=0.05)


def test_auto_mode_quantizes_only_when_needed():
    from synapse.inference.weight_quant import choose_mode
    gb = 1 << 30
    assert choose_mode(2 * gb, 16 * gb, "auto") == "none"
    assert choose_mode(10 * gb, 8 * gb, "auto") == "int8"
    assert choose_mode(40 * gb, 8 * gb, "auto") == "int4"
    assert choose_mode(40 * gb, 8 * gb, "none") == "none"


@pytest.mark.asyncio
@needs_smol
async def test_int8_weights_keep_answers_and_shrink_memory(monkeypatch):
    """Quality gate (teacher-forced over the prompt positions, 6 prompts):
    int8 keeps top-1 at >= 99% with small KL and holds the decoder layers in
    well under half the fp32 bytes. Measured 2026-09-30: top-1 100%,
    KL 0.006, whole shard 0.41x (the fp32 embedding dominates what is left).
    int4 is gated separately (test_int4_mixed_precision_keeps_answers);
    auto mode only picks it when int8 cannot fit."""
    from synapse.inference.shard import Shard
    model_id, n = SMOL
    prompts = ["The capital of France is", "def fibonacci(n):\n    ",
               "Water boils at a temperature of", "In 1492, Christopher Columbus",
               "The quick brown fox jumps over", "Machine learning is a field of"]
    shard = Shard(model_id, 0, n - 1, n)

    async def run(mode):
        monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", mode)
        eng = _engine()
        outs = []
        for p in prompts:
            ids = np.asarray(await eng.encode(shard, p), dtype=np.int64)
            x, _ = await eng.infer_tensor(None, shard, ids)
            outs.append(np.asarray(x).reshape(-1, x.shape[-1]))
        layers = sum(eng._pm.layers[i] is not None for i in range(n))
        from synapse.inference.weight_quant import module_bytes
        return outs, sum(module_bytes(eng._pm.layers[i]) for i in range(n)), layers

    ref, ref_bytes, _ = await run("none")
    got, q_bytes, _ = await run("int8")
    agree, kls = [], []
    for r, g in zip(ref, got):
        agree += list(r.argmax(-1) == g.argmax(-1))
        lr, lg = torch.log_softmax(torch.tensor(r), -1), torch.log_softmax(torch.tensor(g), -1)
        kls.append(float((lr.exp() * (lr - lg)).sum(-1).mean()))
    assert np.mean(agree) >= 0.99, np.mean(agree)
    assert np.mean(kls) <= 0.02, np.mean(kls)
    assert q_bytes < 0.3 * ref_bytes, q_bytes / ref_bytes


_PROMPTS = ["The capital of France is", "def fibonacci(n):\n    ",
            "Water boils at a temperature of", "In 1492, Christopher Columbus",
            "The quick brown fox jumps over", "Machine learning is a field of"]


async def _logits_and_bytes(monkeypatch, mode, keep=None):
    from synapse.inference.shard import Shard
    from synapse.inference.weight_quant import module_bytes
    model_id, n = SMOL
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", mode)
    if keep is None:
        monkeypatch.delenv("SYNAPSE_INT4_KEEP", raising=False)
    else:
        monkeypatch.setenv("SYNAPSE_INT4_KEEP", keep)
    shard = Shard(model_id, 0, n - 1, n)
    eng = _engine()
    outs = []
    for p in _PROMPTS:
        ids = np.asarray(await eng.encode(shard, p), dtype=np.int64)
        x, _ = await eng.infer_tensor(None, shard, ids)
        outs.append(np.asarray(x).reshape(-1, x.shape[-1]))
    return outs, sum(module_bytes(eng._pm.layers[i]) for i in range(n))


def _agreement_and_kl(ref, got):
    agree, kls = [], []
    for r, g in zip(ref, got):
        agree += list(r.argmax(-1) == g.argmax(-1))
        lr, lg = torch.log_softmax(torch.tensor(r), -1), torch.log_softmax(torch.tensor(g), -1)
        kls.append(float((lr.exp() * (lr - lg)).sum(-1).mean()))
    return float(np.mean(agree)), float(np.mean(kls))


@pytest.mark.asyncio
@needs_smol
async def test_int4_mixed_precision_keeps_answers(monkeypatch):
    """int4 quality gate (same teacher-forced measure as int8). Before:
    min/max group fit, everything int4 -> top-1 51%, KL 0.63. Now: per-group
    MSE range search + least-squares scale/zero, and v_proj / down_proj plus
    the two end layers on each side in int8 (Q4_K_M-style). Measured
    2026-11-02: top-1 74%, KL 0.066 at 0.19x the fp32 decoder bytes; the
    better fit alone (SYNAPSE_INT4_KEEP=none) gives KL 0.38 at 0.16x."""
    ref, ref_bytes = await _logits_and_bytes(monkeypatch, "none")
    got, q_bytes = await _logits_and_bytes(monkeypatch, "int4")
    top1, kl = _agreement_and_kl(ref, got)
    assert top1 >= 0.7, top1
    assert kl <= 0.1, kl
    assert q_bytes < 0.22 * ref_bytes, q_bytes / ref_bytes
    pure, pure_bytes = await _logits_and_bytes(monkeypatch, "int4", keep="none")
    assert _agreement_and_kl(ref, pure)[1] <= 0.5
    assert pure_bytes < q_bytes


def test_int4_layer_policy():
    from synapse.inference.weight_quant import layer_mode, quantize_module
    modes = [layer_mode("int4", i, 30) for i in (0, 1, 2, 27, 28, 29)]
    assert modes == ["int8", "int8", "int4", "int4", "int8", "int8"]
    assert layer_mode("int8", 0, 30) == "int8" and layer_mode("none", 0, 30) == "none"

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(64, 64)
            self.self_attn.v_proj = torch.nn.Linear(64, 64)
            self.mlp = torch.nn.Module()
            self.mlp.down_proj = torch.nn.Linear(64, 64)
    b = Block()
    assert quantize_module(b, "int4") == 3
    assert (b.self_attn.q_proj.mode, b.self_attn.v_proj.mode, b.mlp.down_proj.mode) == ("int4", "int8", "int8")


def test_quantized_kernels_match_dequantized_math():
    """The numba kernels compute exactly x @ dequant(W)^T (fp32 accumulate),
    for one row (decode) and several (prefill), with odd sizes."""
    from synapse.inference import qkernels
    from synapse.inference.weight_quant import QuantLinear
    if not qkernels.enabled():
        pytest.skip("numba not installed")
    torch.manual_seed(0)
    for mode in ("int8", "int4"):
        w = torch.randn(70, 100) * 0.05
        q = QuantLinear(w, torch.randn(70), mode)
        for rows in (1, 5):
            x = torch.randn(1, rows, 100)
            ref = torch.nn.functional.linear(x, q.dequantize(), q.bias)
            got = q(x)
            assert torch.allclose(got, ref, atol=1e-4), (mode, rows, float((got - ref).abs().max()))


def test_sharded_checkpoint_fetches_only_needed_files(tmp_path, monkeypatch):
    """A node fetches only the safetensors files that hold its tensors, when
    it first needs them; a later range fetches just what is missing."""
    import json
    from transformers import AutoModelForCausalLM
    from synapse.inference import partial_loader
    from synapse.inference.partial_loader import PartialModel
    ref = AutoModelForCausalLM.from_pretrained(TINY[0])
    ref.save_pretrained(tmp_path, max_shard_size="20KB", safe_serialization=True)
    weight_map = json.loads((tmp_path / "model.safetensors.index.json").read_text())["weight_map"]
    assert len(set(weight_map.values())) > 6, "checkpoint not sharded enough for the test"

    fetched = []
    real = partial_loader._hub_file

    def spy(repo_id, filename):
        fetched.append(filename)
        return real(repo_id, filename)

    monkeypatch.setattr(partial_loader, "_hub_file", spy)

    def files_for(pred):
        return {f for k, f in weight_map.items() if pred(k)}

    def layer(i):
        return lambda k: k.startswith(f"transformer.h.{i}.")

    pm = PartialModel(str(tmp_path), torch.float32, "cpu")
    assert [f for f in fetched if f.endswith(".safetensors")] == [], "shards fetched before any layer was asked for"

    pm.ensure(2, 2, first=False, last=False)  # middle stage
    assert set(pm.files) == files_for(layer(2))
    got = dict(pm.layers[2].named_parameters())
    want = dict(ref.transformer.h[2].named_parameters())
    assert all(torch.equal(got[n], want[n]) for n in want)

    before = set(pm.files)
    pm.ensure(3, 4, first=False, last=True)  # re-plan: this node now ends the pipeline
    need = files_for(lambda k: layer(3)(k) or layer(4)(k) or k.startswith("transformer.ln_f.")
                     or k.startswith("transformer.wte."))  # tied head reads the embedding
    assert set(pm.files) == before | need
    assert len(set(pm.files)) < len(set(weight_map.values()))
    assert sorted(f for f in fetched if f.endswith(".safetensors")) == sorted(pm.files)  # each file once
