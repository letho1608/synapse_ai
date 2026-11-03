"""
LoRA on the layers a node serves.

Before: use_lora loaded the WHOLE model on one machine and wrapped it with
PEFT, while pipeline steps trained the PartialModel layers, so on a cluster
the adapters were never even used. Now each stage injects adapters into its
own layers, trains only those (base frozen, also when int8/int4), and a
3-node pipeline step equals a 1-node step on the same adapters.
"""
import sys

import numpy as np
import pytest
import torch

from test_pipeline_training import TINY, _Cluster, _batch, _cuts, _engine

LORA = {"use_lora": True, "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SYNAPSE_PARTIAL_LOAD", "1")
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "none")


def _cfg():
    from synapse.training.stage_lora import normalize_config
    return normalize_config(LORA)


def _base_state(eng, shard):
    return {f"{i}.{k}": v.detach().clone()
            for i in range(shard.start_layer, shard.end_layer + 1)
            for k, v in eng._pm.layers[i].state_dict().items() if "lora_" not in k}


def _logits(eng, shard, ids):
    def run():
        with torch.no_grad():
            _, h, _ = eng._stage_run(shard, ids, ids, grad=False)
            return eng._get_transformer_components()["lm_head"](h)
    return run()


@pytest.mark.asyncio
async def test_lora_trains_adapters_only_on_served_layers():
    from synapse.inference.shard import Shard
    model_id, n = TINY
    ids, labels = _batch(model_id)
    eng = _engine()
    shard = Shard(model_id, 0, n - 1, n)
    await eng.ensure_shard(shard)
    before_logits = _logits(eng, shard, ids)
    await eng.stage_train_prepare(shard, lora=_cfg())
    # B starts at zero: a fresh adapter does not change the output
    assert torch.allclose(_logits(eng, shard, ids), before_logits, atol=1e-6)
    names = [nm for nm, _ in eng._stage_params(shard)]
    assert names and all("lora_" in nm for nm in names)
    assert all(nm.startswith("layers.") for nm in names), "embeddings must stay frozen"
    base = _base_state(eng, shard)
    losses = []
    for _ in range(6):
        loss, _ = await eng.train("r", shard, ids, labels, np.ones(len(ids)), lora=_cfg())
        losses.append(loss)
    assert eng._model is None, "LoRA loaded the whole model"
    assert losses[-1] < losses[0], losses
    after = _base_state(eng, shard)
    assert all(torch.equal(base[k], after[k]) for k in base), "frozen base weights changed"


@pytest.mark.asyncio
async def test_pipeline_lora_step_equals_single_node_step():
    from synapse.inference.shard import Shard
    model_id, n = TINY
    ids, labels = _batch(model_id)
    lr, m = 1e-2, 2
    cfg = _cfg()

    single = _engine()
    full = Shard(model_id, 0, n - 1, n)
    plan = [(f"node-{i}", Shard(model_id, a, b, n)) for i, (a, b) in enumerate(_cuts(n, 3))]
    engines = [_engine() for _ in plan]
    await single.stage_train_prepare(full, lora=cfg)
    init = {nm: p.detach().clone() for nm, p in single._stage_params(full)}
    for eng, (_, shard) in zip(engines, plan):
        await eng.stage_train_prepare(shard, lora=cfg)
        with torch.no_grad():
            for nm, p in eng._stage_params(shard):
                p.copy_(init[nm])  # same random A on both sides

    cluster = _Cluster(engines)
    loss = await cluster.trainers["node-0"].train_step(plan, ids, labels, lr=lr,
                                                      num_micro_batches=m, lora=cfg)

    # Reference: the same step on one node holding every layer.
    single_cluster = _Cluster([single])
    ref_loss = await single_cluster.trainers["node-0"].train_step(
        [("node-0", full)], ids, labels, lr=lr, num_micro_batches=m, lora=cfg)
    assert loss == pytest.approx(ref_loss, rel=1e-5)
    want = dict(single._stage_params(full))
    worst, compared = 0.0, 0
    for eng, (_, shard) in zip(engines, plan):
        assert eng._model is None
        for nm, p in eng._stage_params(shard):
            worst = max(worst, float((p.detach() - want[nm].detach()).abs().max()))
            compared += 1
    assert compared == len(want)
    assert worst < 1e-5, worst
    assert any(not torch.equal(want[nm], init[nm]) for nm in want), "adapters did not move"


@pytest.mark.asyncio
async def test_lora_over_int8_weights(monkeypatch):
    """QLoRA-style: full fine-tuning of quantized weights is refused, LoRA
    over them trains (gradient flows through the frozen int8 layers)."""
    from synapse.inference.shard import Shard
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "int8")
    model_id, n = TINY
    ids, labels = _batch(model_id)
    eng = _engine()
    shard = Shard(model_id, 0, n - 1, n)
    await eng.ensure_shard(shard)
    if eng._pm.quant_mode == "none":
        pytest.skip("int8 not applied to this model")
    with pytest.raises(RuntimeError, match="LoRA"):
        await eng.stage_train_prepare(shard)
    losses = []
    for _ in range(6):
        loss, _ = await eng.train("r", shard, ids, labels, np.ones(len(ids)), lora=_cfg())
        losses.append(loss)
    assert losses[-1] < losses[0], losses


@pytest.mark.asyncio
async def test_merge_keeps_outputs_and_unwraps():
    from synapse.inference.shard import Shard
    from synapse.training.stage_lora import has_lora
    model_id, n = TINY
    ids, labels = _batch(model_id)
    eng = _engine()
    shard = Shard(model_id, 0, n - 1, n)
    for _ in range(3):
        await eng.train("r", shard, ids, labels, np.ones(len(ids)), lora=_cfg())
    with_adapters = _logits(eng, shard, ids)
    assert eng.stage_lora_merge(shard) > 0
    assert not any(has_lora(eng._pm.layers[i]) for i in range(n))
    assert torch.allclose(_logits(eng, shard, ids), with_adapters, atol=1e-5)
    # back to full fine-tuning mode
    await eng.stage_train_prepare(shard)
    assert any(nm.startswith("embed.") for nm, _ in eng._stage_params(shard))


@pytest.mark.asyncio
async def test_checkpoint_restores_adapters_on_another_node(tmp_path):
    """Failover: a survivor that never trained restores the dead node's
    layers including their adapters (same outputs)."""
    from synapse.inference.shard import Shard
    model_id, n = TINY
    ids, labels = _batch(model_id)
    shard = Shard(model_id, 0, n - 1, n)
    a = _engine()
    for _ in range(3):
        await a.train("r", shard, ids, labels, np.ones(len(ids)), lora=_cfg())
    path = str(tmp_path / "s.weights.pt")
    assert a.save_shard_weights(shard, path)

    b = _engine()
    await b.ensure_shard(shard)
    assert b._restore_shard_weights(shard, path)
    assert b._lora_cfg == a._lora_cfg
    assert torch.allclose(_logits(b, shard, ids), _logits(a, shard, ids), atol=1e-6)


SMOL = ("HuggingFaceTB/SmolLM2-135M-Instruct", 30)


def _cached(model_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(not _cached(SMOL[0]), reason="SmolLM2 not cached")
async def test_fast_decoder_serves_the_adapters(monkeypatch):
    """After LoRA training (adapters only on v_proj, which the old staleness
    check did not look at) the fast inference path must use the adapted
    projections, not its stale fused copy of the base weights."""
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    monkeypatch.setenv("SYNAPSE_FAST_PATH", "1")
    model_id, n = SMOL
    shard = Shard(model_id, 0, n - 1, n)
    eng = PyTorchHFInferenceEngine()
    await eng.ensure_shard(shard)
    ids = np.asarray(await eng.encode(shard, "The capital of France is"), dtype=np.int64)
    x0, _ = eng._forward_transformer_range(shard, ids, "warm", {})  # builds fused plans
    cfg = {"use_lora": True, "lora_r": 4, "lora_alpha": 64, "lora_target_modules": "v_proj"}
    with torch.no_grad():
        await eng.stage_train_prepare(shard, lora=cfg)
        for _, p in eng._stage_params(shard):
            if p.shape[1] == 4:  # lora_B: make the adapter non-zero
                p.normal_(0, 0.05)
    eng._seq_kv.clear()
    fast_out, _ = eng._forward_transformer_range(shard, ids, "after", {})
    ref = _logits(eng, shard, ids[None])
    assert not np.allclose(fast_out, x0, atol=1e-3), "adapters had no effect on the fast path"
    np.testing.assert_allclose(np.asarray(fast_out).reshape(-1)[-ref.shape[-1]:],
                               ref[0, -1].numpy(), atol=2e-3)


def test_messages_samples_format_without_a_chat_template():
    """Models without a chat template (tiny GPT-2) used to drop every
    {"messages": [...]} sample: apply_chat_template raised."""
    from transformers import AutoTokenizer
    from synapse.training.finetune_lora import format_sample_to_text
    tok = AutoTokenizer.from_pretrained(TINY[0])
    assert not getattr(tok, "chat_template", None)
    text = format_sample_to_text({"messages": [{"role": "user", "content": "hi"},
                                               {"role": "assistant", "content": "hello"}]}, tok)
    assert "hi" in text and "hello" in text
