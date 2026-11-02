"""
Behavioral tests: fast Llama-family decoder + paged KV cache.

The fast path (fused q|k|v and gate|up matmuls, SDPA, paged KV) must give
the same logits as running the HF modules, for every way the engine calls
it: prefill, one-token decode, multi-row draft verification after a crop,
and batched decode of several requests. Paged KV must hand blocks back on
crop and when a request ends, and grow when it runs out.
"""
import asyncio
import sys

import numpy as np
import pytest
import torch

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


def test_paged_sequence_crop_release_and_growth():
    from synapse.inference.fast_decoder import KVPool, PagedSequence
    pool = KVPool(n_kv=2, head_dim=4, dtype=torch.float32, device="cpu", block=4, init_blocks=2)
    seq = PagedSequence(pool)
    k = torch.arange(10 * 2 * 4, dtype=torch.float32).view(10, 2, 4)
    seq.write(0, k, -k, seq.slots(0, 10))       # 10 positions -> 3 blocks: pool grows 2 -> 4
    seq.advance(10)
    assert pool.capacity == 4 and len(seq.blocks) == 3
    K, V = seq.read(0, 10)
    assert torch.equal(K[0].transpose(0, 1), k) and torch.equal(V[0].transpose(0, 1), -k)
    seq.crop(5)                                  # keep 2 blocks, return 1
    assert len(seq.blocks) == 2 and pool.used_blocks == 2
    k2 = torch.full((3, 2, 4), 7.0)
    seq.write(0, k2, k2, seq.slots(5, 3))         # rewrite after the crop
    seq.advance(8)
    K, _ = seq.read(0, 8)
    assert torch.equal(K[0].transpose(0, 1)[:5], k[:5]) and torch.equal(K[0].transpose(0, 1)[5:], k2)
    del seq                                      # a finished request returns its blocks
    assert pool.used_blocks == 0


async def _engine(fast: bool, monkeypatch):
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    monkeypatch.setenv("SYNAPSE_FAST_PATH", "1" if fast else "0")
    e = PyTorchHFInferenceEngine()
    shard = Shard(SMOL[0], 0, SMOL[1] - 1, SMOL[1])
    await e.ensure_shard(shard)
    assert (e._fast() is not None) == fast
    return e, shard


@pytest.mark.asyncio
@needs_smol
async def test_fast_path_matches_hf_modules_for_every_call_shape(monkeypatch):
    fast, shard = await _engine(True, monkeypatch)
    slow, _ = await _engine(False, monkeypatch)
    ids = np.asarray(await fast.encode(shard, "The capital of France is Paris and the"), dtype=np.int64)
    draft_ctx = np.concatenate([ids, [11, 12, 13]])
    for e in (fast, slow):
        e.outs = []
        x, st = e._forward_transformer_range(shard, ids, "r", {})            # prefill
        e.outs.append(x)
        nxt = np.concatenate([ids, [int(x[0, -1].argmax())]])
        x, st = e._forward_transformer_range(shard, nxt, "r", st)            # decode
        e.outs.append(x)
        x, st = e._forward_transformer_range(shard, draft_ctx, "r", {"verify_n": 4})  # crop + 4 rows
        e.outs.append(x)
    assert fast._is_paged(fast._seq_kv[f"r@0-{shard.end_layer}"]["cache"])
    for a, b in zip(fast.outs, slow.outs):
        assert a.shape == b.shape
        assert np.abs(a - b).max() < 2e-3, np.abs(a - b).max()
        assert (a.reshape(-1, a.shape[-1]).argmax(-1) == b.reshape(-1, b.shape[-1]).argmax(-1)).all()


@pytest.mark.asyncio
@needs_smol
async def test_fast_batched_decode_matches_solo(monkeypatch):
    e, shard = await _engine(True, monkeypatch)
    prompts = ["Hello there, my name is", "def add(a, b):\n    return", "Water boils at"]
    states, solo = [], []
    for i, p in enumerate(prompts):
        ids = np.asarray(await e.encode(shard, p), dtype=np.int64)
        x, st = e._forward_transformer_range(shard, ids, f"b{i}", {})
        nxt = np.concatenate([ids, [int(x[0, -1].argmax())]])
        states.append((nxt, st))
        xs, _ = e._forward_transformer_range(shard, nxt, f"s{i}", {})        # reference, own request
        solo.append(xs[0, -1])
    jobs = [(shard, nxt, f"b{i}", st) for i, (nxt, st) in enumerate(states)]
    results = e._run_jobs(jobs)
    assert e.batch_stats["batched_forwards"] >= 1
    for (out, _), ref in zip(results, solo):
        assert np.abs(out[0, -1] - ref).max() < 2e-3


@pytest.mark.asyncio
@needs_smol
async def test_fused_weights_share_storage_and_finished_requests_free_kv(monkeypatch):
    e, shard = await _engine(True, monkeypatch)
    before = e._pm.loaded_bytes()
    ids = np.asarray(await e.encode(shard, "one two three four five six seven"), dtype=np.int64)
    e._forward_transformer_range(shard, ids, "q", {})
    assert e._pm.loaded_bytes() == before, "fusing q|k|v / gate|up duplicated weights"
    attn = e._pm.layers[0].self_attn
    plan = e._fast().plan(0, e._pm.layers[0])
    assert attn.q_proj.weight.data_ptr() == plan.qkv_w.data_ptr()
    assert e._fast().pool.used_blocks > 0
    e.clear_cache("q")
    assert e._fast().pool.used_blocks == 0
