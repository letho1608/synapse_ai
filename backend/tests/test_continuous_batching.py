"""
Continuous batching: concurrent requests' decode steps on the same shard run
in ONE forward (weights read once per batch), and every request still gets
exactly the tokens it gets when it runs alone.

Each request keeps its own KV cache and past length; the batched forward
pads pasts/new rows and masks the padding, then writes each request's new
keys/values back into its own cache.
"""
import asyncio
import sys
import time

import numpy as np
import pytest

TINY = ("peft-internal-testing/tiny-random-GPT2LMHeadModel", 5)
SMOL = ("HuggingFaceTB/SmolLM2-135M-Instruct", 30)


def _cached(model_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


MODELS = [pytest.param(TINY, id="tiny-gpt2")] + [
    pytest.param(SMOL, id="smollm2",
                 marks=pytest.mark.skipif(not _cached(SMOL[0]), reason="not cached"))]

PROMPTS = ["The capital of France is", "def add(a, b):",
           "Water boils at", "Once upon a time, in a small village,"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "none")


def _stages(model_id, n, k):
    from synapse.inference.shard import Shard
    cuts = [round(n * (i + 1) / k) - 1 for i in range(k)]
    out, b = [], 0
    for e in cuts:
        out.append(Shard(model_id, b, e, n))
        b = e + 1
    return out


async def _decode(eng, shards, rid, prompt, steps, delta_rows):
    ids = list(prompt)
    for _ in range(steps):
        x = np.array([ids], dtype=np.int64)
        state = {"delta_rows": delta_rows, "verify_n": 1} if delta_rows else None
        for sh in shards:
            x, state = await eng.infer_tensor(rid, sh, x, state)
        ids.append(int(np.asarray(x).reshape(-1, x.shape[-1])[-1].argmax()))
    return ids[len(prompt):]


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MODELS)
@pytest.mark.parametrize("stages,delta_rows", [(1, False), (2, True), (2, False)])
async def test_concurrent_requests_batch_and_match_solo(spec, stages, delta_rows, monkeypatch):
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    model_id, n = spec
    shards = _stages(model_id, n, stages)
    steps = 10

    monkeypatch.setenv("SYNAPSE_BATCH", "0")
    solo_eng = PyTorchHFInferenceEngine()
    prompts = [[int(t) for t in np.asarray(await solo_eng.encode(shards[0], p)).reshape(-1)]
               for p in PROMPTS]
    solo = [await _decode(solo_eng, shards, f"s{i}", p, steps, delta_rows)
            for i, p in enumerate(prompts)]

    monkeypatch.setenv("SYNAPSE_BATCH", "1")
    eng = PyTorchHFInferenceEngine()
    together = await asyncio.gather(*[
        _decode(eng, shards, f"c{i}", p, steps, delta_rows) for i, p in enumerate(prompts)])
    assert together == solo
    assert eng.batch_stats["max_batch"] >= 2, eng.batch_stats
    assert eng.batch_stats["batched_jobs"] >= steps  # most steps really shared a forward


@pytest.mark.asyncio
async def test_lone_request_pays_no_batching_window(monkeypatch):
    """With one active request the drain task never sleeps for company."""
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    model_id, n = TINY
    monkeypatch.setenv("SYNAPSE_BATCH", "1")
    monkeypatch.setenv("SYNAPSE_BATCH_WINDOW_MS", "200")
    eng = PyTorchHFInferenceEngine()
    shard = _stages(model_id, n, 1)[0]
    await _decode(eng, [shard], "warm", [1, 2, 3], 1, False)
    eng.clear_cache("warm")  # request finished
    t0 = time.perf_counter()
    await _decode(eng, [shard], "alone", [1, 2, 3], 5, False)
    assert time.perf_counter() - t0 < 0.9  # 5 steps x 200 ms window would be 1 s
    assert eng.batch_stats["batched_forwards"] == 0


@pytest.mark.asyncio
async def test_failing_job_does_not_break_its_batch_mates(monkeypatch):
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine, KVMissError
    from synapse.inference.shard import Shard
    model_id, n = TINY
    monkeypatch.setenv("SYNAPSE_BATCH", "1")
    eng = PyTorchHFInferenceEngine()
    good = _decode(eng, [Shard(model_id, 0, n - 1, n)], "ok", [1, 2, 3], 3, False)
    bad = eng.infer_tensor("bad", Shard(model_id, 2, n - 1, n),
                           np.zeros((1, 1, 2), dtype=np.float32), {"row_offset": 5})
    res = await asyncio.gather(good, bad, return_exceptions=True)
    assert isinstance(res[1], KVMissError)
    assert len(res[0]) == 3
