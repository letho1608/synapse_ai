"""
Behavioral tests: per-request incremental KV parity on a real model (T3.1).

Loads the cached tiny-random GPT-2 on CPU (no network: local_files_only)
and greedy-decodes: full-refeed baseline (SYNAPSE_SEQ_KV=0) vs incremental
reuse (SYNAPSE_SEQ_KV=1). Asserts TOKEN-IDENTICAL output plus proof the fast
path actually engaged (seq state length == final length). A divergent
continuation must fall back silently and still match full-refeed math.

Perf multiples are NOT asserted here (CPU tiny model); mechanism parity is.
"""
import numpy as np
import pytest

from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
from synapse.inference.shard import Shard

MODEL_ID = "peft-internal-testing/tiny-random-GPT2LMHeadModel"
PROMPT = "The future of AI is"
STEPS = 8


@pytest.fixture(autouse=True)
def _no_broken_directml(monkeypatch):
    """This box's torch_directml has a DLL mismatch vs torch 2.13 CPU:
    importing it AFTER torch is loaded is intermittently FATAL (0xc0000139),
    standalone it raises ImportError. Poison the import so _has_directml()
    takes its clean ImportError path and the engine runs CPU. This masks no
    product behavior (DirectML is irrelevant on CPU); the env owner should
    reinstall a matching torch-directml build for the Intel-GPU path."""
    import sys
    monkeypatch.setitem(sys.modules, "torch_directml", None)


async def _greedy_decode(engine, shard, prompt, request_id):
    tokens = await engine.encode(shard, prompt)
    ids = list(np.asarray(tokens).reshape(-1))
    for _ in range(STEPS):
        out, _ = await engine.infer_tensor(request_id, shard,
                                           np.array(ids, dtype=np.int64))
        nxt = int(np.asarray(out).reshape(-1, out.shape[-1])[0, -1].argmax())
        ids.append(nxt)
    return ids


@pytest.mark.asyncio
async def test_seq_kv_parity_with_tiny_gpt2(monkeypatch):
    import os
    monkeypatch.setenv("SYNAPSE_SEQ_KV", "0")
    engine = PyTorchHFInferenceEngine()
    shard = Shard(MODEL_ID, 0, 4, 5)

    base_ids = await _greedy_decode(engine, shard, PROMPT, "parity-base")

    engine.clear_cache()
    monkeypatch.setenv("SYNAPSE_SEQ_KV", "1")
    seq_ids = await _greedy_decode(engine, shard, PROMPT, "parity-seq")

    assert seq_ids == base_ids, "incremental reuse must match full-refeed math"
    st = engine._seq_kv.get("parity-seq@0-4")
    assert st is not None, "fast path never engaged (vacuous parity)"
    # Last committed input was the final decode input (output has +1 token).
    prompt_len = len(await engine.encode(shard, PROMPT))
    assert st["length"] == prompt_len + STEPS - 1 == len(seq_ids) - 1
    engine.clear_cache("parity-seq")
    assert not any(k.startswith("parity-seq@") for k in engine._seq_kv)


@pytest.mark.asyncio
async def test_seq_kv_divergence_crops_to_common_prefix(monkeypatch):
    """A continuation that diverges from the cached input (e.g. a rejected
    speculative guess) crops the cache back to the common prefix and
    recomputes only the rows after it. Those rows must equal a fresh run."""
    monkeypatch.setenv("SYNAPSE_SEQ_KV", "1")
    engine = PyTorchHFInferenceEngine()
    shard = Shard(MODEL_ID, 0, 4, 5)
    tokens = await engine.encode(shard, PROMPT)
    ids = list(np.asarray(tokens).reshape(-1))
    out, _ = await engine.infer_tensor("div", shard, np.array(ids, dtype=np.int64))
    flipped = ids[:-1] + [(ids[-1] + 7) % 100]
    out2, _ = await engine.infer_tensor("div", shard, np.array(flipped, dtype=np.int64))
    out2 = np.asarray(out2).reshape(-1, np.asarray(out2).shape[-1])
    assert out2.shape[0] == 1  # only the diverging row was recomputed
    assert engine._seq_kv[f"div@0-4"]["length"] == len(flipped)
    out3, _ = await engine.infer_tensor("div-fresh", shard, np.array(flipped, dtype=np.int64))
    out3 = np.asarray(out3).reshape(-1, out2.shape[-1])
    assert np.allclose(out2, out3[-1:], atol=1e-4)
