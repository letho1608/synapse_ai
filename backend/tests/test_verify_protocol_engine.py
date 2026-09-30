"""
The distributed decode protocol on the REAL engine, without the network.

Each step sends [prompt + history + draft] through 2-3 stages with
delta_rows (non-last stages forward only recomputed rows + row_offset) and
per-row INT8 between stages, samples every draft position, and keeps the
verified prefix. The result must equal plain greedy decoding.

The engine always recomputes at least the last verify_n rows (the draft
positions), even when a new draft sits on rows cached by the previous step,
so the last stage can never return fewer rows than positions to sample.
"""
import sys

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


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SYNAPSE_SEQ_KV", "1")
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "none")


def _stages(model_id, n, k):
    from synapse.inference.shard import Shard
    cuts = [round(n * (i + 1) / k) - 1 for i in range(k)]
    out, b = [], 0
    for e in cuts:
        out.append(Shard(model_id, b, e, n))
        b = e + 1
    return out


async def _greedy_reference(eng, shard_full, prompt, steps):
    """No cache, no draft: one full forward per token."""
    ids = list(prompt)
    for _ in range(steps):
        x, _ = await eng.infer_tensor(None, shard_full, np.array(ids, dtype=np.int64))
        ids.append(int(np.asarray(x).reshape(-1, x.shape[-1])[-1].argmax()))
    return ids[len(prompt):]


async def _protocol(eng, shards, prompt, steps, drafter, int8=True):
    from synapse.inference.distributed_engine import verify_speculative_tokens
    from synapse.transport.codec import decode_tensor, encode_tensor
    gen, passes = [], 0
    while len(gen) < steps:
        draft = drafter(prompt, gen)[: max(0, steps - len(gen) - 1)]
        x = np.array([prompt + gen + draft], dtype=np.int64)
        state = {"delta_rows": True, "verify_n": len(draft) + 1}
        for sh in shards:
            x, state = await eng.infer_tensor("proto", sh, x, state)
            if int8 and not sh.is_last_layer():
                x, _ = decode_tensor(encode_tensor(np.asarray(x, np.float32), metadata={},
                                                   quantize_int8=True))
        rows = np.asarray(x).reshape(-1, x.shape[-1])
        n = len(draft) + 1
        assert rows.shape[0] >= n, "last stage returned fewer rows than draft positions"
        sampled = [int(rows[rows.shape[0] - n + j].argmax()) for j in range(n)]
        gen += verify_speculative_tokens(draft, [[t] for t in sampled])
        passes += 1
    eng.clear_cache("proto")
    return gen[:steps], passes


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MODELS)
@pytest.mark.parametrize("k", [2, 3])
async def test_protocol_with_adversarial_drafts_matches_greedy(spec, k):
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    model_id, n = spec
    eng = PyTorchHFInferenceEngine()
    shards = _stages(model_id, n, k)
    prompt = [int(t) for t in np.asarray(await eng.encode(
        shards[0], "The capital of France is Paris. The capital of France is")).reshape(-1)]
    steps = 14
    ref = await _greedy_reference(eng, Shard(model_id, 0, n - 1, n), prompt, steps)

    # Drafts that are right, partly right, and THE SAME every step (so the
    # new draft sits on top of the previous step's cached draft rows).
    def fixed(_p, gen):
        return [ref[len(gen)] if len(gen) < len(ref) else 0, 7, 7]

    def always_same(_p, _gen):
        return [11, 12, 13]

    def correct(_p, gen):
        return ref[len(gen): len(gen) + 3]

    for drafter in (fixed, always_same, correct):
        got, passes = await _protocol(eng, shards, prompt, steps, drafter)
        assert got == ref, drafter.__name__
    # A fully correct draft verifies 3 guesses + 1 per pass.
    _, passes = await _protocol(eng, shards, prompt, steps, correct)
    assert passes <= -(-steps // 4) + 1
