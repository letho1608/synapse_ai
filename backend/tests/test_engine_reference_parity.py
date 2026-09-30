"""
Behavioral tests: the shard engine computes the SAME function as the HF model.

Found 2026-09-29 (measuring INT8 quality): on CPU the engine used "eager"
attention with attention_mask=None, i.e. NO causal mask — every token saw the
future, and SmolLM2 answered "The capital of France is land". Under
transformers 5 the layers also take `past_key_values` (a Cache object) and
return a bare tensor, so the old `past_key_value` per-layer tuples were never
filled while cache "hits" still sliced the input down to the new tokens, and
a non-first pipeline stage got only the new tokens' hidden states. The old
parity test compared two equally-broken paths.

These tests pin the engine to the HF reference: full-shard logits vs
model(input_ids), and multi-step greedy decode through a 2-/3-stage split
with KV reuse vs model.generate. Tiny GPT-2 always runs; the project's
default model runs when cached locally.
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
def _cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SYNAPSE_SEQ_KV", "1")


def _hf(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    m.eval()
    return m


@pytest.mark.parametrize("spec", MODELS)
def test_base_shard_uses_real_layer_count(spec):
    """Models missing from the static layer table were registered with a
    guessed 28 layers, so the API's single-node run of SmolLM2 (30 layers)
    skipped 2 blocks and answered fluent nonsense. The count now comes from
    config.json."""
    from synapse.inference.inference_engine import get_inference_engine
    from synapse.models import build_base_shard
    get_inference_engine("pytorch", None)  # registers the model list
    model_id, n = spec
    assert build_base_shard(model_id, "PyTorchHFInferenceEngine").n_layers == n


@pytest.mark.asyncio
async def test_engine_refuses_wrong_layer_count():
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    model_id, n = TINY
    eng = PyTorchHFInferenceEngine()
    ids = np.arange(4, dtype=np.int64)
    with pytest.raises(RuntimeError, match="n_layers"):
        await eng.infer_tensor(None, Shard(model_id, 0, n - 2, n - 1), ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MODELS)
async def test_full_shard_logits_match_hf(spec):
    import torch
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    model_id, n = spec
    eng = PyTorchHFInferenceEngine()
    shard = Shard(model_id, 0, n - 1, n)
    ids = np.asarray(await eng.encode(shard, "The capital of France is"), dtype=np.int64).reshape(-1)
    got, _ = await eng.infer_tensor(None, shard, ids)
    with torch.no_grad():
        ref = _hf(model_id)(torch.from_numpy(ids)[None]).logits[0].numpy()
    got = np.asarray(got).reshape(ref.shape)
    assert np.allclose(got, ref, atol=1e-3), float(np.abs(got - ref).max())


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MODELS)
@pytest.mark.parametrize("stages", [1, 2, 3])
async def test_split_decode_with_kv_reuse_matches_hf_generate(spec, stages, monkeypatch):
    import torch
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    model_id, n = spec
    steps = 12
    eng = PyTorchHFInferenceEngine()
    cuts = [round(n * (i + 1) / stages) - 1 for i in range(stages)]
    shards, b = [], 0
    for e in cuts:
        shards.append(Shard(model_id, b, e, n))
        b = e + 1
    ids = list(np.asarray(await eng.encode(shards[0], "The capital of France is"),
                          dtype=np.int64).reshape(-1))
    prompt = list(ids)
    for _ in range(steps):
        x = np.array(ids, dtype=np.int64)
        for i, sh in enumerate(shards):
            x, _ = await eng.infer_tensor(f"req-{stages}-{i}", sh, x)
        ids.append(int(np.asarray(x).reshape(-1, x.shape[-1])[-1].argmax()))
    # The fast path really engaged on every stage (not a vacuous pass).
    for i, sh in enumerate(shards):
        st = eng._seq_kv.get(f"req-{stages}-{i}@{sh.start_layer}-{sh.end_layer}")
        assert st is not None and st["length"] == len(ids) - 1
    with torch.no_grad():
        ref = _hf(model_id).generate(torch.tensor([prompt]), max_new_tokens=steps,
                                     do_sample=False, min_new_tokens=steps)[0].tolist()
    assert ids == ref


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MODELS)
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("seq_kv_flag", ["1", "0"])
async def test_delta_convention_matches_hf_generate(spec, stages, seq_kv_flag, monkeypatch):
    """The single-node chat loop (node_extension._run_single_node) and ring
    forwarding send the prompt once, then ONLY the newest token (and each
    stage forwards only the new hidden rows), passing back the returned
    inference_state. This path produced garbage through the real API
    ("Din Son T shallow, ..."): every step ran without context. Works with
    SYNAPSE_SEQ_KV on or off (the flag only gates full-context reuse)."""
    import torch
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    monkeypatch.setenv("SYNAPSE_SEQ_KV", seq_kv_flag)
    model_id, n = spec
    steps = 12
    eng = PyTorchHFInferenceEngine()
    cuts = [round(n * (i + 1) / stages) - 1 for i in range(stages)]
    shards, b = [], 0
    for e in cuts:
        shards.append(Shard(model_id, b, e, n))
        b = e + 1
    prompt = list(np.asarray(await eng.encode(shards[0], "The capital of France is"),
                             dtype=np.int64).reshape(-1))
    out, state, x = [], None, np.array([prompt], dtype=np.int64)
    for _ in range(steps):
        for sh in shards:  # state travels the ring, like the node path
            x, state = await eng.infer_tensor("chat-req", sh, x, state)
        tok = int(np.asarray(x).reshape(-1, x.shape[-1])[-1].argmax())
        out.append(tok)
        x = np.array([[tok]], dtype=np.int64)  # delta: newest token only
    with torch.no_grad():
        ref = _hf(model_id).generate(torch.tensor([prompt]), max_new_tokens=steps,
                                     do_sample=False, min_new_tokens=steps)[0].tolist()
    assert prompt + out == ref
    eng.clear_cache("chat-req")
    assert not eng._seq_kv


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MODELS)
async def test_node_process_tensor_convention_matches_hf_generate(spec):
    """Exactly what the API path does (node._process_prompt ->
    _process_tensor): the prompt goes in as int32 ids, then every step the
    first layer rebuilds prompt + generated as int64 while the loop keeps
    passing the FIRST step's inference_state (stale kv_len). Must be read as
    a full-context extension, not as "append" (which duplicated the prompt)."""
    import torch
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    model_id, n = spec
    steps = 12
    eng = PyTorchHFInferenceEngine()
    shard = Shard(model_id, 0, n - 1, n)
    prompt = np.asarray(await eng.encode(shard, "The capital of France is")).reshape(-1)
    x, first_state = await eng.infer_tensor("api-req", shard, prompt.astype(np.int32), None)
    gen = [int(np.asarray(x).reshape(-1, x.shape[-1])[-1].argmax())]
    for _ in range(steps - 1):
        full = np.concatenate([prompt.astype(np.int64), np.array(gen, dtype=np.int64)])
        x, _ = await eng.infer_tensor("api-req", shard, full, first_state)
        gen.append(int(np.asarray(x).reshape(-1, x.shape[-1])[-1].argmax()))
    with torch.no_grad():
        ref = _hf(model_id).generate(torch.tensor([prompt.astype(np.int64)]), max_new_tokens=steps,
                                     do_sample=False, min_new_tokens=steps)[0].tolist()
    assert prompt.tolist() + gen == ref
