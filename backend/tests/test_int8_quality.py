"""
Behavioral tests: INT8 wire quantization (default ON) and its quality gate.

Codec: per-row symmetric INT8 + the 8 largest-|x| entries per row carried
exactly. Replaces the old single per-tensor scale, which on a real model
flattened hidden states (teacher-forced top-1 agreement ~1-6%), truncated
instead of rounding, returned all-zero float tensors as int8, and silently
dropped request_id when metadata exceeded 1 KiB.

Quality gate (real model, SmolLM2-135M-Instruct = the project default, CPU,
local cache only — skipped if not downloaded): split the model into 2 and 3
pipeline stages, send the inter-stage hidden states through encode/decode,
and compare teacher-forced next-token logits against exact transport.
"""
import os
import sys

import numpy as np
import pytest

from synapse.transport.codec import decode_tensor, encode_tensor

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_LAYERS = 30
PROMPTS = ["The capital of France is", "def fibonacci(n):",
           "Explain photosynthesis in simple terms.", "Hà Nội là thủ đô của",
           "Write a short poem about the sea.", "1, 2, 3, 4, 5,"]
GEN = 24


def _rt(x, **meta):
    return decode_tensor(encode_tensor(x, metadata=meta or None, quantize_int8=True))


def test_roundtrip_error_bounded_and_outliers_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 7, 64)).astype(np.float32)
    x[0, :, 5] = 300.0  # a "massive activation" channel
    y, meta = _rt(x, request_id="r1")
    assert y.dtype == np.float32 and y.shape == x.shape
    assert meta["request_id"] == "r1"
    assert np.array_equal(y[0, :, 5], x[0, :, 5]), "outliers must be carried exactly"
    rest = np.delete(x, 5, axis=2)
    err = np.abs(np.delete(y, 5, axis=2) - rest)
    # Without the outlier the per-row step is max|rest|/127 -> error <= half a step.
    step = np.abs(rest).max(axis=2, keepdims=True) / 127.0
    assert (err <= step / 2 + 1e-6).all()


def test_edge_shapes_and_dtypes():
    for x in (np.zeros((2, 3), np.float32), np.array(1.5, dtype=np.float32),
              np.arange(5, dtype=np.float16), np.zeros((0, 4), np.float32),
              np.full((1, 3), -2.0, np.float64)):
        y, _ = _rt(x)
        assert y.shape == x.shape and y.dtype == x.dtype, (x.shape, x.dtype)
        assert np.allclose(y, x, atol=np.abs(x).max(initial=0) / 100 + 1e-6)
    # Integer tensors (token ids) are never quantized.
    ids = np.arange(10, dtype=np.int64)
    y, _ = _rt(ids)
    assert y.dtype == np.int64 and np.array_equal(y, ids)


def test_prefix_rows_stay_byte_identical():
    """Per-row quantization is deterministic per row: prefix rows decode to
    the same bytes when the sequence grows, so downstream stages keep their
    KV-cache hits under INT8 transport."""
    rng = np.random.default_rng(1)
    full = rng.normal(size=(1, 9, 32)).astype(np.float32)
    a, _ = _rt(full[:, :8])
    b, _ = _rt(full)
    assert np.array_equal(a, b[:, :8])


def test_legacy_per_tensor_payload_still_decodes():
    import json, struct
    from synapse.transport import codec
    x = np.array([[1.0, -2.0]], np.float32)
    scale = 2.0 / 127.0
    q = np.rint(x / scale).astype(np.int8)
    meta = json.dumps({"_q_scale": scale, "_orig_dtype": "float32"}).encode()
    blob = (struct.pack("!4sHHII", codec.MAGIC, codec.DTYPE_MAP[q.dtype], 2, q.nbytes, q.nbytes << 2)
            + struct.pack("!HH", 1, 2) + struct.pack("!H", len(meta)) + meta + q.tobytes())
    y, _ = decode_tensor(blob)
    assert y.dtype == np.float32 and np.allclose(y, x, atol=scale)


def test_large_metadata_kept_not_pruned():
    x = np.ones((1, 4), np.float32)
    big = "s" * 5000
    y, meta = _rt(x, request_id="keep-me", blob=big)
    assert meta["request_id"] == "keep-me" and meta["blob"] == big


def _model_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)
    except Exception:
        return False


@pytest.fixture
def _cpu_only(monkeypatch):
    # Same env guard as test_seq_kv_parity (DirectML DLL mismatch on this box).
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.mark.asyncio
@pytest.mark.skipif(not _model_cached(), reason=f"{MODEL} not in local HF cache")
async def test_int8_quality_gate_real_model(_cpu_only):
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard

    eng = PyTorchHFInferenceEngine()
    full = Shard(MODEL, 0, N_LAYERS - 1, N_LAYERS)

    async def fwd(splits, ids, quant):
        x, b = np.array(ids, dtype=np.int64), 0
        bounds = splits + [N_LAYERS - 1]
        for i, e in enumerate(bounds):
            x, _ = await eng.infer_tensor(None, Shard(MODEL, b, e, N_LAYERS), x)
            b = e + 1
            if i < len(bounds) - 1 and quant:
                x, _ = decode_tensor(encode_tensor(np.asarray(x, np.float32), quantize_int8=True))
        return np.asarray(x).reshape(-1, x.shape[-1])

    seqs = []
    for p in PROMPTS:
        ids = list(np.asarray(await eng.encode(full, p)).reshape(-1))
        n0 = len(ids)
        for _ in range(GEN):
            ids.append(int((await fwd([], ids, False))[-1].argmax()))
        seqs.append((ids, n0))

    def logsm(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    for splits in ([14], [9, 19]):
        agree, kls = [], []
        for ids, n0 in seqs:
            ref = await fwd(splits, ids, False)
            got = await fwd(splits, ids, True)
            sl = slice(n0 - 1, len(ids) - 1)
            agree.append(np.mean(ref[sl].argmax(-1) == got[sl].argmax(-1)))
            lp, lq = logsm(ref[sl]), logsm(got[sl])
            kls.append(np.mean(np.sum(np.exp(lp) * (lp - lq), -1)))
        print(f"stages={len(splits) + 1} top1={np.mean(agree):.3f} min={np.min(agree):.3f} "
              f"KL={np.mean(kls):.2e}")
        assert np.mean(agree) >= 0.98, (splits, agree)
        assert np.min(agree) >= 0.95, (splits, agree)
        assert np.mean(kls) <= 1e-3, (splits, kls)
