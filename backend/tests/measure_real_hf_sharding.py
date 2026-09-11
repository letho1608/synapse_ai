"""Measure real Hugging Face forward vs 2-stage sharded forward.

This is an operator script, not a pytest test. It loads a real HF model through
the shipped PyTorchHFInferenceEngine, runs a full forward, then runs the same
prompt as two layer shards and reports real wall-clock timings and output diff.
"""
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
from synapse.inference.shard import Shard


MODEL_ID = "peft-internal-testing/tiny-random-GPT2LMHeadModel"
PROMPT = "The future of AI is"


async def main():
    engine = PyTorchHFInferenceEngine()
    full = Shard(MODEL_ID, 0, 4, 5)
    stage0 = Shard(MODEL_ID, 0, 1, 5)
    stage1 = Shard(MODEL_ID, 2, 4, 5)

    t0 = time.perf_counter()
    tokens = await engine.encode(full, PROMPT)
    load_encode_s = time.perf_counter() - t0

    # Warm the model once so the measured pass is not dominated by one-time init.
    await engine.infer_tensor("warmup-full", full, tokens)
    engine.clear_cache()

    full_times = []
    full_logits = None
    for i in range(3):
        t = time.perf_counter()
        full_logits, _ = await engine.infer_tensor(f"full-{i}", full, tokens)
        full_times.append(time.perf_counter() - t)
        engine.clear_cache()

    shard_times = []
    hidden_payload_bytes = 0
    shard_logits = None
    for i in range(3):
        engine.clear_cache()
        t = time.perf_counter()
        hidden, _ = await engine.infer_tensor(f"shard0-{i}", stage0, tokens)
        mid = time.perf_counter()
        shard_logits, _ = await engine.infer_tensor(f"shard1-{i}", stage1, hidden)
        end = time.perf_counter()
        hidden_payload_bytes = int(hidden.nbytes)
        shard_times.append({"stage0_s": mid - t, "stage1_s": end - mid, "total_s": end - t})

    assert full_logits is not None
    assert shard_logits is not None
    max_abs_diff = float(np.max(np.abs(full_logits - shard_logits)))
    last_full = int(full_logits[0, -1].argmax())
    last_shard = int(shard_logits[0, -1].argmax())
    seq_len = int(tokens.shape[0])

    result = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "tokens": tokens.tolist(),
        "seq_len": seq_len,
        "load_plus_encode_s": load_encode_s,
        "full_forward_s": full_times,
        "full_forward_avg_s": float(np.mean(full_times)),
        "full_tokens_per_s": float(seq_len / np.mean(full_times)),
        "sharded_forward_s": shard_times,
        "sharded_forward_avg_s": float(np.mean([x["total_s"] for x in shard_times])),
        "sharded_tokens_per_s": float(seq_len / np.mean([x["total_s"] for x in shard_times])),
        "hidden_payload_bytes": hidden_payload_bytes,
        "max_abs_diff": max_abs_diff,
        "last_argmax_full": last_full,
        "last_argmax_shard": last_shard,
        "argmax_match": last_full == last_shard,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
