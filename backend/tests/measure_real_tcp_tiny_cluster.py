"""Measure a real 2-node TCP sharded inference path with a real HF model.

This operator script starts two in-process P2P nodes on real TCP sockets, loads
the same real Hugging Face model into two separate PyTorchHFInferenceEngine
instances, runs shard 0 on node A, sends hidden states over the socket bridge to
node B, runs shard 1 on node B, and sends logits back to node A.
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
from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode


MODEL_ID = "peft-internal-testing/tiny-random-GPT2LMHeadModel"
PROMPT = "The future of AI is"
PORT_A = 6370
PORT_B = 6371


async def wait_connected(node: Libp2pNode, peer_id: str, timeout_s: float = 10.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if node.is_peer_connected(peer_id):
            return True
        await asyncio.sleep(0.05)
    return False


async def main():
    router_a = EventRouter()
    router_b = EventRouter()
    p2p_a = Libp2pNode("TINY-A", PORT_A, router_a)
    p2p_b = Libp2pNode("TINY-B", PORT_B, router_b)
    engine_a = PyTorchHFInferenceEngine()
    engine_b = PyTorchHFInferenceEngine()

    full = Shard(MODEL_ID, 0, 4, 5)
    stage0 = Shard(MODEL_ID, 0, 1, 5)
    stage1 = Shard(MODEL_ID, 2, 4, 5)

    t_load = time.perf_counter()
    tokens = await engine_a.encode(full, PROMPT)
    await engine_b.ensure_shard(stage1)
    load_s = time.perf_counter() - t_load

    await router_a.start()
    await router_b.start()
    asyncio.create_task(p2p_a.start())
    asyncio.create_task(p2p_b.start())
    await asyncio.sleep(0.3)
    await p2p_a.connect_to_peer("TINY-B", f"127.0.0.1:{PORT_B}")
    await p2p_b.connect_to_peer("TINY-A", f"127.0.0.1:{PORT_A}")
    connected = await wait_connected(p2p_a, "TINY-B") and await wait_connected(p2p_b, "TINY-A")
    if not connected:
        raise RuntimeError("P2P nodes did not connect")

    result_queue = asyncio.Queue()
    metrics = {"stage0_s": [], "stage1_s": [], "payload_bytes": []}

    async def handle_b(event):
        data = event.data
        hidden = data["hidden"]
        start_stage1 = time.perf_counter()
        logits, _ = await engine_b.infer_tensor(data["request_id"], stage1, hidden)
        end_stage1 = time.perf_counter()
        await p2p_b.publish("synapse/inference/result/TINY-A", {
            "request_id": data["request_id"],
            "logits": logits,
            "stage0_s": data["stage0_s"],
            "stage1_s": end_stage1 - start_stage1,
            "payload_bytes": int(hidden.nbytes),
            "sent_at": data["sent_at"],
        })

    async def handle_result(event):
        await result_queue.put(event.data)

    router_b.subscribe("synapse/inference/tensor/TINY-B", handle_b)
    router_a.subscribe("synapse/inference/result/TINY-A", handle_result)

    # Warmup both full and sharded paths.
    await engine_a.infer_tensor("warmup-full", full, tokens)
    hidden, _ = await engine_a.infer_tensor("warmup-stage0", stage0, tokens)
    await engine_b.infer_tensor("warmup-stage1", stage1, hidden)
    engine_a.clear_cache()
    engine_b.clear_cache()

    full_times = []
    full_logits = None
    for i in range(3):
        engine_a.clear_cache()
        t = time.perf_counter()
        full_logits, _ = await engine_a.infer_tensor(f"full-{i}", full, tokens)
        full_times.append(time.perf_counter() - t)

    tcp_times = []
    tcp_logits = None
    for i in range(3):
        engine_a.clear_cache()
        engine_b.clear_cache()
        request_id = f"tcp-{i}"
        total_start = time.perf_counter()
        s0_start = time.perf_counter()
        hidden, _ = await engine_a.infer_tensor(request_id, stage0, tokens)
        s0_end = time.perf_counter()
        await p2p_a.publish("synapse/inference/tensor/TINY-B", {
            "request_id": request_id,
            "hidden": hidden,
            "stage0_s": s0_end - s0_start,
            "sent_at": time.perf_counter(),
        })
        result = await asyncio.wait_for(result_queue.get(), timeout=30.0)
        total_s = time.perf_counter() - total_start
        tcp_logits = result["logits"]
        tcp_times.append(total_s)
        metrics["stage0_s"].append(result["stage0_s"])
        metrics["stage1_s"].append(result["stage1_s"])
        metrics["payload_bytes"].append(result["payload_bytes"])

    assert full_logits is not None
    assert tcp_logits is not None
    seq_len = int(tokens.shape[0])
    result = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "seq_len": seq_len,
        "load_two_engines_s": load_s,
        "p2p_connected": connected,
        "full_forward_s": full_times,
        "full_forward_avg_s": float(np.mean(full_times)),
        "full_tokens_per_s": float(seq_len / np.mean(full_times)),
        "tcp_sharded_total_s": tcp_times,
        "tcp_sharded_total_avg_s": float(np.mean(tcp_times)),
        "tcp_sharded_tokens_per_s": float(seq_len / np.mean(tcp_times)),
        "stage0_avg_s": float(np.mean(metrics["stage0_s"])),
        "stage1_avg_s": float(np.mean(metrics["stage1_s"])),
        "hidden_payload_bytes_avg": float(np.mean(metrics["payload_bytes"])),
        "tcp_overhead_avg_s": float(np.mean(tcp_times) - np.mean(metrics["stage0_s"]) - np.mean(metrics["stage1_s"])),
        "max_abs_diff": float(np.max(np.abs(full_logits - tcp_logits))),
        "last_argmax_full": int(full_logits[0, -1].argmax()),
        "last_argmax_tcp_shard": int(tcp_logits[0, -1].argmax()),
    }
    result["argmax_match"] = result["last_argmax_full"] == result["last_argmax_tcp_shard"]
    print(json.dumps(result, indent=2))

    await p2p_a.stop()
    await p2p_b.stop()
    await router_a.stop()
    await router_b.stop()


if __name__ == "__main__":
    asyncio.run(main())
