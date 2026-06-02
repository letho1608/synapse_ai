"""
Benchmark: distributed inference throughput on 3-node pipeline.
3 TCP-connected nodes, simulated compute, EventRouter transport.
Accurate pipeline timing with concurrent micro-batches.
"""
import asyncio, sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode

NODES = [("NODE-A", 5981, 5), ("NODE-B", 5982, 10), ("NODE-C", 5983, 15)]
LAYER_RANGES = [(0, 12), (13, 17), (18, 27)]
TOTAL_LAYERS = 28
GEN_TOKENS = 50

class BenchNode:
    def __init__(self, node_id, port, ms_per_layer):
        self.id = node_id; self.port = port; self.ms_per_layer = ms_per_layer
        self.router = EventRouter()
        self.libp2p = Libp2pNode(node_id, port, self.router)
        self.compute_s = 0.0; self.count = 0
    async def start(self):
        await self.router.start(); asyncio.create_task(self.libp2p.start())
    async def connect_to(self, pid, addr): return await self.libp2p.connect_to_peer(pid, addr)
    async def stop(self):
        await self.libp2p.stop(); await self.router.stop()

async def bench():
    nodes = [BenchNode(nid, p, ms) for nid, p, ms in NODES]
    for n in nodes: await n.start()
    await asyncio.sleep(0.3)
    for n in nodes:
        for o in nodes:
            if n.id != o.id: await n.connect_to(o.id, f"127.0.0.1:{o.port}")
    await asyncio.sleep(0.3)
    for n in nodes:
        for o in nodes:
            if n.id != o.id and not n.libp2p.is_peer_connected(o.id):
                print(f"[FAIL] {n.id} not connected to {o.id}")
                return

    print("[MESH] 3-node full mesh: OK")

    # Compute delay per node per token
    delays = []
    for i in range(3):
        sr, er = LAYER_RANGES[i]
        lc = er - sr + 1
        delays.append(lc * NODES[i][2] / 1000)
    period = max(delays)
    pipeline_latency = sum(delays)
    theoretical_tps = 1.0 / period

    # Set up pipeline: token flow through A -> B -> C
    sem = asyncio.Semaphore(4)  # max concurrent tokens in pipeline

    async def stage_handler(node_idx, event):
        async with sem:
            data = event.data
            step = data["step"]
            # Compute
            await asyncio.sleep(delays[node_idx])
            nodes[node_idx].compute_s += delays[node_idx]
            nodes[node_idx].count += 1
            if node_idx < 2:
                await nodes[node_idx].router.publish(
                    f"synapse/inference/tensor/{NODES[node_idx+1][0]}", data)
            else:
                # Last node: record completion time
                t = time.perf_counter()
                data["t_finish"] = t
                await nodes[0].router.publish("synapse/inference/result/bench", data)

    async def make_handler(idx):
        async def handler(event):
            await stage_handler(idx, event)
        return handler

    for i in range(3):
        nodes[i].router.subscribe(
            f"synapse/inference/tensor/{NODES[i][0]}",
            await make_handler(i))

    completed = []
    result_future = asyncio.Future()
    def on_result(event):
        completed.append(event.data)
        if len(completed) >= GEN_TOKENS and not result_future.done():
            result_future.set_result(True)
    nodes[0].router.subscribe("synapse/inference/result/bench", on_result)

    await asyncio.sleep(0.2)

    print(f"\n[INFER] Generating {GEN_TOKENS} tokens...")
    print(f"  Stage compute per token: A={delays[0]*1000:.0f}ms, "
          f"B={delays[1]*1000:.0f}ms, C={delays[2]*1000:.0f}ms")
    print(f"  Pipeline period (slowest stage): {period*1000:.0f}ms")
    print(f"  Theoretical throughput: {theoretical_tps:.2f} tok/s")
    print(f"  Theoretical latency: {pipeline_latency*1000:.0f}ms")

    # Measure single-token latency
    t0 = time.perf_counter()
    await nodes[0].router.publish("synapse/inference/tensor/NODE-A",
        {"step": -1, "t_start": t0})
    await asyncio.sleep(5.0)
    if completed:
        single_lat = completed[0]["t_finish"] - completed[0]["t_start"]
        completed.clear()
    else:
        single_lat = time.perf_counter() - t0

    # Full benchmark with pipelining
    completed.clear()
    t_start = time.perf_counter()
    for step in range(GEN_TOKENS):
        await nodes[0].router.publish("synapse/inference/tensor/NODE-A",
            {"step": step, "t_start": time.perf_counter()})

    try:
        await asyncio.wait_for(result_future, timeout=30.0)
    except asyncio.TimeoutError:
        pass
    total_time = time.perf_counter() - t_start

    throughput = len(completed) / total_time if total_time > 0 else 0
    first_finish = completed[0]["t_finish"] if completed else t_start
    last_finish = completed[-1]["t_finish"] if completed else t_start
    steady_tokens = len(completed) - 1 if len(completed) > 1 else 1
    steady_time = last_finish - first_finish if len(completed) > 1 else total_time
    steady_tps = steady_tokens / steady_time if steady_time > 0 else 0

    print(f"\n{'='*65}")
    print(f"KET QUA BENCHMARK 3 NODE PHAN TAN")
    print(f"{'='*65}")
    print(f"  Tong token sinh ra:         {len(completed)}")
    print(f"  Tong thoi gian:             {total_time:.3f}s")
    print(f"  Throughput trung binh:      {throughput:.2f} tokens/s")
    print(f"  Throughput steady-state:    {steady_tps:.2f} tokens/s")
    print(f"  Latency 1 token (end2end):  {single_lat*1000:.1f}ms")
    print(f"  Pipeline latency (ly thuyet):{pipeline_latency*1000:.0f}ms")
    print()

    # Compare with theoretical
    eff = steady_tps / theoretical_tps * 100 if theoretical_tps > 0 else 0
    print(f"  Pipeline period thuc te:    {(1.0/steady_tps)*1000:.0f}ms "
          f"(ly thuyet: {period*1000:.0f}ms)")
    print(f"  Hieu suat pipeline:         {eff:.1f}%")
    print()

    total_compute_s = sum(n.compute_s for n in nodes)
    bottleneck_note = NODES[[d for d in delays].index(max(delays))][0]
    print(f"  Nut nghen (bottleneck):     {bottleneck_note} "
          f"({max(delays)*1000:.0f}ms/token)")
    print()
    for i in range(3):
        sr, er = LAYER_RANGES[i]
        lc = er - sr + 1
        print(f"  {NODES[i][0]:<10}: {lc} layers [{sr:2}-{er:2}] "
              f"@{NODES[i][2]}ms/layer = {delays[i]*1000:.0f}ms/tok "
              f"(count={nodes[i].count})")

    for n in nodes: await n.stop()
    return throughput

asyncio.run(bench())
