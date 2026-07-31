"""
B6+B7 cleanup: honest ILP/placeholder-prober removal is verified by grep
evidence; this module carries the behavioral tests that enforce the hygiene
items:

  - `MicroBatch.step` is actually SET by the shipped pipeline paths (was never
    set, so `StageResult.step` always fell back to 0),
  - the SHM transport's retained-segment set (`_active`) is bounded instead of
    growing without limit,
  - the README scaling table equals `benchmark_cluster_speedup.py` output
    (the invariant is enforced by a test, not by hope).
"""
import re
import sys
from pathlib import Path

import numpy as np

from synapse.pipeline.scheduler import MicroBatch
from synapse.pipeline.stage import PipelineExecutor, PipelineStage
from synapse.transport.shm_transport import SharedMemoryTransport

# ---------------------------------------------------------------------------
# MicroBatch.step is set by the shipped pipeline paths
# ---------------------------------------------------------------------------


async def test_micro_batch_step_set_in_single_stage_inference():
    """run_inference(single-stage) must set step = micro-batch index."""
    seen = []

    async def forward(mb):
        seen.append((mb.batch_id, mb.step))
        return np.array([mb.step], dtype=np.float32)

    async def send(_result):
        pass

    stage = PipelineStage(0, "node-a", 1, forward, send)
    executor = PipelineExecutor(stages=[stage])
    results = await executor.run_inference(np.zeros(4), num_micro_batches=4)

    # The forward function received a real step, and the StageResult carries it.
    assert seen == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert [int(r[0]) for r in results] == [0, 1, 2, 3]


async def test_micro_batch_step_flows_through_multi_stage_pipeline():
    """Multi-stage run_inference sets step from the schedule slot and the
    step survives the inter-stage send chain (no hasattr fallback to 0)."""
    seen_stage0 = []

    async def forward0(mb):
        seen_stage0.append((mb.batch_id, mb.step))
        return np.array([mb.batch_id, mb.step], dtype=np.float32)

    async def forward1(mb):
        return np.array([mb.batch_id, mb.step], dtype=np.float32)

    async def send_last(_result):
        pass

    stage1 = PipelineStage(1, "node-b", 2, forward1, send_last)

    async def send0_to_1(result):
        # The shipped send path hands the next stage a fresh MicroBatch
        # carrying the step produced by the previous stage's result.
        mb = MicroBatch(
            batch_id=result.micro_batch_id,
            data=result.data,
            metadata={"step": result.step},
            step=result.step,
        )
        await stage1._input_queue.put(mb)

    stage0 = PipelineStage(0, "node-a", 2, forward0, send0_to_1)

    executor = PipelineExecutor(stages=[stage0, stage1])
    results = await executor.run_inference(np.zeros((2, 4)), num_micro_batches=3)

    # Stage-0 inference slots for 2 stages / 3 micro-batches: steps 0,1,2.
    assert sorted(seen_stage0) == [(0, 0), (1, 1), (2, 2)]
    # The last stage's outputs carry the same steps through the send chain.
    assert sorted((int(r[0]), int(r[1])) for r in results) == [(0, 0), (1, 1), (2, 2)]


# ---------------------------------------------------------------------------
# SHM transport: _active is bounded
# ---------------------------------------------------------------------------


async def test_shm_active_segments_bounded():
    """Retained segment handles are capped (LRU), delivery unaffected."""
    a = SharedMemoryTransport("node-a", max_active_segments=4)
    b = SharedMemoryTransport("node-b", max_active_segments=4)
    await a.start()
    await b.start()
    received: list = []
    b.set_tensor_handler(lambda msg: received.append(msg))
    a.register_peer("node-b:0", "node-b")

    for i in range(20):
        await a.send_tensor(f"req-{i}", 0, np.array([i], dtype=np.int64), "node-b:0")

    # The retained-handle set stays at the cap on both sides.
    assert len(a._active) <= 4
    assert len(b._active) <= 4
    # Delivery is unaffected: every segment still arrives.
    assert [m.request_id for m in received] == [f"req-{i}" for i in range(20)]
    assert b.shm_recvs == 20
    # Eviction is LRU: the newest segments are still retained.
    assert a.segments_sent[-1] in a._active
    assert b.segments_seen[-1] in b._active

    await a.stop()
    await b.stop()


# ---------------------------------------------------------------------------
# README scaling table == benchmark_cluster_speedup.py output
# ---------------------------------------------------------------------------

# README rows: | Nodes | Stage | Latency | Serialize | Micro-batches | Speedup | Efficiency |
_README_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)x\s*\|\s*(\d+)%\s*\|$"
)
# Benchmark rows add Single/Pipeline columns before Speedup.
_BENCH_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)x\s*\|\s*(\d+)%\s*\|$"
)


def _readme_scaling_rows():
    readme = Path(__file__).resolve().parents[2] / "README.md"
    rows = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        m = _README_ROW.match(line)
        if m:
            rows.append(tuple(m.groups()))
    return rows


def _benchmark_scaling_rows():
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from benchmark_cluster_speedup import benchmark_table

    rows = []
    for line in benchmark_table().splitlines():
        m = _BENCH_ROW.match(line)
        if m:
            rows.append(tuple(m.groups()))
    return rows


def test_readme_scaling_table_matches_benchmark_output():
    """The README scaling table is exactly the benchmark's generated table."""
    readme_rows = _readme_scaling_rows()
    bench_rows = _benchmark_scaling_rows()

    assert len(readme_rows) == len(bench_rows) > 0, (
        f"README has {len(readme_rows)} scaling rows, benchmark has "
        f"{len(bench_rows)} — keep the two tables in lockstep."
    )
    assert readme_rows == bench_rows, (
        "README scaling table diverges from benchmark_cluster_speedup.py output.\n"
        f"README rows:    {readme_rows}\n"
        f"Benchmark rows: {bench_rows}\n"
        "Regenerate the README table from `python backend/tests/benchmark_cluster_speedup.py`."
    )
