"""
Behavioral test: the ILP partition path carries a solve time budget and
degrades to the fast LRM method when the budget is exceeded — the cluster
still receives a valid contiguous partition instead of stalling on a solver
that runs unbounded.

Verifies (acceptance: "ILP time-bounded"):
1. compute_partition(method="ilp", time_limit_seconds=...) caps the CBC solve
   (PULP_CBC_CMD timeLimit) — wall time stays bounded on a large instance
   (12 nodes x 32 layers: ~68k ILP constraints).
2. On budget exhaustion the shipped path degrades to LRM, observable via the
   stamped LACPPartition.method_used/solver_status fields and a logged
   warning.
3. The degraded partition is still valid: contiguous ranges, every layer
   covered exactly once.
"""
import logging
import time

import pytest

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def _make_nodes(n: int):
    from synapse.topology.resource_weight import NodeResources

    return {
        f"node{i}": NodeResources(
            gpu_vram_mb=4096 + i * 512,
            gpu_tflops_fp16=10.0 + i * 1.0,
            gpu_tflops_fp32=5.0 + i * 0.5,
            cpu_cores=4 + i,
            system_ram_mb=8192 + i * 1024,
        )
        for i in range(n)
    }


@pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
def test_ilp_time_budget_bounds_solve_and_falls_back(caplog):
    """A large ILP instance (12 nodes x 32 layers) with a 1 s budget returns
    a valid contiguous partition in bounded time and degrades observably."""
    from synapse.topology.partitioner import compute_partition

    nodes = _make_nodes(12)
    total_layers = 32

    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING):
        result = compute_partition(
            nodes=nodes,
            latency_matrix=None,
            total_layers=total_layers,
            model_id="time-budget-model",
            method="ilp",
            time_limit_seconds=1.0,
        )
    elapsed = time.monotonic() - t0

    # Bounded wall time: the CBC solve is capped at 1 s; the remainder is
    # problem construction + LRM fallback, nowhere near an unbounded ILP.
    assert elapsed < 30.0, f"solve+fallback took {elapsed:.1f}s"

    # Still a valid, contiguous partition covering every layer exactly once.
    total = sum(a.layer_count for a in result.assignments)
    assert total == total_layers, f"fallback only covers {total}/{total_layers} layers"
    ordered = sorted(result.assignments, key=lambda a: a.start_layer)
    assert ordered, "no assignments returned"
    assert ordered[0].start_layer == 0
    for prev, curr in zip(ordered, ordered[1:]):
        assert prev.end_layer + 1 == curr.start_layer, (
            f"gap between {prev.node_id} and {curr.node_id}"
        )
    assert ordered[-1].end_layer == total_layers - 1

    # Degradation is observable: shipped path fell back to LRM and the solver
    # did not prove Optimal within the budget.
    assert result.method_used == "lrm", f"expected LRM fallback, got {result.method_used!r}"
    assert result.solver_status != "Optimal", (
        "solver claimed Optimal — no degradation happened"
    )
    assert any(
        "falling back" in r.message for r in caplog.records
    ), "fallback warning was not logged"


@pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
def test_ilp_small_instance_solves_optimal_and_stamps():
    """A small instance with a generous budget still solves optimally through
    the same shipped path and stamps method_used='ilp'."""
    from synapse.topology.partitioner import compute_partition

    result = compute_partition(
        nodes=_make_nodes(3),
        latency_matrix=None,
        total_layers=24,
        model_id="small-model",
        method="ilp",
        time_limit_seconds=30.0,
    )
    assert result.method_used == "ilp", f"expected ILP, got {result.method_used!r}"
    assert result.solver_status == "Optimal", f"unexpected status {result.solver_status!r}"
    total = sum(a.layer_count for a in result.assignments)
    assert total == 24


def test_lrm_path_stamps_defaults():
    """The fast LRM path stamps method_used='lrm' / solver_status='n/a'."""
    from synapse.topology.partitioner import compute_partition

    result = compute_partition(
        nodes=_make_nodes(3),
        latency_matrix=None,
        total_layers=24,
        model_id="lrm-model",
        method="lrm",
        time_limit_seconds=999.0,  # ignored for LRM
    )
    assert result.method_used == "lrm"
    assert result.solver_status == "n/a"
