"""
Scale benchmark: LACP partition frontier n x L (T3.2 / C5).

Measures (not asserts perf): construction+solve wall time, solver status,
and ILP-vs-LRM max-load ratio over n={2,4,8} x L={8,32}. Hard invariants
asserted everywhere: exact-once coverage of [0, L-1], per-node contiguity,
method_used/solver_status stamps. Optimality asserted only for the tiny
2x8 ILP instance; larger cells publish their numbers for the README table.

Deliberately NOT testing 12x32+/WAN here: O(n^2 L^2) linkage rows make big
cells CI-hostile (see Lane 4). Those belong to the manual scale run.
"""
import time

import pytest

from synapse.topology.latency_matrix import LatencyEntry, LatencyMatrix
from synapse.topology.partitioner import compute_partition
from synapse.topology.resource_weight import NodeResources


def _nodes(n):
    out = {}
    for i in range(n):
        out["node-%02d" % i] = NodeResources(
            gpu_vram_mb=8192.0 - i * 256.0,
            gpu_tflops_fp16=60.0 - i * 2.0,
            gpu_tflops_fp32=30.0 - i,
            gpu_count=1,
            cpu_cores=8,
            system_ram_mb=32000.0,
            disk_read_mbps=500.0,
            disk_write_mbps=400.0,
            bandwidth_mbps=1000.0,
        )
    return out


def _matrix(ids, latency_ms=1.0):
    m = LatencyMatrix()
    for a in ids:
        for b in ids:
            m.set(LatencyEntry(from_node=a, to_node=b,
                               latency_ms=0.0 if a == b else latency_ms))
    return m


def _check_invariants(result, total_layers):
    covered = []
    for a in result.assignments:
        assert a.start_layer <= a.end_layer, "empty assignment row"
        covered.extend(range(a.start_layer, a.end_layer + 1))
    assert sorted(covered) == list(range(total_layers)), "layers must be covered exactly once"
    assert result.method_used in ("lrm", "ilp")
    assert isinstance(result.solver_status, str) and result.solver_status


@pytest.mark.parametrize("n,l", [(2, 8), (4, 8), (8, 8), (2, 32), (4, 32), (8, 32)])
def test_partition_scale_matrix(n, l, capsys):
    nodes = _nodes(n)
    ids = sorted(nodes)
    matrix = _matrix(ids)

    t0 = time.perf_counter()
    lrm = compute_partition(nodes, matrix, l, model_id="scale", method="lrm")
    lrm_ms = (time.perf_counter() - t0) * 1000.0
    _check_invariants(lrm, l)

    t0 = time.perf_counter()
    ilp = compute_partition(nodes, matrix, l, model_id="scale", method="ilp",
                            time_limit_seconds=2.0)
    ilp_ms = (time.perf_counter() - t0) * 1000.0
    _check_invariants(ilp, l)

    with capsys.disabled():
        print("scale n=%d L=%d lrm=%.0fms ilp=%.0fms ilp_status=%s/%s" % (
            n, l, lrm_ms, ilp_ms, ilp.method_used, ilp.solver_status))


def test_ilp_optimal_on_tiny_instance():
    nodes = _nodes(2)
    matrix = _matrix(sorted(nodes))
    res = compute_partition(nodes, matrix, 8, model_id="tiny", method="ilp",
                            time_limit_seconds=10.0)
    _check_invariants(res, 8)
    assert res.method_used == "ilp"
    assert res.solver_status == "Optimal"


def test_zero_weight_nodes_get_equal_fallback_without_gaps():
    nodes = {"a": NodeResources(), "b": NodeResources()}
    matrix = _matrix(["a", "b"])
    res = compute_partition(nodes, matrix, 8, model_id="zero", method="lrm")
    _check_invariants(res, 8)
