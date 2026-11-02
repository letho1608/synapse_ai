"""
The master dies: survivors stop planning onto it at once.

Found 2026-11-02: every node kept the dead master's broadcast partition
(still naming the master as a stage) until a new master re-broadcast, which
waits for the election grace period. Each request in that window was sent
to the dead node and paid the mid-request failover timeout. Now a broadcast
partition naming a dead node is ignored and placement leaves dead nodes
out, while the rebalance still sees the pre-failure plan.
"""

from synapse.inference.shard import Shard
from synapse.topology.partitioning_strategy import Partition
import test_measured_placement as _placement  # not `from ... import`: pytest would collect it again


class _FT:
    def __init__(self):
        self.dead = set()

    def is_node_alive(self, nid, unknown=True):
        return nid not in self.dead


def _cluster(monkeypatch):
    node = _placement.TestNodeWiring._node(monkeypatch, ids=("B", "M", "C"))  # we are B, M is master
    node.fault_tolerance = _FT()
    node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}
    node._peer_capabilities = {n: {"layer_profiles": {"m": {"ms_per_layer": 1.0, "max_layers": 100}}}
                               for n in ("M", "C")}
    node._measured_rtt = {"M": 0.0, "C": 0.0}
    base = Shard("m", 0, 29, 30)
    node._authoritative_partitions = [Partition("M", 0.0, 1 / 3), Partition("B", 1 / 3, 2 / 3),
                                      Partition("C", 2 / 3, 1.0)]
    node._authoritative_base_shard = base
    return node, base


def test_survivors_replan_without_the_dead_master(monkeypatch):
    node, base = _cluster(monkeypatch)
    assert node.get_current_shard(base).start_layer == 10  # master's plan while it lives
    node.fault_tolerance.dead.add("M")
    assert node._get_authoritative_partitions(base) is None
    parts = node._compute_lacp_partition(base)
    assert "M" not in {p.node_id for p in parts}
    sh = node.get_current_shard(base)
    covered = sorted((round(p.start * 30), round(p.end * 30)) for p in parts)
    assert covered[0][0] == 0 and covered[-1][1] == 30, "every layer still placed"
    assert sh.model_id == "m" and 0 <= sh.start_layer <= sh.end_layer <= 29
    plan = node.distributed_pipeline._get_partition_assignments(base)
    assert [nid for nid, _ in plan] and "M" not in [nid for nid, _ in plan]


def test_rebalance_still_sees_the_dead_nodes_range(monkeypatch):
    node, base = _cluster(monkeypatch)
    node._authoritative_partitions = None
    node.fault_tolerance.dead.add("M")
    before = {p.node_id for p in node._compute_lacp_partition(base, include_dead=True)}
    assert "M" in before
