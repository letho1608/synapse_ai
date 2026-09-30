"""
Behavioral tests: B3 — latency cache refreshes on topology change;
B4 — broadcast_authoritative_partition is wired (master broadcasts before
distributed requests, assignments prefer the authoritative partition);
B5 — fault-tolerance heartbeats are exchanged over the event mesh
independently of membership, and _pending_rebalance is consumed by
check_fault_tolerance() instead of being written and never read.
"""
import asyncio
import time

import numpy as np
import pytest

from synapse.inference.inference_engine import InferenceEngine
from synapse.inference.shard import Shard
from synapse.orchestration.node import Node
from synapse.routing import EventRouter
from synapse.routing.event_router import Event
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.topology.partitioning_strategy import Partition
from synapse.topology.topology import Topology


class _Engine(InferenceEngine):
    async def encode(self, shard, prompt):
        return np.array([])

    async def sample(self, x):
        return np.array([1])

    async def decode(self, shard, tokens):
        return ""

    async def infer_tensor(self, rid, s, t, state=None):
        return np.array([]), {}

    async def load_checkpoint(self, s, p):
        pass


def _dc(vram_mb, fp16_tflops):
    return DeviceCapabilities(
        model="Test-GPU", chip="Test-Chip", memory=vram_mb,
        flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
        cpu_cores=4, disk_gb=100, system_ram_mb=8192,
    )


def _make_node(own_id="node-a"):
    router = EventRouter()
    libp2p = Libp2pNode(own_id, 0, router)
    node = Node(
        _id=own_id,
        event_router=router,
        libp2p_node=libp2p,
        inference_engine=_Engine(),
        shard_downloader=None,
    )
    return node


def _base_shard():
    return Shard(model_id="qwen", start_layer=0, end_layer=31, n_layers=32)


# ── B3: latency cache refreshes on topology change ─────────────────────────


def test_latency_cache_refreshes_when_topology_grows_and_shrinks():
    """B3: the latency matrix must track the live topology — a cache built
    once (hasattr guard) would keep ghosts or miss newly joined nodes."""
    node = _make_node()
    topo = Topology()
    topo.update_node("node-a", _dc(4096, 2.0))
    topo.update_node("node-b", _dc(6144, 3.0))
    node.topology = topo

    m1 = node._build_latency_matrix()
    assert set(m1.nodes) == {"node-a", "node-b"}
    assert m1.get("node-a", "node-b") is not None
    assert m1.get("node-b", "node-a") is not None

    # node-c joins the cluster.
    topo.update_node("node-c", _dc(8192, 4.0))
    m2 = node._build_latency_matrix()
    assert set(m2.nodes) == {"node-a", "node-b", "node-c"}, (
        "matrix not rebuilt when a node joined — stale cache"
    )
    assert m2.get("node-a", "node-c") is not None
    assert m2.get("node-c", "node-a") is not None

    # node-b leaves the cluster.
    topo.nodes.pop("node-b")
    m3 = node._build_latency_matrix()
    assert set(m3.nodes) == {"node-a", "node-c"}, (
        "matrix not rebuilt when a node left — ghost entries remain"
    )
    assert m3.get("node-a", "node-b") is None


# ── B4: broadcast_authoritative_partition is wired ─────────────────────────


@pytest.mark.asyncio
async def test_master_broadcasts_authoritative_partition_before_distributed(monkeypatch):
    """B4: the master's production distributed path calls
    broadcast_authoritative_partition() before serving, so the authoritative
    partition is computed once and broadcast instead of being a dead method."""
    node = _make_node("master")
    node.election_manager.current_master_id = "master"
    topo = Topology()
    topo.update_node("master", _dc(4096, 2.0))
    topo.update_node("peer", _dc(6144, 3.0))
    node.topology = topo

    ext = node.distributed_pipeline
    calls = {}

    async def fake_distributed(base_shard, prompt, request_id,
                               max_generate_tokens=None, num_micro_batches=1,
                               sampling_state=None):
        calls["base_shard"] = base_shard
        return [1, 2]

    monkeypatch.setattr(ext, "process_prompt_distributed", fake_distributed)

    result = await node._process_prompt(_base_shard(), "hello", request_id="req-b4")

    assert calls, "distributed path did not run"
    assert node._authoritative_partitions is not None, (
        "master never broadcast the authoritative partition"
    )
    assert node._authoritative_base_shard is not None
    assert result is not None and len(result) == 2


@pytest.mark.asyncio
async def test_distributed_assignments_prefer_fresh_authoritative_partition(monkeypatch):
    """B4: once the master has broadcast a partition matching the base shard,
    the distributed pipeline uses it instead of recomputing LACP locally."""
    node = _make_node("master")
    topo = Topology()
    topo.update_node("master", _dc(4096, 2.0))
    topo.update_node("peer", _dc(6144, 3.0))
    node.topology = topo

    ext = node.distributed_pipeline
    base_shard = _base_shard()
    node._authoritative_partitions = [
        Partition(node_id="master", start=0.0, end=0.5),
        Partition(node_id="peer", start=0.5, end=1.0),
    ]
    node._authoritative_base_shard = base_shard

    compute_calls = []
    orig = node._compute_lacp_partition

    def spy(*a, **k):
        compute_calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(node, "_compute_lacp_partition", spy)

    assignments = ext._get_partition_assignments(base_shard)
    assert not compute_calls, (
        "local LACP recompute ran despite a fresh authoritative partition"
    )
    assert [nid for nid, _ in assignments] == ["master", "peer"]

    # A mismatched base shard must fall back to a local recompute.
    node._authoritative_base_shard = Shard(
        model_id="other-model", start_layer=0, end_layer=15, n_layers=16
    )
    assignments2 = ext._get_partition_assignments(base_shard)
    assert compute_calls, "stale/mismatched partition must trigger local recompute"
    assert len(assignments2) == 2


# ── B5: heartbeat exchange + _pending_rebalance consumption ─────────────────


@pytest.mark.asyncio
async def test_ft_heartbeat_records_peer_outside_membership():
    """B5: a fault-tolerance heartbeat over the event mesh keeps liveness
    tracked even for a peer that is neither in topology nor in self.peers."""
    node = _make_node("node-a")
    node._lazy_init_fault_tolerance()

    event = Event(
        topic="synapse/cluster/heartbeat",
        data={"type": "heartbeat", "node_id": "node-ghost"},
        origin="node-ghost",
        timestamp=time.time(),
    )
    await node._on_ft_heartbeat(event)

    assert node.fault_tolerance.is_node_alive("node-ghost"), (
        "peer heartbeat not recorded (membership-only heartbeat would miss it)"
    )


@pytest.mark.asyncio
async def test_ft_heartbeat_loop_publishes_on_router():
    """B5: the shipped heartbeat loop publishes this node's heartbeat on the
    event mesh on the FT cadence."""
    node = _make_node("node-a")
    node._lazy_init_fault_tolerance()
    node.fault_tolerance.HEARTBEAT_INTERVAL = 0.05

    got = []
    node.event_router.subscribe("synapse/cluster/heartbeat", lambda ev: got.append(ev))
    task = asyncio.create_task(node._ft_heartbeat_loop())
    try:
        for _ in range(50):
            if got:
                break
            await asyncio.sleep(0.05)
        assert got, "heartbeat loop never published"
        assert got[0].data.get("node_id") == "node-a"
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_check_fault_tolerance_rebalances_pending_node():
    """B5: a node flagged by the monitor loop (_pending_rebalance) is
    rebalanced by check_fault_tolerance() instead of waiting for the 3-miss
    blacklist threshold; the flag is consumed."""
    node = _make_node("master")
    node.distributed_pipeline  # extension initialized (as node.start() does)
    node._lazy_init_fault_tolerance()
    ft = node.fault_tolerance

    topo = Topology()
    topo.update_node("master", _dc(4096, 2.0))
    topo.update_node("node-a", _dc(6144, 3.0))
    topo.update_node("node-b", _dc(8192, 4.0))
    node.topology = topo

    ft.register_node("node-a")
    ft.register_node("node-b")
    ft._health["node-b"].is_alive = False
    ft._health["node-b"].is_blacklisted = False  # NOT past the blacklist threshold
    ft._pending_rebalance = "node-b"             # monitor loop flagged it

    await node.check_fault_tolerance()

    assert ft._pending_rebalance is None, "pending flag not consumed by rebalance"
    result = node._last_rebalance_result
    assert result is not None and result.success, (
        "pending node was not rebalanced — _pending_rebalance is write-only"
    )
    assert result.removed_node == "node-b"
