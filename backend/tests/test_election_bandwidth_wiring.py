"""
Behavioral tests: multi-resource election wiring + bandwidth propagation.

Verifies the fixes for:
- E1: the production Node wires the LACP multi-resource scorer into the
  ElectionManager, so the elected master's weight derives from the composite
  resource score (not the default 1.0) and a stronger node wins.
- B-bandwidth: measured network bandwidth propagates from the latency matrix
  into NodeResources.bandwidth_mbps, so partitioner/election resource scores
  reflect real link capacity instead of the 1000 Mbps placeholder.
"""
import numpy as np
import pytest

import synapse.orchestration.node as node_module
from synapse.inference.inference_engine import InferenceEngine
from synapse.inference.shard import Shard
from synapse.orchestration.node import Node
from synapse.routing import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.topology.latency_matrix import LatencyEntry, LatencyMatrix
from synapse.topology.resource_weight import NodeResources, apply_measured_bandwidth
from synapse.topology.topology import Topology


def _dc(vram_mb: int, fp16_tflops: float) -> DeviceCapabilities:
    return DeviceCapabilities(
        model="Test-GPU",
        chip="Test-Chip",
        memory=vram_mb,
        flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
        cpu_cores=4,
        disk_gb=100,
        system_ram_mb=8192,
    )


class _DummyEngine(InferenceEngine):
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


def _node_with_topology(own_id="node-a", n_nodes=2):
    """node-a is weak (4 GB / 2 TFLOPS), node-b is strong (6 GB / 3 TFLOPS)."""
    router = EventRouter()
    libp2p = Libp2pNode(own_id, 59999, router)
    node = Node(
        _id=own_id,
        event_router=router,
        libp2p_node=libp2p,
        inference_engine=_DummyEngine(),
        shard_downloader=None,
    )
    node.tflops_proportional = False
    topo = Topology()
    topo.update_node("node-a", _dc(4096, 2.0))
    topo.update_node("node-b", _dc(6144, 3.0))
    node.topology = topo
    return node


class TestElectionWiring:
    def test_election_manager_has_resource_scorer(self):
        node = _node_with_topology()
        assert node.election_manager._resource_scorer is not None

    def test_scorer_differentiates_within_cluster(self):
        node = _node_with_topology()
        score_weak = node._election_resource_scorer("node-a")
        score_strong = node._election_resource_scorer("node-b")
        assert score_strong > score_weak, (
            f"stronger node should score higher: {score_weak} vs {score_strong}"
        )

    def test_election_weight_derives_from_composite(self):
        node = _node_with_topology()
        node.device_capabilities = _dc(4096, 2.0)
        node.election_manager.set_base_compute_weight(4.0, 1.0)
        weight = node.election_manager.compute_weight
        # Composite path (score * 1000) dominates the simple vram+flops path
        # (4.01) and is far from the default 1.0.
        assert weight != 1.0
        assert weight > 4.01

    def test_stronger_node_gets_higher_election_weight(self):
        node_a = _node_with_topology(own_id="node-a")
        node_a.device_capabilities = _dc(4096, 2.0)
        node_a.election_manager.set_base_compute_weight(4.0, 1.0)
        w_a = node_a.election_manager.compute_weight

        node_b = _node_with_topology(own_id="node-b")
        node_b.device_capabilities = _dc(6144, 3.0)
        node_b.election_manager.set_base_compute_weight(6.0, 1.5)
        w_b = node_b.election_manager.compute_weight

        assert w_b > w_a, f"stronger node should win election: {w_a} vs {w_b}"


class TestBandwidthPropagation:
    def test_apply_measured_bandwidth_helper(self):
        nodes = {
            "node-a": NodeResources(gpu_vram_mb=4096, gpu_tflops_fp16=2.0),
            "node-b": NodeResources(gpu_vram_mb=6144, gpu_tflops_fp16=3.0),
        }
        matrix = LatencyMatrix()
        matrix.set(LatencyEntry(from_node="node-a", to_node="node-b",
                                latency_ms=1.0, bandwidth_mbps=500.0))
        apply_measured_bandwidth(nodes, matrix, self_node_id="node-a")
        assert nodes["node-b"].bandwidth_mbps == 500.0
        # Node without a bandwidth entry keeps its placeholder
        assert nodes["node-a"].bandwidth_mbps == 0.0

    def test_apply_measured_bandwidth_no_matrix(self):
        nodes = {"node-a": NodeResources(gpu_vram_mb=4096)}
        apply_measured_bandwidth(nodes, None, self_node_id="node-a")
        assert nodes["node-a"].bandwidth_mbps == 0.0

    def test_production_partition_path_propagates_bandwidth(self):
        node = _node_with_topology()
        # Simulate a completed bandwidth probe (node-a measured node-b).
        node._measured_bandwidth = {"node-b": 500.0}

        captured = []
        original = node_module.compute_partition

        def recording(nodes, latency_matrix=None, total_layers=0, model_id="", **kw):
            captured.append(nodes)
            return original(nodes, latency_matrix, total_layers, model_id, **kw)

        node_module.compute_partition = recording
        try:
            node._compute_lacp_partition(
                Shard(model_id="m", start_layer=0, end_layer=31, n_layers=32)
            )
        finally:
            node_module.compute_partition = original

        assert captured, "compute_partition was not reached"
        assert captured[0]["node-b"].bandwidth_mbps == 500.0, (
            "measured bandwidth must flow into partitioner resource scores"
        )

    def test_election_scorer_uses_measured_bandwidth(self):
        node = _node_with_topology()
        node._measured_bandwidth = {"node-b": 500.0}
        # bandwidth dimension (weight 0.10) now differentiates node-b upward
        # within the cluster scoring used for election weight.
        score_weak = node._election_resource_scorer("node-a")
        score_strong = node._election_resource_scorer("node-b")
        assert score_strong > score_weak
