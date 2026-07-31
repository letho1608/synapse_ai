"""
Behavioral test: ILP reachable from the production partitioner path.

Verifies the fix for H1/ILP: the production node path reads a config knob
(Node.partition_method) and passes it through to compute_partition(method=...).
With method="ilp" the partitioner returns PuLP-solved assignments (contiguous,
full coverage, weighted by resource score); on solver failure it falls back to
a valid LRM partition so the cluster never receives a broken assignment.
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


class TestILPProductionFlow:
    @staticmethod
    def _node_with_topology(n_nodes=2):
        router = EventRouter()
        libp2p = Libp2pNode("node-a", 59999, router)
        node = Node(
            _id="node-a",
            event_router=router,
            libp2p_node=libp2p,
            inference_engine=_DummyEngine(),
            shard_downloader=None,
        )
        # Force the LACP multi-resource path (bypasses TFLOPS-proportional).
        node.tflops_proportional = False
        topo = Topology()
        for i in range(n_nodes):
            topo.update_node(
                f"node-{chr(ord('a') + i)}",
                _dc(4096 + i * 2048, 2.0 + i),
            )
        node.topology = topo
        return node

    @staticmethod
    def _base_shard():
        return Shard(model_id="ilp-model", start_layer=0, end_layer=31, n_layers=32)

    def test_default_partition_method_is_lrm(self):
        node = self._node_with_topology()
        assert node.partition_method == "lrm"

    def test_constructor_exposes_partition_config(self):
        """The production Node accepts partition_method / tflops_proportional
        via the constructor (what the --partition-method / --no-tflops-proportional
        CLI flags pass), so the ILP solver and the multi-resource LACP path are
        selectable by configuration — not only by test-only attribute flips."""
        router = EventRouter()
        libp2p = Libp2pNode("node-cfg", 59999, router)
        node = Node(
            _id="node-cfg",
            event_router=router,
            libp2p_node=libp2p,
            inference_engine=_DummyEngine(),
            shard_downloader=None,
            partition_method="ilp",
            tflops_proportional=False,
        )
        assert node.partition_method == "ilp"
        assert node.tflops_proportional is False
        # Defaults stay backward compatible for existing deployments.
        node2 = Node(
            _id="node-cfg2",
            event_router=router,
            libp2p_node=libp2p,
            inference_engine=_DummyEngine(),
            shard_downloader=None,
        )
        assert node2.partition_method == "lrm"
        assert node2.tflops_proportional is True

    def test_production_path_passes_method_to_compute_partition(self):
        node = self._node_with_topology(n_nodes=2)
        node.partition_method = "ilp"

        calls = []
        original = node_module.compute_partition

        def recording(*args, **kwargs):
            calls.append(kwargs.get("method"))
            return original(*args, **kwargs)

        node_module.compute_partition = recording
        try:
            partitions = node._compute_lacp_partition(self._base_shard())
        finally:
            node_module.compute_partition = original

        assert calls, "compute_partition was not reached"
        assert calls[-1] == "ilp", f"expected method='ilp', got {calls[-1]!r}"
        # ILP-solved assignments: both nodes used, contiguous, full coverage.
        assert len(partitions) == 2
        assert partitions[0].start == pytest.approx(0.0)
        assert partitions[-1].end == pytest.approx(1.0)
        assert partitions[0].end == pytest.approx(partitions[1].start)

    def test_production_ilp_assigns_more_layers_to_stronger_node(self):
        node = self._node_with_topology(n_nodes=2)
        node.partition_method = "ilp"
        # node-a is weaker (2 TFLOPS), node-b is stronger (3 TFLOPS)
        partitions = node._compute_lacp_partition(self._base_shard())
        layers_by_node = {p.node_id: (p.end - p.start) for p in partitions}
        assert layers_by_node["node-b"] > layers_by_node["node-a"], (
            f"ILP should give more layers to the stronger node: {layers_by_node}"
        )

    def test_direct_ilp_call_three_nodes_valid(self):
        from synapse.topology.partitioner import compute_partition
        from synapse.topology.resource_weight import NodeResources

        nodes = {
            f"node{i}": NodeResources(
                gpu_vram_mb=4096 + i * 1024,
                gpu_tflops_fp16=10.0 + i * 2.0,
                gpu_tflops_fp32=5.0 + i,
                cpu_cores=4 + i,
                system_ram_mb=8192 + i * 2048,
            )
            for i in range(3)
        }
        result = compute_partition(
            nodes=nodes,
            latency_matrix=None,
            total_layers=30,
            model_id="test-model",
            method="ilp",
        )
        covered = set()
        for a in result.assignments:
            covered.update(range(a.start_layer, a.end_layer + 1))
        assert covered == set(range(30))
        # Contiguity across nodes
        ordered = sorted(result.assignments, key=lambda a: a.start_layer)
        for prev, curr in zip(ordered, ordered[1:]):
            assert prev.end_layer + 1 == curr.start_layer
