"""
Behavioral test: multi-node LACP partitioning through the production node path.

Verifies the fix for B-latency-matrix: the latency data produced by the node
runtime (_build_latency_matrix) flows type-safely into compute_partition. On a
cluster with 2+ nodes the production path (Node._compute_lacp_partition with
tflops_proportional=False) must partition without the previous
AttributeError ("'dict' object has no attribute 'nodes'") and produce valid,
contiguous assignments covering all model layers.
"""
import numpy as np
import pytest

from synapse.inference.inference_engine import InferenceEngine
from synapse.inference.shard import Shard
from synapse.orchestration.node import Node
from synapse.routing import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.topology.latency_matrix import LatencyMatrix
from synapse.topology.topology import Topology


def _dc(vram_mb: int, fp16_tflops: float, cores: int = 4) -> DeviceCapabilities:
    return DeviceCapabilities(
        model="Test-GPU",
        chip="Test-Chip",
        memory=vram_mb,
        flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
        cpu_cores=cores,
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


class TestMultiNodeLACPFlow:
    @staticmethod
    def _node_with_topology(node_id="node-a", n_nodes=2):
        router = EventRouter()
        libp2p = Libp2pNode(node_id, 59999, router)
        node = Node(
            _id=node_id,
            event_router=router,
            libp2p_node=libp2p,
            inference_engine=_DummyEngine(),
            shard_downloader=None,
        )
        # Force the LACP multi-resource path (the default TFLOPS-proportional
        # path bypasses the latency matrix entirely).
        node.tflops_proportional = False
        topo = Topology()
        for i in range(n_nodes):
            topo.update_node(f"node-{chr(ord('a') + i)}", _dc(4096 + i * 2048, 2.0 + i))
        node.topology = topo
        return node

    def test_build_latency_matrix_returns_latency_matrix(self):
        node = self._node_with_topology(n_nodes=2)
        matrix = node._build_latency_matrix()
        assert matrix is not None
        assert isinstance(matrix, LatencyMatrix)
        assert "node-a" in matrix.nodes and "node-b" in matrix.nodes
        # Diagonal + cross entries present
        assert matrix.get("node-a", "node-b") is not None
        assert matrix.get("node-b", "node-a") is not None

    def test_build_latency_matrix_injects_adaptive_latency(self):
        node = self._node_with_topology(n_nodes=2)
        node._last_forward_latencies = {"node-b": 0.002}  # 2 ms
        matrix = node._build_latency_matrix()
        entry = matrix.get("node-a", "node-b")
        assert entry is not None
        assert entry.latency_ms == pytest.approx(2.0, abs=1e-6)

    def test_production_lacp_partition_three_nodes_no_crash(self):
        node = self._node_with_topology(n_nodes=3)
        base_shard = Shard(model_id="lacp-model", start_layer=0, end_layer=31, n_layers=32)
        partitions = node._compute_lacp_partition(base_shard)

        # Valid contiguous assignments covering all layers
        assert len(partitions) == 3
        assert partitions[0].start == pytest.approx(0.0)
        assert partitions[-1].end == pytest.approx(1.0)
        for prev, curr in zip(partitions, partitions[1:]):
            assert curr.start == pytest.approx(prev.end)
        # Each partition maps to a distinct node in the topology
        assigned = {p.node_id for p in partitions}
        assert assigned == {"node-a", "node-b", "node-c"}

    def test_production_lacp_partition_two_nodes_no_crash(self):
        node = self._node_with_topology(n_nodes=2)
        base_shard = Shard(model_id="lacp-model", start_layer=0, end_layer=31, n_layers=32)
        partitions = node._compute_lacp_partition(base_shard)
        assert len(partitions) == 2
        assert partitions[0].start == pytest.approx(0.0)
        assert partitions[-1].end == pytest.approx(1.0)
        assert partitions[0].end == pytest.approx(partitions[1].start)


class _FakePeer:
    def __init__(self, pid, addr):
        self._pid = pid
        self._addr = addr

    def id(self):
        return self._pid

    def addr(self):
        return self._addr


class TestSHMWiring:
    """Production wiring of the zero-copy shared-memory transport.

    Node.start() brings up a SharedMemoryTransport alongside the TCP transport,
    and update_peers/_register_transport_peers register same-machine peers on
    it so co-located nodes can bypass TCP for tensor traffic.
    """

    async def test_node_wiring_registers_same_machine_peers_on_shm(self):
        from synapse.transport.shm_transport import SharedMemoryTransport
        from synapse.transport.tcp_transport import TensorTCPTransport

        router = EventRouter()
        node = Node(
            _id="node-a",
            event_router=router,
            libp2p_node=Libp2pNode("node-a", 59999, router),
            inference_engine=_DummyEngine(),
            shard_downloader=None,
        )

        # Mirror the transport-start block of Node.start().
        node.tcp_transport = TensorTCPTransport(node_id=node.id)
        await node.tcp_transport.start()
        node.shm_transport = SharedMemoryTransport(node_id=node.id)
        await node.shm_transport.start()

        # One same-machine (loopback) peer, one genuinely remote peer.
        node.peers = [
            _FakePeer("node-b", "127.0.0.1:60123"),
            _FakePeer("node-remote", "10.9.9.9:60124"),
        ]
        # Peers advertise their tensor-transport port in the bridge handshake;
        # an old peer without it stays bridge-only.
        node.libp2p_node._bridge.record_peer_handshake(
            "node-b", {"node_id": "node-b", "tcp_port": 61001})
        node._register_transport_peers()

        # Same-machine peer is reachable via shared memory; remote peer is not.
        assert "127.0.0.1:60123" in node.shm_transport._peer_ids
        assert not "10.9.9.9:60124" in node.shm_transport._peer_ids
        # TCP registers the ADVERTISED tensor port, never the bridge port
        # (the bridge port was registered before, so chunked TCP could never
        # connect between real processes).
        assert "127.0.0.1:61001" in node.tcp_transport._peer_map
        assert "127.0.0.1:60123" not in node.tcp_transport._peer_map
        assert not any(a.startswith("10.9.9.9") for a in node.tcp_transport._peer_map)

        # End-to-end: send through the node's SHM transport to a registered
        # same-machine peer transport; the tensor arrives intact.
        peer_t = SharedMemoryTransport(node_id="node-b")
        await peer_t.start()
        received = []
        peer_t.set_tensor_handler(lambda m: received.append(m))
        node.shm_transport.register_peer("127.0.0.1:60123", "node-b")

        data = np.arange(16, dtype=np.float32)
        ok = await node.shm_transport.send_tensor("w-1", 2, data, "127.0.0.1:60123")
        assert ok is True
        assert len(received) == 1
        assert received[0].request_id == "w-1"
        assert received[0].target_stage == 2
        np.testing.assert_array_equal(received[0].tensor, data)
        assert node.shm_transport.shm_sends == 1
        assert peer_t.shm_recvs == 1

        await node.shm_transport.stop()
        await node.tcp_transport.stop()
        await peer_t.stop()
