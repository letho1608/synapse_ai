import asyncio

from synapse.orchestration.node import Node
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.topology.topology import Topology


class _FakePeer:
    def __init__(self, peer_id: str, caps: DeviceCapabilities, topo: Topology):
        self._peer_id = peer_id
        self._caps = caps
        self._topo = topo

    def id(self) -> str:
        return self._peer_id

    def addr(self) -> str:
        return "127.0.0.1:50051"

    def description(self) -> str:
        return "manual"

    def device_capabilities(self) -> DeviceCapabilities:
        return self._caps

    async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
        return self._topo


def _caps() -> DeviceCapabilities:
    return DeviceCapabilities(
        model="Test GPU",
        chip="Intel Iris Xe",
        memory=2048,
        flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
        cpu_cores=8,
        disk_gb=256,
        system_ram_mb=16384,
        gpu_backend="CPU (x86)",
    )


def _make_node(node_id: str = "node-self") -> Node:
    node = Node(
        _id=node_id,
        server=None,
        inference_engine=None,
        discovery=None,
        shard_downloader=None,
        partitioning_strategy=None,
    )
    node.device_capabilities = _caps()
    return node


def test_collect_topology_keeps_homogeneous_peers() -> None:
    async def _run() -> None:
        node = _make_node()
        caps = _caps()

        peer_topology_a = Topology()
        peer_topology_a.update_node("node-peer-a", caps)

        peer_topology_b = Topology()
        peer_topology_b.update_node("node-peer-b", caps)

        node.peers = [
            _FakePeer("node-peer-a", caps, peer_topology_a),
            _FakePeer("node-peer-b", caps, peer_topology_b),
        ]

        topology = await node.collect_topology(set())
        assert set(topology.nodes.keys()) == {"node-self", "node-peer-a", "node-peer-b"}

    asyncio.run(_run())


def test_collect_topology_prunes_unreachable_remote_nodes() -> None:
    async def _run() -> None:
        node = _make_node()
        caps = _caps()

        remote_topology = Topology()
        remote_topology.update_node("node-peer", caps)
        remote_topology.update_node("stale-node", caps)
        remote_topology.add_edge("node-peer", "stale-node", "TS")

        node.peers = [_FakePeer("node-peer", caps, remote_topology)]

        topology = await node.collect_topology(set())

        assert "node-self" in topology.nodes
        assert "node-peer" in topology.nodes
        assert "stale-node" not in topology.nodes

        for src_id, connections in topology.peer_graph.items():
            assert src_id in topology.nodes
            for conn in connections:
                assert conn.to_id in topology.nodes

    asyncio.run(_run())
