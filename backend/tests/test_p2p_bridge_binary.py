"""
Behavioral tests: binary tensor round-trip over the production P2P bridge.

Verifies the fix for the decode-blocking bug: forward_tensor() publishes
{"bin": encode_tensor(...)}; the bridge serializes the bytes payload to JSON
(hex) and must restore the exact bytes on the receiving side. Previously the
receiver got a hex STRING, decode_tensor() raised, and the exception was
swallowed by the event router — every inter-node tensor was silently dropped.

Also verifies the latency-probe echo: LatencyProber sends a raw 1KB blob that
does not speak the framed handshake protocol; the bridge must echo it back so
bandwidth measurement works against the production listener.
"""
import asyncio
import random
import struct
import time

import numpy as np
import pytest

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.routing.p2p_socket_bridge import P2PSocketBridge
from synapse.transport.codec import decode_tensor, encode_tensor


class _MeshNode:
    """Minimal node: a real Libp2pNode whose P2PSocketBridge listens on port."""

    def __init__(self, node_id: str, port: int):
        self.id = node_id
        self.port = port
        self.er = EventRouter()
        self.libp2p = Libp2pNode(node_id, port, self.er)

    @property
    def bridge(self):
        return self.libp2p._bridge

    async def start(self):
        await self.er.start()
        await self.libp2p.start()
        await asyncio.sleep(0.3)

    async def stop(self):
        await self.libp2p.stop()

    async def connect_to(self, node: "_MeshNode", address: str) -> bool:
        return await self.libp2p.connect_to_peer(node.id, address)


def test_json_serialize_roundtrips_bytes():
    """_json_serialize/_json_deserialize restore the exact bytes object."""
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = {"bin": encode_tensor(expected)}

    serialized = P2PSocketBridge._json_serialize(payload["bin"])
    assert isinstance(serialized, dict)
    assert serialized.get("__bytes__") is True

    restored = P2PSocketBridge._json_deserialize({"bin": serialized})
    assert isinstance(restored["bin"], bytes), (
        "bin must deserialize back to bytes, not a hex string"
    )
    tensor, _meta = decode_tensor(restored["bin"])
    assert tensor.shape == (2, 3)
    assert np.allclose(tensor, expected)


@pytest.mark.asyncio
async def test_binary_tensor_survives_bridge_roundtrip():
    """A tensor published as {"bin": encode_tensor(...)} on node A reaches
    node B's subscriber as decodable bytes — the production inference path."""
    a = _MeshNode("br-a", 59111)
    b = _MeshNode("br-b", 59112)
    await a.start()
    await b.start()

    received = []

    async def on_tensor(event):
        data = event.data
        assert isinstance(data, dict) and "bin" in data
        assert isinstance(data["bin"], bytes), (
            f"bin must arrive as bytes, got {type(data['bin'])}"
        )
        tensor, metadata = decode_tensor(data["bin"])
        received.append((tensor.copy(), metadata))

    b.er.subscribe("synapse/inference/tensor/br-b", on_tensor)

    await a.connect_to(b, "127.0.0.1:59112")
    await asyncio.sleep(0.5)

    expected = np.arange(12, dtype=np.int32).reshape(3, 4)
    payload = {"bin": encode_tensor(
        expected,
        metadata={
            "request_id": "req-1",
            "shard": {
                "model_id": "m", "start_layer": 0, "end_layer": 3, "n_layers": 4,
            },
        },
    )}
    await a.er.publish("synapse/inference/tensor/br-b", payload, origin="local")

    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.1)

    assert received, "tensor never arrived on node B"
    tensor, metadata = received[0]
    assert tensor.shape == (3, 4)
    assert np.array_equal(tensor, expected)
    assert metadata["request_id"] == "req-1"

    await a.stop()
    await b.stop()


@pytest.mark.asyncio
async def test_bridge_echoes_raw_latency_probe():
    """LatencyProber's raw 1KB probe (no framed handshake) is echoed by the
    production listener so RTT/bandwidth can be measured."""
    a = _MeshNode("br-probe-a", 59121)
    await a.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", a.port)
    probe = struct.pack("!dQ", time.time(), random.getrandbits(64)) + b"\x00" * (1024 - 16)
    writer.write(probe)
    await writer.drain()
    echo = await asyncio.wait_for(reader.readexactly(len(probe)), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    assert echo == probe, "bridge did not echo the raw latency probe"

    await a.stop()


@pytest.mark.asyncio
async def test_latency_prober_measures_bandwidth_against_bridge():
    """The production LatencyProber now succeeds against a running bridge
    (probe echo) and returns a positive bandwidth estimate."""
    from synapse.topology.latency_matrix import LatencyProber

    a = _MeshNode("br-bw-a", 59131)
    await a.start()

    prober = LatencyProber()
    entry = await prober.probe("br-bw-b", f"127.0.0.1:{a.port}", timeout=3.0)

    assert entry is not None, "probe failed against a running bridge"
    assert entry.bandwidth_mbps > 0, f"bandwidth not measured: {entry}"

    await a.stop()
