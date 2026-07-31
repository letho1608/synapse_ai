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

Large-frame handling: frames up to the configured cap (default 256 MiB) are
delivered intact even when they exceed the old 10 MiB limit; a frame beyond a
configured cap gets an explicit error packet back to the sender instead of a
silent drop.
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

    def __init__(self, node_id: str, port: int, max_frame_size: int = None):
        self.id = node_id
        self.port = port
        self.er = EventRouter()
        self.libp2p = Libp2pNode(node_id, port, self.er, max_frame_size=max_frame_size)

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


@pytest.mark.asyncio
async def test_large_frame_over_old_10mib_cap_is_delivered():
    """A frame larger than the old hardcoded 10 MiB limit (a 12 MB float32
    tensor expands to ~24 MB of JSON hex) is now delivered intact end-to-end
    with the default 256 MiB cap — the silent-drop regression is gone."""
    a = _MeshNode("br-lg-a", 59141)
    b = _MeshNode("br-lg-b", 59142)
    await a.start()
    await b.start()

    received = []

    async def on_tensor(event):
        data = event.data
        tensor, metadata = decode_tensor(data["bin"])
        received.append((tensor.copy(), metadata))

    b.er.subscribe("synapse/inference/tensor/br-lg-b", on_tensor)

    await a.connect_to(b, "127.0.0.1:59142")
    await asyncio.sleep(0.5)

    # 3M float32 = 12 MB raw; JSON hex encoding doubles the frame to ~24 MB,
    # comfortably past the old 10 MiB cap but inside the new 256 MiB default.
    expected = np.arange(3_000_000, dtype=np.float32).reshape(3000, 1000)
    payload = {"bin": encode_tensor(
        expected,
        metadata={"request_id": "req-big", "shard": {"model_id": "m",
                                                     "start_layer": 0, "end_layer": 8, "n_layers": 8}},
    )}
    await a.er.publish("synapse/inference/tensor/br-lg-b", payload, origin="local")

    for _ in range(100):
        if received:
            break
        await asyncio.sleep(0.1)

    assert received, "large tensor never arrived on node B (silently dropped?)"
    tensor, metadata = received[0]
    assert tensor.shape == (3000, 1000)
    assert np.array_equal(tensor, expected)
    assert metadata["request_id"] == "req-big"

    await a.stop()
    await b.stop()


@pytest.mark.asyncio
async def test_oversized_frame_gets_explicit_error_back_to_sender():
    """With a small configured cap, an oversized frame is NOT silently
    dropped: the receiver answers with an explicit error packet and the
    sender records it (subscriber never sees the payload)."""
    small_cap = 4096
    a = _MeshNode("br-err-a", 59151, max_frame_size=small_cap)
    b = _MeshNode("br-err-b", 59152, max_frame_size=small_cap)
    await a.start()
    await b.start()

    received = []

    async def on_tensor(event):
        received.append(event.data)

    b.er.subscribe("synapse/inference/tensor/br-err-b", on_tensor)

    await a.connect_to(b, "127.0.0.1:59152")
    await asyncio.sleep(0.5)

    # 2K float32 = 8 KB raw -> ~16 KB hex frame, far beyond the 4096 cap.
    big = np.arange(2048, dtype=np.float32)
    await a.er.publish(
        "synapse/inference/tensor/br-err-b",
        {"bin": encode_tensor(big, metadata={"request_id": "req-big2"})},
        origin="local",
    )

    err = None
    for _ in range(50):
        err = a.bridge._last_errors.get("br-err-b")
        if err:
            break
        await asyncio.sleep(0.1)

    assert err is not None, "sender never received an explicit error packet"
    assert err.get("reason") == "frame_too_large", f"unexpected error: {err}"
    assert err.get("max_frame_size") == small_cap
    assert err.get("size", 0) > small_cap
    assert not received, "oversized payload must not reach the subscriber"

    await a.stop()
    await b.stop()
