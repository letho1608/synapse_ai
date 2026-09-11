"""
Behavioral tests: versioned handshake + mixed-version skew (T0.3).

Contract under test (backend/synapse/routing/p2p_socket_bridge.py):
- _send_handshake advertises synapse_version/bridge_version/caps_hash/features;
  old peers ignore unknown keys (.get), new peers record them.
- record_peer_handshake stores versions; is_compatible() is fail-closed ONLY
  on bridge major mismatch; unknown + pre-versioning peers stay compatible.
- negotiated_features() returns the intersection.
- Unknown packet types never kill the read loop (additive-only policy).

Ports 59211/59212: unused by any other test module in this repo.
"""
import asyncio
import json
import struct

import pytest

from synapse.routing.event_router import EventRouter
from synapse.routing.p2p_socket_bridge import (
    P2PSocketBridge,
    BRIDGE_VERSION,
    DEFAULT_FEATURES,
    PACKET_TYPE_HANDSHAKE,
)


async def _make_bridge(node_id, port):
    er = EventRouter()
    try:
        await er.start()
    except Exception:
        pass
    bridge = P2PSocketBridge(node_id, port, er)
    task = asyncio.create_task(bridge.start())
    await asyncio.sleep(0.3)
    return er, bridge, task


async def _stop_bridge(bridge, task):
    try:
        await bridge.stop()
    except Exception:
        pass
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _raw_handshake(port, payload):
    """Speak a hand-rolled handshake to a live bridge; return its reply dict."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        data = json.dumps(payload).encode("utf-8")
        writer.write(struct.pack("!IB", len(data), PACKET_TYPE_HANDSHAKE) + data)
        await writer.drain()
        header = await asyncio.wait_for(reader.readexactly(5), timeout=5.0)
        length = struct.unpack("!I", header[:4])[0]
        body = await asyncio.wait_for(reader.readexactly(length), timeout=5.0)
        return json.loads(body.decode("utf-8"))
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_handshake_records_versions_both_directions():
    """New<->new: versions recorded, compatible, features intersected."""
    _, a, ta = await _make_bridge("hs-a", 59211)
    _, b, tb = await _make_bridge("hs-b", 59212)
    try:
        ok = await a.connect_to_peer("hs-b", "127.0.0.1:59212")
        assert ok, "connect_to_peer failed"
        await asyncio.sleep(0.5)

        ha = a.get_peer_handshake("hs-b")
        hb = b.get_peer_handshake("hs-a")
        assert ha and hb, "handshakes not recorded on both sides"
        assert ha["bridge_version"] == BRIDGE_VERSION
        assert hb["bridge_version"] == BRIDGE_VERSION
        assert ha["synapse_version"] == hb["synapse_version"]
        assert a.is_compatible("hs-b") is True
        assert b.is_compatible("hs-a") is True
        assert a.negotiated_features("hs-b") == sorted(DEFAULT_FEATURES)
        # Unknown peers fail open (no info), never fail closed by accident.
        assert a.is_compatible("no-such-peer") is True
        assert a.negotiated_features("no-such-peer") == []
    finally:
        await _stop_bridge(a, ta)
        await _stop_bridge(b, tb)


@pytest.mark.asyncio
async def test_old_style_handshake_accepted_as_legacy():
    """Pre-versioning peer (node_id/port only) connects and stays compatible."""
    _, srv, ts = await _make_bridge("hs-srv", 59213)
    try:
        reply = await _raw_handshake(59213, {
            "type": "handshake", "node_id": "hs-old", "port": 59999,
        })
        assert reply.get("type") == "handshake"
        assert reply.get("bridge_version") == BRIDGE_VERSION
        assert sorted(reply.get("features", [])) == sorted(DEFAULT_FEATURES)
        await asyncio.sleep(0.3)
        rec = srv.get_peer_handshake("hs-old")
        assert rec is not None, "legacy handshake not recorded"
        assert rec["bridge_version"] is None
        assert srv.is_compatible("hs-old") is True
    finally:
        await _stop_bridge(srv, ts)


@pytest.mark.asyncio
async def test_major_mismatch_flagged_incompatible():
    """bridge_version 999 is recorded and fails closed (no silent mis-mesh)."""
    _, srv, ts = await _make_bridge("hs-srv2", 59214)
    try:
        await _raw_handshake(59214, {
            "type": "handshake", "node_id": "hs-future", "port": 59998,
            "synapse_version": "9.9.9", "bridge_version": 999,
            "features": ["snp1"],
        })
        await asyncio.sleep(0.3)
        rec = srv.get_peer_handshake("hs-future")
        assert rec is not None, "future handshake not recorded"
        assert rec["bridge_version"] == 999
        assert srv.is_compatible("hs-future") is False
    finally:
        await _stop_bridge(srv, ts)


@pytest.mark.asyncio
async def test_unknown_packet_type_does_not_kill_connection():
    """Additive-only: type 0x09 is ignored and later events still deliver."""
    era, a, ta = await _make_bridge("hs-u1", 59215)
    erb, b, tb = await _make_bridge("hs-u2", 59216)
    try:
        ok = await a.connect_to_peer("hs-u2", "127.0.0.1:59216")
        assert ok
        await asyncio.sleep(0.5)

        received = []
        b.event_router.subscribe("synapse/test/unknown-type", received.append)
        _, writer = a._connections["hs-u2"]
        probe = json.dumps({"type": "future", "hello": 1}).encode("utf-8")
        writer.write(struct.pack("!IB", len(probe), 0x09) + probe)
        await writer.drain()
        await asyncio.sleep(0.5)

        assert a.is_peer_connected("hs-u2"), "connection died on unknown packet type"
        await a.event_router.publish(
            "synapse/test/unknown-type", {"n": 42}, origin="local"
        )
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.1)
        assert received, "event stopped delivering after unknown packet type"
        assert received[0].data == {"n": 42}
    finally:
        await _stop_bridge(a, ta)
        await _stop_bridge(b, tb)
