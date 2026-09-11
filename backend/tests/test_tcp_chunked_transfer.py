"""
Behavioral tests: chunked TCP transfer + auto-router + bridge NODELAY (T1.2).

- split/parse round-trip incl. a 130 MiB raw blob (would-be ~260 MiB framed
  over the JSON+hex bridge — the exact ceiling this kills).
- Admission refusal without multi-GiB allocation (declared-total attack).
- Out-of-order reassembly, timeout purge, inconsistent-total ignore.
- Live TensorTCPTransport pair: 12 MiB tensor blob delivers end-to-end and
  decodes bit-exact; bad frames are skipped, connection survives.
- Router decisions: in_process/shm/tcp-chunked/bridge + counters.
- B4: should_quantize exact-dtype default, opt-in only, never non-float.
- Bridge sockets carry TCP_NODELAY on both directions.

Ports 59411/59412: unused by any other test module in this repo.
"""
import asyncio
import socket
import struct
import uuid

import numpy as np
import pytest

from synapse.transport.codec import (
    CHUNK_MAGIC,
    CHUNK_SIZE,
    ChunkReassembler,
    ChunkRefused,
    decode_tensor,
    encode_tensor,
    parse_chunk_frame,
    split_blob,
)
from synapse.transport.router import (
    ROUTE_BRIDGE,
    ROUTE_IN_PROCESS,
    ROUTE_SHM,
    ROUTE_TCP_CHUNKED,
    TransportStats,
    choose_transport,
    chunking_enabled,
    should_quantize,
)
from synapse.transport.tcp_transport import TensorTCPTransport


def test_split_parse_roundtrip_small():
    blob = b"SYNP" + bytes(range(256)) * 100
    tid = uuid.uuid4().bytes
    frames = split_blob(blob, tid)
    assert len(frames) == 1
    got_tid, idx, total, final, payload = parse_chunk_frame(frames[0])
    assert got_tid == tid and (idx, total, final) == (0, 1, True)
    r = ChunkReassembler()
    assert r.add(got_tid, idx, total, final, payload) == blob
    assert r.pending == 0


def test_split_parse_roundtrip_130mib():
    """130 MiB raw would frame to ~260 MiB of JSON+hex: bridge ceiling.

    NOTE: SYNP shape dims are !H (<=65535 by design, wire-frozen), so the
    blob is shaped 3-D like a real activation, not flat.
    """
    raw = np.arange(256 * 512 * 260, dtype=np.float32).reshape(256, 512, 260)
    blob = encode_tensor(raw, metadata={"request_id": "big"})
    assert len(blob) > 130 * 1024 * 1024
    tid = uuid.uuid4().bytes
    frames = split_blob(blob, tid)
    assert len(frames) == (len(blob) + CHUNK_SIZE - 1) // CHUNK_SIZE
    assert len(frames) > 100
    r = ChunkReassembler()
    out = None
    # Deliver out of order: last, first, rest.
    order = [len(frames) - 1, 0] + list(range(1, len(frames) - 1))
    for i in order:
        t, idx, total, final, payload = parse_chunk_frame(frames[i])
        res = r.add(t, idx, total, final, payload)
        if res is not None:
            out = res
    assert out == blob, "reassembly must be byte-exact regardless of order"
    tensor, meta = decode_tensor(out)
    assert np.array_equal(tensor, raw)
    assert meta["request_id"] == "big"


def test_declared_total_attack_refused_without_allocation():
    header = CHUNK_MAGIC + struct.pack("!16sIIBI", uuid.uuid4().bytes, 0, 10 ** 9, 0, 4) + b"xxxx"
    with pytest.raises(ChunkRefused):
        parse_chunk_frame(header)
    with pytest.raises(ValueError):
        parse_chunk_frame(b"SYNP" + b"0" * 40)
    with pytest.raises(ValueError):
        split_blob(b"", uuid.uuid4().bytes)


def test_reassembler_timeout_and_inconsistent_total():
    r = ChunkReassembler(timeout_s=0.05)
    tid = uuid.uuid4().bytes
    assert r.add(tid, 0, 3, False, b"a") is None
    assert r.pending == 1
    # Inconsistent total retransmit is ignored, not corrupting.
    assert r.add(tid, 1, 99, False, b"b") is None
    assert r.pending == 1
    import time
    time.sleep(0.08)
    assert r.purge() == 1
    assert r.pending == 0


@pytest.mark.asyncio
async def test_chunked_tensor_end_to_end_over_tcp():
    """12 MiB tensor blob via send_blob_chunked decodes bit-exact."""
    srv = TensorTCPTransport("tcp-srv", port=59411)
    cli = TensorTCPTransport(node_id="tcp-cli")
    received = []

    def _on_msg(msg):
        received.append(msg)

    srv.set_tensor_handler(_on_msg)
    await srv.start()
    try:
        arr = np.arange(64 * 192 * 256, dtype=np.float32).reshape(64, 192, 256)
        blob = encode_tensor(arr, metadata={
            "request_id": "r1", "orchestrator_id": "tcp-cli",
            "micro_batch_id": 0, "source_stage": 0, "target_stage": 8})
        ok = await cli.send_blob_chunked("127.0.0.1:59411", blob)
        assert ok, "chunked send failed"
        for _ in range(100):
            if received:
                break
            await asyncio.sleep(0.1)
        assert received, "no tensor arrived"
        assert np.array_equal(received[0].tensor, arr)
        assert received[0].micro_batch_id == 0
        assert received[0].target_stage == 8
        assert srv.server.chunks_completed >= 1
    finally:
        await cli.stop()
        await srv.stop()


@pytest.mark.asyncio
async def test_bad_chunk_frame_skipped_connection_survives():
    srv = TensorTCPTransport("tcp-srv2", port=59412)
    cli = TensorTCPTransport(node_id="tcp-cli2")
    received = []
    srv.set_tensor_handler(received.append)
    await srv.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 59412)
        try:
            # Garbage declaring itself a chunk: skipped, loop lives on.
            bad = CHUNK_MAGIC + b"\x00" * 10
            writer.write(struct.pack("!I", len(bad)) + bad)
            await writer.drain()
            await asyncio.sleep(0.3)
            arr = np.arange(8, dtype=np.int32)
            blob = encode_tensor(arr, metadata={
                "request_id": "r2", "micro_batch_id": 0,
                "source_stage": 0, "target_stage": 1})
            ok = await cli.send_blob_chunked("127.0.0.1:59412", blob)
            assert ok
            for _ in range(100):
                if received:
                    break
                await asyncio.sleep(0.1)
            assert received and np.array_equal(received[0].tensor, arr)
        finally:
            writer.close()
    finally:
        await cli.stop()
        await srv.stop()


def test_router_branches_and_counters():
    assert choose_transport("a", "a")[0] == ROUTE_IN_PROCESS
    assert choose_transport("a", "b", "h:1", 100, shm_registered=True)[0] == ROUTE_SHM
    # SHM over cap falls through.
    r, _ = choose_transport("a", "b", "h:1", 600 * 1024 * 1024,
                            shm_registered=True, tcp_registered=True,
                            chunked=True)
    assert r == ROUTE_TCP_CHUNKED
    assert choose_transport("a", "b", "h:1", 100, tcp_registered=True,
                            chunked=True)[0] == ROUTE_TCP_CHUNKED
    assert choose_transport("a", "b", "h:1", 100, tcp_registered=True,
                            chunked=False)[0] == ROUTE_BRIDGE
    assert choose_transport("a", "b", None, 100)[0] == ROUTE_BRIDGE
    assert choose_transport("a", "b", "h:1", 100)[0] == ROUTE_BRIDGE
    s = TransportStats()
    s.record("tcp-chunked")
    s.record("tcp-chunked")
    s.record("bridge")
    assert s.snapshot() == {"tcp-chunked": 2, "bridge": 1}


def test_should_quantize_exact_default(monkeypatch):
    monkeypatch.delenv("SYNAPSE_TRANSPORT_INT8", raising=False)
    assert should_quantize("float32") is False
    assert should_quantize("float16") is False
    monkeypatch.setenv("SYNAPSE_TRANSPORT_INT8", "1")
    assert should_quantize("float32") is True
    assert should_quantize("float16") is True
    assert should_quantize("int64") is False
    assert should_quantize("int8") is False
    assert chunking_enabled({"SYNAPSE_CHUNKED_TCP": "1"}) is True
    assert chunking_enabled({}) is True  # default ON (bridge fallback intact)
    assert chunking_enabled({"SYNAPSE_CHUNKED_TCP": "0"}) is False


@pytest.mark.asyncio
async def test_bridge_sockets_carry_nodelay():
    """T1.2: bridge sets TCP_NODELAY both directions (was Nagle-enabled)."""
    from synapse.routing.event_router import EventRouter
    from synapse.routing.p2p_socket_bridge import P2PSocketBridge

    async def _mk(nid, port):
        er = EventRouter()
        try:
            await er.start()
        except Exception:
            pass
        br = P2PSocketBridge(nid, port, er)
        t = asyncio.create_task(br.start())
        await asyncio.sleep(0.3)
        return br, t

    a, ta = await _mk("nd-a", 59413)
    b, tb = await _mk("nd-b", 59414)
    try:
        assert await a.connect_to_peer("nd-b", "127.0.0.1:59414")
        await asyncio.sleep(0.5)

        def _nodelay(writer):
            sock = writer.get_extra_info("socket")
            assert sock is not None
            return sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)

        _, wa = a._connections["nd-b"]
        assert _nodelay(wa) == 1
        _, wb = b._connections["nd-a"]
        assert _nodelay(wb) == 1
    finally:
        for br, t in ((a, ta), (b, tb)):
            try:
                await br.stop()
            except Exception:
                pass
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
