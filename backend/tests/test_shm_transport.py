"""
Behavioral tests for the LACP zero-copy shared-memory transport.

These tests prove the transport exists as a REAL same-machine path that
bypasses TCP: tensor payloads travel through multiprocessing.shared_memory
segments (asserted via segment names + counters), and the receive side hands
out zero-copy numpy views over the shared buffer.
"""

import mmap
import numpy as np
import pytest
from multiprocessing import shared_memory

from synapse.transport.shm_transport import (
    SharedMemoryTransport,
    host_of,
    is_same_machine,
    SHM_HEADER,
    SHM_MAGIC,
)
from synapse.transport.codec import TensorMessage


def test_host_of_and_same_machine_helpers():
    assert host_of("127.0.0.1:9000") == "127.0.0.1"
    assert host_of("node-a:9000") == "node-a"
    assert host_of("::1:9001") == "::1"

    # Loopback aliases always denote the local machine.
    assert is_same_machine("127.0.0.1:9000", "127.0.0.1:9001")
    assert is_same_machine("localhost:9000", "127.0.0.1:9001")
    assert is_same_machine("127.0.0.1:9000", "::1:9001")
    assert is_same_machine("0.0.0.0:9000", "127.0.0.1:9001")
    # Identical host, different ports -> same machine.
    assert is_same_machine("10.0.0.5:9000", "10.0.0.5:9001")
    # Different hosts -> not the same machine.
    assert not is_same_machine("10.0.0.5:9000", "10.0.0.6:9001")


async def test_zero_copy_delivery_between_same_process_peers():
    a = SharedMemoryTransport("node-a")
    b = SharedMemoryTransport("node-b")
    await a.start()
    await b.start()
    received: list = []
    b.set_tensor_handler(lambda msg: received.append(msg))

    a.register_peer("node-b:0", "node-b")

    tensor = np.arange(100, dtype=np.float32).reshape(10, 10)
    ok = await a.send_tensor(
        "req1",
        1,
        tensor,
        "node-b:0",
        metadata={
            "micro_batch_id": 3,
            "source_stage": 0,
            "orchestrator_id": "orch-1",
            "shard": {"model": "llama-7b", "offset": 42},
        },
    )
    assert ok is True

    # The payload traveled through shared memory: the receiver saw exactly the
    # segment the sender wrote, and both sides count the SHM path.
    assert a.shm_sends == 1
    assert b.shm_recvs == 1
    assert len(b.segments_seen) == 1
    assert b.segments_seen == a.segments_sent

    assert len(received) == 1
    msg = received[0]
    assert isinstance(msg, TensorMessage)
    assert msg.request_id == "req1"
    assert msg.target_stage == 1
    assert msg.source_stage == 0
    assert msg.micro_batch_id == 3
    assert msg.orchestrator_id == "orch-1"
    assert msg.tensor.dtype == np.float32
    assert msg.tensor.shape == (10, 10)
    np.testing.assert_array_equal(msg.tensor, tensor)
    # Nested metadata survives the shared-memory round trip.
    assert msg.metadata["shard"] == {"model": "llama-7b", "offset": 42}

    await a.stop()
    await b.stop()


async def test_receive_is_zero_copy_view_over_shared_buffer():
    a = SharedMemoryTransport("node-a")
    b = SharedMemoryTransport("node-b")
    await a.start()
    await b.start()
    received: list = []
    b.set_tensor_handler(lambda msg: received.append(msg))
    a.register_peer("node-b:0", "node-b")

    tensor = np.arange(50, dtype=np.float64)
    assert await a.send_tensor("req-zc", 0, tensor, "node-b:0") is True
    msg = received[0]

    # The received tensor is a view over the shared segment, not a copy.
    # (On POSIX the buffer is a memoryview; on Windows it is the underlying
    # mmap — either way the array aliases the shared buffer directly.)
    assert isinstance(msg.tensor.base, (memoryview, mmap.mmap))

    # Prove the view aliases the shared buffer: mutate the received tensor and
    # observe the change through an independent handle to the same segment.
    segment_name = b.segments_seen[0]
    shm = shared_memory.SharedMemory(name=segment_name)
    try:
        magic, payload_len, _meta_len = SHM_HEADER.unpack_from(shm.buf, 0)
        assert magic == SHM_MAGIC
        payload = shm.buf[SHM_HEADER.size:SHM_HEADER.size + payload_len]
        before = np.frombuffer(payload, dtype=np.float64).copy()
        np.testing.assert_array_equal(before, tensor)

        msg.tensor[0] = 12345.0
        after = np.frombuffer(payload, dtype=np.float64)
        assert after[0] == 12345.0
        # Release views into the extra handle before closing it.
        del after, before, payload
        shm.close()
    finally:
        if getattr(shm, "_mmap", None) is not None:
            shm.close()

    await a.stop()
    await b.stop()


async def test_messages_delivered_in_order():
    a = SharedMemoryTransport("node-a")
    b = SharedMemoryTransport("node-b")
    await a.start()
    await b.start()
    received: list = []
    b.set_tensor_handler(lambda msg: received.append(msg))
    a.register_peer("node-b:0", "node-b")

    for i in range(20):
        await a.send_tensor(f"req-{i}", 0, np.array([i], dtype=np.int64), "node-b:0")

    assert [m.request_id for m in received] == [f"req-{i}" for i in range(20)]
    assert [int(m.tensor[0]) for m in received] == list(range(20))
    assert b.shm_recvs == 20
    assert len(b.segments_seen) == 20

    await a.stop()
    await b.stop()


async def test_send_to_unregistered_or_remote_peer_returns_false():
    a = SharedMemoryTransport("node-a")
    await a.start()

    # Completely unknown address -> caller must fall back to TCP.
    assert await a.send_tensor("r", 0, np.zeros(4), "10.9.9.9:9000") is False

    # Registered address whose transport is not attached in this process
    # (a genuine remote machine) -> same fallback, no segment written.
    a.register_peer("10.9.9.9:9000", "node-remote")
    assert await a.send_tensor("r", 0, np.zeros(4), "10.9.9.9:9000") is False
    assert a.shm_sends == 0
    assert a.segments_sent == []

    await a.stop()


async def test_stop_releases_active_segments():
    a = SharedMemoryTransport("node-a")
    b = SharedMemoryTransport("node-b")
    await a.start()
    await b.start()
    received: list = []
    b.set_tensor_handler(lambda msg: received.append(msg))
    a.register_peer("node-b:0", "node-b")

    await a.send_tensor("r", 0, np.arange(8), "node-b:0")
    assert len(received) == 1
    # The receiving transport keeps the segment alive while the view is used.
    assert len(b._active) == 1

    await b.stop()
    assert b._active == {}
    assert b.shm_recvs == 1

    # After detach, the peer registry no longer routes to b.
    assert "node-b" not in SharedMemoryTransport._local_registry
    await a.stop()
    assert SharedMemoryTransport._local_registry == {}
