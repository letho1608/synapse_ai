"""
B6+B7 cleanup: honest ILP/placeholder-prober removal is verified by grep
evidence; this module carries the behavioral tests that enforce the hygiene
items:

  - the SHM transport's retained-segment set (`_active`) is bounded instead of
    growing without limit.
"""
import numpy as np

from synapse.transport.shm_transport import SharedMemoryTransport

# ---------------------------------------------------------------------------
# SHM transport: _active is bounded
# ---------------------------------------------------------------------------


async def test_shm_active_segments_bounded():
    """Retained segment handles are capped (LRU), delivery unaffected."""
    a = SharedMemoryTransport("node-a", max_active_segments=4)
    b = SharedMemoryTransport("node-b", max_active_segments=4)
    await a.start()
    await b.start()
    received: list = []
    b.set_tensor_handler(lambda msg: received.append(msg))
    a.register_peer("node-b:0", "node-b")

    for i in range(20):
        await a.send_tensor(f"req-{i}", 0, np.array([i], dtype=np.int64), "node-b:0")

    # The retained-handle set stays at the cap on both sides.
    assert len(a._active) <= 4
    assert len(b._active) <= 4
    # Delivery is unaffected: every segment still arrives.
    assert [m.request_id for m in received] == [f"req-{i}" for i in range(20)]
    assert b.shm_recvs == 20
    # Eviction is LRU: the newest segments are still retained.
    assert a.segments_sent[-1] in a._active
    assert b.segments_seen[-1] in b._active

    await a.stop()
    await b.stop()
