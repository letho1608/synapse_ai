"""
Behavioral test: a dead master (stops publishing election heartbeats) is
demoted through the real election path in a bounded window — much shorter
than the previous ~36 s — while an alive master is never reassigned
(stickiness; covered by test_election_multiresource.py::
test_stickiness_prevents_split_brain, which must stay green).

Drives the shipped ElectionManager with its production TIMEOUT (3 s) and
MASTER_STICKINESS_GRACE_PERIOD (4 s) constants; only the tick cadence is
sped up so the test itself is fast. Expected demotion window ≈ 7-8 s.
"""
import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_dead_master_demoted_in_bounded_window():
    from synapse.orchestration.election import ElectionManager
    from synapse.routing import EventRouter

    router = EventRouter()
    a = ElectionManager("node-a", router, resource_scorer=lambda nid: 1.0)
    b = ElectionManager("node-b", router, resource_scorer=lambda nid: 1.0)
    # Fast ticks for the test; the shipped timeout/grace constants are left at
    # their production values (TIMEOUT=3.0, MASTER_STICKINESS_GRACE_PERIOD=4.0)
    # so the demotion window is the real one.
    a.TICK_INTERVAL = 0.1
    b.TICK_INTERVAL = 0.1

    await router.start()
    await a.start()
    await b.start()
    try:
        # Let the cluster elect once (equal weights, tie broken by id).
        for _ in range(100):
            await asyncio.sleep(0.05)
            if a.current_master_id is not None and b.current_master_id is not None:
                break
        assert a.current_master_id is not None
        assert a.current_master_id == b.current_master_id, (
            f"initial election disagreed: A={a.current_master_id} B={b.current_master_id}"
        )
        master = a.current_master_id
        master_mgr = a if master == "node-a" else b
        survivor = b if master == "node-a" else a

        # Kill the master: stop its election loop so it publishes no more
        # heartbeats (the production death condition for FT demotion).
        await master_mgr.stop()

        t0 = time.monotonic()
        while time.monotonic() - t0 < 20.0:
            await asyncio.sleep(0.1)
            if survivor.current_master_id != master:
                break

        elapsed = time.monotonic() - t0
        assert survivor.current_master_id != master, (
            "dead master was never demoted within 20 s"
        )
        # Shipped window is TIMEOUT(3 s) + GRACE(4 s) ≈ 7-8 s; require well
        # under the previous ~36 s so failover is genuinely responsive.
        assert elapsed < 15.0, f"demotion took {elapsed:.1f}s (expect < 15 s)"

        # The survivor is the new master (only remaining candidate).
        assert survivor.is_master()
        assert survivor.current_master_id == survivor.node_id
    finally:
        await a.stop()
        await b.stop()
        await router.stop()
