"""
Edge-case tests for ElectionManager — dynamic weight, election logic.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.orchestration.election import ElectionManager


async def test_election_loop_runs():
    """Test that election loop starts without crashing (catches missing attrs)."""
    er = EventRouter()
    await er.start()

    em = ElectionManager("node-1", er, compute_weight=100.0)
    await em.start()

    # Let it run a few ticks
    await asyncio.sleep(5)

    assert em.current_master_id == "node-1", f"Expected node-1 as master, got {em.current_master_id}"

    await em.stop()
    await er.stop()
    print("test_election_loop_runs: PASS")


async def test_election_tiebreak():
    """Test that tiebreaker works: higher weight wins, then lexicographic ID."""
    er = EventRouter()
    await er.start()

    em_a = ElectionManager("node-a", er, compute_weight=50.0, dynamic_weight=False)
    em_b = ElectionManager("node-b", er, compute_weight=100.0, dynamic_weight=False)
    em_c = ElectionManager("node-c", er, compute_weight=100.0, dynamic_weight=False)

    await em_a.start()
    await em_b.start()
    await em_c.start()

    # Poll until election converges (up to 10 seconds)
    master_a = master_b = master_c = None
    for _ in range(20):
        await asyncio.sleep(0.5)
        master_a = em_a.current_master_id
        master_b = em_b.current_master_id
        master_c = em_c.current_master_id
        if master_a and master_b and master_c:
            if master_a == master_b == master_c:
                break

    # node-b and node-c both have weight=100, node-b < node-c lexicographically
    assert master_a == "node-b", f"A: Expected node-b, got {master_a}"
    assert master_b == "node-b", f"B: Expected node-b, got {master_b}"
    assert master_c == "node-b", f"C: Expected node-b, got {master_c}"

    await em_a.stop()
    await em_b.stop()
    await em_c.stop()
    await er.stop()
    print("test_election_tiebreak: PASS")


async def test_election_timeout_cleanup():
    """Test that nodes are removed from active_nodes after timeout."""
    er = EventRouter()
    await er.start()

    em = ElectionManager("node-1", er, dynamic_weight=False)
    await em.start()

    # Manually inject an old heartbeat
    now = asyncio.get_event_loop().time()
    em.active_nodes["dead-node"] = {"weight": 1.0, "last_seen": now - 10}  # > TIMEOUT=6s

    await asyncio.sleep(4)  # Let election loop run

    # After cleanup, dead-node should be gone
    assert "dead-node" not in em.active_nodes, f"dead-node not cleaned up: {em.active_nodes}"

    await em.stop()
    await er.stop()
    print("test_election_timeout_cleanup: PASS")


async def test_dynamic_weight_disabled():
    """Test that dynamic_weight=False skips utilization queries."""
    er = EventRouter()
    await er.start()

    em = ElectionManager("node-1", er, compute_weight=100.0)
    em.dynamic_weight = False
    em.base_compute_weight = 100.0
    await em.start()

    await asyncio.sleep(3)

    # Weight should not change when dynamic_weight is disabled
    assert em.compute_weight == 100.0, f"Expected 100.0, got {em.compute_weight}"

    await em.stop()
    await er.stop()
    print("test_dynamic_weight_disabled: PASS")


if __name__ == "__main__":
    results = []
    for name, coro in [
        ("election_loop_runs", test_election_loop_runs),
        ("election_tiebreak", test_election_tiebreak),
        ("election_timeout_cleanup", test_election_timeout_cleanup),
        ("dynamic_weight_disabled", test_dynamic_weight_disabled),
    ]:
        print(f"\n--- Running: {name} ---")
        try:
            asyncio.run(coro())
            results.append((name, "PASS"))
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))

    print("\n" + "=" * 50)
    passed = sum(1 for _, r in results if r == "PASS")
    failed = sum(1 for _, r in results if r == "FAIL")
    for name, result in results:
        print(f"  {name}: {result}")
    print(f"\nTotal: {passed} PASS, {failed} FAIL")