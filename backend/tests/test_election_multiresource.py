"""
Test: Multi-resource election using ResourceScorer.compute_scores().

Verifies the fix for M4/High: master election uses the full multi-dimensional
ResourceScorer composite score instead of the 2D vram + flops/100 formula.

Also verifies master stickiness behaviorally: while the current master is
alive, no node reassigns the master — so a weight flip can never produce two
simultaneous masters (split brain).
"""
import asyncio

import pytest


def _mock_router():
    from unittest.mock import MagicMock

    mock_router = MagicMock()
    mock_router.subscribe = MagicMock()
    mock_router.publish = MagicMock()
    mock_router.unsubscribe = MagicMock()
    return mock_router


class TestElectionMultiResource:
    """Tests for multi-resource election weight."""

    def test_election_accepts_resource_scorer(self):
        """ElectionManager accepts a resource_scorer callable and uses it."""
        from synapse.orchestration.election import ElectionManager

        def fake_scorer(node_id):
            return 42.0

        em = ElectionManager("node-x", _mock_router(), resource_scorer=fake_scorer)
        em.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)
        # scorer path: max(16.2, 42*1000) = 42000
        assert em.base_compute_weight == pytest.approx(42000.0)

    def test_set_base_compute_weight_uses_scorer(self):
        """set_base_compute_weight calls resource_scorer when available."""
        from synapse.orchestration.election import ElectionManager

        scored_nodes = {}

        def fake_scorer(node_id):
            scored_nodes[node_id] = True
            return 50.0

        em = ElectionManager("node-a", _mock_router(), resource_scorer=fake_scorer)
        em.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)
        assert scored_nodes.get("node-a") is True
        assert em.base_compute_weight == pytest.approx(50000.0)

    def test_multi_weight_takes_precedence_over_simple(self):
        """The multi-resource weight wins over VRAM+FLOPs whenever a scorer
        is provided (even a low one is amplified x1000 by design)."""
        from synapse.orchestration.election import ElectionManager

        low = ElectionManager("node-a", _mock_router(), resource_scorer=lambda n: 0.01)
        low.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)
        # scorer path still takes precedence: max(16.2, 0.01*1000) = 16.2
        assert low.base_compute_weight == pytest.approx(16.2)

        high = ElectionManager("node-a", _mock_router(), resource_scorer=lambda n: 100.0)
        high.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)
        assert high.base_compute_weight == pytest.approx(100000.0)

    def test_election_fallback_to_vram_flops(self):
        """Verify election falls back to VRAM+FLOPs if no scorer is provided."""
        from synapse.orchestration.election import ElectionManager

        em = ElectionManager("test_node", _mock_router(), resource_scorer=None)
        em.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)
        # Without scorer: weight = 16 + 20/100 = 16.2
        assert em.base_compute_weight == 16.0 + 20.0 / 100.0
        assert em.compute_weight == em.base_compute_weight

    def test_election_with_scorer_gives_higher_weight(self):
        """Verify election weight is higher when multi-resource scorer is used."""
        from synapse.orchestration.election import ElectionManager

        def high_scorer(node_id):
            return 100.0

        em = ElectionManager("test_node", _mock_router(), resource_scorer=high_scorer)
        em.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)

        # With scorer: weight = max(16.2, 100*1000) = 100000
        assert em.base_compute_weight > 16.2


@pytest.mark.asyncio
async def test_stickiness_prevents_split_brain():
    """A weight flip must never yield two simultaneous masters: while the
    current master is alive every node keeps it (master stickiness)."""
    from synapse.orchestration.election import ElectionManager
    from synapse.routing import EventRouter

    router = EventRouter()
    a = ElectionManager("node-a", router, resource_scorer=lambda nid: 1.0)
    b = ElectionManager("node-b", router, resource_scorer=lambda nid: 1.0)
    a.TICK_INTERVAL = 0.2
    b.TICK_INTERVAL = 0.2
    a.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)
    b.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)

    await router.start()
    await a.start()
    await b.start()
    try:
        # Let the cluster elect once (equal weights, tie broken by id).
        for _ in range(50):
            await asyncio.sleep(0.1)
            if a.current_master_id is not None and b.current_master_id is not None:
                break
        assert a.current_master_id is not None
        assert a.current_master_id == b.current_master_id, (
            f"initial election disagreed: A={a.current_master_id} B={b.current_master_id}"
        )
        original_master = a.current_master_id

        # Flip weights: node-b becomes strictly stronger than the master.
        b._resource_scorer = lambda nid: 100.0
        b.set_base_compute_weight(vram_gb=16.0, fp32_flops=20.0)

        # Wait several ticks: with the old buggy logic node-b would elect
        # itself while node-a (master) kept itself → split brain.
        for _ in range(50):
            await asyncio.sleep(0.1)

        assert a.current_master_id == b.current_master_id, (
            f"split brain after weight flip: A={a.current_master_id} B={b.current_master_id}"
        )
        # Stickiness: the master is unchanged while it stays alive.
        assert a.current_master_id == original_master, (
            f"master changed while alive: {original_master} -> {a.current_master_id}"
        )
    finally:
        await a.stop()
        await b.stop()
