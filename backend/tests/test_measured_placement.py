"""
Measured placement: layer ranges come from measured ms/layer, memory caps
and hop times (DP, EdgeShard-style), not from fixed resource weights.
"""
import sys

import numpy as np
import pytest

from synapse.topology.dp_partition import NodeProfile as P, plan


def _sizes(p):
    return [(nid, b - a + 1) for nid, a, b in p.stages]


def _contiguous(p, total):
    nxt = 0
    for _, a, b in p.stages:
        assert a == nxt and b >= a
        nxt = b + 1
    assert nxt == total


class TestDP:
    def test_equal_nodes_split_evenly_for_throughput(self):
        p = plan([P("a", 1.0, 100), P("b", 1.0, 100)], 30, {("a", "b"): 2.0}, "throughput")
        _contiguous(p, 30)
        assert sorted(n for _, n in _sizes(p)) == [14, 16] and p.cost_ms == 16.0

    def test_latency_objective_avoids_hops_when_one_node_fits(self):
        p = plan([P("a", 1.0, 100), P("b", 1.0, 100)], 30, {("a", "b"): 2.0}, "latency")
        assert len(p.stages) == 1 and p.cost_ms == 30.0

    def test_faster_node_gets_more_layers(self):
        p = plan([P("fast", 1.0, 100), P("slow", 3.0, 100)], 40, {}, "throughput")
        sizes = dict(_sizes(p))
        assert sizes["fast"] > 2.5 * sizes["slow"]

    def test_memory_caps_are_respected_and_force_a_split(self):
        p = plan([P("a", 1.0, 10), P("b", 1.0, 12), P("c", 5.0, 30)], 30, {}, "latency")
        _contiguous(p, 30)
        for nid, n in _sizes(p):
            assert n <= {"a": 10, "b": 12, "c": 30}[nid]

    def test_useless_slow_node_is_left_out(self):
        # "slow" is 50x slower: using it for even one layer loses.
        p = plan([P("a", 1.0, 100), P("b", 1.0, 100), P("slow", 50.0, 100)], 30,
                 {}, "throughput")
        assert "slow" not in [nid for nid, _, _ in p.stages]

    def test_does_not_fit_returns_none(self):
        assert plan([P("a", 1.0, 5), P("b", 1.0, 5)], 30, {}, "throughput") is None

    def test_chain_order_follows_cheap_links(self):
        rtt = {("a", "b"): 50.0, ("a", "c"): 1.0, ("c", "b"): 1.0}
        # throughput: the slowest stage includes its incoming hop, so the
        # 50 ms a-b link must not be in the chain. (Under "latency" every
        # 3-cycle uses all three links, so all orders tie.)
        p = plan([P("a", 1.0, 10), P("b", 1.0, 10), P("c", 1.0, 10)], 30, rtt, "throughput")
        order = [nid for nid, _, _ in p.stages]
        assert order in (["a", "c", "b"], ["b", "c", "a"])

    def test_first_node_pin(self):
        p = plan([P("a", 1.0, 100), P("b", 0.5, 100)], 20, {}, "latency", first_node="a")
        assert p.stages[0][0] == "a"

    def test_matches_brute_force_on_small_cases(self):
        import itertools
        rng = np.random.default_rng(0)
        for _ in range(20):
            nodes = [P(f"n{i}", float(rng.uniform(0.5, 3)), int(rng.integers(3, 12)))
                     for i in range(3)]
            total = int(rng.integers(6, 14))
            rtt = {(a.node_id, b.node_id): float(rng.uniform(0, 3))
                   for a in nodes for b in nodes if a.node_id < b.node_id}
            got = plan(nodes, total, rtt, "throughput")
            best = None
            for r in range(1, 4):
                for order in itertools.permutations(nodes, r):
                    for cuts in itertools.product(range(1, total), repeat=r - 1):
                        if list(cuts) != sorted(set(cuts)):
                            continue
                        bounds = [0, *cuts, total]
                        cost, ok = 0.0, True
                        for i, node in enumerate(order):
                            k = bounds[i + 1] - bounds[i]
                            if k > node.max_layers:
                                ok = False
                                break
                            hop = 0.0 if i == 0 else rtt.get(
                                tuple(sorted((order[i - 1].node_id, node.node_id))), 2.0)
                            cost = max(cost, k * node.ms_per_layer + hop)
                        if ok and (best is None or cost < best):
                            best = cost
            if best is None:
                assert got is None
            else:
                assert got is not None and abs(got.cost_ms - best) < 1e-9


class TestNodeWiring:
    @staticmethod
    def _node(monkeypatch, ids=("A", "B")):
        from synapse.orchestration.node import Node
        from synapse.routing.event_router import EventRouter
        from synapse.routing.libp2p_node import Libp2pNode
        from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
        from synapse.inference.inference_engine import InferenceEngine

        class E(InferenceEngine):
            async def encode(self, s, p): return np.array([])
            async def sample(self, x): return np.array([1])
            async def decode(self, s, t): return ""
            async def infer_tensor(self, r, s, t, st=None): return np.array([]), {}
            async def load_checkpoint(self, s, p): pass

        router = EventRouter()
        node = Node(_id=ids[0], event_router=router,
                    libp2p_node=Libp2pNode(ids[0], 59998, router),
                    inference_engine=E(), shard_downloader=None)
        for nid in ids:
            node.topology.update_node(nid, DeviceCapabilities(
                model=nid, chip="cpu", memory=8000,
                flops=DeviceFlops(fp32=1.0, fp16=1.0, int8=1.0)))
        return node

    def test_placement_uses_gossiped_profiles(self, monkeypatch):
        from synapse.inference.shard import Shard
        node = self._node(monkeypatch)
        node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}
        node._peer_capabilities = {"B": {"layer_profiles": {
            "m": {"ms_per_layer": 3.0, "max_layers": 100}}}}
        node._measured_rtt = {"B": 0.0}  # measured, negligible hop
        parts = node._compute_lacp_partition(Shard("m", 0, 39, 40))
        sizes = {p.node_id: round((p.end - p.start) * 40) for p in parts}
        assert sizes["A"] == 30 and sizes["B"] == 10  # 30 x 1 ms = 10 x 3 ms
        assert node._last_placement["objective"] == "throughput"

    def test_missing_profile_falls_back_to_legacy(self, monkeypatch):
        from synapse.inference.shard import Shard
        node = self._node(monkeypatch)
        node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}
        node._peer_capabilities = {}
        parts = node._compute_lacp_partition(Shard("m", 0, 39, 40))
        assert len(parts) == 2 and getattr(node, "_last_placement", None) is None

    def test_small_measurement_noise_does_not_move_layers(self, monkeypatch):
        from synapse.inference.shard import Shard
        node = self._node(monkeypatch)
        node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}
        node._peer_capabilities = {"B": {"layer_profiles": {"m": {"ms_per_layer": 1.0, "max_layers": 100}}}}
        first = [(p.node_id, p.start, p.end) for p in node._compute_lacp_partition(Shard("m", 0, 29, 30))]
        node._peer_capabilities["B"]["layer_profiles"]["m"]["ms_per_layer"] = 1.05
        again = [(p.node_id, p.start, p.end) for p in node._compute_lacp_partition(Shard("m", 0, 29, 30))]
        assert again == first
        node._peer_capabilities["B"]["layer_profiles"]["m"]["ms_per_layer"] = 3.0
        moved = [(p.node_id, p.start, p.end) for p in node._compute_lacp_partition(Shard("m", 0, 29, 30))]
        assert moved != first

    def test_legacy_switch(self, monkeypatch):
        from synapse.inference.shard import Shard
        monkeypatch.setenv("SYNAPSE_PARTITIONER", "legacy")
        node = self._node(monkeypatch)
        node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}
        node._peer_capabilities = {"B": {"layer_profiles": {"m": {"ms_per_layer": 3.0, "max_layers": 100}}}}
        node._compute_lacp_partition(Shard("m", 0, 39, 40))
        assert getattr(node, "_last_placement", None) is None


@pytest.mark.asyncio
async def test_profile_switches_from_probe_to_real_forwards(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    from synapse.topology.layer_profile import local_profile, layer_params
    model_id, n = "peft-internal-testing/tiny-random-GPT2LMHeadModel", 5
    assert layer_params(model_id) > 0
    eng = PyTorchHFInferenceEngine()
    before = local_profile(eng, model_id)
    assert before["source"] == "probe" and before["ms_per_layer"] > 0 and before["max_layers"] > 0
    sh = Shard(model_id, 0, n - 1, n)
    ids = [1, 2, 3]
    for _ in range(3):
        x, _ = await eng.infer_tensor("p", sh, np.array(ids, dtype=np.int64))
        ids.append(int(np.asarray(x).reshape(-1, x.shape[-1])[-1].argmax()))
    after = local_profile(eng, model_id)
    assert after["source"] == "forward" and after["ms_per_layer"] > 0
