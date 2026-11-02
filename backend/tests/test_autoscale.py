"""
Behavioral tests: autoscale evaluator, advisory-only (T3.3 / C7, P4).

- Unserved model with fit standby -> promote.
- Unknown fit (None) never promotes, even unserved.
- Hot model (rpm/server >= 10 + >= 3x median) -> promote.
- Queue backstop promotes without rate signal.
- Idle multi-server with reporters -> demote; no reporters -> never demote.
- Malformed inputs -> [] (never raises).
"""
import pytest

from synapse.orchestration.autoscale import evaluate


def _fit_yes(_n, _m):
    return True


def _fit_none(_n, _m):
    return None


def _agg(rpm=0.0, reporters=0, queue=0, inflight=0):
    return {"rpm": rpm, "reporters": reporters, "unknown": 0,
            "queue_depth": queue, "in_flight": inflight}


def test_unserved_model_promotes_fit_standby():
    d = evaluate({"m": _agg()}, {}, ["s1"], _fit_yes)
    assert d == [{"action": "promote", "model": "m", "node": "s1",
                  "reason": "unserved-model"}]


def test_unknown_fit_never_promotes():
    assert evaluate({"m": _agg()}, {}, ["s1"], _fit_none) == []
    assert evaluate({"m": _agg(rpm=100.0, reporters=5)}, {"m": ["a"]},
                    ["s1"], _fit_none) == []


def test_hot_model_promotes():
    demand = {"hot": _agg(rpm=30.0, reporters=4), "cold": _agg(rpm=1.0, reporters=4)}
    d = evaluate(demand, {"hot": ["a"], "cold": ["b"]}, ["s1"], _fit_yes)
    assert d == [{"action": "promote", "model": "hot", "node": "s1",
                  "reason": "hot-model"}]


def test_queue_backstop_promotes_without_rates():
    d = evaluate({"m": _agg(queue=64)}, {"m": ["a"]}, ["s1"], _fit_yes)
    assert d == [{"action": "promote", "model": "m", "node": "s1",
                  "reason": "queue-backstop"}]


def test_idle_servers_demoted_only_with_reporters():
    d = evaluate({"m": _agg(rpm=0.5, reporters=3)}, {"m": ["a", "b"]}, [], _fit_yes)
    assert d == [{"action": "demote", "model": "m", "node": "b",
                  "reason": "idle-servers"}]
    # No reporters: no evidence -> no demote.
    assert evaluate({"m": _agg()}, {"m": ["a", "b"]}, [], _fit_yes) == []
    # Single server never demoted.
    assert evaluate({"m": _agg(rpm=0.0, reporters=3)}, {"m": ["a"]}, [], _fit_yes) == []


def test_malformed_inputs_never_raise():
    assert evaluate(None, None, None) == []
    assert evaluate({"m": "junk"}, {"m": ["a"]}, ["s"], _fit_yes) == []
    assert evaluate({}, {}, []) == []


# -- actuation (node side) ----------------------------------------------------

def _wired_node(monkeypatch):
    from test_measured_placement import TestNodeWiring
    node = TestNodeWiring._node(monkeypatch)
    node.election_manager.is_master = lambda: True
    # A is fast; B is slow and behind a long hop, so the latency objective
    # serves from A alone (B on standby) while throughput spreads layers.
    node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}
    node._peer_capabilities = {"B": {"layer_profiles": {"m": {"ms_per_layer": 2.0, "max_layers": 100}}}}
    node._measured_rtt = {"B": 10.0}
    node.partition_objective = "latency"
    return node


@pytest.mark.asyncio
async def test_promote_spreads_layers_then_demote_restores(monkeypatch):
    from synapse.inference.shard import Shard
    node = _wired_node(monkeypatch)
    shard = Shard("m", 0, 39, 40)
    alone = {p.node_id for p in node._compute_lacp_partition(shard)}
    assert alone == {"A"}, "precondition: latency placement leaves B on standby"
    assert node._standby_fits("B", "m") is True

    assert await node._actuate_autoscale([{"action": "promote", "model": "m", "node": "B"}], now=0.0) == "throughput"
    spread = {p.node_id for p in node._compute_lacp_partition(shard)}
    assert spread == {"A", "B"}

    # cooldown: an idle signal right after does nothing
    assert await node._actuate_autoscale([{"action": "demote", "model": "m", "node": "B"}], now=10.0) is None
    assert await node._actuate_autoscale([{"action": "demote", "model": "m", "node": "B"}], now=100.0) == "configured"
    assert {p.node_id for p in node._compute_lacp_partition(shard)} == {"A"}


@pytest.mark.asyncio
async def test_only_the_master_acts_and_no_hint_no_change(monkeypatch):
    node = _wired_node(monkeypatch)
    assert await node._actuate_autoscale([], now=0.0) is None
    node.election_manager.is_master = lambda: False
    assert await node._actuate_autoscale([{"action": "promote", "model": "m"}], now=0.0) is None
    assert getattr(node, "_autoscale_objective", None) is None


@pytest.mark.asyncio
async def test_demand_loop_acts_only_in_on_mode(monkeypatch):
    node = _wired_node(monkeypatch)
    from synapse.inference.shard import Shard
    node.inference_engine.shard = Shard("m", 0, 39, 40)
    node._compute_lacp_partition = lambda *a: [__import__("synapse.topology.partitioning_strategy",
                                                          fromlist=["Partition"]).Partition("A", 0.0, 1.0)]
    # B unserved-for, one server, hot: evaluate() promotes the fitting standby
    node._peer_demand = {"A": {"node_id": "A", "models": {"m": {"rpm": 50.0, "queue_depth": 0}}}}
    monkeypatch.setattr("synapse.topology.capability.aggregate_demand",
                        lambda docs: {"m": {"rpm": 50.0, "reporters": 1, "queue_depth": 0}})
    monkeypatch.setenv("SYNAPSE_AUTOSCALE", "advise")
    await node._log_demand_summary()
    assert node._last_autoscale_hints and getattr(node, "_autoscale_objective", None) is None
    monkeypatch.setenv("SYNAPSE_AUTOSCALE", "on")
    await node._log_demand_summary()
    assert node._autoscale_objective == "throughput"
