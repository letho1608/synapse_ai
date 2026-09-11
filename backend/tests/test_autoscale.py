"""
Behavioral tests: autoscale evaluator, advisory-only (T3.3 / C7, P4).

- Unserved model with fit standby -> promote.
- Unknown fit (None) never promotes, even unserved.
- Hot model (rpm/server >= 10 + >= 3x median) -> promote.
- Queue backstop promotes without rate signal.
- Idle multi-server with reporters -> demote; no reporters -> never demote.
- Malformed inputs -> [] (never raises).
"""
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
