"""
Master-side autoscale evaluator (T3.3 / C7, P4).

Pure function — no node, no network — fully testable. Thresholds mirror
mesh-llm's standby-promotion triggers (DESIGN.md Demand-Aware Rebalancing):
unserved model / >=3x imbalance + >=10 rpm per server / hot single model.
mesh-llm's proxy request counters become synapse's gossip demand aggregate
(topology/capability.aggregate_demand); queue backstop uses gossip
queue_depth instead of proxy counters.

evaluate() only returns hints (mesh-llm /api/model-targets semantics);
the master acts on them in Node._actuate_autoscale (placement objective,
with a cooldown). Unknown fit (None) never promotes; unknown demand never
demotes.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

HOT_RPM_PER_SERVER = 10.0
IMBALANCE_RATIO = 3.0
DEMOTE_RPM_PER_SERVER = 1.0
# Queue backstop: sustained backlog >= this many requests with no free server.
QUEUE_BACKSTOP = 32


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def evaluate(
    demand,
    servers,
    standby,
    fit_fn: Optional[Callable[[str, str], object]] = None,
    hot_rpm: float = HOT_RPM_PER_SERVER,
    imbalance: float = IMBALANCE_RATIO,
    demote_rpm: float = DEMOTE_RPM_PER_SERVER,
) -> List[Dict[str, Any]]:
    """Return advisory decisions: [{action, model, node?, reason}].

    demand: aggregate_demand() output {model: {rpm, reporters, unknown,
      queue_depth, in_flight}}. servers: {model: [node_ids]} currently
    serving. standby: node ids available for promotion. fit_fn(node, model)
    -> True/False/None (None = unknown = never promote on it).
    """
    decisions: List[Dict[str, Any]] = []
    try:
        demand = demand if isinstance(demand, dict) else {}
        servers = servers if isinstance(servers, dict) else {}
        standby = standby if isinstance(standby, list) else []
        fit = fit_fn or (lambda _n, _m: None)
        per_server = {}
        for model, agg in demand.items():
            if not isinstance(agg, dict):
                continue
            n = len(servers.get(model, []))
            rpm = agg.get("rpm", 0.0) or 0.0
            per_server[model] = rpm / max(n, 1)
        median_load = _median(list(per_server.values()))

        for model, agg in demand.items():
            if not isinstance(agg, dict):
                continue
            serving = servers.get(model, [])
            if not isinstance(serving, list):
                serving = []
            serving = [s for s in serving if isinstance(s, str)]
            rpm = agg.get("rpm", 0.0) or 0.0
            load = per_server.get(model, 0.0)

            cands = [s for s in standby if fit(s, model) is True]
            if not serving and cands:
                decisions.append({"action": "promote", "model": model,
                                  "node": cands[0],
                                  "reason": "unserved-model"})
                continue
            if serving and cands:
                has_rates = agg.get("reporters", 0) > 0
                hot = has_rates and load >= hot_rpm and load >= imbalance * median_load
                single_hot = has_rates and len(serving) == 1 and rpm >= hot_rpm
                # Queue backstop needs no rate signal: a visible backlog with
                # a fit standby is evidence by itself.
                queued = agg.get("queue_depth", 0) >= QUEUE_BACKSTOP
                if hot or single_hot or queued:
                    decisions.append({"action": "promote", "model": model,
                                      "node": cands[0],
                                      "reason": ("hot-model" if (hot or single_hot)
                                                 else "queue-backstop")})
            if len(serving) > 1 and agg.get("reporters", 0) > 0:
                if load < demote_rpm:
                    decisions.append({"action": "demote", "model": model,
                                      "node": serving[-1],
                                      "reason": "idle-servers"})

        return decisions
    except Exception as e:
        # Lane B: "no action" must be distinguishable from "crashed".
        try:
            from synapse.helpers import DEBUG
            if DEBUG >= 1:
                print(f"[AUTOSCALE] evaluator crashed, returning no-action: {e}")
        except Exception:
            pass
        return []
