"""
Layer placement from MEASURED costs (EdgeShard-style dynamic programming).

Each node reports how long one decoder layer takes on it (ms_per_layer, from
real forwards or a bandwidth probe) and how many layers fit in its memory
(max_layers). Links cost their measured one-way hop time. For a chain order
of nodes, the DP chooses which nodes to use (device selection) and where to
cut, exactly:

- objective "throughput": minimize the slowest stage, stage = its layers'
  compute + the hop that feeds it (pipeline rate with concurrent requests);
- objective "latency": minimize one token's end-to-end time, i.e. all
  compute + every hop (+ the hop back to the first node).

Orders: every permutation for up to EXHAUSTIVE_MAX nodes, otherwise a
nearest-neighbour chain from the fastest node. This replaces weighting
VRAM/TFLOPS/CPU/... scores with fixed coefficients: the objective is in
milliseconds, built from what was measured.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

EXHAUSTIVE_MAX = 4
DEFAULT_HOP_MS = 2.0  # unmeasured LAN hop


@dataclass
class NodeProfile:
    node_id: str
    ms_per_layer: float
    max_layers: int


@dataclass
class Placement:
    stages: List[Tuple[str, int, int]]  # (node_id, start, end) in order
    cost_ms: float
    objective: str
    order: List[str]


def _hop(rtt: Dict[Tuple[str, str], float], a: str, b: str) -> float:
    if a == b:
        return 0.0
    v = rtt.get((a, b), rtt.get((b, a)))
    return float(v) if v is not None and v >= 0 else DEFAULT_HOP_MS


def _dp_for_order(order: Sequence[NodeProfile], total: int,
                  rtt: Dict[Tuple[str, str], float], objective: str,
                  pin_first: bool = False):
    """Best placement for a fixed chain order, nodes optional (skippable).

    best[i][l] = best cost with layers [0, l) placed on a subsequence of
    order[:i+1] whose last stage is order[i]. The choice of the stage size
    k is vectorized."""
    import numpy as np
    m = len(order)
    INF = math.inf
    best = np.full((m, total + 1), INF)
    back_j = np.full((m, total + 1), -1, dtype=np.int64)
    back_l = np.zeros((m, total + 1), dtype=np.int64)
    latency = objective == "latency"
    for i, node in enumerate(order):
        cap = min(node.max_layers, total)
        if cap <= 0:
            continue
        ks = np.arange(1, cap + 1)
        comp = ks * node.ms_per_layer
        if i == 0 or not pin_first:
            best[i, 1:cap + 1] = comp  # first stage: layers [0, k)
        for j in range(i):
            hop = _hop(rtt, order[j].node_id, node.node_id)
            for l in np.nonzero(np.isfinite(best[j, 1:total]))[0] + 1:
                kmax = min(cap, total - l)
                stage = comp[:kmax] + hop
                c = best[j, l] + stage if latency else np.maximum(best[j, l], stage)
                seg = best[i, l + 1:l + kmax + 1]
                better = c < seg
                if better.any():
                    seg[better] = c[better]
                    back_j[i, l + 1:l + kmax + 1][better] = j
                    back_l[i, l + 1:l + kmax + 1][better] = l
    result = None
    for i in range(m):
        c = best[i, total]
        if not np.isfinite(c):
            continue
        if latency:
            first = _first_of(back_j, back_l, i, total)
            c = c + _hop(rtt, order[i].node_id, order[first].node_id)
        if result is None or c < result[0]:
            result = (float(c), i)
    if result is None:
        return None
    cost, i = result
    stages = []
    l = total
    while True:
        j = int(back_j[i, l])
        start = int(back_l[i, l]) if j >= 0 else 0
        stages.append((order[i].node_id, start, l - 1))
        if j < 0:
            break
        i, l = j, start
    stages.reverse()
    return cost, stages


def _first_of(back_j, back_l, i, l):
    while back_j[i, l] >= 0:
        i, l = int(back_j[i, l]), int(back_l[i, l])
    return i


def _nearest_neighbour_order(profiles: List[NodeProfile],
                             rtt: Dict[Tuple[str, str], float]) -> List[NodeProfile]:
    rest = sorted(profiles, key=lambda p: p.ms_per_layer)
    order = [rest.pop(0)]
    while rest:
        last = order[-1].node_id
        nxt = min(rest, key=lambda p: (_hop(rtt, last, p.node_id), p.ms_per_layer))
        rest.remove(nxt)
        order.append(nxt)
    return order


def plan(profiles: List[NodeProfile], total_layers: int,
         rtt_ms: Optional[Dict[Tuple[str, str], float]] = None,
         objective: str = "throughput", first_node: Optional[str] = None) -> Optional[Placement]:
    """Optimal contiguous placement; None when the model cannot fit.

    first_node pins the first stage (e.g. the node that holds the
    embedding and receives the prompt)."""
    if objective not in ("throughput", "latency"):
        raise ValueError(f"unknown objective {objective!r}")
    usable = [p for p in profiles if p.max_layers > 0 and p.ms_per_layer > 0]
    if not usable or total_layers <= 0:
        return None
    rtt = dict(rtt_ms or {})
    if len(usable) <= EXHAUSTIVE_MAX:
        orders = itertools.permutations(usable)
    else:
        orders = [_nearest_neighbour_order(usable, rtt)]
    best: Optional[Placement] = None
    for order in orders:
        order = list(order)
        if first_node is not None and order[0].node_id != first_node:
            continue
        found = _dp_for_order(order, total_layers, rtt, objective,
                              pin_first=first_node is not None)
        if found is None:
            continue
        cost, stages = found
        if first_node is not None and stages[0][0] != first_node:
            continue
        if best is None or cost < best.cost_ms - 1e-9:
            best = Placement(stages=stages, cost_ms=cost, objective=objective,
                             order=[p.node_id for p in order])
    return best
