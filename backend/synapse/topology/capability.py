"""
Capability + demand gossip over the existing event mesh (T3.3 / C7, P3).

Additive, version-tolerant JSON — old peers ignore these topics entirely:
- synapse/cluster/capabilities: {node_id, model_ids, vram_free_mb,
  tflops_fp16, quant_ok, seq_max, version, ts}. Shard-fit inputs.
- synapse/cluster/demand: {node_id, model_id, req_per_min (None = unknown,
  never 0-as-unknown), queue_depth, in_flight, ts}. Autoscale inputs.

Tri-state honesty: shard_fits() returns True / False / None (unknown).
A None gate never excludes a node; it only marks the decision unmeasured.
Phase 1 is advisory-only (hints for status/topology --json); nothing here
actuates partition or autoscale by itself.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

CAPABILITIES_TOPIC = "synapse/cluster/capabilities"
DEMAND_TOPIC = "synapse/cluster/demand"

# mesh-llm solo-fit headroom borrowed as a SOFT gate (never exclusion).
VRAM_HEADROOM = 1.1
MAX_STORE = 256  # cap per peer-store dict (hygiene)


def _caps_dict(caps) -> Dict[str, Any]:
    if caps is None:
        return {}
    if isinstance(caps, dict):
        return caps
    to_dict = getattr(caps, "to_dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def build_capability_doc(
    node_id: str,
    caps=None,
    model_ids=(),
    vram_free_mb: Optional[float] = None,
    tflops_fp16: Optional[float] = None,
    quant_ok=(),
    seq_max: Optional[int] = None,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build our capability advertisement. Unknowns stay None (never 0)."""
    base = _caps_dict(caps)
    if tflops_fp16 is None:
        flops = base.get("flops") or {}
        if isinstance(flops, dict):
            tflops_fp16 = flops.get("fp16")
    doc = {
        "node_id": node_id,
        "model_ids": list(model_ids or []),
        "vram_free_mb": vram_free_mb,
        "tflops_fp16": tflops_fp16,
        "quant_ok": list(quant_ok or []),
        "seq_max": seq_max,
        "chip": base.get("chip"),
        "gpu_backend": base.get("gpu_backend"),
        "version": version,
        "ts": time.time(),
    }
    return doc


def build_demand_doc(
    node_id: str,
    model_id: str = "",
    req_per_min: Optional[float] = None,
    queue_depth: int = 0,
    in_flight: int = 0,
) -> Dict[str, Any]:
    """Build our demand report. req_per_min None means unknown (no rate
    counter on this node) — consumers must exclude it, never treat as 0."""
    return {
        "node_id": node_id,
        "model_id": model_id or "",
        "req_per_min": req_per_min,
        "queue_depth": int(queue_depth),
        "in_flight": int(in_flight),
        "ts": time.time(),
    }


def shard_fits(cap_doc: Dict[str, Any], layer_bytes,
               num_layers, headroom: float = VRAM_HEADROOM):
    """Soft fit-gate: True / False / None (unknown inputs)."""
    try:
        free = cap_doc.get("vram_free_mb")
        if free is None or layer_bytes is None or num_layers is None:
            return None
        need_mb = float(layer_bytes) * int(num_layers) / (1024 * 1024)
        return float(free) >= need_mb * float(headroom)
    except (TypeError, ValueError):
        return None


def aggregate_demand(docs: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Sum known rates per model. Unknown rates excluded, counted separately."""
    out: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        if not isinstance(d, dict):
            continue
        model = d.get("model_id", "")
        slot = out.setdefault(model, {"rpm": 0.0, "reporters": 0, "unknown": 0,
                                      "queue_depth": 0, "in_flight": 0})
        rpm = d.get("req_per_min")
        if isinstance(rpm, (int, float)):
            slot["rpm"] += float(rpm)
            slot["reporters"] += 1
        else:
            slot["unknown"] += 1
        try:
            slot["queue_depth"] += int(d.get("queue_depth", 0))
            slot["in_flight"] += int(d.get("in_flight", 0))
        except (TypeError, ValueError):
            pass
    return out


def store_bounded(store: Dict[str, Dict[str, Any]], node_id: str,
                   doc: Dict[str, Any]) -> None:
    """Store a peer doc with hygiene cap (evict oldest by ts)."""
    store[node_id] = doc
    while len(store) > MAX_STORE:
        oldest = min(store.items(), key=lambda kv: kv[1].get("ts", 0.0))[0]
        del store[oldest]
