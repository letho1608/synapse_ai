"""
Behavioral tests: capability + demand gossip docs (T3.3 / C7, P3).

Pure-function tests (no node, no network): builders keep unknowns as None
(never 0-as-unknown), aggregation excludes unknown
rates, stores are hygiene-capped.
"""
import time

from synapse.topology.capability import (
    CAPABILITIES_TOPIC,
    DEMAND_TOPIC,
    aggregate_demand,
    build_capability_doc,
    build_demand_doc,
    store_bounded,
)


def test_capability_doc_unknowns_stay_none():
    doc = build_capability_doc("n1", None, version="0.1.0")
    assert doc["node_id"] == "n1"
    assert doc["vram_free_mb"] is None
    assert doc["tflops_fp16"] is None
    assert doc["model_ids"] == []
    assert "ts" in doc
    assert CAPABILITIES_TOPIC == "synapse/cluster/capabilities"
    assert DEMAND_TOPIC == "synapse/cluster/demand"


def test_capability_doc_reads_caps_dict():
    caps = {"chip": "X", "gpu_backend": "CUDA", "flops": {"fp16": 60.0}}
    doc = build_capability_doc("n1", caps, model_ids=["m"], vram_free_mb=8000.0)
    assert doc["tflops_fp16"] == 60.0
    assert doc["chip"] == "X"
    assert doc["model_ids"] == ["m"]


def test_demand_unknown_rate_excluded_not_zero():
    docs = [
        build_demand_doc("a", "m", req_per_min=12.0, queue_depth=3, in_flight=2),
        build_demand_doc("b", "m"),  # unknown rate
    ]
    agg = aggregate_demand(docs)
    assert agg["m"]["rpm"] == 12.0
    assert agg["m"]["reporters"] == 1
    assert agg["m"]["unknown"] == 1
    assert agg["m"]["queue_depth"] == 3
    assert agg["m"]["in_flight"] == 2


def test_aggregate_ignores_malformed_docs():
    agg = aggregate_demand(["junk", None, 42, {"model_id": "m2"}])
    assert agg["m2"]["rpm"] == 0.0
    assert agg["m2"]["reporters"] == 0
    assert agg["m2"]["unknown"] == 1


def test_store_bounded_evicts_oldest():
    store = {}
    for i in range(300):
        store_bounded(store, "n%d" % i, {"node_id": "n%d" % i, "ts": float(i)})
    assert len(store) == 256
    assert "n0" not in store
    assert "n299" in store
