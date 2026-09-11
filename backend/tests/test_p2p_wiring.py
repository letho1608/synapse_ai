"""
Behavioral tests: P2P orphan wiring fixed (C-W1/W4/W5).

Drives the REAL Node handlers with a minimal harness (no full Node, no
sockets): ack answers, example routing, status routing, example-RPC loss.
"""
import asyncio
import types

import numpy as np
import pytest

from synapse.inference.shard import Shard
from synapse.orchestration.node import Node
from synapse.routing.event_router import Event, EventRouter


def _harness(**extra):
    import types as _t
    ns = _t.SimpleNamespace(id="harness", topology_viz=None)
    # Bind the REAL Node helpers under test (not fakes): failures then mean
    # product bugs, not harness gaps.
    ns._decode_p2p_array = _t.MethodType(Node._decode_p2p_array, ns)
    for k, v in extra.items():
        setattr(ns, k, v)
    if not hasattr(ns, "event_router"):
        ns.event_router = EventRouter()
    return ns


def _event(data, origin="peer-x"):
    import time
    return Event(topic="t", data=data, origin=origin, timestamp=time.time())


@pytest.mark.asyncio
async def test_ack_peer_answers_and_ignores_missing_key():
    router = EventRouter()
    got = []
    router.subscribe("ack/1", lambda e: got.append(e.data))
    ns = _harness(event_router=router)
    await Node._ack_peer(ns, {"_ack_topic": "ack/1"})
    assert got == [True]
    await Node._ack_peer(ns, {"nope": 1})  # fire-and-forget, no crash
    await Node._ack_peer(ns, None)


@pytest.mark.asyncio
async def test_on_p2p_example_routes_to_enqueue():
    calls = []

    async def _enqueue(shard, example, target, length, train=False, request_id=None):
        calls.append((shard, example, target, length, train, request_id))

    ns = _harness(enqueue_example=_enqueue)
    ex = np.arange(6, dtype=np.int64)
    tgt = np.arange(6, dtype=np.int64)
    ln = np.array([6], dtype=np.int64)
    shard = Shard("m", 0, 1, 4)
    await Node.on_p2p_example(ns, _event({
        "shard": shard.to_dict(),
        "example": ex.tobytes(), "example_shape": [6], "example_dtype": "int64",
        "target": tgt.tobytes(), "target_shape": [6], "target_dtype": "int64",
        "length": ln.tobytes(), "length_shape": [1], "length_dtype": "int64",
        "train": False, "request_id": "r1",
    }))
    assert len(calls) == 1
    s, e, t, l, train, rid = calls[0]
    assert (s.start_layer, s.end_layer) == (0, 1)
    assert np.array_equal(e, ex) and np.array_equal(t, tgt) and np.array_equal(l, ln)
    assert train is False and rid == "r1"


@pytest.mark.asyncio
async def test_on_p2p_example_bad_payload_does_not_raise():
    async def _boom(*a, **k):
        raise RuntimeError("should not be called")
    ns = _harness(enqueue_example=_boom)
    await Node.on_p2p_example(ns, _event({"shard": None}))
    await Node.on_p2p_example(ns, _event("not-a-dict"))


@pytest.mark.asyncio
async def test_on_p2p_status_direct_routes_to_on_node_status():
    import types as _t
    seen = []
    ns = _harness()
    ns.on_node_status = _t.MethodType(
        lambda self, rid, data: seen.append((rid, data)), ns)
    await Node.on_p2p_status_direct(ns, _event(
        {"request_id": "r9", "status": "mid_phase_2"}, origin="node-z"))
    assert seen == [("r9", {"type": "node_status", "node_id": "node-z",
                            "status": "mid_phase_2"})]


@pytest.mark.asyncio
async def test_on_p2p_example_rpc_replies_real_loss():
    async def _enqueue(shard, example, target, length, train=False, request_id=None):
        assert train is True
        return 1.25

    router = EventRouter()
    ns = _harness(event_router=router, enqueue_example=_enqueue)
    replies = []
    router.subscribe("resp/1", lambda e: replies.append(e.data))
    ex = np.arange(4, dtype=np.int64)
    shard = Shard("m", 0, 1, 4)
    await Node.on_p2p_example_rpc(ns, _event({
        "shard": shard.to_dict(),
        "example": ex.tobytes(), "example_shape": [4], "example_dtype": "int64",
        "target": ex.tobytes(), "target_shape": [4], "target_dtype": "int64",
        "length": np.array([4]).tobytes(), "length_shape": [1], "length_dtype": "int64",
        "train": True, "request_id": "r2",
        "_response_topic": "resp/1",
    }))
    assert replies and replies[0]["loss"] == 1.25
