"""
Behavioral tests: the wiring that makes multi-process inference REALLY split.

Found 2026-09-29 by running 2-3 real backend processes (manual discovery,
separate ports) and reading which layers each one executed:
- every node timed ITSELF out (own heartbeats are never delivered back) and
  rebalanced itself out of the pipeline every few seconds;
- receivers ignored the shard in the message and recomputed "their" shard
  from a local partition view; with equal nodes the tie order differed per
  node, so both ran layers 0..k and bounced hidden states back and forth;
- the last stage sampled at the node default temperature (the request's
  temperature never reached it);
- peers registered each other's BRIDGE port as the tensor-transport address,
  and inbound TCP frames sat in one "__inbound__" reorder bucket waiting for
  micro_batch_id 1, 2, ... so every decode step after the first was dropped;
- SHM was chosen for same-host peers in other processes (in-process only);
- topology broadcast failed on every attempt (pydantic DeviceFlops not JSON).
"""
import asyncio
import json
import time

import numpy as np
import pytest

from synapse.inference.shard import Shard


def test_node_never_times_itself_out():
    from synapse.pipeline.fault_tolerance import FaultToleranceManager
    ft = FaultToleranceManager("self-node")
    ft.register_node("self-node")
    ft.register_node("peer")
    for h in ft._health.values():
        h.last_heartbeat = time.time() - 60
        h.check_timeout(ft.HEARTBEAT_TIMEOUT)
    assert ft.get_dead_nodes() == ["peer"]

    async def one_tick():
        ft._running = True
        ft.HEARTBEAT_INTERVAL = 0.01
        task = asyncio.create_task(ft._monitor_loop())
        await asyncio.sleep(0.05)
        ft._running = False
        await task
    asyncio.run(one_tick())
    assert ft._health["self-node"].is_alive
    assert ft._pending_rebalance != "self-node"


def test_next_stage_comes_from_request_plan():
    from synapse.orchestration.node import Node
    plan = [["a", 0, 9, 30], ["b", 10, 19, 30], ["c", 20, 29, 30]]
    state = {"partition_plan": plan}
    nid, sh = Node._next_stage_from_plan(None, Shard("m", 0, 9, 30), state)
    assert nid == "b" and (sh.start_layer, sh.end_layer, sh.n_layers) == (10, 19, 30)
    nid, sh = Node._next_stage_from_plan(None, Shard("m", 10, 19, 30), state)
    assert nid == "c" and sh.start_layer == 20
    assert Node._next_stage_from_plan(None, Shard("m", 0, 9, 30), {}) is None


@pytest.mark.asyncio
async def test_controller_sends_plan_and_temperature_with_every_stage_input():
    from synapse.inference.distributed_engine import DistributedPipelineController
    from synapse.transport.codec import decode_tensor

    sent = []

    class Router:
        node_id = "orch"

        async def publish(self, topic, data, origin="local"):
            sent.append((topic, data))

        def subscribe(self, *a, **k):
            pass

        def unsubscribe(self, *a, **k):
            pass

    ctl = DistributedPipelineController("orch", inference_engine=None, event_router=Router())
    ctl._plans["r1"] = [["orch", 0, 14, 30], ["peer", 15, 29, 30]]
    ctl._sampling["r1"] = {"temperature": 0.0}
    await ctl._send_input_to_stage("orch", Shard("m", 0, 14, 30),
                                   np.array([[1, 2, 3]], dtype=np.int64), "r1")
    _, meta = decode_tensor(sent[0][1]["bin"])
    st = meta["inference_state"]
    assert st["partition_plan"] == [["orch", 0, 14, 30], ["peer", 15, 29, 30]]
    assert st["temperature"] == 0.0


@pytest.mark.asyncio
async def test_tcp_inbound_delivers_repeated_micro_batch_zero():
    from synapse.transport.codec import encode_tensor
    from synapse.transport.tcp_transport import TensorTCPTransport
    srv = TensorTCPTransport("srv", port=59431)
    cli = TensorTCPTransport("cli")
    got = []
    srv.set_tensor_handler(got.append)
    await srv.start()
    try:
        for step in range(3):  # every decode step reuses micro_batch_id 0
            blob = encode_tensor(np.full((1, 2, 4), step, np.float32),
                                 metadata={"request_id": "r", "micro_batch_id": 0})
            assert await cli.send_blob_chunked("127.0.0.1:59431", blob)
        for _ in range(50):
            if len(got) == 3:
                break
            await asyncio.sleep(0.05)
        assert [int(m.tensor[0, 0, 0]) for m in got] == [0, 1, 2]
    finally:
        await cli.stop()
        await srv.stop()


def test_handshake_advertises_tensor_port_and_serializes_pydantic():
    from synapse.routing.p2p_socket_bridge import P2PSocketBridge
    from synapse.topology.device_capabilities import DeviceFlops
    br = P2PSocketBridge.__new__(P2PSocketBridge)
    br._peer_handshakes = {}
    br.record_peer_handshake("p", {"node_id": "p", "tcp_port": 53765})
    assert br._peer_handshakes["p"]["tcp_port"] == 53765
    payload = json.dumps({"flops": DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0)},
                         default=P2PSocketBridge._json_serialize)
    assert json.loads(payload)["flops"]["fp16"] == 2.0


def test_peer_tensor_address_uses_advertised_port():
    from types import SimpleNamespace
    from synapse.orchestration.node import Node
    bridge = SimpleNamespace(_peer_handshakes={"b": {"tcp_port": 53765}, "old": {}})
    fake = SimpleNamespace(libp2p_node=SimpleNamespace(_bridge=bridge))
    assert Node._peer_tcp_addr(fake, "b", "10.0.0.2:51062") == "10.0.0.2:53765"
    assert Node._peer_tcp_addr(fake, "old", "10.0.0.2:51062") is None  # bridge only


def test_shm_not_chosen_for_peer_in_another_process():
    from synapse.transport.shm_transport import SharedMemoryTransport
    t = SharedMemoryTransport(node_id="shm-a-test")
    t.register_peer("127.0.0.1:51062", "not-in-this-process")
    assert t.is_local_peer("127.0.0.1:51062")
    assert not t.can_deliver("127.0.0.1:51062")


@pytest.mark.asyncio
async def test_event_handler_failures_are_logged_not_swallowed():
    from loguru import logger
    from synapse.routing.event_router import EventRouter
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="ERROR")
    try:
        er = EventRouter()

        async def boom(event):
            raise RuntimeError("handler exploded")
        er.subscribe("synapse/x", boom)
        await er.publish("synapse/x", {})
    finally:
        logger.remove(sink)
    assert any("synapse/x" in l and "handler exploded" in l for l in lines)
