"""
Minimal test: P2P Bridge RPC health check between two nodes on same machine.
"""

import asyncio
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode

async def main():
    # Node A
    router_a = EventRouter()
    libp2p_a = Libp2pNode("node-a", 5991, router_a)

    # Node B
    router_b = EventRouter()
    libp2p_b = Libp2pNode("node-b", 5992, router_b)

    # Start both
    await router_a.start()
    await router_b.start()
    await libp2p_a.start()
    await libp2p_b.start()

    print("Both nodes started")
    await asyncio.sleep(0.5)  # Let servers bind

    # Node A connects to Node B
    ok = await libp2p_a.connect_to_peer("node-b", "127.0.0.1:5992")
    print(f"Node-A connected to Node-B: {ok}")

    await asyncio.sleep(0.5)  # Let connection settle

    # Now do a manual RPC: Node A sends health check request to Node B
    call_id = "test-call-1"
    request_topic = "synapse/rpc/health/request/node-b"
    response_topic = f"synapse/rpc/health/response/{call_id}"

    future = asyncio.get_event_loop().create_future()

    def on_response(event):
        print(f"[Node-A] RPC response received: {event.data}")
        if not future.done():
            future.set_result(event.data)

    # Node B subscribes to health requests
    async def handle_health(event):
        print(f"[Node-B] Health request received: {event.data}")
        response_to = event.data.get("_response_topic")
        if response_to:
            print(f"[Node-B] Publishing response to {response_to}")
            await router_b.publish(response_to, {"status": "ok"}, origin="node-b")

    router_b.subscribe("synapse/rpc/health/request/node-b", handle_health)

    # Node A subscribes to response and sends request
    router_a.subscribe(response_topic, on_response)

    print("Sending health request...")
    await router_a.publish(request_topic, {
        "_response_topic": response_topic,
        "test": True
    }, origin="local")

    print("Waiting for response (15s timeout)...")
    try:
        result = await asyncio.wait_for(future, timeout=15.0)
        print(f"PASS: Got response: {result}")
    except asyncio.TimeoutError:
        print("FAIL: RPC timed out!")

    # Cleanup
    await libp2p_a.stop()
    await libp2p_b.stop()
    await router_a.stop()
    await router_b.stop()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(main())