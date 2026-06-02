"""
Simultaneous P2P connection test - simulates both nodes connecting concurrently.
This is the scenario that triggers the race condition in _handle_incoming_connection
vs connect_to_peer when both nodes try to connect at the same time.
"""

import asyncio
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.networking.p2p_peer_handle import P2PPeerHandle

DC_A = DeviceCapabilities(
    model="NODE-A", chip="RTX 4090", memory=24576,
    flops={"fp32": 82.0, "fp16": 165.0, "int8": 330.0},
    cpu_cores=16, system_ram_mb=64000, gpu_count=1, total_gpu_vram_mb=24576
)
DC_B = DeviceCapabilities(
    model="NODE-B", chip="Core i3", memory=8192,
    flops={"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
    cpu_cores=4, system_ram_mb=8192, gpu_count=0, total_gpu_vram_mb=0
)

NODE_A = "NODE-MASTER-MOCK"
NODE_B = "NODE-WORKER-MOCK"

async def setup_health_handler(node_name: str, router: EventRouter):
    async def handle_health(event):
        print(f"[{node_name}] Health request received! data={event.data}")
        response_topic = event.data.get("_response_topic")
        if response_topic:
            print(f"[{node_name}] Publishing response to {response_topic}")
            await router.publish(response_topic, {"status": "ok"}, origin=node_name)
            print(f"[{node_name}] Response published")

    topic = f"synapse/rpc/health/request/{node_name}"
    router.subscribe(topic, handle_health)
    print(f"[{node_name}] Subscribed to {topic}")

async def test_simultaneous_connect():
    """Test where BOTH nodes call connect_to_peer at the SAME time using asyncio.gather."""
    router_a = EventRouter()
    libp2p_a = Libp2pNode(NODE_A, 5778, router_a)

    router_b = EventRouter()
    libp2p_b = Libp2pNode(NODE_B, 5779, router_b)

    await router_a.start()
    await router_b.start()
    await libp2p_a.start()
    await libp2p_b.start()

    await setup_health_handler(NODE_A, router_a)
    await setup_health_handler(NODE_B, router_b)

    await asyncio.sleep(0.5)

    # SIMULTANEOUS connect - this is the race condition trigger
    print("\n=== SIMULTANEOUS connect: both nodes connect at the same time ===")
    results = await asyncio.gather(
        libp2p_a.connect_to_peer(NODE_B, "127.0.0.1:5779"),
        libp2p_b.connect_to_peer(NODE_A, "127.0.0.1:5778"),
        return_exceptions=True
    )
    print(f"Simultaneous connect results: A→B={results[0]}, B→A={results[1]}")

    await asyncio.sleep(0.5)

    print(f"Node-A connections: {list(libp2p_a._bridge._connections.keys())}")
    print(f"Node-B connections: {list(libp2p_b._bridge._connections.keys())}")

    # Verify both sides see the connection
    a_connected = libp2p_a.is_peer_connected(NODE_B)
    b_connected = libp2p_b.is_peer_connected(NODE_A)
    print(f"A sees B: {a_connected}, B sees A: {b_connected}")

    if not a_connected or not b_connected:
        print("FAIL: Nodes are not connected after simultaneous connect")
        await libp2p_a.stop()
        await libp2p_b.stop()
        return False

    # Create peer handles
    peer_b = P2PPeerHandle(NODE_B, "127.0.0.1:5779", "MAN", DC_B, router_a, libp2p_a)
    peer_a = P2PPeerHandle(NODE_A, "127.0.0.1:5778", "MAN", DC_A, router_b, libp2p_b)

    # Bidirectional health checks
    print("\n=== Bi-directional health checks after simultaneous connect ===")

    # Do BOTH health checks concurrently too
    hc_results = await asyncio.gather(
        peer_b.health_check(),
        peer_a.health_check(),
        return_exceptions=True
    )
    print(f"Health check results: A→B={hc_results[0]}, B→A={hc_results[1]}")

    all_pass = all(r is True for r in hc_results)

    # Do additional rounds to stress test
    for round_num in range(3):
        print(f"\n=== Stress round {round_num + 1} ===")
        r = await asyncio.gather(
            peer_b.health_check(),
            peer_a.health_check(),
            return_exceptions=True
        )
        print(f"Round {round_num + 1}: A→B={r[0]}, B→A={r[1]}")
        if not all(r is True for r in r):
            all_pass = False

    print(f"\n=== Final Result: {'PASS' if all_pass else 'FAIL'} ===")

    await libp2p_a.stop()
    await libp2p_b.stop()
    await router_a.stop()
    await router_b.stop()

    return all_pass

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    success = asyncio.run(test_simultaneous_connect())
    sys.exit(0 if success else 1)