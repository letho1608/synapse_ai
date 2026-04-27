"""
In-process P2P cluster test to debug RPC health check failure.
Both nodes run in the SAME process, sharing no state.
"""

import asyncio
import sys
import os
import json
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["DEBUG"] = "1"
os.environ["GRPC_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.networking.p2p_peer_handle import P2PPeerHandle

DC_STRONG = DeviceCapabilities(
    model="NODE-A", chip="RTX 4090", memory=24576,
    flops={"fp32": 82.0, "fp16": 165.0, "int8": 330.0},
    cpu_cores=16, system_ram_mb=64000, gpu_count=1, total_gpu_vram_mb=24576
)
DC_WEAK = DeviceCapabilities(
    model="NODE-B", chip="Core i3", memory=8192,
    flops={"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
    cpu_cores=4, system_ram_mb=8192, gpu_count=0, total_gpu_vram_mb=0
)

async def setup_health_handler(node_name: str, router: EventRouter):
    """Set up the health RPC handler as the Node class would."""
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

async def main():
    # Create components for Node-A
    router_a = EventRouter()
    libp2p_a = Libp2pNode("NODE-MASTER-MOCK", 5678, router_a)

    # Create components for Node-B
    router_b = EventRouter()
    libp2p_b = Libp2pNode("NODE-WORKER-MOCK", 5679, router_b)

    # Start routers and P2P nodes
    await router_a.start()
    await router_b.start()
    await libp2p_a.start()
    await libp2p_b.start()

    # Set up health handlers (simulating _setup_p2p_rpc_handlers)
    await setup_health_handler("NODE-MASTER-MOCK", router_a)
    await setup_health_handler("NODE-WORKER-MOCK", router_b)

    await asyncio.sleep(0.5)  # Let TCP servers bind

    # Create peer handles
    peer_b = P2PPeerHandle("NODE-WORKER-MOCK", "127.0.0.1:5679", "MAN", DC_WEAK, router_a, libp2p_a)
    peer_a = P2PPeerHandle("NODE-MASTER-MOCK", "127.0.0.1:5678", "MAN", DC_STRONG, router_b, libp2p_b)

    print("\n=== Step 1: Node-A health check Node-B ===")
    result_a = await peer_b.health_check()  # peer_b is on router_a, checking NODE-WORKER-MOCK
    print(f"Node-A health check result: {result_a}")

    print("\n=== Step 2: Node-B health check Node-A ===")
    result_b = await peer_a.health_check()  # peer_a is on router_b, checking NODE-MASTER-MOCK
    print(f"Node-B health check result: {result_b}")

    print(f"\n=== Final Results ===")
    print(f"Node-A → Node-B: {'PASS' if result_a else 'FAIL'}")
    print(f"Node-B → Node-A: {'PASS' if result_b else 'FAIL'}")

    # Cleanup
    await libp2p_a.stop()
    await libp2p_b.stop()
    await router_a.stop()
    await router_b.stop()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(main())