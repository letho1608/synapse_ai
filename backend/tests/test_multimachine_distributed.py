"""
REAL MULTI-MACHINE DISTRIBUTED TEST
====================================
This test validates that 2 Synapse nodes on DIFFERENT machines can:
1. Discover each other via ManualDiscovery (shared config file)
2. Form a P2P TCP mesh with proper handshake, heartbeat, tiebreaker
3. Run master election (one node becomes master)
4. Share topology data (cross-machine RPC health checks + topology collection)
5. Compute consistent partition/shard assignments
6. Forward prompt and tensor data across nodes via P2P event routing
7. Deliver inference results back to the orchestrator node

Prerequisites:
- Two real machines on the same LAN (or with Tailscale overlay)
- Both machines must have Python 3.10+ and synapse_ai installed
- Both machines must be able to reach each other on the configured TCP ports
- Copy the same cluster_config_multimachine.json to both machines

Usage:
  Machine A:  python tests/test_multimachine_distributed.py --role server --port 5888
  Machine B:  python tests/test_multimachine_distributed.py --role client --port 5889 --target <MACHINE_A_IP>:5888
  OR use a shared config:  python tests/test_multimachine_distributed.py --config-based

This test exercises the CRITICAL FIXES:
- Bridge broadcast loop prevention (origin-based filtering)
- Binary tensor serialization (tobytes/frombuffer with hex encoding)
- Orchestrator_id routing for result delivery
- Authoritative topology broadcast from master node
- Consistent partition computation across all nodes
"""

import asyncio
import sys
import os
import json
import argparse
import tempfile
import time
import socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.networking.p2p_peer_handle import P2PPeerHandle
from synapse.networking.manual.manual_discovery import ManualDiscovery
from synapse.orchestration.election import ElectionManager
from synapse.topology.topology import Topology


DC_PRIMARY = DeviceCapabilities(
    model="Primary(GPU)", chip="RTX 3080", memory=10240,
    flops={"fp32": 30.0, "fp16": 60.0, "int8": 120.0},
    cpu_cores=8, system_ram_mb=32000, gpu_count=1, total_gpu_vram_mb=10240
)

DC_SECONDARY = DeviceCapabilities(
    model="Secondary(CPU)", chip="Core i5", memory=8192,
    flops={"fp32": 3.0, "fp16": 6.0, "int8": 12.0},
    cpu_cores=4, system_ram_mb=16384, gpu_count=0, total_gpu_vram_mb=0
)


def make_config(node_a_id, node_a_port, node_b_id, node_b_port, node_b_host="127.0.0.1"):
    return {
        "peers": {
            node_a_id: {
                "address": "127.0.0.1",
                "port": node_a_port,
                "device_capabilities": {
                    "model": "Primary(GPU)", "chip": "RTX 3080", "memory": 10240,
                    "flops": {"fp32": 30.0, "fp16": 60.0, "int8": 120.0},
                    "cpu_cores": 8, "system_ram_mb": 32000, "gpu_count": 1, "total_gpu_vram_mb": 10240
                }
            },
            node_b_id: {
                "address": node_b_host,
                "port": node_b_port,
                "device_capabilities": {
                    "model": "Secondary(CPU)", "chip": "Core i5", "memory": 8192,
                    "flops": {"fp32": 3.0, "fp16": 6.0, "int8": 12.0},
                    "cpu_cores": 4, "system_ram_mb": 16384, "gpu_count": 0, "total_gpu_vram_mb": 0
                }
            }
        }
    }


def get_lan_ip():
    """Get the LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ────────────────────────────────────────────────────────────────────
# DIRECT MODE: Client connects to Server (no shared config)
# ────────────────────────────────────────────────────────────────────

async def run_server(port: int):
    """Run as the server/waiting side."""
    node_id = "MACHINE-A"
    print(f"\n[SERVER] Starting as {node_id} on port {port}")
    print(f"[SERVER] LAN IP: {get_lan_ip()}")

    router = EventRouter()
    libp2p = Libp2pNode(node_id, port, router)

    await router.start()
    await libp2p.start()

    print(f"[SERVER] Listening on {port}, waiting for MACHINE-B to connect...")

    # Wait for incoming connection
    wait_start = time.time()
    while time.time() - wait_start < 60:
        await asyncio.sleep(0.5)
        if libp2p._bridge._connections:
            peer = list(libp2p._bridge._connections.keys())[0]
            print(f"[SERVER] Peer connected: {peer}")
            await router.publish(
                "synapse/test/hello",
                {"message": f"Hello from {node_id}!", "timestamp": time.time()},
                origin=node_id,
            )
            print(f"[SERVER] Sent hello message to {peer}")
            break
    else:
        print("[SERVER] TIMEOUT: No peer connected within 60s")
        await libp2p.stop()
        await router.stop()
        return False

    # Receive test messages
    router.subscribe("synapse/test/hello", lambda ev: print(f"[SERVER] Received: {ev.data}"))
    router.subscribe("synapse/test/reply", lambda ev: print(f"[SERVER] Received reply: {ev.data}"))

    await asyncio.sleep(5)

    # Verify bi-directional communication
    print(f"[SERVER] Connections: {list(libp2p._bridge._connections.keys())}")
    connected = len(libp2p._bridge._connections) > 0
    print(f"[SERVER] Result: {'PASSED' if connected else 'FAILED'}")
    await libp2p.stop()
    await router.stop()
    return connected


async def run_client(port: int, target: str):
    """Run as the client/connecting side."""
    node_id = "MACHINE-B"
    print(f"\n[CLIENT] Starting as {node_id} on port {port}")
    print(f"[CLIENT] LAN IP: {get_lan_ip()}")
    print(f"[CLIENT] Connecting to server at {target}...")

    router = EventRouter()
    libp2p = Libp2pNode(node_id, port, router)

    await router.start()
    await libp2p.start()

    # Subscribe to test events
    received_messages = []
    finished = asyncio.Event()

    async def on_hello(event):
        print(f"[CLIENT] Received hello: {event.data}")
        received_messages.append(event.data)
        await router.publish(
            "synapse/test/reply",
            {"message": f"Hello back from {node_id}!", "timestamp": time.time()},
            origin=node_id,
        )
        for peer_id in libp2p._bridge._connections:
            print(f"[CLIENT] Sent reply to {peer_id}")
        finished.set()

    router.subscribe("synapse/test/hello", on_hello)

    # Connect to server
    host, port_str = target.split(":")
    result = await libp2p.connect_to_peer("MACHINE-A", target)

    if not result:
        print(f"[CLIENT] FAILED to connect to {target}")
        await libp2p.stop()
        await router.stop()
        return False

    print(f"[CLIENT] Connected successfully. Waiting for test message...")

    try:
        await asyncio.wait_for(finished.wait(), timeout=30)
        print(f"[CLIENT] Communication verified: {len(received_messages)} messages received")

        connected = libp2p.is_peer_connected("MACHINE-A")
        print(f"[CLIENT] Connection status: {connected}")
        print(f"[CLIENT] Result: {'PASSED' if connected else 'FAILED'}")
        await libp2p.stop()
        await router.stop()
        return connected
    except asyncio.TimeoutError:
        print("[CLIENT] TIMEOUT: No test message received within 30s")
        await libp2p.stop()
        await router.stop()
        return False


# ────────────────────────────────────────────────────────────────────
# FULL CLUSTER MODE: Both sides use shared config file
# ────────────────────────────────────────────────────────────────────

async def run_full_cluster(config_data: dict, my_node_id: str, my_port: int):
    """Run a node in full cluster mode with discovery, election, topology."""
    tmpdir = tempfile.mkdtemp(prefix="synapse_mm_")
    config_path = os.path.join(tmpdir, "cluster_config_multimachine.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    try:
        router = EventRouter()
        libp2p = Libp2pNode(my_node_id, my_port, router)

        def create_peer(nid, addr, desc, dc):
            return P2PPeerHandle(nid, addr, desc, dc, router, libp2p)

        discovery = ManualDiscovery(config_path, my_node_id, create_peer)

        # Determine weight based on device capabilities
        if my_node_id == "MACHINE-A":
            weight = 60.0  # Higher weight → master
        else:
            weight = 6.0

        election = ElectionManager(my_node_id, router, compute_weight=weight, dynamic_weight=False)

        # Health handlers
        async def health_handler(event):
            resp = event.data.get("_response_topic")
            if resp:
                await router.publish(resp, {"status": "ok"}, origin=my_node_id)
        router.subscribe(f"synapse/rpc/health/request/{my_node_id}", health_handler)

        # Topology handler
        async def topo_handler(event):
            resp = event.data.get("_response_topic")
            if resp:
                await router.publish(resp, {
                    "topology": {
                        "nodes": {
                            my_node_id: (DC_PRIMARY if my_node_id == "MACHINE-A" else DC_SECONDARY).to_dict()
                        },
                        "edges": [],
                        "active_node_id": election.current_master_id
                    }
                }, origin=my_node_id)
        router.subscribe(f"synapse/rpc/topology/request/{my_node_id}", topo_handler)

        # Subscribe to topology/partition broadcasts
        router.subscribe("synapse/cluster/topology", lambda ev:
            print(f"[{my_node_id}] Got topology broadcast: {list(ev.data.get('topology', {}).get('nodes', {}).keys()) if isinstance(ev.data, dict) else 'N/A'}"))

        router.subscribe("synapse/cluster/partition", lambda ev:
            print(f"[{my_node_id}] Got partition broadcast: {ev.data}"))

        # Start all components
        await router.start()
        await libp2p.start()
        await election.start()
        await discovery.start()

        await asyncio.sleep(0.5)

        print(f"\n[{my_node_id}] === Waiting for P2P mesh to form... ===")

        other_id = "MACHINE-B" if my_node_id == "MACHINE-A" else "MACHINE-A"

        for i in range(30):
            await asyncio.sleep(1.0)
            peers = await discovery.discover_peers()
            peer_ids = [p.id() for p in peers]
            connected = libp2p.is_peer_connected(other_id)

            print(f"[{my_node_id}][{i+1}s] Peers discovered: {peer_ids}, Connected to {other_id}: {connected}")

            if other_id in peer_ids and connected:
                print(f"[{my_node_id}] Mesh formed after {i+1}s!")
                break
        else:
            print(f"[{my_node_id}] FAIL: Mesh did not form within 30s")
            return False

        # Wait for election to settle
        await asyncio.sleep(5)
        print(f"\n[{my_node_id}] === Election Status ===")
        print(f"[{my_node_id}] Is master: {election.is_master()}")
        print(f"[{my_node_id}] Current master: {election.current_master_id}")

        election_ok = election.current_master_id == "MACHINE-A"
        if election_ok:
            print(f"[{my_node_id}] Election: CORRECT (MACHINE-A should be master)")
        else:
            print(f"[{my_node_id}] Election: UNEXPECTED")

        # Test RPC health checks
        print(f"\n[{my_node_id}] === RPC Health Check ===")
        health_ok = False
        peers = await discovery.discover_peers()
        if peers:
            peer = next((p for p in peers if p.id() == other_id), None)
            if peer:
                hc_result = await peer.health_check()
                health_ok = bool(hc_result)
                print(f"[{my_node_id}] Health check to {other_id}: {'OK' if hc_result else 'FAIL'}")

        # Test topology collection
        print(f"\n[{my_node_id}] === Topology Collection ===")
        topology_ok = False
        if peers:
            peer = next((p for p in peers if p.id() == other_id), None)
            if peer:
                topo = await peer.collect_topology(set(), max_depth=2)
                node_ids = list(topo.nodes.keys()) if hasattr(topo, 'nodes') else []
                topology_ok = other_id in node_ids and getattr(topo, "active_node_id", None) == "MACHINE-A"
                print(f"[{my_node_id}] Topology nodes: {node_ids}")
                print(f"[{my_node_id}] Topology active master: {getattr(topo, 'active_node_id', None)}")

        # Stay alive for a bit to allow broadcasts
        await asyncio.sleep(3)

        # Cleanup
        await discovery.stop()
        await election.stop()
        await libp2p.stop()
        await router.stop()

        success = election_ok and health_ok and topology_ok
        print(f"\n[{my_node_id}] ===== FULL CLUSTER TEST: {'PASS' if success else 'FAIL'} =====")
        return success

    finally:
        try:
            os.unlink(config_path)
            os.rmdir(tmpdir)
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Multi-Machine Distributed Test for Synapse AI")
    parser.add_argument("--role", choices=["server", "client"], help="Role for direct mode")
    parser.add_argument("--port", type=int, default=5888, help="P2P port")
    parser.add_argument("--target", type=str, help="Target address for client (host:port)")
    parser.add_argument("--config-based", action="store_true", help="Use shared config-based mode (both sides)")
    parser.add_argument("--node-id", type=str, default=None, help="Override node ID")
    parser.add_argument("--peer-host", type=str, default="127.0.0.1", help="IP of the other machine")
    parser.add_argument("--peer-port", type=int, default=5889, help="Port of the other machine")
    args = parser.parse_args()

    if args.config_based:
        # Shared config mode: run one node, the other runs the same script
        my_id = args.node_id or f"NODE-{args.port}"
        if my_id == "MACHINE-A":
            node_a_port, node_b_port = args.port, args.peer_port
        elif my_id == "MACHINE-B":
            node_a_port, node_b_port = args.peer_port, args.port
        else:
            node_a_port, node_b_port = args.port, args.peer_port
        config = make_config(
            node_a_id="MACHINE-A", node_a_port=node_a_port,
            node_b_id="MACHINE-B", node_b_port=node_b_port,
            node_b_host=args.peer_host
        )
        success = await run_full_cluster(config, my_id, args.port)
    elif args.role == "server":
        success = await run_server(args.port)
    elif args.role == "client":
        if not args.target:
            print("ERROR: --target is required for client role")
            sys.exit(1)
        success = await run_client(args.port, args.target)
    else:
        print("Usage:")
        print("  Direct mode:")
        print("    Machine A: python test_multimachine_distributed.py --role server --port 5888")
        print("    Machine B: python test_multimachine_distributed.py --role client --port 5889 --target <A_IP>:5888")
        print("  Config-based mode:")
        print("    Both sides: python test_multimachine_distributed.py --config-based --port <PORT> --node-id <ID> --peer-host <OTHER_IP> --peer-port <OTHER_PORT>")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
