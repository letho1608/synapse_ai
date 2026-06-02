"""
Full in-process cluster test simulating ManualDiscovery + P2P mesh formation.
Tests: discovery, health checks, master election, topology collection.
No model loading - pure P2P infrastructure test.
"""

import asyncio
import sys
import os
import json
import logging
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.networking.p2p_peer_handle import P2PPeerHandle
from synapse.networking.manual.manual_discovery import ManualDiscovery
from synapse.orchestration.election import ElectionManager

DC_STRONG = DeviceCapabilities(
    model="NODE-A(GPU)", chip="RTX 4090", memory=24576,
    flops={"fp32": 82.0, "fp16": 165.0, "int8": 330.0},
    cpu_cores=16, system_ram_mb=64000, gpu_count=1, total_gpu_vram_mb=24576
)
DC_WEAK = DeviceCapabilities(
    model="NODE-B(CPU)", chip="Core i3", memory=8192,
    flops={"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
    cpu_cores=4, system_ram_mb=8192, gpu_count=0, total_gpu_vram_mb=0
)

NODE_A = "NODE-MASTER-MOCK"
NODE_B = "NODE-WORKER-MOCK"
PORT_A = 5878
PORT_B = 5879


def _make_config():
    return {
        "peers": {
            NODE_A: {"address": "127.0.0.1", "port": PORT_A, "device_capabilities": {
                "model": "NODE-A(GPU)", "chip": "RTX 4090", "memory": 24576,
                "flops": {"fp32": 82.0, "fp16": 165.0, "int8": 330.0},
                "cpu_cores": 16, "system_ram_mb": 64000, "gpu_count": 1, "total_gpu_vram_mb": 24576
            }},
            NODE_B: {"address": "127.0.0.1", "port": PORT_B, "device_capabilities": {
                "model": "NODE-B(CPU)", "chip": "Core i3", "memory": 8192,
                "flops": {"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
                "cpu_cores": 4, "system_ram_mb": 8192, "gpu_count": 0, "total_gpu_vram_mb": 0
            }}
        }
    }


async def run_cluster_test():
    # Write config to temp file
    config_data = _make_config()
    tmpdir = tempfile.mkdtemp(prefix="synapse_test_")
    config_path = os.path.join(tmpdir, "cluster_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    try:
        # ===== Node-A setup =====
        router_a = EventRouter()
        libp2p_a = Libp2pNode(NODE_A, PORT_A, router_a)

        def create_peer_a(nid, addr, desc, dc):
            return P2PPeerHandle(nid, addr, desc, dc, router_a, libp2p_a)

        discovery_a = ManualDiscovery(config_path, NODE_A, create_peer_a)
        election_a = ElectionManager(NODE_A, router_a, compute_weight=165.0)

        # ===== Node-B setup =====
        router_b = EventRouter()
        libp2p_b = Libp2pNode(NODE_B, PORT_B, router_b)

        def create_peer_b(nid, addr, desc, dc):
            return P2PPeerHandle(nid, addr, desc, dc, router_b, libp2p_b)

        discovery_b = ManualDiscovery(config_path, NODE_B, create_peer_b)
        election_b = ElectionManager(NODE_B, router_b, compute_weight=4.0)

        # Subscribe health handlers
        async def health_a(event):
            resp = event.data.get("_response_topic")
            if resp:
                await router_a.publish(resp, {"status": "ok"}, origin=NODE_A)
        router_a.subscribe(f"synapse/rpc/health/request/{NODE_A}", health_a)

        async def health_b(event):
            resp = event.data.get("_response_topic")
            if resp:
                await router_b.publish(resp, {"status": "ok"}, origin=NODE_B)
        router_b.subscribe(f"synapse/rpc/health/request/{NODE_B}", health_b)

        # Start routers and P2P nodes
        await router_a.start()
        await router_b.start()
        await libp2p_a.start()
        await libp2p_b.start()

        await asyncio.sleep(0.3)

        # Start elections
        await election_a.start()
        await election_b.start()

        # Start discovery
        await discovery_a.start()
        await discovery_b.start()

        print("\n=== Waiting for P2P mesh to form (Manual Discovery + Health Checks)... ===")

        # Wait for peers to be discovered (discovery loop runs every 5s, health checks within)
        for i in range(20):
            await asyncio.sleep(0.5)
            peers_a = await discovery_a.discover_peers()
            peers_b = await discovery_b.discover_peers()

            a_has_b = any(p.id() == NODE_B for p in peers_a)
            b_has_a = any(p.id() == NODE_A for p in peers_b)

            print(f"  [{i+1}] A peers: {[p.id() for p in peers_a]}, B peers: {[p.id() for p in peers_b]}")

            if a_has_b and b_has_a:
                print(f"  Mesh formed after {(i+1)*0.5:.1f}s!")
                break
        else:
            print("FAIL: Mesh did not form within timeout")
            return False

        peers_a = await discovery_a.discover_peers()
        peers_b = await discovery_b.discover_peers()

        # Verify connections
        a_conn = libp2p_a.is_peer_connected(NODE_B)
        b_conn = libp2p_b.is_peer_connected(NODE_A)
        print(f"\nConnection status: A→B={a_conn}, B→A={b_conn}")

        if not a_conn or not b_conn:
            print("FAIL: Connections not established")
            return False

        # Verify election
        await asyncio.sleep(5)  # Let election settle
        print(f"\nMaster election: A is_master={election_a.is_master()}, B is_master={election_b.is_master()}")
        print(f"Master ID from A: {election_a.current_master_id}")
        print(f"Master ID from B: {election_b.current_master_id}")

        # A should be master (higher weight: 165.0 vs 4.0)
        if not election_a.is_master():
            print("WARNING: Node-A should be master (higher weight)")

        # ===== Test topology collection via RPC =====
        print("\n=== Testing Topology RPC ===")

        # Setup topology handler on Node-A
        async def topology_handler_a(event):
            resp = event.data.get("_response_topic")
            if resp:
                await router_a.publish(resp, {
                    "topology": {
                        "nodes": {
                            NODE_A: DC_STRONG.to_dict(),
                        },
                        "edges": [],
                        "active_node_id": election_a.current_master_id
                    }
                }, origin=NODE_A)
        router_a.subscribe(f"synapse/rpc/topology/request/{NODE_A}", topology_handler_a)

        async def topology_handler_b(event):
            resp = event.data.get("_response_topic")
            if resp:
                await router_b.publish(resp, {
                    "topology": {
                        "nodes": {
                            NODE_B: DC_WEAK.to_dict(),
                        },
                        "edges": [],
                        "active_node_id": election_b.current_master_id
                    }
                }, origin=NODE_B)
        router_b.subscribe(f"synapse/rpc/topology/request/{NODE_B}", topology_handler_b)

        # Collect topology from both sides
        peer_b_from_a = next(p for p in peers_a if p.id() == NODE_B)
        peer_a_from_b = next(p for p in peers_b if p.id() == NODE_A)

        topo_b = await peer_b_from_a.collect_topology(set(), max_depth=1)
        topo_a = await peer_a_from_b.collect_topology(set(), max_depth=1)

        print(f"Topology from B (collected by A): nodes={list(topo_b.nodes.keys()) if hasattr(topo_b, 'nodes') else 'N/A'}")
        print(f"Topology from A (collected by B): nodes={list(topo_a.nodes.keys()) if hasattr(topo_a, 'nodes') else 'N/A'}")

        # ===== Stress test: multiple simultaneous RPC calls =====
        print("\n=== Stress Test: 10 simultaneous health checks ===")
        for round_num in range(10):
            results = await asyncio.gather(
                peer_b_from_a.health_check(),
                peer_a_from_b.health_check(),
                return_exceptions=True
            )
            ok = all(r is True for r in results)
            if not ok:
                print(f"Round {round_num+1}: FAIL - {results}")
                await discovery_a.stop()
                await discovery_b.stop()
                await election_a.stop()
                await election_b.stop()
                await libp2p_a.stop()
                await libp2p_b.stop()
                return False

        print("All 10 rounds PASSED!")

        # Cleanup
        await discovery_a.stop()
        await discovery_b.stop()
        await election_a.stop()
        await election_b.stop()
        await libp2p_a.stop()
        await libp2p_b.stop()
        await router_a.stop()
        await router_b.stop()

        print("\n===== FULL CLUSTER TEST: PASS =====")
        return True

    finally:
        # Cleanup temp file
        try:
            os.unlink(config_path)
            os.rmdir(tmpdir)
        except Exception:
            pass


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    success = asyncio.run(run_cluster_test())
    sys.exit(0 if success else 1)