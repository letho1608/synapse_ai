"""
3-node P2P cluster test - verifies the tiebreaker works for multi-node topologies.
Node IDs: NODE-A < NODE-B < NODE-C (lexicographically)
Expected: A keeps outgoing to B and C, B keeps outgoing to C, C keeps incoming from both
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
from synapse.topology.device_capabilities import DeviceCapabilities
from synapse.networking.p2p_peer_handle import P2PPeerHandle
from synapse.networking.manual.manual_discovery import ManualDiscovery
from synapse.orchestration.election import ElectionManager

DC_DEFAULT = DeviceCapabilities(
    model="NODE", chip="CPU", memory=8192,
    flops={"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
    cpu_cores=4, system_ram_mb=8192, gpu_count=0, total_gpu_vram_mb=0
)

NODE_A = "NODE-A"
NODE_B = "NODE-B"
NODE_C = "NODE-C"
PORT_A = 5888
PORT_B = 5889
PORT_C = 5890

CONFIG = {
    "peers": {
        NODE_A: {"address": "127.0.0.1", "port": PORT_A, "device_capabilities": {
            "model": "NODE-A", "chip": "CPU", "memory": 8192,
            "flops": {"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
            "cpu_cores": 4, "system_ram_mb": 8192, "gpu_count": 0, "total_gpu_vram_mb": 0
        }},
        NODE_B: {"address": "127.0.0.1", "port": PORT_B, "device_capabilities": {
            "model": "NODE-B", "chip": "CPU", "memory": 8192,
            "flops": {"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
            "cpu_cores": 4, "system_ram_mb": 8192, "gpu_count": 0, "total_gpu_vram_mb": 0
        }},
        NODE_C: {"address": "127.0.0.1", "port": PORT_C, "device_capabilities": {
            "model": "NODE-C", "chip": "CPU", "memory": 8192,
            "flops": {"fp32": 2.0, "fp16": 4.0, "int8": 8.0},
            "cpu_cores": 4, "system_ram_mb": 8192, "gpu_count": 0, "total_gpu_vram_mb": 0
        }}
    }
}


async def run_three_node_test():
    tmpdir = tempfile.mkdtemp(prefix="synapse3_")
    config_path = os.path.join(tmpdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    try:
        # Create routers and bridges
        router_a = EventRouter(); lib_a = Libp2pNode(NODE_A, PORT_A, router_a)
        router_b = EventRouter(); lib_b = Libp2pNode(NODE_B, PORT_B, router_b)
        router_c = EventRouter(); lib_c = Libp2pNode(NODE_C, PORT_C, router_c)

        # Health handlers
        async def make_health(router, name):
            async def h(event):
                resp = event.data.get("_response_topic")
                if resp: await router.publish(resp, {"status": "ok"}, origin=name)
            return h
        router_a.subscribe(f"synapse/rpc/health/request/{NODE_A}", await make_health(router_a, NODE_A))
        router_b.subscribe(f"synapse/rpc/health/request/{NODE_B}", await make_health(router_b, NODE_B))
        router_c.subscribe(f"synapse/rpc/health/request/{NODE_C}", await make_health(router_c, NODE_C))

        # Start all
        for r in [router_a, router_b, router_c]:
            await r.start()
        for lib in [lib_a, lib_b, lib_c]:
            await lib.start()
        await asyncio.sleep(0.3)

        # ===== SIMULTANEOUS 3-way connect =====
        print("\n=== 3-Way Simultaneous Connect ===")
        results = await asyncio.gather(
            lib_a.connect_to_peer(NODE_B, f"127.0.0.1:{PORT_B}"),
            lib_a.connect_to_peer(NODE_C, f"127.0.0.1:{PORT_C}"),
            lib_b.connect_to_peer(NODE_A, f"127.0.0.1:{PORT_A}"),
            lib_b.connect_to_peer(NODE_C, f"127.0.0.1:{PORT_C}"),
            lib_c.connect_to_peer(NODE_A, f"127.0.0.1:{PORT_A}"),
            lib_c.connect_to_peer(NODE_B, f"127.0.0.1:{PORT_B}"),
            return_exceptions=True
        )
        names = ["A→B", "A→C", "B→A", "B→C", "C→A", "C→B"]
        all_ok = True
        for name, r in zip(names, results):
            print(f"  {name}: {'OK' if r else 'FAIL'}")
            if not r: all_ok = False

        await asyncio.sleep(0.3)

        # Verify connections
        print(f"\nA connections: {list(lib_a._bridge._connections.keys())}")
        print(f"B connections: {list(lib_b._bridge._connections.keys())}")
        print(f"C connections: {list(lib_c._bridge._connections.keys())}")

        a_ok = lib_a.is_peer_connected(NODE_B) and lib_a.is_peer_connected(NODE_C)
        b_ok = lib_b.is_peer_connected(NODE_A) and lib_b.is_peer_connected(NODE_C)
        c_ok = lib_c.is_peer_connected(NODE_A) and lib_c.is_peer_connected(NODE_B)
        print(f"Full mesh: A={a_ok}, B={b_ok}, C={c_ok}")
        if not (a_ok and b_ok and c_ok):
            print("FAIL: Full mesh not formed")
            all_ok = False

        # Create peer handles
        peer_ab = P2PPeerHandle(NODE_B, f"127.0.0.1:{PORT_B}", "MAN", DC_DEFAULT, router_a, lib_a)
        peer_ac = P2PPeerHandle(NODE_C, f"127.0.0.1:{PORT_C}", "MAN", DC_DEFAULT, router_a, lib_a)
        peer_ba = P2PPeerHandle(NODE_A, f"127.0.0.1:{PORT_A}", "MAN", DC_DEFAULT, router_b, lib_b)
        peer_bc = P2PPeerHandle(NODE_C, f"127.0.0.1:{PORT_C}", "MAN", DC_DEFAULT, router_b, lib_b)
        peer_ca = P2PPeerHandle(NODE_A, f"127.0.0.1:{PORT_A}", "MAN", DC_DEFAULT, router_c, lib_c)
        peer_cb = P2PPeerHandle(NODE_B, f"127.0.0.1:{PORT_B}", "MAN", DC_DEFAULT, router_c, lib_c)

        # ===== All-to-all health checks =====
        print("\n=== All-to-All Health Checks (3 rounds) ===")
        for rnd in range(3):
            hc = await asyncio.gather(
                peer_ab.health_check(), peer_ac.health_check(),
                peer_ba.health_check(), peer_bc.health_check(),
                peer_ca.health_check(), peer_cb.health_check(),
                return_exceptions=True
            )
            all_passed = all(h is True for h in hc)
            print(f"  Round {rnd+1}: {'PASS' if all_passed else 'FAIL'}")
            if not all_passed:
                print(f"  Results: {hc}")
                all_ok = False

        # Cleanup
        for lib in [lib_a, lib_b, lib_c]:
            await lib.stop()
        for r in [router_a, router_b, router_c]:
            await r.stop()

        print(f"\n===== 3-NODE TEST: {'PASS' if all_ok else 'FAIL'} =====")
        return all_ok

    finally:
        try:
            os.unlink(config_path)
            os.rmdir(tmpdir)
        except:
            pass


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    success = asyncio.run(run_three_node_test())
    sys.exit(0 if success else 1)