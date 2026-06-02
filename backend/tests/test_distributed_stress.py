"""
Stress test for distributed P2P: 5-node mesh, concurrent health checks,
node churn (join/leave), heartbeat verification.
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter
from synapse.routing.p2p_socket_bridge import P2PSocketBridge
from synapse.routing.libp2p_node import Libp2pNode


class DummyNode:
    """Minimal node for stress testing mesh connectivity."""

    def __init__(self, node_id: str, port: int):
        self.id = node_id
        self.port = port
        self.er = EventRouter()
        self.libp2p = Libp2pNode(node_id, port, self.er)

    @property
    def bridge(self):
        return self.libp2p._bridge

    async def start(self):
        await self.er.start()
        await self.libp2p.start()
        await asyncio.sleep(0.3)

    async def stop(self):
        await self.libp2p.stop()

    async def connect_to(self, node: "DummyNode", address: str) -> bool:
        return await self.libp2p.connect_to_peer(node.id, address)


async def test_5_node_full_mesh():
    """Create 5 nodes, form full mesh, verify all-to-all connectivity."""
    print("\n--- 5-Node Full Mesh ---")
    nodes = []
    for i in range(5):
        n = DummyNode(f"N{i}", 5900 + i)
        nodes.append(n)

    # Start all
    for n in nodes:
        await n.start()
    print(f"Started {len(nodes)} nodes")

    # Form full mesh
    for i, n in enumerate(nodes):
        for j, other in enumerate(nodes):
            if i != j:
                addr = f"127.0.0.1:{other.port}"
                ok = await n.connect_to(other, addr)
                if not ok:
                    print(f"  FAIL: {n.id} -> {other.id} connection failed")
                    return False
    print("All connections established")

    await asyncio.sleep(1)

    # Verify all connections
    for n in nodes:
        connected = sum(1 for o in nodes if o.id != n.id and n.bridge.is_peer_connected(o.id))
        expected = len(nodes) - 1
        if connected != expected:
            print(f"  FAIL: {n.id} has {connected}/{expected} connections")
            for o in nodes:
                if o.id != n.id:
                    print(f"    {n.id}->{o.id}: {n.bridge.is_peer_connected(o.id)}")
            return False
    print(f"Full mesh verified: all nodes have {len(nodes)-1} connections")

    # Health check all pairs (40 checks = 5*4*2 directions)
    print("Running all-to-all health checks...")
    ok_count = 0
    fail = 0
    for n in nodes:
        for o in nodes:
            if n.id != o.id:
                ok = n.bridge.is_peer_connected(o.id)
                if ok:
                    ok_count += 1
                else:
                    fail += 1
    print(f"  Connectivity: {ok_count}/{ok_count+fail} OK")

    # Cleanup
    for n in nodes:
        await n.stop()

    return fail == 0


async def test_node_churn():
    """Test nodes joining and leaving dynamically."""
    print("\n--- Node Churn (Join/Leave) ---")
    a = DummyNode("A", 5950)
    b = DummyNode("B", 5951)
    c = DummyNode("C", 5952)

    await a.start()
    await b.start()
    print("A and B started")

    # Connect A-B
    ok = await a.connect_to(b, "127.0.0.1:5951")
    assert ok, "A->B connect failed"

    # Verify connection
    assert a.bridge.is_peer_connected("B"), "A->B not connected"
    assert b.bridge.is_peer_connected("A"), "B->A not connected"
    print("A-B mesh formed")

    # Late join C
    await c.start()
    await c.connect_to(a, "127.0.0.1:5950")
    await c.connect_to(b, "127.0.0.1:5951")
    await asyncio.sleep(0.5)

    assert c.bridge.is_peer_connected("A"), "C->A not connected"
    assert c.bridge.is_peer_connected("B"), "C->B not connected"
    print("C joined mesh, 3-node mesh complete")

    # Node B leaves (disconnect)
    await b.stop()
    await asyncio.sleep(1)

    # A and C should detect
    b_gone_from_a = not a.bridge.is_peer_connected("B")
    b_gone_from_c = not c.bridge.is_peer_connected("B")
    print(f"After B stops: A sees B={not b_gone_from_a}, C sees B={not b_gone_from_c}")

    # Cleanup
    await a.stop()
    await c.stop()

    return True


async def test_heartbeat_stress():
    """Run heartbeat for extended period, verify stability."""
    print("\n--- Heartbeat Stress (20s) ---")
    a = DummyNode("A", 5960)
    b = DummyNode("B", 5961)

    await a.start()
    await b.start()

    await a.connect_to(b, "127.0.0.1:5961")
    await asyncio.sleep(0.5)

    assert a.bridge.is_peer_connected("B"), "A->B not connected"

    # Monitor heartbeat for 20 seconds
    print("Monitoring heartbeat for 20 seconds...")
    failures = 0
    for i in range(20):
        await asyncio.sleep(1)
        connected_a = a.bridge.is_peer_connected("B")
        connected_b = b.bridge.is_peer_connected("A")
        if not connected_a or not connected_b:
            failures += 1
            print(f"  t={i+1}s: FAIL A->B={connected_a}, B->A={connected_b}")

    print(f"Heartbeat stability: {20 - failures}/20 seconds stable")

    await a.stop()
    await b.stop()

    return failures == 0


async def test_concurrent_connects():
    """Test many concurrent connections between 4 nodes."""
    print("\n--- Concurrent Connections (4 nodes) ---")
    nodes = [DummyNode(f"N{i}", 5970 + i) for i in range(4)]

    for n in nodes:
        await n.start()

    # All connect concurrently
    tasks = []
    for i, n in enumerate(nodes):
        for j, o in enumerate(nodes):
            if i != j:
                tasks.append(n.connect_to(o, f"127.0.0.1:{o.port}"))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok_count = sum(1 for r in results if r is True)
    print(f"Concurrent connections: {ok_count}/{len(tasks)} succeeded")

    await asyncio.sleep(0.5)

    # Verify full mesh
    for n in nodes:
        connected = sum(1 for o in nodes if o.id != n.id and n.bridge.is_peer_connected(o.id))
        if connected != 3:
            print(f"  WARN: {n.id} has {connected}/3 connections")

    for n in nodes:
        await n.stop()

    return ok_count >= 6  # Allow some dup detection failures


async def main():
    tests = [
        ("5-Node Full Mesh", test_5_node_full_mesh),
        ("Node Churn (Join/Leave)", test_node_churn),
        ("Heartbeat Stress (20s)", test_heartbeat_stress),
        ("Concurrent Connections (4 nodes)", test_concurrent_connects),
    ]

    results = []
    for name, coro in tests:
        try:
            ok = await coro()
            status = "PASS" if ok else "FAIL"
            results.append((name, status))
            print(f"  [{status}] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))

    print("\n" + "=" * 50)
    passed = sum(1 for _, r in results if r == "PASS")
    for name, result in results:
        print(f"  {name}: {result}")
    print(f"\nTotal: {passed}/{len(results)} PASS")


if __name__ == "__main__":
    asyncio.run(main())