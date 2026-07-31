"""
Behavioral test: the production bandwidth probe measures a real
transfer-time payload (not a single-1KB RTT), aggregates multiple samples,
and the measured value demonstrably flows probe → latency matrix →
NodeResources → partition/election resource scores.

The whole chain runs against the shipped code: a real Libp2pNode/P2PSocketBridge
listens on a port, the LatencyProber opens a raw TCP connection to it, the
bridge echoes the probe, and the resulting entry lands in a LatencyMatrix and
then in resource scoring via apply_measured_bandwidth.
"""
import asyncio

import pytest

from synapse.routing.event_router import EventRouter
from synapse.routing.libp2p_node import Libp2pNode


class _BridgeNode:
    """A real Libp2pNode whose P2PSocketBridge listens on a port."""

    def __init__(self, node_id: str, port: int):
        self.id = node_id
        self.port = port
        self.er = EventRouter()
        self.libp2p = Libp2pNode(node_id, port, self.er)

    async def start(self):
        await self.er.start()
        await self.libp2p.start()
        await asyncio.sleep(0.3)

    async def stop(self):
        await self.libp2p.stop()
        await self.er.stop()


@pytest.mark.asyncio
async def test_real_probe_bandwidth_flows_into_scoring():
    from synapse.topology.latency_matrix import LatencyProber
    from synapse.topology.resource_weight import (
        NodeResources,
        ResourceScorer,
        apply_measured_bandwidth,
    )

    target = _BridgeNode("bw-target", 59401)
    await target.start()
    try:
        prober = LatencyProber()

        # Real probe against the production listener (bridge echo). The
        # 64 KiB payload makes the estimate transfer-time-based.
        entry = await prober.probe("bw-self", f"127.0.0.1:{target.port}", timeout=3.0)
        assert entry is not None, "probe failed against the production bridge"
        assert entry.bandwidth_mbps > 50.0, (
            "bandwidth looks like a 1KB-RTT proxy "
            f"(got {entry.bandwidth_mbps:.1f} Mbps on a loopback link)"
        )
        assert LatencyProber.PROBE_SIZE >= 65536, (
            "probe payload must be transfer-time-sized, not a 1KB RTT"
        )
        assert LatencyProber.PROBE_COUNT >= 5, "probe must aggregate samples"

        # Flow step 1: probe -> latency matrix (via probe_all).
        matrix = await prober.probe_all(
            "bw-self", {"bw-target": f"127.0.0.1:{target.port}"}
        )
        me = matrix.get("bw-self", "bw-target")
        assert me is not None, "measured entry missing from the latency matrix"
        # A second independent measurement: same magnitude class (loopback,
        # transfer-time-based), not a 1KB-RTT proxy (~10 Mbps).
        assert me.bandwidth_mbps > 50.0, (
            f"matrix bandwidth {me.bandwidth_mbps:.1f} Mbps looks like a 1KB-RTT proxy"
        )

        # Flow step 2: matrix -> NodeResources.bandwidth_mbps (replaces the
        # 1000 Mbps placeholder for the measured link).
        nodes = {
            "bw-self": NodeResources(gpu_vram_mb=8192, gpu_tflops_fp16=20.0),
            "bw-target": NodeResources(gpu_vram_mb=4096, gpu_tflops_fp16=10.0),
        }
        apply_measured_bandwidth(nodes, matrix, self_node_id="bw-self")
        assert nodes["bw-target"].bandwidth_mbps == pytest.approx(
            me.bandwidth_mbps, rel=0.01
        ), "measured bandwidth did not flow into NodeResources"

        # Flow step 3: NodeResources -> partition/election scoring: the
        # measured link must change the bandwidth dimension of the composite
        # score (vs. the all-placeholder baseline).
        scorer = ResourceScorer()
        with_bw = scorer.compute_scores(nodes)["bw-target"][0]
        baseline = {
            "bw-self": NodeResources(gpu_vram_mb=8192, gpu_tflops_fp16=20.0),
            "bw-target": NodeResources(gpu_vram_mb=4096, gpu_tflops_fp16=10.0),
        }
        without_bw = scorer.compute_scores(baseline)["bw-target"][0]
        assert with_bw != without_bw, (
            "measured bandwidth did not affect the resource score"
        )
    finally:
        await target.stop()
