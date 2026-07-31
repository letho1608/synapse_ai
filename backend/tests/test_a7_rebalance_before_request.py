"""
Behavioral test: A7 — rebalancing works before the first distributed request.

The rebalance path in node.check_fault_tolerance() is gated on
node._distributed_pipeline. That extension used to be initialized only
lazily by the first distributed request, so a node that died before any
request was ever rebalanced. node.start() now initializes the distributed
pipeline extension unconditionally; this test drives the real start() path
(no distributed request ever runs) and asserts a dead node is rebalanced.
"""
import tempfile

import numpy as np
import pytest

from synapse.inference.inference_engine import InferenceEngine
from synapse.orchestration.node import Node
from synapse.routing import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.topology.topology import Topology


class _Engine(InferenceEngine):
    async def encode(self, shard, prompt):
        return np.array([])

    async def sample(self, x):
        return np.array([1])

    async def decode(self, shard, tokens):
        return ""

    async def infer_tensor(self, rid, s, t, state=None):
        return np.array([]), {}

    async def load_checkpoint(self, s, p):
        pass


def _dc(vram_mb, fp16_tflops):
    return DeviceCapabilities(
        model="Test-GPU", chip="Test-Chip", memory=vram_mb,
        flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
        cpu_cores=4, disk_gb=100, system_ram_mb=8192,
    )


@pytest.mark.asyncio
async def test_rebalance_runs_before_any_distributed_request():
    """A7: node.start() initializes the distributed pipeline extension
    unconditionally, so check_fault_tolerance() rebalances a dead node even
    though no distributed request ever initialized it."""
    router = EventRouter()
    libp2p = Libp2pNode("master", 60021, router)
    node = Node(
        _id="master",
        event_router=router,
        libp2p_node=libp2p,
        inference_engine=_Engine(),
        shard_downloader=None,
    )
    node.tflops_proportional = False

    try:
        await node.start()

        # No distributed request ever ran — the extension must already exist
        # because start() initialized it unconditionally (not lazily by a
        # request) and started its ResultRouter.
        from synapse.pipeline.node_extension import DistributedPipelineExtension
        assert isinstance(node._distributed_pipeline, DistributedPipelineExtension), (
            "node.start() did not initialize the distributed pipeline extension"
        )

        # Populate the cluster the way a live multi-node mesh would (start()
        # collects a self-only topology; the periodic peer loop never
        # replaces it). node-b dies before any distributed request.
        topo = Topology()
        topo.update_node("node-a", _dc(4096, 2.0))
        topo.update_node("node-b", _dc(6144, 3.0))
        node.topology = topo

        ft = node.fault_tolerance
        # Keep the monitor loop out of the way (its 2 s cadence would mark
        # silently-idle nodes dead); node-b dies explicitly below.
        ft.HEARTBEAT_INTERVAL = 100.0
        ft.HEARTBEAT_TIMEOUT = 60.0
        with tempfile.TemporaryDirectory(prefix="a7-ckpt-") as tmpdir:
            ft.checkpoint_dir = tmpdir
            ft.register_node("node-a")
            ft.register_node("node-b")
            ft._health["node-b"].is_alive = False
            ft._health["node-b"].is_blacklisted = True

            await node.check_fault_tolerance()

        result = node._last_rebalance_result
        assert result is not None, (
            "rebalance never ran — check_fault_tolerance is still gated on a "
            "request-initialized pipeline"
        )
        assert result.success, f"rebalance failed: {result}"
        assert result.removed_node == "node-b", (
            f"expected rebalance for node-b, got {result.removed_node}"
        )
    finally:
        await node.stop()
