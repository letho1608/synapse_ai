"""
Test LACP — Distributed Pipeline Integration

Validates:
1. Results are awaited per step (TestResultWaiter)
2. Node._build_latency_matrix produces a real matrix from peers
"""

import pytest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.routing.event_router import EventRouter


class TestLatencyMatrixIntegration:
    def test_node_latency_matrix_single_node(self):
        from synapse.orchestration.node import Node
        from synapse.routing.event_router import EventRouter
        from synapse.routing.libp2p_node import Libp2pNode
        from synapse.inference.inference_engine import InferenceEngine

        node_id = "test-node-lat"
        router = EventRouter()
        libp2p = Libp2pNode(node_id, 59999, router)

        class MockEngine(InferenceEngine):
            async def encode(self, shard, prompt): return np.array([])
            async def sample(self, x): return np.array([1])
            async def decode(self, shard, tokens): return ""
            async def infer_tensor(self, rid, s, t, state=None): return np.array([]), {}
            async def load_checkpoint(self, s, p): pass

        node = Node(
            _id=node_id,
            event_router=router,
            libp2p_node=libp2p,
            inference_engine=MockEngine(),
            shard_downloader=None,
        )

        matrix = node._build_latency_matrix()
        assert matrix is None


class TestResultWaiter:
    """The step-result waiter subscribes BEFORE the step is sent and accepts
    only the last stage's result for its micro-batch (B2 batch identity), or
    a kv_miss reply from any stage. (Replaces _collect_result, which
    subscribed after sending and could miss a fast local result.)"""

    @staticmethod
    async def _controller():
        from synapse.inference.distributed_engine import DistributedPipelineController
        router = EventRouter()
        controller = DistributedPipelineController(
            node_id="orchestrator", inference_engine=None, event_router=router)
        await controller.start()
        return controller, router

    @pytest.mark.asyncio
    async def test_receives_result_published_after_subscribe(self):
        controller, router = await self._controller()
        waiter = controller._result_waiter("rid-1", 0, 5)
        try:
            await router.publish("synapse/inference/result/rid-1",
                                 {"tokens": [101, 102, 103], "micro_batch_id": 0,
                                  "source_stage_id": 5})
            result = await waiter.wait(3.0)
            assert result is not None and result.tokens == [101, 102, 103]
        finally:
            waiter.close()
            await controller.stop()

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        controller, _ = await self._controller()
        waiter = controller._result_waiter("rid-2", 0, 5)
        try:
            assert await waiter.wait(0.1) is None
        finally:
            waiter.close()
            await controller.stop()

    @pytest.mark.asyncio
    async def test_rejects_wrong_micro_batch_and_wrong_stage(self):
        controller, router = await self._controller()
        waiter = controller._result_waiter("rid-3", 1, 5)
        topic = "synapse/inference/result/rid-3"
        try:
            await router.publish(topic, {"tokens": [999], "micro_batch_id": 0, "source_stage_id": 5})
            await router.publish(topic, {"tokens": [777], "micro_batch_id": 1, "source_stage_id": 3})
            assert await waiter.wait(0.2) is None, "a foreign result was accepted"
            await router.publish(topic, {"tokens": [101, 102], "micro_batch_id": 1, "source_stage_id": 5})
            result = await waiter.wait(3.0)
            assert result.tokens == [101, 102] and result.micro_batch_id == 1
        finally:
            waiter.close()
            await controller.stop()

    @pytest.mark.asyncio
    async def test_kv_miss_from_any_stage_is_accepted(self):
        controller, router = await self._controller()
        waiter = controller._result_waiter("rid-4", 2, 5)
        try:
            await router.publish("synapse/inference/result/rid-4",
                                 {"tokens": [], "kv_miss": True, "micro_batch_id": 2,
                                  "source_stage_id": None})
            result = await waiter.wait(3.0)
            assert result is not None and result.metadata["kv_miss"] is True
        finally:
            waiter.close()
            await controller.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])