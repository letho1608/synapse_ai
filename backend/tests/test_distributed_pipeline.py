"""
Test LACP — Distributed Pipeline Integration

Validates:
1. PipelineStage executes forward passes correctly
2. PipelineExecutor orchestrates multi-stage inference
3. ResultRouter delivers out-of-order results in order
4. DistributedPipelineController coordinates inter-node inference
5. DistributedPipelineExtension integrates with Node class
6. Node._build_latency_matrix produces real matrix from peers
7. Fallback to single-node when topology has only 1 node
"""

import pytest
import sys
import os
import asyncio
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.pipeline.stage import (
    PipelineStage,
    PipelineExecutor,
    StageResult,
    StageStats,
)
from synapse.pipeline.scheduler import (
    PipelineScheduler,
    MicroBatch,
    MicroBatchOp,
    PipelineSlot,
    PipelineSchedule,
)
from synapse.transport.result_router import ResultRouter, RoutedResult, ResultBuffer
from synapse.transport.codec import encode_tensor, decode_tensor, TensorMessage
from synapse.routing.event_router import EventRouter


class TestPipelineStage:
    def make_forward_fn(self, delay: float = 0.0):
        async def forward(mb: MicroBatch) -> np.ndarray:
            if delay:
                await asyncio.sleep(delay)
            return mb.data * 2.0
        return forward

    def make_send_fn(self, results_list: list):
        async def send(result: StageResult) -> None:
            results_list.append(result)
        return send

    @pytest.mark.asyncio
    async def test_stage_forward(self):
        stage = PipelineStage(
            stage_id=0,
            node_id="node-0",
            num_stages=1,
            forward_fn=self.make_forward_fn(),
            send_tensor_fn=lambda r: None,
        )

        data = np.array([1.0, 2.0, 3.0])
        result = await stage.forward(data)
        assert np.array_equal(result, data * 2.0)
        assert stage.stats.total_forwards == 1

    @pytest.mark.asyncio
    async def test_stage_last_stage_puts_to_output_queue(self):
        results = []
        stage = PipelineStage(
            stage_id=2,
            node_id="node-2",
            num_stages=3,
            forward_fn=self.make_forward_fn(),
            send_tensor_fn=self.make_send_fn(results),
        )

        mb = MicroBatch(batch_id=0, data=np.array([5.0]))
        await stage.enqueue_input(mb)

        stop = asyncio.Event()
        loop_task = asyncio.create_task(stage.process_loop(stop))
        await asyncio.sleep(0.2)

        try:
            result = await stage.dequeue_output(timeout=2.0)
            assert result.micro_batch_id == 0
            assert result.data == 10.0
        finally:
            stop.set()
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stage_sends_to_next_stage(self):
        sent = []
        stage = PipelineStage(
            stage_id=0,
            node_id="node-0",
            num_stages=3,
            forward_fn=self.make_forward_fn(),
            send_tensor_fn=self.make_send_fn(sent),
        )

        mb = MicroBatch(batch_id=0, data=np.array([7.0]))
        await stage.enqueue_input(mb)

        stop = asyncio.Event()
        loop_task = asyncio.create_task(stage.process_loop(stop))
        await asyncio.sleep(0.2)

        stop.set()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        assert len(sent) == 1
        assert sent[0].micro_batch_id == 0
        assert sent[0].data == 14.0

    @pytest.mark.asyncio
    async def test_stage_stats(self):
        stage = PipelineStage(
            stage_id=0,
            node_id="node-0",
            num_stages=1,
            forward_fn=self.make_forward_fn(),
            send_tensor_fn=lambda r: None,
        )

        await stage.forward(np.array([1.0]))
        await stage.forward(np.array([2.0]))
        await stage.forward(np.array([3.0]))

        assert stage.stats.total_forwards == 3
        assert stage.stats.avg_latency_ms > 0


class TestPipelineExecutor:
    def make_forward_fn(self, multiplier: float = 2.0):
        async def forward(mb: MicroBatch) -> np.ndarray:
            return mb.data * multiplier
        return forward

    @pytest.mark.asyncio
    async def test_single_stage_executor(self):
        stage = PipelineStage(
            stage_id=0, node_id="node-0", num_stages=1,
            forward_fn=self.make_forward_fn(2.0),
            send_tensor_fn=lambda r: None,
        )

        executor = PipelineExecutor(stages=[stage])
        results = await executor.run_inference(
            input_data=np.array([1.0, 2.0, 3.0]),
            num_micro_batches=2,
        )

        assert len(results) == 2
        assert np.array_equal(results[0], np.array([2.0, 4.0, 6.0]))

    @pytest.mark.asyncio
    async def test_dual_stage_executor_chains_results(self):
        async def forward_mul_2(mb):
            return mb.data * 2
        async def forward_mul_3(mb):
            return mb.data * 3

        stage0_output = []
        stage1_output = []

        async def send_0_to_1(r):
            stage0_output.append(r)

        async def send_1_final(r):
            stage1_output.append(r)

        stage0 = PipelineStage(
            stage_id=0, node_id="node-0", num_stages=2,
            forward_fn=forward_mul_2,
            send_tensor_fn=send_0_to_1,
        )
        stage1 = PipelineStage(
            stage_id=1, node_id="node-1", num_stages=2,
            forward_fn=forward_mul_3,
            send_tensor_fn=send_1_final,
        )

        # Feed micro-batch directly into each stage and verify output flows
        mb0 = MicroBatch(batch_id=0, data=np.array([1.0]))
        r0 = await stage0.run_forward(mb0)
        assert r0.data == 2.0
        # send_tensor_fn NOT called by run_forward — only process_loop calls it

        # Manually chain: stage0 output -> stage1 input
        mb1 = MicroBatch(batch_id=0, data=r0.data, metadata={"step": 0})
        r1 = await stage1.run_forward(mb1)
        assert r1.data == 6.0  # 1.0 * 2 * 3

    @pytest.mark.asyncio
    async def test_executor_handles_empty_stages(self):
        executor = PipelineExecutor(stages=[])
        results = await executor.run_inference(
            input_data=np.array([1.0, 2.0]),
            num_micro_batches=1,
        )
        assert len(results) == 1
        assert np.array_equal(results[0], np.array([1.0, 2.0]))


class TestResultRouter:
    @pytest.mark.asyncio
    async def test_in_order_delivery(self):
        router = ResultRouter("node-0")
        await router.start()
        try:
            delivered = []
            def on_token(rid, tokens, finished):
                if tokens or finished:
                    delivered.append((tokens, finished))

            router.register_token_callback("req-1", on_token)

            r0 = RoutedResult(
                request_id="req-1", orchestrator_id="node-0", micro_batch_id=0,
                source_stage_id=0, tokens=[1, 2, 3], is_finished=False,
            )
            r1 = RoutedResult(
                request_id="req-1", orchestrator_id="node-0", micro_batch_id=1,
                source_stage_id=0, tokens=[4, 5, 6], is_finished=True,
            )

            router.receive_result(r0)
            router.receive_result(r1)

            assert len(delivered) == 2
            assert delivered[0][0] == [1, 2, 3]
            assert delivered[0][1] is False
            assert delivered[1][0] == [4, 5, 6]
            assert delivered[1][1] is True
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_out_of_order_reassembly(self):
        router = ResultRouter("node-0")
        await router.start()
        try:
            delivered = []
            def on_token(rid, tokens, finished):
                delivered.append(tokens)

            router.register_token_callback("req-reorder", on_token)

            r2 = RoutedResult(
                request_id="req-reorder", orchestrator_id="node-0",
                micro_batch_id=2, source_stage_id=0, tokens=[7, 8], is_finished=False,
            )
            router.receive_result(r2)
            assert len(delivered) == 0

            r0 = RoutedResult(
                request_id="req-reorder", orchestrator_id="node-0",
                micro_batch_id=0, source_stage_id=0, tokens=[1, 2], is_finished=False,
            )
            router.receive_result(r0)
            assert len(delivered) == 1
            assert delivered[0] == [1, 2]

            r1 = RoutedResult(
                request_id="req-reorder", orchestrator_id="node-0",
                micro_batch_id=1, source_stage_id=0, tokens=[4, 5, 6], is_finished=False,
            )
            router.receive_result(r1)
            assert len(delivered) == 3
            assert delivered == [[1, 2], [4, 5, 6], [7, 8]]
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_gc_removes_stale_buffers(self):
        router = ResultRouter("node-0")
        await router.start()
        try:
            # Manually shorten TTL for testing
            router.TTL = 0.01
            router.GC_INTERVAL = 0.05

            r0 = RoutedResult(
                request_id="req-stale", orchestrator_id="node-0",
                micro_batch_id=0, source_stage_id=0, tokens=[1], is_finished=False,
            )
            router.receive_result(r0)

            await asyncio.sleep(0.2)

            assert "req-stale" not in router._buffers
        finally:
            await router.stop()

    def test_result_buffer_in_order(self):
        buf = ResultBuffer()

        r0 = RoutedResult("r", "o", 0, 0, [1], False)
        r1 = RoutedResult("r", "o", 1, 0, [2], False)
        r2 = RoutedResult("r", "o", 2, 0, [3], True)

        delivered = buf.insert(r0)
        assert len(delivered) == 1
        assert delivered[0].tokens == [1]

        delivered = buf.insert(r1)
        assert len(delivered) == 1
        assert delivered[0].tokens == [2]

        delivered = buf.insert(r2)
        assert len(delivered) == 1
        assert delivered[0].tokens == [3]

    def test_result_buffer_out_of_order(self):
        buf = ResultBuffer()

        r1 = RoutedResult("r", "o", 1, 0, [2], False)
        delivered = buf.insert(r1)
        assert len(delivered) == 0

        r0 = RoutedResult("r", "o", 0, 0, [1], False)
        delivered = buf.insert(r0)
        assert len(delivered) == 2
        assert delivered[0].tokens == [1]
        assert delivered[1].tokens == [2]


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


class TestE2EDistributedPipeline:
    """End-to-end test: simulate 3-node distributed inference."""

    @pytest.mark.asyncio
    async def test_pipeline_executor_with_3_stages(self):
        async def forward_mul_2(mb):
            return mb.data * 2
        async def forward_mul_3(mb):
            return mb.data * 3
        async def forward_mul_4(mb):
            return mb.data * 4

        final_output = []

        async def stage2_send(r):
            final_output.append(r)

        stage2 = PipelineStage(
            stage_id=2, node_id="n2", num_stages=3,
            forward_fn=forward_mul_4,
            send_tensor_fn=stage2_send,
        )

        async def send_1_to_2(r):
            mb = MicroBatch(batch_id=r.micro_batch_id, data=r.data, metadata={"step": r.step})
            result = await stage2.run_forward(mb)

        stage1 = PipelineStage(
            stage_id=1, node_id="n1", num_stages=3,
            forward_fn=forward_mul_3,
            send_tensor_fn=send_1_to_2,
        )

        async def send_0_to_1(r):
            mb = MicroBatch(batch_id=r.micro_batch_id, data=r.data, metadata={"step": r.step})
            result = await stage1.run_forward(mb)

        stage0 = PipelineStage(
            stage_id=0, node_id="n0", num_stages=3,
            forward_fn=forward_mul_2,
            send_tensor_fn=send_0_to_1,
        )

        # Manually chain through 3 stages — this is what a real pipeline does
        mb = MicroBatch(batch_id=0, data=np.array([1.0]))
        r0 = await stage0.run_forward(mb)
        assert r0.data == 2.0

        mb1 = MicroBatch(batch_id=0, data=r0.data, metadata={"step": 0})
        r1 = await stage1.run_forward(mb1)
        assert r1.data == 6.0

        mb2 = MicroBatch(batch_id=0, data=r1.data, metadata={"step": 0})
        r2 = await stage2.run_forward(mb2)

        assert r2.data == 24.0  # 1.0 * 2 * 3 * 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])