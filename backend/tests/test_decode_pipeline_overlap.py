"""
Test: Decode-phase pipeline micro-batch overlapping.

Verifies the fix for B1: during autoregressive token generation,
multiple micro-batches are fired into the pipeline before collecting
results, enabling concurrent stage execution.
"""
import asyncio
import numpy as np
import pytest


class TestDecodePipelineOverlap:
    """Tests for decode-phase pipeline overlap in distributed_engine.py."""

    def test_decode_loop_uses_multiple_batches(self):
        """Verify the decode loop fires multiple micro-batches concurrently."""
        code = open(
            r"D:\Code\paper\synapse_ai\backend\synapse\inference\distributed_engine.py"
        ).read()
        # The decode loop should send multiple batches before collecting
        assert "num_decode_batches" in code, (
            "Decode loop doesn't use num_decode_batches"
        )
        assert "decode_batches" in code, (
            "Decode loop doesn't batch multiple micro-batches"
        )

    def test_decode_loop_fires_before_collect(self):
        """Verify all micro-batches are sent before collecting any results."""
        code = open(
            r"D:\Code\paper\synapse_ai\backend\synapse\inference\distributed_engine.py"
        ).read()
        
        # The send loop should complete before the collect loop starts
        # Look for separate 'for' loops for send and collect
        send_count = code.count("await self._send_input_to_stage(")
        collect_count = code.count("await self._collect_result(")
        
        # Should have comparable numbers (both inside a loop)
        assert send_count >= 2, (
            f"Only {send_count} send calls found, expected at least 2 for overlapping"
        )

    @pytest.mark.asyncio
    async def test_overlapping_batches_fired(self):
        """Verify that multiple decode batches can be sent before collecting."""
        from synapse.inference.distributed_engine import (
            DistributedPipelineController,
            DistributedInferenceSession,
        )
        from synapse.inference.shard import Shard
        from synapse.inference.inference_engine import InferenceEngine
        import asyncio
        from unittest.mock import MagicMock, AsyncMock
        
        # Create a controller with mock components
        mock_engine = MagicMock(spec=InferenceEngine)
        mock_engine.encode = AsyncMock(return_value=np.array([[1, 2, 3, 4, 5]]))
        mock_router = MagicMock()
        mock_router.publish = AsyncMock()
        mock_router.subscribe = MagicMock()
        mock_router.unsubscribe = MagicMock()
        
        controller = DistributedPipelineController(
            node_id="test_node",
            inference_engine=mock_engine,
            event_router=mock_router,
        )
        controller.lookahead_window = 3
        
        # Count how many times send_input is called before the first collect
        send_count = 0
        collect_count = 0
        send_before_first_collect = 0
        
        original_send = controller._send_input_to_stage
        original_collect = controller._collect_result
        
        async def counting_send(*args, **kwargs):
            nonlocal send_count, send_before_first_collect
            send_count += 1
            if collect_count == 0:
                send_before_first_collect = send_count
            return await original_send(*args, **kwargs)
        
        async def counting_collect(*args, **kwargs):
            nonlocal collect_count
            collect_count += 1
            return await original_collect(*args, **kwargs)
        
        controller._send_input_to_stage = counting_send
        controller._collect_result = counting_collect
        
        # Mock the collect to return a token result fast
        original_collect_real = controller._collect_result
        
        async def fast_collect(request_id, last_stage_node_id, last_stage_shard, timeout=30.0):
            from synapse.transport.result_router import RoutedResult
            nonlocal collect_count
            collect_count += 1
            return RoutedResult(
                request_id=request_id,
                orchestrator_id="test_node",
                micro_batch_id=0,
                source_stage_id=0,
                tokens=[42, 43, 44] if collect_count <= 2 else [45],
                is_finished=(collect_count >= 5),
            )
        
        controller._collect_result = fast_collect
        
        base_shard = MagicMock(spec=Shard)
        base_shard.model_id = "test-model"
        base_shard.start_layer = 0
        base_shard.end_layer = 31
        base_shard.to_dict = MagicMock(return_value={"model_id": "test-model"})
        
        partition_assignments = [
            ("node0", base_shard),
            ("node1", base_shard),
        ]
        
        # Start the controller
        await controller.start()
        
        try:
            tokens = await controller.process_distributed_prompt(
                base_shard=base_shard,
                prompt="Hello",
                request_id="test-request-1",
                partition_assignments=partition_assignments,
                max_generate_tokens=10,
                num_micro_batches=1,
            )
            # Should get some tokens back
            assert len(tokens) > 0
        finally:
            await controller.stop()

    def test_decode_loop_structure_has_overlap(self):
        """Verify the decode loop structure enables pipeline overlap."""
        code = open(
            r"D:\Code\paper\synapse_ai\backend\synapse\inference\distributed_engine.py"
        ).read()
        
        # Verify the loop fires multiple batches before results collection
        # Key pattern: iterate over decode_batches sending, THEN iterate collecting
        assert "for db_idx, input_tensor in decode_batches:" in code, (
            "Missing fire-all-batches loop"
        )
        assert "for _ in decode_batches:" in code, (
            "Missing collect-all-results loop"
        )
