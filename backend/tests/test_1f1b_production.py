"""
Test: 1F1B schedule generator wiring — constructor contracts.

The grep/source-inspection tests that previously lived here (scanning
production files for "generate_1f1b_schedule" strings, including hardcoded
absolute paths) were replaced by behavioral tests that DRIVE the real shipped
functions: see test_1f1b_training_production.py, which runs
DistributedTrainer.train_1f1b() through real PipelineStage objects and spies
on the schedule arguments. These two tests keep the light constructor-level
contract checks.
"""
import pytest


class Test1F1BProduction:
    """Tests for 1F1B production wiring."""

    def test_distributed_trainer_has_train_1f1b_method(self):
        """Verify DistributedTrainer has a train_1f1b method."""
        from synapse.pipeline.gradient_sync import DistributedTrainer
        assert hasattr(DistributedTrainer, "train_1f1b")
        assert callable(DistributedTrainer.train_1f1b)

    def test_pipeline_stage_accepts_backward_fn(self):
        """Verify PipelineStage constructor accepts backward_fn parameter."""
        from synapse.pipeline.stage import PipelineStage
        import numpy as np

        async def dummy_fn(mb):
            return np.array([0.0])

        async def dummy_send(r):
            pass

        stage = PipelineStage(
            stage_id=0, node_id="test", num_stages=1,
            forward_fn=dummy_fn, send_tensor_fn=dummy_send,
            backward_fn=dummy_fn,
        )
        assert stage._backward_fn is not None
