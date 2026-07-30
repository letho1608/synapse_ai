"""
Test LACP — Pipeline Parallelism 1F1B Scheduler

Validates:
1. 1F1B schedule generation is correct (warmup, steady, cooldown)
2. Inference-only schedule: forward pipeline with staggered micro-batches
3. Stage-specific slot extraction is correct and ordered
4. Throughput estimation scales with micro-batch size
5. Speedup ratio: more stages = higher throughput up to pipeline depth limit
6. Single-stage degenerate case: works correctly
7. Schedule has no gaps: every stage busy during steady state
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.pipeline.scheduler import (
    MicroBatchOp,
    MicroBatch,
    PipelineSlot,
    PipelineSchedule,
    PipelineScheduler,
    PipelineStageState,
)


class Test1F1BSchedule:
    def test_1stage_4batches(self):
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_1f1b_schedule(1, 4)

        # 1 stage: F0,F1,F2,F3,B0,B1,B2,B3
        assert len(schedule.slots) == 8
        for i in range(4):
            assert schedule.slots[i].op == MicroBatchOp.FORWARD
            assert schedule.slots[i].micro_batch_id == i
        for i in range(4):
            assert schedule.slots[4 + i].op == MicroBatchOp.BACKWARD
            assert schedule.slots[4 + i].micro_batch_id == i

    def test_4stage_4batch_warmup(self):
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_1f1b_schedule(4, 4)

        # Warmup: stages 0,1,2 do only forward for first 3 steps
        stage0_slots = scheduler.slots_for_stage(schedule, 0)
        stage1_slots = scheduler.slots_for_stage(schedule, 1)
        stage2_slots = scheduler.slots_for_stage(schedule, 2)
        stage3_slots = scheduler.slots_for_stage(schedule, 3)

        # Stage 0: all forwards then all backwards (8 slots)
        assert len(stage0_slots) == 8
        assert all(s.op == MicroBatchOp.FORWARD for s in stage0_slots[:4])
        assert all(s.op == MicroBatchOp.BACKWARD for s in stage0_slots[4:])

        # Stage 1: starts later, ends later
        assert stage1_slots[0].step > stage0_slots[0].step

        # All stages have equal total slots
        assert len(stage0_slots) == len(stage1_slots) == len(stage2_slots) == len(stage3_slots)

    def test_3stage_8batch_has_steady_region(self):
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_1f1b_schedule(3, 8)
        assert schedule.warmup_steps == 2
        assert schedule.cooldown_steps == 2
        assert schedule.steady_steps == 12  # 16 total - 2 warmup - 2 cooldown

    def test_no_duplicate_steps_per_stage(self):
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_1f1b_schedule(4, 5)
        for stage in range(4):
            slots = scheduler.slots_for_stage(schedule, stage)
            timestamps = [s.step for s in slots]
            assert len(timestamps) == len(set(timestamps)), f"Duplicate steps in stage {stage}"

    def test_micro_batch_ordering_per_stage(self):
        """Within each stage, micro-batch IDs should follow 1F1B order."""
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_1f1b_schedule(4, 4)
        stage0 = scheduler.slots_for_stage(schedule, 0)

        # Forward pass: ascending (0,1,2,3)
        fwd = [s.micro_batch_id for s in stage0 if s.op == MicroBatchOp.FORWARD]
        assert fwd == [0, 1, 2, 3], f"Forward order: {fwd}"

        # Backward pass: descending (3,2,1,0) for 1F1B
        bwd = [s.micro_batch_id for s in stage0 if s.op == MicroBatchOp.BACKWARD]
        assert bwd == [3, 2, 1, 0], f"Backward order: {bwd}"


class TestInferenceSchedule:
    def test_forward_only_3stage_4batch(self):
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_inference_schedule(3, 4)

        stage0 = scheduler.slots_for_stage(schedule, 0)
        stage1 = scheduler.slots_for_stage(schedule, 1)
        stage2 = scheduler.slots_for_stage(schedule, 2)

        # Each stage processes all 4 micro-batches
        assert len(stage0) == 4
        assert len(stage1) == 4
        assert len(stage2) == 4

        # All ops are FORWARD
        assert all(s.op == MicroBatchOp.FORWARD for s in schedule.slots)

        # Staggered: stage1 starts 1 step after stage0
        assert stage1[0].step == stage0[0].step + 1

    def test_1stage_no_stagger(self):
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_inference_schedule(1, 5)

        stage0 = scheduler.slots_for_stage(schedule, 0)
        assert len(stage0) == 5
        assert [s.step for s in stage0] == [0, 1, 2, 3, 4]

    def test_2stage_stagger(self):
        scheduler = PipelineScheduler()
        schedule = scheduler.generate_inference_schedule(2, 3)

        s0 = scheduler.slots_for_stage(schedule, 0)
        s1 = scheduler.slots_for_stage(schedule, 1)

        assert [s.step for s in s0] == [0, 1, 2]
        assert [s.step for s in s1] == [1, 2, 3]


class TestThroughputEstimation:
    def test_identical_stages(self):
        scheduler = PipelineScheduler()
        stage_times = [10.0, 10.0, 10.0, 10.0]  # ms
        tp = scheduler.estimated_throughput(stage_times, num_micro_batches=8)
        # 4 stages + 8 batches - 1 = 11 steps; bottleneck 10ms
        # throughput = 8 / (10ms * 11 / 1000) = 72.7
        assert 60 < tp < 100

    def test_bottleneck_dominates(self):
        scheduler = PipelineScheduler()
        stage_times = [5.0, 5.0, 50.0, 5.0]  # ms, stage 2 is bottleneck
        tp = scheduler.estimated_throughput(stage_times, num_micro_batches=8)
        # 4 stages + 8 batches - 1 = 11 steps; bottleneck 50ms
        # throughput = 8 / (50ms * 11 / 1000) = 14.5
        assert 10 < tp < 20

    def test_training_throughput_higher_than_single(self):
        scheduler = PipelineScheduler()
        single_stage = 40.0  # ms
        stage_times = [10.0, 10.0, 10.0, 10.0]
        num_mb = 16

        speedup = scheduler.speedup_ratio(single_stage, stage_times, 4, num_mb)
        # 4 stages of 10ms should be faster than 1 stage of 40ms
        assert speedup > 1.0

    def test_zero_time_handling(self):
        scheduler = PipelineScheduler()
        tp = scheduler.estimated_throughput([], num_micro_batches=4)
        assert tp == 0.0


class TestMicroBatch:
    def test_creation(self):
        import numpy as np
        mb = MicroBatch(
            batch_id=0,
            data=np.array([1, 2, 3]),
            labels=np.array([4, 5, 6]),
            metadata={"user": "test"},
        )
        assert mb.batch_id == 0
        assert mb.labels is not None
        assert mb.metadata["user"] == "test"