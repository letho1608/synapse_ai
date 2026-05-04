"""
LACP 3.0 — Pipeline Parallelism Scheduler (1F1B / GPipe)

Implements the One-Forward-One-Backward (1F1B) schedule for pipeline
parallelism across multiple nodes. This eliminates pipeline bubbles
after warmup, keeping all nodes busy.

Schedule visualization (3 stages, 4 micro-batches):

  Time -->
  S0: [F0][F1][F2][F3][B0][B1][B2][B3]
  S1:     [F0][F1][F2][F3][B0][B1][B2][B3]
  S2:         [F0][F1][F2][F3][B0][B1][B2][B3]

  Fx = forward pass for micro-batch x
  Bx = backward pass for micro-batch x

For inference-only (no backward), this becomes a simple forward pipeline
with micro-batch overlapping:

  Time -->
  S0: [F0][F1][F2][F3]
  S1:     [F0][F1][F2][F3]
  S2:         [F0][F1][F2][F3]

  After warmup of (stages-1) steps, throughput = 1 / max(stage_time).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Awaitable, Any, Tuple
import numpy as np


class MicroBatchOp(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass
class MicroBatch:
    """A single micro-batch flowing through the pipeline."""
    batch_id: int
    data: np.ndarray  # input token IDs or embeddings
    labels: Optional[np.ndarray] = None  # for training
    attention_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSlot:
    """A single operation slot in the pipeline schedule."""
    step: int           # global step number
    stage: int           # pipeline stage index (0-based)
    op: MicroBatchOp     # FORWARD or BACKWARD
    micro_batch_id: int  # which micro-batch
    node_id: str         # which node runs this


@dataclass
class PipelineSchedule:
    """Complete 1F1B schedule for a training/inference step."""
    slots: List[PipelineSlot] = field(default_factory=list)
    num_stages: int = 0
    num_micro_batches: int = 0
    warmup_steps: int = 0
    steady_steps: int = 0
    cooldown_steps: int = 0


class PipelineScheduler:
    """
    Generates 1F1B schedules and orchestrates micro-batch flow through
    the pipeline stages.
    """

    def __init__(self, num_micro_batches: int = 4):
        self.num_micro_batches = num_micro_batches
        self._running = False

    def generate_1f1b_schedule(
        self,
        num_stages: int,
        num_micro_batches: int,
        node_ids: Optional[List[str]] = None,
    ) -> PipelineSchedule:
        """
        Generate a 1F1B schedule for GPipe-style pipeline parallelism.

        The schedule structure:
        - Warmup: (num_stages - 1) forward passes to fill the pipeline
        - Steady: num_micro_batches * 2 - (num_stages - 1) ops alternating F/B
        - Cooldown: (num_stages - 1) backward passes to drain
        """
        schedule = PipelineSchedule(
            num_stages=num_stages,
            num_micro_batches=num_micro_batches,
        )

        if num_stages <= 1:
            # Single stage: just sequential F then B
            for mb in range(num_micro_batches):
                schedule.slots.append(PipelineSlot(
                    step=mb, stage=0, op=MicroBatchOp.FORWARD,
                    micro_batch_id=mb, node_id="stage-0"
                ))
            for mb in range(num_micro_batches):
                schedule.slots.append(PipelineSlot(
                    step=num_micro_batches + mb, stage=0, op=MicroBatchOp.BACKWARD,
                    micro_batch_id=mb, node_id="stage-0"
                ))
            schedule.warmup_steps = num_micro_batches
            schedule.steady_steps = 0
            schedule.cooldown_steps = num_micro_batches
            return schedule

        warmup = num_stages - 1
        cooldown = num_stages - 1

        # Each stage has a different schedule
        for stage in range(num_stages):
            nid = node_ids[stage] if node_ids and stage < len(node_ids) else f"stage-{stage}"

            for mb in range(num_micro_batches):
                step = stage + mb
                schedule.slots.append(PipelineSlot(
                    step=step, stage=stage, op=MicroBatchOp.FORWARD,
                    micro_batch_id=mb, node_id=nid
                ))

            for i, mb in enumerate(range(num_micro_batches - 1, -1, -1)):
                step = stage + num_micro_batches + i
                schedule.slots.append(PipelineSlot(
                    step=step, stage=stage, op=MicroBatchOp.BACKWARD,
                    micro_batch_id=mb, node_id=nid
                ))

        schedule.warmup_steps = warmup
        schedule.steady_steps = num_micro_batches * 2 - warmup * 2
        schedule.cooldown_steps = cooldown

        return schedule

    def generate_inference_schedule(
        self,
        num_stages: int,
        num_micro_batches: int,
        node_ids: Optional[List[str]] = None,
    ) -> PipelineSchedule:
        """
        Simplified forward-only pipeline for inference.
        Each stage processes micro-batches in a staggered fashion.
        """
        schedule = PipelineSchedule(
            num_stages=num_stages,
            num_micro_batches=num_micro_batches,
        )

        for stage in range(num_stages):
            nid = node_ids[stage] if node_ids and stage < len(node_ids) else f"stage-{stage}"
            for mb in range(num_micro_batches):
                step = stage + mb
                schedule.slots.append(PipelineSlot(
                    step=step, stage=stage, op=MicroBatchOp.FORWARD,
                    micro_batch_id=mb, node_id=nid
                ))

        warmup = num_stages - 1
        cooldown = num_stages - 1
        schedule.warmup_steps = warmup if warmup > 0 else 0
        schedule.steady_steps = num_micro_batches - warmup if num_micro_batches > warmup else 0
        schedule.cooldown_steps = cooldown if cooldown > 0 else 0

        return schedule

    def slots_for_stage(
        self, schedule: PipelineSchedule, stage_id: int
    ) -> List[PipelineSlot]:
        """Get all slots for a specific stage, sorted by step."""
        return sorted(
            [s for s in schedule.slots if s.stage == stage_id],
            key=lambda s: s.step
        )

    def estimated_throughput(
        self,
        per_stage_times_ms: List[float],
        num_micro_batches: int,
        is_training: bool = False,
    ) -> float:
        """
        Estimate pipeline throughput (samples/sec).

        Total time = bottleneck_ms * (num_stages + num_micro_batches - 1)
        Throughput = num_micro_batches / total_time_seconds
        """
        if not per_stage_times_ms:
            return 0.0

        bottleneck_ms = max(per_stage_times_ms)
        num_stages = len(per_stage_times_ms)

        total_steps = num_stages + num_micro_batches - 1
        total_time_s = (bottleneck_ms * total_steps) / 1000.0

        if total_time_s <= 0:
            return 0.0

        return num_micro_batches / total_time_s

    @staticmethod
    def speedup_ratio(
        single_stage_time_ms: float,
        per_stage_times_ms: List[float],
        num_stages: int,
        num_micro_batches: int,
    ) -> float:
        """
        Compute speedup ratio vs single-node execution.

        single_node_time = num_micro_batches * single_stage_time_ms
        pipeline_time = bottleneck_ms * (num_stages + num_micro_batches - 1)
        speedup = single_node_time / pipeline_time
        """
        if num_stages <= 1:
            return 1.0

        bottleneck_ms = max(per_stage_times_ms)
        total_steps = num_stages + num_micro_batches - 1
        single_time = num_micro_batches * single_stage_time_ms
        pipeline_time = bottleneck_ms * total_steps

        if pipeline_time <= 0:
            return 0.0

        return single_time / pipeline_time


@dataclass
class PipelineStageState:
    """Runtime state for a single pipeline stage."""
    stage: PipelineStage
    current_step: int = 0
    pending_input: Optional[np.ndarray] = None
    pending_output: Optional[np.ndarray] = None
    completed_batches: int = 0
    total_time_ms: float = 0.0
    batches_per_second: float = 0.0