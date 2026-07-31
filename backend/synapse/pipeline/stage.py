"""
LACP — Pipeline Stage Executor

Executes forward passes on a node's assigned layer range with:
- Micro-batch processing from 1F1B schedule
- KV cache persistence per request
- Async tensor forwarding to next stage
- Graceful degradation on failure
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from synapse.pipeline.scheduler import (
    MicroBatchOp,
    MicroBatch,
    PipelineSlot,
    PipelineSchedule,
    PipelineScheduler,
)


@dataclass
class StageResult:
    micro_batch_id: int
    data: np.ndarray
    step: int
    elapsed_ms: float
    # "forward" or "backward" — lets the pipeline send path distinguish the
    # backward phase's gradient feedback from forward activations/losses.
    op: str = "forward"


@dataclass
class StageStats:
    total_forwards: int = 0
    total_backwards: int = 0
    total_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    failures: int = 0

    @property
    def avg_latency_ms(self) -> float:
        total = self.total_forwards + self.total_backwards
        return self.total_latency_ms / total if total > 0 else 0.0


class PipelineStage:
    """
    A single pipeline stage that executes inference on a contiguous layer range.

    Each stage:
    1. Receives micro-batches from the previous stage (or tokenizer)
    2. Runs forward/backward pass on its assigned layer range
    3. Sends results to the next stage via async forwarder
    """

    def __init__(
        self,
        stage_id: int,
        node_id: str,
        num_stages: int,
        forward_fn: Callable,
        send_tensor_fn: Callable,
        recv_tensor_fn: Optional[Callable] = None,
        backward_fn: Optional[Callable] = None,
    ):
        self.stage_id = stage_id
        self.node_id = node_id
        self.num_stages = num_stages
        self._forward_fn = forward_fn
        self._send_tensor_fn = send_tensor_fn
        self._recv_tensor_fn = recv_tensor_fn
        self._backward_fn = backward_fn

        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1

        self._input_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._output_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._done: Dict[int, asyncio.Event] = {}

        self.stats = StageStats()

    async def enqueue_input(self, mb: MicroBatch) -> None:
        await self._input_queue.put(mb)

    async def dequeue_output(self, timeout: float = 30.0) -> StageResult:
        return await asyncio.wait_for(self._output_queue.get(), timeout=timeout)

    async def run_forward(self, micro_batch: MicroBatch) -> StageResult:
        start = time.perf_counter()
        try:
            result_data = await self._forward_fn(micro_batch)
        except Exception:
            self.stats.failures += 1
            raise

        elapsed = (time.perf_counter() - start) * 1000
        self.stats.total_forwards += 1
        self.stats.total_latency_ms += elapsed

        result = StageResult(
            micro_batch_id=micro_batch.batch_id,
            data=result_data,
            step=micro_batch.step if hasattr(micro_batch, "step") else 0,
            elapsed_ms=elapsed,
        )
        return result

    async def process_loop(self, stop_event: asyncio.Event) -> None:
        """Process both forward and backward passes as scheduled.

        Graceful drain: once stop is requested the stage still consumes every
        slot still queued before exiting — the 1F1B backward chain unwinds
        upstream AFTER the last stage produces its final result, so a
        hard loop-top exit would drop the last backward slots."""
        while True:
            if stop_event.is_set() and self._input_queue.empty():
                break
            try:
                mb = await asyncio.wait_for(self._input_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            try:
                op = mb.metadata.get("op", "forward") if mb.metadata else "forward"
                if op == "backward" and self._backward_fn is not None:
                    result = await self.run_backward(mb)
                else:
                    result = await self.run_forward(mb)

                if not self.is_last_stage:
                    await self._send_tensor_fn(result)
                else:
                    await self._output_queue.put(result)
            except Exception:
                continue

    async def forward(self, input_data: np.ndarray, micro_batch_id: int = 0, metadata: Optional[Dict] = None) -> np.ndarray:
        mb = MicroBatch(batch_id=micro_batch_id, data=input_data, metadata=metadata or {})
        result = await self.run_forward(mb)
        return result.data

    async def run_backward(self, micro_batch: MicroBatch) -> StageResult:
        """Execute a backward pass on this stage's layer range."""
        if self._backward_fn is None:
            raise RuntimeError("No backward function configured for this stage")
        start = time.perf_counter()
        try:
            result_data = await self._backward_fn(micro_batch)
        except Exception:
            self.stats.failures += 1
            raise

        elapsed = (time.perf_counter() - start) * 1000
        self.stats.total_backwards += 1
        self.stats.total_latency_ms += elapsed

        result = StageResult(
            micro_batch_id=micro_batch.batch_id,
            data=result_data,
            step=micro_batch.step if hasattr(micro_batch, "step") else 0,
            elapsed_ms=elapsed,
            op="backward",
        )
        return result


class PipelineExecutor:
    """
    Orchestrates a full pipeline execution across multiple stages/nodes.

    Usage:
        executor = PipelineExecutor(stages=[stage0, stage1, stage2],
                                     scheduler=PipelineScheduler())
        results = await executor.run_inference(input_tensor, num_micro_batches=4)
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        scheduler: Optional[PipelineScheduler] = None,
        timeout_per_step: float = 30.0,
    ):
        self.stages = stages
        self.scheduler = scheduler or PipelineScheduler()
        self.timeout_per_step = timeout_per_step
        self.num_stages = len(stages)

    async def run_inference(
        self,
        input_data: np.ndarray,
        num_micro_batches: int = 1,
    ) -> List[np.ndarray]:
        """
        Run forward-only pipeline inference.

        Args:
            input_data: Initial input tensor (prompt embedding or first token)
            num_micro_batches: Number of micro-batches to process

        Returns:
            List of result tensors from the last stage, one per micro-batch
        """
        if self.num_stages == 0:
            return [input_data]

        schedule = self.scheduler.generate_inference_schedule(
            self.num_stages, num_micro_batches
        )

        results: List[Optional[np.ndarray]] = [None] * num_micro_batches

        if self.num_stages == 1:
            stage = self.stages[0]
            for mb_id in range(num_micro_batches):
                mb = MicroBatch(
                    batch_id=mb_id,
                    data=input_data.copy() if mb_id > 0 else input_data,
                    metadata={"step": mb_id},
                )
                result = await stage.run_forward(mb)
                results[mb_id] = result.data
            return [r for r in results if r is not None]

        stop_events = {s.stage_id: asyncio.Event() for s in self.stages}
        loop_tasks = [
            asyncio.create_task(s.process_loop(stop_events[s.stage_id]))
            for s in self.stages
        ]

        try:
            stage0_slots = self.scheduler.slots_for_stage(schedule, 0)

            for slot in sorted(stage0_slots, key=lambda s: s.step):
                mb = MicroBatch(
                    batch_id=slot.micro_batch_id,
                    data=input_data.copy(),
                    metadata={"step": slot.step},
                )
                await self.stages[0].enqueue_input(mb)

            for mb_id in range(num_micro_batches):
                result = await self.stages[-1].dequeue_output(
                    timeout=self.timeout_per_step
                )
                results[result.micro_batch_id] = result.data

        finally:
            for evt in stop_events.values():
                evt.set()
            for task in loop_tasks:
                task.cancel()
            await asyncio.gather(*loop_tasks, return_exceptions=True)

        return [r for r in results if r is not None]

    @staticmethod
    async def run_pipeline_on_node(
        stage: PipelineStage,
        input_data: np.ndarray,
        num_micro_batches: int = 1,
        timeout: float = 300.0,
    ) -> List[StageResult]:
        results: List[StageResult] = []
        for mb_id in range(num_micro_batches):
            mb = MicroBatch(
                batch_id=mb_id,
                data=input_data.copy() if mb_id > 0 else input_data,
                metadata={"step": mb_id},
            )
            result = await asyncio.wait_for(
                stage.run_forward(mb), timeout=timeout / num_micro_batches
            )
            results.append(result)
        return results