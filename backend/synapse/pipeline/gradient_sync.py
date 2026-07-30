"""
LACP - Distributed Gradient Aggregation (All-Reduce) & DDP

Ring all-reduce for synchronizing gradients across N nodes.
Each node has a full model copy (DDP style).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable, Any
import numpy as np

from synapse.pipeline.scheduler import (
    MicroBatch,
    MicroBatchOp,
    PipelineSchedule,
    PipelineScheduler,
)
from synapse.pipeline.stage import PipelineStage


@dataclass
class GradientShard:
    node_id: str
    layer_range: tuple[int, int]
    gradients: Dict[str, np.ndarray]
    step: int = 0
    loss: float = 0.0

    def total_bytes(self) -> int:
        return sum(g.nbytes for g in self.gradients.values())


class RingAllReducer:

    def __init__(self, num_nodes: int, ring_order: Optional[List[str]] = None):
        self.num_nodes = num_nodes
        self.ring_order = ring_order or []
        self._accumulated: Dict[str, Dict[str, np.ndarray]] = {}

    def get_predecessor(self, node_id: str) -> Optional[str]:
        if not self.ring_order or node_id not in self.ring_order:
            return None
        idx = self.ring_order.index(node_id)
        return self.ring_order[(idx - 1) % len(self.ring_order)]

    def get_successor(self, node_id: str) -> Optional[str]:
        if not self.ring_order or node_id not in self.ring_order:
            return None
        idx = self.ring_order.index(node_id)
        return self.ring_order[(idx + 1) % len(self.ring_order)]

    def accumulate(self, node_id: str, gradients: Dict[str, np.ndarray]):
        if node_id not in self._accumulated:
            self._accumulated[node_id] = {}
        for param, grad in gradients.items():
            if param in self._accumulated[node_id]:
                self._accumulated[node_id][param] += grad
            else:
                self._accumulated[node_id][param] = grad.copy()

    def get_accumulated(self, node_id: str) -> Dict[str, np.ndarray]:
        return self._accumulated.get(node_id, {})

    def reset(self, node_id: str):
        self._accumulated.pop(node_id, None)


class DistributedTrainer:
    """
    DDP-style distributed training with ring all-reduce.

    Each node:
      1. Computes local gradients on its full model copy
      2. Ring all-reduce to sum gradients across all nodes
      3. Each node updates weights with globally summed & scaled gradients
    """

    def __init__(
        self,
        node_id: str,
        world_size: int,
        ring_order: Optional[List[str]] = None,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
        quantize_grads: bool = True,
    ):
        self.node_id = node_id
        self.world_size = world_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.quantize_grads = quantize_grads
        self.reducer = RingAllReducer(world_size, ring_order=ring_order)
        self._train_step: int = 0
        self._loss_history: List[float] = []

    def _quantize_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Quantize gradients to FP16 to save 50% bandwidth."""
        return {k: v.astype(np.float16) for k, v in gradients.items()}

    async def step(
        self,
        local_gradients: Dict[str, np.ndarray],
        local_loss: float,
        send_fn: Optional[Callable[[str, Dict[str, np.ndarray]], Awaitable[None]]] = None,
        recv_fn: Optional[Callable[[str], Awaitable[Dict[str, np.ndarray]]]] = None,
    ) -> Dict[str, np.ndarray]:
        self._train_step += 1

        self.reducer.reset(self.node_id)
        
        # Prepare gradients for ring transfer
        current_gradients = local_gradients
        if self.quantize_grads:
            current_gradients = self._quantize_gradients(current_gradients)
            
        self.reducer.accumulate(self.node_id, current_gradients)

        predecessor = self.reducer.get_predecessor(self.node_id)
        successor = self.reducer.get_successor(self.node_id)

        for _ in range(max(1, self.world_size) - 1):
            if send_fn and successor:
                # Fire and forget or await? Usually await to ensure sequential ring flow
                await send_fn(successor, current_gradients)
            if recv_fn and predecessor:
                received = await recv_fn(predecessor)
                self.reducer.accumulate(self.node_id, received)
                current_gradients = received

        reduced = self.reducer.get_accumulated(self.node_id)

        # Gradient clipping
        total_norm = 0.0
        for grad in reduced.values():
            total_norm += np.sum(grad.astype(np.float32) ** 2)
        total_norm = np.sqrt(total_norm)

        scale = self.max_grad_norm / max(total_norm, self.max_grad_norm)
        if scale < 1.0:
            reduced = {k: (v.astype(np.float32) * scale) for k, v in reduced.items()}
        else:
            reduced = {k: v.astype(np.float32) for k, v in reduced.items()}

        updates = {}
        for param, grad in reduced.items():
            updates[param] = -self.learning_rate * (grad / self.world_size)

        self._loss_history.append(local_loss)
        if len(self._loss_history) > 100:
            self._loss_history.pop(0)

        return updates

    @property
    def average_loss(self) -> float:
        if not self._loss_history:
            return 0.0
        return sum(self._loss_history) / len(self._loss_history)

    async def train_1f1b(
        self,
        stages: List[PipelineStage],
        num_micro_batches: int,
        input_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Execute a training step using 1F1B pipeline schedule.
        
        Uses PipelineScheduler.generate_1f1b_schedule() to coordinate 
        forward and backward passes across all pipeline stages.
        
        Args:
            stages: List of PipelineStage objects, one per node.
            num_micro_batches: Number of micro-batches for the pipeline.
            input_data: Input tensor for training.
            labels: Optional target labels for loss computation.
            
        Returns:
            List of loss values per micro-batch.
        """
        scheduler = PipelineScheduler(num_micro_batches=num_micro_batches)
        num_stages = len(stages)
        node_ids = [s.node_id for s in stages]
        
        schedule = scheduler.generate_1f1b_schedule(num_stages, num_micro_batches, node_ids=node_ids)
        
        # Build per-stage slot map
        stage_slots = scheduler.slots_for_stage(schedule, 0)
        for s in stages:
            if s.stage_id > 0:
                stage_slots = scheduler.slots_for_stage(schedule, s.stage_id)
        
        # Execute pipeline
        stop_events = {s.stage_id: asyncio.Event() for s in stages}
        loop_tasks = [
            asyncio.create_task(s.process_loop(stop_events[s.stage_id]))
            for s in stages
        ]
        
        losses = []
        try:
            for slot in sorted(schedule.slots, key=lambda s: s.step):
                if slot.stage == 0:
                    mb_data = input_data.copy() if slot.micro_batch_id > 0 else input_data
                    op_name = "backward" if slot.op == MicroBatchOp.BACKWARD else "forward"
                    mb = MicroBatch(
                        batch_id=slot.micro_batch_id,
                        data=mb_data,
                        labels=labels,
                        metadata={"op": op_name, "step": slot.step},
                    )
                    await stages[0].enqueue_input(mb)
            
            # Collect results from last stage
            num_ops = len([s for s in schedule.slots if s.stage == num_stages - 1])
            for _ in range(num_ops):
                result = await stages[-1].dequeue_output(timeout=30.0)
                if result is not None:
                    losses.append(float(np.mean(result.data)) if hasattr(result.data, "__iter__") else 0.0)
        finally:
            for evt in stop_events.values():
                evt.set()
            for task in loop_tasks:
                task.cancel()
            await asyncio.gather(*loop_tasks, return_exceptions=True)
        
        return losses