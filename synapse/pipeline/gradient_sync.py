"""
LACP 3.0 — Distributed Gradient Aggregation (All-Reduce) & DDP

Implements distributed data-parallel gradient synchronization for training
across pipeline stages. Uses ring all-reduce for efficiency over N nodes.

Protocol:
  1. Each node computes local gradients on its shard
  2. Nodes perform ring all-reduce to sum gradients across all shards
  3. Each node updates its local weights with the globally summed gradients
  4. Gradient scaling (1/N) ensures correct update magnitude

The all-reduce runs asynchronously — nodes start the next forward pass
while gradients are being reduced in the background.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable
import numpy as np


@dataclass
class GradientShard:
    """A shard of gradients to be reduced."""
    node_id: str
    layer_range: tuple[int, int]  # (start_layer, end_layer)
    gradients: Dict[str, np.ndarray]  # param_name -> gradient tensor
    step: int = 0
    loss: float = 0.0

    def total_bytes(self) -> int:
        return sum(g.nbytes for g in self.gradients.values())


@dataclass
class ReducedGradients:
    """Result of all-reduce: summed gradients from all nodes."""
    gradients: Dict[str, np.ndarray] = field(default_factory=dict)
    total_loss: float = 0.0
    num_contributors: int = 0
    step: int = 0


class RingAllReducer:
    """
    Ring all-reduce implementation specialized for pipeline parallelism.

    In a pipeline of N stages, each stage computes gradients for its shard.
    Ring all-reduce passes gradients through the ring (N-1) times,
    each node accumulating from its predecessor and forwarding.
    After (N-1) steps, every node has the FULL summed gradients.

    For inference-only mode, this reducer is a no-op.
    """

    def __init__(self, num_nodes: int, ring_order: Optional[List[str]] = None):
        """
        Args:
            num_nodes: Total number of nodes in the ring.
            ring_order: Optional explicit node ordering for the ring.
                        Default: sorted by node_id.
        """
        self.num_nodes = num_nodes
        self.ring_order = ring_order or []

        # Per-node accumulated gradients
        self._accumulated: Dict[str, Dict[str, np.ndarray]] = {}

    def get_predecessor(self, node_id: str) -> Optional[str]:
        """Get the predecessor in the ring."""
        if not self.ring_order or node_id not in self.ring_order:
            return None
        idx = self.ring_order.index(node_id)
        return self.ring_order[(idx - 1) % len(self.ring_order)]

    def get_successor(self, node_id: str) -> Optional[str]:
        """Get the successor in the ring."""
        if not self.ring_order or node_id not in self.ring_order:
            return None
        idx = self.ring_order.index(node_id)
        return self.ring_order[(idx + 1) % len(self.ring_order)]

    def accumulate(self, node_id: str, gradients: Dict[str, np.ndarray]):
        """Accumulate incoming gradients from predecessor into local buffer."""
        if node_id not in self._accumulated:
            self._accumulated[node_id] = {}

        for param, grad in gradients.items():
            if param in self._accumulated[node_id]:
                self._accumulated[node_id][param] += grad
            else:
                self._accumulated[node_id][param] = grad.copy()

    def get_accumulated(self, node_id: str) -> Dict[str, np.ndarray]:
        """Get the fully accumulated gradients for a node."""
        return self._accumulated.get(node_id, {})

    def reset(self, node_id: str):
        self._accumulated.pop(node_id, None)

    def scale_for_update(
        self, gradients: Dict[str, np.ndarray], world_size: int
    ) -> Dict[str, np.ndarray]:
        """Scale gradients by 1/world_size for correct SGD update."""
        return {k: v / world_size for k, v in gradients.items()}


class DistributedTrainer:
    """
    Coordinates distributed training across pipeline stages.

    Training loop:
      1. Forward pass through pipeline (1F1B schedule)
      2. Each node computes local gradients on its shard
      3. Ring all-reduce to aggregate gradients globally
      4. Each node updates weights with scaled global gradients
      5. Repeat for next step
    """

    def __init__(
        self,
        node_id: str,
        world_size: int,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.node_id = node_id
        self.world_size = world_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self.reducer = RingAllReducer(world_size)
        self._train_step: int = 0
        self._total_loss: float = 0.0
        self._loss_history: List[float] = []

    async def step(
        self,
        local_gradients: Dict[str, np.ndarray],
        local_loss: float,
        send_fn: Optional[Callable[[str, Dict[str, np.ndarray]], Awaitable[None]]] = None,
        recv_fn: Optional[Callable[[], Awaitable[Dict[str, np.ndarray]]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perform one training step: all-reduce gradients and compute weight updates.

        Args:
            local_gradients: Gradients computed locally on this node's shard.
            local_loss: Loss value from this node's shard.
            send_fn: Async function to send gradients to the successor node.
            recv_fn: Async function to receive gradients from the predecessor.

        Returns:
            Dict of parameter name -> weight update values (already scaled).
        """
        self._train_step += 1

        # Phase 1: Initialize accumulation with own gradients
        self.reducer.reset(self.node_id)
        self.reducer.accumulate(self.node_id, local_gradients)

        # Phase 2: Ring all-reduce — pass gradients around ring (N-1) times
        current_gradients = local_gradients
        predecessor = self.reducer.get_predecessor(self.node_id)
        successor = self.reducer.get_successor(self.node_id)

        for _ in range(self.world_size - 1):
            if send_fn and successor:
                # Send our current accumulated to successor
                await send_fn(successor, current_gradients)

            if recv_fn and predecessor:
                # Receive from predecessor
                received = await recv_fn(predecessor)
                self.reducer.accumulate(self.node_id, received)
                current_gradients = received

        # Phase 3: Get fully reduced gradients
        reduced = self.reducer.get_accumulated(self.node_id)

        # Phase 4: Gradient clipping
        total_norm = 0.0
        for grad in reduced.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        scale = self.max_grad_norm / max(total_norm, self.max_grad_norm)
        if scale < 1.0:
            reduced = {k: v * scale for k, v in reduced.items()}

        # Phase 5: Scale by 1/world_size and apply learning rate
        updates = {}
        for param, grad in reduced.items():
            updates[param] = -self.learning_rate * (grad / self.world_size)

        self._total_loss += local_loss
        self._loss_history.append(local_loss)

        if len(self._loss_history) > 100:
            self._loss_history.pop(0)

        return updates

    @property
    def average_loss(self) -> float:
        if not self._loss_history:
            return 0.0
        return sum(self._loss_history) / len(self._loss_history)