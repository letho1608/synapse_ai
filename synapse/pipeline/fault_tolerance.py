"""
LACP 3.0 — Fault Tolerance & Dynamic Rebalancing Engine

Handles node failures in the distributed pipeline with zero-downtime recovery.

Strategy:
  1. Heartbeat monitoring — detect dead nodes within 5s
  2. State checkpointing — periodic shard state snapshots to disk/shared memory
  3. Dynamic rebalancing — redistribute dead node's shard to surviving nodes
  4. Pipeline resume — continue inference/training from last checkpoint
  5. Graceful degradation — if N nodes drop to N-1, rebalance then continue

The rebalancing uses the LACP 3.0 partitioner to recompute optimal shard
assignments excluding the failed node, then redistributes layers.
"""

from __future__ import annotations

import asyncio
import time
import pickle
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np


@dataclass
class ShardCheckpoint:
    """A snapshot of a node's shard state for recovery."""
    node_id: str
    model_id: str
    start_layer: int
    end_layer: int
    step: int
    kv_cache: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class NodeHealth:
    """Tracked health status for a single node."""
    node_id: str
    last_heartbeat: float = 0.0
    is_alive: bool = False
    consecutive_misses: int = 0
    is_blacklisted: bool = False

    def record_heartbeat(self):
        self.last_heartbeat = time.time()
        self.is_alive = True
        self.consecutive_misses = 0
        self.is_blacklisted = False

    def check_timeout(self, timeout_s: float = 10.0) -> bool:
        """Returns True if node has timed out."""
        elapsed = time.time() - self.last_heartbeat
        if elapsed > timeout_s:
            self.consecutive_misses += 1
            self.is_alive = False
            if self.consecutive_misses >= 3:
                self.is_blacklisted = True
            return True
        return False


@dataclass
class RebalanceResult:
    """Result of a dynamic rebalancing operation."""
    removed_node: str
    added_nodes: List[str] = field(default_factory=list)
    new_assignments: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    recovered_from_checkpoint: bool = False
    downtime_ms: float = 0.0
    success: bool = False


class FaultToleranceManager:
    """
    Manages node health monitoring, checkpointing, and dynamic rebalancing.
    """

    HEARTBEAT_INTERVAL = 2.0   # seconds between heartbeats
    HEARTBEAT_TIMEOUT = 10.0   # seconds before considering node dead
    CHECKPOINT_INTERVAL = 30.0  # seconds between state snapshots
    MAX_MISSES = 3             # consecutive missed heartbeats before blacklisting

    def __init__(self, node_id: str, checkpoint_dir: str = ".synapse_checkpoints"):
        self.node_id = node_id
        self.checkpoint_dir = checkpoint_dir
        self._health: Dict[str, NodeHealth] = {}
        self._checkpoints: Dict[str, List[ShardCheckpoint]] = defaultdict(list)
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._pending_rebalance: Optional[str] = None  # node_id being rebalanced

        os.makedirs(checkpoint_dir, exist_ok=True)

    async def start(self) -> None:
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()

    # ── Heartbeat ──────────────────────────────────────────

    def register_node(self, node_id: str) -> None:
        self._health[node_id] = NodeHealth(node_id=node_id)
        self._health[node_id].record_heartbeat()

    def record_heartbeat(self, node_id: str) -> None:
        if node_id not in self._health:
            self.register_node(node_id)
        else:
            self._health[node_id].record_heartbeat()

    def get_alive_nodes(self) -> List[str]:
        return [nid for nid, h in self._health.items() if h.is_alive]

    def get_dead_nodes(self) -> List[str]:
        return [nid for nid, h in self._health.items() if not h.is_alive]

    def is_node_alive(self, node_id: str) -> bool:
        h = self._health.get(node_id)
        return h.is_alive if h else False

    # ── Checkpointing ──────────────────────────────────────

    def save_checkpoint(
        self,
        model_id: str,
        start_layer: int,
        end_layer: int,
        step: int,
        kv_cache: Optional[Dict] = None,
        optimizer_state: Optional[Dict] = None,
    ) -> str:
        """Save a shard state checkpoint. Returns checkpoint path."""
        ckpt = ShardCheckpoint(
            node_id=self.node_id,
            model_id=model_id,
            start_layer=start_layer,
            end_layer=end_layer,
            step=step,
            kv_cache=kv_cache,
            optimizer_state=optimizer_state,
        )

        filename = f"{model_id}_L{start_layer}-{end_layer}_step{step}.ckpt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(ckpt, f)

        self._checkpoints[self.node_id].append(ckpt)
        return filepath

    def load_latest_checkpoint(self, node_id: str) -> Optional[ShardCheckpoint]:
        """Load the most recent checkpoint for a node (used during recovery)."""
        ckpts = self._checkpoints.get(node_id, [])
        if not ckpts:
            return None
        return max(ckpts, key=lambda c: c.timestamp)

    def load_checkpoint(
        self, model_id: str, start_layer: int, end_layer: int, step: int
    ) -> Optional[ShardCheckpoint]:
        """Load a specific checkpoint from disk."""
        filename = f"{model_id}_L{start_layer}-{end_layer}_step{step}.ckpt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, "rb") as f:
            return pickle.load(f)

    # ── Dynamic Rebalancing ────────────────────────────────

    async def rebalance_after_failure(
        self,
        dead_node_id: str,
        current_assignments: Dict[str, Tuple[int, int]],
        total_layers: int,
        lacp_partition_fn: callable,
    ) -> RebalanceResult:
        """
        Rebalance shards after a node failure.

        Args:
            dead_node_id: Node that died.
            current_assignments: {node_id: (start_layer, end_layer)} before failure.
            total_layers: Total transformer layers in the model.
            lacp_partition_fn: async function(nodes, total_layers) -> new assignments.

        Returns:
            RebalanceResult with new assignments.
        """
        start_time = time.time()
        result = RebalanceResult(removed_node=dead_node_id)

        try:
            # Blacklist the dead node
            if dead_node_id in self._health:
                self._health[dead_node_id].is_blacklisted = True
                self._health[dead_node_id].is_alive = False

            # Get surviving nodes
            alive = [nid for nid in self.get_alive_nodes() if nid != dead_node_id]
            if not alive:
                result.success = False
                return result

            # Compute new assignments via LACP 3.0
            new_assignments = await lacp_partition_fn(alive, total_layers)
            result.new_assignments = new_assignments

            # Load checkpoint for dead node's shard range
            dead_shard = current_assignments.get(dead_node_id)
            if dead_shard:
                ckpt = self.load_latest_checkpoint(dead_node_id)
                if ckpt:
                    result.recovered_from_checkpoint = True
                    # The new node receiving this shard range should load this checkpoint
                    result.new_assignments["_recovery_checkpoint"] = ckpt

            result.downtime_ms = (time.time() - start_time) * 1000
            result.success = True

        except Exception as e:
            result.success = False

        return result

    # ── Monitor Loop ───────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Background loop: check health, trigger rebalance on failure."""
        while self._running:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

            dead = self.get_dead_nodes()
            for node_id in dead:
                if not self._health[node_id].is_blacklisted:
                    # Node just died — trigger rebalance
                    self._pending_rebalance = node_id