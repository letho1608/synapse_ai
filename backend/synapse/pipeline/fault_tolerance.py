"""
LACP — Fault Tolerance & Dynamic Rebalancing Engine

Handles node failures in the distributed pipeline with zero-downtime recovery.

Strategy:
  1. Heartbeat monitoring — detect dead nodes within 5s
  2. State checkpointing — periodic shard state snapshots to disk/shared memory
  3. Dynamic rebalancing — redistribute dead node's shard to surviving nodes
  4. Pipeline resume — continue inference/training from last checkpoint
  5. Graceful degradation — if N nodes drop to N-1, rebalance then continue

The rebalancing uses the LACP partitioner to recompute optimal shard
assignments excluding the failed node, then redistributes layers.
"""

from __future__ import annotations

import asyncio
import time
import json
import os
import glob
from collections import defaultdict
from dataclasses import dataclass, field, asdict
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
    # The checkpoint recovered for the dead node (loaded from disk when the
    # checkpoint was written by a different process), and the surviving node
    # whose new shard range absorbs the dead node's layers — that node must
    # load this checkpoint to resume training/inference from the last state.
    recovery_checkpoint: Optional[ShardCheckpoint] = None
    recovery_target_node: Optional[str] = None
    downtime_ms: float = 0.0
    success: bool = False


class FaultToleranceManager:
    """
    Manages node health monitoring, checkpointing, and dynamic rebalancing.
    """

    HEARTBEAT_INTERVAL = 2.0   # seconds between heartbeats
    HEARTBEAT_TIMEOUT = 5.0    # seconds before considering node dead (matches README claim)
    CHECKPOINT_INTERVAL = 30.0  # seconds between state snapshots
    MAX_MISSES = 3             # consecutive missed heartbeats before blacklisting

    def __init__(self, node_id: str, checkpoint_dir: str = None):
        if checkpoint_dir is None:
            from synapse.loading import get_checkpoints_dir
            checkpoint_dir = get_checkpoints_dir()
        self.node_id = node_id
        self.checkpoint_dir = checkpoint_dir
        self._health: Dict[str, NodeHealth] = {}
        self._checkpoints: Dict[str, List[ShardCheckpoint]] = defaultdict(list)
        # Recovered cross-node checkpoints staged for the surviving node that
        # takes over a dead node's shard range: {target_node_id: ShardCheckpoint}.
        self._recovery_checkpoints: Dict[str, ShardCheckpoint] = {}
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
        """Save a shard state checkpoint to JSON. Returns checkpoint path."""
        ckpt = ShardCheckpoint(
            node_id=self.node_id,
            model_id=model_id,
            start_layer=start_layer,
            end_layer=end_layer,
            step=step,
            kv_cache=kv_cache,
            optimizer_state=optimizer_state,
        )

        # Prefix the filename with the node id so checkpoints written by
        # different nodes never collide for identical shard ranges, and so a
        # cross-node recovery (a *different* FaultToleranceManager looking for
        # this node's state) can find the file on the shared checkpoint dir.
        filename = f"{self.node_id}_{model_id}_L{start_layer}-{end_layer}_step{step}.ckpt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Use JSON-safe serialization (convert numpy arrays if present)
        data = asdict(ckpt)
        self._json_serialize(data, filepath)

        self._checkpoints[self.node_id].append(ckpt)
        return filepath

    @staticmethod
    def _json_serialize(data: dict, filepath: str) -> None:
        """Serialize dict to JSON, converting numpy types safely."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj
        sanitized = convert(data)
        with open(filepath, "w") as f:
            json.dump(sanitized, f, default=str, indent=2)

    @staticmethod
    def _json_deserialize(filepath: str) -> Optional[dict]:
        """Deserialize JSON file safely."""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return None

    def load_latest_checkpoint(self, node_id: str) -> Optional[ShardCheckpoint]:
        """Load the most recent checkpoint for a node (used during recovery).

        Checks the in-memory checkpoints saved by this process first, then
        falls back to scanning the shared checkpoint directory for files this
        node wrote to disk. The disk fallback is what makes *cross-node*
        recovery work: the orchestrator's own memory never holds checkpoints
        saved by a different (dead) node — only the on-disk state does.
        """
        ckpts = list(self._checkpoints.get(node_id, []))
        if ckpts:
            return max(ckpts, key=lambda c: c.timestamp)

        # Cross-node fallback: scan disk for files saved by this node.
        prefix = f"{node_id}_"
        newest: Optional[ShardCheckpoint] = None
        try:
            entries = os.listdir(self.checkpoint_dir)
        except OSError:
            return None
        for fname in entries:
            if not fname.startswith(prefix) or not fname.endswith(".ckpt"):
                continue
            data = self._json_deserialize(os.path.join(self.checkpoint_dir, fname))
            if data is None:
                continue
            ckpt = ShardCheckpoint(**{k: v for k, v in data.items() if k in ShardCheckpoint.__dataclass_fields__})
            if newest is None or ckpt.timestamp > newest.timestamp:
                newest = ckpt
        return newest

    def load_checkpoint(
        self, model_id: str, start_layer: int, end_layer: int, step: int
    ) -> Optional[ShardCheckpoint]:
        """Load a specific checkpoint from disk (JSON format).

        Matches both the legacy filename (no node prefix) and the current
        node-prefixed format. If several nodes saved the same shard range, the
        most recently modified file wins.
        """
        legacy = f"{model_id}_L{start_layer}-{end_layer}_step{step}.ckpt"
        patterns = (
            f"*_{model_id}_L{start_layer}-{end_layer}_step{step}.ckpt",
            legacy,
        )
        candidates = []
        for pattern in patterns:
            candidates.extend(glob.glob(os.path.join(self.checkpoint_dir, pattern)))
        candidates = [c for c in candidates if os.path.isfile(c)]
        if not candidates:
            return None
        filepath = max(candidates, key=os.path.getmtime)

        data = self._json_deserialize(filepath)
        if data is None:
            return None
        return ShardCheckpoint(**{k: v for k, v in data.items() if k in ShardCheckpoint.__dataclass_fields__})

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

            # Compute new assignments via LACP
            new_assignments = await lacp_partition_fn(alive, total_layers)
            result.new_assignments = new_assignments

            # Load checkpoint for dead node's shard range (cross-node: this
            # falls back to the on-disk state the dead node itself saved).
            dead_shard = current_assignments.get(dead_node_id)
            if dead_shard:
                ckpt = self.load_latest_checkpoint(dead_node_id)
                if ckpt:
                    result.recovered_from_checkpoint = True
                    # Find the surviving node whose new shard range absorbs
                    # the dead node's layers — that node must resume from this
                    # checkpoint. (Never stuff the checkpoint into the
                    # assignments dict; that is a per-node layer map.)
                    target = None
                    for nid, (s, e) in new_assignments.items():
                        if s <= dead_shard[0] <= e:
                            target = nid
                            break
                    if target is None and new_assignments:
                        target = next(iter(new_assignments))
                    result.recovery_checkpoint = ckpt
                    result.recovery_target_node = target
                    if target:
                        # Stage the recovered state so the survivor can pull it
                        # via get_recovery_checkpoint() and resume training.
                        self._recovery_checkpoints[target] = ckpt

            result.downtime_ms = (time.time() - start_time) * 1000
            result.success = True

        except Exception as e:
            result.success = False

        return result

    def get_recovery_checkpoint(self, node_id: str) -> Optional[ShardCheckpoint]:
        """Return the cross-node checkpoint staged for a node during the last
        rebalance, or None if no recovery state was recovered for it."""
        return self._recovery_checkpoints.get(node_id)

    # ── Monitor Loop ───────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Background loop: check health, trigger rebalance on failure."""
        while self._running:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

            # Check every registered node for timeout
            for nid in list(self._health.keys()):
                h = self._health[nid]
                h.check_timeout(self.HEARTBEAT_TIMEOUT)

            dead = self.get_dead_nodes()
            for node_id in dead:
                if not self._health[node_id].is_blacklisted:
                    # Node just died — this would trigger rebalance
                    # Rebalance is invoked externally via check_fault_tolerance() in node.py
                    self._pending_rebalance = node_id