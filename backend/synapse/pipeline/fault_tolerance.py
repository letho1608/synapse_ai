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
from typing import Dict, List, Optional, Tuple, Any
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
    # Absolute path of the on-disk JSON file this checkpoint was saved to /
    # loaded from. Filled by save_checkpoint() and by the disk-scan loaders;
    # None for in-memory-only checkpoints. Lets the recovery path hand the
    # exact file to the engine's checkpoint-load path.
    checkpoint_path: Optional[str] = None
    # Shard layer-weights file (engine.save_shard_weights) holding this
    # node's range as of `weights_step`; None = metadata-only checkpoint.
    # Recovery hands this to engine.load_checkpoint() so the survivor
    # restores real weights instead of resuming degraded.
    weights_path: Optional[str] = None
    weights_step: Optional[int] = None


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
    # NOTE (T1.1): there is intentionally NO periodic snapshot interval.
    # A former CHECKPOINT_INTERVAL constant was dead (no loop ever read it);
    # snapshots are written per training step via node_extension hooks, not on
    # a timer, so an inference-only cluster writes nothing. Do not re-add an
    # interval saver without retention bounds (see manifest cap below).
    MAX_MISSES = 3             # consecutive missed heartbeats before blacklisting
    MANIFEST_KEEP = 10         # max manifest entries retained per (node, model)

    def __init__(self, node_id: str, checkpoint_dir: str = None):
        if checkpoint_dir is None:
            # Was the HF hub cache root: shard checkpoints landed next to the
            # user's model downloads. Own directory, overridable.
            checkpoint_dir = os.getenv("SYNAPSE_CHECKPOINT_DIR") or os.path.join(
                os.path.expanduser("~"), ".synapse", "checkpoints")
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
        return [nid for nid, h in self._health.items()
                if not h.is_alive and nid != self.node_id]

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
        weights_path: Optional[str] = None,
        weights_step: Optional[int] = None,
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
            weights_path=weights_path,
            weights_step=weights_step,
        )

        # Prefix the filename with the node id so checkpoints written by
        # different nodes never collide for identical shard ranges, and so a
        # cross-node recovery (a *different* FaultToleranceManager looking for
        # this node's state) can find the file on the shared checkpoint dir.
        filename = (f"{self._safe_name(self.node_id)}_{self._safe_name(model_id)}"
                    f"_L{start_layer}-{end_layer}_step{step}.ckpt")
        filepath = os.path.join(self.checkpoint_dir, filename)
        ckpt.checkpoint_path = filepath

        # Use JSON-safe serialization (convert numpy arrays if present)
        data = asdict(ckpt)
        self._json_serialize(data, filepath)

        self._checkpoints[self.node_id].append(ckpt)
        # T1.1: index the file in the sidecar manifest (best-effort: a
        # manifest failure must never fail the checkpoint write itself).
        # Hash the text-mode round-trip (not raw bytes): on Windows the file
        # is written with \r\n translation, so a raw-byte hash would never
        # match the receiver's text hash. Both sides hash canonical text.
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                canonical = f.read()
            self._update_manifest(ckpt, len(canonical.encode("utf-8")),
                                  self._sha256_text(canonical))
        except Exception:
            pass
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
        prefix = f"{self._safe_name(node_id)}_"
        newest: Optional[ShardCheckpoint] = None
        try:
            entries = os.listdir(self.checkpoint_dir)
        except OSError:
            return None
        for fname in entries:
            if not fname.startswith(prefix) or not fname.endswith(".ckpt"):
                continue
            data = self._json_deserialize(os.path.join(self.checkpoint_dir, fname))
            # The prefix alone also matches node "a_b" when asked for "a".
            if data is None or data.get("node_id", node_id) != node_id:
                continue
            ckpt = ShardCheckpoint(**{k: v for k, v in data.items() if k in ShardCheckpoint.__dataclass_fields__})
            ckpt.checkpoint_path = os.path.join(self.checkpoint_dir, fname)
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
        safe_model = self._safe_name(model_id)
        legacy = f"{safe_model}_L{start_layer}-{end_layer}_step{step}.ckpt"
        patterns = (
            f"*_{safe_model}_L{start_layer}-{end_layer}_step{step}.ckpt",
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
        ckpt = ShardCheckpoint(**{k: v for k, v in data.items() if k in ShardCheckpoint.__dataclass_fields__})
        ckpt.checkpoint_path = filepath
        return ckpt

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

        except Exception:
            result.success = False

        return result

    @staticmethod
    def _safe_name(name: str) -> str:
        """Confine a node/model id to a flat filename segment (H5 traversal).

        Separators become "_" rather than being cut off, so HF repo ids
        ("org/name") stay distinct per org and never create subdirectories;
        ":" is also replaced (it opens an NTFS alternate data stream).
        """
        base = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_"
                       for c in str(name or ""))
        return base.strip(".") or "unknown"

    def _confine(self, filename: str) -> str:
        """Resolve filename inside checkpoint_dir or raise (H5 traversal)."""
        candidate = (self.checkpoint_dir and os.path.abspath(self.checkpoint_dir)) or os.path.abspath(".")
        target = os.path.abspath(os.path.join(candidate, os.path.basename(filename)))
        if os.path.commonpath([candidate, target]) != candidate:
            raise ValueError("checkpoint path escapes checkpoint_dir: %r" % filename)
        return target


    @staticmethod
    def _sha256_text(text: str) -> str:
        """Canonical-text hash shared with the P2P pull path (T1.1)."""
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _manifest_path(self, node_id: str, model_id: str) -> str:
        return self._confine(
            "%s_%s.manifest.json" % (self._safe_name(node_id), self._safe_name(model_id))
        )

    def _update_manifest(self, ckpt: ShardCheckpoint, size: int, sha256: str) -> None:
        """Append ckpt to the sidecar manifest (atomic tmp+rename, capped)."""
        mpath = self._manifest_path(ckpt.node_id, ckpt.model_id)
        manifest: Dict[str, Any] = {"node_id": ckpt.node_id, "model_id": ckpt.model_id,
                                    "files": [], "updated_at": time.time()}
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("files"), list):
                manifest = loaded
        except (OSError, ValueError):
            pass
        files = [e for e in manifest["files"]
                 if not (e.get("step") == ckpt.step and e.get("start_layer") == ckpt.start_layer
                         and e.get("end_layer") == ckpt.end_layer)]
        files.append({"step": ckpt.step, "start_layer": ckpt.start_layer,
                      "end_layer": ckpt.end_layer,
                      "path": os.path.basename(ckpt.checkpoint_path or ""),
                      "size": size, "sha256": sha256, "timestamp": ckpt.timestamp})
        files.sort(key=lambda e: (e.get("step", 0), e.get("timestamp", 0)))
        manifest["files"] = files[-self.MANIFEST_KEEP:]
        manifest["updated_at"] = time.time()
        tmp = mpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, mpath)

    def get_checkpoint_manifest(self, node_id: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Read the sidecar manifest index, or None if absent/corrupt."""
        try:
            with open(self._manifest_path(node_id, model_id), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def latest_manifest_step(self, node_id: str, model_id: str) -> Optional[int]:
        """Newest step in the manifest index (G5 monotonicity source)."""
        manifest = self.get_checkpoint_manifest(node_id, model_id)
        if not manifest:
            return None
        steps = [e.get("step") for e in manifest.get("files", []) if isinstance(e.get("step"), int)]
        return max(steps) if steps else None

    def get_recovery_checkpoint(self, node_id: str) -> Optional[ShardCheckpoint]:
        """Return the cross-node checkpoint staged for a node during the last
        rebalance, or None if no recovery state was recovered for it."""
        return self._recovery_checkpoints.get(node_id)

    # ── Monitor Loop ───────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Background loop: check health, trigger rebalance on failure."""
        while self._running:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

            # Check every registered node for timeout. We are alive by
            # definition: nobody delivers our own heartbeat back to us, so
            # timing ourselves out made every node declare ITSELF dead every
            # few seconds and rebalance itself out of the pipeline.
            for nid in list(self._health.keys()):
                h = self._health[nid]
                if nid == self.node_id:
                    h.record_heartbeat()
                    continue
                h.check_timeout(self.HEARTBEAT_TIMEOUT)

            dead = self.get_dead_nodes()
            for node_id in dead:
                if not self._health[node_id].is_blacklisted:
                    # Node just died — this would trigger rebalance
                    # Rebalance is invoked externally via check_fault_tolerance() in node.py
                    self._pending_rebalance = node_id