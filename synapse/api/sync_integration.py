"""
Integration layer to track AllReduce sync operations with real-time status updates.
Works with metrics_streaming to push sync progress to WebSocket clients.
"""

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SyncMetrics:
    """Metrics for a single sync operation."""
    sync_id: str
    strategy: str  # "tree", "ring", "mesh"
    nodes: int
    data_size_mb: float
    reduce_phase_ms: float
    broadcast_phase_ms: float
    total_ms: float
    success: bool
    error: Optional[str] = None


class AdaptiveAllReduceWithTracking:
    """
    Wrapper around AllReduce that provides real-time status tracking
    via metrics_streaming module.
    """
    
    def __init__(self, adaptive_allreduce_manager, sync_tracker):
        self.manager = adaptive_allreduce_manager
        self.tracker = sync_tracker
        self.sync_count = 0
    
    async def execute_with_tracking(
        self,
        flatten_grads: np.ndarray,
        node_rank: int,
        ring_successor,
        model_id: str,
        timeout: float = 120.0
    ) -> bool:
        """
        Execute AllReduce with real-time status tracking.
        
        Reports:
        - Starting (0%)
        - Reduce phase (30%)
        - Broadcast phase (70%)
        - Completed (100%)
        """
        self.sync_count += 1
        sync_id = f"sync_{model_id}_{self.sync_count}"
        
        nodes = self.manager.world_size
        data_size_mb = flatten_grads.nbytes / (1024 * 1024)
        strategy = self._select_strategy(nodes)
        
        start_time = time.time()
        
        try:
            # Report start
            await self.tracker.start_sync(strategy, nodes, data_size_mb)
            
            # --- Phase 1: Reduce (30% of time) ---
            await self.tracker.update_sync_phase(
                sync_id, 1, 10, "reduce_start"
            )
            
            reduce_start = time.time()
            reduce_success = await self._execute_reduce_phase(
                flatten_grads, node_rank, ring_successor, timeout * 0.3
            )
            reduce_ms = (time.time() - reduce_start) * 1000
            
            if not reduce_success:
                await self.tracker.fail_sync(sync_id, "Reduce phase failed")
                return False
            
            await self.tracker.update_sync_phase(
                sync_id, 1, 30, "reduce_complete"
            )
            
            # --- Phase 2: Broadcast (40% of time) ---
            await self.tracker.update_sync_phase(
                sync_id, 2, 40, "broadcast_start"
            )
            
            broadcast_start = time.time()
            broadcast_success = await self._execute_broadcast_phase(
                flatten_grads, node_rank, ring_successor, timeout * 0.4
            )
            broadcast_ms = (time.time() - broadcast_start) * 1000
            
            if not broadcast_success:
                await self.tracker.fail_sync(sync_id, "Broadcast phase failed")
                return False
            
            await self.tracker.update_sync_phase(
                sync_id, 2, 75, "broadcast_complete"
            )
            
            # --- Phase 3: Finalize (30% of time) ---
            await self.tracker.update_sync_phase(
                sync_id, 3, 90, "finalize"
            )
            
            total_ms = (time.time() - start_time) * 1000
            
            # Report completion
            await self.tracker.complete_sync(sync_id, total_ms)
            
            return True
            
        except Exception as e:
            error_msg = f"{strategy} failed: {str(e)}"
            await self.tracker.fail_sync(sync_id, error_msg)
            return False
    
    def _select_strategy(self, nodes: int) -> str:
        """Select strategy based on node count."""
        if nodes <= 8:
            return "tree"
        elif nodes <= 16:
            return "ring"
        else:
            return "mesh"
    
    async def _execute_reduce_phase(
        self,
        flatten_grads: np.ndarray,
        node_rank: int,
        ring_successor,
        timeout: float
    ) -> bool:
        """Execute reduce phase with timeout."""
        try:
            # Send gradient chunk to successor
            # (simplified - actual implementation in allreduce_strategies.py)
            await asyncio.wait_for(
                self._send_chunk_async(flatten_grads, ring_successor),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False
    
    async def _execute_broadcast_phase(
        self,
        flatten_grads: np.ndarray,
        node_rank: int,
        ring_successor,
        timeout: float
    ) -> bool:
        """Execute broadcast phase with timeout."""
        try:
            await asyncio.wait_for(
                self._receive_chunk_async(flatten_grads, ring_successor),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False
    
    async def _send_chunk_async(self, data: np.ndarray, target) -> None:
        """Send chunk to target (placeholder)."""
        if hasattr(target, 'transfer_chunk'):
            await target.transfer_chunk(0, data, "reduce")
    
    async def _receive_chunk_async(self, data: np.ndarray, source) -> None:
        """Receive chunk from source (placeholder)."""
        await asyncio.sleep(0.01)  # Simulate


class TrainingStepMetricsRecorder:
    """Records metrics for each training step and emits them via streaming."""
    
    def __init__(self, job_id: str, metrics_collector):
        self.job_id = job_id
        self.collector = metrics_collector
        self.step_timer = {}
    
    async def start_step(self) -> None:
        """Mark start of a training step."""
        self.step_timer["step_start"] = time.time()
    
    async def end_forward(self) -> None:
        """Mark end of forward pass."""
        start = self.step_timer.get("step_start", time.time())
        forward_ms = (time.time() - start) * 1000
        await self.collector.record_forward_pass(forward_ms)
        self.step_timer["forward_end"] = time.time()
    
    async def end_backward(self) -> None:
        """Mark end of backward pass."""
        start = self.step_timer.get("forward_end", time.time())
        backward_ms = (time.time() - start) * 1000
        await self.collector.record_backward_pass(backward_ms)
        self.step_timer["backward_end"] = time.time()
    
    async def end_sync(self, strategy: str = "ring") -> None:
        """Mark end of sync phase."""
        start = self.step_timer.get("backward_end", time.time())
        sync_ms = (time.time() - start) * 1000
        await self.collector.record_sync_pass(sync_ms, strategy)
    
    async def emit_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        progress_pct: int,
        lr: float,
        sync_strategy: Optional[str] = None,
        sync_bytes: Optional[int] = None,
        tokens_per_sec: Optional[float] = None
    ) -> None:
        """Emit complete step metrics."""
        await self.collector.emit_step_metrics(
            epoch=epoch,
            step=step,
            loss=loss,
            progress_pct=progress_pct,
            lr=lr,
            sync_strategy=sync_strategy,
            sync_bytes=sync_bytes,
            tokens_per_sec=tokens_per_sec
        )
