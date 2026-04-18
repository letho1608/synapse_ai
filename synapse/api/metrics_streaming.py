"""
Real-time WebSocket metrics streaming for training progress.
Replaces polling with push-based WebSocket updates.
"""

import asyncio
import json
import time
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetricsFrame:
    """Single frame of training metrics for streaming."""
    timestamp: float
    epoch: int
    step: int
    loss: Optional[float]
    progress_pct: int
    forward_ms: Optional[float]
    backward_ms: Optional[float]
    sync_ms: Optional[float]
    sync_strategy: Optional[str]  # "tree", "ring", "mesh"
    sync_bytes: Optional[int]
    tokens_per_sec: Optional[float]
    learning_rate: float


@dataclass
class SyncStatusFrame:
    """Sync operation status during AllReduce."""
    timestamp: float
    sync_id: str
    status: str  # "starting", "reduce", "broadcast", "completed", "failed"
    strategy: str  # "tree", "ring", "mesh"
    phase: int  # current phase (1-3)
    total_phases: int
    nodes_involved: int
    progress_pct: int
    data_size_mb: float
    latency_ms: Optional[float]
    error: Optional[str]


class MetricsStreamManager:
    """
    Manages WebSocket connections and broadcasts training metrics in real-time.
    Replaces dashboard polling with push-based updates.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}  # job_id -> [callbacks]
        self.metrics_history: Dict[str, List[TrainingMetricsFrame]] = {}
        self.sync_history: Dict[str, List[SyncStatusFrame]] = {}
        self.active_jobs: Dict[str, bool] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, job_id: str, callback: Callable) -> None:
        """Register a WebSocket callback for a job."""
        async with self._lock:
            if job_id not in self.subscribers:
                self.subscribers[job_id] = []
            self.subscribers[job_id].append(callback)
            self.metrics_history[job_id] = []
            self.sync_history[job_id] = []
            self.active_jobs[job_id] = True
            logger.info(f"Subscribed to metrics for job {job_id}")
    
    async def unsubscribe(self, job_id: str, callback: Callable) -> None:
        """Unregister a WebSocket callback."""
        async with self._lock:
            if job_id in self.subscribers:
                self.subscribers[job_id] = [
                    cb for cb in self.subscribers[job_id] 
                    if cb != callback
                ]
                if not self.subscribers[job_id]:
                    del self.subscribers[job_id]
                    del self.metrics_history[job_id]
                    del self.sync_history[job_id]
                    del self.active_jobs[job_id]
                    logger.info(f"Unsubscribed from metrics for job {job_id}")
    
    async def emit_metrics(self, job_id: str, frame: TrainingMetricsFrame) -> None:
        """Emit a metrics frame to all subscribers of this job."""
        async with self._lock:
            if job_id not in self.active_jobs:
                return
            
            # Store in history (keep last 1000 frames per job)
            if job_id not in self.metrics_history:
                self.metrics_history[job_id] = []
            self.metrics_history[job_id].append(frame)
            if len(self.metrics_history[job_id]) > 1000:
                self.metrics_history[job_id].pop(0)
        
        # Broadcast to all subscribers
        if job_id in self.subscribers:
            payload = {
                "type": "training_metrics",
                "job_id": job_id,
                "data": asdict(frame)
            }
            tasks = [
                asyncio.create_task(cb(payload))
                for cb in self.subscribers[job_id]
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def emit_sync_status(self, job_id: str, frame: SyncStatusFrame) -> None:
        """Emit a sync status frame to all subscribers."""
        async with self._lock:
            if job_id not in self.active_jobs:
                return
            
            # Store in history
            if job_id not in self.sync_history:
                self.sync_history[job_id] = []
            self.sync_history[job_id].append(frame)
            if len(self.sync_history[job_id]) > 500:
                self.sync_history[job_id].pop(0)
        
        # Broadcast to all subscribers
        if job_id in self.subscribers:
            payload = {
                "type": "sync_status",
                "job_id": job_id,
                "data": asdict(frame)
            }
            tasks = [
                asyncio.create_task(cb(payload))
                for cb in self.subscribers[job_id]
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_metrics_history(self, job_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve recent metrics history for a job."""
        async with self._lock:
            if job_id not in self.metrics_history:
                return []
            history = self.metrics_history[job_id][-limit:]
            return [asdict(m) for m in history]
    
    async def get_sync_history(self, job_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve recent sync status history for a job."""
        async with self._lock:
            if job_id not in self.sync_history:
                return []
            history = self.sync_history[job_id][-limit:]
            return [asdict(s) for s in history]


class TrainingMetricsCollector:
    """
    Collects training metrics during a training session.
    Updates MetricsStreamManager in real-time.
    """
    
    def __init__(self, job_id: str, stream_manager: MetricsStreamManager):
        self.job_id = job_id
        self.stream_manager = stream_manager
        self.step_times: Dict[str, float] = {}
        self.last_sync_ms = 0.0
        self.sync_count = 0
    
    async def record_forward_pass(self, duration_ms: float) -> None:
        """Record forward pass duration."""
        self.step_times["forward"] = duration_ms
    
    async def record_backward_pass(self, duration_ms: float) -> None:
        """Record backward pass duration."""
        self.step_times["backward"] = duration_ms
    
    async def record_sync_pass(self, duration_ms: float, strategy: str = "ring") -> None:
        """Record gradient sync (AllReduce) duration."""
        self.step_times["sync"] = duration_ms
        self.last_sync_ms = duration_ms
        self.sync_count += 1
    
    async def emit_step_metrics(
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
        """Emit metrics for a training step."""
        frame = TrainingMetricsFrame(
            timestamp=time.time(),
            epoch=epoch,
            step=step,
            loss=loss,
            progress_pct=progress_pct,
            forward_ms=self.step_times.get("forward"),
            backward_ms=self.step_times.get("backward"),
            sync_ms=self.step_times.get("sync"),
            sync_strategy=sync_strategy,
            sync_bytes=sync_bytes,
            tokens_per_sec=tokens_per_sec,
            learning_rate=lr
        )
        await self.stream_manager.emit_metrics(self.job_id, frame)


class SyncStatusTracker:
    """Tracks AllReduce sync operations and reports status."""
    
    def __init__(self, job_id: str, stream_manager: MetricsStreamManager):
        self.job_id = job_id
        self.stream_manager = stream_manager
        self.sync_start_time: Dict[str, float] = {}
        self.sync_id_counter = 0
    
    async def start_sync(self, strategy: str, nodes_involved: int, data_size_mb: float) -> str:
        """Mark start of a sync operation."""
        sync_id = f"sync_{self.sync_id_counter}"
        self.sync_id_counter += 1
        self.sync_start_time[sync_id] = time.time()
        
        frame = SyncStatusFrame(
            timestamp=time.time(),
            sync_id=sync_id,
            status="starting",
            strategy=strategy,
            phase=0,
            total_phases=3,  # reduce, broadcast, finalize
            nodes_involved=nodes_involved,
            progress_pct=0,
            data_size_mb=data_size_mb,
            latency_ms=None,
            error=None
        )
        await self.stream_manager.emit_sync_status(self.job_id, frame)
        return sync_id
    
    async def update_sync_phase(
        self,
        sync_id: str,
        phase: int,
        progress_pct: int,
        phase_name: str = ""
    ) -> None:
        """Update sync operation progress."""
        elapsed_ms = (time.time() - self.sync_start_time.get(sync_id, time.time())) * 1000
        
        frame = SyncStatusFrame(
            timestamp=time.time(),
            sync_id=sync_id,
            status=phase_name or f"phase_{phase}",
            strategy="ring",  # Will be overridden
            phase=phase,
            total_phases=3,
            nodes_involved=0,  # Passed separately
            progress_pct=progress_pct,
            data_size_mb=0,
            latency_ms=elapsed_ms,
            error=None
        )
        await self.stream_manager.emit_sync_status(self.job_id, frame)
    
    async def complete_sync(self, sync_id: str, total_duration_ms: float) -> None:
        """Mark sync operation as completed."""
        frame = SyncStatusFrame(
            timestamp=time.time(),
            sync_id=sync_id,
            status="completed",
            strategy="ring",
            phase=3,
            total_phases=3,
            nodes_involved=0,
            progress_pct=100,
            data_size_mb=0,
            latency_ms=total_duration_ms,
            error=None
        )
        await self.stream_manager.emit_sync_status(self.job_id, frame)
        if sync_id in self.sync_start_time:
            del self.sync_start_time[sync_id]
    
    async def fail_sync(self, sync_id: str, error_msg: str) -> None:
        """Mark sync operation as failed."""
        elapsed_ms = (time.time() - self.sync_start_time.get(sync_id, time.time())) * 1000
        
        frame = SyncStatusFrame(
            timestamp=time.time(),
            sync_id=sync_id,
            status="failed",
            strategy="ring",
            phase=3,
            total_phases=3,
            nodes_involved=0,
            progress_pct=0,
            data_size_mb=0,
            latency_ms=elapsed_ms,
            error=error_msg
        )
        await self.stream_manager.emit_sync_status(self.job_id, frame)
        if sync_id in self.sync_start_time:
            del self.sync_start_time[sync_id]


# Global instance (thread-safe)
_metrics_stream_manager: Optional[MetricsStreamManager] = None


def get_metrics_stream_manager() -> MetricsStreamManager:
    """Get or create global metrics stream manager."""
    global _metrics_stream_manager
    if _metrics_stream_manager is None:
        _metrics_stream_manager = MetricsStreamManager()
    return _metrics_stream_manager
