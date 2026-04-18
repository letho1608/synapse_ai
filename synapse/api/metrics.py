"""
Real-time Metrics Collection and Broadcasting System
Collects system and performance metrics and streams to WebSocket clients
"""

import asyncio
import json
import psutil
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for a node"""
    timestamp: float
    node_id: str
    ring_rank: int = -1
    
    # System metrics
    cpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    system_memory_mb: float = 0.0
    network_bandwidth_in_mbps: float = 0.0
    network_bandwidth_out_mbps: float = 0.0
    
    # Inference metrics
    active_inference_requests: int = 0
    avg_token_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # Training metrics
    active_training_requests: int = 0
    gradient_sync_latency_ms: float = 0.0
    training_throughput_examples_per_sec: float = 0.0
    current_training_loss: Optional[float] = None
    
    # Network metrics
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    active_peer_connections: int = 0
    
    # Shard info
    shard_start_layer: int = 0
    shard_end_layer: int = 0
    shard_model_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Remove None values to reduce message size
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class MetricsSnapshot:
    """Aggregated metrics for all nodes"""
    timestamp: float
    node_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    cluster_avg_latency_ms: float = 0.0
    cluster_total_throughput: float = 0.0
    cluster_training_loss: Optional[float] = None


class MetricsCollector:
    """
    Collects real-time metrics from nodes and maintains history.
    Supports WebSocket broadcasting and REST API queries.
    """
    
    def __init__(self, max_history_per_node: int = 1000):
        self.max_history_per_node = max_history_per_node
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = {}
        self.connected_clients: List = []  # WebSocket clients
        self._lock = asyncio.Lock()
        
        # Performance tracking
        self.collection_times: List[float] = []
        self.broadcast_times: List[float] = []
    
    async def record_metric(self, metric: PerformanceMetrics):
        """Record a metric and broadcast to WebSocket clients"""
        async with self._lock:
            # Add to history
            if metric.node_id not in self.metrics_history:
                self.metrics_history[metric.node_id] = []
            
            self.metrics_history[metric.node_id].append(metric)
            
            # Keep sliding window
            if len(self.metrics_history[metric.node_id]) > self.max_history_per_node:
                self.metrics_history[metric.node_id].pop(0)
        
        # Broadcast to clients (async, non-blocking)
        await self.broadcast_metric(metric)
    
    async def broadcast_metric(self, metric: PerformanceMetrics):
        """Send metric update to all connected WebSocket clients"""
        start_time = time.time()
        message = json.dumps(metric.to_dict(), default=str)
        
        disconnected_clients = []
        
        for client in self.connected_clients:
            try:
                # Send with timeout
                await asyncio.wait_for(
                    client.send_text(message),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                print(f"⚠ WebSocket send timeout for {client}")
                disconnected_clients.append(client)
            except Exception as e:
                # Client disconnected or error
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            try:
                self.connected_clients.remove(client)
            except ValueError:
                pass
        
        # Track broadcast time
        broadcast_time = (time.time() - start_time) * 1000
        self.broadcast_times.append(broadcast_time)
        if len(self.broadcast_times) > 100:
            self.broadcast_times.pop(0)
    
    async def get_node_metrics(self, node_id: str, limit: int = 100) -> List[Dict]:
        """Get recent metrics for a node"""
        async with self._lock:
            if node_id not in self.metrics_history:
                return []
            
            metrics = self.metrics_history[node_id][-limit:]
            return [m.to_dict() for m in metrics]
    
    async def get_cluster_metrics(self) -> MetricsSnapshot:
        """Get aggregated cluster metrics"""
        async with self._lock:
            snapshot = MetricsSnapshot(timestamp=time.time())
            
            latencies = []
            throughputs = []
            losses = []
            
            for node_id, metrics_list in self.metrics_history.items():
                if metrics_list:
                    latest = metrics_list[-1]
                    snapshot.node_metrics[node_id] = latest
                    
                    if latest.avg_token_latency_ms > 0:
                        latencies.append(latest.avg_token_latency_ms)
                    if latest.tokens_per_second > 0:
                        throughputs.append(latest.tokens_per_second)
                    if latest.current_training_loss is not None:
                        losses.append(latest.current_training_loss)
            
            if latencies:
                snapshot.cluster_avg_latency_ms = float(np.mean(latencies))
            if throughputs:
                snapshot.cluster_total_throughput = float(np.sum(throughputs))
            if losses:
                snapshot.cluster_training_loss = float(np.mean(losses))
        
        return snapshot
    
    def register_websocket(self, websocket):
        """Register a WebSocket client for metric broadcasts"""
        self.connected_clients.append(websocket)
    
    def unregister_websocket(self, websocket):
        """Unregister a WebSocket client"""
        try:
            self.connected_clients.remove(websocket)
        except ValueError:
            pass
    
    async def get_performance_stats(self) -> Dict:
        """Get performance statistics of the metrics system itself"""
        return {
            "collection_times_ms": {
                "avg": float(np.mean(self.collection_times)) if self.collection_times else 0,
                "max": float(np.max(self.collection_times)) if self.collection_times else 0,
                "min": float(np.min(self.collection_times)) if self.collection_times else 0,
            },
            "broadcast_times_ms": {
                "avg": float(np.mean(self.broadcast_times)) if self.broadcast_times else 0,
                "max": float(np.max(self.broadcast_times)) if self.broadcast_times else 0,
                "min": float(np.min(self.broadcast_times)) if self.broadcast_times else 0,
            },
            "connected_clients": len(self.connected_clients),
            "total_nodes": len(self.metrics_history),
        }


class SystemMetricsCollector:
    """
    Collects system-level metrics (CPU, GPU, network, etc.)
    """
    
    def __init__(self):
        self.net_io_last = None
        self.net_io_timestamp = None
        self.prev_active_connections = 0
    
    async def collect_system_metrics(self) -> Dict:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory
            virtual_memory = psutil.virtual_memory()
            system_memory_mb = virtual_memory.used / (1024 * 1024)
            
            # Network bandwidth
            net_io = psutil.net_io_counters()
            bandwidth_in = 0.0
            bandwidth_out = 0.0
            
            if self.net_io_last and self.net_io_timestamp:
                time_delta = time.time() - self.net_io_timestamp
                bytes_in_delta = net_io.bytes_recv - self.net_io_last.bytes_recv
                bytes_out_delta = net_io.bytes_sent - self.net_io_last.bytes_sent
                
                # Convert to Mbps
                bandwidth_in = (bytes_in_delta * 8) / (time_delta * 1_000_000)
                bandwidth_out = (bytes_out_delta * 8) / (time_delta * 1_000_000)
            
            self.net_io_last = net_io
            self.net_io_timestamp = time.time()
            
            # GPU memory (if available)
            gpu_memory_mb = await self._get_gpu_memory()
            
            return {
                'cpu_usage_percent': float(cpu_usage),
                'system_memory_mb': float(system_memory_mb),
                'network_bandwidth_in_mbps': float(bandwidth_in),
                'network_bandwidth_out_mbps': float(bandwidth_out),
                'gpu_memory_mb': float(gpu_memory_mb),
                'total_bytes_sent': int(net_io.bytes_sent),
                'total_bytes_received': int(net_io.bytes_recv),
            }
        
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            return {}
    
    async def _get_gpu_memory(self) -> float:
        """Get GPU memory usage if available"""
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.memory_allocated()) / (1024 * 1024)
        except Exception:
            pass
        return 0.0


class InferenceMetricsTracker:
    """Tracks inference-specific metrics"""
    
    def __init__(self):
        self.token_latencies: List[float] = []
        self.active_requests: int = 0
        self.total_tokens: int = 0
        self.window_size = 100
    
    def record_token_latency(self, latency_ms: float):
        """Record latency of a single token"""
        self.token_latencies.append(latency_ms)
        if len(self.token_latencies) > self.window_size:
            self.token_latencies.pop(0)
        self.total_tokens += 1
    
    def get_avg_latency_ms(self) -> float:
        """Get average token latency over recent window"""
        if not self.token_latencies:
            return 0.0
        return float(np.mean(self.token_latencies))
    
    def get_throughput(self) -> float:
        """Get tokens per second"""
        if not self.token_latencies:
            return 0.0
        # Rough estimate: window_size / sum of window latencies (in seconds)
        total_time_sec = sum(self.token_latencies) / 1000
        return len(self.token_latencies) / total_time_sec if total_time_sec > 0 else 0.0


class TrainingMetricsTracker:
    """Tracks training-specific metrics"""
    
    def __init__(self):
        self.grad_sync_latencies: List[float] = []
        self.training_losses: List[float] = []
        self.active_training_requests: int = 0
        self.window_size = 50
        self.last_sync_time: Optional[float] = None
        self.last_loss: Optional[float] = None
    
    def record_grad_sync_latency(self, latency_ms: float):
        """Record gradient sync latency"""
        self.grad_sync_latencies.append(latency_ms)
        if len(self.grad_sync_latencies) > self.window_size:
            self.grad_sync_latencies.pop(0)
    
    def record_training_loss(self, loss: float):
        """Record training loss"""
        self.training_losses.append(loss)
        self.last_loss = loss
        if len(self.training_losses) > self.window_size:
            self.training_losses.pop(0)
    
    def get_avg_grad_sync_latency_ms(self) -> float:
        """Get average gradient sync latency"""
        if not self.grad_sync_latencies:
            return 0.0
        return float(np.mean(self.grad_sync_latencies))
    
    def get_current_loss(self) -> Optional[float]:
        """Get current training loss"""
        return self.last_loss


# Integration helper for Node class
class MetricsIntegration:
    """Helper class to integrate metrics into Node"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.metrics_collector = MetricsCollector()
        self.system_metrics = SystemMetricsCollector()
        self.inference_metrics = InferenceMetricsTracker()
        self.training_metrics = TrainingMetricsTracker()
    
    async def collect_and_report(
        self,
        ring_rank: int = -1,
        active_peers: int = 0,
        current_shard = None
    ) -> PerformanceMetrics:
        """Collect all metrics and create PerformanceMetrics"""
        system_metrics = await self.system_metrics.collect_system_metrics()
        
        metric = PerformanceMetrics(
            timestamp=time.time(),
            node_id=self.node_id,
            ring_rank=ring_rank,
            **system_metrics,
            
            # Inference metrics
            active_inference_requests=self.inference_metrics.active_requests,
            avg_token_latency_ms=self.inference_metrics.get_avg_latency_ms(),
            tokens_per_second=self.inference_metrics.get_throughput(),
            
            # Training metrics
            active_training_requests=self.training_metrics.active_training_requests,
            gradient_sync_latency_ms=self.training_metrics.get_avg_grad_sync_latency_ms(),
            current_training_loss=self.training_metrics.get_current_loss(),
            
            # Network metrics
            active_peer_connections=active_peers,
            
            # Shard info
            shard_start_layer=current_shard.start_layer if current_shard else 0,
            shard_end_layer=current_shard.end_layer if current_shard else 0,
            shard_model_id=current_shard.model_id if current_shard else "",
        )
        
        return metric
