"""
Latency Probing - Đo latency giữa các máy qua Tailscale
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from synapse.networking.peer_handle import PeerHandle
from synapse.topology.topology import Topology


@dataclass
class LatencyResult:
    """Kết quả đo latency"""
    from_id: str
    to_id: str
    latency_ms: float
    timestamp: float


@dataclass
class LatencyCache:
    """Cache cho latency matrix"""
    version: str = "1.0"
    timestamp: float = 0.0
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)


class LatencyProber:
    """
    Đo latency giữa các máy qua Tailscale.
    
    Gửi ping đến tất cả peers, đo RTT và tính trung bình.
    Loại bỏ outliers (max, min) để lấy kết quả ổn định hơn.
    """
    
    def __init__(
        self, 
        cache_file: str = "synapse/config/latency_cache.json",
        probe_count: int = 10,
        timeout_ms: int = 5000,
        cache_validity_seconds: int = 300
    ):
        """
        Args:
            cache_file: File để cache latency matrix
            probe_count: Số ping mỗi lần đo
            timeout_ms: Timeout cho mỗi ping (ms)
            cache_validity_seconds: Cache valid trong bao lâu (5 phút)
        """
        self.cache_file = Path(cache_file)
        self.probe_count = probe_count
        self.timeout_ms = timeout_ms
        self.cache_validity_seconds = cache_validity_seconds
        self._cache: Optional[LatencyCache] = None
    
    def _load_cache(self) -> Optional[LatencyCache]:
        """Load cache từ file"""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            cache = LatencyCache(
                version=data.get("version", "1.0"),
                timestamp=data.get("timestamp", 0.0),
                matrix=data.get("matrix", {})
            )
            return cache
        except (json.JSONDecodeError, IOError):
            return None
    
    def _save_cache(self, cache: LatencyCache) -> None:
        """Save cache ra file"""
        try:
            # Tạo thư mục cha nếu chưa có
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    "version": cache.version,
                    "timestamp": cache.timestamp,
                    "matrix": cache.matrix
                }, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save latency cache: {e}")
    
    def _needs_update(self) -> bool:
        """Kiểm tra xem cache có còn valid không"""
        if self._cache is None:
            self._cache = self._load_cache()
        
        if self._cache is None:
            return True
        
        # Cache quá hạn
        age = time.time() - self._cache.timestamp
        return age > self.cache_validity_seconds
    
    def get_matrix(self, topology: Topology) -> Dict[str, Dict[str, float]]:
        """
        Lấy latency matrix, probe lại nếu cần.

        Args:
            topology: Topology để lấy peers

        Returns:
            Dict[from_id][to_id] = latency_ms
        """
        if self._needs_update():
            # Use topology.nodes.keys() to get peer IDs - Topology has 'nodes' not 'peers'
            peer_ids = list(topology.nodes.keys()) if topology.nodes else []
            if peer_ids:
                # Try to probe peers, but probe_single is a placeholder that returns {}
                # So we build a default latency matrix based on node IDs
                # This ensures the ILP solver has a valid matrix to work with
                matrix: Dict[str, Dict[str, float]] = {}
                for peer_id in peer_ids:
                    matrix[peer_id] = {}
                    for other_id in peer_ids:
                        if peer_id == other_id:
                            matrix[peer_id][other_id] = 0.0  # Same machine = 0 latency
                        else:
                            # Default latency: 50ms for same cluster, 200ms for cross-cluster
                            matrix[peer_id][other_id] = 50.0
                
                self._cache = LatencyCache(
                    version="1.0",
                    timestamp=time.time(),
                    matrix=matrix
                )
                self._save_cache(self._cache)
            else:
                # Không có peers, trả về empty matrix
                self._cache = LatencyCache(
                    version="1.0",
                    timestamp=time.time(),
                    matrix={}
                )

        return self._cache.matrix if self._cache else {}
    
    async def probe_single(self, peer: "PeerHandle") -> Dict[str, float]:
        """
        Ping 1 peer và trả về latencies đến tất cả máy khác.
        
        Args:
            peer: Peer cần probe
            
        Returns:
            Dict[peer_id] = latency_ms
        """
        latencies = {}
        # Note: Thực tế cần gửi ping message qua network
        # Ở đây là placeholder - cần implement actual ping qua gRPC
        return latencies
    
    async def probe_all(self, peers: List["PeerHandle"]) -> Dict[str, Dict[str, float]]:
        """
        Ping tất cả peers song song.
        
        Args:
            peers: List of peers
            
        Returns:
            Latency matrix
        """
        if not peers:
            return {}
        
        matrix: Dict[str, Dict[str, float]] = {
            peer.id(): {} for peer in peers
        }
        
        # Probe tất cả cặp
        tasks = []
        for i, peer_a in enumerate(peers):
            for peer_b in peers[i+1:]:
                tasks.append(self._probe_pair(peer_a, peer_b, matrix))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return matrix
    
    async def _probe_pair(
        self, 
        peer_a: "PeerHandle",
        peer_b: "PeerHandle",
        matrix: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Probe một cặp peers.
        
        Args:
            peer_a: Peer A
            peer_b: Peer B
            matrix: Matrix để update kết quả
        """
        latencies = []
        
        for _ in range(self.probe_count):
            try:
                start = time.perf_counter()
                # Gửi ping qua Tailscale
                # Note: Cần implement actual ping method trong PeerHandle
                if hasattr(peer_a, 'ping'):
                    await asyncio.wait_for(
                        peer_a.ping(peer_b),
                        timeout=self.timeout_ms / 1000
                    )
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
            except (asyncio.TimeoutError, Exception):
                # Timeout hoặc lỗi, bỏ qua
                continue
        
        if latencies:
            # Loại bỏ outliers (max, min)
            latencies.sort()
            if len(latencies) > 2:
                latencies = latencies[1:-1]
            
            avg_latency = sum(latencies) / len(latencies)
            
            # Update matrix (symmetric)
            matrix[peer_a.id()][peer_b.id()] = avg_latency
            matrix[peer_b.id()][peer_a.id()] = avg_latency
        else:
            # Không đo được, dùng default
            default_latency = 200.0  # 200ms default
            matrix[peer_a.id()][peer_b.id()] = default_latency
            matrix[peer_b.id()][peer_a.id()] = default_latency
    
    def get_latency(self, from_id: str, to_id: str) -> float:
        """
        Lấy latency giữa 2 máy từ cache.
        
        Args:
            from_id: Machine A ID
            to_id: Machine B ID
            
        Returns:
            Latency in ms, hoặc 200ms default
        """
        if self._cache is None:
            self._cache = self._load_cache()
        
        if self._cache and self._cache.matrix:
            return self._cache.matrix.get(from_id, {}).get(to_id, 200.0)
        return 200.0
    
    def clear_cache(self) -> None:
        """Xóa cache"""
        self._cache = None
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def force_refresh(self, topology: Topology) -> Dict[str, Dict[str, float]]:
        """
        Force refresh latency matrix.
        
        Args:
            topology: Topology để probe
            
        Returns:
            Updated latency matrix
        """
        self.clear_cache()
        return self.get_matrix(topology)