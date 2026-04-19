"""
LACP Partitioning - Latency-Aware Collaborative Partitioning

Main entry point cho LACP strategy:
1. Probe latency giữa các máy
2. Hierarchical clustering
3. ILP-based partitioning tối ưu
"""
from typing import List, Optional, Dict, Any
import asyncio

from synapse.topology.partitioning_strategy import PartitioningStrategy, Partition
from synapse.topology.topology import Topology
from synapse.inference.shard import Shard
from synapse.topology.latency_clustering import HierarchicalClusterer, MachineCluster
from synapse.topology.ilp_partitioner import ILPPartitioner, LayerProfile
from synapse.networking.latency_probing import LatencyProber
from synapse.resources.resource_manager_registry import ResourceManagerRegistry


class LACPPartitioningStrategy(PartitioningStrategy):
    """
    Latency-Aware Collaborative Partitioning (LACP).
    
    Thuật toán 3 giai đoạn:
    1. Latency Probing: Đo latency thực giữa các máy
    2. Hierarchical Clustering: Nhóm máy theo latency
    3. ILP Partitioning: Chia layers tối ưu dùng ILP
    
    Benefits so với RingMemoryWeightedPartitioningStrategy:
    - Đo latency thực tế thay vì ước lượng
    - Nhóm máy gần nhau để giảm cross-network traffic
    - ILP tối ưu hơn greedy approach
    - Tận dụng CPU RAM, Disk, Multi-GPU
    """
    
    def __init__(
        self,
        latency_cache_file: str = "synapse/config/latency_cache.json",
        cluster_threshold_ms: float = 50.0,
        ilp_timeout: int = 60,
        enable_resource_managers: bool = True
    ):
        """
        Args:
            latency_cache_file: File để cache latency matrix
            cluster_threshold_ms: Ngưỡng latency để merge clusters
            ilp_timeout: Timeout cho ILP solver (seconds)
            enable_resource_managers: Enable resource managers
        """
        self.latency_cache_file = latency_cache_file
        self.cluster_threshold_ms = cluster_threshold_ms
        self.ilp_timeout = ilp_timeout
        self.enable_resource_managers = enable_resource_managers
        
        # Lazy initialization
        self._latency_prober: Optional[LatencyProber] = None
        self._clusterer: Optional[HierarchicalClusterer] = None
        self._ilp_solver: Optional[ILPPartitioner] = None
        self._resource_managers: Optional[ResourceManagerRegistry] = None
        
        # Cached values
        self._cached_latency_matrix: Optional[Dict[str, Dict[str, float]]] = None
        self._cached_clusters: Optional[List[MachineCluster]] = None
        # Cache kết quả phân vùng cuối cùng
        self._cached_partitions: Optional[List[Partition]] = None
        self._cached_topology_fingerprint: Optional[str] = None
        self._cached_model_id: Optional[str] = None

    def _get_topology_fingerprint(self, topology: Topology) -> str:
        """Tạo fingerprint duy nhất cho topology dựa trên node IDs và memory."""
        nodes_info = []
        for node_id in sorted(topology.nodes.keys()):
            node = topology.nodes[node_id]
            nodes_info.append(f"{node_id}:{node.memory}")
        return "|".join(nodes_info)
    
    def partition(self, topology: Topology, base_shard: Optional[Shard] = None) -> List[Partition]:
        """
        Main entry point - chia model dựa trên LACP với cơ chế cache.
        
        Args:
            topology: Topology chứa các nodes
            base_shard: Tham chiếu shard model đang chạy
            
        Returns:
            List of Partition
        """
        nodes = list(topology.nodes.keys())
        if len(nodes) <= 1:
            node_id = nodes[0] if nodes else "localhost"
            return [Partition(node_id=node_id, start=0.0, end=1.0)]

        # --- Kiểm tra Cache ---
        current_fingerprint = self._get_topology_fingerprint(topology)
        current_model_id = base_shard.model_id if base_shard else "unknown"
        
        if (self._cached_partitions and 
            self._cached_topology_fingerprint == current_fingerprint and 
            self._cached_model_id == current_model_id):
            return self._cached_partitions

        # 1. Get latency matrix
        latency_matrix = self._get_latency_prober().get_matrix(topology)
        
        # 2. Cluster machines
        clusters = self._get_clusterer().cluster(
            latency_matrix,
            threshold_ms=self.cluster_threshold_ms
        )
        
        # 3. Get layer profiles (memory, compute per layer)
        layer_profiles = self._get_layer_profiles(topology, base_shard)
        
        # 4. ILP partitioning
        partitions = self._get_ilp_solver().find_optimal(
            topology=topology,
            latency_matrix=latency_matrix,
            clusters=clusters,
            layer_profiles=layer_profiles,
            timeout=self.ilp_timeout
        )
        partitions = sorted(partitions, key=lambda p: (p.start, p.end, p.node_id))
        
        # Cập nhật cache
        self._cached_latency_matrix = latency_matrix
        self._cached_clusters = clusters
        self._cached_partitions = partitions
        self._cached_topology_fingerprint = current_fingerprint
        self._cached_model_id = current_model_id
        
        return partitions
    
    def _get_latency_prober(self) -> LatencyProber:
        """Get lazy-initialized latency prober"""
        if self._latency_prober is None:
            self._latency_prober = LatencyProber(self.latency_cache_file)
        return self._latency_prober
    
    def _get_clusterer(self) -> HierarchicalClusterer:
        """Get lazy-initialized clusterer"""
        if self._clusterer is None:
            self._clusterer = HierarchicalClusterer(
                default_threshold_ms=self.cluster_threshold_ms
            )
        return self._clusterer
    
    def _get_ilp_solver(self) -> ILPPartitioner:
        """Get lazy-initialized ILP solver"""
        if self._ilp_solver is None:
            self._ilp_solver = ILPPartitioner(timeout=self.ilp_timeout)
        return self._ilp_solver
    
    def _get_layer_profiles(self, topology: Topology, base_shard: Optional[Shard] = None) -> List[LayerProfile]:
        """
        Get layer profiles từ topology.
        
        Nếu base_shard được truyền thì tra cứu thông tin model thực tế từ model_list,
        ước lượng số lượng layer và RAM chuẩn. Nếu không có, giữ base model mẫu.
        
        Args:
            topology: Topology
            base_shard: Thông tin current inference shard
            
        Returns:
            List of LayerProfile
        """
        from synapse.model_list import HF_MODEL_LAYERS, HF_MODEL_PARAMS
        
        num_layers = 32
        memory_per_layer_mb = 500.0  # default cho model 7B
        
        if base_shard:
            model_key = base_shard.model_id
            
            # Khởi tạo mặc định theo base_shard
            if base_shard.n_layers and base_shard.n_layers > 0:
                num_layers = base_shard.n_layers
                # Heuristic thực tế theo số tầng:
                # ≤16 tầng → model cực nhỏ (<100M params) → ~50MB/layer
                # ≤24 tầng → model nhỏ (0.5B-1.5B) → ~100MB/layer
                # ≤36 tầng → model trung (3B-7B)     → ~400MB/layer
                # >36 tầng → model lớn (13B+)         → ~800MB/layer
                if num_layers <= 16:
                    memory_per_layer_mb = 50.0
                elif num_layers <= 24:
                    memory_per_layer_mb = 100.0
                elif num_layers <= 36:
                    memory_per_layer_mb = 400.0
                else:
                    memory_per_layer_mb = 800.0

            # Sử dụng Local DB nếu có
            from synapse.helpers import _load_model_db
            db = _load_model_db()
            db_entry = None
            for entry in db:
                if entry.get("name") == model_key:
                    db_entry = entry
                    break
                    
            if db_entry:
                if db_entry.get("num_hidden_layers"):
                    num_layers = db_entry["num_hidden_layers"]
                
                # Estimate total mem from recommended_ram_gb
                mem_gb = db_entry.get("recommended_ram_gb", 14.0)
                total_mem_mb = mem_gb * 1024
                # 20% buffer thay vì cộng cứng 300MB
                memory_per_layer_mb = (total_mem_mb / num_layers) * 1.2
                
            elif model_key and model_key in HF_MODEL_LAYERS:
                num_layers = HF_MODEL_LAYERS[model_key]
                param_str = HF_MODEL_PARAMS.get(model_key, "7B")
                
                # Estimate total mem based on B/M param string
                if "B" in param_str:
                    try:
                        billions = float(param_str.replace("B", ""))
                        total_mem_mb = billions * 2000  # FP16: 2 bytes per param + overhead
                    except ValueError:
                        total_mem_mb = 7000 * 2
                elif "M" in param_str:
                    try:
                        millions = float(param_str.replace("M", ""))
                        total_mem_mb = millions * 2
                    except ValueError:
                        total_mem_mb = 1000
                else:
                    total_mem_mb = 7000 * 2
                
                # 20% buffer (thay vì +300MB cố định)
                memory_per_layer_mb = (total_mem_mb / num_layers) * 1.2
        
        return [
            LayerProfile(
                layer_id=i,
                memory_mb=memory_per_layer_mb,
                compute_flops=0.0
            )
            for i in range(num_layers)
        ]
    
    def _get_resource_managers(self) -> Optional[ResourceManagerRegistry]:
        """Get lazy-initialized resource managers"""
        if not self.enable_resource_managers:
            return None
        
        if self._resource_managers is None:
            try:
                from synapse.resources import ResourceManagerRegistry
                self._resource_managers = ResourceManagerRegistry()
            except ImportError:
                return None
        
        return self._resource_managers
    
    def get_latency_matrix(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get cached latency matrix"""
        return self._cached_latency_matrix
    
    def get_clusters(self) -> Optional[List[MachineCluster]]:
        """Get cached clusters"""
        return self._cached_clusters
    
    def get_resource_managers(self) -> Optional[ResourceManagerRegistry]:
        """Get resource managers"""
        return self._get_resource_managers()
    
    def force_refresh(self, topology: Topology) -> List[Partition]:
        """
        Force refresh latency matrix và recalculate partitions.
        
        Args:
            topology: Topology
            
        Returns:
            Updated partitions
        """
        # Clear cache
        self._latency_prober = None
        self._cached_latency_matrix = None
        self._cached_clusters = None
        
        # Recalculate
        return self.partition(topology)
    
    def set_layer_profiles(self, profiles: List[LayerProfile]) -> None:
        """
        Set layer profiles (gọi trước khi partition).
        
        Args:
            profiles: List of LayerProfile
        """
        # Store for later use in _get_layer_profiles
        self._custom_layer_profiles = profiles
    
    def _get_layer_profiles_with_custom(
        self, 
        topology: Topology
    ) -> List[LayerProfile]:
        """Get layer profiles, dùng custom nếu có"""
        if hasattr(self, '_custom_layer_profiles'):
            return self._custom_layer_profiles
        return self._get_layer_profiles(topology)
    
    def get_cluster_for_machine(
        self, 
        machine_id: str
    ) -> Optional[MachineCluster]:
        """
        Get cluster chứa machine.
        
        Args:
            machine_id: Machine ID
            
        Returns:
            MachineCluster hoặc None
        """
        if self._cached_clusters is None:
            return None
        return self._get_clusterer().get_cluster_for_machine(
            machine_id, self._cached_clusters
        )
    
    def are_in_same_cluster(
        self, 
        machine_a: str, 
        machine_b: str
    ) -> bool:
        """
        Kiểm tra 2 machines có trong cùng cluster không.
        
        Args:
            machine_a: Machine A ID
            machine_b: Machine B ID
            
        Returns:
            True nếu cùng cluster
        """
        if self._cached_clusters is None:
            return False
        return self._get_clusterer().are_in_same_cluster(
            machine_a, machine_b, self._cached_clusters
        )
    
    def __repr__(self) -> str:
        return (
            f"LACPPartitioningStrategy("
            f"cluster_threshold={self.cluster_threshold_ms}ms, "
            f"ilp_timeout={self.ilp_timeout}s"
            f")"
        )