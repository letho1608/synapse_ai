"""
Latency Clustering - Hierarchical clustering dựa trên latency matrix
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class MachineCluster:
    """
    Một cluster chứa các machines có latency gần nhau.
    """
    id: int
    machine_ids: List[str]
    avg_latency: float  # Average latency trong cluster
    
    def contains(self, machine_id: str) -> bool:
        """Kiểm tra machine có trong cluster không"""
        return machine_id in self.machine_ids
    
    def __repr__(self) -> str:
        return f"MachineCluster(id={self.id}, machines={len(self.machine_ids)}, avg_latency={self.avg_latency:.2f}ms)"


class HierarchicalClusterer:
    """
    Hierarchical clustering dựa trên latency matrix.
    
    Sử dụng agglomerative clustering để nhóm các machines
    có latency thấp vào cùng cluster.
    """
    
    def __init__(self, default_threshold_ms: float = 50.0):
        """
        Args:
            default_threshold_ms: Ngưỡng latency để merge clusters (ms)
        """
        self.default_threshold_ms = default_threshold_ms
    
    def cluster(
        self,
        latency_matrix: Dict[str, Dict[str, float]],
        threshold_ms: Optional[float] = None
    ) -> List[MachineCluster]:
        """
        Thực hiện agglomerative clustering.
        
        Args:
            latency_matrix: Dict[machine_id][machine_id] = latency_ms
            threshold_ms: Ngưỡng latency để merge (ms)
            
        Returns:
            List of MachineCluster
        """
        if not latency_matrix:
            return []
        
        if not SCIPY_AVAILABLE:
            # Fallback: không dùng scipy, trả về mỗi machine là 1 cluster
            return self._simple_cluster(latency_matrix)
        
        threshold = threshold_ms or self.default_threshold_ms
        machine_ids = list(latency_matrix.keys())
        n = len(machine_ids)
        
        if n < 2:
            # Chỉ có 1 machine
            return [MachineCluster(
                id=0,
                machine_ids=machine_ids,
                avg_latency=0.0
            )]
        
        # Build distance matrix (upper triangular form)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                lat = latency_matrix[machine_ids[i]].get(machine_ids[j], 200.0)
                distances.append(lat)
        
        # Hierarchical clustering
        dist_array = np.array(distances)
        Z = linkage(dist_array, method='average')
        
        # Cut dendrogram at threshold
        cluster_labels = fcluster(Z, t=threshold, criterion='distance')
        
        # Build clusters
        cluster_dict: Dict[int, List[str]] = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(machine_ids[i])
        
        # Create MachineCluster objects
        result = []
        for cluster_id, machine_list in cluster_dict.items():
            avg_lat = self._calc_avg_cluster_latency(machine_list, latency_matrix)
            result.append(MachineCluster(
                id=cluster_id,
                machine_ids=machine_list,
                avg_latency=avg_lat
            ))
        
        return result
    
    def _simple_cluster(
        self, 
        latency_matrix: Dict[str, Dict[str, float]]
    ) -> List[MachineCluster]:
        """
        Simple fallback clustering khi scipy không available.
        Mỗi machine thành 1 cluster.
        """
        machine_ids = list(latency_matrix.keys())
        return [MachineCluster(
            id=i,
            machine_ids=[mid],
            avg_latency=0.0
        ) for i, mid in enumerate(machine_ids)]
    
    def _calc_avg_cluster_latency(
        self,
        machine_ids: List[str],
        latency_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """Tính average latency trong cluster"""
        total = 0.0
        count = 0
        for i, m1 in enumerate(machine_ids):
            for m2 in machine_ids[i + 1:]:
                lat = latency_matrix.get(m1, {}).get(m2, 200.0)
                total += lat
                count += 1
        return total / count if count > 0 else 0.0
    
    def get_cluster_for_machine(
        self,
        machine_id: str,
        clusters: List[MachineCluster]
    ) -> Optional[MachineCluster]:
        """
        Tìm cluster chứa machine.
        
        Args:
            machine_id: Machine ID
            clusters: List of clusters
            
        Returns:
            MachineCluster hoặc None
        """
        for cluster in clusters:
            if cluster.contains(machine_id):
                return cluster
        return None
    
    def are_in_same_cluster(
        self,
        machine_a: str,
        machine_b: str,
        clusters: List[MachineCluster]
    ) -> bool:
        """
        Kiểm tra 2 machines có trong cùng cluster không.
        
        Args:
            machine_a: Machine A ID
            machine_b: Machine B ID
            clusters: List of clusters
            
        Returns:
            True nếu cùng cluster
        """
        cluster_a = self.get_cluster_for_machine(machine_a, clusters)
        if cluster_a is None:
            return False
        return cluster_a.contains(machine_b)
    
    def get_cross_cluster_penalty(
        self,
        machine_a: str,
        machine_b: str,
        clusters: List[MachineCluster],
        base_penalty: float = 1000.0
    ) -> float:
        """
        Tính penalty khi 2 machines khác cluster.
        
        Args:
            machine_a: Machine A ID
            machine_b: Machine B ID
            clusters: List of clusters
            base_penalty: Penalty cơ bản
            
        Returns:
            Penalty value
        """
        if self.are_in_same_cluster(machine_a, machine_b, clusters):
            return 0.0
        
        # Penalty cao hơn nếu machines ở xa nhau
        cluster_a = self.get_cluster_for_machine(machine_a, clusters)
        cluster_b = self.get_cluster_for_machine(machine_b, clusters)
        
        if cluster_a and cluster_b:
            # Penalty = base + difference in avg latency
            return base_penalty + abs(cluster_a.avg_latency - cluster_b.avg_latency)
        
        return base_penalty