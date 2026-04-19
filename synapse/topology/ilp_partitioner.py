"""
ILP Partitioner - ILP-based partitioning cho LACP
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    import pulp as _pulp_module
    from pulp import (
        LpProblem, LpMinimize, LpVariable, LpBinary,
        LpSum, PULP_CBC_CMD
    )
    LP_STATUS_OPTIMAL = _pulp_module.LpStatusOptimal  # = 1
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    LP_STATUS_OPTIMAL = 1  # fallback constant
    # Define placeholder classes for when pulp is not available
    LpProblem = None
    LpMinimize = None
    LpVariable = None
    LpBinary = None
    LpSum = None
    PULP_CBC_CMD = None

from synapse.topology.partitioning_strategy import Partition
from synapse.topology.topology import Topology
from synapse.topology.latency_clustering import MachineCluster


@dataclass
class LayerProfile:
    """
    Profile của một layer (memory, compute).
    """
    layer_id: int
    memory_mb: float
    compute_flops: float = 0.0


class ILPPartitioner:
    """
    ILP-based partitioner cho LACP.
    
    Tìm partition tối ưu bằng cách giải ILP problem:
    - Minimize weighted latency cost
    - Respect memory constraints
    - Consider cluster assignments
    """
    
    def __init__(self, timeout: int = 60):
        """
        Args:
            timeout: Timeout cho ILP solver (seconds)
        """
        self.timeout = timeout
    
    def find_optimal(
        self,
        topology: Topology,
        latency_matrix: Dict[str, Dict[str, float]],
        clusters: List[MachineCluster],
        layer_profiles: List[LayerProfile],
        timeout: Optional[int] = None
    ) -> List[Partition]:
        """
        Tìm partition tối ưu.
        
        Args:
            topology: Topology với các nodes
            latency_matrix: Latency giữa các machines
            clusters: Machine clusters
            layer_profiles: Profile của các layers
            timeout: Timeout override
            
        Returns:
            List of Partition
        """
        if not PULP_AVAILABLE:
            # Fallback to greedy
            return self._greedy_fallback(topology, latency_matrix, layer_profiles)
        
        try:
            return self._solve_ilp(
                topology, latency_matrix, clusters, layer_profiles,
                timeout or self.timeout
            )
        except Exception as e:
            print(f"ILP solving failed: {e}, falling back to greedy")
            return self._greedy_fallback(topology, latency_matrix, layer_profiles)
    
    def _solve_ilp(
        self,
        topology: Topology,
        latency_matrix: Dict[str, Dict[str, float]],
        clusters: List[MachineCluster],
        layer_profiles: List[LayerProfile],
        timeout: int
    ) -> List[Partition]:
        """
        Giải ILP problem.
        
        Variables:
        - x[i][m] = 1 nếu layer i được gán cho machine m
        
        Objective:
        - Minimize sum of weighted latency cost cho consecutive layers
        
        Constraints:
        - Each layer assigned to exactly one machine
        - Memory constraint per machine
        """
        machines = sorted(topology.nodes.keys())
        n_layers = len(layer_profiles)
        n_machines = len(machines)
        
        if n_machines == 0 or n_layers == 0:
            return []
        
        # Create LP problem
        prob = LpProblem("LACP_Partitioning", LpMinimize)
        
        # Variables: x[i][m] = 1 if layer i assigned to machine m
        x = {}
        for i in range(n_layers):
            for m in range(n_machines):
                x[i, m] = LpVariable(
                    f"x_{i}_{m}",
                    cat=LpBinary
                )
        
        # Objective: Minimize weighted latency cost
        latency_cost = []
        for i in range(n_layers - 1):
            for m1 in range(n_machines):
                for m2 in range(n_machines):
                    if m1 != m2:
                        machine_id_1 = machines[m1]
                        machine_id_2 = machines[m2]
                        lat = latency_matrix.get(machine_id_1, {}).get(
                            machine_id_2, 200.0
                        )
                        # y[i,i+1] >= x[i][m1] + x[i+1][m2] - 1
                        y = LpVariable(f"y_{i}_{m1}_{m2}", cat=LpBinary)
                        prob += y >= x[i, m1] + x[i + 1, m2] - 1
                        latency_cost.append(lat * y)
        
        prob += LpSum(latency_cost)
        
        # Constraint 1: Each layer assigned to exactly one machine
        for i in range(n_layers):
            prob += LpSum(x[i, m] for m in range(n_machines)) == 1
        
        # Constraint 2: Memory constraint per machine
        machine_memory = {
            m: topology.nodes[m].memory
            for m in machines
        }
        for m in range(n_machines):
            mem_constraint = []
            for i in range(n_layers):
                mem_constraint.append(
                    layer_profiles[i].memory_mb * x[i, m]
                )
            prob += LpSum(mem_constraint) <= machine_memory[machines[m]]
        
        # Solve
        solver = PULP_CBC_CMD(timeLimit=timeout)
        prob.solve(solver)
    
        # Check solution status before extracting
        # LpStatusOptimal = 1, LpStatusNotSolved = 0, LpStatusInfeasible = -1, etc.
        if prob.status != LP_STATUS_OPTIMAL:
            # Solution is not optimal (could be infeasible, undefined, etc.)
            # Fallback to greedy
            print(f"[ILP] Status={prob.status} (not optimal), falling back to greedy")
            return self._greedy_fallback(topology, latency_matrix, layer_profiles)
    
        # Extract solution
        partitions = self._extract_partitions(
            x, machines, layer_profiles, n_layers
        )
    
        return partitions
    
    def _extract_partitions(
        self,
        x: Dict[Tuple[int, int], Any],
        machines: List[str],
        layer_profiles: List[LayerProfile],
        n_layers: int
    ) -> List[Partition]:
        """Extract partition boundaries từ ILP solution"""
        # Determine which machine each layer belongs to
        layer_to_machine = {}
        for i in range(n_layers):
            for m, machine_id in enumerate(machines):
                var = x.get((i, m))
                if var and var.value() and var.value() > 0.5:
                    layer_to_machine[i] = machine_id
                    break
        
        # Group consecutive layers by machine
        partitions = []
        current_machine = None
        start_layer = 0
        
        for i in range(n_layers):
            machine = layer_to_machine.get(i)
            if machine is None:
                machine = machines[0]  # Default fallback
            
            if machine != current_machine:
                if current_machine is not None:
                    # Save previous partition
                    partitions.append(Partition(
                        node_id=current_machine,
                        start=start_layer / n_layers,
                        end=i / n_layers
                    ))
                current_machine = machine
                start_layer = i
        
        # Last partition
        if current_machine is not None:
            partitions.append(Partition(
                node_id=current_machine,
                start=start_layer / n_layers,
                end=1.0
            ))
        
        return partitions
    
    def _greedy_fallback(
        self,
        topology: Topology,
        latency_matrix: Dict[str, Dict[str, float]],
        layer_profiles: List[LayerProfile]
    ) -> List[Partition]:
        """
        Fallback algorithm nếu ILP timeout, unavailable, hoặc infeasible.
        Sử dụng phân bổ tỉ lệ theo RAM (Ring Memory Weighted) để tránh bị 
        fragmentation tạo ra nhiều shard trên cùng một máy, dẫn đến swap memory.
        """
        machines = sorted(topology.nodes.keys(), key=lambda m: (-topology.nodes[m].memory, m))
        if not machines:
            return []

        total_mem = sum(topology.nodes[m].memory for m in machines)
        if total_mem <= 0:
            total_mem = 1.0

        partitions = []
        current_start = 0.0

        for i, m in enumerate(machines):
            fraction = topology.nodes[m].memory / total_mem
            end = current_start + fraction
            
            # Đảm bảo partition cuối cùng luôn bao phủ đến hết
            if i == len(machines) - 1:
                end = 1.0
                
            if end > current_start:
                partitions.append(Partition(
                    node_id=m, 
                    start=current_start, 
                    end=end
                ))
            current_start = end

        return partitions
    
    def calculate_leaf_score(
        self,
        machine_id: str,
        topology: Topology,
        clusters: List[MachineCluster]
    ) -> float:
        """
        Tính leaf score cho machine.
        
        Leaf node = machine có ít external connections.
        Score càng cao = càng được ưu tiên.
        
        Args:
            machine_id: Machine ID
            topology: Topology
            clusters: Clusters
            
        Returns:
            Leaf score (0-1)
        """
        from synapse.topology.latency_clustering import HierarchicalClusterer
        
        clusterer = HierarchicalClusterer()
        external_connections = 0
        
        for peer_id in topology.nodes.keys():
            if peer_id == machine_id:
                continue
            # Đếm kết nối ra ngoài cluster
            if not clusterer.are_in_same_cluster(machine_id, peer_id, clusters):
                external_connections += 1
        
        # Score cao nếu ít external connections
        return 1.0 / (1.0 + external_connections)
    
    def calculate_download_score(
        self,
        machine_id: str,
        model_id: str,
        cache_dir: str
    ) -> float:
        """
        Tính download score cho machine.
        
        Machine đã download model → tiết kiệm thời gian, được ưu tiên.
        
        Args:
            machine_id: Machine ID
            model_id: Model ID
            cache_dir: Cache directory
            
        Returns:
            Download score (0-1)
        """
        from pathlib import Path
        
        cache_path = Path(cache_dir) / model_id / machine_id
        if cache_path.exists():
            # Đã download đầy đủ
            return 1.0
        elif (cache_path / ".downloading").exists():
            # Đang download một phần
            return 0.5
        else:
            # Chưa download
            return 0.0