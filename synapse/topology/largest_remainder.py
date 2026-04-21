from typing import List, Dict, Optional, Tuple
from loguru import logger
from synapse.topology.partitioning_strategy import PartitioningStrategy, Partition
from synapse.topology.topology import Topology
from synapse.inference.shard import Shard

class LargestRemainderPartitioningStrategy(PartitioningStrategy):
    """
    LACP 2.0: Robust partitioning using the Largest Remainder Method.
    Extracted from Exo's logic and adapted for Synapse.
    """
    def __init__(self, use_latency: bool = True):
        self.use_latency = use_latency

    def partition(self, topology: Topology, base_shard: Optional[Shard] = None) -> List[Partition]:
        """
        Divide model layers across nodes based on their memory and (optionally) latency weights.
        """
        nodes = topology.nodes
        if not nodes:
            return []

        # 1. Determine total layers
        num_layers = base_shard.n_layers if base_shard and base_shard.n_layers > 0 else 32
        
        # 2. Calculate Weights for each node
        # We start with Memory as the primary weight.
        weights = {}
        for node_id, node in nodes.items():
            # Memory weight (in MB)
            weights[node_id] = float(node.memory)
            
        # 3. Apply Largest Remainder Algorithm
        allocated_counts = self._allocate_layers(num_layers, weights)

        # 4. Convert to Partitions [start_pct, end_pct)
        partitions = []
        current_layer = 0
        
        # Sort nodes by ID for deterministic assignment
        sorted_node_ids = sorted(nodes.keys())
        
        for node_id in sorted_node_ids:
            count = allocated_counts.get(node_id, 0)
            if count == 0: continue
            
            start_pct = current_layer / num_layers
            current_layer += count
            end_pct = current_layer / num_layers
            
            partitions.append(Partition(
                node_id=node_id,
                start=start_pct,
                end=end_pct
            ))
            
        return partitions

    def _allocate_layers(self, num_layers: int, weights: Dict[str, float]) -> Dict[str, int]:
        """
        Core Largest Remainder Algorithm.
        """
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Fallback: even distribution
            total_weight = len(weights)
            weights = {nid: 1.0 for nid in weights}

        # Step 1: Initial allocation (floor of proportional share)
        shares = {nid: (w / total_weight) * num_layers for nid, w in weights.items()}
        allocated = {nid: int(s) for nid, s in shares.items()}
        remaining = num_layers - sum(allocated.values())

        # Step 2: Distribute remaining layers to nodes with largest fractional parts
        fractions = {nid: s - int(s) for nid, s in shares.items()}
        sorted_nodes_by_fraction = sorted(fractions.keys(), key=lambda k: fractions[k], reverse=True)

        for i in range(remaining):
            nid = sorted_nodes_by_fraction[i]
            allocated[nid] += 1

        return allocated
