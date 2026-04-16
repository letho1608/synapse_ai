from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    
    # Only use throughput if ALL nodes have a valid, non-zero measurement.
    # If any node has 0, it means it's newly joined or hasn't profiled yet, 
    # so we should fall back to hardware-based weighting for fairness.
    all_have_tp = all(node[1].warmup_throughput > 0 for node in nodes)
    total_throughput = sum(node[1].warmup_throughput for node in nodes)
    
    if all_have_tp and total_throughput > 0:
      # Sort by throughput descending
      nodes.sort(key=lambda x: (x[1].warmup_throughput, x[0]), reverse=True)
      partitions = []
      start = 0
      for node in nodes:
        share = node[1].warmup_throughput / total_throughput
        end = round(start + share, 5)
        partitions.append(Partition(node[0], start, end, weight=share))
        start = end
      return partitions

    # Fallback to FLOPS (Compute)
    # Ensure every node has at least a baseline weight (0.1 TFLOPS) so they aren't ignored
    # even if their specs haven't been fully retrieved yet.
    total_flops = sum(max(0.1, node[1].flops.fp16) for node in nodes)
    
    if total_flops > 0:
      nodes.sort(key=lambda x: (x[1].flops.fp16, x[0]), reverse=True)
      partitions = []
      start = 0
      for node in nodes:
        share = max(0.1, node[1].flops.fp16) / total_flops
        end = round(start + share, 5)
        partitions.append(Partition(node[0], start, end, weight=share))
        start = end
      return partitions
      
    # Fallback to Memory if FLOPS info is missing
    # Ensure every node has at least 1GB baseline
    total_memory = sum(max(1, node[1].memory) for node in nodes)
    nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True)
    partitions = []
    start = 0
    for node in nodes:
      share = max(1, node[1].memory) / total_memory
      end = round(start + share, 5)
      partitions.append(Partition(node[0], start, end, weight=share))
      start = end
    return partitions
