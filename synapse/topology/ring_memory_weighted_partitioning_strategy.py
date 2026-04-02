from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    
    # Try to partition by FLOPS (Compute) first
    total_flops = sum(node[1].flops.fp16 for node in nodes)
    
    if total_flops > 0:
      # Sort by FLOPS descending
      nodes.sort(key=lambda x: (x[1].flops.fp16, x[0]), reverse=True)
      partitions = []
      start = 0
      for node in nodes:
        share = node[1].flops.fp16 / total_flops
        end = round(start + share, 5)
        partitions.append(Partition(node[0], start, end))
        start = end
      return partitions
      
    # Fallback to Memory if FLOPS info is missing (total_flops == 0)
    nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True)
    total_memory = sum(node[1].memory for node in nodes)
    partitions = []
    start = 0
    for node in nodes:
      end = round(start + (node[1].memory/total_memory), 5)
      partitions.append(Partition(node[0], start, end))
      start = end
    return partitions
