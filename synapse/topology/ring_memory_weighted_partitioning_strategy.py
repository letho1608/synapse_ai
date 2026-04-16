from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    
    # Try to partition by Warmup Throughput (Real Performance) first
    total_throughput = sum(node[1].warmup_throughput for node in nodes)
    
    if total_throughput > 0:
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
    total_flops = sum(node[1].flops.fp16 for node in nodes)
    
    if total_flops > 0:
      # Sort by FLOPS descending
      nodes.sort(key=lambda x: (x[1].flops.fp16, x[0]), reverse=True)
      partitions = []
      start = 0
      for node in nodes:
        # Cơ bản: Chia theo FLOPS
        # Nâng cao (Giai đoạn 2): Kết hợp Băng thông mạng
        # score = node[1].flops.fp16 * (node[1].bandwidth_mbps / 1000.0 if node[1].bandwidth_mbps > 0 else 1.0)
        share = node[1].flops.fp16 / total_flops
        end = round(start + share, 5)
        # Sử dụng trọng số tính toán được để gán vào Partition
        partitions.append(Partition(node[0], start, end, weight=share))
        start = end
      return partitions
      
    # Fallback to Memory if FLOPS info is missing (total_flops == 0)
    nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True)
    total_memory = sum(node[1].memory for node in nodes)
    partitions = []
    start = 0
    for node in nodes:
      share = node[1].memory / total_memory
      end = round(start + share, 5)
      partitions.append(Partition(node[0], start, end, weight=share))
      start = end
    return partitions
