from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from .topology import Topology
from synapse.inference.shard import Shard

@dataclass
class Partition:
  node_id: str
  start: float
  end: float
  weight: float = 1.0

class PartitioningStrategy(ABC):
  @abstractmethod
  def partition(self, topology: Topology) -> List[Partition]:
    pass

def map_partitions_to_shards(partitions: List[Partition], total_layers: int, model_id: str) -> List[Shard]:
  # Helper function to convert partitions to Shard objects
  results = []
  current_start = 0
  for i, p in enumerate(partitions):
    # Tính số lượng layer cho partition này dựa trên share (end - start)
    share = p.end - p.start
    n_shard_layers = max(1, round(share * total_layers))
    
    s_layer = current_start
    e_layer = s_layer + n_shard_layers - 1
    
    # Đảm bảo node cuối cùng luôn bao phủ tới layer cuối cùng (n_layers - 1)
    if i == len(partitions) - 1 or e_layer >= total_layers:
      e_layer = total_layers - 1
      
    # Trường hợp hy hữu s_layer vượt quá giới hạn
    if s_layer >= total_layers:
      s_layer = total_layers - 1
      e_layer = total_layers - 1

    results.append(Shard(
      model_id=model_id,
      start_layer=s_layer,
      end_layer=e_layer,
      n_layers=total_layers
    ))
    
    # Node tiếp theo bắt đầu từ layer kế tiếp
    current_start = e_layer + 1
    
  return results
