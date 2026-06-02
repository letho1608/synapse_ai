from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from .topology import Topology
from synapse.inference.shard import Shard
from synapse.topology.device_capabilities import device_capabilities
from synapse.topology.resource_weight import NodeResources
import asyncio


# Partitions shard-space into pieces of contiguous shards, represented by floating point range [start, end) between 0 and 1
@dataclass
class Partition:
  node_id: str
  start: float
  end: float


class PartitioningStrategy(ABC):
  @abstractmethod
  def partition(self, topology: Topology, base_shard: Optional[Shard] = None) -> List[Partition]:
    pass


class TFLOPSProportionalPartitioner(PartitioningStrategy):
    """
    Cake-inspired TFLOPS-proportional sharding.

    Each node's layer allocation is directly proportional to its GPU TFLOPS
    relative to the cluster total. This is simpler and more predictable than
    multi-resource scoring.

    Formula:
        node_i_layers = total_layers * (node_i_tflops / total_tflops)

    Uses Largest Remainder Method for integer allocation.
    """
    def __init__(self, min_layers_per_node: int = 1):
        self.min_layers_per_node = min_layers_per_node

    def _get_tflops(self, node_id: str, topology: Topology) -> float:
        dc = topology.nodes.get(node_id)
        if dc is None:
            return 0.0
        flops = dc.flops if hasattr(dc, "flops") else {}
        if hasattr(flops, "fp16"):
            return float(flops.fp16)
        if isinstance(flops, dict):
            return float(flops.get("fp16", 0))
        return 0.0

    def partition(self, topology: Topology, base_shard: Optional[Shard] = None) -> List[Partition]:
        num_layers = base_shard.n_layers if base_shard and base_shard.n_layers > 0 else 32
        node_ids = list(topology.nodes.keys())
        if not node_ids:
            return []

        if len(node_ids) == 1:
            return [Partition(node_id=node_ids[0], start=0.0, end=1.0)]

        tflops_map = {nid: self._get_tflops(nid, topology) for nid in node_ids}
        total_tflops = sum(tflops_map.values())

        if total_tflops == 0:
            equal_share = 1.0 / len(node_ids)
            return [Partition(node_id=nid, start=i * equal_share, end=(i + 1) * equal_share) for i, nid in enumerate(node_ids)]

        shares = {nid: (tflops / total_tflops) * num_layers for nid, tflops in tflops_map.items()}
        allocated = {nid: max(self.min_layers_per_node, int(s)) for nid, s in shares.items()}
        remainder = num_layers - sum(allocated.values())

        if remainder > 0:
            fractions = {nid: shares[nid] - int(shares[nid]) for nid in shares}
            sorted_ids = sorted(fractions.keys(), key=lambda k: fractions[k], reverse=True)
            for i in range(remainder):
                allocated[sorted_ids[i % len(sorted_ids)]] += 1

        current_layer = 0
        partitions = []
        for nid in node_ids:
            layers = allocated.get(nid, 0)
            if layers <= 0:
                continue
            start = current_layer / num_layers
            end = (current_layer + layers) / num_layers
            partitions.append(Partition(node_id=nid, start=start, end=end))
            current_layer += layers

        if current_layer < num_layers and partitions:
            partitions[-1].end = 1.0

        return partitions


def map_partitions_to_shards(partitions: List[Partition], num_layers: int, model_id: str) -> List[Shard]:
  shards = []
  for i, partition in enumerate(partitions):
    start_layer = int(partition.start*num_layers)
    end_layer = int(partition.end*num_layers) - 1

    # Ensure the last partition covers up to num_layers - 1
    if i == len(partitions) - 1:
      end_layer = num_layers - 1

    # Ensure no empty shards
    if start_layer <= end_layer:
      shards.append(Shard(model_id, start_layer, end_layer, num_layers))

  # Ensure full coverage
  if shards and shards[-1].end_layer < num_layers - 1:
    shards[-1] = Shard(model_id, shards[-1].start_layer, num_layers - 1, num_layers)

  return shards