from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Set
import numpy as np
from synapse.inference.shard import Shard
from synapse.topology.device_capabilities import DeviceCapabilities
from synapse.topology.topology import Topology


class PeerHandle(ABC):
  @abstractmethod
  def id(self) -> str:
    pass

  @abstractmethod
  def addr(self) -> str:
    pass

  @abstractmethod
  def description(self) -> str:
    pass

  @abstractmethod
  def device_capabilities(self) -> DeviceCapabilities:
    pass

  @abstractmethod
  async def connect(self) -> None:
    pass

  @abstractmethod
  async def is_connected(self) -> bool:
    pass

  @abstractmethod
  async def disconnect(self) -> None:
    pass

  @abstractmethod
  async def health_check(self) -> bool:
    pass

  @abstractmethod
  async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_tensor(self, shard: Shard, tensor: np.array, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    pass

  @abstractmethod
  async def collect_topology(self, visited: Set[str], max_depth: int) -> Topology:
    pass

  @abstractmethod
  async def send_example(self, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray, train: bool, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def sync_weights(self, model_id: str, weights: np.ndarray, step: int) -> np.ndarray:
    pass

  @abstractmethod
  async def test_network(self, payload: bytes) -> float:
    pass

  @abstractmethod
  async def profile_hardware(self, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray, n_iters: int, skip_iters: int) -> Tuple[float, float]:
    pass

  @abstractmethod
  async def setup_ring(self, rank: int, world_size: int, successor_url: str) -> None:
    pass

  @abstractmethod
  async def transfer_chunk(self, chunk_index: int, tensor: np.ndarray, step_type: str) -> None:
    pass

  @abstractmethod
  async def trigger_ring_allreduce(self, model_id: str) -> None:
    pass
