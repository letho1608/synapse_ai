import numpy as np
import os
from synapse.helpers import DEBUG  # Make sure to import DEBUG

from typing import Tuple, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
from .shard import Shard

if TYPE_CHECKING:
    from synapse.loading import ShardDownloader


class InferenceEngine(ABC):
  session = {}

  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    pass

  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    pass

  @abstractmethod
  async def load_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_checkpoint(self, shard: Shard, path: str):
    pass

  async def train(self, request_id: str, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray, backgrad: Optional[np.ndarray] = None, loss: str = "forward") -> Tuple[Optional[float], Optional[np.ndarray]]:
      """Training phân tán (chia layer): last layer tính loss; các layer khác nhận backgrad. Subclass nên override."""
      # Default: chỉ forward, không training (对于 chỉ forward qua các layer)
      return None, None
  
  async def evaluate(self, request_id: str, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray) -> Optional[float]:
      """Đánh giá (eval) phân tán. Subclass nên override."""
      # Default: chỉ forward, không tính loss
      return None

  async def save_session(self, key, value):
    self.session[key] = value

  async def clear_session(self):
    self.session.clear()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    tokens = await self.encode(shard, prompt)
    
    # Kiểm tra tokens rỗng - đây là nguyên nhân gốc rễ
    if tokens.size == 0 or len(tokens) == 0:
      # Nếu prompt không encode được token nào, trả về logits rỗng
      # Điều này có thể xảy ra với prompt rỗng hoặc chỉ có whitespace
      if DEBUG >= 2:
        print(f"[{request_id}] Warning: Empty tokens from prompt '{prompt}', returning empty logits")
      vocab_size = 50257  # Default GPT-2 vocab size, sẽ được override nếu model có vocab_size
      empty_logits = np.zeros((1, 0, vocab_size), dtype=np.float32)
      return empty_logits, inference_state
    
    if shard.model_id != 'stable-diffusion-2-1-base':
      x = tokens.reshape(1, -1)
    else:
      x = tokens
    output_data, inference_state = await self.infer_tensor(request_id, shard, x, inference_state)

    return output_data, inference_state


inference_engine_classes = {
  "pytorch": "PyTorchHFInferenceEngine",
}


def get_inference_engine(inference_engine_name: str, shard_downloader: "ShardDownloader" = None):
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")
  
  supported_engines = ["pytorch"]

  if inference_engine_name == "pytorch":
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine, HFShardDownloader, register_pytorch_models
    register_pytorch_models()
    downloader = shard_downloader
    if downloader is None:
      downloader = HFShardDownloader()
    return PyTorchHFInferenceEngine(shard_downloader=downloader)

  raise ValueError(f"Unsupported inference engine: '{inference_engine_name}'. Supported engines: {supported_engines}")
