"""
PyTorch + Hugging Face inference engine.
Phân tán: load model từ HF. Danh sách model lấy từ synapse.model_list (download + web UI).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from synapse.helpers import DEBUG
from synapse.inference.inference_engine import InferenceEngine
from synapse.inference.shard import Shard
from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS, resolve_hf_id


def register_pytorch_models() -> None:
    """Đăng ký model từ model_list vào registry (build_base_shard)."""
    from synapse.models import register_model
    for model_id, hf_id in HF_MODELS.items():
        n_layers = HF_MODEL_LAYERS.get(model_id, 28)
        register_model(model_id=model_id, num_layers=n_layers, repo_path=hf_id)


def resolve_hf_model_id(model_id: str) -> str:
    """Tên model -> HF repo_id (dùng trong engine)."""
    return resolve_hf_id(model_id)


class HFShardDownloader:
    """Tải model từ Hugging Face (cache local). Không tải theo shard file; shard = layer range khi chạy."""

    def __init__(self):
        self._on_progress = None  # optional AsyncCallbackSystem

    @property
    def on_progress(self):
        if self._on_progress is None:
            from synapse.helpers import AsyncCallbackSystem
            self._on_progress = AsyncCallbackSystem()
        return self._on_progress

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        hf_id = resolve_hf_model_id(shard.model_id)
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id=hf_id)
            return Path(path)
        except Exception as e:
            if DEBUG >= 1:
                print(f"[HFShardDownloader] ensure_shard {shard.model_id} -> {hf_id}: {e}")
            return Path(".")

    async def get_shard_download_status(self, inference_engine_name: str):
        return
        yield


class PyTorchHFInferenceEngine(InferenceEngine):
    """Engine: load model từ Hugging Face, chạy inference (full hoặc theo layer range cho phân tán)."""

    def __init__(self, shard_downloader: Optional[Any] = None):
        self.shard_downloader = shard_downloader or HFShardDownloader()
        self.shard: Optional[Shard] = None
        self.tokenizer = None
        self._model = None
        self._model_id: Optional[str] = None
        self._device = "cuda" if self._has_cuda() else "cpu"
        self._force_cpu_training = False

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _get_hf_id(self, model_id: str) -> str:
        return resolve_hf_model_id(model_id)

    async def ensure_shard(self, shard: Shard) -> None:
        if self._model_id == shard.model_id and self._model is not None:
            return
        hf_id = self._get_hf_id(shard.model_id)
        if DEBUG >= 1:
            print(f"[PyTorchHF] Loading {shard.model_id} -> {hf_id}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            load_kw = {"trust_remote_code": True}
            load_model_kw = {
                "trust_remote_code": True,
                "torch_dtype": "auto",
                "device_map": "auto" if self._has_cuda() else None,
            }
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_id, **load_kw, local_files_only=True)
                self._model = AutoModelForCausalLM.from_pretrained(hf_id, **load_model_kw, local_files_only=True)
                if DEBUG >= 1:
                    print(f"[PyTorchHF] Loaded from cache (local): {shard.model_id}")
            except (OSError, ValueError) as _e:
                if DEBUG >= 1:
                    print(f"[PyTorchHF] Cache miss or incomplete, loading from Hub: {_e}")
                self.tokenizer = AutoTokenizer.from_pretrained(hf_id, **load_kw)
                self._model = AutoModelForCausalLM.from_pretrained(hf_id, **load_model_kw)
            if (self._model.device.type if hasattr(self._model, "device") else "cpu") == "cpu":
                self._model = self._model.to(self._device)
            self._model_id = shard.model_id
            self.shard = shard
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF load {hf_id}: {e}") from e

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        await self.ensure_shard(shard)
        if self.tokenizer is None:
            return np.array([], dtype=np.int32)
        enc = self.tokenizer.encode(prompt, return_tensors="np")
        return enc.flatten().astype(np.int32)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        await self.ensure_shard(shard)
        if self.tokenizer is None or tokens is None or len(tokens) == 0:
            return ""
        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

    async def sample(self, x: np.ndarray, temp: float = 0.0, generated_ids: Optional[List[int]] = None, repetition_penalty: float = 1.1) -> np.ndarray:
        """Sample token từ logits (last position). Áp dụng repetition_penalty nếu có generated_ids."""
        try:
            import torch
            logits = torch.from_numpy(x).float()
            if logits.dim() == 3:
                logits = logits[0, -1, :].clone()
            else:
                logits = logits[0, :].clone()
            if generated_ids and repetition_penalty != 1.0:
                for tid in set(generated_ids):
                    if 0 <= tid < logits.shape[-1]:
                        if logits[tid].item() > 0:
                            logits[tid] = logits[tid] / repetition_penalty
                        else:
                            logits[tid] = logits[tid] * repetition_penalty
            if temp <= 0:
                token_id = logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits / temp, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
            return np.array([token_id], dtype=np.int32)
        except Exception as e:
            if DEBUG >= 1:
                print(f"[PyTorchHF] sample error: {e}")
            return np.array([0], dtype=np.int32)

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        await self.ensure_shard(shard)
        inference_state = inference_state or {}
        try:
            import torch
            if self._model is None:
                vocab_size = 32000
                return np.zeros((1, 0, vocab_size), dtype=np.float32), inference_state
            device = next(self._model.parameters()).device
            if input_data.dtype != np.int64:
                input_data = input_data.astype(np.int64)
            input_ids = torch.from_numpy(input_data).to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            with torch.no_grad():
                out = self._model(input_ids)
            logits = out.logits if hasattr(out, "logits") else out[0]
            logits_np = logits.float().cpu().numpy()
            return logits_np, inference_state
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF infer_tensor: {e}") from e

    async def load_checkpoint(self, shard: Shard, path: str) -> None:
        pass

    async def save_checkpoint(self, shard: Shard, path: str) -> None:
        if self._model is not None and self.tokenizer is not None:
            try:
                self._model.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[PyTorchHF] save_checkpoint: {e}")

    def _get_optimizer(self, lr: float = 2e-5):
        """Lazy tạo optimizer cho training (1 node hoặc last layer)."""
        if getattr(self, "_optimizer", None) is None and self._model is not None:
            try:
                import torch
                self._optimizer = torch.optim.AdamW(
                    list(self._model.parameters()),
                    lr=lr,
                    weight_decay=0.01,
                )
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[PyTorchHF] _get_optimizer: {e}")
        return getattr(self, "_optimizer", None)

    async def train(
        self,
        request_id: str,
        shard: Shard,
        example: np.ndarray,
        target: np.ndarray,
        length: np.ndarray,
        backgrad: Optional[np.ndarray] = None,
        loss: str = "forward",
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Training chia tải: last layer (hoặc 1 node) chạy full forward + loss + backward + step. Nhiều node (back_gradient) chưa implement."""
        await self.ensure_shard(shard)
        if loss == "back_gradient":
            if DEBUG >= 1:
                print("[PyTorchHF] train(back_gradient) chưa implement cho nhiều node; bỏ qua.")
            return None, None
        try:
            import torch
            import torch.nn.functional as F
            if self._model is None:
                return None, None

            def _run_train_step(device) -> float:
                ex = example.astype(np.int64) if example.dtype != np.int64 else example
                tg = target.astype(np.int64) if target.dtype != np.int64 else target
                input_ids = torch.from_numpy(ex).to(device)
                labels = torch.from_numpy(tg).to(device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)

                self._model.train()
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self._model(input_ids)
                else:
                    out = self._model(input_ids)

                logits = out.logits if hasattr(out, "logits") else out[0]
                # Causal LM: shift; ignore_index=-100
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss_val = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

                opt = self._get_optimizer()
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
                    loss_val.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    opt.step()
                return float(loss_val.item())

            device = next(self._model.parameters()).device
            if self._force_cpu_training and device.type != "cpu":
                if DEBUG >= 1:
                    print("[PyTorchHF] CPU fallback is enabled; moving model to CPU for training.")
                self._model = self._model.to("cpu")
                self._optimizer = None
                self._device = "cpu"
                device = next(self._model.parameters()).device

            try:
                return _run_train_step(device), None
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as oom:
                if device.type != "cuda":
                    raise
                if DEBUG >= 1:
                    print("[PyTorchHF] CUDA OOM during training. Falling back to CPU training.")
                try:
                    import gc

                    self._force_cpu_training = True
                    self._model = self._model.to("cpu")
                    self._optimizer = None
                    self._device = "cpu"
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as switch_err:
                    raise RuntimeError(f"PyTorchHF train: CUDA OOM and CPU fallback failed: {switch_err}") from oom

                cpu_device = next(self._model.parameters()).device
                if cpu_device.type != "cpu":
                    cpu_device = torch.device("cpu")
                return _run_train_step(cpu_device), None
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF train: {e}") from e

    async def evaluate(
        self,
        request_id: str,
        shard: Shard,
        example: np.ndarray,
        target: np.ndarray,
        length: np.ndarray,
    ) -> Optional[float]:
        """Đánh giá: forward, tính loss (không backward)."""
        await self.ensure_shard(shard)
        try:
            import torch
            import torch.nn.functional as F
            if self._model is None:
                return None
            device = next(self._model.parameters()).device
            if example.dtype != np.int64:
                example = example.astype(np.int64)
            if target.dtype != np.int64:
                target = target.astype(np.int64)
            input_ids = torch.from_numpy(example).to(device)
            labels = torch.from_numpy(target).to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            self._model.eval()
            with torch.no_grad():
                out = self._model(input_ids)
                logits = out.logits if hasattr(out, "logits") else out[0]
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss_val = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            return float(loss_val.item())
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            return None
