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
from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS, resolve_hf_id, VISION_MODELS


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
        self._is_lora_prepared = False

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

        is_vision = shard.model_id in VISION_MODELS
        if is_vision:
            try:
                from huggingface_hub import snapshot_download
                import sys
                import torch
                import json
                repo_path = snapshot_download(repo_id=hf_id)
                if repo_path not in sys.path:
                    sys.path.insert(0, repo_path)
                import model as crnn_model
                
                with open(f"{repo_path}/vocab.json", "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                
                if "idx_to_char" in vocab:
                    idx_dict = vocab["idx_to_char"]
                else:
                    idx_dict = vocab
                    
                num_classes = max([int(k) for k in idx_dict.keys()]) + 1
                
                self._model = crnn_model.CRNN(vocab_size=num_classes, hidden_size=256)
                weights_path = f"{repo_path}/vietnamese_crnn_model.pth"
                checkpoint = torch.load(weights_path, map_location=self._device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self._model.load_state_dict(checkpoint['model_state_dict'])
                    if 'img_width' in checkpoint:
                        self.img_width = checkpoint['img_width']
                else:
                    self._model.load_state_dict(checkpoint)
                self._model = self._model.to(self._device)
                self._model.eval()
                
                self.tokenizer = vocab
                self._model_id = shard.model_id
                self.shard = shard
                if DEBUG >= 1: print(f"[PyTorchHF] Loaded Vision local CRNN: {shard.model_id}")
                return
            except Exception as e:
                raise RuntimeError(f"PyTorchHF load vision {hf_id}: {e}") from e

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
            if self._model.device.type if hasattr(self._model, "device") else "cpu" == "cpu":
                self._model = self._model.to(self._device)
            self._model_id = shard.model_id
            self.shard = shard
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF load {hf_id}: {e}") from e

    async def infer_vision(self, shard: Shard, image_bytes: bytes) -> str:
        await self.ensure_shard(shard)
        try:
            import torch
            import cv2
            import numpy as np
            import torchvision.transforms as transforms
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if np.mean(img_array) < 127:
                img_array = 255 - img_array
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            h, w = img_array.shape
            new_h = 64
            img_width = getattr(self, 'img_width', 448)
            new_w = int(w * (new_h / h))
            if new_w > img_width:
                new_w = img_width
                
            img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            padded = np.full((new_h, img_width), 255, dtype=np.uint8)
            padded[:, :new_w] = img_resized
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            image_tensor = transform(padded.copy()).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                logits = self._model(image_tensor)
                
            vocab = self.tokenizer
            if "idx_to_char" in vocab:
                idx_dict = vocab["idx_to_char"]
            else:
                idx_dict = vocab
            idx_to_char = {int(k): v for k, v in idx_dict.items()}
            
            beam_width = 10
            log_probs = logits.squeeze(1).cpu().numpy()
            seq_len, num_classes = log_probs.shape
            beams = [(0.0, [], -1)]
            
            for t in range(seq_len):
                new_beams = []
                for score, seq, last_idx in beams:
                    for c in range(num_classes):
                        new_score = score + log_probs[t, c]
                        if c == 0:
                            new_beams.append((new_score, seq[:], -1))
                        elif c == last_idx:
                            new_beams.append((new_score, seq[:], last_idx))
                        else:
                            new_beams.append((new_score, seq + [c], c))
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_width]
                
            best_seq = beams[0][1]
            text = "".join([idx_to_char.get(i, '') for i in best_seq])
            return text
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise Exception(f"Vision inference error: {e}")

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
                # Nhường nhịp cho event loop trước khi vào compute nặng
                await asyncio.sleep(0.01)
                out = self._model(input_ids)
                # Nhường nhịp sau compute
                await asyncio.sleep(0.01)
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
        """Lazy tạo optimizer cho training. CHỈ tối ưu các tham số requires_grad=True (quan trọng cho LoRA)."""
        if getattr(self, "_optimizer", None) is None and self._model is not None:
            try:
                import torch
                # CHỈ lấy các tham số trainable (cho LoRA/PEFT)
                trainable_params = [p for p in self._model.parameters() if p.requires_grad]
                
                if not trainable_params:
                    if DEBUG >= 1: print("[PyTorchHF] [WARNING] No trainable parameters found! Is LoRA enabled?")
                    trainable_params = list(self._model.parameters())
                    
                self._optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=lr,
                    weight_decay=0.01,
                )
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[PyTorchHF] _get_optimizer: {e}")
        return getattr(self, "_optimizer", None)

    def _prepare_model_for_training(self):
        """Ốp LoRA (PEFT) và Gradient Checkpointing để tiết kiệm VRAM."""
        if self._is_lora_prepared or self._model is None:
            return
        
        try:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            if DEBUG >= 1: print("[PyTorchHF] Preparing model for training with LoRA...")
            
            # 1. Gradient Checkpointing: Tiết kiệm VRAM cực lớn (đánh đổi bằng tốc độ)
            try:
                self._model.gradient_checkpointing_enable()
            except Exception:
                pass
            
            # 2. Cấu hình LoRA (Low-Rank Adaptation)
            # target_modules thường là q_proj, v_proj cho Llama/Qwen
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self._model = get_peft_model(self._model, config)
            self._model.print_trainable_parameters()
            
            self._is_lora_prepared = True
            # Reset optimizer để nó bắt được các tham số LoRA mới
            self._optimizer = None
            
        except ImportError:
            if DEBUG >= 1: print("[PyTorchHF] [WARNING] peft not installed. Falling back to FULL fine-tuning (HIGH VRAM USAGE).")
        except Exception as e:
            if DEBUG >= 1:
                print(f"[PyTorchHF] LoRA preparation error: {e}")
                import traceback
                traceback.print_exc()

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
        """
        Huấn luyện phân tán: 
        Hỗ trợ cả Causal LLM (CrossEntropy) và Vision/OCR (CTCLoss).
        """
        await self.ensure_shard(shard)
        if self._model is not None and not self._is_lora_prepared and len(target.shape) > 0 and len(example.shape) < 4:
            # Chỉ ốp LoRA cho LLM (vison/ocr có kiến trúc riêng thường đã nhỏ)
            self._prepare_model_for_training()

        if loss == "back_gradient":
            if DEBUG >= 1:
                print("[PyTorchHF] train(back_gradient) chưa implement cho nhiều node; bỏ qua.")
            return None, None
            
        try:
            import torch
            import torch.nn.functional as F
            from torch import nn
            
            if self._model is None:
                return None, None
                
            device = self._device
            self._model.train()
            
            # 1. Kiểm tra nếu là dữ liệu Vision (OCR)
            # example shape: [B, 1, H, W]
            is_vision = len(target.shape) > 0 and len(example.shape) == 4
            
            if is_vision:
                # --- OCR TRAINING (CTCLoss) ---
                imgs = torch.from_numpy(example).to(device)
                labels = torch.from_numpy(target).to(device)
                label_lens = torch.from_numpy(length).to(device)
                
                # Forward
                preds = self._model(imgs) # [T, B, C]
                T = preds.size(0)
                batch_size = imgs.size(0)
                input_lens = torch.full(size=(batch_size,), fill_value=T, dtype=torch.int32).to(device)
                
                # Log Softmax
                preds = preds.log_softmax(2)
                
                # CTCLoss (blank=0 mặc định trong ocr_trainer)
                criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
                loss_val = criterion(preds, labels, input_lens, label_lens)
            else:
                # --- LLM TRAINING (CrossEntropy) ---
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
                    
                # Nhường nhịp trước compute nặng
                await asyncio.sleep(0.01)
                out = self._model(input_ids)
                await asyncio.sleep(0.01)
                logits = out.logits if hasattr(out, "logits") else out[0]
                
                # Causal LM shift
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss_val = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            # 2. Backward & Step
            opt = self._get_optimizer()
            if opt is not None:
                opt.zero_grad()
                # Nhường nhịp trước backward
                await asyncio.sleep(0.01)
                loss_val.backward()
                await asyncio.sleep(0.01)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                
            return float(loss_val.item()), None
            
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF train error: {e}") from e

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
