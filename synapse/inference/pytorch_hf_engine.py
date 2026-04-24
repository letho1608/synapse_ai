"""
PyTorch + Hugging Face inference engine.
Phân tán: load model từ HF. Danh sách model lấy từ synapse.model_list (download + web UI).
"""

import torch
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
            
            # Check if this shard covers a specific layer range (distributed inference)
            is_first = shard.is_first_layer()
            is_last = shard.is_last_layer()
            start_layer = shard.start_layer
            end_layer = shard.end_layer
            n_layers = shard.n_layers
            
            # For distributed inference with layer range, we need to handle hidden states
            # If not first layer, we need to use cached hidden states from previous shard
            if not is_first and inference_state.get("hidden_states") is not None:
                hidden_states = inference_state["hidden_states"]
                if isinstance(hidden_states, np.ndarray):
                    hidden_states = torch.from_numpy(hidden_states).to(device)
                # Use hidden states as input instead of token IDs
                input_hidden = hidden_states
                use_hidden_states_input = True
            else:
                input_hidden = None
                use_hidden_states_input = False
            
            # Get attention mask and position_ids from inference_state
            attention_mask = None
            position_ids = None
            if inference_state.get("attention_mask") is not None:
                attn_mask = inference_state["attention_mask"]
                if isinstance(attn_mask, np.ndarray):
                    attention_mask = torch.from_numpy(attn_mask).to(device)
            if inference_state.get("position_ids") is not None:
                pos_ids = inference_state["position_ids"]
                if isinstance(pos_ids, np.ndarray):
                    position_ids = torch.from_numpy(pos_ids).to(device)
            
            # Get KV cache from inference_state (for autoregressive generation)
            # KV cache format: list of (past_key, past_value) tuples per layer
            past_key_values = None
            if inference_state.get("past_key_values") is not None:
                pkv = inference_state["past_key_values"]
                if isinstance(pkv, list):
                    past_key_values = []
                    for item in pkv:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            k, v = item
                            if isinstance(k, np.ndarray):
                                k = torch.from_numpy(k).to(device)
                            if isinstance(v, np.ndarray):
                                v = torch.from_numpy(v).to(device)
                            past_key_values.append((k, v))
                        else:
                            past_key_values = None
                            break
            
            with torch.no_grad():
                # Check if this is a distributed shard (not full model)
                is_distributed_shard = not (is_first and is_last)
        
                # Enable KV cache capture for autoregressive generation
                # This is needed when past_key_values is provided or when we need to build KV cache
                capture_kv = past_key_values is not None or is_last
                self._capture_kv_cache = capture_kv
        
                if is_distributed_shard:
                    # Distributed inference: run only our layer range
                    hidden = await self._run_layer_range(
                        input_ids, input_hidden, start_layer, end_layer, n_layers,
                        is_first, is_last, attention_mask, position_ids, past_key_values
                    )
        
                    # Store hidden states for next shard
                    inference_state["hidden_states"] = hidden.float().cpu().numpy()
        
                    # If last layer, compute logits
                    if is_last:
                        logits = self._compute_logits(hidden)
                    else:
                        # Return zeros with correct shape for intermediate layers
                        vocab_size = self._get_vocab_size()
                        seq_len = hidden.shape[1]
                        logits = torch.zeros(1, seq_len, vocab_size).to(device)
                else:
                    # Full model inference (single node)
                    out = self._model(input_ids)
                    logits = out.logits if hasattr(out, "logits") else out[0]
        
                # Store attention mask and position_ids for distributed inference
                if is_first and not is_last:
                    # Create attention mask (all 1s for causal model)
                    seq_len = input_ids.shape[1]
                    inference_state["attention_mask"] = torch.ones(1, seq_len, device=device).cpu().numpy()
                    # Create position_ids [0, 1, 2, ..., seq_len-1]
                    inference_state["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0).cpu().numpy()
        
                # Store captured KV cache for next autoregressive step
                if hasattr(self, '_captured_kv_cache') and self._captured_kv_cache:
                    inference_state["past_key_values"] = [
                        self._captured_kv_cache[i] if i in self._captured_kv_cache else None
                        for i in range(len(self._captured_kv_cache))
                    ]
                    if DEBUG >= 2:
                        print(f"[infer_tensor] Stored KV cache for {len(self._captured_kv_cache)} layers")
        
                logits_np = logits.float().cpu().numpy()
                return logits_np, inference_state
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF infer_tensor: {e}") from e
    
    async def _run_layer_range(
        self,
        input_ids: torch.Tensor,
        input_hidden: Optional[torch.Tensor],
        start_layer: int,
        end_layer: int,
        n_layers: int,
        is_first: bool,
        is_last: bool,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None
    ) -> torch.Tensor:
        """Run model for a specific layer range. Returns hidden states at the end of the range.
        
        For KV cache extraction (autoregressive generation), set capture_kv_cache=True.
        The captured KV cache will be stored in self._captured_kv_cache.
        """
        import torch
    
        # Get the base model (e.g., Qwen2Model, LlamaModel)
        base_model = self._model.base_model if hasattr(self._model, 'base_model') else self._model
    
        # Get embeddings if first layer
        if is_first:
            # Get embeddings
            if hasattr(base_model, 'embed_tokens'):
                hidden = base_model.embed_tokens(input_ids)
            elif hasattr(base_model, 'get_input_embeddings'):
                hidden = base_model.get_input_embeddings()(input_ids)
            elif hasattr(self._model, 'get_input_embeddings'):
                hidden = self._model.get_input_embeddings()(input_ids)
            else:
                # Fallback: run first layer through model
                out = self._model(input_ids, output_hidden_states=True, return_dict=True)
                all_hidden = out.hidden_states
                return all_hidden[min(end_layer + 1, len(all_hidden) - 1)]
        else:
            hidden = input_hidden
    
        # Get the transformer layers
        # Different model architectures have different attribute names
        layers = None
        if hasattr(base_model, 'layers'):
            layers = base_model.layers
        elif hasattr(base_model, 'h'):
            layers = base_model.h
        elif hasattr(base_model, 'transformer_blocks'):
            layers = base_model.transformer_blocks
        elif hasattr(base_model, 'blocks'):
            layers = base_model.blocks
    
        if layers is None:
            # Fallback: run full model
            if is_first:
                out = self._model(input_ids, output_hidden_states=True, return_dict=True)
                return out.hidden_states[min(end_layer + 1, len(out.hidden_states) - 1)]
            else:
                # Can't do much without layers reference
                raise RuntimeError("Cannot find transformer layers in model")
    
        # Determine if we need to capture KV cache (for autoregressive generation)
        capture_kv = past_key_values is not None or getattr(self, '_capture_kv_cache', False)
        
        # KV cache capture hooks storage
        captured_kv = {} if capture_kv else None
        hooks = []
    
        def create_kv_hook(layer_idx):
            """Create a forward hook to capture key/value tensors from attention."""
            def hook_fn(module, input, output):
                # Attention output format: (hidden_states, None, past_key_value) or similar
                # For standard attention: output[1] or output[2] contains past_kv
                if isinstance(output, tuple) and len(output) >= 3:
                    # Most HF models return: (hidden_states, attention_weights, past_key_value)
                    # past_key_value is a tuple of (key, value) tensors
                    past_kv = output[2] if len(output) > 2 else None
                    if past_kv is not None and isinstance(past_kv, tuple) and len(past_kv) == 2:
                        captured_kv[layer_idx] = (
                            past_kv[0].detach().cpu().numpy(),
                            past_kv[1].detach().cpu().numpy()
                        )
            return hook_fn
    
        # Register hooks on attention layers if capturing KV
        if capture_kv:
            for layer_idx in range(start_layer, min(end_layer + 1, len(layers))):
                layer = layers[layer_idx]
                # Find attention module within the layer
                # Different architectures: LlamaAttention, Qwen2Attention, etc.
                attention = None
                for name, module in layer.named_modules():
                    module_name = type(module).__name__
                    if 'Attention' in module_name or 'Attn' in module_name:
                        attention = module
                        break
                
                if attention is not None:
                    hook = attention.register_forward_hook(create_kv_hook(layer_idx))
                    hooks.append(hook)
                    if DEBUG >= 2:
                        print(f"[_run_layer_range] Registered KV hook on layer {layer_idx}, attention type: {type(attention).__name__}")
    
        # Run through our layer range
        # For models like Qwen2, Llama: layers[i] is a LlamaDecoderLayer or similar
        for layer_idx in range(start_layer, end_layer + 1):
            if layer_idx >= len(layers):
                break
    
            layer = layers[layer_idx]
    
            # Each layer takes hidden states and attention mask (if needed)
            # Most modern HF models use: layer(hidden_states, attention_mask=...)
            if hasattr(layer, '_use_cache'):
                # Some models need this
                layer._use_cache = capture_kv
    
            # Prepare layer inputs - pass position_ids for RoPE support
            layer_kwargs = {}
            if attention_mask is not None:
                layer_kwargs['attention_mask'] = attention_mask
            if position_ids is not None:
                layer_kwargs['position_ids'] = position_ids
            
            # For autoregressive generation with KV cache, pass past_key_values
            if capture_kv and past_key_values is not None and layer_idx < len(past_key_values):
                layer_kwargs['past_key_value'] = past_key_values[layer_idx]
    
            try:
                # Try standard interface with all kwargs
                layer_output = layer(hidden, use_cache=capture_kv, **layer_kwargs)
                if isinstance(layer_output, tuple):
                    hidden = layer_output[0]
                else:
                    hidden = layer_output
            except TypeError:
                # Try without attention_mask
                try:
                    layer_kwargs.pop('attention_mask', None)
                    layer_output = layer(hidden, use_cache=capture_kv, **layer_kwargs)
                    if isinstance(layer_output, tuple):
                        hidden = layer_output[0]
                    else:
                        hidden = layer_output
                except Exception as e:
                    # Fallback: run full model with our hidden states as input
                    if DEBUG >= 2:
                        print(f"[_run_layer_range] Layer forward failed, falling back: {e}")
                    # Use hidden states as inputs_embeds to the model
                    if hasattr(self._model.base_model, 'inputs_embeds'):
                        out = self._model.base_model(inputs_embeds=hidden)
                    else:
                        # Last resort: run full model from beginning
                        out = self._model(input_ids, output_hidden_states=True, return_dict=True)
                        if hasattr(out, 'hidden_states') and out.hidden_states:
                            return out.hidden_states[min(end_layer + 1, len(out.hidden_states) - 1)]
                        elif hasattr(out, 'logits'):
                            # If no hidden_states, return logits and let caller handle
                            return hidden # Return our computed hidden, not logits
                        else:
                            return hidden
    
        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()
    
        # Store captured KV cache for caller to retrieve
        if capture_kv and captured_kv:
            self._captured_kv_cache = captured_kv
            if DEBUG >= 2:
                print(f"[_run_layer_range] Captured KV for layers: {list(captured_kv.keys())}")
    
        # Apply final layer norm if last layer (some models need this)
        if is_last:
            if hasattr(base_model, 'final_layer_norm'):
                hidden = base_model.final_layer_norm(hidden)
            elif hasattr(base_model, 'layer_norm'):
                hidden = base_model.layer_norm(hidden)
            elif hasattr(self._model, 'model') and hasattr(self._model.model, 'layer_norm'):
                hidden = self._model.model.layer_norm(hidden)
    
        return hidden
    
    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states."""
        # Get lm_head
        if hasattr(self._model, 'lm_head'):
            logits = self._model.lm_head(hidden_states)
        elif hasattr(self._model.base_model, 'lm_head'):
            logits = self._model.base_model.lm_head(hidden_states)
        else:
            # Fallback: use model output
            out = self._model.base_model(inputs_embeds=hidden_states)
            logits = out.logits if hasattr(out, 'logits') else out[0]
        return logits
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size from model."""
        if hasattr(self._model, 'config') and hasattr(self._model.config, 'vocab_size'):
            return self._model.config.vocab_size
        elif hasattr(self._model, 'lm_head') and hasattr(self._model.lm_head, 'out_features'):
            return int(self._model.lm_head.out_features)
        elif hasattr(self._model.base_model, 'lm_head') and hasattr(self._model.base_model.lm_head, 'out_features'):
            return int(self._model.base_model.lm_head.out_features)
        return 32000 # Default
    
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
        """
        Huấn luyện phân tán: 
        Hỗ trợ cả Causal LLM (CrossEntropy) và Vision/OCR (CTCLoss).
        """
        await self.ensure_shard(shard)
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
                    
                out = self._model(input_ids)
                logits = out.logits if hasattr(out, "logits") else out[0]
                
                # Causal LM shift
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss_val = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            # 2. Backward & Step
            opt = self._get_optimizer()
            if opt is not None:
                opt.zero_grad()
                loss_val.backward()
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
