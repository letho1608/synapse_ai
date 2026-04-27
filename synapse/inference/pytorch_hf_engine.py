"""
PyTorch + Hugging Face inference engine.
Phân tán: load model từ HF. Danh sách model lấy từ synapse.model_list (download + web UI).
"""

import numpy as np
import inspect
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
    """Engine: load model từ Hugging Face, chạy inference (full hoặc theo layer range cho phân tán).
    
    Hỗ trợ tự nhận diện phần cứng và tối ưu hóa tự động:
    - Chọn attn_implementation tốt nhất (sdpa / eager fallback)
    - Chọn torch_dtype phù hợp (bfloat16 / float16 / float32)
    - Bật Quantization (4-bit / 8-bit) nếu VRAM thấp
    """

    def __init__(self, shard_downloader: Optional[Any] = None):
        self.shard_downloader = shard_downloader or HFShardDownloader()
        self.shard: Optional[Shard] = None
        self.tokenizer = None
        self._model = None
        self._model_id: Optional[str] = None
        self._device = "cuda" if self._has_cuda() else "cpu"
        self._kv_caches: Dict[str, Any] = {} # request_id -> list of past_key_values per layer
        # Cache kết quả nhận diện phần cứng để không phải phát hiện lại nhiều lần
        self._hw_profile: Optional[Dict[str, Any]] = None

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _has_directml() -> bool:
        try:
            import torch_directml
            return torch_directml.is_available()
        except Exception:
            return False

    def _detect_hardware_profile(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Nhận diện cấu hình phần cứng của node hiện tại và trả về một profile
        để định hướng cách nạp mô hình một cách tối ưu nhất.

        Returns dict với các key:
            - device (str): 'cuda' hoặc 'cpu' (hoặc 'dml')
            - torch_dtype: dtype tối ưu nhất
            - attn_implementation (str): 'sdpa', hoặc 'eager'
            - load_in_4bit (bool): True nếu VRAM quá nhỏ so với size model
            - load_in_8bit (bool): True nếu VRAM hơi thọt so với size model
            - device_map (str|None): 'auto' nếu tự động phân mảng
            - vram_gb (float): VRAM khả dụng (0 nếu CPU-only)
            - supports_bf16 (bool): True nếu GPU hỗ trợ BFloat16
        """
        import torch

        profile: Dict[str, Any] = {
            "device": "cpu",
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
            "load_in_4bit": False,
            "load_in_8bit": False,
            "device_map": None,
            "vram_gb": 0.0,
            "supports_bf16": False,
        }

        if self._has_cuda():
            profile["device"] = "cuda"
        elif self._has_directml():
            import torch_directml
            profile["device"] = "dml"
            profile["torch_dtype"] = torch.float16
            import psutil
            # Estimate roughly for shared GPU memory
            profile["vram_gb"] = (psutil.virtual_memory().total / (1024 ** 3)) * 0.5 
            if DEBUG >= 1: print("[PyTorchHF][HW] DirectML detected. Using 'dml' device.")
            self._hw_profile = profile
            return profile
        else:
            profile["torch_dtype"] = "auto"
            profile["attn_implementation"] = "eager"
            if DEBUG >= 1: print(f"[PyTorchHF][HW] CPU-only node → Dtype: auto | Attn: eager")
            self._hw_profile = profile
            return profile

        # --- CUDA branch ---
        try:
            dev = torch.cuda.current_device()
            total_vram_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
            vram_gb = total_vram_gb - (torch.cuda.memory_allocated(dev) / (1024 ** 3))
            profile["vram_gb"] = vram_gb

            major, _ = torch.cuda.get_device_capability(dev)
            supports_bf16 = (major >= 8)
            profile["supports_bf16"] = supports_bf16
            profile["torch_dtype"] = torch.bfloat16 if supports_bf16 else torch.float16

            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                profile["attn_implementation"] = "sdpa"

            required_vram = 0.0
            if model_id:
                from synapse.model_list import HF_MODEL_PARAMS, HF_MODELS
                param_str = HF_MODEL_PARAMS.get(model_id)
                if not param_str:
                    for name, repo_id in HF_MODELS.items():
                        if repo_id == model_id:
                            param_str = HF_MODEL_PARAMS.get(name)
                            break
                            
                if param_str:
                    try:
                        p = float(param_str.lower().replace('b', '').replace('m', ''))
                        if 'm' in param_str.lower(): p = p / 1000.0
                        required_vram = p * 2.0 # FP16 ~ 2 bytes per param
                    except Exception:
                        pass
                        
            # Auto-quantization logic replacing simple cut-offs
            if required_vram > 0 and total_vram_gb < (required_vram * 1.2):
                try:
                    import bitsandbytes
                    if total_vram_gb < (required_vram * 0.7):
                        profile["load_in_4bit"] = True
                    else:
                        profile["load_in_8bit"] = True
                except ImportError:
                    pass
            elif required_vram == 0.0 and total_vram_gb <= 12.0:
                try:
                    import bitsandbytes
                    if total_vram_gb <= 6.0:
                        profile["load_in_4bit"] = True
                    else:
                        profile["load_in_8bit"] = True
                except ImportError:
                    pass

            if DEBUG >= 1:
                quant = "4bit" if profile["load_in_4bit"] else ("8bit" if profile["load_in_8bit"] else "none")
                print(f"[PyTorchHF][HW] CUDA: {total_vram_gb:.1f}GB (Req: {required_vram:.1f}GB) | Quant: {quant} | Attn: {profile['attn_implementation']}")

        except Exception as e:
            if DEBUG >= 1: print(f"[PyTorchHF][HW] CUDA Error: {e}")
            profile["torch_dtype"] = torch.float16

        self._hw_profile = profile
        return profile

    def _get_hf_id(self, model_id: str) -> str:
        return resolve_hf_model_id(model_id)

    def _get_transformer_components(self):
        """Return transformer layers and projection modules for supported causal LM layouts."""
        if self._model is None:
            return None

        model = self._model

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return {
                "layers": list(model.model.layers),
                "embed": getattr(model.model, "embed_tokens", None),
                "final_norm": getattr(model.model, "norm", None),
                "lm_head": getattr(model, "lm_head", None),
            }

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return {
                "layers": list(model.transformer.h),
                "embed": getattr(model.transformer, "wte", None),
                "final_norm": getattr(model.transformer, "ln_f", None),
                "lm_head": getattr(model, "lm_head", None),
            }

        if hasattr(model, "layers"):
            return {
                "layers": list(model.layers),
                "embed": getattr(model, "embed_tokens", None),
                "final_norm": getattr(model, "norm", None),
                "lm_head": getattr(model, "lm_head", None),
            }

        return None

    def _get_rotary_embedding_module(self):
        """Best-effort lấy rotary embedding module cho các kiến trúc Qwen/Llama khác version."""
        if self._model is None:
            return None

        # Newer HF layouts thường có model.rotary_emb
        if hasattr(self._model, "model") and hasattr(self._model.model, "rotary_emb"):
            return self._model.model.rotary_emb

        # Fallback: rotary_emb có thể nằm trong self_attn của layer đầu tiên
        components = self._get_transformer_components()
        if components and components.get("layers"):
            first_layer = components["layers"][0]
            self_attn = getattr(first_layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "rotary_emb"):
                return self_attn.rotary_emb

        return None

    def _build_position_embeddings(self, hidden_states, position_ids):
        """Dựng position_embeddings cho rotary embedding module."""
        rotary_emb = self._get_rotary_embedding_module()
        if rotary_emb is None:
            return None

        # Đa số Qwen/Llama dùng chữ ký rotary_emb(x, position_ids).
        out = rotary_emb(hidden_states, position_ids)
        return out

    def _call_transformer_block(self, block, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None, past_key_value=None, cache_position=None):
        # Gọi theo chữ ký thật của layer thay vì thử nhiều fallback.
        params = inspect.signature(block.forward).parameters
        kwargs: Dict[str, Any] = {}

        if "attention_mask" in params and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in params and position_ids is not None:
            kwargs["position_ids"] = position_ids
        if "position_embeddings" in params and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if "past_key_value" in params:
            kwargs["past_key_value"] = past_key_value
        if "cache_position" in params and cache_position is not None:
            kwargs["cache_position"] = cache_position
        if "use_cache" in params:
            kwargs["use_cache"] = True
        if "output_attentions" in params:
            kwargs["output_attentions"] = False

        output = block(hidden_states, **kwargs)

        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def _forward_transformer_range(self, shard: Shard, input_data: np.ndarray, request_id: Optional[str] = None):
        import torch

        components = self._get_transformer_components()
        if not components:
            raise RuntimeError("Unsupported causal LM architecture for distributed shard execution")

        layers = components["layers"]
        if shard.start_layer < 0 or shard.end_layer >= len(layers):
            raise RuntimeError(
                f"Shard range out of bounds for model {shard.model_id}: {shard.start_layer}-{shard.end_layer} / {len(layers)}"
            )

        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        if np.issubdtype(input_data.dtype, np.floating):
            if input_data.ndim == 2:
                input_data = input_data[np.newaxis, :]
            hidden_states = torch.from_numpy(input_data).to(device=device, dtype=dtype)
        else:
            input_ids = torch.from_numpy(input_data.astype(np.int64)).to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            embed = components["embed"]
            if embed is None:
                raise RuntimeError(f"Model {shard.model_id} does not expose input embeddings")
            hidden_states = embed(input_ids)

        # ---------------------------------------------------------------------
        # KV Cache & Position ID Management
        # ---------------------------------------------------------------------
        request_id = request_id or "default"
        if request_id not in self._kv_caches:
            self._kv_caches[request_id] = {}
        
        # Determine sequence length and offset
        # We look at the cache of the first layer in our shard to see how many tokens we've seen
        cache = self._kv_caches[request_id]
        past_len = 0
        if shard.start_layer in cache and cache[shard.start_layer] is not None:
            # past_key_value format: (key, value) or tuple(tuple...)
            # We assume (batch, num_heads, seq_len, head_dim) for most HF models
            k = cache[shard.start_layer][0]
            past_len = k.shape[-2]

        seq_len = hidden_states.shape[1]
        
        # position_ids must start from past_len
        position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Some newer models use cache_position
        cache_position = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)

        # Build position embeddings (RoPE)
        position_embeddings = self._build_position_embeddings(hidden_states, position_ids)

        # Prepare attention mask (optional, eager path might need it if not None but usually None is causal)
        attention_mask = None

        with torch.no_grad():
            for layer_index in range(shard.start_layer, shard.end_layer + 1):
                layer_cache = cache.get(layer_index)
                
                # Forward pass for a single layer block
                # Output might be (hidden_states, next_kv_cache, ...)
                # But since use_cache=True, we expect a tuple back.
                block = layers[layer_index]
                
                # Our helper _call_transformer_block will call the block and return JUST hidden_states
                # We need to manually capture the KV cache if it's returned.
                # Let's override the block call here to capture KV cache.
                
                params = inspect.signature(block.forward).parameters
                kwargs: Dict[str, Any] = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "position_embeddings": position_embeddings,
                    "past_key_value": layer_cache,
                    "use_cache": True,
                }
                if "cache_position" in params:
                    kwargs["cache_position"] = cache_position
                
                # Actual block call
                output = block(hidden_states, **kwargs)
                
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    if len(output) > 1:
                        cache[layer_index] = output[1]
                elif hasattr(output, "last_hidden_state"):
                    hidden_states = output.last_hidden_state
                    if hasattr(output, "past_key_value"):
                        cache[layer_index] = output.past_key_value
                else:
                    hidden_states = output

            if not shard.is_last_layer():
                return hidden_states.float().cpu().numpy()

            final_norm = components["final_norm"]
            if final_norm is not None:
                hidden_states = final_norm(hidden_states)

            lm_head = components["lm_head"]
            if lm_head is None:
                raise RuntimeError(f"Model {shard.model_id} does not expose an output head")

            logits = lm_head(hidden_states)
            return logits.float().cpu().numpy()

    def clear_cache(self, request_id: Optional[str] = None):
        if request_id:
            self._kv_caches.pop(request_id, None)
            if DEBUG >= 2: print(f"[PyTorchHF] Cleared KV cache for request: {request_id}")
        else:
            self._kv_caches.clear()
            if DEBUG >= 1: print("[PyTorchHF] Cleared all KV caches")

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

        # --- Nhận diện phần cứng ---
        hw = self._detect_hardware_profile(shard.model_id)
        load_model_kw: Dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": hw["attn_implementation"],
            "torch_dtype": hw["torch_dtype"],
        }

        # Ưu tiên ÉP model vào GPU duy nhất nếu có thể để tránh device_map đẩy sang CPU
        if hw["device"] == "cuda":
            if hw["load_in_4bit"] or hw["load_in_8bit"]:
                # Nếu dùng quantization, phải dùng device_map
                load_model_kw["device_map"] = "auto"
                from transformers import BitsAndBytesConfig
                if hw["load_in_4bit"]:
                    load_model_kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=hw["torch_dtype"],
                        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
                    )
                else:
                    load_model_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                # KHÔNG dùng device_map='auto' cho 1 máy để tránh nó đẩy layer sang CPU
                pass

        if DEBUG >= 1:
            print(f"[PyTorchHF] Nạp model {shard.model_id} lên {hw['device']}...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, local_files_only=True)
                self._model = AutoModelForCausalLM.from_pretrained(hf_id, **load_model_kw, local_files_only=True)
            except (OSError, ValueError):
                # --- [P2P LAN Cache Sync] ---
                try:
                    import aiohttp
                    node_ref = getattr(self, "node", None)
                    if node_ref:
                        nodes = await node_ref.get_tailscale_nodes()
                        from synapse.networking.tailscale.tailscale_helpers import get_synapse_api_urls_from_node_list
                        from synapse.helpers import get_all_ip_addresses_and_interfaces
                        self_ips = set(ip for ip, _ in get_all_ip_addresses_and_interfaces() if ip)
                        urls = get_synapse_api_urls_from_node_list(nodes, api_port=52415, only_synapse_nodes=True, exclude_ips=self_ips)
                        if urls:
                            master_url = urls[0] # Node cao điểm nhất (Election Master)
                            if DEBUG >= 1: print(f"[PyTorchHF] Requesting P2P LAN Cache sync from: {master_url}...")
                            # Trong thực tế: Dùng aiohttp_client để scan `repo_files` và tải đè local_dir.
                            # Tạm thời dùng fallback gốc vì thay đổi HF_HOME/blob hash rất dễ gây lỗi nạp (Corrupt file).
                except Exception as e:
                    if DEBUG >= 2: print(f"[P2P Sync Exception]: {e}")
                
                # Fallback to direct HF download
                self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(hf_id, **load_model_kw)

            # ÉP MODEL LÊN GPU (Nếu không dùng quantization/device_map)
            if hw["device"] == "cuda" and "device_map" not in load_model_kw:
                self._model = self._model.to("cuda")

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
        enc = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="np")
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
            if self._model is None:
                vocab_size = 32000
                return np.zeros((1, 0, vocab_size), dtype=np.float32), inference_state
            output = self._forward_transformer_range(shard, input_data, request_id=request_id)
            return output, inference_state
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
