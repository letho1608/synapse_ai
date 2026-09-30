"""
PyTorch + Hugging Face inference engine.
Phân tán: load model từ HF. Danh sách model lấy từ synapse.model_list (download + web UI).
"""

import asyncio
import numpy as np
import inspect
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from synapse.helpers import DEBUG
from synapse.inference.inference_engine import (
    InferenceEngine, SHARD_WEIGHTS_FORMAT, SHARD_WEIGHTS_SUFFIX,
)
from synapse.inference.shard import Shard
from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS, resolve_hf_id


class KVMissError(RuntimeError):
    """A stage was asked to append rows after a prefix it does not hold
    (e.g. it restarted or replaced a dead node). The orchestrator answers by
    resending the step with full hidden states (delta_rows off)."""


def register_pytorch_models() -> None:
    """Đăng ký model từ model_list vào registry (build_base_shard)."""
    from synapse.models import register_model
    for model_id, hf_id in HF_MODELS.items():
        # 0 = unknown: build_base_shard reads num_hidden_layers from the
        # model's config.json. A guessed default (was 28) silently skipped
        # layers on any deeper model (SmolLM2-135M has 30).
        n_layers = HF_MODEL_LAYERS.get(model_id) or HF_MODEL_LAYERS.get(hf_id) or 0
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


class PyTorchHFInferenceEngine(InferenceEngine):
    """Engine: load model từ Hugging Face, chạy inference (full hoặc theo layer range cho phân tán).
    
    Hỗ trợ tự nhận diện phần cứng và tối ưu hóa tự động:
    - Chọn attn_implementation tốt nhất (sdpa / eager fallback)
    - Chọn torch_dtype phù hợp (bfloat16 / float16 / float32)
    - Bật Quantization (4-bit / 8-bit) nếu VRAM thấp
    """

    @staticmethod
    def _choose_quant(total_vram_gb: float, required_vram: float):
        """Quant rung selection (T3.4-rung2): '4bit' / '8bit' / None."""
        if required_vram > 0 and total_vram_gb < (required_vram * 1.2):
            return "4bit" if total_vram_gb < (required_vram * 0.7) else "8bit"
        if required_vram == 0.0 and total_vram_gb <= 12.0:
            return "4bit" if total_vram_gb <= 6.0 else "8bit"
        return None

    def __init__(self, shard_downloader: Optional[Any] = None):
        self.shard_downloader = shard_downloader or HFShardDownloader()
        self.shard: Optional[Shard] = None
        self.tokenizer = None
        self._model = None
        self._model_id: Optional[str] = None
        self._device = "cuda" if self._has_cuda() else "cpu"
        # Cache kết quả nhận diện phần cứng để không phải phát hiện lại nhiều lần
        self._hw_profile: Optional[Dict[str, Any]] = None
        # Per-request incremental KV state (T3.1, SYNAPSE_SEQ_KV=1 only):
        # request_id -> {"ids": 1-D int64 prefix, "length": int, "cache": HF Cache}.
        # Reused ONLY when the new input extends the stored prefix exactly,
        # so decode matches full-refeed math by construction (parity-gated).
        self._seq_kv: Dict[str, Dict[str, Any]] = {}
        import threading
        self._forward_lock = threading.Lock()
        # Layer-range loading (default): only the served layers are in memory.
        # Training/evaluation switch to the full model (_ensure_full).
        self._pm = None
        self._tok_model_id: Optional[str] = None
        # Continuous batching of decode steps across requests (see _submit).
        self._pending: List[Any] = []
        self._drain_task = None
        self._active: Dict[str, float] = {}
        self._fwd_ewma_s = 0.0
        self._load_lock = None
        self.layer_ms: Dict[str, float] = {}  # model -> decode ms per layer (EWMA)
        self.batch_stats = {"forwards": 0, "batched_forwards": 0,
                            "batched_jobs": 0, "max_batch": 0}

    @staticmethod
    def _seq_kv_enabled() -> bool:
        """Per-request incremental KV fast path (T3.1).

        Default ON (parity proven token-identical on tiny-GPT2 CPU,
        fallback on any doubt). SYNAPSE_SEQ_KV=0 restores legacy scan.
        """
        return os.getenv("SYNAPSE_SEQ_KV", "1") == "1"

    @staticmethod
    def _seq_rows(input_data: np.ndarray) -> np.ndarray:
        """Input as one row per sequence position: token ids -> (seq,),
        hidden states [1, seq, hidden] / [seq, hidden] -> (seq, hidden)."""
        arr = np.ascontiguousarray(input_data)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.reshape(-1, arr.shape[-1])
        # Token ids compare by value: callers mix int32 (encode) and int64.
        return arr.reshape(-1).astype(np.int64, copy=False)

    def _seq_kv_try_hit(self, input_data: np.ndarray, request_id: Optional[str],
                        min_new: int = 1):
        """Return (reusable_len, state) from per-request state, or (0, None).

        reusable_len is the longest common prefix of this input and the
        previous call's input (rows compared byte-exact), capped so at least
        one row is recomputed. An exact extension reuses everything; after
        a rejected speculative guess the cache is cropped back to the
        divergence point instead of recomputing the whole context. Works for
        token ids (first stage) and hidden states (later stages): upstream
        stages return prefix rows byte-identical to the previous call, and
        per-row INT8 wire quantization is deterministic per row.
        """
        if not request_id:
            return 0, None
        try:
            rows = self._seq_rows(input_data)
            prev = self._seq_kv.get(request_id)
            if not isinstance(prev, dict) or prev.get("cache") is None:
                return 0, None
            prev_rows = prev.get("ids")
            if not isinstance(prev_rows, np.ndarray) or prev_rows.dtype != rows.dtype:
                return 0, None
            plen = int(prev.get("length", 0))
            if plen <= 0 or rows.shape[1:] != prev_rows.shape[1:]:
                return 0, None
            n = min(plen, rows.shape[0])
            if rows.ndim == 1:
                diff = np.nonzero(rows[:n] != prev_rows[:n])[0]
            else:
                diff = np.nonzero(np.any(rows[:n] != prev_rows[:n], axis=1))[0]
            common = int(diff[0]) if diff.size else n
            # The caller needs outputs for the last `min_new` positions (the
            # draft being verified), so those rows are always recomputed.
            common = min(common, rows.shape[0] - max(1, int(min_new)))
            if common <= 0:
                return 0, None
            return common, prev
        except Exception:
            return 0, None

    @staticmethod
    def _causal_mask(seq_len: int, past_len: int, dtype, device):
        """Additive 4-D mask [1, 1, seq_len, past_len + seq_len]: query i
        (absolute position past_len + i) sees keys 0..past_len + i only."""
        import torch
        total = past_len + seq_len
        q = torch.arange(past_len, total, device=device).unsqueeze(1)
        k = torch.arange(total, device=device).unsqueeze(0)
        mask = torch.zeros(seq_len, total, dtype=dtype, device=device)
        mask.masked_fill_(k > q, torch.finfo(dtype).min)
        return mask[None, None]

    def _seq_kv_commit(self, input_data: np.ndarray, request_id: Optional[str],
                       cache, out_hidden=None) -> None:
        """Record this call's input rows + cache (+ full output hidden states
        for non-last stages) for the next incremental step.

        Always stored (even with SYNAPSE_SEQ_KV=0): delta-convention callers
        send only new tokens and depend on it; the flag only gates reuse for
        full-context callers."""
        if not request_id:
            return
        try:
            rows = self._seq_rows(input_data)
            self._seq_kv[request_id] = {"ids": rows.copy(), "length": rows.shape[0],
                                        "cache": cache, "out": out_hidden}
        except Exception:
            pass

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
            profile["device"] = "dml"
            profile["torch_dtype"] = torch.float16
            import psutil
            # Estimate roughly for shared GPU memory
            profile["vram_gb"] = (psutil.virtual_memory().total / (1024 ** 3)) * 0.5 
            if DEBUG >= 1: print("[PyTorchHF][HW] DirectML detected. Using 'dml' device.")
            self._hw_profile = profile
            return profile
        else:
            # fp32 on CPU: "auto" picked bf16 from the config, which most CPUs
            # run slowly and which drifts from reference numerics.
            profile["torch_dtype"] = torch.float32
            profile["attn_implementation"] = "eager"
            if DEBUG >= 1: print("[PyTorchHF][HW] CPU-only node → Dtype: float32 | Attn: eager")
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

            # Best attention kernel that really exists here (probe: fa2 needs
            # the flash_attn package + CUDA; triton has no HF equivalent).
            from synapse.inference.attn_backend import probe_attn_backend
            best = probe_attn_backend(prefer=("fa2", "sdpa", "eager"))["name"]
            profile["attn_implementation"] = {
                "fa2": "flash_attention_2", "sdpa": "sdpa"}.get(best, "eager")

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
                        
            # Auto-quantization rung selection is a pure function (tested).
            quant_needed = PyTorchHFInferenceEngine._choose_quant(
                total_vram_gb, required_vram)
            if quant_needed:
                try:
                    import bitsandbytes  # noqa: F401
                    profile["load_in_4bit"] = quant_needed == "4bit"
                    profile["load_in_8bit"] = quant_needed == "8bit"
                except ImportError:
                    print(f"[PyTorchHF][HW] WARNING: {quant_needed} wanted "
                          f"(VRAM {total_vram_gb:.1f}GB < req {required_vram:.1f}GB) "
                          f"but bitsandbytes is not installed -> loading UNQUANTIZED")
            profile["quant_wanted"] = quant_needed
            profile["quant_active"] = ("4bit" if profile["load_in_4bit"]
                                       else ("8bit" if profile["load_in_8bit"] else "none"))

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
        if self._pm is not None:
            from synapse.inference.partial_loader import LayerView
            pm = self._pm
            return {"layers": LayerView(pm), "embed": pm.embed, "pos_embed": pm.pos_embed,
                    "final_norm": pm.final_norm, "lm_head": pm.lm_head}
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
                # GPT-2 style learned absolute positions, added to the token
                # embeddings on the first stage (no rotary embedding).
                "pos_embed": getattr(model.transformer, "wpe", None),
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
        if self._pm is not None:
            return self._pm.rotary
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

    def _prepare_forward(self, shard: Shard, input_data: np.ndarray,
                         request_id: Optional[str] = None,
                         inference_state: Optional[dict] = None):
        """Classify the call against the request's KV cache and build the
        new rows' hidden states (see _forward_transformer_range)."""
        import torch

        components = self._get_transformer_components()
        if not components:
            raise RuntimeError("Unsupported causal LM architecture for distributed shard execution")

        layers = components["layers"]
        if shard.start_layer < 0 or shard.end_layer >= len(layers):
            raise RuntimeError(
                f"Shard range out of bounds for model {shard.model_id}: {shard.start_layer}-{shard.end_layer} / {len(layers)}"
            )
        if shard.n_layers != len(layers):
            # A wrong n_layers makes the "last" shard stop early and apply
            # the LM head to a mid-model hidden state: fluent garbage.
            raise RuntimeError(
                f"Shard n_layers={shard.n_layers} but {shard.model_id} has {len(layers)} "
                f"transformer layers; refusing to run a partial model")

        device, dtype = self._device_dtype()

        is_hidden_states = np.issubdtype(input_data.dtype, np.floating)
        if is_hidden_states:
            if input_data.ndim == 2:
                input_data = input_data[np.newaxis, :]
            hidden_states_full = torch.from_numpy(input_data).to(device=device, dtype=dtype)
        else:
            input_ids_full = torch.from_numpy(input_data.astype(np.int64)).to(device)
            if input_ids_full.dim() == 1:
                input_ids_full = input_ids_full.unsqueeze(0)
            embed = components["embed"]
            if embed is None:
                raise RuntimeError(f"Model {shard.model_id} does not expose input embeddings")
            hidden_states_full = embed(input_ids_full)

        # ---------------------------------------------------------------------
        # KV cache: one HF Cache object per (request, shard range) (T3.1),
        # updated in place by the attention layers (layer_idx = absolute
        # layer index, so a mid-model shard works). Never shared across
        # requests: an in-place-mutated Cache must not sit under two prefixes.
        # ---------------------------------------------------------------------
        from transformers import DynamicCache

        kv_key = f"{request_id}@{shard.start_layer}-{shard.end_layer}" if request_id else None
        new_rows = self._seq_rows(input_data)
        stored = self._seq_kv.get(kv_key) if kv_key else None
        # Unambiguous classification: an input that exactly extends the
        # stored input is full-context (e.g. node._process_tensor rebuilds
        # prompt + generated ids even when it also forwards a kv_len);
        # otherwise a caller-supplied kv_len matching the stored length means
        # "append these new rows".
        state_in = inference_state or {}
        # delta_rows: the caller (distributed decode) wants each non-last stage
        # to forward only the rows it recomputed plus their offset, instead of
        # the hidden states for the whole context on every step.
        want_delta_rows = bool(state_in.get("delta_rows"))
        row_offset = state_in.get("row_offset") if is_hidden_states else None

        prev_out = None
        delta = False
        mode = "full"
        if row_offset is not None:
            # Upstream sent rows [row_offset, row_offset + n): our cache must
            # already hold every position before row_offset.
            row_offset = int(row_offset)
            if row_offset == 0 and (not isinstance(stored, dict) or stored.get("cache") is None):
                stored = {"cache": DynamicCache(), "length": 0,
                          "ids": new_rows[:0].copy()}  # first step: nothing cached yet
            if (not isinstance(stored, dict) or stored.get("cache") is None
                    or int(stored.get("length", 0)) < row_offset):
                raise KVMissError(f"{kv_key}: need {row_offset} cached rows, have "
                                  f"{int(stored.get('length', 0)) if isinstance(stored, dict) else 0}")
            past_len, cache = row_offset, stored["cache"]
            if int(stored["length"]) > row_offset:
                cache.crop(row_offset)
            hidden_states = hidden_states_full
            all_rows = np.concatenate([stored["ids"][:row_offset],
                                       new_rows.astype(stored["ids"].dtype)])
            mode = "offset"
        else:
            ext_len, prev = self._seq_kv_try_hit(
                input_data, kv_key, min_new=int(state_in.get("verify_n") or 1))
            caller_kv = (state_in.get("kv_len") or {}).get(kv_key)
            full_ext = bool(ext_len) and ext_len == int(prev.get("length", 0)) \
                and len(new_rows) > ext_len
            delta = (not full_ext and isinstance(stored, dict) and stored.get("cache") is not None
                     and caller_kv is not None and caller_kv == stored.get("length")
                     and not want_delta_rows)
            if ext_len > 0 and not self._seq_kv_enabled():
                ext_len = 0  # SYNAPSE_SEQ_KV=0: full-context callers recompute

            if delta:
                past_len, cache = int(stored["length"]), stored["cache"]
                hidden_states = hidden_states_full
                all_rows = np.concatenate([stored["ids"], new_rows.astype(stored["ids"].dtype)])
                mode = "delta"
            else:
                past_len = ext_len
                if past_len > 0 and not shard.is_last_layer() and not want_delta_rows:
                    prev_out = prev.get("out")
                    if prev_out is None or prev_out.shape[1] < past_len:
                        past_len = 0  # cannot rebuild the full output -> recompute
                    else:
                        prev_out = prev_out[:, :past_len]
                if past_len > 0:
                    cache = prev["cache"]
                    if int(prev.get("length", 0)) > past_len:
                        cache.crop(past_len)  # drop rows after the divergence
                    hidden_states = hidden_states_full[:, past_len:, :]
                    mode = "reuse"
                    if DEBUG >= 2: print(f"[Cache] SeqHit! Reusing prefix length {past_len}")
                else:
                    prev_out = None
                    hidden_states = hidden_states_full
                    cache = DynamicCache()
                all_rows = new_rows

        seq_len = hidden_states.shape[1]
        if DEBUG >= 1:
            print(f"[PyTorchHF] forward layers {shard.start_layer}-{shard.end_layer}/{shard.n_layers} "
                  f"req={request_id} new_rows={seq_len} past={past_len} "
                  f"mode={mode} "
                  f"input={'hidden' if is_hidden_states else 'ids'}")
        pos_embed = components.get("pos_embed")
        if pos_embed is not None and not is_hidden_states:
            # GPT-2 style absolute positions (first stage only).
            position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long,
                                        device=device).unsqueeze(0)
            hidden_states = hidden_states + pos_embed(position_ids)
        return {
            "shard": shard, "components": components, "layers": layers,
            "hidden": hidden_states, "past_len": int(past_len), "cache": cache,
            "kv_key": kv_key, "all_rows": all_rows, "state": inference_state,
            "want_delta_rows": want_delta_rows, "delta": delta, "stored": stored,
            "prev_out": prev_out, "device": device, "request_id": request_id,
        }

    def _forward_transformer_range(self, shard: Shard, input_data: np.ndarray,
                                   request_id: Optional[str] = None,
                                   inference_state: Optional[dict] = None):
        """Run layers [start, end] of the shard; returns (output, state).

        Caller conventions, all served from a per-(request, shard) KV cache:
        - full context: every call passes prompt + history (+ draft). The
          longest common prefix with the previous input is reused (cropping
          rows after a rejected draft); non-last stages return rows for ALL
          positions, or only the recomputed ones with delta_rows.
        - row_offset (hidden states, delta_rows): rows start at row_offset;
          the cache is cropped there and they are appended.
        - delta: the caller passes back the returned state (which carries
          kv_len) and only the NEW tokens / hidden rows (single-node decode
          loop, ring forwarding). They are appended after the cached context
          and only their rows are returned.
        """
        ctx = self._prepare_forward(shard, input_data, request_id, inference_state)
        hidden = self._run_layers(ctx)
        return self._finish_forward(ctx, hidden)

    def _run_layers(self, ctx) -> Any:
        import torch
        shard, layers, device = ctx["shard"], ctx["layers"], ctx["device"]
        hidden_states, past_len, cache = ctx["hidden"], ctx["past_len"], ctx["cache"]
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device).unsqueeze(0)
        cache_position = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
        position_embeddings = self._build_position_embeddings(hidden_states, position_ids)
        # Causal mask over [past + new] keys. Explicit because "eager"
        # attention (the CPU default) applies NO causal masking when
        # attention_mask is None — every token attended to future tokens.
        attention_mask = self._causal_mask(seq_len, past_len, hidden_states.dtype, device)
        return self._apply_blocks(layers, shard, hidden_states, attention_mask, position_ids,
                                  position_embeddings, cache, cache_position)

    @staticmethod
    def _apply_blocks(layers, shard, hidden_states, attention_mask, position_ids,
                      position_embeddings, cache, cache_position):
        import torch
        with torch.no_grad():
            for layer_index in range(shard.start_layer, shard.end_layer + 1):
                block = layers[layer_index]
                params = inspect.signature(block.forward).parameters
                kwargs: Dict[str, Any] = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "use_cache": True,
                }
                if "position_embeddings" in params:
                    kwargs["position_embeddings"] = position_embeddings
                # transformers >= 4.56 renamed past_key_value -> past_key_values.
                if "past_key_values" in params:
                    kwargs["past_key_values"] = cache
                else:
                    kwargs["past_key_value"] = cache
                if "cache_position" in params:
                    kwargs["cache_position"] = cache_position

                output = block(hidden_states, **kwargs)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                elif hasattr(output, "last_hidden_state"):
                    hidden_states = output.last_hidden_state
                else:
                    hidden_states = output
        return hidden_states

    def _run_layers_batched(self, ctxs) -> List[Any]:
        """One forward for several requests' decode steps on the same shard.

        Each request keeps its own cache and past length: past keys/values
        are right-padded to the longest past and masked, the new rows are
        right-padded and masked, and every request's new keys/values are
        written back into its own cache afterwards. Weights are read once
        per batch instead of once per request (the decode bottleneck)."""
        import torch
        from transformers import DynamicCache
        shard, layers, device = ctxs[0]["shard"], ctxs[0]["layers"], ctxs[0]["device"]
        dtype = ctxs[0]["hidden"].dtype
        pasts = [c["past_len"] for c in ctxs]
        news = [c["hidden"].shape[1] for c in ctxs]
        P, N, B = max(pasts), max(news), len(ctxs)
        H = ctxs[0]["hidden"].shape[-1]
        hidden = torch.zeros(B, N, H, dtype=dtype, device=device)
        for i, c in enumerate(ctxs):
            hidden[i, :news[i]] = c["hidden"][0]
        batch_cache = DynamicCache()
        for li in range(shard.start_layer, shard.end_layer + 1):
            ref = ctxs[0]["cache"].layers[li]
            kh, dk = ref.keys.shape[1], ref.keys.shape[3]
            vh, dv = ref.values.shape[1], ref.values.shape[3]
            K = torch.zeros(B, kh, P, dk, dtype=ref.keys.dtype, device=device)
            V = torch.zeros(B, vh, P, dv, dtype=ref.values.dtype, device=device)
            for i, c in enumerate(ctxs):
                layer = c["cache"].layers[li]
                K[i, :, :pasts[i]] = layer.keys[0, :, :pasts[i]]
                V[i, :, :pasts[i]] = layer.values[0, :, :pasts[i]]
            batch_cache.update(K, V, li)
        neg = torch.finfo(dtype).min
        mask = torch.full((B, 1, N, P + N), neg, dtype=dtype, device=device)
        pos = torch.zeros(B, N, dtype=torch.long, device=device)
        for i in range(B):
            p, n = pasts[i], news[i]
            mask[i, 0, :, :p] = 0.0
            tri = torch.triu(torch.full((N, N), neg, dtype=dtype, device=device), diagonal=1)
            mask[i, 0, :, P:] = tri
            pos[i] = torch.arange(p, p + N, device=device)
        cache_position = torch.arange(P, P + N, dtype=torch.long, device=device)
        position_embeddings = self._build_position_embeddings(hidden, pos)
        out = self._apply_blocks(layers, shard, hidden, mask, pos, position_embeddings,
                                 batch_cache, cache_position)
        results = []
        for i, c in enumerate(ctxs):
            p, n = pasts[i], news[i]
            for li in range(shard.start_layer, shard.end_layer + 1):
                src = batch_cache.layers[li]
                dst = c["cache"].layers[li]
                dst.keys = torch.cat([dst.keys[:, :, :p], src.keys[i:i + 1, :, P:P + n]], dim=2)
                dst.values = torch.cat([dst.values[:, :, :p], src.values[i:i + 1, :, P:P + n]], dim=2)
            results.append(out[i:i + 1, :n])
        return results

    def _finish_forward(self, ctx, hidden_states):
        import torch
        shard, components = ctx["shard"], ctx["components"]
        kv_key, all_rows, cache = ctx["kv_key"], ctx["all_rows"], ctx["cache"]
        past_len, stored, prev_out = ctx["past_len"], ctx["stored"], ctx["prev_out"]
        with torch.no_grad():
            out_state = dict(ctx["state"] or {})
            if kv_key:
                kv_map = dict(out_state.get("kv_len") or {})
                kv_map[kv_key] = int(all_rows.shape[0])
                out_state["kv_len"] = kv_map

            out_state.pop("row_offset", None)
            if not shard.is_last_layer() and ctx["want_delta_rows"]:
                # Forward only the recomputed rows; the next stage crops its
                # own cache to row_offset and appends them.
                out_state["row_offset"] = int(past_len)
                if kv_key:
                    self._seq_kv_commit(all_rows, kv_key, cache, out_hidden=None)
                return hidden_states.float().cpu().numpy(), out_state
            if not shard.is_last_layer():
                full_out = None
                if ctx["delta"]:
                    old = stored.get("out")
                    if old is not None:
                        full_out = torch.cat([old, hidden_states], dim=1)
                else:
                    # Full-context callers need rows for EVERY position (the
                    # next stage matches on prefix rows): cached prefix
                    # output + the new rows.
                    if prev_out is not None and past_len > 0:
                        hidden_states = torch.cat([prev_out, hidden_states], dim=1)
                    full_out = hidden_states
                if kv_key:
                    self._seq_kv_commit(all_rows, kv_key, cache, out_hidden=full_out)
                return hidden_states.float().cpu().numpy(), out_state

            if kv_key:
                # T3.1: record this call for the next incremental step.
                self._seq_kv_commit(all_rows, kv_key, cache)

            final_norm = components["final_norm"]
            if final_norm is not None:
                hidden_states = final_norm(hidden_states)

            lm_head = components["lm_head"]
            if lm_head is None:
                raise RuntimeError(f"Model {shard.model_id} does not expose an output head")

            logits = lm_head(hidden_states)
            return logits.float().cpu().numpy(), out_state

    def _record_layer_ms(self, ctx, seconds: float) -> None:
        """EWMA of one decoder layer's decode-step time (ms) per model, from
        real forwards; published for measured partitioning."""
        sh = ctx["shard"]
        if ctx["past_len"] <= 0 or ctx["hidden"].shape[1] > self.BATCH_MAX_NEW_ROWS:
            return  # prefill costs differently; decode is what pipelines run
        n = sh.end_layer - sh.start_layer + 1
        ms = seconds * 1000.0 / max(1, n)
        prev = self.layer_ms.get(sh.model_id)
        self.layer_ms[sh.model_id] = ms if prev is None else 0.8 * prev + 0.2 * ms

    def _run_jobs(self, jobs) -> List[Any]:
        """Run queued forwards (under the forward lock). Decode steps of
        different requests on the same shard share one batched forward;
        prefills and everything else run one by one. Returns, per job,
        (output, state) or the exception it raised."""
        results: List[Any] = [None] * len(jobs)
        prepared = {}
        for idx, (shard, data, rid, state) in enumerate(jobs):
            try:
                prepared[idx] = self._prepare_forward(shard, data, rid, state)
            except Exception as e:
                results[idx] = e
        groups: Dict[Any, List[int]] = {}
        for idx, ctx in prepared.items():
            sh = ctx["shard"]
            key = (sh.model_id, sh.start_layer, sh.end_layer)
            if ctx["past_len"] > 0 and ctx["hidden"].shape[1] <= self.BATCH_MAX_NEW_ROWS \
                    and all(ctx["cache"].layers[li].keys.shape[2] == ctx["past_len"]
                            for li in (sh.start_layer, sh.end_layer)):
                groups.setdefault(key, []).append(idx)
            else:
                groups.setdefault(("solo", idx), []).append(idx)
        for key, idxs in groups.items():
            try:
                if len(idxs) == 1:
                    import time as _time
                    c = prepared[idxs[0]]
                    t0 = _time.perf_counter()
                    outs = [self._run_layers(c)]
                    self._record_layer_ms(c, _time.perf_counter() - t0)
                else:
                    outs = self._run_layers_batched([prepared[i] for i in idxs])
                    self.batch_stats["batched_forwards"] += 1
                    self.batch_stats["batched_jobs"] += len(idxs)
                    self.batch_stats["max_batch"] = max(self.batch_stats["max_batch"], len(idxs))
                for i, h in zip(idxs, outs):
                    results[i] = self._finish_forward(prepared[i], h)
            except Exception as e:
                for i in idxs:
                    results[i] = e
            self.batch_stats["forwards"] += 1
        return results

    def clear_cache(self, request_id: Optional[str] = None):
        if request_id:
            # T3.1: per-request incremental state MUST die with the request,
            # otherwise a later request reusing the id inherits stale KVs.
            prefix = f"{request_id}@"
            for k in [k for k in self._seq_kv if k.startswith(prefix)]:
                del self._seq_kv[k]
            self._active.pop(request_id, None)  # finished: no longer batch company
        else:
            self._seq_kv.clear()
            if DEBUG >= 1: print("[PyTorchHF] Cleared all KV caches")

    @staticmethod
    def _partial_enabled() -> bool:
        """SYNAPSE_PARTIAL_LOAD=0 restores whole-model loading everywhere."""
        return os.getenv("SYNAPSE_PARTIAL_LOAD", "1") != "0"

    def _device_dtype(self):
        import torch
        if self._pm is not None:
            return torch.device(self._pm.device), self._pm.dtype
        p = next(self._model.parameters())
        return p.device, p.dtype

    def _loading_lock(self):
        # One load at a time: concurrent first requests raced transformers'
        # lazy imports in two threads ("cannot import name AutoTokenizer").
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

    async def ensure_shard(self, shard: Shard) -> None:
        async with self._loading_lock():
            await self._ensure_shard_locked(shard)

    async def _ensure_shard_locked(self, shard: Shard) -> None:
        if self._partial_enabled() and self._model is None:
            try:
                await self._ensure_partial(shard)
                return
            except FileNotFoundError as e:
                # No safetensors (e.g. .bin-only repo): whole-model fallback.
                if DEBUG >= 1:
                    print(f"[PyTorchHF] layer-range load unavailable ({e}); loading full model")
        await self._ensure_full(shard)

    async def _ensure_partial(self, shard: Shard) -> None:
        """Materialize only [start, end] (+ embedding / head at the ends)."""
        pm = self._pm
        loop = asyncio.get_running_loop()
        if pm is None or self._model_id != shard.model_id:
            from synapse.inference.partial_loader import PartialModel
            from synapse.inference import weight_quant as WQ
            import torch
            hf_id = self._get_hf_id(shard.model_id)
            hw = self._detect_hardware_profile(shard.model_id)
            device = "cuda" if hw["device"] == "cuda" else "cpu"
            dtype = hw["torch_dtype"] if device == "cuda" else torch.float32

            def _build():
                pm_ = PartialModel(hf_id, dtype, device, hw["attn_implementation"])
                per_layer = pm_.layer_param_count() * torch.tensor([], dtype=dtype).element_size()
                need = per_layer * (shard.end_layer - shard.start_layer + 1)
                pm_.quant_mode = WQ.choose_mode(need, self._available_bytes(device),
                                                WQ.requested_mode())
                return pm_


            with self._forward_lock:
                self._seq_kv.clear()
            self.tokenizer = await loop.run_in_executor(None, self._load_tokenizer, hf_id)
            pm = await loop.run_in_executor(None, _build)
            self._pm = pm
            self._model_id = shard.model_id
            self._tok_model_id = shard.model_id
            if DEBUG >= 1:
                print(f"[PyTorchHF] {shard.model_id}: layer-range loading on {device} "
                      f"({dtype}), weight quant={pm.quant_mode}")
        first, last = shard.is_first_layer(), shard.is_last_layer()
        needs = any(i not in pm.layers for i in range(shard.start_layer, shard.end_layer + 1)) \
            or (first and pm.embed is None) or (last and pm.lm_head is None)
        if needs:
            def _do():
                with self._forward_lock:
                    pm.ensure(shard.start_layer, shard.end_layer, first, last)
            await loop.run_in_executor(None, _do)
        self.shard = shard

    @staticmethod
    def _available_bytes(device: str) -> int:
        try:
            if device == "cuda":
                import torch
                free, _ = torch.cuda.mem_get_info()
                return int(free)
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            return 0

    def loaded_layers(self) -> List[int]:
        """Absolute indices of the decoder layers held in memory."""
        if self._pm is not None:
            return sorted(self._pm.layers)
        comps = self._get_transformer_components() if self._model is not None else None
        return list(range(len(comps["layers"]))) if comps else []

    def release_layers(self, keep: List[int]) -> List[int]:
        """Free loaded layers outside `keep` (layer-range mode only)."""
        if self._pm is None:
            return []
        with self._forward_lock:
            freed = self._pm.release(keep)
            if freed:
                self._seq_kv.clear()
        return freed

    async def _ensure_full(self, shard: Shard) -> None:
        """Whole HF model (training, evaluation, non-safetensors repos)."""
        if self._model_id == shard.model_id and self._model is not None:
            return
        self._pm = None
        hf_id = self._get_hf_id(shard.model_id)
        if DEBUG >= 1:
            print(f"[PyTorchHF] Loading {shard.model_id} -> {hf_id}")

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
            loop = asyncio.get_event_loop()
    
            def _tok_local():
                return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, local_files_only=True)
            def _model_local():
                return AutoModelForCausalLM.from_pretrained(hf_id, **load_model_kw, local_files_only=True)
            def _tok_remote():
                return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            def _model_remote():
                return AutoModelForCausalLM.from_pretrained(hf_id, **load_model_kw)
    
            try:
                self.tokenizer = await loop.run_in_executor(None, _tok_local)
                self._model = await loop.run_in_executor(None, _model_local)
            except (OSError, ValueError):
                # --- [P2P LAN Cache Sync] ---
                try:
                    node_ref = getattr(self, "node", None)
                    if node_ref:
                        nodes = await node_ref.get_tailscale_nodes()
                        from synapse.networking.tailscale.tailscale_helpers import get_synapse_api_urls_from_node_list
                        from synapse.helpers import get_all_ip_addresses_and_interfaces
                        self_ips = set(ip for ip, _ in get_all_ip_addresses_and_interfaces() if ip)
                        urls = get_synapse_api_urls_from_node_list(nodes, api_port=52415, only_synapse_nodes=True, exclude_ips=self_ips)
                        if urls:
                            master_url = urls[0]
                            if DEBUG >= 1: print(f"[PyTorchHF] Requesting P2P LAN Cache sync from: {master_url}...")
                except Exception as e:
                    if DEBUG >= 2: print(f"[P2P Sync Exception]: {e}")
    
                self.tokenizer = await loop.run_in_executor(None, _tok_remote)
                self._model = await loop.run_in_executor(None, _model_remote)
    
            # ÉP MODEL LÊN GPU (Nếu không dùng quantization/device_map)
            if hw["device"] == "cuda" and "device_map" not in load_model_kw:
                self._model = self._model.to("cuda")
    
            self._model_id = shard.model_id
            self._tok_model_id = shard.model_id
            self.shard = shard
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF load {hf_id}: {e}") from e

    @staticmethod
    def _load_tokenizer(hf_id: str):
        """Local cache first, then the Hub (unless offline)."""
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True,
                                                 local_files_only=True)
        except (OSError, ValueError):
            return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    async def _ensure_tokenizer(self, shard: Shard) -> None:
        """Tokenizer only: encoding must not pull every layer into memory
        (the orchestrator encodes with the full-model base shard)."""
        if self.tokenizer is not None and self._tok_model_id in (None, shard.model_id):
            return
        if self._model is not None and self._model_id == shard.model_id:
            return
        hf_id = self._get_hf_id(shard.model_id)

        self.tokenizer = await asyncio.get_running_loop().run_in_executor(
            None, self._load_tokenizer, hf_id)
        self._tok_model_id = shard.model_id

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        async with self._loading_lock():
            await self._ensure_tokenizer(shard)
        if self.tokenizer is None:
            return np.array([], dtype=np.int32)
        enc = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="np")
        return enc.flatten().astype(np.int32)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        async with self._loading_lock():
            await self._ensure_tokenizer(shard)
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

    # Decode steps with at most this many new rows join a batched forward.
    BATCH_MAX_NEW_ROWS = 8

    @staticmethod
    def _batching_enabled() -> bool:
        """SYNAPSE_BATCH=0 runs every forward on its own."""
        return os.getenv("SYNAPSE_BATCH", "1") != "0"

    @staticmethod
    def _batch_window_s() -> float:
        try:
            return max(0.0, float(os.getenv("SYNAPSE_BATCH_WINDOW_MS", "3"))) / 1000.0
        except ValueError:
            return 0.003

    async def _submit(self, shard, input_data, request_id, inference_state):
        """Queue one forward; a single drain task runs whatever is queued
        together (continuous batching at step granularity). It waits a few
        ms for company only while more than one request is active, so a
        lone request pays no batching delay."""
        import time as _time
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        now = _time.monotonic()
        if request_id:
            self._active[request_id] = now
        horizon = max(0.25, 3.0 * self._fwd_ewma_s)
        for rid in [r for r, t in self._active.items() if now - t > horizon]:
            del self._active[rid]
        self._pending.append(((shard, input_data, request_id, inference_state), fut))
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = loop.create_task(self._drain())
        return await fut

    async def _drain(self) -> None:
        while self._pending:
            if len(self._active) > 1:
                await asyncio.sleep(self._batch_window_s())
            batch, self._pending = self._pending, []
            jobs = [j for j, _ in batch]

            def _locked():
                with self._forward_lock:
                    return self._run_jobs(jobs)
            import time as _time
            t0 = _time.monotonic()
            try:
                results = await asyncio.to_thread(_locked)
            except Exception as e:  # never strand a waiter
                results = [e] * len(batch)
            dt = _time.monotonic() - t0
            self._fwd_ewma_s = dt if not self._fwd_ewma_s else 0.8 * self._fwd_ewma_s + 0.2 * dt
            for (_, fut), res in zip(batch, results):
                if fut.done():
                    continue
                if isinstance(res, BaseException):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)

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
            if self._model is None and self._pm is None:
                vocab_size = 32000
                return np.zeros((1, 0, vocab_size), dtype=np.float32), inference_state
            
            if not self._batching_enabled():
                def _locked():
                    # One forward at a time: concurrent calls for the same
                    # request would mutate the same in-place KV Cache.
                    with self._forward_lock:
                        return self._forward_transformer_range(
                            shard, input_data, request_id, inference_state)
                return await asyncio.to_thread(_locked)
            return await self._submit(shard, input_data, request_id, inference_state)
        except KVMissError:
            raise  # the node turns this into a kv_miss reply, not a crash
        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF infer_tensor: {e}") from e

    async def load_checkpoint(self, shard: Shard, path: str) -> bool:
        """Restore weights from a directory written by save_checkpoint.

        Returns True only when weights were actually restored. Returns False
        (loudly, never silently) for metadata-only JSON checkpoints, missing
        paths, or any load failure — red-team B1: a silent pass let failover
        report success while resuming from scratch.

        Scope: full-model directory restore. Sharded partial-range restore
        (loading only [start_layer, end_layer] weights) is NOT implemented;
        callers serving a partial shard must treat False as degraded resume
        (see Node.check_fault_tolerance).
        """
        try:
            import os
            if path and os.path.isfile(path) and path.endswith(SHARD_WEIGHTS_SUFFIX):
                return self._restore_shard_weights(shard, path)
            if not path or not os.path.isdir(path):
                if DEBUG >= 1:
                    print(f"[PyTorchHF] load_checkpoint: no weights at {path!r} "
                          f"(metadata-only or missing) -> restore no-op")
                return False
            names = set(os.listdir(path))
            has_config = "config.json" in names
            has_weights = any(
                n == "pytorch_model.bin" or n.endswith(".safetensors") or n.endswith(".bin")
                for n in names
            )
            if not (has_config and has_weights):
                if DEBUG >= 1:
                    print(f"[PyTorchHF] load_checkpoint: {path!r} is not a weight "
                          f"dir -> restore no-op")
                return False
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import asyncio as _aio
            hw = self._detect_hardware_profile(shard.model_id)
            kw: Dict[str, Any] = {
                "trust_remote_code": True,
                "attn_implementation": hw["attn_implementation"],
                "torch_dtype": hw["torch_dtype"],
                "local_files_only": True,
            }
            loop = _aio.get_event_loop()
            self.tokenizer = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                                            local_files_only=True))
            self._model = await loop.run_in_executor(
                None, lambda: AutoModelForCausalLM.from_pretrained(path, **kw))
            self._pm = None  # a full-model restore replaces layer-range state
            self._seq_kv.clear()
            if hw["device"] == "cuda" and "device_map" not in kw:
                self._model = self._model.to("cuda")
            self._model_id = shard.model_id
            self.shard = shard
            if DEBUG >= 1:
                print(f"[PyTorchHF] load_checkpoint: restored weights from {path!r}")
            return True
        except Exception as e:
            if DEBUG >= 1:
                print(f"[PyTorchHF] load_checkpoint failed for {path!r}: {e} -> restore no-op")
            return False

    def save_shard_weights(self, shard: Shard, path: str) -> bool:
        """Write layers [start_layer, end_layer] of the loaded model to `path`.

        Only the shard's own decoder layers are saved (not the full model), keyed
        by absolute layer index so any node holding the same base model can
        restore them into its layer stack. Written to a temp file then
        os.replace()d, so a crash mid-write never leaves a torn checkpoint.
        """
        comps = self._get_transformer_components()
        if comps is None or self._model_id != shard.model_id:
            return False
        try:
            import os
            import torch
            layers = comps["layers"]
            if not (0 <= shard.start_layer <= shard.end_layer < len(layers)):
                return False
            payload = {
                "format": SHARD_WEIGHTS_FORMAT,
                "model_id": shard.model_id,
                "start_layer": shard.start_layer,
                "end_layer": shard.end_layer,
                "layers": {
                    i: {k: v.detach().to("cpu").clone() for k, v in layers[i].state_dict().items()}
                    for i in range(shard.start_layer, shard.end_layer + 1)
                },
            }
            tmp = path + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f"[PyTorchHF] save_shard_weights failed for {path!r}: {e}")
            return False

    def _restore_shard_weights(self, shard: Shard, path: str) -> bool:
        """Copy saved layers covering [shard.start_layer, shard.end_layer] into
        the already-loaded model. Partial coverage or a model mismatch returns
        False (degraded) rather than resuming with a half-restored range."""
        comps = self._get_transformer_components()
        if comps is None or self._model_id != shard.model_id:
            print(f"[PyTorchHF] load_checkpoint: base model {shard.model_id!r} not loaded "
                  f"-> cannot restore shard weights from {path!r}")
            return False
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != SHARD_WEIGHTS_FORMAT or payload.get("model_id") != shard.model_id:
            print(f"[PyTorchHF] load_checkpoint: {path!r} is not a shard-weights file "
                  f"for {shard.model_id!r}")
            return False
        saved = payload.get("layers", {})
        wanted = range(shard.start_layer, shard.end_layer + 1)
        layers = comps["layers"]
        missing = [i for i in wanted if i not in saved or i >= len(layers)]
        if missing:
            print(f"[PyTorchHF] load_checkpoint: {path!r} lacks layers {missing} "
                  f"-> restore refused (degraded)")
            return False
        with torch.no_grad():
            for i in wanted:
                layers[i].load_state_dict(saved[i], strict=True)
        # Cached per-request KV was computed with the old weights.
        self._seq_kv.clear()
        print(f"[PyTorchHF] load_checkpoint: restored layers "
              f"{shard.start_layer}-{shard.end_layer} from {path!r}")
        return True

    async def save_checkpoint(self, shard: Shard, path: str) -> None:
        if self._model is None and self._pm is not None:
            await self._ensure_full(shard)
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

    async def _back_gradient(
        self,
        request_id: str,
        shard: Shard,
        example: np.ndarray,
        target: np.ndarray,
        length: np.ndarray,
        backgrad: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """Backward phase: real gradients via torch autograd.

        Re-runs the stage's forward on its micro-batch, computes the loss,
        and backprops through the real model. Returns ``(None, grads)`` where
        ``grads["__input_grad__"]`` is the gradient of the loss w.r.t. the
        stage's input (the embedding output — the tensor the upstream stage's
        backward consumes) and the remaining entries are per-parameter
        gradients. When ``backgrad`` (the gradient handed back from the
        downstream stage through the pipeline send path) is shape-compatible
        with the model logits, it is used as the backward seed instead of the
        local loss.

        Returns ``(None, None)`` when no model/torch is available — the
        caller treats that as "no backward gradients for this micro-batch".
        """
        try:
            import torch
            import torch.nn.functional as F

            if self._model is None:
                return None, None

            device = self._device
            self._model.train()

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

            input_grad: Optional[np.ndarray] = None
            capture_handle = None
            get_embeddings = getattr(self._model, "get_input_embeddings", None)
            if callable(get_embeddings):
                emb_module = get_embeddings()
                if emb_module is not None:
                    def capture(module, grad_input, grad_output):
                        nonlocal input_grad
                        g = grad_output[0] if grad_output else None
                        if g is not None:
                            input_grad = g.detach().float().cpu().numpy()
                    capture_handle = emb_module.register_full_backward_hook(capture)

            try:
                out = self._model(input_ids)
                if hasattr(out, "logits"):
                    logits = out.logits
                elif isinstance(out, torch.Tensor):
                    # Plain nn.Module: the forward output IS the logits tensor.
                    # (out[0] would silently drop the batch dim.)
                    logits = out
                else:
                    logits = out[0]
                if backgrad is not None and tuple(np.asarray(backgrad).shape) == tuple(logits.shape):
                    # Downstream gradient is shape-compatible with our output:
                    # seed the backward with it (real external-gradient
                    # backprop) instead of the local loss.
                    seed = torch.from_numpy(
                        np.asarray(backgrad, dtype=np.float32)
                    ).to(device)
                    torch.autograd.backward(logits, grad_tensors=seed)
                else:
                    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                    shift_labels = labels[..., 1:].contiguous().view(-1)
                    loss_val = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                    loss_val.backward()
            finally:
                if capture_handle is not None:
                    capture_handle.remove()

            grads: Dict[str, Any] = {}
            if input_grad is not None:
                grads["__input_grad__"] = input_grad
            for name, p in self._model.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.detach().float().cpu().numpy()
            return None, (grads if grads else None)
        except Exception as e:
            if DEBUG >= 1:
                print(f"[PyTorchHF] back_gradient: {e}")
            return None, None

    async def train(
        self,
        request_id: str,
        shard: Shard,
        example: np.ndarray,
        target: np.ndarray,
        length: np.ndarray,
        backgrad: Optional[np.ndarray] = None,
        loss: str = "forward",
        sync_gradients: bool = False,
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        # Training runs the whole HF model (every stage re-runs its forward
        # locally), so it needs all layers, not the served range.
        await self._ensure_full(shard)
        if loss == "back_gradient":
            return await self._back_gradient(
                request_id, shard, example, target, length, backgrad
            )

        try:
            import torch
            import torch.nn.functional as F

            if self._model is None:
                return None, None

            device = self._device
            self._model.train()

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
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss_val = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            opt = self._get_optimizer()
            if opt is not None:
                opt.zero_grad()
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)

                if sync_gradients:
                    grads = {}
                    for name, param in self._model.named_parameters():
                        if param.grad is not None:
                            grads[name] = param.grad.detach().cpu().numpy().copy()
                    return float(loss_val.item()), grads
                else:
                    opt.step()

            return float(loss_val.item()), None

        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"PyTorchHF train error: {e}") from e

    async def apply_gradient_updates(self, shard: Shard, updates: dict) -> None:
        if self._model is None:
            return
        try:
            import torch
            for name, param in self._model.named_parameters():
                if name in updates:
                    update_val = torch.from_numpy(updates[name]).to(param.device).to(param.dtype)
                    param.data.add_(update_val)
            if DEBUG >= 2:
                print(f"[PyTorchHF] Applied {len(updates)} gradient updates")
        except Exception as e:
            if DEBUG >= 1:
                print(f"[PyTorchHF] apply_gradient_updates error: {e}")

    async def evaluate(
        self,
        request_id: str,
        shard: Shard,
        example: np.ndarray,
        target: np.ndarray,
        length: np.ndarray,
    ) -> Optional[float]:
        """Đánh giá: forward, tính loss (không backward)."""
        await self._ensure_full(shard)
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
        except Exception:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            return None
