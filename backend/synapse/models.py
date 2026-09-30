"""
Model Registry - Quản lý mô hình trong hệ thống

Thay thế model_cards cũ bằng hệ thống registry linh hoạt hơn
"""

from synapse.inference.shard import Shard
from typing import Optional, Dict, Any
from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS, resolve_hf_id


# Registry đơn giản để quản lý mô hình
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# repo id -> transformer layer count read from its config.json.
_CONFIG_LAYERS: Dict[str, int] = {}


def layers_from_config(repo_id: Optional[str]) -> int:
    """Transformer block count from the model's config.json (0 if unknown).

    Local HF cache first; downloads just config.json otherwise (skipped when
    HF_HUB_OFFLINE=1). The layer count must be exact: a shard whose
    end_layer stops short of the real last block silently skips layers.
    """
    if not repo_id or "/" not in repo_id:
        return 0
    if repo_id in _CONFIG_LAYERS:
        return _CONFIG_LAYERS[repo_id]
    import json
    import os
    path = None
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo_id, "config.json")
        path = cached if isinstance(cached, str) else None
        if path is None and os.environ.get("HF_HUB_OFFLINE", "0") != "1":
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id, "config.json", etag_timeout=10)
    except Exception:
        path = None
    n = 0
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg = cfg.get("text_config", cfg) if isinstance(cfg.get("text_config"), dict) else cfg
            for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
                if isinstance(cfg.get(key), int) and cfg[key] > 0:
                    n = cfg[key]
                    break
        except Exception:
            n = 0
    if n:
        _CONFIG_LAYERS[repo_id] = n
    return n


def register_model(
    model_id: str,
    num_layers: int,
    repo_path: Optional[str] = None,
    **kwargs
):
    """
    Đăng ký mô hình vào registry
    
    Args:
        model_id: ID của mô hình
        num_layers: Số lượng layers
        repo_path: Path đến mô hình (local path)
        **kwargs: Các tham số khác
    """
    _MODEL_REGISTRY[model_id] = {
        "layers": num_layers,
        "repo_path": repo_path,
        **kwargs
    }


def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
    """
    Lấy repo path của mô hình
    
    Args:
        model_id: ID của mô hình
        inference_engine_classname: Tên inference engine
        
    Returns:
        Path đến mô hình hoặc None
    """
    if model_id not in _MODEL_REGISTRY:
        # Dự phòng: Nếu có trong danh sách HF thì dùng chính model_id làm path
        if model_id in HF_MODELS or any(v == model_id for v in HF_MODELS.values()):
            return model_id
        resolved = resolve_hf_id(model_id)
        if resolved:
            return resolved
        return None
    
    model_info = _MODEL_REGISTRY[model_id]
    
    # Nếu có repo_path, dùng nó
    if "repo_path" in model_info and model_info["repo_path"]:
        return model_info["repo_path"]
    
    # Nếu có repo dict (legacy), dùng nó
    if "repo" in model_info and isinstance(model_info["repo"], dict):
        return model_info["repo"].get(inference_engine_classname)
    
    # Mặc định: dùng model_id làm path
    return model_id


def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
    """
    Tạo base shard cho mô hình
    
    Args:
        model_id: ID của mô hình
        inference_engine_classname: Tên inference engine
        
    Returns:
        Shard object hoặc None
    """
    if model_id == "__none__":
        return None
        
    n_layers = 0
    
    if model_id in _MODEL_REGISTRY:
        model_info = _MODEL_REGISTRY[model_id]
        n_layers = model_info.get("layers", 0)
    elif model_id in HF_MODEL_LAYERS:
        n_layers = HF_MODEL_LAYERS[model_id]
    else:
        resolved = resolve_hf_id(model_id)
        if resolved in HF_MODEL_LAYERS:
            n_layers = HF_MODEL_LAYERS[resolved]
        else:
            normalized = model_id.lower().replace(":", "-").replace("_", "-")
            if normalized in HF_MODEL_LAYERS:
                n_layers = HF_MODEL_LAYERS[normalized]
            else:
                for k, v in HF_MODEL_LAYERS.items():
                    if k.lower() == normalized:
                        n_layers = v
                        break
                # Also check registered models with normalized key
                if n_layers < 1:
                    if normalized in _MODEL_REGISTRY:
                        n_layers = _MODEL_REGISTRY[normalized].get("layers", 0)
                    else:
                        for reg_key, reg_info in _MODEL_REGISTRY.items():
                            if reg_key.lower() == normalized:
                                n_layers = reg_info.get("layers", 0)
                                break
    
    if n_layers < 1:
        # Unknown statically: read the real count from config.json.
        repo = get_repo(model_id, inference_engine_classname) or resolve_hf_id(model_id)
        n_layers = layers_from_config(repo)
        if n_layers >= 1 and model_id in _MODEL_REGISTRY:
            _MODEL_REGISTRY[model_id]["layers"] = n_layers

    if n_layers < 1:
        return None

    return Shard(model_id, 0, 0, n_layers)


def build_full_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
    """Tạo full shard (tất cả layers)"""
    base_shard = build_base_shard(model_id, inference_engine_classname)
    if base_shard is None:
        return None
    return Shard(base_shard.model_id, 0, base_shard.n_layers - 1, base_shard.n_layers)


# ============================================================================
# Đăng ký các mô hình mặc định
# ============================================================================

# Qwen 2.5 1.5B
register_model(
    model_id="qwen2.5-1.5b",
    num_layers=28,
    repo_path="qwen2.5-1.5b",  # Local path
    pretty_name="Qwen 2.5 1.5B Instruct",
    model_type="llama",  # Dùng Llama architecture
    hidden_size=1536,
    vocab_size=151936
)

# GPT-2 Small
register_model(
    model_id="gpt2",
    num_layers=12,
    repo_path="gpt2",
    pretty_name="GPT-2 Small",
    model_type="gpt2",
    hidden_size=768,
    vocab_size=50257
)


# Custom models có thể được thêm vào đây hoặc từ file config
