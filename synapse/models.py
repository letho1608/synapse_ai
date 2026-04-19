"""
Model Registry - Quản lý mô hình trong hệ thống

Thay thế model_cards cũ bằng hệ thống registry linh hoạt hơn
"""

from synapse.inference.shard import Shard
from typing import Optional, List, Dict, Any
from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS


# Registry đơn giản để quản lý mô hình
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


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


def get_pretty_name(model_id: str) -> Optional[str]:
    """Lấy tên đẹp của mô hình"""
    if model_id not in _MODEL_REGISTRY:
        return model_id
    return _MODEL_REGISTRY[model_id].get("pretty_name", model_id)


def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
    """
    Tạo base shard cho mô hình
    
    Args:
        model_id: ID của mô hình
        inference_engine_classname: Tên inference engine
        
    Returns:
        Shard object hoặc None
    """
    n_layers = 0
    
    if model_id in _MODEL_REGISTRY:
        model_info = _MODEL_REGISTRY[model_id]
        n_layers = model_info.get("layers", 0)
    elif model_id in HF_MODEL_LAYERS:
        # Dự phòng từ danh sách động (hf_models.json)
        n_layers = HF_MODEL_LAYERS[model_id]
    else:
        # Thử tìm kiếm mờ trong HF_MODEL_LAYERS
        for k, v in HF_MODEL_LAYERS.items():
            if k.lower() == model_id.lower():
                n_layers = v
                break
    
    if n_layers < 1:
        return None
    
    return Shard(model_id, 0, 0, n_layers)


def build_full_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
    """Tạo full shard (tất cả layers)"""
    base_shard = build_base_shard(model_id, inference_engine_classname)
    if base_shard is None:
        return None
    return Shard(base_shard.model_id, 0, base_shard.n_layers - 1, base_shard.n_layers)


def get_supported_models(supported_inference_engine_lists: Optional[List[List[str]]] = None) -> List[str]:
    """
    Lấy danh sách mô hình được hỗ trợ
    
    Args:
        supported_inference_engine_lists: Danh sách inference engines
        
    Returns:
        List model IDs
    """
    models = set(_MODEL_REGISTRY.keys())
    models.update(HF_MODELS.values())
    return sorted(list(models))


def list_models() -> List[str]:
    """Liệt kê tất cả mô hình"""
    models = set(_MODEL_REGISTRY.keys())
    models.update(HF_MODELS.values())
    return sorted(list(models))


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Lấy thông tin mô hình"""
    if model_id in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_id]
    
    if model_id in HF_MODELS.values():
        # Tạo info cơ bản cho model từ HF
        return {
            "layers": HF_MODEL_LAYERS.get(model_id, 0),
            "pretty_name": model_id
        }
    return None


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
