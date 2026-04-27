"""
Danh sách model dùng cho: download (HF), hiển thị web UI, và inference.
Một nguồn duy nhất cho download và web UI (PyTorch/HF).
"""

import json
import os
from pathlib import Path
from typing import Dict

# Các dictionary được dùng chung trên toàn cấu trúc của dự án, nay sẽ nạp động từ file JSON
HF_MODELS: Dict[str, str] = {}
HF_MODEL_PARAMS: Dict[str, str] = {}
HF_MODEL_LAYERS: Dict[str, int] = {}
VISION_MODELS = set()

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Tương thích ngược với các alias cũ đang dùng ở CLI/API/tests.
LEGACY_MODEL_ALIASES = {
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
}

def _load_dynamic_config():
    db_path = Path(os.path.dirname(__file__)) / "data" / "hf_models.json"
    try:
        if db_path.exists():
            with open(db_path, "r", encoding="utf-8") as f:
                db_data = json.load(f)
            
            for entry in db_data:
                repo_id = entry.get("name")
                if not repo_id:
                    continue
                
                # Bản thân repo_id chính là khóa tìm kiếm và tên trên UI
                HF_MODELS[repo_id] = repo_id
                
                # Nạp tham số
                if entry.get("parameter_count"):
                    HF_MODEL_PARAMS[repo_id] = entry["parameter_count"]
                    
                # Nạp số layer
                if entry.get("num_hidden_layers"):
                    HF_MODEL_LAYERS[repo_id] = entry["num_hidden_layers"]
                
                # Kiểm tra vision capability
                if "vision" in entry.get("capabilities", []) or "vision" in entry.get("pipeline_tag", "").lower():
                    VISION_MODELS.add(repo_id)
    except Exception:
        pass


def _apply_legacy_aliases() -> None:
    """Bổ sung alias cũ vào map động để giữ tương thích cho code hiện hữu."""
    for alias, target in LEGACY_MODEL_ALIASES.items():
        if target not in HF_MODELS.values():
            continue
        HF_MODELS[alias] = target
        if target in HF_MODEL_PARAMS:
            HF_MODEL_PARAMS[alias] = HF_MODEL_PARAMS[target]
        if target in HF_MODEL_LAYERS:
            HF_MODEL_LAYERS[alias] = HF_MODEL_LAYERS[target]
        if target in VISION_MODELS:
            VISION_MODELS.add(alias)

# Chạy nạp dữ liệu một lần khi module load
_load_dynamic_config()
_apply_legacy_aliases()


def resolve_hf_id(model_name: str) -> str:
    """Tên model -> Hugging Face repo_id (dùng download + UI)."""
    if not model_name:
        return DEFAULT_HF_MODEL  # Fallback default
        
    key = model_name.strip()
    if key.lower().startswith("ollama/"):  # tương thích prefix cũ
        key = key[7:].strip()

    # Ưu tiên alias tường minh để tránh match nhầm bản coder/awq/gptq.
    alias_target = LEGACY_MODEL_ALIASES.get(key.lower())
    if alias_target:
        return alias_target
        
    # Check if exact match exists
    if key in HF_MODELS:
        return HF_MODELS[key]
        
    # Check lowercase mapping
    for k, v in HF_MODELS.items():
        if k.lower() == key.lower():
            return v

    # Tương thích với định dạng family:size (vd. qwen2.5:3b), có ranking để chọn bản chuẩn.
    if ":" in key:
        family, size = key.lower().split(":", 1)
        candidates = []
        for _, repo_id in HF_MODELS.items():
            rid = repo_id.lower()
            if family in rid and size in rid:
                score = 0
                if "instruct" in rid:
                    score += 3
                if "coder" in rid and "coder" not in family and "coder" not in key.lower():
                    score -= 2
                if any(tag in rid for tag in ("-awq", "-gptq", "-gguf")):
                    score -= 3
                candidates.append((score, len(rid), repo_id))

        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            return candidates[0][2]
            
    # Always fallback to the raw key, meaning they explicitly typed a repo_id not in initial scan
    return key
