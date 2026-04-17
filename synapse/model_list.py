"""
Danh sách model dùng cho: download (HF), hiển thị web UI, và inference.
Một nguồn duy nhất cho download và web UI (PyTorch/HF).
"""

from typing import Dict

import json
import os
from pathlib import Path
from typing import Dict

# Các dictionary được dùng chung trên toàn cấu trúc của dự án, nay sẽ nạp động từ file JSON
HF_MODELS: Dict[str, str] = {}
HF_MODEL_PARAMS: Dict[str, str] = {}
HF_MODEL_LAYERS: Dict[str, int] = {}
VISION_MODELS = set()

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

# Chạy nạp dữ liệu một lần khi module load
_load_dynamic_config()


def resolve_hf_id(model_name: str) -> str:
    """Tên model -> Hugging Face repo_id (dùng download + UI)."""
    if not model_name:
        return "Qwen/Qwen2.5-1.5B-Instruct"  # Fallback default
        
    key = model_name.strip()
    if key.lower().startswith("ollama/"):  # tương thích prefix cũ
        key = key[7:]
        
    # Check if exact match exists
    if key in HF_MODELS:
        return HF_MODELS[key]
        
    # Check lowercase mapping
    for k, v in HF_MODELS.items():
        if k.lower() == key.lower():
            return v
    
    # Tương thích với các định dạng ollama cũ (những người quen gõ "qwen2.5:3b")
    # Chúng ta tự động biến nó thành việc tìm kiếm gần đúng trong DB
    for k, v in HF_MODELS.items():
        if key.split(":")[0].lower() in k.lower():
            # Ưu tiên các model có parameter count trùng
            if ":" in key and key.split(":")[1].lower() in v.lower():
                return v
            
    # Always fallback to the raw key, meaning they explicitly typed a repo_id not in initial scan
    return key
