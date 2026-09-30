"""
Danh sách model dùng cho: download (HF), hiển thị web UI, và inference.
Hoàn toàn load từ hf_models.json - không còn hardcoded.
"""

import json
import os
from pathlib import Path
from typing import Dict

HF_MODELS: Dict[str, str] = {}
HF_MODEL_PARAMS: Dict[str, str] = {}
HF_MODEL_LAYERS: Dict[str, int] = {}
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

def _load_dynamic_config():
    db_path = Path(os.path.dirname(__file__)) / "config" / "hf_models.json"
    try:
        if db_path.exists():
            with open(db_path, "r", encoding="utf-8") as f:
                db_data = json.load(f)

            for entry in db_data:
                repo_id = entry.get("name")
                if not repo_id:
                    continue

                HF_MODELS[repo_id] = repo_id

                if entry.get("parameter_count"):
                    HF_MODEL_PARAMS[repo_id] = entry["parameter_count"]

                if entry.get("num_hidden_layers"):
                    HF_MODEL_LAYERS[repo_id] = entry["num_hidden_layers"]
    except Exception:
        pass

_load_dynamic_config()


def resolve_hf_id(model_name: str) -> str:
    """Tên model -> Hugging Face repo_id (dùng download + UI)."""
    if not model_name:
        return DEFAULT_MODEL

    key = model_name.strip()
    if key.lower().startswith("ollama/"):
        key = key[7:]

    if key in HF_MODELS:
        return HF_MODELS[key]

    for k, v in HF_MODELS.items():
        if k.lower() == key.lower():
            return v
            
    norm_key = key.lower().replace("-", "").replace("_", "")
    norm_parts = norm_key.split(":")
    
    best_match = None
    for k, v in HF_MODELS.items():
        norm_k = k.lower().replace("-", "").replace("_", "")
        
        if norm_parts[0] in norm_k:
            if len(norm_parts) > 1 and norm_parts[1] in v.lower():
                if "coder" not in norm_key and "coder" in v.lower():
                    if best_match is None:
                        best_match = v
                    continue
                if best_match is None or ("instruct" in v.lower() and "instruct" not in best_match.lower()):
                    best_match = v
                if "instruct" in v.lower() and "coder" not in v.lower():
                    return v

    return best_match if best_match else model_name