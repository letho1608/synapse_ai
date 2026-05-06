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
VISION_MODELS = set()
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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

                if "vision" in entry.get("capabilities", []) or "vision" in entry.get("pipeline_tag", "").lower():
                    VISION_MODELS.add(repo_id)
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

    for k, v in HF_MODELS.items():
        if key.split(":")[0].lower() in k.lower():
            if ":" in key and key.split(":")[1].lower() in v.lower():
                return v

    return DEFAULT_MODEL