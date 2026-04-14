"""
Danh sách model dùng cho: download (HF), hiển thị web UI, và inference.
Một nguồn duy nhất cho download và web UI (PyTorch/HF).
"""

from typing import Dict

# 22 model: tên hiển thị / dùng trong API -> Hugging Face repo_id (download + web UI)
HF_MODELS: Dict[str, str] = {
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi3:medium": "microsoft/Phi-3-medium-4k-instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    "codellama:7b": "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek-coder:6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "neural-chat:7b": "Intel/neural-chat-7b-v3-3",
    "openhermes2.5:7b": "Teknium/OpenHermes-2.5-Mistral-7B",
    "solar:10.7b": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "smollm2:360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "tinyllama:1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "starcoder2:15b": "bigcode/starcoder2-15b",
    "nous-hermes2:10b": "NousResearch/Hermes-2-Pro-Mistral-7B",
    "dcaybe1.0:8.7m": "dcaybe/dcaybe-1.0-8.7m",
}

# Tham số (hiển thị web: 0.5B, 1.5B, 7B...)
HF_MODEL_PARAMS: Dict[str, str] = {
    "qwen2.5:0.5b": "0.5B", "qwen2.5:1.5b": "1.5B", "qwen2.5:3b": "3B", "qwen2.5:7b": "7B", "qwen2.5:14b": "14B",
    "llama3.2:1b": "1B", "llama3.2:3b": "3B", "llama3.1:8b": "8B",
    "phi3:mini": "3.8B", "phi3:medium": "14B", "mistral:7b": "7B", "gemma2:2b": "2B", "gemma2:9b": "9B",
    "codellama:7b": "7B", "deepseek-coder:6.7b": "6.7B", "neural-chat:7b": "7B", "openhermes2.5:7b": "7B",
    "solar:10.7b": "10.7B", "smollm2:360m": "360M", "tinyllama:1.1b": "1.1B", "starcoder2:15b": "15B",
    "nous-hermes2:10b": "7B",
    "dcaybe1.0:8.7m": "8.7M",
}

# Số layer (chỉ dùng cho inference / sharding)
HF_MODEL_LAYERS: Dict[str, int] = {
    "qwen2.5:0.5b": 14, "qwen2.5:1.5b": 28, "qwen2.5:3b": 28, "qwen2.5:7b": 28, "qwen2.5:14b": 40,
    "llama3.2:1b": 16, "llama3.2:3b": 28, "llama3.1:8b": 32,
    "phi3:mini": 32, "phi3:medium": 40, "mistral:7b": 32, "gemma2:2b": 26, "gemma2:9b": 42,
    "codellama:7b": 32, "deepseek-coder:6.7b": 28, "neural-chat:7b": 32, "openhermes2.5:7b": 32,
    "solar:10.7b": 36, "smollm2:360m": 12, "tinyllama:1.1b": 22, "starcoder2:15b": 40,
    "nous-hermes2:10b": 32,
    "dcaybe1.0:8.7m": 10,
}

# Các model chuyên biệt Computer Vision (đầu vào hình ảnh) thay vì LLM (đầu vào text).
VISION_MODELS = {
    "dcaybe1.0:8.7m"
}


def resolve_hf_id(model_name: str) -> str:
    """Tên model -> Hugging Face repo_id (dùng download + UI)."""
    key = (model_name or "").strip().lower()
    if key.startswith("ollama/"):  # tương thích prefix cũ
        key = key[7:]
    return HF_MODELS.get(key) or HF_MODELS.get(
        key.split(":")[0] + ":3b" if ":" in key else key
    ) or "Qwen/Qwen2.5-1.5B-Instruct"
