import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from synapse.helpers import DEBUG

HF_API = "https://huggingface.co/api/models"
HF_MODELS_JSON_PATH = Path(os.path.dirname(__file__)) / "data" / "hf_models.json"

MOE_CONFIGS = {
    "mixtral": {"num_experts": 8, "active_experts": 2},
    "deepseek_v2": {"num_experts": 64, "active_experts": 6},
    "deepseek_v3": {"num_experts": 256, "active_experts": 8},
    "qwen3_moe": {"num_experts": 128, "active_experts": 8},
    "llama4": {"num_experts": 16, "active_experts": 1},
}

QUANT_BPP = {
    "F32":    4.0,
    "F16":    2.0,
    "BF16":   2.0,
    "Q8_0":   1.0,
    "Q6_K":   0.75,
    "Q5_K_M": 0.625,
    "Q4_K_M": 0.5,
    "Q4_0":   0.5,
    "Q3_K_M": 0.4375,
    "Q2_K":   0.3125,
    "AWQ-4bit": 0.5,
    "GPTQ-Int4": 0.5,
}

RUNTIME_OVERHEAD = 1.2

def _auth_headers() -> dict:
    headers = {"User-Agent": "synapse-discovery/1.0"}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def fetch_model_info(repo_id: str) -> dict | None:
    url = f"{HF_API}/{repo_id}"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if DEBUG >= 1: print(f"[Discovery] HTTP {e.code} for {repo_id}")
    except Exception as e:
        if DEBUG >= 1: print(f"[Discovery] Error fetching {repo_id}: {e}")
    return None

def fetch_config_json(repo_id: str) -> dict | None:
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None

def format_param_count(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        val = total_params / 1_000_000_000
        return f"{val:.1f}B" if val != int(val) else f"{int(val)}B"
    elif total_params >= 1_000_000:
        val = total_params / 1_000_000
        return f"{val:.0f}M"
    else:
        return f"{total_params / 1_000:.0f}K"

def estimate_ram(total_params: int, quant: str) -> tuple[float, float]:
    bpp = QUANT_BPP.get(quant, 0.5)
    model_size_gb = (total_params * bpp) / (1024**3)
    min_ram_gb = model_size_gb * RUNTIME_OVERHEAD
    rec_ram_gb = model_size_gb * 2.0
    return round(max(min_ram_gb, 1.0), 1), round(max(rec_ram_gb, 2.0), 1)

def extract_provider(repo_id: str) -> str:
    org = repo_id.split("/")[0].lower() if "/" in repo_id else repo_id
    mapping = {
        "meta-llama": "Meta", "mistralai": "Mistral AI", "qwen": "Alibaba",
        "microsoft": "Microsoft", "google": "Google", "deepseek-ai": "DeepSeek"
    }
    return mapping.get(org, org.title())

def infer_context_length(config: dict | None) -> int:
    if not config: return 4096
    keys = ["max_position_embeddings", "max_sequence_length", "sliding_window"]
    for key in keys:
        if key in config and isinstance(config[key], int) and config[key] > 0:
            return config[key]
    if "text_config" in config and isinstance(config["text_config"], dict):
        tc = config["text_config"]
        for key in keys:
            if key in tc and isinstance(tc[key], int) and tc[key] > 0:
                return tc[key]
    return 4096

def detect_moe(repo_id: str, config: dict | None, architecture: str, total_params: int) -> dict:
    result = {"is_moe": False, "num_experts": None, "active_experts": None, "active_parameters": None}
    num_experts = config.get("num_local_experts") or config.get("num_experts") if config else None
    active_experts = config.get("num_experts_per_tok") or config.get("top_k_experts") if config else None
    
    if architecture in MOE_CONFIGS:
        moe = MOE_CONFIGS[architecture]
        num_experts = num_experts or moe["num_experts"]
        active_experts = active_experts or moe["active_experts"]
        
    if num_experts and active_experts:
        result["is_moe"] = True
        result["num_experts"] = num_experts
        result["active_experts"] = active_experts
        shared_fraction = 0.05
        shared = int(total_params * shared_fraction)
        expert_pool = total_params - shared
        per_expert = expert_pool // num_experts
        result["active_parameters"] = shared + active_experts * per_expert
    return result

def infer_use_case(repo_id: str, pipeline_tag: str | None) -> str:
    rid = repo_id.lower()
    if "instruct" in rid or "chat" in rid: return "Instruction following, chat"
    if "coder" in rid: return "Code generation and completion"
    if "r1" in rid or "reason" in rid: return "Advanced reasoning, chain-of-thought"
    return "General purpose"

def scrape_model(repo_id: str) -> dict | None:
    if DEBUG >= 1: print(f"[Discovery] Inspecting model: {repo_id}")
    info = fetch_model_info(repo_id)
    if not info: return None

    safetensors = info.get("safetensors", {})
    total_params = safetensors.get("total")
    if not total_params:
        params_by_dtype = safetensors.get("parameters", {})
        if params_by_dtype:
            total_params = max(params_by_dtype.values())
    
    if not total_params:
        if DEBUG >= 1: print(f"[Discovery] Could not find parameter count for {repo_id}")
        # Default fallback to 7B if absolutely nothing is found
        total_params = 7_000_000_000

    config = info.get("config", {})
    full_config = fetch_config_json(repo_id)
    
    # Safely get structural layer info
    num_hidden_layers = 32
    if full_config:
        num_hidden_layers = full_config.get("num_hidden_layers", full_config.get("n_layer", full_config.get("num_layers", 32)))
    elif config:
        num_hidden_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", 32)))

    quant = "Q4_K_M"  # Target quantization for estimation
    min_r, rec_r = estimate_ram(total_params, quant)
    architecture = (full_config or {}).get("model_type", "unknown")
    moe_info = detect_moe(repo_id, full_config, architecture, total_params)
    
    result = {
        "name": repo_id,
        "provider": extract_provider(repo_id),
        "parameter_count": format_param_count(total_params),
        "parameters_raw": total_params,
        "min_ram_gb": min_r,
        "recommended_ram_gb": rec_r,
        "min_vram_gb": min_r,
        "quantization": quant,
        "context_length": infer_context_length(full_config),
        "use_case": infer_use_case(repo_id, info.get("pipeline_tag")),
        "pipeline_tag": info.get("pipeline_tag") or "text-generation",
        "architecture": architecture,
        "hf_downloads": info.get("downloads", 0),
        "hf_likes": info.get("likes", 0),
        "release_date": (info.get("createdAt") or "")[:10] or None,
        "num_hidden_layers": num_hidden_layers,
        "_discovered": True
    }
    
    if moe_info["is_moe"]:
        result.update(moe_info)
        
    return result

def discover_and_save_model(repo_id: str) -> dict | None:
    """Scrapes HF, appends to hf_models.json, returns dict"""
    model_data = scrape_model(repo_id)
    if not model_data:
        return None
        
    try:
        # Load existing
        existing_data = []
        if HF_MODELS_JSON_PATH.exists():
            with open(HF_MODELS_JSON_PATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        
        # Check if already exists
        updated = False
        for i, entry in enumerate(existing_data):
            if entry.get("name", "").lower() == repo_id.lower():
                existing_data[i].update(model_data)
                updated = True
                break
                
        if not updated:
            existing_data.append(model_data)
        
        # Save back
        HF_MODELS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HF_MODELS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
        if DEBUG >= 1: print(f"[Discovery] Successfully locally cached {repo_id}")
        return model_data
        
    except Exception as e:
        if DEBUG >= 1: print(f"[Discovery] Failed to write cache for {repo_id}: {e}")
        return model_data

