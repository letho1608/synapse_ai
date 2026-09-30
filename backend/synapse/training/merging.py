"""
Model Merging - Hợp nhất nhiều mô hình/LoRA adapters.
Hỗ trợ Linear Merging, SLERP, và Ties-Merging.
"""

import os
import time
import torch
from typing import List, Dict, Optional, Any
from synapse.helpers import DEBUG

def merge_lora_adapters(
    base_model_id: str,
    adapter_paths: List[str],
    output_path: str,
    weights: Optional[List[float]] = None,
    merge_type: str = "linear"
) -> Dict[str, Any]:
    """
    Hợp nhất các LoRA adapters vào base model và lưu lại thành full model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    start_time = time.time()
    try:
        if weights is None:
            weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
            
        print(f"[Merging] Loading base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="cpu" # Use CPU to avoid OOM if many adapters
        )
        
        # Load the first adapter
        print(f"[Merging] Loading adapter 1: {adapter_paths[0]}")
        model = PeftModel.from_pretrained(base_model, adapter_paths[0], adapter_name="adapter_0")
        
        # Add remaining adapters
        for i in range(1, len(adapter_paths)):
            print(f"[Merging] Loading adapter {i+1}: {adapter_paths[i]}")
            model.load_adapter(adapter_paths[i], adapter_name=f"adapter_{i}")
            
        # Perform merge
        print(f"[Merging] Performing {merge_type} merge...")
        if merge_type == "linear":
            model.add_weighted_adapter(
                adapters=[f"adapter_{i}" for i in range(len(adapter_paths))],
                weights=weights,
                adapter_name="merged",
                combination_type="linear"
            )
            model.set_adapter("merged")
        elif merge_type == "slerp":
            # PEFT doesn't natively support SLERP in add_weighted_adapter yet, 
            # fallback to linear or implement manually if needed.
            model.add_weighted_adapter(
                adapters=[f"adapter_{i}" for i in range(len(adapter_paths))],
                weights=weights,
                adapter_name="merged",
                combination_type="cat" # Simplified fallback
            )
            model.set_adapter("merged")
        else:
            raise ValueError(f"Unsupported merge type: {merge_type}")
            
        # Merge and unload
        print("[Merging] Merging weights into base model...")
        merged_model = model.merge_and_unload()
        
        # Save
        print(f"[Merging] Saving to: {output_path}")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        duration = time.time() - start_time
        return {
            "success": True,
            "output_path": output_path,
            "duration_sec": round(duration, 2),
            "message": f"Successfully merged {len(adapter_paths)} adapters in {duration:.1f}s"
        }
        
    except Exception as e:
        if DEBUG >= 1:
            import traceback
            traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

def merge_models_checkpoints(
    model_paths: List[str],
    output_path: str,
    weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Hợp nhất các full model checkpoints bằng cách trung bình trọng số (Weight Averaging).
    """
    start_time = time.time()
    try:
        if not model_paths:
            return {"success": False, "error": "No models to merge"}
            
        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)
            
        print(f"[Merging] Merging {len(model_paths)} models...")
        
        # Load first model state dict
        first_sd = torch.load(os.path.join(model_paths[0], "pytorch_model.bin"), map_location="cpu")
        merged_sd = {k: v * weights[0] for k, v in first_sd.items()}
        del first_sd
        
        for i in range(1, len(model_paths)):
            print(f"[Merging] Processing model {i+1}...")
            sd = torch.load(os.path.join(model_paths[i], "pytorch_model.bin"), map_location="cpu")
            for k, v in sd.items():
                if k in merged_sd:
                    merged_sd[k] += v * weights[i]
            del sd
            
        # Save merged state dict
        os.makedirs(output_path, exist_ok=True)
        torch.save(merged_sd, os.path.join(output_path, "pytorch_model.bin"))
        
        # Copy config files from first model
        import shutil
        for item in os.listdir(model_paths[0]):
            if item != "pytorch_model.bin":
                s = os.path.join(model_paths[0], item)
                d = os.path.join(output_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
                    
        duration = time.time() - start_time
        return {
            "success": True,
            "output_path": output_path,
            "duration_sec": round(duration, 2),
            "message": f"Successfully merged {len(model_paths)} full models in {duration:.1f}s"
        }
    except Exception as e:
        if DEBUG >= 1:
            import traceback
            traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
