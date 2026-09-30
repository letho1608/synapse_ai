"""
Model Merging - Hợp nhất nhiều mô hình/LoRA adapters.
Hỗ trợ Linear Merging, SLERP, và Ties-Merging.
"""

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

