import traceback
from os import PathLike
import os
from typing import Union
from transformers import AutoTokenizer
try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None
    if DEBUG >= 1:
        print("[WARNING] AutoProcessor could not be imported from transformers. Vision models may not work.")

import numpy as np
from synapse.helpers import DEBUG
from pathlib import Path
from synapse.loading import get_models_dir

def ensure_downloads_dir():
    """Ensure downloads directory exists"""
    return get_models_dir()


async def resolve_tokenizer(repo_id: Union[str, PathLike]):
  if repo_id is None:
    raise ValueError("repo_id cannot be None")
  local_path = ensure_downloads_dir() / str(repo_id).replace("/", "--")
  if DEBUG >= 2: print(f"Checking if local path exists to load tokenizer from local {local_path=}")
  try:
    if local_path and os.path.exists(local_path):
      if DEBUG >= 2: print(f"Resolving tokenizer for {repo_id=} from {local_path=}")
      return await _resolve_tokenizer(local_path)
  except:
    if DEBUG >= 5: print(f"Local check for {local_path=} failed. Resolving tokenizer for {repo_id=} normally...")
    if DEBUG >= 5: traceback.print_exc()
  return await _resolve_tokenizer(repo_id)


async def _resolve_tokenizer(repo_id_or_local_path: Union[str, PathLike]):
  # Xử lý đặc biệt cho các model Qwen sử dụng Sliding Window Attention
  is_qwen = "qwen" in str(repo_id_or_local_path).lower()
  
  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {repo_id_or_local_path}")
    
    # Cấu hình cho tokenizer với các tham số phù hợp với Sliding Window Attention
    tokenizer_kwargs = {
        "trust_remote_code": True,
        "use_fast": False
    }
    
    # Thêm xử lý đặc biệt cho model Qwen
    if is_qwen:
        # Qwen models có thể cần cấu hình attention đặc biệt
        tokenizer_kwargs["use_fast"] = True
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id_or_local_path, **tokenizer_kwargs)
    return tokenizer
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {repo_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  raise ValueError(f"Unsupported model: {repo_id_or_local_path}")
