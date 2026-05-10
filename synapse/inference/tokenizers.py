import asyncio
import traceback
from os import PathLike
import os
from typing import Union
from transformers import AutoTokenizer, AutoProcessor
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
  is_qwen = "qwen" in str(repo_id_or_local_path).lower()

  tokenizer_kwargs = {
    "trust_remote_code": True,
    "use_fast": False
  }
  if is_qwen:
    tokenizer_kwargs["use_fast"] = True

  loop = asyncio.get_event_loop()
  def _load():
    return AutoTokenizer.from_pretrained(repo_id_or_local_path, **tokenizer_kwargs)

  try:
    tokenizer = await loop.run_in_executor(None, _load)
    return tokenizer
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {repo_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  raise ValueError(f"Unsupported model: {repo_id_or_local_path}")
