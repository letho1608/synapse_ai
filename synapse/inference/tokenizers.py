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
  print(f"DEBUG_CRASH: _resolve_tokenizer called for {repo_id_or_local_path}", flush=True)
  # try:
  #   if DEBUG >= 4: print(f"Trying AutoProcessor for {repo_id_or_local_path}")
  #   processor = AutoProcessor.from_pretrained(repo_id_or_local_path, use_fast=True if "Mistral-Large" in f"{repo_id_or_local_path}" else False, trust_remote_code=True)
  #   if not hasattr(processor, 'eos_token_id'):
  #     processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
  #   if not hasattr(processor, 'encode'):
  #     processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
  #   if not hasattr(processor, 'decode'):
  #     processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
  #   return processor
  # except Exception as e:
  #   if DEBUG >= 4: print(f"Failed to load processor for {repo_id_or_local_path}. Error: {e}")
  #   if DEBUG >= 4: print(traceback.format_exc())

  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {repo_id_or_local_path}")
    print(f"DEBUG_CRASH: Calling AutoTokenizer.from_pretrained for {repo_id_or_local_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id_or_local_path, trust_remote_code=True, use_fast=False)
    print(f"DEBUG_CRASH: AutoTokenizer.from_pretrained success", flush=True)
    return tokenizer
  except Exception as e:
    print(f"DEBUG_CRASH: AutoTokenizer.from_pretrained failed: {e}", flush=True)
    if DEBUG >= 4: print(f"Failed to load tokenizer for {repo_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  raise ValueError(f"Unsupported model: {repo_id_or_local_path}")
