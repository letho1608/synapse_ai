import asyncio
import traceback
from os import PathLike
import os
from typing import Dict, Union
from transformers import AutoTokenizer
from synapse.helpers import DEBUG
from synapse.loading import get_models_dir

def ensure_downloads_dir():
    """Ensure downloads directory exists"""
    return get_models_dir()


# Loaded tokenizers by repo id. The API resolves the tokenizer on every chat
# request; reloading it each time (from_pretrained + a Hub round trip when
# online) cost ~2 s of time-to-first-token per request.
_TOKENIZERS: Dict[str, object] = {}
_TOKENIZER_LOCKS: Dict[tuple, asyncio.Lock] = {}  # (repo, loop): a lock is loop-bound


async def resolve_tokenizer(repo_id: Union[str, PathLike]):
  if repo_id is None:
    raise ValueError("repo_id cannot be None")
  key = str(repo_id)
  cached = _TOKENIZERS.get(key)
  if cached is not None:
    return cached
  lock = _TOKENIZER_LOCKS.setdefault((key, id(asyncio.get_running_loop())), asyncio.Lock())
  async with lock:
    cached = _TOKENIZERS.get(key)
    if cached is None:
      cached = _TOKENIZERS[key] = await _resolve_tokenizer_uncached(repo_id)
  return cached


async def _resolve_tokenizer_uncached(repo_id: Union[str, PathLike]):
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
