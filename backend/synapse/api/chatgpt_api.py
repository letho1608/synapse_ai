import re
import subprocess
import uuid
import time
import asyncio
import json
import os
import platform
import psutil
import numpy as np
from pathlib import Path
from typing import List, Literal, Union, Dict, Optional, Set, Any
from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError
from aiohttp.web_exceptions import HTTPMethodNotAllowed, HTTPNotFound
import aiohttp_cors
import traceback
import signal
from synapse import DEBUG, VERSION
from synapse.terminal_log import get_lines as get_terminal_log_lines
from synapse.helpers import PrefixDict, shutdown, get_all_ip_addresses_and_interfaces
from synapse.networking.tailscale.tailscale_helpers import get_self_tailscale_info
from synapse.orchestration import Node
from synapse.model_list import HF_MODELS, HF_MODEL_PARAMS, resolve_hf_id  # Danh sách + metadata cho web UI
from typing import Callable
from collections import defaultdict
from synapse.training.finetune_lora import load_raw_data_async, format_sample_to_text
from synapse.training.preprocessing import preprocess_async
from synapse.loading import get_checkpoints_dir
from synapse.models import build_base_shard, get_repo
from synapse.inference.tokenizers import resolve_tokenizer


# Built-in liveness page served at GET / (the project is CLI-first; there is
# no separate web frontend).
_DEFAULT_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Synapse — LACP Control Plane</title>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3; }
  header { padding: 16px 24px; background: #161b22; border-bottom: 1px solid #30363d; }
  header h1 { margin: 0; font-size: 18px; }
  main { padding: 24px; max-width: 800px; margin: 0 auto; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-top: 16px; }
  .ok { color: #3fb950; } .err { color: #f85149; }
  pre { white-space: pre-wrap; font-size: 12px; color: #8b949e; }
</style>
</head>
<body>
<header><h1>Synapse <span style="color:#58a6ff;font-size:12px;border:1px solid #1f6feb;padding:2px 8px;border-radius:999px">LACP</span></h1></header>
<main>
  <div class="card" id="status"><h2>Status</h2><p class="ok">API reachable</p></div>
  <div class="card"><h2>Cluster topology</h2><pre id="topo">loading…</pre></div>
</main>
<script>
async function refresh() {
  try {
    const info = await (await fetch('/v1/system/info')).json();
    document.getElementById('status').innerHTML = '<h2>Status</h2><p class="ok">' +
      (info.distributed_status || 'single-node') + ' · v' + (info.version || '?') + '</p>';
  } catch (e) {
    document.getElementById('status').innerHTML = '<h2>Status</h2><p class="err">API unavailable</p>';
  }
  try {
    const topo = await (await fetch('/v1/topology')).json();
    document.getElementById('topo').textContent = JSON.stringify(topo, null, 2);
  } catch (e) {
    document.getElementById('topo').textContent = 'Topology unavailable';
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class Message:
  def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]):
    self.role = role
    self.content = content

  def to_dict(self):
    return {"role": self.role, "content": self.content}


class ChatCompletionRequest:
  def __init__(self, model: str, messages: List[Message], temperature: float):
    self.model = model
    self.messages = messages
    self.temperature = temperature

  def to_dict(self):
    return {"model": self.model, "messages": [message.to_dict() for message in self.messages], "temperature": self.temperature}


def _clean_completion_content(text: str) -> str:
  """Cắt bớt nội dung khi model lặp rác (vd. API/API/API, Xcode/API). Không thay thế nội dung thật của model."""
  if not text or not isinstance(text, str):
    return text
  bad_patterns = [
    r"/API(/API)+",           # /API/API/API...
    r"API(/API)+",            # API/API/API...
    r"Xcode/API",             # Xcode/API...
    r"As the code/API",       # As the code/API...
    r"I can code/API",        # I can code/API...
  ]
  out = text.strip()
  for pat in bad_patterns:
    m = re.search(pat, out, re.IGNORECASE)
    if m:
      out = out[: m.start()].rstrip()
      break
  if re.search(r"API/API", out, re.IGNORECASE):
    idx = out.lower().find("api/api")
    if idx >= 0:
      out = out[: idx].rstrip()
  return out


class StreamDetokenizer:
  """Incremental detokenization for SSE streaming.

  Decoding each token on its own and cleaning (strip) every piece dropped
  all spaces and newlines from streamed answers ("Herearefivefruits:1.Banana"
  instead of "Here are five fruits:\n\n1. Banana"). Instead: decode the whole
  answer so far and emit only the new suffix; hold back while the text ends
  in an incomplete UTF-8 sequence (U+FFFD); never strip trailing whitespace
  mid-stream."""

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.ids: List[int] = []
    self.sent = ""

  def _decode(self) -> str:
    try:
      return self.tokenizer.decode(self.ids, skip_special_tokens=True)
    except TypeError:
      return self.tokenizer.decode(self.ids)

  def push(self, ids) -> str:
    self.ids.extend(int(t) for t in ids)
    full = self._decode()
    if full.endswith("\ufffd"):
      return ""
    text = full.lstrip()
    cleaned = _clean_completion_content(text)
    if cleaned != text.strip():  # a garbage pattern cut the answer here
      text = cleaned
    if not text.startswith(self.sent):
      return ""  # normalization rewrote earlier text; never resend it
    delta = text[len(self.sent):]
    self.sent = text
    return delta


def stream_chunk(chat_request, request_id: str, content: str) -> dict:
  """One chat.completion.chunk carrying `content` verbatim."""
  return {
    "id": f"chatcmpl-{request_id}",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": chat_request.model,
    "system_fingerprint": f"synapse_{VERSION}",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                 "logprobs": None, "finish_reason": None}],
  }


def generate_completion(
  chat_request: ChatCompletionRequest,
  tokenizer,
  prompt: str,
  request_id: str,
  tokens: List[int],
  stream: bool,
  finish_reason: Union[Literal["length", "stop"], None],
  object_type: Literal["chat.completion", "text_completion"],
) -> dict:
  try:
    raw_content = tokenizer.decode(tokens, skip_special_tokens=True)
  except TypeError:
    raw_content = tokenizer.decode(tokens)
  content = _clean_completion_content(raw_content)
  completion = {
    "id": f"chatcmpl-{request_id}",
    "object": object_type,
    "created": int(time.time()),
    "model": chat_request.model,
    "system_fingerprint": f"synapse_{VERSION}",
    "choices": [{
      "index": 0,
      "message": {"role": "assistant", "content": content},
      "logprobs": None,
      "finish_reason": finish_reason,
    }],
  }

  if not stream:
    completion["usage"] = {
      "prompt_tokens": len(tokenizer.encode(prompt)),
      "completion_tokens": len(tokens),
      "total_tokens": len(tokenizer.encode(prompt)) + len(tokens),
    }

  choice = completion["choices"][0]
  if object_type.startswith("chat.completion"):
    key_name = "delta" if stream else "message"
    choice[key_name] = {"role": "assistant", "content": content}
  elif object_type == "text_completion":
    choice["text"] = content
  else:
    ValueError(f"Unsupported response type: {object_type}")

  return completion


def _sampling_state(data) -> Optional[dict]:
  """Per-request sampling knobs carried in inference_state (the node samples
  with them; before this the request's temperature was ignored)."""
  t = data.get("temperature") if isinstance(data, dict) else None
  try:
    return {"temperature": max(0.0, float(t))} if t is not None else None
  except (TypeError, ValueError):
    return None


def _concat_prompt(messages: List[Message]) -> str:
  """Plain concatenation for tokenizers without a usable chat template (GPT-2)."""
  return "\n".join(m.content or "" for m in messages if m.role in ("user", "assistant", "system"))


def build_prompt(tokenizer, messages: List[Message]):
  if getattr(tokenizer, "chat_template", None) is None:
    return _concat_prompt(messages)
  conversation = [m.to_dict() for m in messages]
  try:
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
  except (ValueError, AttributeError):
    return _concat_prompt(messages)
  if DEBUG >= 3: print(f"!!! Prompt: {prompt}")
  return prompt


def parse_message(data: dict):
  if "role" not in data or "content" not in data:
    raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
  return Message(data["role"], data["content"])


def parse_chat_request(data: dict, default_model: str):
  return ChatCompletionRequest(
    data.get("model", default_model),
    [parse_message(msg) for msg in data["messages"]],
    data.get("temperature", 0.0),
  )


class PromptSession:
  def __init__(self, request_id: str, timestamp: int, prompt: str):
    self.request_id = request_id
    self.timestamp = timestamp
    self.prompt = prompt


# Routes anyone on the network may call without SYNAPSE_API_KEY: chat and
# read-only cluster status. Everything else (quit, settings, model pull /
# delete / merge, datasets, training, logs) changes or exposes the machine.
PUBLIC_ROUTES = {
  ("GET", "/"), ("GET", "/healthcheck"), ("GET", "/models"), ("GET", "/v1/models"),
  ("POST", "/chat/completions"), ("POST", "/v1/chat/completions"),
  ("POST", "/chat/token/encode"), ("POST", "/v1/chat/token/encode"),
  ("GET", "/topology"), ("GET", "/v1/topology"), ("GET", "/v1/distributed/status"),
}
# Reachable without a key even when one is set (liveness probes).
OPEN_ROUTES = {("GET", "/"), ("GET", "/healthcheck")}


def _is_loopback(remote: Optional[str]) -> bool:
  import ipaddress
  try:
    return ipaddress.ip_address((remote or "").split("%")[0]).is_loopback
  except ValueError:
    return False


def check_access(method: str, path: str, remote: Optional[str], auth_header: str,
                 query_token: str, api_key: Optional[str]):
  """None if the request may proceed, else (status, message).

  With SYNAPSE_API_KEY set, every route but the liveness ones needs
  `Authorization: Bearer <key>` (the websocket may pass `?token=`). Without a
  key, chat and status stay open to the LAN, while admin routes answer only
  to this machine, so a neighbour cannot /quit the node or delete models."""
  import hmac
  if method == "OPTIONS":
    return None
  route = (method, path)
  if api_key:
    if route in OPEN_ROUTES:
      return None
    token = auth_header[7:] if auth_header.startswith("Bearer ") else query_token
    if not token:
      return 401, "Unauthorized. Missing or invalid Authorization Bearer token."
    if not hmac.compare_digest(token.encode(), api_key.encode()):
      return 403, "Forbidden. Invalid API Key."
    return None
  if route in PUBLIC_ROUTES or _is_loopback(remote):
    return None
  return 403, "Forbidden. Admin routes are local-only; set SYNAPSE_API_KEY to allow remote access."


class ChatGPTAPI:
  def __init__(
    self,
    node: Node,
    inference_engine_classname: str,
    response_timeout: int = 90,
    on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None,
    default_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    chatgpt_api_port: int = 52415,
  ):
    self.node = node
    self.inference_engine_classname = inference_engine_classname
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    
    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        denied = check_access(request.method, request.path, request.remote,
                              request.headers.get("Authorization", ""),
                              request.query.get("token", ""), os.getenv("SYNAPSE_API_KEY"))
        if denied:
            status, msg = denied
            return web.json_response({"error": msg}, status=status)
        return await handler(request)

    self.app = web.Application(client_max_size=10*1024*1024, middlewares=[auth_middleware])  # 10MB limit for text payloads
    self.prompts: PrefixDict[str, PromptSession] = PrefixDict()
    self.prev_token_lens: Dict[str, int] = {}
    self.stream_tasks: Dict[str, asyncio.Task] = {}
    self.default_model = default_model or "HuggingFaceTB/SmolLM2-135M-Instruct"
    self.discovery_module = "tailscale" # Default
    self.token_queues = defaultdict(asyncio.Queue)
    self.activity_logs = [] # In-memory activity logs
    self.log_activity("System Started", "-", "success")
    self.chatgpt_api_port = chatgpt_api_port
    self._training_job: Optional[Dict] = None  # job_id, model, dataset, epochs, batch_size, status, progress, error
    _base_dir = Path(__file__).resolve().parent.parent
    self._data_dir = _base_dir / "data"  # Chỉ dataset (json/jsonl)
    self._settings_path = _base_dir / "config" / "settings.json"  # Cấu hình (tách khỏi data)
    self._linked_datasets_path = _base_dir / "config" / "linked_datasets.json"
    self.linked_datasets = {} # name -> path
    self._load_settings()
    self._load_linked_datasets()

    # Get the callback system and register our handler
    self.token_callback = node.on_token.register("chatgpt-api-token-handler")
    def safe_handle_tokens(_request_id, tokens, is_finished):
        """Safely handle tokens with error handling"""
        async def _handle():
            try:
                await self.handle_tokens(_request_id, tokens, is_finished)
            except Exception as e:
                if DEBUG >= 1:
                    print(f"Error in handle_tokens for {_request_id}: {e}")
                import traceback
                traceback.print_exc()
        # Create task and ensure it runs - THIS IS THE FIX
        task = asyncio.create_task(_handle())
        # Store task to prevent garbage collection
        if not hasattr(self, '_token_tasks'):
            self._token_tasks = set()
        self._token_tasks.add(task)
        task.add_done_callback(self._token_tasks.discard)
    
    # Register the callback to trigger safe_handle_tokens when tokens arrive
    self.token_callback.on_next(lambda rid, tok, fin: safe_handle_tokens(rid, tok, fin))
    self.system_prompt = system_prompt

    cors = aiohttp_cors.setup(self.app)
    cors_options = aiohttp_cors.ResourceOptions(
      allow_credentials=True,
      expose_headers="*",
      allow_headers="*",
      allow_methods="*",
    )
    cors.add(self.app.router.add_get("/models", self.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models", self.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_get("/healthcheck", self.handle_healthcheck), {"*": cors_options})
    cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_get("/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/distributed/status", self.handle_get_distributed_status), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/training/distributed/start", self.handle_post_training_distributed), {"*": cors_options})
    
    # New endpoints for Dashboard
    cors.add(self.app.router.add_get("/v1/system/stats", self.handle_get_system_stats), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/ws/metrics", self.handle_ws_metrics), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/system/info", self.handle_get_system_info), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models/list", self.handle_get_models_list), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models/search", self.handle_get_models_search), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models/compatibility", self.handle_get_models_compatibility), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/network/nodes", self.handle_get_network_nodes), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets", self.handle_get_datasets), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/upload", self.handle_post_datasets_upload), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/delete", self.handle_post_datasets_delete), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets/preview", self.handle_get_datasets_preview), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets/download", self.handle_get_datasets_download), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/create", self.handle_post_datasets_create), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/rename", self.handle_post_datasets_rename), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets/validate", self.handle_get_datasets_validate), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/link", self.handle_post_datasets_link), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/update", self.handle_post_datasets_update), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/activity", self.handle_get_activity_logs), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/logs/terminal", self.handle_get_terminal_logs), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/training/start", self.handle_post_training), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/training/status", self.handle_get_training_status), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models/status", self.handle_get_models_status), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/models/pull", self.handle_post_models_pull), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/models/delete", self.handle_post_models_delete), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/models/merge", self.handle_post_models_merge), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models/checkpoints", self.handle_get_model_checkpoints), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/settings", self.handle_get_settings), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/settings", self.handle_post_settings), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/cluster/resources", self.handle_get_cluster_resources), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/download/progress", self.handle_get_download_progress), {"*": cors_options})
    # CLI-first (PLAN Part E): no web frontend. GET / is a built-in liveness
    # page (status + topology), registered unconditionally.
    self.app.router.add_get("/", self.handle_root)
      
    self.app.middlewares.append(self.timeout_middleware)
    self.app.middlewares.append(self.log_request)

  def _apply_optimization_settings(self, opts: Dict[str, Any]) -> Optional[str]:
    """Validate and apply optimization knobs; returns an error or None.

    - lookahead_window (0..16): n-gram draft length, node-wide, immediate.
    - load_quantization (auto|none|int8|int4): weight quant for layers the
      engine loads from now on (already-loaded layers keep their format).
    - transport_quantization (int8|none): hidden states on the wire, immediate.
    - lacp_priority (throughput|latency): measured-placement objective,
      used at the next placement."""
    q_alias = {"8bit": "int8", "4bit": "int4", "off": "none"}
    if "lookahead_window" in opts:
      try:
        lw = int(opts["lookahead_window"])
      except (TypeError, ValueError):
        return "lookahead_window must be an integer"
      if not 0 <= lw <= 16:
        return "lookahead_window must be 0..16"
    if "load_quantization" in opts:
      lq = q_alias.get(str(opts["load_quantization"]).lower(), str(opts["load_quantization"]).lower())
      if lq not in ("auto", "none", "int8", "int4"):
        return "load_quantization must be auto|none|int8|int4"
    if "transport_quantization" in opts:
      tq = str(opts["transport_quantization"]).lower()
      if tq not in ("int8", "none", "fp32", "off"):
        return "transport_quantization must be int8|none"
    if "lacp_priority" in opts:
      lp = str(opts["lacp_priority"]).lower()
      if lp not in ("throughput", "latency"):
        return "lacp_priority must be throughput|latency"
    if "lookahead_window" in opts:
      self.node.lookahead_window = lw
    if "load_quantization" in opts:
      os.environ["SYNAPSE_WEIGHT_QUANT"] = lq
    if "transport_quantization" in opts:
      os.environ["SYNAPSE_TRANSPORT_INT8"] = "1" if tq == "int8" else "0"
    if "lacp_priority" in opts:
      self.node.partition_objective = lp
    return None

  def _load_settings(self) -> None:
    try:
      if self._settings_path.is_file():
        with open(self._settings_path, "r", encoding="utf-8") as f:
          data = json.load(f)
          if "default_model" in data and isinstance(data["default_model"], str):
            self.default_model = data["default_model"].strip()
          if "discovery_module" in data and isinstance(data["discovery_module"], str):
            self.discovery_module = data["discovery_module"].strip()
          # C-W12: system_prompt round-trips too (server knobs like ports
          # belong to the launcher argv, not this file — documented).
          if "system_prompt" in data and isinstance(data["system_prompt"], str):
            self.system_prompt = data["system_prompt"]
          opts = {k: data[k] for k in ("load_quantization", "lookahead_window",
                                       "transport_quantization", "lacp_priority") if k in data}
          if opts and getattr(self, "node", None) is not None:
            err = self._apply_optimization_settings(opts)
            if err and DEBUG >= 1:
              print(f"settings.json: {err} (ignored)")
    except Exception:
      pass

  def _load_linked_datasets(self) -> None:
    try:
      if self._linked_datasets_path.is_file():
        with open(self._linked_datasets_path, "r", encoding="utf-8") as f:
          self.linked_datasets = json.load(f)
    except Exception:
      self.linked_datasets = {}

  def _save_linked_datasets(self) -> None:
    try:
      self._linked_datasets_path.parent.mkdir(parents=True, exist_ok=True)
      with open(self._linked_datasets_path, "w", encoding="utf-8") as f:
        json.dump(self.linked_datasets, f, ensure_ascii=False, indent=2)
    except Exception as e:
      if DEBUG >= 1: print(f"_save_linked_datasets error: {e}")

  def _save_settings(self) -> None:
    """Lưu settings hiện tại ra synapse/config/settings.json."""
    try:
      self._settings_path.parent.mkdir(parents=True, exist_ok=True)
      data = {
        "default_model": self.default_model,
        "discovery_module": getattr(self, "discovery_module", "tailscale"),
        "system_prompt": getattr(self, "system_prompt", ""),
        "load_quantization": os.getenv("SYNAPSE_WEIGHT_QUANT", "auto"),
        "lookahead_window": getattr(getattr(self, "node", None), "lookahead_window", 3),
        "transport_quantization": "int8" if os.getenv("SYNAPSE_TRANSPORT_INT8", "1") == "1" else "none",
        "lacp_priority": getattr(getattr(self, "node", None), "partition_objective", None) or
          os.getenv("SYNAPSE_PARTITION_OBJECTIVE", "throughput"),
      }
      with open(self._settings_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
      if DEBUG >= 1:
        print(f"_save_settings error: {e}")


  async def handle_quit(self, request):
    if DEBUG >= 1: print("Received quit signal")
    response = web.json_response({"detail": "Quit signal received"}, status=200)
    await response.prepare(request)
    try:
      await response.write_eof()
    except (ClientConnectionResetError, ConnectionResetError, OSError):
      pass  # Client đã đóng connection
    await shutdown(signal.SIGINT, asyncio.get_event_loop(), self.node.server)

  async def timeout_middleware(self, app, handler):
    async def middleware(request):
      try:
        return await asyncio.wait_for(handler(request), timeout=self.response_timeout)
      except HTTPNotFound:
        if DEBUG >= 1:
          print(f"[WARN] Not Found: {request.method} {request.path}")
        return web.json_response({"detail": "Not Found", "path": request.path}, status=404)
      except HTTPMethodNotAllowed:
        if DEBUG >= 1:
          print(f"[WARN] Method Not Allowed: {request.method} {request.path}")
        return web.json_response({"detail": f"Method Not Allowed: dùng {request.path} với method khác (GET/POST)."}, status=405)
      except asyncio.TimeoutError:
        return web.json_response({"detail": "Request timed out"}, status=408)

    return middleware

  async def log_request(self, app, handler):
    async def middleware(request):
      # Xử lý GET /v1/distributed/status ngay trong middleware để tránh 404 (route có thể không match)
      raw_path = (request.path or "").strip()
      if request.method == "GET" and "distributed/status" in raw_path:
        return await self.handle_get_distributed_status(request)
      # CRITICAL: Always log POST requests to /chat/completions for debugging
      # Hide high-frequency polling requests (topology, node lists) to
      # reduce noise. Download progress is a REAL route now
      # (handle_get_download_progress), so it stays visible when failing.
      is_polling_request = (
        request.method == "GET" and (
          "/v1/topology" in request.path or
          "/v1/network/nodes" in request.path
        )
      )
      
      if request.method == "POST" and "/chat/completions" in request.path:
        print(f"[INFO] POST request received: {request.method} {request.path} from {request.remote}")
        print(f"[INFO] Request headers: {dict(request.headers)}")
        print(f"[INFO] Request URL: {request.url}")
        print(f"[INFO] Request can_read_body: {request.can_read_body}")
      elif DEBUG >= 3 and not is_polling_request:
        print(f"Received request: {request.method} {request.path}")
      try:
        # Only log handler call for non-polling requests or POST requests
        if not is_polling_request or (request.method == "POST" and "/chat/completions" in request.path):
          print(f"[INFO] Calling handler for {request.method} {request.path}...")
        response = await handler(request)
        if request.method == "POST" and "/chat/completions" in request.path:
          print(f"[INFO] POST request handled successfully, response status: {response.status if hasattr(response, 'status') else 'N/A'}")
        return response
      except HTTPNotFound:
        if DEBUG >= 1:
          print(f"[WARN] Not Found: {request.method} {request.path}")
        return web.json_response({"detail": "Not Found", "path": request.path}, status=404)
      except HTTPMethodNotAllowed:
        if DEBUG >= 1:
          print(f"[WARN] Method Not Allowed: {request.method} {request.path}")
        return web.json_response({"detail": f"Method Not Allowed: dùng {request.path} với method khác (GET/POST)."}, status=405)
      except Exception as e:
        print(f"[ERROR] Error in middleware for {request.method} {request.path}: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"detail": f"Internal server error: {str(e)}"}, status=500)

    return middleware


  async def handle_root(self, request):
    # T2.1 --headless (via SYNAPSE_HEADLESS=1, set by `synapse serve
    # --headless`): liveness stub only. API + /healthcheck + /v1/* are
    # unaffected — this narrows GET / alone, never the server.
    import os as _os
    if _os.getenv("SYNAPSE_HEADLESS", "0") == "1":
      return web.Response(status=204)
    return web.Response(text=_DEFAULT_DASHBOARD_HTML, content_type="text/html")

  def _get_gpu_info(self) -> str:
    """Lấy thông tin GPU từ device_capabilities (hỗ trợ NVIDIA, AMD, Apple, Intel, Ascend)."""
    try:
      caps = getattr(self.node, "device_capabilities", None)
      if caps is not None:
        # Dùng SystemSpecs nếu có
        specs = getattr(caps, "_specs", None)
        if specs is None:
          # Detection costs seconds (PowerShell/CIM on Windows): once per
          # process, shared with get_system_stats_dict.
          specs = getattr(self, "_cached_system_specs", None)
          if specs is None:
            try:
              from synapse.topology.device_capabilities import SystemSpecs
              specs = self._cached_system_specs = SystemSpecs.detect()
            except Exception:
              specs = None
        if specs is not None:
          if getattr(specs, "has_gpu", False) and getattr(specs, "gpu_name", None):
            vram = getattr(specs, "gpu_vram_gb", 0) or 0
            unified = " [Unified]" if getattr(specs, "unified_memory", False) else ""
            return f"{specs.gpu_name} ({vram:.1f} GB VRAM{unified})"
          return "Không có GPU — CPU-only"
    except Exception:
      pass
    # Fallback cuối: nvidia-smi (tương thích ngược)
    try:
      out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
      )
      if out.returncode == 0 and out.stdout.strip():
        lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        return "; ".join(lines[:4]) if lines else "N/A"
    except Exception:
      pass
    return "N/A"

  def _get_gpu_utilization(self) -> Optional[int]:
    """Lay % su dung GPU tu realtime monitor (cache), fallback to nvidia-smi."""
    try:
        if hasattr(self, "node") and self.node and hasattr(self.node, "monitor"):
            metrics = self.node._current_metrics
            if metrics.gpu_util_pct > 0:
                return int(metrics.gpu_util_pct)
    except Exception:
        pass
    try:
      out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
      )
      if out.returncode == 0 and out.stdout.strip():
        # Có thể nhiều dòng (nhiều GPU); lấy giá trị đầu hoặc trung bình
        values = []
        for line in out.stdout.strip().split("\n"):
          line = line.strip()
          if not line:
            continue
          # nvidia-smi có thể trả "45" hoặc "45 %"
          val = line.replace("%", "").strip()
          try:
            values.append(int(val))
          except ValueError:
            continue
        if values:
          return round(sum(values) / len(values))  # trung bình nhiều GPU
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
      pass
    return None

  async def handle_healthcheck(self, request):
    payload = {"status": "ok"}
    if self.node and self.node.inference_engine:
      payload["inference_ready"] = True
      engine = self.node.inference_engine
      payload["model_loaded"] = getattr(engine, "_model", None) is not None
    else:
      payload["inference_ready"] = False
      payload["model_loaded"] = False
    return web.json_response(payload)


  async def handle_get_models(self, request):
    """Danh sách model: từ HF_MODELS (download + web UI) và hf_models.json."""
    from synapse.helpers import _load_model_db
    db = _load_model_db()
    all_names = set(HF_MODELS.keys())
    for entry in db:
        if entry.get("name"):
            all_names.add(entry["name"])
            
    # "ready" = weights config present in the local HF cache (it used to be
    # True for every listed model, including 34B ones never downloaded).
    try:
      from huggingface_hub import try_to_load_from_cache
    except Exception:
      try_to_load_from_cache = None

    def _cached(name):
      if try_to_load_from_cache is None:
        return False
      try:
        return isinstance(try_to_load_from_cache(resolve_hf_id(name), "config.json"), str)
      except Exception:
        return False
    models_list = [{"id": name, "object": "model", "owned_by": "pytorch", "ready": _cached(name)}
                   for name in sorted(all_names)]
    return web.json_response({"object": "list", "data": models_list})

  async def handle_get_models_status(self, request):
    """Trạng thái model: danh sách từ HF_MODELS (dùng cho web UI)."""
    try:
      ips = get_all_ip_addresses_and_interfaces()
      host = next((ip for ip in ips if not ip.startswith("127.")), "localhost")
      base_url = f"http://{host}:52415"
    except Exception:
      base_url = "http://localhost:52415"
    return web.json_response({
      "enabled": True,
      "models": sorted(HF_MODELS.keys()),
      "base_url": base_url,
    })

  def _get_hf_downloaded_names(self) -> Set[str]:
    """Model đã tải đủ (có trong HF cache và load được local_files_only). Tránh hiển thị 'đã tải' khi mới tải dở."""
    try:
      from huggingface_hub import scan_cache_dir, snapshot_download
      cache = scan_cache_dir()
      repo_ids_in_cache: Set[str] = set()
      for repo in getattr(cache, "repos", []) or []:
        rid = getattr(repo, "repo_id", None)
        if rid:
          repo_ids_in_cache.add(rid)
      complete: Set[str] = set()
      
      # 1. Quét HF_MODELS truyền thống
      for name, repo_id in HF_MODELS.items():
        if repo_id not in repo_ids_in_cache:
          continue
        try:
          snapshot_download(repo_id=repo_id, local_files_only=True)
          complete.add(name)
        except Exception:
          pass
          
      # 2. Quét từ Discovery Database
      try:
          from synapse.helpers import _load_model_db
          db = _load_model_db()
          for entry in db:
              rid = entry.get("name")
              if rid and rid in repo_ids_in_cache:
                  try:
                      snapshot_download(repo_id=rid, local_files_only=True)
                      complete.add(rid) # Save HF repo id as the alias
                  except Exception:
                      pass
      except Exception:
          pass
          
      return complete
    except Exception:
      return set()

  def _get_hf_cache_size_by_repo(self) -> Dict[str, int]:
    """Trả về dict repo_id -> size_on_disk (bytes) từ HF cache."""
    try:
      from huggingface_hub import scan_cache_dir
      cache = scan_cache_dir()
      out: Dict[str, int] = {}
      for repo in getattr(cache, "repos", []) or []:
        rid = getattr(repo, "repo_id", None)
        size = getattr(repo, "size_on_disk", 0) or 0
        if rid:
          out[rid] = size
      return out
    except Exception:
      return {}

  async def handle_get_models_list(self, request):
    """Danh sách cho web UI: đã tải (HF cache) + có thể tải (mọi model từ db). Truyền full thông số UI/UX."""
    try:
      downloaded_names = self._get_hf_downloaded_names()
      cache_sizes = self._get_hf_cache_size_by_repo()
      
      from synapse.helpers import _load_model_db
      db = _load_model_db()
              
      downloaded = []
      for n in sorted(downloaded_names):
        repo_id = HF_MODELS.get(n, n)
        size_bytes = cache_sizes.get(repo_id, 0)
        size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else None
        
        entry = next((e for e in db if e.get("name") == repo_id), {})
        param_size = entry.get("parameter_count", HF_MODEL_PARAMS.get(n, "-"))
        
        downloaded.append({
          "name": n,
          "status": "downloaded",
          "repo_id": repo_id,
          "size_gb": size_gb,
          "parameter_size": param_size,
          "quantization_level": entry.get("quantization", "FP16/BF16"),
          "memory": f"{entry['min_ram_gb']} - {entry['recommended_ram_gb']}GB" if entry.get("min_ram_gb") else "-",
          "fit": "-",
          "speed": "-",
          "family": entry.get("provider", "-"),
          "architecture": entry.get("architecture", "-"),
          "pipeline_tag": entry.get("pipeline_tag", "")
        })
        
      available = []
      # Thêm các model từ DB mà chưa được tải
      for entry in db:
          name = entry.get("name")
          if name and name not in downloaded_names:
              available.append({
                  "name": name,
                  "status": "available",
                  "repo_id": name,
                  "parameter_size": entry.get("parameter_count", "-"),
                  "family": entry.get("provider", "-"),
                  "quantization_level": entry.get("quantization", "-"),
                  "memory": f"{entry['min_ram_gb']} - {entry['recommended_ram_gb']}GB" if entry.get("min_ram_gb") else "-",
                  "architecture": entry.get("architecture", "-"),
                  "pipeline_tag": entry.get("pipeline_tag", ""),
                  "hf_downloads": entry.get("hf_downloads", 0)
              })
              
      # Sort available models by downloads descending for better UI ranking
      available.sort(key=lambda x: x.get("hf_downloads", 0), reverse=True)
      
      return web.json_response({
        "downloaded": downloaded,
        "available": available,
      })
    except Exception:
      if DEBUG >= 1:
        traceback.print_exc()
      # Fallback
      available = [{"name": n, "status": "available"} for n in sorted(HF_MODELS.keys())]
      return web.json_response({"downloaded": [], "available": available})

  async def handle_get_models_search(self, request):
    """Tìm kiếm models trên Hugging Face API."""
    try:
      query = request.query.get("q", "").strip()
      if not query:
        return web.json_response({"data": []})
      
      from synapse.model_discovery import search_hugging_face
      results = await asyncio.get_event_loop().run_in_executor(None, lambda: search_hugging_face(query))
      
      # Mark models that are already downloaded
      downloaded_names = self._get_hf_downloaded_names()
      for r in results:
        if r["name"] in downloaded_names:
          r["isDownloaded"] = True
          
      return web.json_response({"data": results})
    except Exception as e:
      return web.json_response({"detail": str(e)}, status=500)

  # Weight formats skipped when pulling only this node's part (the
  # safetensors it needs are fetched per file below).
  _WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.h5", "*.msgpack",
                      "*.gguf", "*.onnx", "*.ot", "*.tflite"]

  def _pull_node_part(self, model_name: str, hf_id: str):
    """Fetch config/tokenizer plus only the checkpoint files of this node's
    current layer range. Returns ((start, end), files), or None when the
    range or a safetensors index is unknown (caller pulls the whole repo)."""
    try:
      base = build_base_shard(model_name, self.inference_engine_classname)
      if base is None:
        return None
      shard = self.node.get_current_shard(base)
      from huggingface_hub import snapshot_download
      from synapse.inference.partial_loader import PartialModel
      import torch
      snapshot_download(repo_id=hf_id, ignore_patterns=self._WEIGHT_PATTERNS)
      pm = PartialModel(hf_id, torch.float32, "cpu")  # meta skeleton only, nothing loaded
      files = pm.prefetch(shard.start_layer, shard.end_layer,
                          shard.is_first_layer(), shard.is_last_layer())
      return (shard.start_layer, shard.end_layer), files
    except Exception as e:
      if DEBUG >= 1: print(f"[pull] node-part pull failed ({e}); pulling the whole repo")
      return None

  async def handle_post_models_pull(self, request):
    """Tải model: Đầu tiên kiểm tra trong HF_MODELS, nếu không có thì scrape metadata và tiến hành HF snapshot_download."""
    try:
      data = await request.json()
      model_name = (data.get("model") or data.get("name") or "").strip()
      if not model_name:
        return web.json_response({"success": False, "message": "Thiếu model"}, status=400)
        
      is_new_discovery = False
      hf_id = HF_MODELS.get(model_name)
      
      if not hf_id:
          if "/" in model_name:
              from synapse.model_discovery import discover_and_save_model
              if DEBUG >= 1: print(f"Không tìm thấy '{model_name}' trong danh sách cứng, bắt đầu tiến trình khám phá...")
              try:
                  # This blocks, but it's okay for pulling
                  discovered = discover_and_save_model(model_name)
                  if discovered:
                      hf_id = model_name
                      is_new_discovery = True
                  else:
                      return web.json_response({"success": False, "message": f"Model '{model_name}' không hợp lệ hoặc không có trên HF."}, status=400)
              except Exception as e:
                  return web.json_response({"success": False, "message": f"Lỗi cào dữ liệu model: {e}"}, status=400)
          else:
              return web.json_response({"success": False, "message": f"Model '{model_name}' không có trong danh sách. Hãy nhập đúng repo ID (ví dụ: unsloth/Qwen)."}, status=400)
      else:
          hf_id = resolve_hf_id(model_name)
          
      part = None if data.get("full") else await asyncio.to_thread(self._pull_node_part, model_name, hf_id)
      if part is None:
        from huggingface_hub import snapshot_download
        await asyncio.to_thread(snapshot_download, repo_id=hf_id)
        msg = f"Đã tải {model_name} -> {hf_id} (Hugging Face cache)"
      else:
        (start, end), files = part
        msg = (f"Đã tải phần của node này: {model_name} -> {hf_id}, layer {start}-{end}, "
               f"{len(files)} file trọng số")
      if is_new_discovery:
          msg += " [Đã lưu metadata cục bộ]"
          
      return web.json_response({
        "success": True,
        "message": msg,
      })
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_models_delete(self, request):
    """Xóa model khỏi HF cache (danh sách từ HF_MODELS)."""
    try:
      data = await request.json()
      model_name = (data.get("model") or data.get("name") or "").strip()
      if not model_name:
        return web.json_response({"success": False, "message": "Thiếu model"}, status=400)
      from huggingface_hub import scan_cache_dir
      hf_id = resolve_hf_id(model_name)
      cache_info = scan_cache_dir()
      rev_hashes = []
      for repo in getattr(cache_info, "repos", []) or []:
        if getattr(repo, "repo_id", "") == hf_id:
          for rev in getattr(repo, "revisions", []) or []:
            rev_hashes.append(getattr(rev, "commit_hash", ""))
          break
      if rev_hashes:
        strategy = cache_info.delete_revisions(*rev_hashes)
        strategy.execute()
      return web.json_response({
        "success": True,
        "message": f"Đã xóa cache HF cho {model_name}" if rev_hashes else f"Không tìm thấy cache for {model_name}",
      })
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_get_model_checkpoints(self, request):
    """Liệt kê các checkpoints đã lưu trong thư mục checkpoints/."""
    try:
      from synapse.loading import get_checkpoints_base_dir
      cp_dir = get_checkpoints_base_dir()
      if not cp_dir.exists():
        return web.json_response({"data": []})
      
      checkpoints = []
      for model_dir in cp_dir.iterdir():
        if model_dir.is_dir():
          # Kiểm tra xem có phải adapter (LoRA) hay full model
          is_adapter = (model_dir / "adapter_config.json").exists()
          is_full = (model_dir / "config.json").exists()
          
          if is_adapter or is_full:
            checkpoints.append({
              "name": model_dir.name,
              "path": str(model_dir.absolute()),
              "type": "adapter" if is_adapter else "full",
              "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(model_dir.stat().st_ctime))
            })
            
      return web.json_response({"data": checkpoints})
    except Exception as e:
      return web.json_response({"detail": str(e)}, status=500)

  async def handle_post_models_merge(self, request):
    """Hợp nhất models/adapters. Body: base_model, adapters (list of paths), output_name, merge_type."""
    try:
      from synapse.training.merging import merge_lora_adapters
      body = await request.json()
      base_model = body.get("base_model")
      adapters = body.get("adapters", []) # List of paths or names in checkpoints/
      output_name = body.get("output_name")
      merge_type = body.get("merge_type", "linear")
      
      if not base_model or not adapters or not output_name:
        return web.json_response({"success": False, "message": "Thiếu base_model, adapters hoặc output_name"}, status=400)
      
      from synapse.loading import get_checkpoints_base_dir
      cp_dir = get_checkpoints_base_dir()
      
      # Resolve adapter paths
      resolved_adapters = []
      for a in adapters:
        p = Path(a)
        if not p.is_absolute():
          p = cp_dir / a
        if p.exists():
          resolved_adapters.append(str(p.absolute()))
        else:
          return web.json_response({"success": False, "message": f"Adapter không tồn tại: {a}"}, status=400)
          
      output_path = str((cp_dir / output_name).absolute())
      
      # Run merging in a separate thread to not block the event loop
      loop = asyncio.get_event_loop()
      result = await loop.run_in_executor(
        None, 
        lambda: merge_lora_adapters(resolve_hf_id(base_model), resolved_adapters, output_path, merge_type=merge_type)
      )
      
      if result.get("success"):
        self.log_activity("Model Merged", output_name, "success")
        return web.json_response(result)
      else:
        self.log_activity("Merge Failed", output_name, "failed")
        return web.json_response(result, status=500)
        
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_chat_token_encode(self, request):
    """Token encode: dùng chat completions với model trong danh sách."""
    return web.json_response(
      {"error": "Dùng chat completions với model trong danh sách (Quản lý mô hình)."},
      status=400,
    )

  async def handle_post_chat_completions(self, request):
    request_id = None
    print(f"[INFO] handle_post_chat_completions called: path={request.path}, remote={request.remote}")
    print(f"[INFO] Request method: {request.method}, content_type: {request.content_type}")
    try:
      print("[INFO] Reading request JSON...")
      data = await request.json()
      print(f"[INFO] Request data received: model={data.get('model')}, stream={data.get('stream')}, messages_count={len(data.get('messages', []))}")
      max_tokens = data.get("max_tokens")
      if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = None
      model_from_request = data.get("model") or self.default_model
      if str(model_from_request).strip().lower() == "default":
        # `synapse benchmark` without --model sends "default": the server's model.
        model_from_request = self.default_model
      if model_from_request and model_from_request.startswith("gpt-"):
        model_from_request = self.default_model
      # Tương thích client cũ: bỏ prefix "ollama/" nếu có
      model_id_raw = (model_from_request[7:] if model_from_request.startswith("ollama/") else model_from_request).strip()
      model_id = resolve_hf_id(model_id_raw)
      
      self.log_activity("Inference Task", model_id, "success")

      # Engine PyTorch: dùng process_prompt phân tán (nhiều node / layer sharding)
      if self.inference_engine_classname == "PyTorchHFInferenceEngine":
        base_shard = build_base_shard(model_id, self.inference_engine_classname)
        if not base_shard:
          return web.json_response(
            {"error": {"message": f"Model '{model_id}' không có trong registry PyTorch/HF. Chọn model từ danh sách.", "code": "model_not_found"}},
            status=400,
          )

        request_id = str(uuid.uuid4())
        self.token_queues[request_id] = asyncio.Queue()
        try:
          repo = get_repo(model_id, self.inference_engine_classname)
          tokenizer = await resolve_tokenizer(repo)
        except Exception as e:
          if DEBUG >= 1:
            traceback.print_exc()
          return web.json_response(
            {"error": {"message": f"Không load được tokenizer cho {model_id}: {e}", "code": "tokenizer_error"}},
            status=500,
          )
        chat_request = parse_chat_request(data, self.default_model)
        has_system_message = any(msg.role == "system" for msg in chat_request.messages)
        if not has_system_message and self.system_prompt:
          # The configured system prompt (settings / --system-prompt). It used
          # to be replaced by a hardcoded "answer in Vietnamese" line, which
          # derailed small English models into garbage.
          chat_request.messages.insert(0, Message(role="system", content=self.system_prompt))
        prompt = build_prompt(tokenizer, chat_request.messages)
        stream = data.get("stream", False)
        if self.on_chat_completion_request:
          self.on_chat_completion_request(request_id, chat_request, prompt)
        process_task = asyncio.create_task(
          self.node.process_prompt(base_shard, prompt, request_id=request_id, max_generate_tokens=max_tokens, inference_state=_sampling_state(data))
        )
        try:
          if stream:
            response = web.StreamResponse(
              status=200,
              headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
            await response.prepare(request)
            
            all_tokens_stream: List[int] = []
            detok = StreamDetokenizer(tokenizer)
            process_done_at: Optional[float] = None
            
            while True:
              try:
                tokens, is_finished = await asyncio.wait_for(
                  self.token_queues[request_id].get(),
                  timeout=float(self.response_timeout),
                )
              except asyncio.TimeoutError:
                if process_task.done():
                  if process_done_at is None:
                    process_done_at = time.time()
                  task_exc = process_task.exception()
                  if task_exc:
                    if DEBUG >= 1:
                      print(f"Inference task failed for {request_id}: {task_exc}")
                    return web.json_response(
                      {"error": {"message": f"Inference failed: {task_exc}", "code": "inference_failed"}},
                      status=500,
                    )
                  if time.time() - process_done_at >= 5.0:
                    if DEBUG >= 1:
                      print(f"Grace timeout waiting for streamed tokens: {request_id}")
                    break
                  continue
                else:
                  # Timeout real
                  break

              all_tokens_stream.extend(tokens)
              delta = detok.push(tokens)
              if delta:
                await response.write(f"data: {json.dumps(stream_chunk(chat_request, request_id, delta))}\n\n".encode())
              if is_finished:
                break
            
            # Finalize the stream (with exact usage, OpenAI-style, so clients
            # never have to estimate the token count from words)
            final = generate_completion(chat_request, tokenizer, prompt, request_id, [], True, 'stop', 'chat.completion')
            try:
              n_prompt = len(tokenizer.encode(prompt))
            except Exception:
              n_prompt = 0
            final["usage"] = {"prompt_tokens": n_prompt, "completion_tokens": len(all_tokens_stream),
                              "total_tokens": n_prompt + len(all_tokens_stream)}
            await response.write(f"data: {json.dumps(final)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
          else:
            all_tokens = []
            process_done_at: Optional[float] = None
            while True:
              try:
                tokens, is_finished = await asyncio.wait_for(
                  self.token_queues[request_id].get(),
                  timeout=10.0,
                )
              except asyncio.TimeoutError:
                if process_task.done():
                  if process_done_at is None:
                    process_done_at = time.time()
                  task_exc = process_task.exception()
                  if task_exc:
                    if DEBUG >= 1:
                      print(f"Inference task failed for {request_id}: {task_exc}")
                    return web.json_response(
                      {"error": {"message": f"Inference failed: {task_exc}", "code": "inference_failed"}},
                      status=500,
                    )
                  if time.time() - process_done_at >= 5.0:
                    if DEBUG >= 1:
                      print(f"Grace timeout waiting for tokens: {request_id}")
                    break
                continue
              all_tokens.extend(tokens)
              if is_finished:
                break
            completion = generate_completion(
              chat_request,
              tokenizer,
              prompt,
              request_id,
              all_tokens,
              False,
              "stop",
              "chat.completion",
            )
            return web.json_response(completion)
        finally:
          if not process_task.done():
            process_task.cancel()
            try:
              await process_task
            except asyncio.CancelledError:
              pass
            except Exception as cleanup_e:
              if DEBUG >= 1:
                print(f"[WARN] process_task cleanup error for {request_id}: {cleanup_e}")
                traceback.print_exc()
      return web.json_response(
        {"error": {"message": "Chọn model trong danh sách (Quản lý mô hình).", "code": "model_not_found"}},
        status=400,
      )
    except Exception as outer_e:
      if DEBUG >= 1:
        print(f"Unhandled exception in handle_post_chat_completions: {outer_e}")
        traceback.print_exc()
      return web.json_response(
        {"detail": f"Internal server error: {str(outer_e)}"},
        status=500,
      )
    finally:
      if request_id and request_id in self.token_queues:
        if DEBUG >= 2:
          print(f"Cleaning up token queue: {request_id=}")
        del self.token_queues[request_id]

  async def handle_get_topology(self, request):
    try:
      topology = self.node.current_topology
      if topology:
        return web.json_response(topology.to_json())
      else:
        return web.json_response({})
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error getting topology: {str(e)}"}, status=500)

  def _get_partitions_count(self) -> int:
    """Số partition (node) trong topology; 1 = chỉ máy này."""
    try:
      get_auth = getattr(self.node, "_get_authoritative_partitions", None)
      if callable(get_auth):
        partitions = get_auth()
        if partitions:
          return len(partitions)
      if not self.node.topology or not self.node.topology.nodes:
        peers = getattr(self.node, "peers", []) or []
        return max(1, 1 + len(peers))
      partitions = self.node._compute_lacp_partition()
      if partitions:
        return len(partitions)
      peers = getattr(self.node, "peers", []) or []
      return max(1, len(self.node.topology.nodes), 1 + len(peers))
    except Exception:
      return 1

  async def handle_get_distributed_status(self, request):
    """Trạng thái phân tán: nhiều node cùng xử lý 1 request (inference/training)."""
    try:
      n = self._get_partitions_count()
      payload = {
        "partitions_count": n,
        "multi_node_inference": n > 1,
        "multi_node_training": n > 1,
        "message": "Luồng phân tán: nhiều node (partitions_count > 1).",
      }
      # C-W9: surface in-memory health signals (all optional, getattr-guarded
      # so older nodes simply omit them). Operators can now observe degraded
      # recovery, route mix, quant state, and version skew via API.
      try:
        node = getattr(self, "node", None)
        if node is not None:
          degraded = getattr(node, "_last_recovery_degraded", None)
          if degraded:
            payload["recovery_degraded"] = degraded
          standby = getattr(node, "_last_standby", None)
          if standby is not None:
            payload["standby_nodes"] = list(standby)
          hints = getattr(node, "_last_autoscale_hints", None)
          if hints:
            payload["autoscale_hints"] = hints
          stats = getattr(node, "transport_stats", None)
          counts = stats.snapshot() if stats is not None else {}
          if counts:
            payload["transport_counts"] = counts
          eng = getattr(node, "inference_engine", None)
          prof = getattr(eng, "_hw_profile", None)
          if isinstance(prof, dict):
            payload["quant_active"] = prof.get("quant_active")
          bridge = getattr(getattr(node, "libp2p_node", None), "_bridge", None)
          if bridge is not None:
            try:
              from synapse.routing.p2p_socket_bridge import BRIDGE_VERSION as _BV
              payload["bridge_version"] = _BV
            except Exception:
              pass
            payload["peer_versions"] = {
              pid: {"bridge_version": h.get("bridge_version"),
                    "synapse_version": h.get("synapse_version")}
              for pid, h in ((pid, bridge.get_peer_handshake(pid) or {})
                             for pid in list(getattr(bridge, "_peer_handshakes", {}) or {}))
            }
      except Exception:
        pass
      return web.json_response(payload)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": str(e)}, status=500)

  async def handle_get_download_progress(self, request):
    """C-W8: download progress was referenced in middleware but had no route
    (404 for every poller). Reads the live node_download_progress map."""
    try:
      raw = getattr(getattr(self, "node", None), "node_download_progress", {}) or {}
      progress = {}
      for node_id, event in raw.items():
        try:
          progress[node_id] = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        except Exception:
          progress[node_id] = {"status": "unknown"}
      return web.json_response({"ok": True, "progress": progress})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"ok": False, "progress": {}, "degraded": True,
                                "detail": str(e)[:200]}, status=500)

  async def handle_post_training_distributed(self, request):
    """Training phân tán: chạy job trên node nhận request. Chia tải theo layer (nhiều máy) dùng pipeline khi engine hỗ trợ train()."""
    resp = await self.handle_post_training(request)
    if resp.status == 200 and resp.text:
      try:
        data = json.loads(resp.text)
        data["message"] = "Training chia tải (pipeline) đã bắt đầu. Trang Training sẽ cập nhật tiến trình."
        return web.json_response(data)
      except Exception:
        pass
    return resp

  def get_system_stats_dict(self):
    try:
      if hasattr(self, "node") and self.node and self.node._current_metrics:
        m = self.node._current_metrics
        cpu_percent = m.cpu_util_pct
        memory_percent = m.ram_util_pct
        memory_used_gb = round(m.ram_used_mb / 1024, 2)
        memory_total_gb = round(m.ram_total_mb / 1024, 2)
      else:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = round(memory.used / (1024**3), 2)
        memory_total_gb = round(memory.total / (1024**3), 2)
      uptime = time.time() - psutil.boot_time()

      # Format uptime
      days = int(uptime // (24 * 3600))
      hours = int((uptime % (24 * 3600)) // 3600)
      minutes = int((uptime % 3600) // 60)
      uptime_str = f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"
      
      # Additional Hardcore Telemetry
      try:
          disk = psutil.disk_usage('/')
          disk_total_gb = round(disk.total / (1024**3), 1)
          disk_used_gb = round(disk.used / (1024**3), 1)
          disk_percent = disk.percent
      except:
          disk_total_gb = disk_used_gb = disk_percent = 0

      try:
          swap = psutil.swap_memory()
          swap_total_gb = round(swap.total / (1024**3), 1)
          swap_used_gb = round(swap.used / (1024**3), 1)
          swap_percent = swap.percent
      except:
          swap_total_gb = swap_used_gb = swap_percent = 0

      try:
          net_io = psutil.net_io_counters()
          net_sent_mb = round(net_io.bytes_sent / (1024**2), 1)
          net_recv_mb = round(net_io.bytes_recv / (1024**2), 1)
          net_pkts_sent = net_io.packets_sent
          net_pkts_recv = net_io.packets_recv
          net_errin = net_io.errin
          net_errout = net_io.errout
          net_dropin = net_io.dropin
          net_dropout = net_io.dropout
      except:
          net_sent_mb = net_recv_mb = net_pkts_sent = net_pkts_recv = net_errin = net_errout = net_dropin = net_dropout = 0

      try:
          if hasattr(os, 'getloadavg'):
              load1, load5, load15 = [round(x, 2) for x in os.getloadavg()]
          else:
              load1 = load5 = load15 = 0.0
      except:
          load1 = load5 = load15 = 0.0
          
      try:
          cpu_freq = psutil.cpu_freq()
          cpu_freq_current = round(cpu_freq.current, 1) if cpu_freq else 0
          cpu_freq_max = round(cpu_freq.max, 1) if cpu_freq else 0
      except:
          cpu_freq_current = cpu_freq_max = 0
          
      try:
          cpu_stats = psutil.cpu_stats()
          ctx_switches = cpu_stats.ctx_switches
          interrupts = cpu_stats.interrupts
      except:
          ctx_switches = interrupts = 0
          
      try:
          disk_io = psutil.disk_io_counters()
          disk_read_mb = round(disk_io.read_bytes / (1024**2), 1) if disk_io else 0
          disk_write_mb = round(disk_io.write_bytes / (1024**2), 1) if disk_io else 0
          disk_read_count = disk_io.read_count if disk_io else 0
          disk_write_count = disk_io.write_count if disk_io else 0
      except:
          disk_read_mb = disk_write_mb = disk_read_count = disk_write_count = 0
          
      try:
          pids = psutil.pids()
          process_count = len(pids)
      except:
          process_count = 0

      downloaded_names = self._get_hf_downloaded_names()
      models_count = len(downloaded_names)

      # GPU info from new cross-platform hardware detection
      gpu_info_str = self._get_gpu_info()
      gpu_utilization = self._get_gpu_utilization()

      # Try to get structured GPU data from SystemSpecs (cached to prevent blocking event loop)
      gpu_name = None
      gpu_vram_gb = None
      unified_memory = False
      try:
        if not hasattr(self, '_cached_system_specs'):
            from synapse.topology.device_capabilities import SystemSpecs
            self._cached_system_specs = SystemSpecs.detect()
        
        if self._cached_system_specs.has_gpu and self._cached_system_specs.gpu_name:
          gpu_name = self._cached_system_specs.gpu_name
          gpu_vram_gb = round(self._cached_system_specs.gpu_vram_gb, 1) if self._cached_system_specs.gpu_vram_gb else None
          unified_memory = bool(self._cached_system_specs.unified_memory)
      except Exception:
        pass

      return {
        "cpu": cpu_percent,
        "memory": memory_percent,
        "memory_used_gb": memory_used_gb,
        "memory_total_gb": memory_total_gb,
        "uptime": uptime_str,
        "gpu": gpu_info_str,         # legacy string for old UI elements
        "gpu_name": gpu_name,        # structured — new UI
        "gpu_vram_gb": gpu_vram_gb,  # structured — new UI
        "unified_memory": unified_memory,
        "gpu_utilization": gpu_utilization,
        "models_count": models_count,
        "gpu_vram_used_mb": self.node._current_metrics.gpu_vram_used_mb if hasattr(self, "node") and self.node else 0,
        "gpu_power_w": self.node._current_metrics.gpu_power_w if hasattr(self, "node") and self.node else 0,
        "gpu_temp_c": self.node._current_metrics.gpu_temp_c if hasattr(self, "node") and self.node else 0,
        
        # New telemetry
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "disk_percent": disk_percent,
        "swap_total_gb": swap_total_gb,
        "swap_used_gb": swap_used_gb,
        "swap_percent": swap_percent,
        "net_sent_mb": net_sent_mb,
        "net_recv_mb": net_recv_mb,
        "net_pkts_sent": net_pkts_sent,
        "net_pkts_recv": net_pkts_recv,
        "net_errin": net_errin,
        "net_errout": net_errout,
        "net_dropin": net_dropin,
        "net_dropout": net_dropout,
        "load_avg": [load1, load5, load15],
        "cpu_freq_current": cpu_freq_current,
        "cpu_freq_max": cpu_freq_max,
        "ctx_switches": ctx_switches,
        "interrupts": interrupts,
        "disk_read_mb": disk_read_mb,
        "disk_write_mb": disk_write_mb,
        "disk_read_count": disk_read_count,
        "disk_write_count": disk_write_count,
        "process_count": process_count
      }
    except Exception as e:
      return {"detail": str(e), "error": True}

  async def handle_get_system_stats(self, request):
    # Off the event loop: the first call runs hardware detection (PowerShell /
    # CIM queries, ~4-5 s on Windows), which stalled every in-flight request.
    return web.json_response(await asyncio.to_thread(self.get_system_stats_dict))

  async def handle_ws_metrics(self, request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    try:
      while True:
        stats = await asyncio.to_thread(self.get_system_stats_dict)
        # Wrap it in the format expected by frontend
        await ws.send_json({"type": "metrics", "stats": stats})
        await asyncio.sleep(1)
    except Exception:
      pass
    return ws

  async def handle_get_system_info(self, request):
      """Thông tin dự án / phiên bản cho trang About."""
      try:
          from synapse.topology.device_capabilities import SystemSpecs
          specs = SystemSpecs.detect()
          hw = {
              "cpu": specs.cpu_name,
              "cpu_cores": specs.total_cpu_cores,
              "ram_total_gb": round(specs.total_ram_gb, 1),
              "ram_available_gb": round(specs.available_ram_gb, 1),
              "gpu": specs.gpu_name if specs.has_gpu else None,
              "gpu_vram_gb": round(specs.gpu_vram_gb, 1) if specs.has_gpu and specs.gpu_vram_gb else None,
              "unified_memory": bool(specs.unified_memory),
              "gpu_count": specs.gpu_count,
              "backend": specs.backend.value if specs.backend else "Unknown",
              "os": platform.system() + " " + platform.release(),
          }
      except Exception:
          hw = {}
          
      # Detect network discovery mode
      try:
          discovery_type = "Unknown"
          if hasattr(self, "discovery_module") and self.discovery_module:
              discovery_type = self.discovery_module
          elif hasattr(self.node, "discovery") and self.node.discovery:
              discovery_type = self.node.discovery.__class__.__name__.replace("Discovery", "").lower()
      except Exception:
          discovery_type = "Unknown"
      try:
        from synapse.inference.attn_backend import available_backends
        info_attn = available_backends()
      except Exception:
        info_attn = []
      return web.json_response({
          "project": "Synapse AI",
          "version": VERSION,
          "hardware": hw,
          "network_discovery": discovery_type,
          "attention_backends": info_attn,
      })
  
  
  async def handle_get_models_compatibility(self, request):
      """
      Lấy danh sách model với thông tin tương thích phần cứng.
      GET /api/v1/models/compatibility?filter=Perfect&sort=memory&page=1&limit=20
      """
      try:
          # Parse query params
          filter_fit = request.query.get("filter", None)  # Perfect, Good, Marginal, Too Tight
          sort_by = request.query.get("sort", "memory")  # memory, params, fit_score
          page = int(request.query.get("page", 1))
          limit = min(int(request.query.get("limit", 20)), 100)
          
          from synapse.helpers import (
              _estimate_model_memory_gb, _score_fit, _fit_emoji,
              _load_model_db, _QUANT_SPEED
          )
          from synapse.topology.device_capabilities import SystemSpecs
          
          # Detect hardware
          specs = SystemSpecs.detect()
          
          # Determine available memory budget
          if specs.has_gpu and specs.gpu_vram_gb and specs.gpu_vram_gb > 0:
              budget_gb = specs.gpu_vram_gb
          elif specs.unified_memory and specs.gpu_vram_gb:
              budget_gb = specs.gpu_vram_gb
          else:
              budget_gb = specs.available_ram_gb
          
          # Load model database
          db = _load_model_db()
          
          # Calculate compatibility for each model
          results = []
          for entry in db:
              model_name = entry.get("name", "")
              params_b = entry.get("parameters_raw", 0) / 1e9 if entry.get("parameters_raw") else 7.0
              ctx = entry.get("context_length", 4096)
              quant = entry.get("quantization", "Q4_K_M")
              
              mem_required = _estimate_model_memory_gb(params_b, quant, ctx)
              fit = _score_fit(mem_required, budget_gb)
              fit_emoji = _fit_emoji(fit)
              
              # Calculate fit score for sorting (lower is better)
              if mem_required > budget_gb:
                  fit_score = 1000 + (mem_required - budget_gb)
              else:
                  fit_score = (mem_required / budget_gb) * 100
              
              quant_speed = _QUANT_SPEED.get(quant, 1.0)
              base_speed = 45.0 if specs.backend and specs.backend.value == "CUDA" else 35.0
              estimated_speed = round(base_speed * quant_speed, 1)
              
              results.append({
                  "name": model_name,
                  "parameters": f"{params_b:.1f}B",
                  "parameters_raw": params_b,
                  "quantization": quant,
                  "context_length": ctx,
                  "memory_needed_gb": round(mem_required, 2),
                  "fit": fit,
                  "fit_emoji": fit_emoji,
                  "fit_score": round(fit_score, 2),
                  "estimated_speed": estimated_speed,
              })
          
          # Apply filter
          if filter_fit:
              results = [r for r in results if r["fit"] == filter_fit]
          
          # Sort results
          if sort_by == "params":
              results.sort(key=lambda x: x["parameters_raw"], reverse=True)
          elif sort_by == "fit_score":
              results.sort(key=lambda x: x["fit_score"])
          else:  # sort by memory
              results.sort(key=lambda x: x["memory_needed_gb"])
          
          # Paginate
          total = len(results)
          start = (page - 1) * limit
          end = start + limit
          paginated = results[start:end]
          
          return web.json_response({
              "total": total,
              "page": page,
              "limit": limit,
              "total_pages": (total + limit - 1) // limit,
              "budget_gb": round(budget_gb, 1),
              "hardware": {
                  "cpu": specs.cpu_name,
                  "cpu_cores": specs.total_cpu_cores,
                  "ram_total_gb": round(specs.total_ram_gb, 1),
                  "ram_available_gb": round(specs.available_ram_gb, 1),
                  "gpu": specs.gpu_name if specs.has_gpu else None,
                  "gpu_vram_gb": round(specs.gpu_vram_gb, 1) if specs.has_gpu and specs.gpu_vram_gb else None,
                  "unified_memory": bool(specs.unified_memory),
              },
              "models": paginated,
          })
      except Exception as e:
          import traceback
          if DEBUG >= 2:
              traceback.print_exc()
          return web.json_response({"error": str(e)}, status=500)


  async def handle_get_settings(self, request):
    """Trả về cài đặt hiện tại."""
    return web.json_response({
      "default_model": getattr(self, "default_model", ""),
      "discovery_module": getattr(self, "discovery_module", "tailscale"),
      "lacp_priority": getattr(self.node, "partition_objective", None) or
        os.getenv("SYNAPSE_PARTITION_OBJECTIVE", "throughput"),
      "load_quantization": os.getenv("SYNAPSE_WEIGHT_QUANT", "auto"),
      "lookahead_window": getattr(self.node, "lookahead_window", 3),
      "transport_quantization": "int8" if os.getenv("SYNAPSE_TRANSPORT_INT8", "1") == "1" else "none"
    })

  async def handle_post_settings(self, request):
    """Lưu cài đặt và áp dụng ngay lập tức."""
    try:
      body = await request.json()
      model = (body.get("default_model") or "").strip()
      if model:
        self.default_model = model
      
      discovery_module = (body.get("discovery_module") or "").strip()
      requires_restart = False
      if discovery_module in ["udp", "tailscale", "manual"]:
        old_module = getattr(self, "discovery_module", "tailscale")
        self.discovery_module = discovery_module
        if old_module != discovery_module:
            requires_restart = True
            
      # Optimization settings: validated, applied now, persisted. (They were
      # stored but never reached the node / engine / router before.)
      err = self._apply_optimization_settings({
        k: body[k] for k in ("load_quantization", "lookahead_window",
                             "transport_quantization", "lacp_priority") if k in body})
      if err:
        return web.json_response({"error": err}, status=400)
      self._save_settings()
      self.log_activity("Cài đặt đã lưu", "-", "success")
      
      if requires_restart:
          async def delayed_restart():
              import asyncio, os
              await asyncio.sleep(1.0)
              os._exit(42)
          asyncio.create_task(delayed_restart())
          return web.json_response({"success": True, "restarted": True, "default_model": self.default_model, "discovery_module": getattr(self, "discovery_module", "")})
      
      return web.json_response({"success": True, "restarted": False, "default_model": self.default_model, "discovery_module": getattr(self, "discovery_module", "")})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_get_network_nodes(self, request):
    try:
      discovery_type = "Unknown"
      if hasattr(self, "discovery_module") and self.discovery_module:
          discovery_type = self.discovery_module
      elif hasattr(self.node, "discovery") and self.node.discovery:
          discovery_type = self.node.discovery.__class__.__name__.replace("Discovery", "").lower()
        
      results = []
      if discovery_type == "tailscale":
        results = await self.node.get_tailscale_nodes()
      else:
        if hasattr(self.node, "discovery") and self.node.discovery:
          peers = await self.node.discovery.discover_peers()
          for p in peers:
            results.append({
              "device_id": p.id(),
              "name": p.id(),
              "addresses": [p.addr()],
              "last_seen": None,
              "is_self": False
            })
            
      return web.json_response({"data": results, "type": discovery_type})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"detail": str(e), "data": [], "type": "error"}, status=500)

  def _dataset_sample_count(self, file_path: Path) -> Optional[int]:
    """Đếm số mẫu: .jsonl = số dòng không rỗng, .json = len(list)."""
    try:
      if file_path.suffix == ".jsonl":
        count = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
          for line in f:
            if line.strip():
              count += 1
        return count
      if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
          data = json.load(f)
        return len(data) if isinstance(data, list) else 1
    except Exception:
      return None
    return None

  # File cấu hình trong _data_dir — không hiển thị trong danh sách dataset
  _DATASET_EXCLUDE_NAMES = frozenset({"settings.json"})

  async def handle_get_datasets(self, request):
    try:
      data_dir = self._data_dir
      datasets = []
      # 1. Quét file trong synapse/data
      if data_dir.exists():
        for f in data_dir.glob("**/*.json*"):
          if not f.is_file() or f.name in self._DATASET_EXCLUDE_NAMES:
            continue
          stats = f.stat()
          sample_count = self._dataset_sample_count(f)
          datasets.append({
            "name": f.name,
            "path": str(f),
            "size_mb": round(stats.st_size / (1024 * 1024), 2),
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.st_ctime)),
            "sample_count": sample_count,
            "type": "file"
          })
      
      # 2. Thêm các linked folders
      for name, path_str in self.linked_datasets.items():
        p = Path(path_str)
        if p.exists():
          stats = p.stat()
          datasets.append({
            "name": f"[Linked] {name}",
            "original_name": name,
            "path": path_str,
            "size_mb": 0, # Thư mục không tính size nhanh được
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.st_ctime)),
            "sample_count": None,
            "type": "link"
          })
          
      return web.json_response({"data": datasets})
    except Exception as e:
      return web.json_response({"detail": str(e)}, status=500)

  async def handle_post_datasets_upload(self, request):
    try:
      reader = await request.multipart()
      field = await reader.next()
      if field.name != "file" or not field.filename:
        return web.json_response({"success": False, "message": "Cần gửi file với field 'file'"}, status=400)
      safe_name = Path(field.filename).name
      if not (safe_name.endswith(".json") or safe_name.endswith(".jsonl")):
        return web.json_response({"success": False, "message": "Chỉ chấp nhận file .json hoặc .jsonl"}, status=400)
      self._data_dir.mkdir(parents=True, exist_ok=True)
      dest = self._data_dir / safe_name
      size = 0
      with open(dest, "wb") as out:
        while True:
          chunk = await field.read_chunk()
          if not chunk:
            break
          size += len(chunk)
          out.write(chunk)
      self.log_activity("Dataset Uploaded", safe_name, "success")
      return web.json_response({
        "success": True,
        "name": safe_name,
        "path": str(dest),
        "size_mb": round(size / (1024 * 1024), 2),
      })
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_datasets_delete(self, request):
    try:
      body = await request.json()
      name = (body.get("name") or body.get("path") or "").strip()
      if not name:
        return web.json_response({"success": False, "message": "Thiếu name hoặc path"}, status=400)
      
      # Kiểm tra xem có phải xóa link không
      found_link = None
      for lname, lpath in self.linked_datasets.items():
        if lname == name or lpath == name or f"[Linked] {lname}" == name:
          found_link = lname
          break
      
      if found_link:
        del self.linked_datasets[found_link]
        self._save_linked_datasets()
        self.log_activity("Link Removed", found_link, "info")
        return web.json_response({"success": True, "message": f"Đã gỡ liên kết {found_link}"})

      path = Path(name)
      if not path.is_absolute():
        path = self._data_dir / path.name
      path = path.resolve()
      data_dir = self._data_dir.resolve()
      if not str(path).startswith(str(data_dir)) or path == data_dir:
        return web.json_response({"success": False, "message": "Không được xóa file ngoài thư mục dataset"}, status=400)
      if not path.is_file():
        return web.json_response({"success": False, "message": "File không tồn tại"}, status=404)
      path.unlink()
      self.log_activity("Dataset Deleted", path.name, "success")
      return web.json_response({"success": True, "name": path.name})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_get_datasets_preview(self, request):
    """Trả về vài dòng đầu của dataset (jsonl) hoặc preview json hoặc folder nội dung."""
    try:
      path_str = request.query.get("path") or request.query.get("name") or ""
      if not path_str:
        return web.json_response({"success": False, "message": "Thiếu path hoặc name"}, status=400)
      
      max_lines = int(request.query.get("lines", 50))
      
      # Kiểm tra link trước
      is_link = False
      path = None
      for lname, lpath in self.linked_datasets.items():
        if lname == path_str or lpath == path_str or f"[Linked] {lname}" == path_str:
          path = Path(lpath)
          is_link = True
          break
      
      if not is_link:
        path = self._resolve_dataset_path(path_str, name_only_ok=False)
      
      if path is None or not path.exists():
        return web.json_response({"success": False, "message": "Dataset không tồn tại hoặc không được phép"}, status=404)
      
      lines = []
      if path.is_file():
        if path.suffix == ".jsonl":
          with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
              if i >= max_lines: break
              if line.strip():
                lines.append(line.strip()[:1000])
        elif path.suffix == ".json":
          with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
          if isinstance(data, list):
            lines = [json.dumps(x, ensure_ascii=False)[:500] for x in data[:max_lines]]
          else:
            lines = [json.dumps(data, ensure_ascii=False)[:1000]]
      elif path.is_dir():
          lines.append(f"--- Thư mục: {path.name} ---")
          lines.append("Danh sách 10 file đầu tiên:")
          for f in list(path.iterdir())[:10]:
              lines.append(f.name + (" (Folder)" if f.is_dir() else ""))

      return web.json_response({"success": True, "preview": lines})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_datasets_link(self, request):
    """Liên kết một thư mục cục bộ làm dataset."""
    try:
      body = await request.json()
      name = (body.get("name") or "").strip()
      path_str = (body.get("path") or "").strip().strip('"').strip("'")
      # Thay thế \ thành / để tránh lỗi escape chars trong Python (đặc biệt là \f, \U)
      path_str = path_str.replace('\\', '/')
      
      if not name or not path_str:
        return web.json_response({"success": False, "message": "Thiếu tên hoặc đường dẫn"}, status=400)
      
      print(f"[DatasetLink] Thử liên kết đường dẫn: {path_str}")
      path = Path(path_str).resolve()
      
      if not path.exists():
        # Thử lại với Path gốc nếu resolve() làm sai lệch ở một số môi trường Windows
        path = Path(path_str)
        if not path.exists():
            print(f"[DatasetLink] [LỖI] Đường dẫn không tồn tại: {path_str}")
            return web.json_response({"success": False, "message": f"Đường dẫn không tồn tại trên hệ thống: {path_str}"}, status=400)
      
      self.linked_datasets[name] = str(path.absolute())
      self._save_linked_datasets()
      self.log_activity("Dataset Linked", name, "success")
      return web.json_response({"success": True, "message": f"Đã liên kết thành công: {name}"})
    except Exception as e:
      return web.json_response({"success": False, "message": str(e)}, status=500)

  def _resolve_dataset_path(self, path_str: str, name_only_ok: bool = True) -> Optional[Path]:
    """Resolve path/name thành Path. Hỗ trợ chuẩn hóa dấu gạch chéo Windows và Linked Datasets."""
    path_str = (path_str or "").strip().strip('"').strip("'")
    if not path_str:
      return None
    
    # Chuẩn hóa dấu gạch chéo để tránh lỗi escape chars (\f, \U, ...)
    normalized_str = path_str.replace('\\', '/')
    path = Path(normalized_str)
    
    print(f"[DEBUG] Resolve Path Input: {path_str}")
    
    # Chuyển thành đường dẫn tuyệt đối
    if not path.is_absolute():
      path = (self._data_dir / path.name)
    
    try:
      path = path.resolve()
    except Exception as e:
      print(f"[DEBUG] Resolve Error: {e}")
      # Vẫn tiếp tục với path hiện tại nếu resolve thất bại (có thể là file chưa tồn tại)
    
    print(f"[DEBUG] Final Resolved Path: {path}")

    # Kiểm tra xem có nằm trong whitelist linked không
    is_linked = False
    for lname, lpath_raw in self.linked_datasets.items():
        try:
            lpath = Path(lpath_raw).resolve()
            # Kiểm tra xem 'path' có nằm trong hoặc chính là 'lpath'
            path.relative_to(lpath)
            is_linked = True
            print(f"[DEBUG] Matched Whitelist (Linked): {lname} ({lpath})")
            break
        except (ValueError, Exception):
            continue
            
    # Kiểm tra trong data_dir mặc định
    data_dir = self._data_dir.resolve()
    is_in_data = False
    try:
        path.relative_to(data_dir)
        is_in_data = True
        print("[DEBUG] Matched Whitelist (Default Data Dir)")
    except (ValueError, Exception):
        pass

    if not is_linked and not is_in_data:
      print("[DEBUG] [LỖI] Truy cập bị từ chối: Đường dẫn không nằm trong Whitelist")
      return None

    if not path.exists():
      if not (name_only_ok and path.parent == data_dir):
        print(f"[DEBUG] [LỖI] File/Folder không tồn tại trên ổ đĩa: {path}")
        return None
    
    return path

  async def handle_get_datasets_download(self, request):
    """Tải file dataset về (attachment). Query: path hoặc name."""
    try:
      path_str = request.query.get("path") or request.query.get("name") or ""
      
      # Kiểm tra link trước
      for lname, lpath in self.linked_datasets.items():
        if lname == path_str or f"[Linked] {lname}" == path_str:
          # Không cho phép tải cả folder, chỉ cho phép preview
          return web.json_response({"success": False, "message": "Không hỗ trợ tải cả thư mục, chỉ có thể xem preview."}, status=400)

      path = self._resolve_dataset_path(path_str, name_only_ok=False)
      if path is None or not path.is_file():
        return web.json_response({"success": False, "message": "File không tồn tại hoặc không được phép"}, status=404)
      return web.FileResponse(
        path,
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
      )
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_datasets_create(self, request):
    """Tạo dataset mới: empty, từ template, hoặc từ nội dung nhập (content). Body: name, format, template, content (tùy chọn)."""
    try:
      body = await request.json()
      name = (body.get("name") or body.get("filename") or "").strip()
      if not name:
        return web.json_response({"success": False, "message": "Thiếu name"}, status=400)
      if not (name.endswith(".json") or name.endswith(".jsonl")):
        return web.json_response({"success": False, "message": "Tên file phải có đuôi .json hoặc .jsonl"}, status=400)
      fmt = (body.get("format") or "jsonl").strip().lower()
      if fmt not in ("json", "jsonl"):
        fmt = "jsonl"
      template = (body.get("template") or body.get("type") or "empty").strip().lower()
      if template not in ("empty", "text", "messages", "alpaca"):
        template = "empty"
      content = (body.get("content") or body.get("data") or "").strip()
      self._data_dir.mkdir(parents=True, exist_ok=True)
      path = self._data_dir / name
      if path.exists():
        return web.json_response({"success": False, "message": "File đã tồn tại"}, status=400)
      if content:
        if fmt == "jsonl":
          lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
          for i, ln in enumerate(lines):
            try:
              json.loads(ln)
            except json.JSONDecodeError as e:
              return web.json_response({"success": False, "message": f"Dòng {i + 1} không phải JSON hợp lệ: {e}"}, status=400)
          path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
          try:
            data = json.loads(content)
            if not isinstance(data, list):
              return web.json_response({"success": False, "message": "Nội dung JSON phải là mảng (array)"}, status=400)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
          except json.JSONDecodeError as e:
            return web.json_response({"success": False, "message": f"JSON không hợp lệ: {e}"}, status=400)
      elif template == "empty":
        if fmt == "jsonl":
          path.write_text("", encoding="utf-8")
        else:
          path.write_text("[]", encoding="utf-8")
      else:
        if template == "text":
          sample = {"text": "Câu hỏi mẫu và câu trả lời mẫu."}
        elif template == "messages":
          sample = {"messages": [{"role": "user", "content": "Câu hỏi mẫu"}, {"role": "assistant", "content": "Câu trả lời mẫu"}]}
        else:
          sample = {"instruction": "Hướng dẫn mẫu", "input": "", "output": "Câu trả lời mẫu"}
        if fmt == "jsonl":
          path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
          path.write_text(json.dumps([sample], ensure_ascii=False, indent=2), encoding="utf-8")
      self.log_activity("Dataset Created", name, "success")
      return web.json_response({"success": True, "name": name, "path": str(path)})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_datasets_rename(self, request):
    """Đổi tên dataset. Body: name (hoặc path), new_name."""
    try:
      body = await request.json()
      name = (body.get("name") or body.get("path") or "").strip()
      new_name = (body.get("new_name") or "").strip()
      if not name or not new_name:
        return web.json_response({"success": False, "message": "Thiếu name hoặc new_name"}, status=400)
      if not (new_name.endswith(".json") or new_name.endswith(".jsonl")):
        return web.json_response({"success": False, "message": "new_name phải có đuôi .json hoặc .jsonl"}, status=400)
      path = self._resolve_dataset_path(name, name_only_ok=False)
      if path is None or not path.is_file():
        return web.json_response({"success": False, "message": "File không tồn tại"}, status=404)
      new_path = path.parent / new_name
      if new_path.exists():
        return web.json_response({"success": False, "message": "Tên mới đã tồn tại"}, status=400)
      path.rename(new_path)
      self.log_activity("Dataset Renamed", f"{path.name} → {new_name}", "success")
      return web.json_response({"success": True, "name": new_name, "path": str(new_path)})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  def _validate_dataset_file(self, path: Path) -> Dict:
    """Kiểm tra format file: valid_count, errors (list {line, message}), format_detected (text|messages|alpaca|unknown)."""
    result = {"valid_count": 0, "errors": [], "format_detected": "unknown", "total_lines": 0}
    formats_seen = set()
    try:
      if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
          for line_num, line in enumerate(f, 1):
            result["total_lines"] += 1
            if not line.strip():
              continue
            try:
              obj = json.loads(line)
              if not isinstance(obj, dict):
                result["errors"].append({"line": line_num, "message": "Mỗi dòng phải là JSON object"})
                continue
              if "text" in obj:
                formats_seen.add("text")
              if "messages" in obj:
                formats_seen.add("messages")
              if "instruction" in obj or "output" in obj:
                formats_seen.add("alpaca")
              result["valid_count"] += 1
            except json.JSONDecodeError as e:
              result["errors"].append({"line": line_num, "message": str(e)})
      else:
        with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
        if not isinstance(data, list):
          result["errors"].append({"line": 1, "message": "File JSON phải là array"})
          return result
        result["total_lines"] = len(data)
        for i, obj in enumerate(data):
          if not isinstance(obj, dict):
            result["errors"].append({"line": i + 1, "message": "Mỗi phần tử phải là object"})
            continue
          if "text" in obj:
            formats_seen.add("text")
          if "messages" in obj:
            formats_seen.add("messages")
          if "instruction" in obj or "output" in obj:
            formats_seen.add("alpaca")
          result["valid_count"] += 1
      if formats_seen:
        result["format_detected"] = "messages" if "messages" in formats_seen else "alpaca" if "alpaca" in formats_seen else "text"
      return result
    except Exception as e:
      result["errors"].append({"line": 0, "message": str(e)})
      return result

  async def handle_get_datasets_validate(self, request):
    """Kiểm tra format dataset. Query: path hoặc name. Trả về valid_count, errors, format_detected."""
    try:
      path_str = request.query.get("path") or request.query.get("name") or ""
      path = self._resolve_dataset_path(path_str, name_only_ok=False)
      if path is None or not path.is_file():
        return web.json_response({"success": False, "message": "File không tồn tại"}, status=404)
      result = self._validate_dataset_file(path)
      return web.json_response({"success": True, **result})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_datasets_update(self, request):
    """Cập nhật nội dung dataset. Body: path hoặc name, content."""
    try:
      body = await request.json()
      name = (body.get("name") or body.get("path") or "").strip()
      content = (body.get("content") or "").strip()
      if not name or not content:
        return web.json_response({"success": False, "message": "Thiếu name hoặc content"}, status=400)
      
      path = self._resolve_dataset_path(name, name_only_ok=False)
      if path is None or not path.is_file():
        return web.json_response({"success": False, "message": "File không tồn tại hoặc không được phép chỉnh sửa"}, status=404)
        
      # Validate content before saving
      if path.suffix == ".jsonl":
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
          try:
            json.loads(ln)
          except json.JSONDecodeError as e:
            return web.json_response({"success": False, "message": f"Dòng {i + 1} không phải JSON hợp lệ: {e}"}, status=400)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
      else:
        try:
          data = json.loads(content)
          if not isinstance(data, list):
            return web.json_response({"success": False, "message": "Nội dung JSON phải là mảng (array)"}, status=400)
          path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except json.JSONDecodeError as e:
          return web.json_response({"success": False, "message": f"JSON không hợp lệ: {e}"}, status=400)
          
      self.log_activity("Dataset Updated", path.name, "success")
      return web.json_response({"success": True, "message": "Đã cập nhật dataset thành công"})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_get_activity_logs(self, request):
    # Return the last 50 activities, reversed (newest first)
    return web.json_response({"data": list(reversed(self.activity_logs[-50:]))})

  async def handle_get_terminal_logs(self, request):
    """Log thật từ stdout/stderr (như terminal)."""
    try:
      lines = get_terminal_log_lines()
      return web.json_response({"data": lines})
    except Exception as e:
      return web.json_response({"data": [], "detail": str(e)})

  def log_activity(self, action: str, model: str = "-", status: str = "success"):
    entry = {
      "time": time.strftime('%H:%M %p'),
      "action": action,
      "model": model,
      "status": status
    }
    self.activity_logs.append(entry)
    # Keep only last 100 logs in memory
    if len(self.activity_logs) > 100:
      self.activity_logs.pop(0)

  async def _run_training_job_distributed(self, job: Dict) -> None:
    """Training chia tai (mac dinh): dung pipeline Node + Ring All-Reduce, 1 node hoac nhieu node."""
    job_id = job.get("job_id", "")
    model = job.get("model", "")
    dataset_path = (job.get("dataset") or "").strip()
    epochs = int(job.get("epochs", 3))
    max_steps = int(job.get("max_steps", 0) or 0)
    batch_size = int(job.get("batch_size", 1))
    max_length = min(int(job.get("max_length", 128)), 2048)
    min_max_length = 64
    min_chars = int(job.get("min_chars", 10))
    max_chars = int(job.get("max_chars", 100000))
    skip_dedup = job.get("skip_dedup", False)
    save_every = int(job.get("save_every", 5))
    checkpoint_dir = (job.get("checkpoint_dir") or "").strip() or str(get_checkpoints_dir(model))

    def _is_cuda_oom_error(exc: Exception) -> bool:
      msg = str(exc).lower()
      patterns = ("cuda out of memory", "cuda_oom", "outofmemoryerror", "cublas_status_alloc_failed",)
      return any(p in msg for p in patterns)

    def _clear_cuda_cache() -> None:
      try:
        import torch
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
      except Exception:
        pass

    try:
      import torch
      if torch.cuda.is_available():
        total_vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
        if total_vram_gb <= 3.0:
          if batch_size > 1:
            batch_size = 1
          if max_length > 128:
            max_length = 128
    except Exception:
      pass
    job["effective_batch_size"] = batch_size
    job["effective_max_length"] = max_length
    job["oom_retries"] = 0
    try:
      path = self._resolve_dataset_path(dataset_path)
      if path is None:
        job["status"] = "failed"
        job["error"] = f"Dataset khong ton tai hoac khong hop le: {dataset_path}"
        self.log_activity("Training Failed", model, "failed")
        return
      raw_data = await load_raw_data_async(path)
      if not raw_data:
        job["status"] = "failed"
        job["error"] = "Dataset rong"
        self.log_activity("Training Failed", model, "failed")
        return

      pp = await preprocess_async(raw_data, format_type="auto", min_chars=min_chars, max_chars=max_chars, dedup=not skip_dedup,)
      raw_data = pp["processed_samples"]
      job["preprocess"] = {
        "original_count": pp["original_count"],
        "empty_removed": pp["empty_removed"],
        "short_removed": pp["short_removed"],
        "duplicate_removed": pp["duplicate_removed"],
        "validation_errors": pp["validation_errors"],
        "valid_count": pp["valid_count"],
        "format_detected": pp["format_detected"],
      }
      if not raw_data:
        job["status"] = "failed"
        job["error"] = "Khong co mau hop le sau tien xu ly"
        self.log_activity("Training Failed", model, "failed")
        return

      # Logic Training LLM (Causal LM)
      base_shard = build_base_shard(model, self.inference_engine_classname)
      if base_shard is None:
        job["status"] = "failed"
        job["error"] = f"Model không hỗ trợ: {model}"
        self.log_activity("Training Failed", model, "failed")
        return
      await self.node.inference_engine.ensure_shard(base_shard)
      tokenizer = getattr(self.node.inference_engine, "tokenizer", None)
      if tokenizer is None:
        job["status"] = "failed"
        job["error"] = "Không lấy được tokenizer từ engine"
        self.log_activity("Training Failed", model, "failed")
        return
      if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
      pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
      texts = []
      for sample in raw_data:
        try:
          t = format_sample_to_text(sample, tokenizer)
          if t and t.strip():
            texts.append(t)
        except Exception:
          continue
      if not texts:
        job["status"] = "failed"
        job["error"] = "Không có mẫu hợp lệ sau khi format"
        self.log_activity("Training Failed", model, "failed")
        return

      def tokenize_batch(batch_texts: List[str], current_max_length: int):
        # Dynamic padding theo batch giảm VRAM đáng kể so với padding="max_length" toàn cục.
        enc = tokenizer(
          batch_texts,
          truncation=True,
          max_length=current_max_length,
          padding=True,
          return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
          attention_mask = (input_ids != pad_id).astype(np.int64)
        else:
          attention_mask = attention_mask.astype(np.int64)
        labels = np.where(attention_mask > 0, input_ids, -100).astype(np.int64)
        lengths = np.maximum(attention_mask.sum(axis=1), 1).astype(np.int64)
        return input_ids, labels, lengths

      planned_steps = (len(texts) + batch_size - 1) // batch_size * epochs
      if max_steps > 0:
        total_steps = max(1, min(planned_steps, max_steps))
      else:
        total_steps = max(1, planned_steps)
      step_count = 0
      effective_max_length = max_length
      world_size = len(getattr(self.node, "peers", [])) + 1
      # LACP DistributedPipelineExtension is exposed via the real
      # `distributed_pipeline` property (lazy-initialized); never read the
      # non-existent `_pipeline_extension` attribute, which silently disabled
      # multi-node 1F1B training and fell back to single-node enqueue.
      ext = getattr(self.node, "distributed_pipeline", None)
      lr_val = float((job.get("learning_rate") or "1e-5").strip()) if job.get("learning_rate") else 1e-5
      save_every = int(job.get("save_every", 5))
      checkpoint_dir = (job.get("checkpoint_dir") or "").strip() or str(get_checkpoints_dir(model))
      # LoRA: every node trains adapters on the layers it serves (no
      # whole-model load); the settings travel with each pipeline step.
      from synapse.training.stage_lora import normalize_config
      lora_cfg = normalize_config(job) if job.get("use_lora") else None
      if lora_cfg is not None and ext is None:
        raise RuntimeError("LoRA needs the pipeline extension (distributed_pipeline)")
      job["lora_applied"] = lora_cfg is not None

      for epoch in range(epochs):
        if self._training_job is None or self._training_job.get("job_id") != job_id:
          return
        if max_steps > 0 and step_count >= max_steps:
          break
        indices = np.random.permutation(len(texts)) if epoch > 0 else np.arange(len(texts))
        for start in range(0, len(indices), batch_size):
          if self._training_job is None or self._training_job.get("job_id") != job_id:
            return
          if max_steps > 0 and step_count >= max_steps:
            break
          batch_idx = indices[start : start + batch_size]
          pending_batches = [[texts[i] for i in batch_idx]]
          while pending_batches:
            if max_steps > 0 and step_count >= max_steps:
              break
            current_batch_texts = pending_batches.pop(0)
            if not current_batch_texts:
              continue

            try:
              example_np, target_np, length_np = tokenize_batch(current_batch_texts, effective_max_length)

              if ext is not None and (world_size > 1 or lora_cfg is not None):
                import uuid
                loss = await ext.train_step_distributed(
                  base_shard, example_np, target_np,
                  request_id=str(uuid.uuid4()),
                  learning_rate=lr_val,
                  schedule="1f1b",
                  **({"lora": lora_cfg} if lora_cfg else {}),
                )
              else:
                loss = await self.node.enqueue_example(
                  base_shard, example_np, target_np, length_np,
                  request_id=str(uuid.uuid4()), train=True,
                )
              if loss is None:
                raise RuntimeError("Training step trả về loss=None")
              if isinstance(loss, (int, float)):
                job["loss"] = round(float(loss), 4)
              step_count += 1
              job["current_step"] = step_count
              job["current_epoch"] = epoch
              job["progress"] = min(100, int(100 * step_count / total_steps))
              # Nhường event loop để endpoint status vẫn phản hồi được trong lúc train.
              await asyncio.sleep(0)
            except Exception as e:
              if not _is_cuda_oom_error(e):
                if DEBUG >= 1:
                  traceback.print_exc()
                job["status"] = "failed"
                job["error"] = str(e)
                self.log_activity("Training Failed", model, "failed")
                return

              job["oom_retries"] = int(job.get("oom_retries", 0)) + 1
              _clear_cuda_cache()

              if effective_max_length > min_max_length:
                prev_len = effective_max_length
                effective_max_length = max(min_max_length, effective_max_length // 2)
                job["effective_max_length"] = effective_max_length
                pending_batches.insert(0, current_batch_texts)
                if DEBUG >= 1:
                  print(
                    f"[training:{job_id}] CUDA OOM -> giảm max_length {prev_len} -> {effective_max_length} và retry"
                  )
                continue

              if len(current_batch_texts) > 1:
                mid = max(1, len(current_batch_texts) // 2)
                left = current_batch_texts[:mid]
                right = current_batch_texts[mid:]
                if right:
                  pending_batches.insert(0, right)
                if left:
                  pending_batches.insert(0, left)
                if DEBUG >= 1:
                  print(
                    f"[training:{job_id}] CUDA OOM ở max_length={effective_max_length} -> tách batch {len(current_batch_texts)} thành {len(left)} + {len(right)}"
                  )
                continue

              if DEBUG >= 1:
                traceback.print_exc()
              job["status"] = "failed"
              job["error"] = (
                "CUDA OOM dù đã giảm max_length và batch tối thiểu. "
                "Hãy chọn model nhỏ hơn hoặc giảm max_length trong Training settings."
              )
              self.log_activity("Training Failed", model, "failed")
              return

        if save_every > 0 and (epoch + 1) % save_every == 0:
          try:
            await self.node.coordinate_save(base_shard, epoch + 1, str(checkpoint_dir))
          except Exception:
            pass

      if self._training_job and self._training_job.get("job_id") == job_id:
        job["status"] = "completed"
        job["progress"] = 100
        job["current_epoch"] = epochs
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log_activity("Training Completed", model, "success")
    except asyncio.CancelledError:
      if self._training_job and self._training_job.get("job_id") == job_id:
        job["status"] = "cancelled"
        self.log_activity("Training Cancelled", model, "cancelled")
    except Exception as e:
      if self._training_job and self._training_job.get("job_id") == job_id:
        job["status"] = "failed"
        job["error"] = str(e)
        self.log_activity("Training Failed", model, "failed")
      if DEBUG >= 2:
        traceback.print_exc()

  async def _run_training_job(self, job: Dict) -> None:
    """Training chia tải (pipeline Node). Không còn LoRA 1 node."""
    try:
      await self._run_training_job_distributed(job)
    except asyncio.CancelledError:
      model = job.get("model", "")
      if self._training_job and self._training_job.get("job_id") == job.get("job_id"):
        self._training_job["status"] = "cancelled"
        self.log_activity("Training Cancelled", model, "cancelled")
    except Exception as e:
      model = job.get("model", "")
      if self._training_job and self._training_job.get("job_id") == job.get("job_id"):
        self._training_job["status"] = "failed"
        self._training_job["error"] = str(e)
        self.log_activity("Training Failed", model, "failed")

  async def _run_training_job_deferred(self, job: Dict) -> None:
    # Cho phép HTTP response /v1/training/start được flush trước khi bắt đầu train nặng.
    await asyncio.sleep(0.05)
    await self._run_training_job(job)

  async def handle_post_training(self, request):
    try:
      body = await request.json() if request.can_read_body else {}
      model = (body.get("model") or "").strip()
      dataset = (body.get("dataset") or body.get("dataset_path") or "").strip()
      epochs = int(body.get("epochs", 10))
      batch_size = max(1, int(body.get("batch_size", 1)))
      learning_rate = (body.get("learning_rate") or "").strip() or None
      save_every = int(body.get("save_every", 5))
      max_steps = int(body.get("max_steps", 0))
      warmup_steps = int(body.get("warmup_steps", 0))
      max_length = min(int(body.get("max_length", 128)), 2048)
      output_model_name = (body.get("output_model_name") or body.get("output_name") or "").strip() or None
      checkpoint_dir = (body.get("checkpoint_dir") or "").strip() or None
      system_prompt = (body.get("system_prompt") or body.get("system") or "").strip() or None
      min_chars = int(body.get("min_chars", 10))
      max_chars = int(body.get("max_chars", 100000))
      skip_dedup = body.get("skip_dedup", False)
      use_lora = body.get("use_lora", False)
      lora_r = int(body.get("lora_r", 8))
      lora_alpha = int(body.get("lora_alpha", 16))
      lora_dropout = float(body.get("lora_dropout", 0.05))
      lora_target_modules = body.get("lora_target_modules") or "q_proj,v_proj,k_proj,o_proj"
      if not model:
        return web.json_response({"success": False, "message": "Thieu model"}, status=400)
      if not dataset:
        return web.json_response({"success": False, "message": "Thieu dataset (path)"}, status=400)
      job_id = str(uuid.uuid4())
      self._training_job = {
        "job_id": job_id,
        "model": model,
        "dataset": dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_every": save_every,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "max_length": max_length,
        "output_model_name": output_model_name,
        "checkpoint_dir": checkpoint_dir,
        "system_prompt": system_prompt,
        "base_model_id": model,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "skip_dedup": skip_dedup,
        "use_lora": use_lora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "status": "started",
        "progress": 0,
        "current_epoch": 0,
        "current_step": 0,
        "loss": None,
        "error": None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
      }
      self.log_activity("Training Started", model, "started")
      asyncio.create_task(self._run_training_job_deferred(self._training_job))
      return web.json_response({
        "success": True,
        "status": "started",
        "job_id": job_id,
        "message": "Training chia tải (pipeline) đã bắt đầu. Trang Training sẽ cập nhật tiến trình.",
      })
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_get_training_status(self, request):
    if self._training_job is None:
      return web.json_response({"status": "idle", "job": None})
    return web.json_response({"status": self._training_job.get("status", "idle"), "job": self._training_job})

  async def handle_get_cluster_resources(self, request):
    """
    Tổng hợp tài nguyên + layer sharding của tất cả node trong cluster.
    Dùng cho trang Cluster UI để trực quan hóa LACP 2.0 / Discrete Sharding.
    Trả về: danh sách node với chip, TFLOPS, layer_start/end indices, share_pct, cpu_pct, gpu_util.
    """
    try:
      topology = self.node.current_topology
      partitions = []
      
      # Lấy số layer thực tế từ model đang chạy hoặc fallback 32
      total_layers = 32
      try:
        if hasattr(self.node.inference_engine, "num_layers") and self.node.inference_engine.num_layers:
          total_layers = self.node.inference_engine.num_layers
      except Exception:
        pass

      strategy_name = "LACP (Multi-Resource + Latency Clustering)"
      if topology and self.node.topology.nodes:
        try:
          partitions = self.node._compute_lacp_partition()
        except Exception:
          partitions = []

      # Lấy tailscale nodes để có tên thân thiện
      try:
        ts_nodes = await self.node.get_tailscale_nodes()
      except Exception:
        ts_nodes = []
      total_tailscale_devices = len(ts_nodes)
      ts_map = {n.get("device_id", ""): n for n in ts_nodes if n.get("device_id")}

      # Lấy metrics máy này
      try:
        cpu_pct_self = psutil.cpu_percent()
        ram_self = psutil.virtual_memory()
        ram_used_self = round(ram_self.used / (1024**3), 2)
        ram_total_self = round(ram_self.total / (1024**3), 2)
      except Exception:
        cpu_pct_self = 0
        ram_used_self = 0
        ram_total_self = 0
      gpu_util_self = self._get_gpu_utilization()

      # Tổng TFLOPS
      nodes_raw = list(topology.all_nodes()) if topology else []
      total_flops = sum(n[1].flops.fp16 for n in nodes_raw) if nodes_raw else 0

      result = []
      for partition in partitions:
        node_id = partition.node_id
        caps = topology.get_node(node_id) if topology else None
        is_self = (node_id == self.node.id)

        ts_info = ts_map.get(node_id, {})
        display_name = ts_info.get("name") or ("Máy này" if is_self else node_id[:12])
        
        # Hardware
        chip = caps.chip if caps else "Unknown"
        flops_fp16 = caps.flops.fp16 if caps else 0
        vram_gb = round(caps.memory / 1024, 1) if caps and caps.memory else 0
        ram_gb = round(caps.system_ram_mb / 1024, 1) if caps and caps.system_ram_mb else 0
        gpu_backend = caps.gpu_backend if caps else "Unknown"
        has_gpu = vram_gb > 0 and gpu_backend not in ("CPU (x86)", "CPU (ARM)", "Unknown")

        # Layer range: Chuyển đổi từ float range [0.0, 1.0] sang discrete indices [0, total_layers-1]
        start_layer = int(partition.start * total_layers)
        end_layer = max(start_layer, int(partition.end * total_layers) - 1)
        # Ensure last partition covers everything
        if partition == partitions[-1]:
          end_layer = total_layers - 1
        layer_count = (end_layer - start_layer + 1)
        
        share_pct = round((partition.end - partition.start) * 100, 1)
        flops_share_pct = round(flops_fp16 / total_flops * 100, 1) if total_flops > 0 else share_pct

        # Real-time metrics (chỉ máy này)
        cpu_pct = cpu_pct_self if is_self else None
        gpu_util = gpu_util_self if is_self else None
        ram_used = ram_used_self if is_self else None

        node_ip = ts_info.get("addresses", [None])[0] or node_id
        result.append({
          "node_id": node_id,
          "name": display_name,
          "is_self": is_self,
          "ip": node_ip,
          "chip": chip,
          "gpu_backend": gpu_backend,
          "has_gpu": has_gpu,
          "vram_gb": vram_gb,
          "ram_gb": ram_gb,
          "flops_fp16": round(flops_fp16, 2),
          "layer_start": start_layer,
          "layer_end": end_layer,
          "layer_count": layer_count,
          "layer_start_pct": round(start_layer / total_layers * 100, 1),
          "layer_end_pct": round((end_layer + 1) / total_layers * 100, 1),
          "share_pct": share_pct,
          "flops_share_pct": flops_share_pct,
          "cpu_cores": caps.cpu_cores if caps else 0,
          "disk_gb": caps.disk_gb if caps else 0,
          "cpu_pct": cpu_pct,
          "gpu_utilization": gpu_util,
          "ram_used_gb": ram_used,
          "ram_total_gb": ram_gb,
        })

      # Fallback nếu 1 node
      if not result:
        try:
          result.append({
            "node_id": self.node.id,
            "name": "Máy này",
            "is_self": True,
            "ip": "127.0.0.1",
            "chip": self.node.device_capabilities.chip if self.node.device_capabilities else "Unknown",
            "gpu_backend": self.node.device_capabilities.gpu_backend if self.node.device_capabilities else "Unknown",
            "has_gpu": True, # Probable
            "vram_gb": round(self.node.device_capabilities.memory / 1024, 1) if self.node.device_capabilities else 0,
            "ram_gb": round(self.node.device_capabilities.system_ram_mb / 1024, 1) if self.node.device_capabilities else 0,
            "flops_fp16": round(self.node.device_capabilities.flops.fp16, 2) if self.node.device_capabilities else 0,
            "layer_start": 0,
            "layer_end": total_layers - 1,
            "layer_count": total_layers,
            "layer_start_pct": 0.0,
            "layer_end_pct": 100.0,
            "share_pct": 100.0,
            "flops_share_pct": 100.0,
            "cpu_cores": self.node.device_capabilities.cpu_cores if self.node.device_capabilities else 0,
            "disk_gb": self.node.device_capabilities.disk_gb if self.node.device_capabilities else 0,
            "cpu_pct": cpu_pct_self,
            "gpu_utilization": self._get_gpu_utilization(),
            "ram_used_gb": ram_used_self,
            "ram_total_gb": ram_total_self,
          })
        except Exception: pass

      # Peer graph từ topology
      peer_graph = {}
      if topology:
        try:
          peer_graph = {
            nid: [{
              "from_id": c.from_id,
              "to_id": c.to_id,
              "description": c.description,
            } for c in conns]
            for nid, conns in topology.peer_graph.items()
          }
        except Exception:
          pass

      # Lấy latency matrix
      latency_matrix = {}
      if hasattr(self.node, "_latency_cache"):
        latency_matrix = self.node._latency_cache

      return web.json_response({
        "nodes": result,
        "total_nodes": len(result),
        "total_tailscale_devices": total_tailscale_devices,
        "total_layers": total_layers,
        "strategy_name": strategy_name,
        "peer_graph": peer_graph,
        "latency_matrix": latency_matrix,
        "training_job": self._training_job,
      })
    except Exception:
      if DEBUG >= 1: traceback.print_exc()
      return web.json_response({"nodes": [], "total_nodes": 0, "total_layers": 32, "strategy_name": "Error"}, status=500)


  async def handle_tokens(self, request_id: str, tokens: List[int], is_finished: bool):
    token_len = len(tokens) if isinstance(tokens, (list, tuple)) else "?"
    if DEBUG >= 1:
      print(f"[handle_tokens] request_id={request_id[:8]}..., len(tokens)={token_len}, is_finished={is_finished}, queue_exists={request_id in self.token_queues}")
    if request_id not in self.token_queues:
      if DEBUG >= 3:
        print(f"[DEBUG] Queue not found for {request_id[:8]}... (token from non-origin node)")
      return
    await self.token_queues[request_id].put((tokens, is_finished))

  async def run(self, host: str = "0.0.0.0", port: int = 52415):

    runner = None
    try:
      runner = web.AppRunner(self.app)
      await runner.setup()
      site = web.TCPSite(runner, host, port)
      try:
        await site.start()
      except OSError as e:
        from synapse.helpers import check_ports_free, format_port_conflicts
        conflicts = check_ports_free({"api": port})
        msg = format_port_conflicts(conflicts) if conflicts else (
          "Cannot bind API port %s: %s" % (port, e))
        print(msg)
        try:
          await runner.cleanup()
        except Exception:
          pass
        raise RuntimeError(msg) from e
      display_host = "127.0.0.1" if host == "0.0.0.0" else host

      print("[INFO] Available endpoints:")
      print(f"[INFO]   - POST http://{display_host}:{port}/v1/chat/completions")
      print(f"[INFO]   - GET  http://{display_host}:{port}/v1/models")
      print(f"[INFO]   - GET  http://{display_host}:{port}/ (status page)")
      try:
        _, tailscale_ips = await get_self_tailscale_info()
        if tailscale_ips:
          print("[INFO]   (Tailscale - máy khác truy cập):")
          for ip in tailscale_ips:
            print(f"[INFO]   - POST http://{ip}:{port}/v1/chat/completions")
            print(f"[INFO]   - GET  http://{ip}:{port}/v1/models")
            print(f"[INFO]   - GET  http://{ip}:{port}/ (status page)")
        else:
          lan_ips = [ip for ip, _ in get_all_ip_addresses_and_interfaces() if ip and ip not in ("127.0.0.1", "::1", "localhost")]
          if lan_ips:
            print("[INFO]   (Tailscale chưa bật — máy khác truy cập qua LAN):")
            for ip in lan_ips:
              print(f"[INFO]   - GET  http://{ip}:{port}/ (status page)")
          else:
            print(f"[INFO]   (Tailscale): chưa bật — máy khác dùng IP LAN của máy này, port {port}")
      except Exception:
        lan_ips = [ip for ip, _ in get_all_ip_addresses_and_interfaces() if ip and ip not in ("127.0.0.1", "::1", "localhost")]
        if lan_ips:
          print(f"[INFO]   (Máy khác truy cập qua LAN): GET http://{lan_ips[0]}:{port}/ (status page)")
        else:
          print(f"[INFO]   (Máy khác truy cập): dùng IP LAN máy này, port {port}")
      
      # Giữ server chạy cho đến khi bị cancel
      try:
        print("DEBUG: API Server entering wait loop...")
        await asyncio.Event().wait()  # Chờ vô hạn cho đến khi task bị cancel
        print("DEBUG: API Server wait loop finished (unexpectedly)!")
      except asyncio.CancelledError:
        print("[INFO] API server shutdown requested")
        raise  # Re-raise để cleanup được thực hiện
    except asyncio.CancelledError:
      # Normal shutdown
      print("[INFO] API server shutting down...")
    except Exception as e:
      print(f"[ERROR] ERROR starting API server: {e}")
      import traceback
      traceback.print_exc()
      raise
    finally:
      # Cleanup runner
      if runner:
        try:
          await runner.cleanup()
          print("[INFO] API server cleanup completed")
        except Exception as e:
          print(f"[WARNING] Error during runner cleanup: {e}")
