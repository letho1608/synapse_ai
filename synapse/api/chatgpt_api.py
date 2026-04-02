import re
import subprocess
import uuid
import time
import asyncio
import json
import os
import psutil
from pathlib import Path
from transformers import AutoTokenizer
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
from synapse.networking.tailscale.tailscale_helpers import get_synapse_api_urls_from_node_list, get_self_tailscale_info
from synapse.orchestration import Node
from synapse.model_list import HF_MODELS, HF_MODEL_PARAMS, resolve_hf_id  # Danh sách + metadata cho web UI
from typing import Callable, Optional
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from synapse.loading import RepoProgressEvent
import tempfile
from synapse.apputil import create_animation_mp4
from collections import defaultdict
from synapse.training.finetune_lora import load_raw_data, format_sample_to_text
from synapse.models import build_base_shard, get_repo
from synapse.inference.tokenizers import resolve_tokenizer
from synapse.api.function_calling import build_tool_instructions, parse_tool_call, parse_tool_calls, run_tool_and_format, run_tools_parallel
from synapse.api.tools.intent_trigger import detect_tool_intent
from synapse.api.tools.format_vi import format_tool_result_vi


class Message:
  def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]], tools: Optional[List[Dict]] = None):
    self.role = role
    self.content = content
    self.tools = tools

  def to_dict(self):
    data = {"role": self.role, "content": self.content}
    if self.tools:
      data["tools"] = self.tools
    return data


class ChatCompletionRequest:
  def __init__(self, model: str, messages: List[Message], temperature: float, tools: Optional[List[Dict]] = None):
    self.model = model
    self.messages = messages
    self.temperature = temperature
    self.tools = tools

  def to_dict(self):
    return {"model": self.model, "messages": [message.to_dict() for message in self.messages], "temperature": self.temperature, "tools": self.tools}


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


def remap_messages(messages: List[Message]) -> List[Message]:
  remapped_messages = []
  last_image = None
  for message in messages:
    if not isinstance(message.content, list):
      remapped_messages.append(message)
      continue

    remapped_content = []
    for content in message.content:
      if isinstance(content, dict):
        if content.get("type") in ["image_url", "image"]:
          image_url = content.get("image_url", {}).get("url") or content.get("image")
          if image_url:
            last_image = {"type": "image", "image": image_url}
            remapped_content.append({"type": "text", "text": "[An image was uploaded but is not displayed here]"})
        else:
          remapped_content.append(content)
      else:
        remapped_content.append(content)
    remapped_messages.append(Message(role=message.role, content=remapped_content))

  if last_image:
    # Replace the last image placeholder with the actual image content
    for message in reversed(remapped_messages):
      for i, content in enumerate(message.content):
        if isinstance(content, dict):
          if content.get("type") == "text" and content.get("text") == "[An image was uploaded but is not displayed here]":
            message.content[i] = last_image
            return remapped_messages

  return remapped_messages


def build_prompt(tokenizer, _messages: List[Message], tools: Optional[List[Dict]] = None):
  messages = remap_messages(_messages)
  
  # Check if tokenizer has chat template (GPT-2 doesn't have one)
  if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
    # For models without chat template (like GPT-2), concatenate messages
    prompt_parts = []
    for msg in messages:
      content = msg.content if msg.content is not None else ""
      if msg.role == "user":
        prompt_parts.append(content)
      elif msg.role == "assistant":
        prompt_parts.append(content)
      elif msg.role == "system":
        prompt_parts.append(content)
    return "\n".join(prompt_parts)
  
  chat_template_args = {"conversation": [m.to_dict() for m in messages], "tokenize": False, "add_generation_prompt": True}
  if tools: 
    chat_template_args["tools"] = tools

  try:
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt: {prompt}")
    return prompt
  except (ValueError, AttributeError):
    # If chat template fails, fallback to simple concatenation
    prompt_parts = []
    for msg in messages:
      content = msg.content if msg.content is not None else ""
      if msg.role == "user":
        prompt_parts.append(content)
      elif msg.role == "assistant":
        prompt_parts.append(content)
      elif msg.role == "system":
        prompt_parts.append(content)
    return "\n".join(prompt_parts)
  except UnicodeEncodeError:
    # Handle Unicode encoding by ensuring everything is UTF-8
    chat_template_args["conversation"] = [
      {k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v 
       for k, v in m.to_dict().items()}
      for m in messages
    ]
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt (UTF-8 encoded): {prompt}")
    return prompt


def parse_message(data: dict):
  if "role" not in data or "content" not in data:
    raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
  return Message(data["role"], data["content"], data.get("tools"))


def parse_chat_request(data: dict, default_model: str):
  return ChatCompletionRequest(
    data.get("model", default_model),
    [parse_message(msg) for msg in data["messages"]],
    data.get("temperature", 0.0),
    data.get("tools", None),
  )


class PromptSession:
  def __init__(self, request_id: str, timestamp: int, prompt: str):
    self.request_id = request_id
    self.timestamp = timestamp
    self.prompt = prompt


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
    self.app = web.Application(client_max_size=100*1024*1024)  # 100MB to support image upload
    self.prompts: PrefixDict[str, PromptSession] = PrefixDict()
    self.prev_token_lens: Dict[str, int] = {}
    self.stream_tasks: Dict[str, asyncio.Task] = {}
    self.default_model = default_model or "qwen2.5:1.5b"
    self.token_queues = defaultdict(asyncio.Queue)
    self.activity_logs = [] # In-memory activity logs
    self.log_activity("System Started", "-", "success")
    self.chatgpt_api_port = chatgpt_api_port
    self._training_job: Optional[Dict] = None  # job_id, model, dataset, epochs, batch_size, status, progress, error
    self._data_dir = Path("synapse/data")  # Chỉ dataset (json/jsonl)
    self._settings_path = Path("synapse/config/settings.json")  # Cấu hình (tách khỏi data)
    self._load_settings()

    # Get the callback system and register our handler
    self.token_callback = node.on_token.register("chatgpt-api-token-handler")
    def safe_handle_tokens(_request_id, tokens, is_finished):
      """Safely handle tokens with error handling"""
      async def _handle():
        try:
          await self.handle_tokens(_request_id, tokens, is_finished)
        except Exception as e:
          if DEBUG >= 1:
            print(f"[ChatGPTAPI] Error in handle_tokens for {_request_id}: {e}")
            import traceback
            traceback.print_exc()
      # Create task and ensure it runs
      task = asyncio.create_task(_handle())
      # Store task to prevent garbage collection
      if not hasattr(self, '_token_tasks'):
        self._token_tasks = set()
      self._token_tasks.add(task)
      task.add_done_callback(self._token_tasks.discard)
    self.token_callback.on_next(safe_handle_tokens)
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
    cors.add(self.app.router.add_post("/v1/image/generations", self.handle_post_image_generations), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/download/progress", self.handle_get_download_progress), {"*": cors_options})
    cors.add(self.app.router.add_get("/modelpool", self.handle_model_support), {"*": cors_options})
    cors.add(self.app.router.add_get("/healthcheck", self.handle_healthcheck), {"*": cors_options})
    cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})
    cors.add(self.app.router.add_delete("/models/{model_name}", self.handle_delete_model), {"*": cors_options})
    cors.add(self.app.router.add_get("/initial_models", self.handle_get_initial_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/create_animation", self.handle_create_animation), {"*": cors_options})
    cors.add(self.app.router.add_post("/download", self.handle_post_download), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_get("/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/distributed/status", self.handle_get_distributed_status), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/training/distributed/start", self.handle_post_training_distributed), {"*": cors_options})
    
    # New endpoints for Dashboard
    cors.add(self.app.router.add_get("/v1/system/stats", self.handle_get_system_stats), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/system/info", self.handle_get_system_info), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/tailscale/nodes", self.handle_get_tailscale_nodes), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets", self.handle_get_datasets), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/upload", self.handle_post_datasets_upload), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/delete", self.handle_post_datasets_delete), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets/preview", self.handle_get_datasets_preview), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets/download", self.handle_get_datasets_download), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/create", self.handle_post_datasets_create), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/datasets/rename", self.handle_post_datasets_rename), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/datasets/validate", self.handle_get_datasets_validate), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/activity", self.handle_get_activity_logs), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/logs/terminal", self.handle_get_terminal_logs), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/training/start", self.handle_post_training), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/training/status", self.handle_get_training_status), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models/status", self.handle_get_models_status), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models/list", self.handle_get_models_list), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/models/pull", self.handle_post_models_pull), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/models/delete", self.handle_post_models_delete), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/settings", self.handle_get_settings), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/settings", self.handle_post_settings), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/settings/save", self.handle_post_settings), {"*": cors_options})
    # Static: PHẢI dùng prefix /_static/ (không dùng "/") vì add_static("/") match mọi path và chỉ cho GET/HEAD -> POST sẽ 405
    if "__compiled__" not in globals():
      self.static_dir = Path(__file__).parent.parent/"tinychat"
      self.app.router.add_get("/", self.handle_root)
      self.app.router.add_static("/_static/", self.static_dir, name="static")
      
    self.app.middlewares.append(self.timeout_middleware)
    self.app.middlewares.append(self.log_request)

  def _load_settings(self) -> None:
    """Load settings từ file (synapse/config/settings.json). Nếu có file cũ synapse/data/settings.json thì migrate sang config."""
    try:
      old_path = self._data_dir / "settings.json"
      if self._settings_path.is_file():
        with open(self._settings_path, "r", encoding="utf-8") as f:
          data = json.load(f)
        if isinstance(data.get("default_model"), str) and data["default_model"].strip():
          self.default_model = data["default_model"].strip()
      elif old_path.is_file():
        with open(old_path, "r", encoding="utf-8") as f:
          data = json.load(f)
        if isinstance(data.get("default_model"), str) and data["default_model"].strip():
          self.default_model = data["default_model"].strip()
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._settings_path, "w", encoding="utf-8") as f:
          json.dump({"default_model": self.default_model}, f, ensure_ascii=False, indent=2)
        try:
          old_path.unlink()
        except Exception:
          pass
    except Exception:
      pass

  def _save_settings(self) -> None:
    """Lưu settings hiện tại ra synapse/config/settings.json."""
    try:
      self._settings_path.parent.mkdir(parents=True, exist_ok=True)
      data = {"default_model": self.default_model}
      with open(self._settings_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
      if DEBUG >= 1:
        print(f"[ChatGPTAPI] _save_settings error: {e}")

  def _get_model_cards(self) -> Dict[str, Dict]:
    """Danh sách model hợp lệ: từ HF_MODELS (download + web UI)."""
    return {m: {} for m in HF_MODELS}

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
          print(f"[ChatGPTAPI] [WARN] Not Found: {request.method} {request.path}")
        return web.json_response({"detail": "Not Found", "path": request.path}, status=404)
      except HTTPMethodNotAllowed:
        if DEBUG >= 1:
          print(f"[ChatGPTAPI] [WARN] Method Not Allowed: {request.method} {request.path}")
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
      # Hide polling requests (download/progress, topology) to reduce noise
      is_polling_request = (
        request.method == "GET" and (
          "/v1/download/progress" in request.path or
          "/v1/topology" in request.path or
          "/v1/tailscale/nodes" in request.path or
          "/modelpool" in request.path
        )
      )
      
      if request.method == "POST" and "/chat/completions" in request.path:
        print(f"[ChatGPTAPI] [INFO] POST request received: {request.method} {request.path} from {request.remote}")
        print(f"[ChatGPTAPI] [INFO] Request headers: {dict(request.headers)}")
        print(f"[ChatGPTAPI] [INFO] Request URL: {request.url}")
        print(f"[ChatGPTAPI] [INFO] Request can_read_body: {request.can_read_body}")
      elif DEBUG >= 3 and not is_polling_request:
        print(f"Received request: {request.method} {request.path}")
      try:
        # Only log handler call for non-polling requests or POST requests
        if not is_polling_request or (request.method == "POST" and "/chat/completions" in request.path):
          print(f"[ChatGPTAPI] [INFO] Calling handler for {request.method} {request.path}...")
        response = await handler(request)
        if request.method == "POST" and "/chat/completions" in request.path:
          print(f"[ChatGPTAPI] [INFO] POST request handled successfully, response status: {response.status if hasattr(response, 'status') else 'N/A'}")
        return response
      except HTTPNotFound:
        if DEBUG >= 1:
          print(f"[ChatGPTAPI] [WARN] Not Found: {request.method} {request.path}")
        return web.json_response({"detail": "Not Found", "path": request.path}, status=404)
      except HTTPMethodNotAllowed:
        if DEBUG >= 1:
          print(f"[ChatGPTAPI] [WARN] Method Not Allowed: {request.method} {request.path}")
        return web.json_response({"detail": f"Method Not Allowed: dùng {request.path} với method khác (GET/POST)."}, status=405)
      except Exception as e:
        print(f"[ChatGPTAPI] [ERROR] Error in middleware for {request.method} {request.path}: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"detail": f"Internal server error: {str(e)}"}, status=500)

    return middleware

  async def _safe_write(self, response, data: bytes) -> bool:
    """
    Safely write data to response stream, handling client disconnections.
    Returns True if write was successful, False if client disconnected.
    """
    try:
      await response.write(data)
      return True
    except (ClientConnectionResetError, ConnectionResetError, OSError, TypeError) as e:
      if DEBUG >= 2: print(f"Client disconnected during write: {e}")
      return False
    except Exception as e:
      if DEBUG >= 2: print(f"Unexpected error during write: {e}")
      return False

  async def handle_root(self, request):
    # Serve new production dashboard (no-cache để luôn lấy bản mới, tránh hiển thị usage tiếng Anh từ cache)
    dashboard_path = self.static_dir / "dashboard.html"
    if dashboard_path.exists():
      resp = web.FileResponse(dashboard_path)
      resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
      resp.headers["Pragma"] = "no-cache"
      resp.headers["Expires"] = "0"
      return resp
    # Fallback to old index.html if dashboard doesn't exist
    return web.FileResponse(self.static_dir / "index.html")

  def _get_gpu_info(self) -> str:
    """Lấy thông tin GPU từ device_capabilities (hỗ trợ NVIDIA, AMD, Apple, Intel, Ascend)."""
    try:
      caps = getattr(self.node, "device_capabilities", None)
      if caps is not None:
        # Dùng SystemSpecs nếu có
        specs = getattr(caps, "_specs", None)
        if specs is None:
          try:
            from synapse.topology.device_capabilities import SystemSpecs
            specs = SystemSpecs.detect()
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
    """Lấy % sử dụng GPU (nvidia-smi utilization.gpu). Trả về None nếu không có GPU hoặc lỗi."""
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

  async def handle_model_support(self, request):
    try:
      response = web.StreamResponse(status=200, reason='OK', headers={ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' })
      await response.prepare(request)
      async for path, s in self.node.shard_downloader.get_shard_download_status(self.inference_engine_classname):
        model_data = { s.shard.model_id: { "downloaded": s.downloaded_bytes == s.total_bytes, "download_percentage": 100 if s.downloaded_bytes == s.total_bytes else 100 * float(s.downloaded_bytes) / float(s.total_bytes), "total_size": s.total_bytes, "total_downloaded": s.downloaded_bytes } }
        if not await self._safe_write(response, f"data: {json.dumps(model_data)}\n\n".encode()):
          # Client đã đóng connection, dừng loop
          break
      # Chỉ ghi [DONE] nếu client vẫn còn kết nối
      await self._safe_write(response, b"data: [DONE]\n\n")
      return response

    except Exception as e:
      print(f"Error in handle_model_support: {str(e)}")
      traceback.print_exc()
      return web.json_response({"detail": f"Server error: {str(e)}"}, status=500)

  async def _get_synapse_api_urls(self) -> List[str]:
    """Danh sách Synapse API URL của các node khác (để training phân tán). Loại trừ máy này."""
    self_ips: Set[str] = set()
    try:
      for ip, _ in get_all_ip_addresses_and_interfaces():
        if ip:
          self_ips.add(ip.strip())
      self_ips.add("127.0.0.1")
      self_ips.add("localhost")
    except Exception:
      pass
    try:
      nodes: List[Dict] = await self.node.get_tailscale_nodes()
      return get_synapse_api_urls_from_node_list(
        nodes,
        api_port=self.chatgpt_api_port,
        only_synapse_nodes=True,
        exclude_ips=self_ips,
      )
    except Exception as e:
      if DEBUG >= 1:
        print(f"[ChatGPTAPI] _get_synapse_api_urls: {e}")
    return []

  async def _trigger_training_on_peers(self, body: dict) -> None:
    """Gửi cùng lệnh training start tới tất cả node khác (phân tán). Chạy nền."""
    urls = await self._get_synapse_api_urls()
    if not urls:
      return
    import aiohttp
    async def post_one(base_url: str) -> None:
      url = f"{base_url.rstrip('/')}/v1/training/start"
      try:
        async with aiohttp.ClientSession() as session:
          async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status in (200, 201) and DEBUG >= 1:
              print(f"[ChatGPTAPI] Training phân tán: đã gửi start tới {base_url}")
      except Exception as e:
        if DEBUG >= 1:
          print(f"[ChatGPTAPI] Training phân tán gửi tới {base_url}: {e}")
    await asyncio.gather(*[post_one(u) for u in urls], return_exceptions=True)

  async def handle_get_models(self, request):
    """Danh sách model: từ HF_MODELS (download + web UI)."""
    models_list = [{"id": name, "object": "model", "owned_by": "pytorch", "ready": True} for name in sorted(HF_MODELS)]
    return web.json_response({"object": "list", "data": models_list})

  async def handle_get_models_status(self, request):
    """Trạng thái model: danh sách từ HF_MODELS (dùng cho web UI)."""
    return web.json_response({
      "enabled": True,
      "models": sorted(HF_MODELS.keys()),
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
      for name, repo_id in HF_MODELS.items():
        if repo_id not in repo_ids_in_cache:
          continue
        try:
          snapshot_download(repo_id=repo_id, local_files_only=True)
          complete.add(name)
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
    """Danh sách cho web UI: đã tải (HF cache) + có thể tải (22 model từ HF_MODELS). Có size_gb, parameter_size, quantization_level."""
    try:
      downloaded_names = self._get_hf_downloaded_names()
      cache_sizes = self._get_hf_cache_size_by_repo()
      downloaded = []
      for n in sorted(downloaded_names):
        repo_id = HF_MODELS.get(n, "")
        size_bytes = cache_sizes.get(repo_id, 0)
        size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else None
        downloaded.append({
          "name": n,
          "status": "downloaded",
          "repo_id": repo_id,
          "size_gb": size_gb,
          "parameter_size": HF_MODEL_PARAMS.get(n, "-"),
          "quantization_level": "FP16/BF16",
        })
      available = [{"name": n, "status": "available"} for n in sorted(HF_MODELS.keys()) if n not in downloaded_names]
      return web.json_response({
        "downloaded": downloaded,
        "available": available,
      })
    except Exception as e:
      if DEBUG >= 1:
        traceback.print_exc()
      # Fallback: trả về toàn bộ 22 model ở "có thể tải" để trang quản lý luôn hiển thị danh sách
      available = [{"name": n, "status": "available"} for n in sorted(HF_MODELS.keys())]
      return web.json_response({"downloaded": [], "available": available})

  async def handle_post_models_pull(self, request):
    """Tải model: HF snapshot_download (danh sách từ HF_MODELS)."""
    try:
      data = await request.json()
      model_name = (data.get("model") or data.get("name") or "").strip()
      if not model_name:
        return web.json_response({"success": False, "message": "Thiếu model"}, status=400)
      if model_name not in HF_MODELS:
        return web.json_response({"success": False, "message": f"Model '{model_name}' không có trong danh sách."}, status=400)
      from huggingface_hub import snapshot_download
      hf_id = resolve_hf_id(model_name)
      snapshot_download(repo_id=hf_id)
      return web.json_response({
        "success": True,
        "message": f"Đã tải {model_name} -> {hf_id} (Hugging Face cache)",
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
        "message": f"Đã xóa cache HF cho {model_name}" if rev_hashes else f"Không tìm thấy cache cho {model_name}",
      })
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_post_chat_token_encode(self, request):
    """Token encode: dùng chat completions với model trong danh sách."""
    return web.json_response(
      {"error": "Dùng chat completions với model trong danh sách (Quản lý mô hình)."},
      status=400,
    )

  async def handle_get_download_progress(self, request):
    progress_data = {}
    for node_id, progress_event in self.node.node_download_progress.items():
      if isinstance(progress_event, RepoProgressEvent):
        if progress_event.status != "in_progress": continue
        progress_data[node_id] = progress_event.to_dict()
      else:
        print(f"Unknown progress event type: {type(progress_event)}. {progress_event}")
    return web.json_response(progress_data)

  async def handle_post_chat_completions(self, request):
    request_id = None
    print(f"[ChatGPTAPI] [INFO] handle_post_chat_completions called: path={request.path}, remote={request.remote}")
    print(f"[ChatGPTAPI] [INFO] Request method: {request.method}, content_type: {request.content_type}")
    try:
      print(f"[ChatGPTAPI] [INFO] Reading request JSON...")
      data = await request.json()
      print(f"[ChatGPTAPI] [INFO] Request data received: model={data.get('model')}, stream={data.get('stream')}, messages_count={len(data.get('messages', []))}")
      model_from_request = data.get("model") or self.default_model
      if model_from_request and model_from_request.startswith("gpt-"):
        model_from_request = self.default_model
      # Tương thích client cũ: bỏ prefix "ollama/" nếu có
      model_id = (model_from_request[7:] if model_from_request.startswith("ollama/") else model_from_request).strip()

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
        # Trigger tool theo từ khóa (model nhỏ thường không output TOOL_CALL đúng format)
        last_user = None
        for msg in reversed(chat_request.messages):
          if msg.role == "user":
            last_user = msg.content if isinstance(msg.content, str) else ""
            break
        if last_user:
          intent_tools = detect_tool_intent(last_user)
          if intent_tools:
            try:
              from synapse.api.tools.registry import execute_tool
              parts = []
              for tool_name, tool_args in intent_tools:
                raw = await execute_tool(tool_name, tool_args)
                text_vi = format_tool_result_vi(tool_name, raw)
                parts.append(text_vi)
              tool_result_vi = "\n".join(parts)
              inject = "\n\n[Dữ liệu thời gian thực đã lấy]\n" + tool_result_vi + "\n\nHãy trả lời ngắn gọn bằng tiếng Việt dựa trên dữ liệu trên."
              for msg in reversed(chat_request.messages):
                if msg.role == "user":
                  msg.content = (msg.content or "") + inject
                  break
            except Exception as e:
              if DEBUG >= 1:
                traceback.print_exc()
        # Inject system prompt và tool instructions
        tool_instructions = build_tool_instructions()
        has_system_message = any(msg.role == "system" for msg in chat_request.messages)
        if self.system_prompt:
          if not has_system_message:
            system_content = self.system_prompt + "\n\n" + tool_instructions
            system_message = Message(role="system", content=system_content)
            chat_request.messages.insert(0, system_message)
          else:
            for msg in chat_request.messages:
              if msg.role == "system":
                msg.content = (msg.content or "") + "\n\n" + tool_instructions
                break
        elif not has_system_message:
          system_message = Message(role="system", content=tool_instructions)
          chat_request.messages.insert(0, system_message)
        prompt = build_prompt(tokenizer, chat_request.messages, chat_request.tools)
        stream = data.get("stream", False)
        if self.on_chat_completion_request:
          self.on_chat_completion_request(request_id, chat_request, prompt)
        process_task = asyncio.create_task(
          self.node.process_prompt(base_shard, prompt, request_id=request_id)
        )
        try:
          if stream:
            # Thu thập full response trước (để kiểm tra tool call), sau đó stream
            response = web.StreamResponse(
              status=200,
              headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
            await response.prepare(request)
            all_tokens_stream: List[int] = []
            while True:
              try:
                tokens, is_finished = await asyncio.wait_for(
                  self.token_queues[request_id].get(),
                  timeout=float(self.response_timeout),
                )
              except asyncio.TimeoutError:
                if DEBUG >= 1:
                  print(f"[ChatGPTAPI] Timeout waiting for tokens: {request_id}")
                break
              all_tokens_stream.extend(tokens)
              if is_finished:
                break
            decoded_stream = tokenizer.decode(all_tokens_stream)
            tool_calls_stream = parse_tool_calls(decoded_stream)
            tokens_to_stream = all_tokens_stream
            prompt_for_stream = prompt
            if tool_calls_stream:
              if DEBUG >= 1:
                print(f"[ChatGPTAPI] Tool calls (stream, {len(tool_calls_stream)}): {[name for name, _ in tool_calls_stream]}")
              try:
                # Nếu có nhiều tools, chạy song song
                if len(tool_calls_stream) > 1:
                  result_text_s = await run_tools_parallel(tool_calls_stream)
                  result_msg_s = f"Kết quả từ {len(tool_calls_stream)} tools:\n{result_text_s}\n\nDựa vào kết quả trên trả lời ngắn gọn cho user."
                else:
                  name_s, args_s = tool_calls_stream[0]
                  result_text_s = await run_tool_and_format(name_s, args_s)
                  result_msg_s = f"Kết quả tool:\n{result_text_s}\n\nDựa vào kết quả trên trả lời ngắn gọn cho user."
                new_messages_s = list(chat_request.messages) + [
                  Message("assistant", decoded_stream),
                  Message("user", result_msg_s),
                ]
                new_prompt_s = build_prompt(tokenizer, new_messages_s, None)
                request_id_s = str(uuid.uuid4())
                self.token_queues[request_id_s] = asyncio.Queue()
                asyncio.create_task(self.node.process_prompt(base_shard, new_prompt_s, request_id=request_id_s))
                tokens_to_stream = []
                while True:
                  try:
                    tk, fin = await asyncio.wait_for(self.token_queues[request_id_s].get(), timeout=float(self.response_timeout))
                  except asyncio.TimeoutError:
                    break
                  tokens_to_stream.extend(tk)
                  if fin:
                    break
                prompt_for_stream = new_prompt_s
                if request_id_s in self.token_queues:
                  del self.token_queues[request_id_s]
              except Exception:
                pass
            for i in range(0, len(tokens_to_stream), 1):
              chunk = generate_completion(
                chat_request, tokenizer, prompt_for_stream, request_id, tokens_to_stream[i : i + 1], True, None, "chat.completion"
              )
              await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await response.write(f"data: {json.dumps(generate_completion(chat_request, tokenizer, prompt_for_stream, request_id, [], True, 'stop', 'chat.completion'))}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
          else:
            all_tokens = []
            while True:
              try:
                tokens, is_finished = await asyncio.wait_for(
                  self.token_queues[request_id].get(),
                  timeout=float(self.response_timeout),
                )
              except asyncio.TimeoutError:
                if DEBUG >= 1:
                  print(f"[ChatGPTAPI] Timeout waiting for tokens: {request_id}")
                break
              all_tokens.extend(tokens)
              if is_finished:
                break
            decoded_first = tokenizer.decode(all_tokens)
            # Thử parse multiple tool calls trước
            tool_calls = parse_tool_calls(decoded_first)
            if tool_calls:
              if DEBUG >= 1:
                print(f"[ChatGPTAPI] Tool calls ({len(tool_calls)}): {[name for name, _ in tool_calls]}")
              try:
                # Nếu có nhiều tools, chạy song song
                if len(tool_calls) > 1:
                  result_text = await run_tools_parallel(tool_calls)
                  result_msg = f"Kết quả từ {len(tool_calls)} tools:\n{result_text}\n\nDựa vào kết quả trên trả lời ngắn gọn cho user."
                else:
                  name, args = tool_calls[0]
                  result_text = await run_tool_and_format(name, args)
                  result_msg = f"Kết quả tool:\n{result_text}\n\nDựa vào kết quả trên trả lời ngắn gọn cho user."
                new_messages = list(chat_request.messages) + [
                  Message("assistant", decoded_first),
                  Message("user", result_msg),
                ]
                new_prompt = build_prompt(tokenizer, new_messages, None)
                request_id_2 = str(uuid.uuid4())
                self.token_queues[request_id_2] = asyncio.Queue()
                process_task_2 = asyncio.create_task(
                  self.node.process_prompt(base_shard, new_prompt, request_id=request_id_2)
                )
                all_tokens_2 = []
                while True:
                  try:
                    tokens_2, is_finished_2 = await asyncio.wait_for(
                      self.token_queues[request_id_2].get(),
                      timeout=float(self.response_timeout),
                    )
                  except asyncio.TimeoutError:
                    break
                  all_tokens_2.extend(tokens_2)
                  if is_finished_2:
                    break
                completion = generate_completion(
                  chat_request, tokenizer, new_prompt, request_id_2, all_tokens_2, False, "stop", "chat.completion"
                )
                if request_id_2 in self.token_queues:
                  del self.token_queues[request_id_2]
                if not process_task_2.done():
                  process_task_2.cancel()
                  try:
                    await process_task_2
                  except asyncio.CancelledError:
                    pass
                return web.json_response(completion)
              except Exception as tool_e:
                if DEBUG >= 1:
                  traceback.print_exc()
                completion = generate_completion(
                  chat_request, tokenizer, prompt, request_id, all_tokens, False, "stop", "chat.completion"
                )
                return web.json_response(completion)
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
      return web.json_response(
        {"error": {"message": "Chọn model trong danh sách (Quản lý mô hình).", "code": "model_not_found"}},
        status=400,
      )
    except Exception as outer_e:
      if DEBUG >= 1:
        print(f"[ChatGPTAPI] Unhandled exception in handle_post_chat_completions: {outer_e}")
        traceback.print_exc()
      return web.json_response(
        {"detail": f"Internal server error: {str(outer_e)}"},
        status=500,
      )
    finally:
      if request_id and request_id in self.token_queues:
        if DEBUG >= 2:
          print(f"[ChatGPTAPI] Cleaning up token queue: {request_id=}")
        del self.token_queues[request_id]

  async def handle_post_image_generations(self, request):
    """Image generation: endpoint tạm trả 501."""
    return web.json_response(
      {"error": {"message": "Image generation chưa hỗ trợ. Dùng model vision hoặc API khác."}},
      status=501,
    )

  async def handle_delete_model(self, request):
    """Xóa model: xóa khỏi HF cache (danh sách từ HF_MODELS)."""
    model_id = (request.match_info.get("model_name") or "").strip()
    if not model_id:
      return web.json_response({"detail": "model_name required"}, status=400)
    try:
      from huggingface_hub import scan_cache_dir
      hf_id = resolve_hf_id(model_id)
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
        "status": "success",
        "message": f"Đã xóa cache HF cho {model_id}" if rev_hashes else f"Không tìm thấy cache cho {model_id}",
      })
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": str(e)}, status=500)

  async def handle_get_initial_models(self, request):
    """Danh sách model: từ HF_MODELS; downloaded = có trong HF cache."""
    downloaded_names = self._get_hf_downloaded_names()
    model_data = {}
    for mid in sorted(HF_MODELS.keys()):
      model_data[mid] = {
        "name": mid,
        "downloaded": mid in downloaded_names,
        "download_percentage": 100 if mid in downloaded_names else 0,
        "total_size": None,
        "total_downloaded": None,
        "loading": False,
      }
    return web.json_response(model_data)

  async def handle_create_animation(self, request):
    try:
      data = await request.json()
      replacement_image_path = data.get("replacement_image_path")
      device_name = data.get("device_name", "Local Device")
      prompt_text = data.get("prompt", "")

      if DEBUG >= 2: print(f"Creating animation with params: replacement_image={replacement_image_path}, device={device_name}, prompt={prompt_text}")

      if not replacement_image_path:
        return web.json_response({"error": "replacement_image_path is required"}, status=400)

      # Create temp directory if it doesn't exist
      tmp_dir = Path(tempfile.gettempdir())/"synapse_animations"
      tmp_dir.mkdir(parents=True, exist_ok=True)

      # Generate unique output filename in temp directory
      output_filename = f"animation_{uuid.uuid4()}.mp4"
      output_path = str(tmp_dir/output_filename)

      if DEBUG >= 2: print(f"Animation temp directory: {tmp_dir}, output file: {output_path}, directory exists: {tmp_dir.exists()}, directory permissions: {oct(tmp_dir.stat().st_mode)[-3:]}")

      # Create the animation
      create_animation_mp4(replacement_image_path, output_path, device_name, prompt_text)

      return web.json_response({"status": "success", "output_path": output_path})

    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"error": str(e)}, status=500)

  async def handle_post_download(self, request):
    """Download model: HF snapshot_download (danh sách từ HF_MODELS)."""
    try:
      data = await request.json()
      model_name = (data.get("model") or "").strip()
      if not model_name:
        return web.json_response({"error": "model parameter is required"}, status=400)
      if model_name not in HF_MODELS:
        return web.json_response({"error": f"Model '{model_name}' không có trong danh sách."}, status=400)
      from huggingface_hub import snapshot_download
      hf_id = resolve_hf_id(model_name)
      asyncio.get_event_loop().run_in_executor(None, lambda: snapshot_download(repo_id=hf_id))
      return web.json_response({"status": "success", "message": f"Download started for model: {model_name}"})
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"error": str(e)}, status=500)

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
      if not self.node.partitioning_strategy or not self.node.topology:
        return 1
      partitions = self.node.partitioning_strategy.partition(self.node.topology)
      return len(partitions) if partitions else 1
    except Exception:
      return 1

  async def handle_get_distributed_status(self, request):
    """Trạng thái phân tán: nhiều node cùng xử lý 1 request (inference/training)."""
    try:
      n = self._get_partitions_count()
      return web.json_response({
        "partitions_count": n,
        "multi_node_inference": n > 1,
        "multi_node_training": n > 1,
        "message": "Luồng phân tán: nhiều node (partitions_count > 1).",
      })
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": str(e)}, status=500)

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

  async def handle_get_system_stats(self, request):
    try:
      cpu_percent = psutil.cpu_percent()
      memory = psutil.virtual_memory()
      uptime = time.time() - psutil.boot_time()

      # Format uptime
      days = int(uptime // (24 * 3600))
      hours = int((uptime % (24 * 3600)) // 3600)
      minutes = int((uptime % 3600) // 60)
      uptime_str = f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"

      downloaded_names = self._get_hf_downloaded_names()
      models_count = len(downloaded_names)

      # GPU info from new cross-platform hardware detection
      gpu_info_str = self._get_gpu_info()
      gpu_utilization = self._get_gpu_utilization()

      # Try to get structured GPU data from SystemSpecs
      gpu_name = None
      gpu_vram_gb = None
      unified_memory = False
      try:
        from synapse.topology.device_capabilities import SystemSpecs
        specs = SystemSpecs.detect()
        if specs.has_gpu and specs.gpu_name:
          gpu_name = specs.gpu_name
          gpu_vram_gb = round(specs.gpu_vram_gb, 1) if specs.gpu_vram_gb else None
          unified_memory = bool(specs.unified_memory)
      except Exception:
        pass

      return web.json_response({
        "cpu": cpu_percent,
        "memory": memory.percent,
        "memory_used_gb": round(memory.used / (1024**3), 2),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "uptime": uptime_str,
        "gpu": gpu_info_str,         # legacy string for old UI elements
        "gpu_name": gpu_name,        # structured — new UI
        "gpu_vram_gb": gpu_vram_gb,  # structured — new UI
        "unified_memory": unified_memory,
        "gpu_utilization": gpu_utilization,
        "models_count": models_count,
      })
    except Exception as e:
      return web.json_response({"detail": str(e)}, status=500)

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
      }
    except Exception:
      hw = {}
    return web.json_response({
      "project": "Synapse AI",
      "version": VERSION,
      "hardware": hw,
    })


  async def handle_get_settings(self, request):
    """Trả về cài đặt hiện tại (default model)."""
    return web.json_response({
      "default_model": self.default_model,
    })

  async def handle_post_settings(self, request):
    """Lưu cài đặt (default_model) và ghi file."""
    try:
      body = await request.json()
      model = (body.get("default_model") or "").strip()
      if model:
        self.default_model = model
      self._save_settings()
      self.log_activity("Cài đặt đã lưu", "-", "success")
      return web.json_response({"success": True, "default_model": self.default_model})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  async def handle_get_tailscale_nodes(self, request):
    try:
      nodes = await self.node.get_tailscale_nodes()
      return web.json_response({"data": nodes})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"detail": str(e), "data": []}, status=500)

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
    """Trả về vài dòng đầu của dataset (jsonl) hoặc preview json."""
    try:
      path_str = request.query.get("path") or request.query.get("name") or ""
      if not path_str:
        return web.json_response({"success": False, "message": "Thiếu path hoặc name"}, status=400)
      path = Path(path_str)
      if not path.is_absolute():
        path = self._data_dir / path.name
      path = path.resolve()
      data_dir = self._data_dir.resolve()
      if not str(path).startswith(str(data_dir)) or not path.is_file():
        return web.json_response({"success": False, "message": "File không tồn tại hoặc không được phép"}, status=404)
      max_lines = min(50, int(request.query.get("lines", 20)))
      lines = []
      if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
          for i, line in enumerate(f):
            if i >= max_lines:
              break
            if line.strip():
              lines.append(line.strip())
      else:
        with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
        if isinstance(data, list):
          lines = [json.dumps(x, ensure_ascii=False)[:500] for x in data[:max_lines]]
        else:
          lines = [json.dumps(data, ensure_ascii=False)[:1000]]
      return web.json_response({"success": True, "preview": lines})
    except Exception as e:
      if DEBUG >= 2:
        traceback.print_exc()
      return web.json_response({"success": False, "message": str(e)}, status=500)

  def _resolve_dataset_path(self, path_str: str, name_only_ok: bool = True) -> Optional[Path]:
    """Resolve path/name thành Path trong _data_dir. Trả về None nếu không hợp lệ."""
    path_str = (path_str or "").strip()
    if not path_str:
      return None
    path = Path(path_str)
    if not path.is_absolute():
      path = self._data_dir / path.name
    path = path.resolve()
    data_dir = self._data_dir.resolve()
    if not str(path).startswith(str(data_dir)) or path == data_dir:
      return None
    if not path.is_file() and not (name_only_ok and path.parent == data_dir):
      return None
    return path

  async def handle_get_datasets_download(self, request):
    """Tải file dataset về (attachment). Query: path hoặc name."""
    try:
      path_str = request.query.get("path") or request.query.get("name") or ""
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
    """Training chia tải (mặc định): dùng pipeline Node (enqueue_example), 1 node hoặc nhiều node."""
    job_id = job.get("job_id", "")
    model = job.get("model", "")
    dataset_path = (job.get("dataset") or "").strip()
    epochs = int(job.get("epochs", 3))
    batch_size = int(job.get("batch_size", 4))
    max_length = min(int(job.get("max_length", 512)), 2048)
    try:
      path = Path(dataset_path)
      if not path.is_absolute():
        base = self._data_dir
        if (base / path.name).exists():
          path = base / path.name
        elif (base / path).exists():
          path = base / path
      if not path.exists():
        job["status"] = "failed"
        job["error"] = f"Dataset không tồn tại: {path}"
        self.log_activity("Training Failed", model, "failed")
        return
      raw_data = load_raw_data(path)
      if not raw_data:
        job["status"] = "failed"
        job["error"] = "Dataset rỗng"
        self.log_activity("Training Failed", model, "failed")
        return
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
      def tokenize_one(text):
        enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="np")
        ids = enc["input_ids"][0]
        lab = np.array([x if x != pad_id else -100 for x in ids], dtype=np.int64)
        return ids.astype(np.int64), lab, np.array([ids.shape[0]], dtype=np.int64)
      total_steps = max(1, (len(texts) + batch_size - 1) // batch_size * epochs)
      step_count = 0
      for epoch in range(epochs):
        if self._training_job is None or self._training_job.get("job_id") != job_id:
          return
        indices = np.random.permutation(len(texts)) if epoch > 0 else np.arange(len(texts))
        for start in range(0, len(indices), batch_size):
          if self._training_job is None or self._training_job.get("job_id") != job_id:
            return
          batch_idx = indices[start : start + batch_size]
          batch_texts = [texts[i] for i in batch_idx]
          batch_ids, batch_labels, batch_lengths = [], [], []
          for t in batch_texts:
            ids, lab, ln = tokenize_one(t)
            batch_ids.append(ids)
            batch_labels.append(lab)
            batch_lengths.append(ln)
          example_np = np.stack(batch_ids).astype(np.int64)
          target_np = np.stack(batch_labels).astype(np.int64)
          length_np = np.concatenate(batch_lengths).astype(np.int64)
          try:
            loss = await self.node.enqueue_example(base_shard, example_np, target_np, length_np, request_id=str(uuid.uuid4()), train=True)
            if loss is not None and isinstance(loss, (int, float)):
              job["loss"] = round(float(loss), 4)
          except Exception as e:
            if DEBUG >= 1:
              traceback.print_exc()
            job["status"] = "failed"
            job["error"] = str(e)
            self.log_activity("Training Failed", model, "failed")
            return
          step_count += 1
          job["current_step"] = step_count
          job["current_epoch"] = epoch
          job["progress"] = min(100, int(100 * step_count / total_steps))
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

  async def handle_post_training(self, request):
    try:
      body = await request.json() if request.can_read_body else {}
      model = (body.get("model") or "").strip()
      dataset = (body.get("dataset") or body.get("dataset_path") or "").strip()
      epochs = int(body.get("epochs", 10))
      batch_size = int(body.get("batch_size", 16))
      learning_rate = (body.get("learning_rate") or "").strip() or None
      save_every = int(body.get("save_every", 5))
      max_steps = int(body.get("max_steps", 0))
      warmup_steps = int(body.get("warmup_steps", 0))
      output_model_name = (body.get("output_model_name") or body.get("output_name") or "").strip() or None
      checkpoint_dir = (body.get("checkpoint_dir") or "").strip() or None
      system_prompt = (body.get("system_prompt") or body.get("system") or "").strip() or None
      if not model:
        return web.json_response({"success": False, "message": "Thiếu model"}, status=400)
      if not dataset:
        return web.json_response({"success": False, "message": "Thiếu dataset (path)"}, status=400)
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
        "output_model_name": output_model_name,
        "checkpoint_dir": checkpoint_dir,
        "system_prompt": system_prompt,
        "base_model_id": model,
        "status": "started",
        "progress": 0,
        "current_epoch": 0,
        "current_step": 0,
        "loss": None,
        "error": None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
      }
      self.log_activity("Training Started", model, "started")
      asyncio.create_task(self._run_training_job(self._training_job))
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

  async def handle_tokens(self, request_id: str, tokens: List[int], is_finished: bool):
    token_len = len(tokens) if isinstance(tokens, (list, tuple)) else "?"
    if DEBUG >= 1:
      print(f"[ChatGPTAPI] [handle_tokens] request_id={request_id[:8]}..., len(tokens)={token_len}, is_finished={is_finished}, queue_exists={request_id in self.token_queues}")
    if request_id not in self.token_queues:
      if DEBUG >= 1:
        print(f"[ChatGPTAPI] [ERROR] Queue not found for {request_id[:8]}...")
      return
    await self.token_queues[request_id].put((tokens, is_finished))

  async def run(self, host: str = "0.0.0.0", port: int = 52415):

    runner = None
    try:
      runner = web.AppRunner(self.app)
      await runner.setup()
      site = web.TCPSite(runner, host, port)
      await site.start()
      display_host = "127.0.0.1" if host == "0.0.0.0" else host

      print(f"[ChatGPTAPI] [INFO] Available endpoints:")
      print(f"[ChatGPTAPI] [INFO]   - POST http://{display_host}:{port}/v1/chat/completions")
      print(f"[ChatGPTAPI] [INFO]   - GET  http://{display_host}:{port}/v1/models")
      print(f"[ChatGPTAPI] [INFO]   - GET  http://{display_host}:{port}/ (web UI)")
      try:
        _, tailscale_ips = await get_self_tailscale_info()
        if tailscale_ips:
          print(f"[ChatGPTAPI] [INFO]   (Tailscale - máy khác truy cập):")
          for ip in tailscale_ips:
            print(f"[ChatGPTAPI] [INFO]   - POST http://{ip}:{port}/v1/chat/completions")
            print(f"[ChatGPTAPI] [INFO]   - GET  http://{ip}:{port}/v1/models")
            print(f"[ChatGPTAPI] [INFO]   - GET  http://{ip}:{port}/ (web UI)")
        else:
          lan_ips = [ip for ip, _ in get_all_ip_addresses_and_interfaces() if ip and ip not in ("127.0.0.1", "::1", "localhost")]
          if lan_ips:
            print(f"[ChatGPTAPI] [INFO]   (Tailscale chưa bật — máy khác truy cập qua LAN):")
            for ip in lan_ips:
              print(f"[ChatGPTAPI] [INFO]   - GET  http://{ip}:{port}/ (web UI)")
          else:
            print(f"[ChatGPTAPI] [INFO]   (Tailscale): chưa bật — máy khác dùng IP LAN của máy này, port {port}")
      except Exception:
        lan_ips = [ip for ip, _ in get_all_ip_addresses_and_interfaces() if ip and ip not in ("127.0.0.1", "::1", "localhost")]
        if lan_ips:
          print(f"[ChatGPTAPI] [INFO]   (Máy khác truy cập qua LAN): GET http://{lan_ips[0]}:{port}/ (web UI)")
        else:
          print(f"[ChatGPTAPI] [INFO]   (Máy khác truy cập): dùng IP LAN máy này, port {port}")
      
      # Giữ server chạy cho đến khi bị cancel
      try:
        print("DEBUG: API Server entering wait loop...")
        await asyncio.Event().wait()  # Chờ vô hạn cho đến khi task bị cancel
        print("DEBUG: API Server wait loop finished (unexpectedly)!")
      except asyncio.CancelledError:
        print(f"[ChatGPTAPI] [INFO] API server shutdown requested")
        raise  # Re-raise để cleanup được thực hiện
    except asyncio.CancelledError:
      # Normal shutdown
      print(f"[ChatGPTAPI] [INFO] API server shutting down...")
    except Exception as e:
      print(f"[ChatGPTAPI] [ERROR] ERROR starting API server: {e}")
      import traceback
      traceback.print_exc()
      raise
    finally:
      # Cleanup runner
      if runner:
        try:
          await runner.cleanup()
          print(f"[ChatGPTAPI] [INFO] API server cleanup completed")
        except Exception as e:
          print(f"[ChatGPTAPI] [WARNING] Error during runner cleanup: {e}")

  def base64_decode(self, base64_string):
    #decode and reshape image
    if base64_string.startswith('data:image'):
      base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_data))
    W, H = (dim - dim%64 for dim in (img.width, img.height))
    if W != img.width or H != img.height:
      if DEBUG >= 2: print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
      img = img.resize((W, H), Image.NEAREST)  # use desired downsampling filter
    img = np.array(img)
    img = (img[:, :, :3].astype(np.float32)/255)*2 - 1
    img = img[None]
    return img
