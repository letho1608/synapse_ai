import os
import sys
import asyncio
from typing import Callable, TypeVar, Optional, Dict, Generic, Tuple, List
import socket
import random
import psutil
import uuid
import socket
import re
import subprocess
from pathlib import Path
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import traceback

DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.0.1"

# Ensure UTF-8 output for Windows terminals to avoid UnicodeEncodeError with emojis
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def get_system_info():
  """Return a human-readable system info string using detected hardware."""
  try:
    from synapse.topology.device_capabilities import SystemSpecs
    specs = SystemSpecs.detect()
    gpu_part = f", GPU: {specs.gpu_name} ({specs.gpu_vram_gb:.1f}GB VRAM)" if specs.has_gpu and specs.gpu_name else ", GPU: None"
    return f"{specs.cpu_name} ({specs.total_cpu_cores} cores), RAM: {specs.total_ram_gb:.1f}GB{gpu_part}"
  except Exception:
    return "Unknown System"


# ---------------------------------------------------------------------------
# Model Weight Analysis (ported from llmit/llmfit_core)
# ---------------------------------------------------------------------------

# Quantization bytes-per-parameter table
_QUANT_BPP = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0,
    "Q8_0": 1.05, "Q6_K": 0.80, "Q5_K_M": 0.68,
    "Q4_K_M": 0.58, "Q4_0": 0.58, "Q3_K_M": 0.48, "Q2_K": 0.37,
    "mlx-4bit": 0.55, "mlx-8bit": 1.0,
}
_QUANT_HIERARCHY = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
_QUANT_SPEED = {
    "F16": 0.6, "BF16": 0.6, "Q8_0": 0.8, "Q6_K": 0.95,
    "Q5_K_M": 1.0, "Q4_K_M": 1.15, "Q4_0": 1.15,
    "Q3_K_M": 1.25, "Q2_K": 1.35, "mlx-4bit": 1.15, "mlx-8bit": 0.85,
}


def _quant_bpp(quant: str) -> float:
    return _QUANT_BPP.get(quant, 0.58)


def _estimate_model_memory_gb(params_b: float, quant: str, ctx: int) -> float:
    """Estimate total memory required (GB) for a model.
    Formula: model_weights + KV_cache + runtime_overhead
    """
    bpp = _quant_bpp(quant)
    model_mem = params_b * bpp
    kv_cache = 0.000008 * params_b * ctx
    overhead = 0.5
    return model_mem + kv_cache + overhead


def _best_quant_for_budget(params_b: float, ctx: int, budget_gb: float) -> Optional[tuple]:
    """Find best quantization that fits within budget. Returns (quant, mem_gb) or None."""
    for q in _QUANT_HIERARCHY:
        mem = _estimate_model_memory_gb(params_b, q, ctx)
        if mem <= budget_gb:
            return (q, mem)
    # Try halving context
    half_ctx = ctx // 2
    if half_ctx >= 1024:
        for q in _QUANT_HIERARCHY:
            mem = _estimate_model_memory_gb(params_b, q, half_ctx)
            if mem <= budget_gb:
                return (q, mem)
    return None


def _score_fit(mem_required: float, mem_available: float) -> str:
    """Score memory fit level. Returns: Perfect, Good, Marginal, Too Tight."""
    if mem_required > mem_available:
        return "Too Tight"
    ratio = mem_required / mem_available if mem_available > 0 else float("inf")
    if ratio <= 0.6:
        return "Perfect"
    elif ratio <= 0.8:
        return "Good"
    else:
        return "Marginal"


def _fit_emoji(fit: str) -> str:
    return {"Perfect": "🟢", "Good": "🟡", "Marginal": "🟠", "Too Tight": "🔴"}.get(fit, "⚪")


def _load_model_db() -> list:
    """Load hf_models.json from synapse/data/."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "hf_models.json")
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _fuzzy_find_model(model_name: str, db: list) -> Optional[dict]:
    """Find the best matching model entry from hf_models.json using fuzzy matching."""
    if not db or not model_name:
        return None

    name_lower = model_name.lower().replace(":", "-").replace("_", "-")

    # Attempt 1: Exact substring match on name
    for entry in db:
        if name_lower in entry.get("name", "").lower():
            return entry

    # Attempt 2: Match by parameter count keywords (e.g., "7b", "1.5b")
    import re
    param_match = re.search(r"(\d+\.?\d*)\s*b", name_lower)
    if param_match:
        param_str = param_match.group(0)  # e.g., "7b" or "1.5b"
        for entry in db:
            entry_name = entry.get("name", "").lower()
            if param_str in entry_name:
                # Also check family name
                for keyword in ["qwen", "llama", "mistral", "phi", "gemma", "deepseek"]:
                    if keyword in name_lower and keyword in entry_name:
                        return entry

    # Attempt 3: Keyword family match
    for keyword in ["qwen", "llama", "mistral", "phi", "gemma", "deepseek", "gpt"]:
        if keyword in name_lower:
            for entry in db:
                if keyword in entry.get("name", "").lower():
                    return entry

    return None


def check_model_hardware_fit(model_name: str) -> None:
    """
    Check if the selected model fits on current hardware.
    Prints a colored summary to console. Called automatically at startup.
    Ported from llmit/llmfit_core/fit.py.
    """
    try:
        from synapse.topology.device_capabilities import SystemSpecs
        specs = SystemSpecs.detect()
    except Exception as e:
        if DEBUG >= 1:
            print(f"[model_check] Could not detect hardware: {e}")
        return

    db = _load_model_db()
    entry = _fuzzy_find_model(model_name, db)

    # Determine available memory budget
    if specs.has_gpu and specs.gpu_vram_gb and specs.gpu_vram_gb > 0:
        budget_gb = specs.gpu_vram_gb
        memory_label = f"GPU VRAM: {budget_gb:.1f}GB"
        run_mode = "GPU"
    elif specs.unified_memory and specs.gpu_vram_gb:
        budget_gb = specs.gpu_vram_gb
        memory_label = f"Unified Memory: {budget_gb:.1f}GB"
        run_mode = "Unified"
    else:
        budget_gb = specs.available_ram_gb
        memory_label = f"System RAM: {budget_gb:.1f}GB available"
        run_mode = "CPU"

    print(f"\n{'='*55}")
    print(f"  SYSTEM - Hardware Check")
    print(f"{'='*55}")
    print(f"  CPU  : {specs.cpu_name} ({specs.total_cpu_cores} cores)")
    print(f"  RAM  : {specs.total_ram_gb:.1f}GB total, {specs.available_ram_gb:.1f}GB available")
    if specs.has_gpu and specs.gpu_name:
        unified_tag = " [Unified]" if specs.unified_memory else ""
        print(f"  GPU  : {specs.gpu_name} ({specs.gpu_vram_gb:.1f}GB VRAM{unified_tag})")
    else:
        print(f"  GPU  : Not detected")
    print(f"  Mode : {run_mode} ({memory_label})")
    print(f"{'='*55}")

    if entry:
        params_b = entry.get("parameters_raw", 0) / 1e9 if entry.get("parameters_raw") else 7.0
        ctx = entry.get("context_length", 4096)
        default_quant = entry.get("quantization", "Q4_K_M")
        min_ram = entry.get("min_ram_gb", params_b * 0.6)

        mem_required = _estimate_model_memory_gb(params_b, default_quant, ctx)
        fit = _score_fit(mem_required, budget_gb)

        print(f"  Model: {entry.get('name', model_name)}")
        print(f"  Params: {params_b:.1f}B | Quant: {default_quant} | Ctx: {ctx:,} tokens")
        print(f"  Memory needed: ~{mem_required:.1f}GB")
        print(f"  Fit Status: {fit}")

        if fit == "Too Tight":
            best = _best_quant_for_budget(params_b, ctx, budget_gb)
            if best:
                q, mem = best
                print(f"\n  [SUGGESTION] Try quantization '{q}' -> needs ~{mem:.1f}GB")
            else:
                candidates = []
                for m in db:
                    p = m.get("parameters_raw", 0) / 1e9 if m.get("parameters_raw") else 0
                    if p > 0:
                        mem_est = _estimate_model_memory_gb(p, "Q4_K_M", 4096)
                        if mem_est <= budget_gb * 0.85:
                            candidates.append((m["name"], p, mem_est))
                candidates.sort(key=lambda x: x[1], reverse=True)
                if candidates:
                    print(f"\n  [ADVICE] Smaller models for your hardware:")
                    for name, p, mem in candidates[:3]:
                        print(f"     * {name} ({p:.1f}B, ~{mem:.1f}GB)")
    else:
        import re
        param_match = re.search(r"(\d+\.?\d*)\s*b", model_name.lower())
        if param_match:
            params_b = float(param_match.group(1))
            mem_required = _estimate_model_memory_gb(params_b, "Q4_K_M", 4096)
            fit = _score_fit(mem_required, budget_gb)
            print(f"  Model: {model_name} (est. {params_b:.1f}B params)")
            print(f"  Memory needed: ~{mem_required:.1f}GB (Q4_K_M)")
            print(f"  Fit Status: {fit}")
        else:
            print(f"  Model: {model_name} (metadata not found)")

    print(f"{'='*55}\n")


def find_available_port(host: str = "", min_port: int = 49152, max_port: int = 65535) -> int:
  used_ports_file = os.path.join(tempfile.gettempdir(), "synapse_used_ports")

  def read_used_ports():
    if os.path.exists(used_ports_file):
      with open(used_ports_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]
    return []

  def write_used_port(port, used_ports):
    with open(used_ports_file, "w") as f:
      for p in used_ports[-19:] + [port]:
        f.write(f"{p}\n")

  used_ports = read_used_ports()
  available_ports = set(range(min_port, max_port + 1)) - set(used_ports)

  while available_ports:
    port = random.choice(list(available_ports))
    if DEBUG >= 2: print(f"Trying to find available port {port=}")
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
      write_used_port(port, used_ports)
      return port
    except socket.error:
      available_ports.remove(port)

  raise RuntimeError("No available ports in the specified range")


def terminal_link(uri, label=None):
  if label is None:
    label = uri
  parameters = ""

  # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
  escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

  return escape_mask.format(parameters, uri, label)


T = TypeVar("T")
K = TypeVar("K")


class AsyncCallback(Generic[T]):
  def __init__(self) -> None:
    self.condition: asyncio.Condition = asyncio.Condition()
    self.result: Optional[Tuple[T, ...]] = None
    self.observers: list[Callable[..., None]] = []

  async def wait(self, check_condition: Callable[..., bool], timeout: Optional[float] = None) -> Tuple[T, ...]:
    async with self.condition:
      await asyncio.wait_for(self.condition.wait_for(lambda: self.result is not None and check_condition(*self.result)), timeout)
      assert self.result is not None  # for type checking
      return self.result

  def on_next(self, callback: Callable[..., None]) -> None:
    self.observers.append(callback)

  def set(self, *args: T) -> None:
    self.result = args
    for observer in self.observers:
      observer(*args)
    asyncio.create_task(self.notify())

  async def notify(self) -> None:
    async with self.condition:
      self.condition.notify_all()


class AsyncCallbackSystem(Generic[K, T]):
  def __init__(self) -> None:
    self.callbacks: Dict[K, AsyncCallback[T]] = {}

  def register(self, name: K) -> AsyncCallback[T]:
    if name not in self.callbacks:
      self.callbacks[name] = AsyncCallback[T]()
    return self.callbacks[name]

  def deregister(self, name: K) -> None:
    if name in self.callbacks:
      del self.callbacks[name]

  def trigger(self, name: K, *args: T) -> None:
    if name in self.callbacks:
      self.callbacks[name].set(*args)

  def trigger_all(self, *args: T) -> None:
    for callback in self.callbacks.values():
      callback.set(*args)


K = TypeVar('K', bound=str)
V = TypeVar('V')


class PrefixDict(Generic[K, V]):
  def __init__(self):
    self.items: Dict[K, V] = {}

  def add(self, key: K, value: V) -> None:
    self.items[key] = value

  def find_prefix(self, argument: str) -> List[Tuple[K, V]]:
    return [(key, value) for key, value in self.items.items() if argument.startswith(key)]

  def find_longest_prefix(self, argument: str) -> Optional[Tuple[K, V]]:
    matches = self.find_prefix(argument)
    if len(matches) == 0:
      return None

    return max(matches, key=lambda x: len(x[0]))


def is_valid_uuid(val):
  try:
    uuid.UUID(str(val))
    return True
  except ValueError:
    return False


def get_or_create_node_id():
  NODE_ID_FILE = Path(tempfile.gettempdir())/".synapse_node_id"
  try:
    if NODE_ID_FILE.is_file():
      with open(NODE_ID_FILE, "r") as f:
        stored_id = f.read().strip()
      if is_valid_uuid(stored_id):
        if DEBUG >= 2: print(f"Retrieved existing node ID: {stored_id}")
        return stored_id
      else:
        if DEBUG >= 2: print("Stored ID is not a valid UUID. Generating a new one.")

    new_id = str(uuid.uuid4())
    with open(NODE_ID_FILE, "w") as f:
      f.write(new_id)

    if DEBUG >= 2: print(f"Generated and stored new node ID: {new_id}")
    return new_id
  except IOError as e:
    if DEBUG >= 2: print(f"IO error creating node_id: {e}")
    return str(uuid.uuid4())
  except Exception as e:
    if DEBUG >= 2: print(f"Unexpected error creating node_id: {e}")
    return str(uuid.uuid4())


def pretty_print_bytes(size_in_bytes: int) -> str:
  if size_in_bytes < 1024:
    return f"{size_in_bytes} B"
  elif size_in_bytes < 1024**2:
    return f"{size_in_bytes / 1024:.2f} KB"
  elif size_in_bytes < 1024**3:
    return f"{size_in_bytes / (1024 ** 2):.2f} MB"
  elif size_in_bytes < 1024**4:
    return f"{size_in_bytes / (1024 ** 3):.2f} GB"
  else:
    return f"{size_in_bytes / (1024 ** 4):.2f} TB"


def pretty_print_bytes_per_second(bytes_per_second: int) -> str:
  if bytes_per_second < 1024:
    return f"{bytes_per_second} B/s"
  elif bytes_per_second < 1024**2:
    return f"{bytes_per_second / 1024:.2f} KB/s"
  elif bytes_per_second < 1024**3:
    return f"{bytes_per_second / (1024 ** 2):.2f} MB/s"
  elif bytes_per_second < 1024**4:
    return f"{bytes_per_second / (1024 ** 3):.2f} GB/s"
  else:
    return f"{bytes_per_second / (1024 ** 4):.2f} TB/s"


def get_all_ip_addresses_and_interfaces():
    ip_addresses = []
    try:
        # Get local machine IP addresses using Windows socket approach
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith("127."):
            ip_addresses.append((local_ip, "Local Area Connection"))
        
        # Also add localhost
        ip_addresses.append(("127.0.0.1", "Loopback"))
        
    except Exception as e:
        if DEBUG >= 1: print(f"Failed to get IP addresses: {e}")
        if DEBUG >= 1: traceback.print_exc()
    
    if not ip_addresses:
        if DEBUG >= 1: print("Failed to get any IP addresses. Defaulting to localhost.")
        return [("localhost", "lo")]
    
    return list(set(ip_addresses))



async def get_interface_priority_and_type(ifname: str) -> Tuple[int, str]:
  # Local container/virtual interfaces
  if ('bridge' in ifname.lower() or 'docker' in ifname.lower()):
    return (7, "Container Virtual")

  # Loopback interface
  if 'loopback' in ifname.lower():
    return (6, "Loopback")

  # Regular ethernet detection
  if 'ethernet' in ifname.lower():
    return (4, "Ethernet")

  # WiFi detection
  if ('wifi' in ifname.lower() or 'wireless' in ifname.lower() or 'wi-fi' in ifname.lower()):
    return (3, "WiFi")

  # VPN interfaces
  if ('vpn' in ifname.lower() or 'tunnel' in ifname.lower()):
    return (1, "External Virtual")

  # Other interfaces
  return (2, "Other")


async def shutdown(signal, loop, server):
  """Gracefully shutdown the server and close the asyncio loop."""
  print(f"Received exit signal {signal.name}...")
  print("Thank you for using synapse.")
  server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  [task.cancel() for task in server_tasks]
  print(f"Cancelling {len(server_tasks)} outstanding tasks")
  await asyncio.gather(*server_tasks, return_exceptions=True)
  await server.stop()


def is_frozen():
  return getattr(sys, 'frozen', False) or os.path.basename(sys.executable) == "synapse" \
    or '__nuitka__' in globals() or getattr(sys, '__compiled__', False)

