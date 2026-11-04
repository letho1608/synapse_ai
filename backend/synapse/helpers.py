import os
import sys
import asyncio
import hashlib
from typing import Callable, TypeVar, Optional, Dict, Generic, Tuple
import socket
import psutil
from pathlib import Path
import tempfile
import json
import traceback

DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.1.0"


def _get_mac_address() -> str:
    """Get the primary MAC address of this machine."""
    mac = None
    try:
        import psutil
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == psutil.AF_LINK and addr.address:
                    mac = addr.address
                    break
            if mac:
                break
    except Exception:
        pass
    if not mac:
        mac = "00:00:00:00:00:00"
    return mac


def _cleanup_old_node_id_files():
    """Remove old .synapse_node_id files from temp directory."""
    try:
        old_files = [
            Path(tempfile.gettempdir()) / ".synapse_node_id",
            Path(tempfile.gettempdir()) / "synapse_node_id",
        ]
        for f in old_files:
            if f.is_file():
                f.unlink()
                if DEBUG >= 2:
                    print(f"Cleaned up old node ID file: {f}")
    except Exception as e:
        if DEBUG >= 2:
            print(f"Failed to cleanup old node ID files: {e}")

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

# Quantization bytes-per-parameter table — SINGLE SOURCE OF TRUTH (B4).
# model_discovery imports this (no local copy); values include a runtime
# margin, so fit advice stays conservative. Do not fork this table.
QUANT_BPP = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0,
    "Q8_0": 1.05, "Q6_K": 0.80, "Q5_K_M": 0.68,
    "Q4_K_M": 0.58, "Q4_0": 0.58, "Q3_K_M": 0.48, "Q2_K": 0.37,
    "AWQ-4bit": 0.55, "GPTQ-Int4": 0.55,
    "mlx-4bit": 0.55, "mlx-8bit": 1.0,
}
# Legacy private alias (same object, do not diverge).
_QUANT_BPP = QUANT_BPP
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


def get_tailscale_env():
    """Canonical Tailscale credentials (C-W10).

    Reads TAILSCALE_API_KEY (alias TS_API_KEY) and TAILSCALE_TAILNET
    (aliases TS_TAILNET, then legacy TAILNET_NAME). Both backend/main.py
    and Node.get_tailscale_nodes() must use this — never divergent names.
    Returns (api_key, tailnet), empty strings when unset.
    """
    api_key = (os.getenv("TAILSCALE_API_KEY", "")
               or os.getenv("TS_API_KEY", ""))
    tailnet = (os.getenv("TAILSCALE_TAILNET", "")
               or os.getenv("TS_TAILNET", "")
               or os.getenv("TAILNET_NAME", ""))
    return api_key, tailnet


def check_ports_free(wanted) -> list:
    """Preflight: try binding each {name: port} (TCP). Returns a list of
    conflict dicts {name, port, holder} — empty when all are free.

    holder names the owning process (e.g. "pid 5020 (python.exe)") via
    psutil, or "" when it cannot be determined. Never raises; a bind
    failure for any other reason reports holder="unbindable: <err>".
    Sockets are closed immediately, no lingering.

    No SO_REUSEADDR on the probe (deliberate): with it, Windows lets a
    second bind succeed and the conflict would go undetected. A TIME_WAIT
    leftover is NOT reported — only a live LISTEN socket counts, verified
    via psutil so a just-closed backend never blocks an immediate restart.
    """
    conflicts = []
    try:
        items = list(wanted.items())
    except Exception:
        return conflicts
    for name, port in items:
        try:
            port = int(port)
        except (TypeError, ValueError):
            continue
        probe = None
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("0.0.0.0", port))
        except OSError:
            holder = _listening_holder(port)
            if holder is None:
                continue  # transient (TIME_WAIT) — port is effectively free
            conflicts.append({"name": name, "port": port, "holder": holder})
        except Exception as e:
            conflicts.append({"name": name, "port": port,
                              "holder": "unbindable: %s" % (e,)})
        finally:
            try:
                if probe is not None:
                    probe.close()
            except Exception:
                pass
    return conflicts


def _listening_holder(port: int):
    """Return 'pid N (name)' if something is LISTENing on port, else None."""
    try:
        import psutil as _psutil
        for conn in _psutil.net_connections(kind="tcp"):
            try:
                if conn.laddr and conn.laddr.port == port and \
                        conn.status == _psutil.CONN_LISTEN:
                    pid = conn.pid or 0
                    pname = ""
                    try:
                        pname = _psutil.Process(pid).name() if pid else ""
                    except Exception:
                        pass
                    return "pid %d (%s)" % (pid, pname or "?")
            except Exception:
                continue
    except Exception:
        pass
    return None


def format_port_conflicts(conflicts) -> str:
    """Human-readable fix-it message for check_ports_free() output."""
    lines = ["Port conflict: another process holds ports this node needs."]
    for c in conflicts:
        lines.append("  - %s port %s held by %s" % (
            c.get("name", "?"), c.get("port", "?"), c.get("holder", "?") or "?"))
    lines.append("Fix (Windows): netstat -ano | findstr :<port>  ->  taskkill /PID <pid> /F")
    lines.append("Or run with free ports: --node-port <free> --chatgpt-api-port <free>")
    lines.append("(Often another node already running on this machine, or a launcher "
                 "that was killed and left its backend holding the ports.)")
    return "\n".join(lines)


def get_all_local_ips() -> set:
    """All local IPv4 addresses, incl. loopback (B1: single copy).

    Canonical home of the former _get_all_local_ips() triple in
    udp_discovery / tailscale_discovery / p2p_socket_bridge.
    """
    ips = set()
    try:
        for _nic, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if getattr(socket, "AF_INET", None) and a.family == socket.AF_INET and a.address:
                    ips.add(a.address.strip())
    except Exception:
        pass
    if not ips:
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip and not ip.startswith("127."):
                ips.add(ip)
        except Exception:
            pass
    ips.add("127.0.0.1")
    return ips


def _load_model_db() -> list:
    """Load hf_models.json from synapse/config/."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), "config", "hf_models.json")
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_model_db(db: list) -> bool:
    """Save hf_models.json to synapse/config/."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), "config", "hf_models.json")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        if DEBUG >= 1: print(f"[helpers] Failed to save model db: {e}")
        return False


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
    print("  SYSTEM - Hardware Check")
    print(f"{'='*55}")
    print(f"  CPU  : {specs.cpu_name} ({specs.total_cpu_cores} cores)")
    print(f"  RAM  : {specs.total_ram_gb:.1f}GB total, {specs.available_ram_gb:.1f}GB available")
    if specs.has_gpu and specs.gpu_name:
        unified_tag = " [Unified]" if specs.unified_memory else ""
        print(f"  GPU  : {specs.gpu_name} ({specs.gpu_vram_gb:.1f}GB VRAM{unified_tag})")
    else:
        print("  GPU  : Not detected")
    print(f"  Mode : {run_mode} ({memory_label})")
    print(f"{'='*55}")

    if entry:
        params_b = entry.get("parameters_raw", 0) / 1e9 if entry.get("parameters_raw") else 7.0
        ctx = entry.get("context_length", 4096)
        default_quant = entry.get("quantization", "Q4_K_M")
        entry.get("min_ram_gb", params_b * 0.6)

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
                    print("\n  [ADVICE] Smaller models for your hardware:")
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


def get_or_create_node_id(port: int = None):
    """
    Generate a deterministic node ID based on MAC address + port.
    This ensures each node on the same machine with different ports
    gets a unique ID, and the ID remains stable across restarts.
    """
    _cleanup_old_node_id_files()

    mac = _get_mac_address()
    unique_input = f"{mac}:{port}" if port else mac
    node_id = hashlib.sha256(unique_input.encode()).hexdigest()[:32]

    if DEBUG >= 2:
        print(f"Generated deterministic node ID: {node_id} (MAC={mac}, port={port})")
    return node_id


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
        # ALWAYS add localhost FIRST -- P2P on same machine must use loopback.
        # Sending on non-loopback first causes TCP loops and 5s handshake timeouts.
        ip_addresses.append(("127.0.0.1", "Loopback"))

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip and not local_ip.startswith("127."):
            ip_addresses.append((local_ip, "Local Area Connection"))

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


PARENT_PID_ENV = "SYNAPSE_PARENT_PID"


def start_parent_watchdog(interval_s: float = 1.0, _exit=os._exit, _own_start=None):
    """Exit this process once the launcher named by $SYNAPSE_PARENT_PID dies.

    Launchers (main.py, `serve`, `console`) spawn the backend as a child; if
    the launcher is killed (TerminateProcess on Windows, SIGKILL on POSIX) no
    cleanup code runs and the orphaned backend keeps holding its ports. A
    process with that PID created after us is a recycled PID, not the
    launcher. Returns the thread, or None when the env var is unset/invalid.
    A launcher that is already gone (killed while we were importing) makes
    the thread exit the process on its first check.
    """
    import threading
    import time as _time
    raw = os.environ.get(PARENT_PID_ENV, "").strip()
    try:
        pid = int(raw)
    except ValueError:
        return None
    own_start = _own_start
    if own_start is None:
        try:
            own_start = psutil.Process().create_time()
        except psutil.Error:
            own_start = float("inf")

    def _alive() -> bool:
        try:
            return psutil.Process(pid).create_time() <= own_start
        except psutil.Error:
            return False

    def _watch():
        while _alive():
            _time.sleep(interval_s)
        print("[synapse] launcher process %d exited -> stopping backend" % pid,
              file=sys.stderr, flush=True)
        _exit(0)

    t = threading.Thread(target=_watch, name="parent-watchdog", daemon=True)
    t.start()
    return t


