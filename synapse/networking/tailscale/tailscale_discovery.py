import asyncio
import time
import traceback
import socket
import aiohttp
from typing import List, Dict, Callable, Tuple, Set
from synapse.networking.discovery import Discovery
from synapse.networking.peer_handle import PeerHandle
from synapse.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.helpers import DEBUG, DEBUG_DISCOVERY
from .tailscale_helpers import get_device_id, get_self_tailscale_info, update_device_attributes, get_device_attributes, get_tailscale_devices, Device, first_ip_from_addresses


def _get_all_local_ips() -> Set[str]:
  """Lấy mọi IPv4 của máy này (kể cả Tailscale 100.x) để nhận diện 'máy này' khi không có tailscale CLI."""
  ips = set()
  try:
    import psutil
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
  return ips


class TailscaleDiscovery(Discovery):
  def __init__(
    self,
    node_id: str,
    node_port: int,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
    discovery_interval: int = 5,
    discovery_timeout: int = 120,
    update_interval: int = 15,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
    tailscale_api_key: str = None,
    tailnet: str = None,
    allowed_node_ids: List[str] = None,
  ):
    self.node_id = node_id
    self.node_port = node_port
    self.create_peer_handle = create_peer_handle
    self.discovery_interval = discovery_interval
    self.discovery_timeout = discovery_timeout
    self.update_interval = update_interval
    self.device_capabilities = device_capabilities
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
    self.peer_fail_counts: Dict[str, int] = {} # Mới: Theo dõi số lần Health Check thất bại liên tiếp
    self.probe_failure_cache: Dict[str, float] = {} # Mới: Cache các IP bị lỗi (IP -> timestamp) để tránh spam
    self._tailscale_devices_cache: Dict[str, Device] = {} # Mới: Cache danh sách thiết bị
    self._tailscale_cache_timestamp: float = 0 # Mới: Timestamp của cache
    # ✅ IMPROVEMENT 1: Adaptive Peer Timeout
    self.peer_timeout_adaptive = True
    self.peer_timeout_base = 60  # 1 minute for dynamic envs (vs 600s fixed)
    self.peer_timeout_max = 600
    self.peer_failure_history: Dict[str, List[float]] = {}  # peer_id -> list of failure timestamps
    self.discovery_task = None
    self.cleanup_task = None
    self.tailscale_api_key = tailscale_api_key
    self.tailnet = tailnet
    self.allowed_node_ids = allowed_node_ids
    self._device_id = None
    self.update_task = None

  async def start(self):
    if DEBUG_DISCOVERY >= 1: print("TailscaleDiscovery: Getting device capabilities...")
    self.device_capabilities = await device_capabilities()
    
    if DEBUG_DISCOVERY >= 1: print("TailscaleDiscovery: Starting discovery tasks...")
    self.discovery_task = asyncio.create_task(self.task_discover_peers())
    self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())
    self.update_task = asyncio.create_task(self.task_update_device_posture_attributes())
    if DEBUG_DISCOVERY >= 1: print("TailscaleDiscovery: All tasks started")

  async def task_update_device_posture_attributes(self):
    while True:
      try:
        await self.update_device_posture_attributes()
        if DEBUG_DISCOVERY >= 2:
          print(f"Updated device posture attributes")
      except Exception as e:
        print(f"Error updating device posture attributes: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.update_interval)

  async def get_device_id(self):
    if self._device_id:
      return self._device_id
    self._device_id = await get_device_id()
    return self._device_id

  async def update_device_posture_attributes(self):
    try:
      if DEBUG_DISCOVERY >= 1: print("TailscaleDiscovery: Updating device posture attributes...")
      device_id = await self.get_device_id()
      if DEBUG_DISCOVERY >= 1: print(f"TailscaleDiscovery: Device ID: {device_id}")
      await update_device_attributes(device_id, self.tailscale_api_key, self.node_id, self.node_port, self.device_capabilities)
      if DEBUG_DISCOVERY >= 1: print("TailscaleDiscovery: Device posture attributes updated successfully")
    except Exception as e:
      print(f"TailscaleDiscovery: Error updating device posture attributes: {e}")
      if DEBUG_DISCOVERY >= 1: traceback.print_exc()

  async def task_discover_peers(self):
    while True:
      try:
        devices: dict[str, Device] = await get_tailscale_devices(self.tailscale_api_key, self.tailnet)
        current_time = time.time()

        # Tăng ngưỡng lên 600 giây (10 phút) để tránh node bị ẩn khi chưa kịp refresh coordinator
        active_devices = {name: device for name, device in devices.items() if device.last_seen is not None and (current_time - device.last_seen.timestamp()) < 600}

        if DEBUG_DISCOVERY >= 4: print(f"Found tailscale devices: {devices}")
        if DEBUG_DISCOVERY >= 2: print(f"Active tailscale devices: {len(active_devices)}/{len(devices)}")
        if DEBUG_DISCOVERY >= 2: print("Time since last seen tailscale devices", [(current_time - device.last_seen.timestamp()) for device in devices.values()])

        for device in active_devices.values():
          if device.name == self.node_id: continue
          peer_host = device.addresses[0]
          try:
            peer_id, peer_port, device_capabilities = await get_device_attributes(device.device_id, self.tailscale_api_key)
          except (asyncio.TimeoutError, OSError, aiohttp.ClientError) as e:
            if DEBUG_DISCOVERY >= 1:
              peer_id, peer_port, device_capabilities = await get_device_attributes(device.device_id, self.tailscale_api_key)

          # Fallback: Nếu không lấy được ID qua API (do giới hạn gói Free), 
          # hãy thử "dò" trực tiếp qua IP trên cổng mặc định (node_port của chính mình)
          if not peer_id:
            peer_host = first_ip_from_addresses(device.addresses)
            if peer_host:
              fallback_port = self.node_port 
              if DEBUG >= 1: print(f"TailscaleDiscovery: Attributes missing for {device.name}, probing direct {peer_host}:{fallback_port}...")
            # Kiểm tra xem IP này có đang bị "cấm" do lỗi không (cooldown 60s)
          if peer_host in self.probe_failure_cache:
            if current_time - self.probe_failure_cache[peer_host] < 60:
              continue
            else:
              del self.probe_failure_cache[peer_host]

          if not peer_id:
            probe_handle = None
            try:
              # Tạo một handle tạm thời để probe
              probe_handle = self.create_peer_handle("probe", f"{peer_host}:{fallback_port}", "Probe", UNKNOWN_DEVICE_CAPABILITIES)
              # Tăng timeout dò máy lên 30 giây để bù đắp độ trễ mạng Tailscale và máy bận
              peer_id, device_capabilities = await asyncio.wait_for(probe_handle.probe(), timeout=30.0)
              peer_port = fallback_port
              if DEBUG >= 1: print(f"TailscaleDiscovery: Probe SUCCESS for {device.name} -> Node ID: {peer_id}")
            except Exception as e:
              # Dùng repr(e) để không bị chuỗi rỗng (ví dụ: TimeoutError)
              print(f"[WARNING] TailscaleDiscovery: Probe FAILED for {device.name} at {peer_host}:{fallback_port}. Reason: {repr(e)}")
              # Lưu lỗi vào cache để không thử lại ngay lập tức (tránh spam/nghẽn mạng)
              self.probe_failure_cache[peer_host] = current_time
            finally:
              # QUAN TRỌNG: Phải đóng handle probe dù thành công hay thất bại để tránh rò rỉ channel gRPC
              if probe_handle:
                try:
                  await probe_handle.disconnect()
                except Exception:
                  pass

          if not peer_id:
            if DEBUG_DISCOVERY >= 4: print(f"{device.device_id} does not have synapse node attributes. skipping.")
            continue

          if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
            if DEBUG_DISCOVERY >= 2: print(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list")
            continue

          if peer_id == self.node_id:
            if DEBUG >= 1: print(f"TailscaleDiscovery: Skipping device {device.name} (ID {peer_id}) because it is THIS machine (same Node ID).")
            continue

          if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
            new_peer_handle = self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", "TS", device_capabilities)
            try:
              if not await new_peer_handle.health_check():
                print(f"[WARNING] TailscaleDiscovery: Peer {peer_id} at {peer_host}:{peer_port} health_check FAILED. Port {peer_port} likely blocked by firewall.")
                continue
              # Thành công thì reset đếm lỗi
              self.peer_fail_counts[peer_id] = 0
            except Exception as e:
              print(f"[WARNING] TailscaleDiscovery: Peer {peer_id} health_check ERROR: {e}")
              continue

            if DEBUG >= 1: print(f"TailscaleDiscovery: ADDING peer {peer_id} at {peer_host}:{peer_port}")
            self.known_peers[peer_id] = (
              new_peer_handle,
              current_time,
              current_time,
            )
          else:
            # Máy đã biết, kiểm tra sức khỏe và áp dụng Grace Period
            peer_handle = self.known_peers[peer_id][0]
            is_healthy = False
            try:
              is_healthy = await peer_handle.health_check()
            except Exception:
              is_healthy = False

            if is_healthy:
              self.peer_fail_counts[peer_id] = 0
              self.known_peers[peer_id] = (peer_handle, self.known_peers[peer_id][1], current_time)
            else:
              # Thất bại: Tăng biến đếm lỗi thay vì xóa ngay
              count = self.peer_fail_counts.get(peer_id, 0) + 1
              self.peer_fail_counts[peer_id] = count
              
              if count >= 10: # Cho phép lỗi tối đa 10 lần liên tiếp (Grace Period ~ 120s+)
                print(f"[WARNING] Peer {peer_id} at {peer_host}:{peer_port} is not healthy after {count} attempts. Removing from cluster.")
                if peer_id in self.known_peers: del self.known_peers[peer_id]
                del self.peer_fail_counts[peer_id]
              else:
                print(f"[INFO] Peer {peer_id} missed health check ({count}/10). Keeping in list for recovery...")
                # Vẫn cập nhật last_seen để không bị cleanup_task xóa nhầm
                self.known_peers[peer_id] = (peer_handle, self.known_peers[peer_id][1], current_time)

      except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        if DEBUG_DISCOVERY >= 1:
          print(f"Tailscale discovery: API timeout/kết nối lỗi ({type(e).__name__}), thử lại sau.")
      except Exception as e:
        print(f"Error in discover peers: {e}")
        if DEBUG_DISCOVERY >= 1:
          print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)

  async def stop(self):
    if self.discovery_task:
      self.discovery_task.cancel()
    if self.cleanup_task:
      self.cleanup_task.cancel()
    if self.update_task:
      self.update_task.cancel()
    if self.discovery_task or self.cleanup_task or self.update_task:
      await asyncio.gather(self.discovery_task, self.cleanup_task, self.update_task, return_exceptions=True)

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2:
          print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    return [peer_handle for peer_handle, _, _ in self.known_peers.values()]

  def _calculate_adaptive_timeout(self, peer_id: str) -> int:
    """
    ✅ IMPROVEMENT 1: Calculate timeout based on peer failure history
    
    Peers with frequent failures get shorter timeouts for faster churn detection
    Peers with no failures keep maximum timeout
    """
    if not self.peer_timeout_adaptive:
      return self.discovery_timeout
    
    # Count failures in last 5 minutes
    now = time.time()
    recent_failures = len([
        t for t in self.peer_failure_history.get(peer_id, [])
        if now - t < 300  # 5 minute window
    ])
    
    # More failures = shorter timeout (detect churn faster)
    if recent_failures > 3:
      # Exponential backoff: timeout decreases as failures increase
      timeout = max(self.peer_timeout_base, self.peer_timeout_max // (recent_failures + 1))
      if DEBUG_DISCOVERY >= 2:
        print(f"  Adaptive timeout for {peer_id}: {timeout}s (recent_failures={recent_failures})")
      return timeout
    
    # Default: Base timeout + slight variance based on history depth
    return min(self.peer_timeout_base + (60 * max(0, 1 - recent_failures / 5)), self.peer_timeout_max)

  async def task_cleanup_peers(self):
    while True:
      try:
        current_time = time.time()
        peers_to_remove = []

        peer_ids = list(self.known_peers.keys())
        results = await asyncio.gather(*[self.check_peer(peer_id, current_time) for peer_id in peer_ids], return_exceptions=True)

        for peer_id, should_remove in zip(peer_ids, results):
          if should_remove: peers_to_remove.append(peer_id)

        if DEBUG_DISCOVERY >= 2:
          print(
            "Peer statuses:", {
              peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, health_check={await peer_handle.health_check()}, connected_at={connected_at}, last_seen={last_seen}"
              for peer_handle, connected_at, last_seen in self.known_peers.values()
            }
          )

        for peer_id in peers_to_remove:
          if peer_id in self.known_peers:
            del self.known_peers[peer_id]
            if DEBUG_DISCOVERY >= 2: print(f"Removed peer {peer_id} due to inactivity or failed health check.")
      except Exception as e:
        print(f"Error in cleanup peers: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)

  async def check_peer(self, peer_id: str, current_time: float) -> bool:
    peer_handle, connected_at, last_seen = self.known_peers.get(peer_id, (None, None, None))
    if peer_handle is None: return False

    try:
      is_connected = await peer_handle.is_connected()
      health_ok = await peer_handle.health_check()
    except Exception as e:
      if DEBUG_DISCOVERY >= 2: print(f"Error checking peer {peer_id}: {e}")
      # Record failure in history for adaptive timeout
      if peer_id not in self.peer_failure_history:
        self.peer_failure_history[peer_id] = []
      self.peer_failure_history[peer_id].append(current_time)
      # Keep only recent failures (last 5 minutes)
      cutoff = current_time - 300
      self.peer_failure_history[peer_id] = [
          t for t in self.peer_failure_history[peer_id] if t > cutoff
      ]
      return True

    # ✅ IMPROVEMENT 1: Use adaptive timeout instead of fixed discovery_timeout
    timeout = self._calculate_adaptive_timeout(peer_id)
    
    should_remove = (
        (not is_connected and current_time - connected_at > timeout) 
        or (current_time - last_seen > timeout) 
        or (not health_ok)
    )
    
    if not health_ok or (current_time - last_seen > timeout):
      # Record failure for adaptive timeout calculation
      if peer_id not in self.peer_failure_history:
        self.peer_failure_history[peer_id] = []
      self.peer_failure_history[peer_id].append(current_time)
      # Keep only recent failures (last 5 minutes)
      cutoff = current_time - 300
      self.peer_failure_history[peer_id] = [
          t for t in self.peer_failure_history[peer_id] if t > cutoff
      ]
    else:
      # Successful check: clear failure history for this peer
      if peer_id in self.peer_failure_history:
        self.peer_failure_history[peer_id] = []
    
    return should_remove

  async def get_devices_for_ui(self) -> List[Dict]:
    """Trả về danh sách thiết bị Tailscale cho Web UI giám sát (có kèm thông tin Synapse nếu có)."""
    try:
      our_device_id, our_tailscale_ips = await get_self_tailscale_info()
      our_ips_for_match = list(our_tailscale_ips) if our_tailscale_ips else list(_get_all_local_ips())
      hostname = (socket.gethostname() or "").strip().lower()
      # Dùng cache nếu còn hiệu lực (30s)
      current_time = time.time()
      if current_time - self._tailscale_cache_timestamp > 30:
        devices = await get_tailscale_devices(self.tailscale_api_key, self.tailnet)
        self._tailscale_devices_cache = devices
        self._tailscale_cache_timestamp = current_time
      else:
        devices = self._tailscale_devices_cache
      current_time = time.time()
      result = []
      for name, device in devices.items():
        last_seen_sec = (current_time - device.last_seen.timestamp()) if device.last_seen else None
        addrs = device.addresses or []
        addrs_flat = [str(a) if isinstance(a, str) else str(a.get("addr") or a.get("ip") or "") for a in addrs]
        match_id = our_device_id is not None and str(device.device_id) == str(our_device_id)
        match_ip = bool(our_ips_for_match and any(
          ip == addr or ip in addr or addr.startswith(ip + "/") or addr.startswith(ip + ":")
          for ip in our_ips_for_match for addr in addrs_flat if addr
        ))
        match_name = (name == self.node_id) or (self.node_id and (name == self.node_id or self.node_id in name))
        match_hostname = bool(hostname and name and hostname in name.lower())
        is_self = match_id or match_ip or match_name or match_hostname
        row = {
          "name": name,
          "device_id": device.device_id,
          "addresses": device.addresses or [],
          "last_seen_iso": device.last_seen.isoformat() if device.last_seen else None,
          "last_seen_sec_ago": round(last_seen_sec) if last_seen_sec is not None else None,
          "is_self": is_self,
          "is_synapse_node": False,
          "synapse_node_id": None,
          "synapse_port": None,
          "model": None,
          "memory_gb": None,
          "flops_fp16": None,
          "flops_fp32": None,
          "cpu_cores": None,
          "disk_gb": None,
          "system_ram_gb": None,
        }
        try:
          peer_id, peer_port, caps = await get_device_attributes(device.device_id, self.tailscale_api_key)
          if peer_id:
            row["is_synapse_node"] = True
            row["synapse_node_id"] = peer_id
            row["synapse_port"] = peer_port
            row["model"] = caps.model or ""
            row["memory_gb"] = caps.memory // 1024 if caps.memory else None
            row["flops_fp16"] = float(caps.flops.fp16) if caps.flops else 0.0
            row["flops_fp32"] = float(caps.flops.fp32) if caps.flops else 0.0
            row["cpu_cores"] = caps.cpu_cores
            row["disk_gb"] = caps.disk_gb
            row["system_ram_gb"] = (getattr(caps, "system_ram_mb", 0) or 0) // 1024
        except Exception:
          pass
        # Máy này: luôn dùng device_capabilities local (gọi mới mỗi lần) để hiển thị đúng
        if is_self:
          try:
            caps_local = await device_capabilities()
          except Exception:
            caps_local = self.device_capabilities
          if caps_local:
            row["is_synapse_node"] = True
            row["synapse_node_id"] = self.node_id
            row["synapse_port"] = self.node_port
            row["model"] = caps_local.model or ""
            row["memory_gb"] = caps_local.memory // 1024 if caps_local.memory else 0
            row["flops_fp16"] = float(caps_local.flops.fp16) if caps_local.flops else 0.0
            row["flops_fp32"] = float(caps_local.flops.fp32) if caps_local.flops else 0.0
            row["cpu_cores"] = caps_local.cpu_cores
            row["disk_gb"] = caps_local.disk_gb
            row["system_ram_gb"] = (getattr(caps_local, "system_ram_mb", 0) or 0) // 1024
        result.append(row)
      return result
    except Exception as e:
      if DEBUG_DISCOVERY >= 1:
        print(f"TailscaleDiscovery get_devices_for_ui: {e}")
        traceback.print_exc()
      return []
