import json
import asyncio
import aiohttp
import re
from typing import Dict, Any, Tuple, List, Optional
from synapse.helpers import DEBUG_DISCOVERY
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from datetime import datetime, timezone

# Timeout cho các request tới Tailscale API (giây) - tránh TimeoutError khi mạng chậm
TAILSCALE_API_TIMEOUT = aiohttp.ClientTimeout(total=45, connect=15)
TAILSCALE_GET_ATTRIBUTES_RETRIES = 2
TAILSCALE_RETRY_DELAY = 2.0


class Device:
  def __init__(self, device_id: str, name: str, addresses: List[str], last_seen: Optional[datetime] = None):
    self.device_id = device_id
    self.name = name
    self.addresses = addresses
    self.last_seen = last_seen

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'Device':
    return cls(device_id=data.get('id', ''), name=data.get('name', ''), addresses=data.get('addresses', []), last_seen=cls.parse_datetime(data.get('lastSeen')))

  @staticmethod
  def parse_datetime(date_string: Optional[str]) -> Optional[datetime]:
    if not date_string:
      return None
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


async def get_device_id() -> str:
  info = await get_self_tailscale_info()
  if info[0] is None:
    raise Exception("Could not get Tailscale device ID. Is tailscale CLI installed and running?")
  return info[0]


async def get_self_tailscale_info() -> Tuple[Optional[str], List[str]]:
  """Trả về (device_id, danh sách Tailscale IP của máy này) để nhận diện 'máy này' trong danh sách devices."""
  try:
    process = await asyncio.create_subprocess_exec('tailscale', 'status', '--json', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
      return (None, [])
    data = json.loads(stdout.decode())
    self_data = data.get('Self') or {}
    device_id = self_data.get('ID')
    if device_id is not None:
      device_id = str(device_id)
    ips = list(self_data.get('TailscaleIPs') or [])
    return (device_id, ips)
  except Exception:
    return (None, [])


async def update_device_attributes(device_id: str, api_key: str, node_id: str, node_port: int, device_capabilities: DeviceCapabilities):
  async with aiohttp.ClientSession(timeout=TAILSCALE_API_TIMEOUT) as session:
    base_url = f"https://api.tailscale.com/api/v2/device/{device_id}/attributes"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    attributes = {
      "custom:synapse_node_id": node_id.replace('-', '_'), "custom:synapse_node_port": node_port, "custom:synapse_device_capability_chip": sanitize_attribute(device_capabilities.chip),
      "custom:synapse_device_capability_model": sanitize_attribute(device_capabilities.model), "custom:synapse_device_capability_memory": str(device_capabilities.memory),
      "custom:synapse_device_capability_flops_fp16": str(device_capabilities.flops.fp16), "custom:synapse_device_capability_flops_fp32": str(device_capabilities.flops.fp32),
      "custom:synapse_device_capability_flops_int8": str(device_capabilities.flops.int8),
      "custom:synapse_device_capability_cpu_cores": str(device_capabilities.cpu_cores), "custom:synapse_device_capability_disk_gb": str(device_capabilities.disk_gb),
      "custom:synapse_device_capability_system_ram_mb": str(getattr(device_capabilities, "system_ram_mb", 0)),
    }

    for attr_name, attr_value in attributes.items():
      url = f"{base_url}/{attr_name}"
      data = {"value": str(attr_value).replace(' ', '_')}  # Ensure all values are strings for JSON
      async with session.post(url, headers=headers, json=data) as response:
        if response.status == 200:
          if DEBUG_DISCOVERY >= 1: print(f"Updated device posture attribute {attr_name} for device {device_id}")
        elif response.status == 404:
          # 404 means the feature is likely not enabled or available, suppress the error to avoid spam
          if DEBUG_DISCOVERY >= 2: print(f"Device posture attribute {attr_name} not found (404). This feature may not be enabled.")
        else:
          print(f"Failed to update device posture attribute {attr_name}: {response.status} {await response.text()}")


def _safe_float(val: Any, default: float = 0) -> float:
  """Chuyển giá trị sang float an toàn (string, None, số)."""
  if val is None:
    return default
  try:
    return float(val)
  except (TypeError, ValueError):
    return default


def _attributes_to_dict(attributes: Any) -> Dict[str, str]:
  """Chuẩn hóa attributes từ Tailscale: có thể là dict hoặc list [{key, value}, ...]."""
  if not attributes:
    return {}
  if isinstance(attributes, dict):
    return {k: str(v) if v is not None else "" for k, v in attributes.items()}
  if isinstance(attributes, list):
    out = {}
    for item in attributes:
      if isinstance(item, dict):
        k = item.get("key") or item.get("name") or item.get("id")
        v = item.get("value")
        if k is not None:
          out[str(k)] = str(v) if v is not None else ""
    return out
  return {}


async def get_device_attributes(device_id: str, api_key: str) -> Tuple[str, int, DeviceCapabilities]:
  url = f"https://api.tailscale.com/api/v2/device/{device_id}/attributes"
  headers = {'Authorization': f'Bearer {api_key}'}
  last_error = None
  for attempt in range(TAILSCALE_GET_ATTRIBUTES_RETRIES + 1):
    try:
      async with aiohttp.ClientSession(timeout=TAILSCALE_API_TIMEOUT) as session:
        async with session.get(url, headers=headers) as response:
          if response.status == 200:
            data = await response.json()
            raw_attrs = data.get("attributes", {})
            attributes = _attributes_to_dict(raw_attrs)
            node_id = (attributes.get("custom:synapse_node_id") or "").replace('_', '-')
            node_port = int(attributes.get("custom:synapse_node_port", 0))
            device_capabilities = DeviceCapabilities(
              model=(attributes.get("custom:synapse_device_capability_model") or "").replace('_', ' '),
              chip=(attributes.get("custom:synapse_device_capability_chip") or "").replace('_', ' '),
              memory=int(attributes.get("custom:synapse_device_capability_memory", 0)),
              flops=DeviceFlops(
                fp16=_safe_float(attributes.get("custom:synapse_device_capability_flops_fp16"), 0),
                fp32=_safe_float(attributes.get("custom:synapse_device_capability_flops_fp32"), 0),
                int8=_safe_float(attributes.get("custom:synapse_device_capability_flops_int8"), 0)
              ),
              cpu_cores=int(attributes.get("custom:synapse_device_capability_cpu_cores", 0)),
              disk_gb=int(attributes.get("custom:synapse_device_capability_disk_gb", 0)),
              system_ram_mb=int(attributes.get("custom:synapse_device_capability_system_ram_mb", 0)),
            )
            return node_id, node_port, device_capabilities
          else:
            print(f"Failed to fetch posture attributes for {device_id}: {response.status}")
            return "", 0, DeviceCapabilities(model="", chip="", memory=0, flops=DeviceFlops(fp16=0, fp32=0, int8=0), cpu_cores=0, disk_gb=0, system_ram_mb=0)
    except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as e:
      last_error = e
      if attempt < TAILSCALE_GET_ATTRIBUTES_RETRIES:
        if DEBUG_DISCOVERY >= 1:
          print(f"Tailscale API timeout/error for device {device_id}, retry {attempt + 1}/{TAILSCALE_GET_ATTRIBUTES_RETRIES} in {TAILSCALE_RETRY_DELAY}s: {e}")
        await asyncio.sleep(TAILSCALE_RETRY_DELAY)
      else:
        if DEBUG_DISCOVERY >= 1:
          print(f"Tailscale API failed for device {device_id} after retries: {last_error}")
        raise last_error
  return "", 0, DeviceCapabilities(model="", chip="", memory=0, flops=DeviceFlops(fp16=0, fp32=0, int8=0), cpu_cores=0, disk_gb=0, system_ram_mb=0)


def parse_device_attributes(data: Dict[str, str]) -> Dict[str, Any]:
  result = {}
  prefix = "custom:synapse_"
  for key, value in data.items():
    if key.startswith(prefix):
      attr_name = key.replace(prefix, "")
      if attr_name in ["node_id", "node_port", "device_capability_chip", "device_capability_model"]:
        # Đảm bảo value không phải None trước khi gọi replace
        result[attr_name] = (value or "").replace('_', ' ')
      elif attr_name in ["device_capability_memory", "device_capability_flops_fp16", "device_capability_flops_fp32", "device_capability_flops_int8"]:
        result[attr_name] = float(value)
  return result


def sanitize_attribute(value: str) -> str:
  # Replace invalid characters with underscores
  if value is None:
    return ""
  sanitized_value = re.sub(r'[^a-zA-Z0-9_.]', '_', str(value))
  # Truncate to 50 characters
  return sanitized_value[:50]


async def get_tailscale_devices(api_key: str, tailnet: str) -> Dict[str, Device]:
  async with aiohttp.ClientSession(timeout=TAILSCALE_API_TIMEOUT) as session:
    url = f"https://api.tailscale.com/api/v2/tailnet/{tailnet}/devices"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with session.get(url, headers=headers) as response:
      response.raise_for_status()
      data = await response.json()

      devices = {}
      for device_data in data.get("devices", []):
        device = Device.from_dict(device_data)
        devices[device.name] = device

      return devices


def first_ip_from_addresses(addresses: Any) -> Optional[str]:
  """Lấy IP đầu tiên từ danh sách addresses (node Tailscale). Bỏ /cidr nếu có."""
  if not addresses:
    return None
  for a in addresses:
    s = str(a) if isinstance(a, str) else str((a or {}).get("addr") or (a or {}).get("ip") or "")
    s = (s or "").strip()
    if "/" in s:
      s = s.split("/")[0].strip()
    if s and all(c in "0123456789." for c in s):
      return s
  return None


def get_synapse_api_urls_from_node_list(
  nodes: List[Dict[str, Any]],
  api_port: int = 52415,
  only_synapse_nodes: bool = True,
  exclude_ips: Optional[set] = None,
) -> List[str]:
  """
  Từ danh sách node, trả về danh sách Synapse API base URL (http://ip:api_port).
  exclude_ips: set IP (str) để loại trừ (vd IP của máy này).
  """
  exclude_ips = exclude_ips or set()
  seen: set = set()
  urls: List[str] = []
  for node in nodes:
    if only_synapse_nodes and not node.get("is_synapse_node"):
      continue
    ip = first_ip_from_addresses(node.get("addresses") or [])
    if not ip or ip in exclude_ips:
      continue
    url = f"http://{ip}:{api_port}"
    key = url.lower()
    if key not in seen:
      seen.add(key)
      urls.append(url)
  return urls
