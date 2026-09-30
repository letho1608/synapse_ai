import asyncio
import time
import traceback
import socket
import aiohttp
from typing import List, Dict, Callable, Tuple
from synapse.networking.discovery import Discovery, prune_stale_peers
from synapse.networking.peer_handle import PeerHandle
from synapse.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.helpers import DEBUG, DEBUG_DISCOVERY
from .tailscale_helpers import get_device_id, update_device_attributes, get_device_attributes, get_tailscale_devices, Device


class TailscaleDiscovery(Discovery):
    def __init__(
        self,
        node_id: str,
        node_port: int,
        create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
        discovery_interval: int = 5,
        discovery_timeout: int = 30,
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
        # Probe once: posture updates need a local Tailscale device identity.
        # Without the CLI they can never succeed — skip the task instead of
        # erroring every update_interval forever.
        self._tailscale_cli = False
        try:
            from synapse.networking.tailscale.tailscale_helpers import get_self_tailscale_info
            _device_id, _ = await get_self_tailscale_info()
            self._tailscale_cli = bool(_device_id)
        except Exception:
            self._tailscale_cli = False
        if self._tailscale_cli:
            self.update_task = asyncio.create_task(self.task_update_device_posture_attributes())
        else:
            self.update_task = None
            print("TailscaleDiscovery: no Tailscale CLI/device identity — posture updates OFF "
                  "(logged once, not every interval). On LAN without Tailscale, prefer "
                  "--discovery-module udp (same subnet, zero config) or manual (cluster_config.json).")
        if not self.tailscale_api_key and DEBUG_DISCOVERY >= 1:
            print("TailscaleDiscovery: no API key — peer discovery via API is inert; same advice as above.")
        if DEBUG_DISCOVERY >= 1: print("TailscaleDiscovery: All tasks started")

    async def task_update_device_posture_attributes(self):
        while True:
            try:
                await self.update_device_posture_attributes()
                if DEBUG_DISCOVERY >= 2:
                    print("Updated device posture attributes")
            except Exception as e:
                if DEBUG_DISCOVERY >= 1:
                    print(f"TailscaleDiscovery: Unexpected error in posture update loop: {e}")
                    traceback.print_exc()
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
        except (asyncio.TimeoutError, aiohttp.ClientError, socket.gaierror) as e:
            if DEBUG_DISCOVERY >= 1:
                print(f"TailscaleDiscovery: API timeout/connection error while updating attributes ({type(e).__name__}). Will retry later.")
        except Exception as e:
            print(f"TailscaleDiscovery: Error updating device posture attributes: {e}")
            if DEBUG_DISCOVERY >= 1: traceback.print_exc()

    async def task_discover_peers(self):
        while True:
            try:
                devices: dict[str, Device] = await get_tailscale_devices(self.tailscale_api_key, self.tailnet)
                current_time = time.time()

                active_devices = {name: device for name, device in devices.items() if device.last_seen is not None and (current_time - device.last_seen.timestamp()) < 30}

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
                            print(f"Discover peers: skip device {device.device_id} ({device.name}) after timeout/error: {e}")
                        continue

                    # Fallback: if node_id is empty (Tailscale Free plan without attributes),
                    # use device.name as the peer identifier and self.node_port as default
                    if not peer_id:
                        if DEBUG_DISCOVERY >= 2:
                            print(f"Discover peers: {device.name} ({device.device_id}) has no attributes (Tailscale Free?). Using name as fallback peer_id.")
                        peer_id = device.name
                        peer_port = peer_port if peer_port > 0 else self.node_port

                    if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
                        if DEBUG_DISCOVERY >= 2: print(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list")
                        continue

                    if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
                        new_peer_handle = self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", "TS", device_capabilities)
                        if not await new_peer_handle.health_check():
                            if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Skipping.")
                            continue

                        if DEBUG >= 1: print(f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
                        self.known_peers[peer_id] = (
                            new_peer_handle,
                            current_time,
                            current_time,
                        )
                    else:
                        if not await self.known_peers[peer_id][0].health_check():
                            if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.")
                            if peer_id in self.known_peers: del self.known_peers[peer_id]
                            continue
                        self.known_peers[peer_id] = (self.known_peers[peer_id][0], self.known_peers[peer_id][1], current_time)

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
        tasks = [t for t in (self.discovery_task, self.cleanup_task,
                             self.update_task) if t is not None]
        for t in tasks:
            t.cancel()
        self.discovery_task = None
        self.cleanup_task = None
        self.update_task = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        if wait_for_peers > 0:
            while len(self.known_peers) < wait_for_peers:
                if DEBUG_DISCOVERY >= 2:
                    print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
                await asyncio.sleep(0.1)
        return [peer_handle for peer_handle, _, _ in self.known_peers.values()]

    async def task_cleanup_peers(self):
        while True:
            try:
                await prune_stale_peers(self.known_peers, self.discovery_timeout, DEBUG_DISCOVERY)
            except Exception as e:
                print(f"Error in cleanup peers: {e}")
                print(traceback.format_exc())
            finally:
                await asyncio.sleep(self.discovery_interval)
