# Tailscale discovery và chia sẻ tài nguyên node.
from .tailscale_discovery import TailscaleDiscovery
from .tailscale_helpers import (
    get_synapse_api_urls_from_node_list,
    first_ip_from_addresses,
    get_tailscale_devices,
    get_self_tailscale_info,
    Device,
)

__all__ = [
    "TailscaleDiscovery",
    "get_synapse_api_urls_from_node_list",
    "first_ip_from_addresses",
    "get_tailscale_devices",
    "get_self_tailscale_info",
    "Device",
]
