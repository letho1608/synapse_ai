from abc import ABC, abstractmethod
from typing import List
from .peer_handle import PeerHandle


class Discovery(ABC):
  @abstractmethod
  async def start(self) -> None:
    pass

  @abstractmethod
  async def stop(self) -> None:
    pass

  @abstractmethod
  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    pass


async def peer_is_stale(peer_handle, connected_at: float, last_seen: float,
                        now: float, timeout: float) -> bool:
  """True when a discovered peer should be dropped: disconnected past the
  timeout, not seen within it, failing its health check, or unreachable."""
  try:
    is_connected = await peer_handle.is_connected()
    health_ok = await peer_handle.health_check()
  except Exception:
    return True
  return ((not is_connected and now - connected_at > timeout)
          or (now - last_seen > timeout) or (not health_ok))


async def prune_stale_peers(known_peers: dict, timeout: float, debug: int = 0) -> None:
  """Remove stale entries from a discovery's {peer_id: (handle, connected_at,
  last_seen, ...)} map (shared by the UDP and Tailscale discoveries)."""
  import asyncio
  import time
  now = time.time()
  ids = list(known_peers.keys())
  entries = [known_peers[i] for i in ids]
  stale = await asyncio.gather(
    *[peer_is_stale(e[0], e[1], e[2], now, timeout) for e in entries],
    return_exceptions=True)
  for peer_id, bad in zip(ids, stale):
    if bad is True and peer_id in known_peers:
      del known_peers[peer_id]
      if debug >= 2:
        print(f"Removed peer {peer_id} due to inactivity or failed health check.")
