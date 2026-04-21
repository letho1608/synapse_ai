import asyncio
from typing import List, Callable, Optional, Any
from loguru import logger
import json

from synapse.routing.p2p_socket_bridge import P2PSocketBridge

class Libp2pNode:
    """
    Libp2pNode manages the P2P Mesh connectivity.
    It integrates with Tailscale for peer discovery and uses P2PSocketBridge 
    for cross-process event relay.
    """
    def __init__(self, node_id: str, port: int, event_router: Any):
        self.node_id = node_id
        self.port = port
        self.event_router = event_router
        self._bridge = P2PSocketBridge(node_id, port, event_router)
        self._running = False

    async def start(self, bootstrap_peers: List[str] = None):
        """
        Starts the P2P node and the socket bridge.
        """
        self._running = True
        
        # Start the background socket bridge
        asyncio.create_task(self._bridge.start())
        
        if bootstrap_peers:
            for peer in bootstrap_peers:
                await self.connect_to_peer(peer)

    async def connect_to_peer(self, peer_addr: str):
        """Establish connection to a specific peer."""
        if not self._running: return
        
        # peer_addr is expected to be "host:port"
        # We need a peer_id as well. In this simplified version, we can use 
        # a temporary ID that will be updated during handshake.
        # But discovery modules usually provide ID.
        # For now, let's assume P2PSocketBridge.connect_to_peer takes host:port
        # and manages IDs from handshake.
        
        # Refactoring connect_to_peer to be simpler
        try:
            # We don't have peer_id yet, let's use the address as key
            await self._bridge.connect_to_peer(f"temp-{peer_addr}", peer_addr)
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_addr}: {e}")

    async def publish(self, topic: str, message: Any):
        """Publish a message to the mesh via the bridge."""
        if not self._running:
            return
        
        # Locally bridging to EventRouter first, which then triggers bridge broadcast
        await self.event_router.publish(topic, message, origin=self.node_id)

    async def stop(self):
        """Shut down the P2P node and bridge."""
        self._running = False
        await self._bridge.stop()
        logger.info("P2P Node stopped.")
