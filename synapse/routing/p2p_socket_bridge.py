import asyncio
import json
import struct
from typing import Dict, List, Set, Any, Optional, Tuple
from loguru import logger
from synapse.routing.event_router import EventRouter, Event

class P2PSocketBridge:
    """
    P2PSocketBridge provides a real TCP transport for the EventRouter.
    It allows multiple processes to share a common event-driven mesh.
    """
    def __init__(self, node_id: str, port: int, event_router: EventRouter):
        self.node_id = node_id
        self.port = port
        self.event_router = event_router
        self._running = False
        self._server: Optional[asyncio.AbstractServer] = None
        self._connections: Dict[str, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self._peer_addresses: Dict[str, str] = {} # node_id -> address
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the TCP server and join the mesh."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_incoming_connection, "0.0.0.0", self.port
        )
        
        # Subscribe to local events that need to be broadcasted
        self.event_router.subscribe("synapse/#", self._on_local_event)
        
        async with self._server:
            await self._server.serve_forever()

    async def stop(self):
        """Clean up connections and stop server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        for task in self._tasks:
            task.cancel()
        
        for writer in [w for _, w in self._connections.values()]:
            writer.close()
            await writer.wait_closed()
        
        logger.info("P2P Socket Bridge stopped.")

    async def connect_to_peer(self, peer_id: str, address: str):
        """Establish a permanent connection to a peer."""
        if peer_id == self.node_id: return
        if peer_id in self._connections: return
        
        # address format: "host:port"
        try:
            host, port = address.split(":")
            reader, writer = await asyncio.open_connection(host, int(port))
            
            # Send our handshake
            await self._send_handshake(writer)
            
            self._connections[peer_id] = (reader, writer)
            self._peer_addresses[peer_id] = address
            self._tasks.append(asyncio.create_task(self._read_loop(peer_id, reader)))
            logger.info(f"Connected to peer: {peer_id} at {address}")
        except (ConnectionRefusedError, OSError) as e:
            # WinError 1225 is often a ConnectionRefusedError on Windows
            if "1225" in str(e) or isinstance(e, ConnectionRefusedError):
                logger.warning(f"Connection refused by peer {peer_id} at {address}. Peer may be offline or starting up.")
            else:
                logger.error(f"Failed to connect to peer {peer_id} at {address}: {e}")
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id} at {address}: {e}")

    async def _handle_incoming_connection(self, reader, writer):
        """Handle a new incoming peer connection."""
        addr = writer.get_extra_info('peername')
        logger.debug(f"Incoming connection from {addr}")
        
        try:
            # 1. Wait for handshake
            peer_info = await self._read_packet(reader)
            if not peer_info or peer_info.get("type") != "handshake":
                logger.warning(f"Invalid handshake from {addr}")
                writer.close()
                return
            
            peer_id = peer_info.get("node_id")
            if not peer_id:
                writer.close()
                return

            if peer_id in self._connections:
                logger.debug(f"Peer {peer_id} already connected. Closing old connection.")
                _, old_writer = self._connections[peer_id]
                old_writer.close()

            self._connections[peer_id] = (reader, writer)
            self._tasks.append(asyncio.create_task(self._read_loop(peer_id, reader)))
            logger.info(f"Peer {peer_id} connected from {addr}")
            
            # 2. Send our handshake back
            await self._send_handshake(writer)
            
        except Exception as e:
            logger.error(f"Error handling incoming connection from {addr}: {e}")
            writer.close()

    async def _on_local_event(self, event: Event):
        """React to events published on the local EventRouter."""
        # Only broadcast events that originated locally to avoid loops
        if event.origin != "local" and event.origin != self.node_id:
            return
            
        # Don't broadcast UI-only or very high frequency internal events if not needed
        # (Though synapse/# covers everything including election)
        
        packet = {
            "type": "event",
            "topic": event.topic,
            "data": event.data,
            "origin": self.node_id,
            "timestamp": event.timestamp
        }
        
        await self._broadcast_packet(packet)

    async def _broadcast_packet(self, packet: Dict):
        """Send a packet to all connected peers."""
        data = json.dumps(packet).encode('utf-8')
        length_prefix = struct.pack("!I", len(data))
        
        disconnected = []
        for peer_id, (_, writer) in self._connections.items():
            try:
                writer.write(length_prefix + data)
                await writer.drain()
            except Exception as e:
                logger.warning(f"Lost connection to peer {peer_id}: {e}")
                disconnected.append(peer_id)
        
        for peer_id in disconnected:
            self._connections.pop(peer_id, None)

    async def _read_loop(self, peer_id: str, reader: asyncio.StreamReader):
        """Read packets from a specific peer stream."""
        try:
            while self._running:
                packet = await self._read_packet(reader)
                if packet is None: break
                
                if packet.get("type") == "event":
                    topic = packet.get("topic")
                    data = packet.get("data")
                    origin = packet.get("origin")
                    
                    # Prevent re-publishing back to the network (origin check)
                    # We publish locally with the remote origin ID
                    await self.event_router.publish(topic, data, origin=origin)
                    
                elif packet.get("type") == "ping":
                    # Potentially handle pings for health check
                    pass
        except Exception as e:
            logger.debug(f"Read loop for {peer_id} ended: {e}")
        finally:
            self._connections.pop(peer_id, None)

    async def _read_packet(self, reader: asyncio.StreamReader) -> Optional[Dict]:
        """Read a single length-prefixed JSON packet."""
        try:
            header = await reader.readexactly(4)
            length = struct.unpack("!I", header)[0]
            data = await reader.readexactly(length)
            return json.loads(data.decode('utf-8'))
        except (asyncio.IncompleteReadError, ConnectionError):
            return None
        except Exception as e:
            logger.error(f"Error reading packet: {e}")
            return None

    async def _send_handshake(self, writer: asyncio.StreamWriter):
        """Send identification handshake to a peer."""
        handshake = {
            "type": "handshake",
            "node_id": self.node_id,
            "port": self.port
        }
        data = json.dumps(handshake).encode('utf-8')
        writer.write(struct.pack("!I", len(data)) + data)
        await writer.drain()

