import asyncio
import json
import struct
import socket
import time
from typing import Dict, List, Set, Any, Optional, Tuple
from loguru import logger
from synapse.routing.event_router import EventRouter, Event

def _get_all_local_ips() -> Set[str]:
    """Get all IPv4 addresses of this machine to detect self-connections."""
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
    ips.add("127.0.0.1")  # Always include loopback
    return ips

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
        self._pending_handshake_futures: Dict[str, asyncio.Future] = {} # temp_id -> Future[str]
        self._temp_id_map: Dict[str, str] = {} # temp_id -> real_node_id
        # Heartbeat & reconnection state
        self._last_pong: Dict[str, float] = {}  # peer_id -> last pong timestamp
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = 5.0
        self._pong_timeout = 15.0
        self._reconnect_backoff: Dict[str, float] = {}  # peer_id -> current backoff seconds
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}  # peer_id -> reconnect task
        self._max_backoff = 60.0
        self._base_backoff = 1.0

    async def start(self):
        """Start the TCP server and join the mesh."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_incoming_connection, "0.0.0.0", self.port
        )

        # Subscribe to local events that need to be broadcasted
        self.event_router.subscribe("synapse/#", self._on_local_event)

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        async with self._server:
            await self._server.serve_forever()

    async def stop(self):
        """Clean up connections and stop server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        for task in self._tasks:
            task.cancel()
        for task in self._reconnect_tasks.values():
            task.cancel()

        for writer in [w for _, w in self._connections.values()]:
            writer.close()
        await asyncio.gather(*[w.wait_closed() for _, w in self._connections.values()], return_exceptions=True)

        logger.info("P2P Socket Bridge stopped.")

    async def connect_to_peer(self, peer_id: str, address: str, timeout: float = 5.0) -> bool:
        """Establish a permanent connection to a peer.
        Returns True if connection succeeded.
        """
        if peer_id == self.node_id: return True
        # Check if already connected and alive
        if peer_id in self._connections:
            _, writer = self._connections[peer_id]
            if not writer.is_closing():
                return True
            else:
                # Connection is dead, cleanup
                self._connections.pop(peer_id, None)

        # Also check temp-id mapping for stale temp-id entries
        if peer_id in self._temp_id_map:
            real_id = self._temp_id_map.pop(peer_id)
            self._connections.pop(peer_id, None)

        # Early self-connection detection: resolve host and compare to local IPs.
        # Only skip if BOTH IP is local AND port matches our own port.
        # When two nodes run on same machine, allow connecting to different-process nodes.
        try:
            _host, _port_str = address.split(":")
            resolved_ip = socket.gethostbyname(_host)
            target_port = int(_port_str)
            if resolved_ip in _get_all_local_ips() and target_port == self.port:
                logger.debug(f"Skipping self-connection to {address} (local IP + same port)")
                return False
        except Exception:
            pass

        # address format: "host:port"
        try:
            host, port_str = address.split(":")
            reader, writer = await asyncio.open_connection(host, int(port_str))

            # Send our handshake
            await self._send_handshake(writer)

            # Read peer's handshake with timeout to get real peer_id
            try:
                handshake_packet = await asyncio.wait_for(self._read_packet(reader), timeout=timeout)
                if not handshake_packet or handshake_packet.get("type") != "handshake":
                    logger.warning(f"Invalid or missing handshake from {peer_id} at {address}")
                    writer.close()
                    await writer.wait_closed()
                    return False
                actual_peer_id = handshake_packet.get("node_id")
            except asyncio.TimeoutError:
                logger.warning(f"Handshake timeout for {peer_id} at {address}")
                writer.close()
                await writer.wait_closed()
                return False
            except Exception as e:
                logger.error(f"Error during handshake with {peer_id} at {address}: {e}")
                writer.close()
                await writer.wait_closed()
                return False

            # Reject if handshake reveals this is a self-connection
            if actual_peer_id == self.node_id:
                logger.debug(f"Handshake resolved to self ({self.node_id}), closing self-connection to {address}")
                writer.close()
                await writer.wait_closed()
                return False

            # Clean up any stale temp-id entry for this peer
            stale_temps = [tid for tid, rid in list(self._temp_id_map.items())
                           if rid == actual_peer_id or self._peer_addresses.get(tid) == address]
            for stale_temp in stale_temps:
                self._temp_id_map.pop(stale_temp, None)
                if stale_temp in self._connections:
                    old_r, old_w = self._connections.pop(stale_temp)
                    old_w.close()

            # TIEBREAKER: Lower node_id's outgoing connection is canonical.
            # Both sides must use the same TCP connection for full-duplex comms.
            if self.node_id < actual_peer_id:
                # I have lower ID → my outgoing wins
                if actual_peer_id in self._connections:
                    _, old_writer = self._connections[actual_peer_id]
                    old_writer.close()
                self._connections[actual_peer_id] = (reader, writer)
                self._peer_addresses[actual_peer_id] = address
                self._tasks.append(asyncio.create_task(self._read_loop(actual_peer_id, reader)))
                self._mark_peer_alive(actual_peer_id)
                logger.info(f"Connected to peer (outgoing): {actual_peer_id} at {address}")
            else:
                # I have higher ID → prefer peer's outgoing connection
                if actual_peer_id in self._connections:
                    # Incoming already established (simultaneous connect)
                    writer.close()
                    await writer.wait_closed()
                    self._mark_peer_alive(actual_peer_id)
                    logger.info(f"Connected to peer (incoming): {actual_peer_id} at {address}")
                else:
                    # No incoming yet — keep my outgoing (single-direction / late join)
                    self._connections[actual_peer_id] = (reader, writer)
                    self._peer_addresses[actual_peer_id] = address
                    self._tasks.append(asyncio.create_task(self._read_loop(actual_peer_id, reader)))
                    self._mark_peer_alive(actual_peer_id)
                    logger.info(f"Connected to peer (outgoing, single-dir): {actual_peer_id} at {address}")

            return True
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id} at {address}: {e}")
            return False

    def is_peer_connected(self, peer_id: str) -> bool:
        """Check if a peer is currently connected (used by P2PPeerHandle)."""
        if peer_id not in self._connections:
            for temp_id, real_id in self._temp_id_map.items():
                if real_id == peer_id:
                    return peer_id in self._connections
            return False
        _, writer = self._connections[peer_id]
        return not writer.is_closing()

    def _mark_peer_alive(self, peer_id: str):
        """Reset heartbeat tracking for a peer after successful connection/pong."""
        self._last_pong[peer_id] = time.time()
        self._reconnect_backoff.pop(peer_id, None)

    async def _heartbeat_loop(self):
        """Periodically send pings and check for pong timeouts."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            if not self._running:
                break
            now = time.time()
            disconnected = []
            for peer_id, (_, writer) in list(self._connections.items()):
                try:
                    ping = {"type": "ping", "timestamp": now}
                    data = json.dumps(ping).encode("utf-8")
                    writer.write(struct.pack("!I", len(data)) + data)
                    await writer.drain()
                except Exception:
                    disconnected.append(peer_id)
                else:
                    last = self._last_pong.get(peer_id, now)
                    if now - last > self._pong_timeout:
                        logger.warning(f"Pong timeout for {peer_id} ({now - last:.0f}s)")
                        disconnected.append(peer_id)

            for peer_id in disconnected:
                self._connections.pop(peer_id, None)
                self._last_pong.pop(peer_id, None)
                self._schedule_reconnect(peer_id)

    def _schedule_reconnect(self, peer_id: str):
        """Schedule a reconnection attempt with exponential backoff."""
        if peer_id in self._reconnect_tasks and not self._reconnect_tasks[peer_id].done():
            return
        if peer_id not in self._peer_addresses:
            return
        task = asyncio.create_task(self._do_reconnect(peer_id))
        self._reconnect_tasks[peer_id] = task

    async def _do_reconnect(self, peer_id: str):
        """Attempt reconnection with exponential backoff."""
        backoff = self._reconnect_backoff.get(peer_id, self._base_backoff)
        while self._running and peer_id not in self._connections:
            await asyncio.sleep(backoff)
            if not self._running:
                return
            address = self._peer_addresses.get(peer_id)
            if not address:
                return
            logger.info(f"Reconnecting to {peer_id} at {address} (backoff={backoff:.1f}s)")
            try:
                ok = await self.connect_to_peer(peer_id, address)
                if ok:
                    logger.info(f"Reconnected to {peer_id}")
                    return
            except Exception as e:
                logger.debug(f"Reconnect failed for {peer_id}: {e}")
            backoff = min(backoff * 2, self._max_backoff)
            self._reconnect_backoff[peer_id] = backoff

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

            # Resolve pending handshake future for this specific peer (matched by temp_id)
            for temp_id, future in list(self._pending_handshake_futures.items()):
                if not future.done() and (self._temp_id_map.get(temp_id) == peer_id
                                           or self._peer_addresses.get(temp_id) == f"{addr[0]}:{addr[1]}"):
                    future.set_result(peer_id)

            # TIEBREAKER: Lower node_id's outgoing connection is canonical.
            # Both sides must use the same TCP connection for full-duplex comms.
            if self.node_id < peer_id:
                # I have lower ID → my outgoing should win.
                # Only store if connect_to_peer hasn't already stored the outgoing.
                if peer_id not in self._connections:
                    # Outgoing hasn't run yet, keep this incoming temporarily
                    self._connections[peer_id] = (reader, writer)
                    self._tasks.append(asyncio.create_task(self._read_loop(peer_id, reader)))
                    logger.debug(f"Peer {peer_id} connected (incoming, temporary) from {addr}")
                # Send handshake back so peer's connect_to_peer can complete
                await self._send_handshake(writer)
                return

            # I have higher ID → peer's outgoing is canonical
            # This incoming connection IS the canonical one
            if peer_id in self._connections:
                logger.debug(f"Peer {peer_id} already connected. Closing old (outgoing) connection.")
                _, old_writer = self._connections[peer_id]
                old_writer.close()

            self._connections[peer_id] = (reader, writer)
            self._tasks.append(asyncio.create_task(self._read_loop(peer_id, reader)))
            self._mark_peer_alive(peer_id)
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
            print(f"[DEBUG-BRIDGE] {self.node_id}: SKIP broadcast topic={event.topic} origin={event.origin} (not local, not self)", flush=True)
            return

        print(f"[DEBUG-BRIDGE] {self.node_id}: broadcasting topic={event.topic} origin={event.origin}", flush=True)
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
        topic = packet.get('topic', 'unknown')

        print(f"[DEBUG-BROADCAST] {self.node_id}: broadcasting '{topic}' to {len(self._connections)} peers: {list(self._connections.keys())}", flush=True)

        disconnected = []
        for peer_id, (_, writer) in self._connections.items():
            try:
                print(f"[DEBUG-BROADCAST] {self.node_id}: writing to {peer_id}, writer_closing={writer.is_closing()}, writer_addr={writer.get_extra_info('peername')}", flush=True)
                length_prefix = struct.pack("!I", len(data))
                writer.write(length_prefix + data)
                await writer.drain()
                print(f"[DEBUG-BROADCAST] {self.node_id}: drain complete for {peer_id}", flush=True)
            except Exception as e:
                print(f"[DEBUG-BROADCAST] {self.node_id}: ERROR writing to {peer_id}: {e}", flush=True)
                logger.warning(f"Lost connection to peer {peer_id}: {e}")
                disconnected.append(peer_id)

        for peer_id in disconnected:
            self._connections.pop(peer_id, None)
            self._last_pong.pop(peer_id, None)
            self._schedule_reconnect(peer_id)

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
                    print(f"[DEBUG-BRIDGE] {self.node_id}: _read_loop received from {peer_id} topic={topic} origin={origin}", flush=True)

                    # Prevent re-publishing back to the network (origin check)
                    await self.event_router.publish(topic, data, origin=origin)
                    print(f"[DEBUG-BRIDGE] {self.node_id}: _read_loop published locally topic={topic}", flush=True)

                elif packet.get("type") == "handshake":
                    # Handle handshake that arrives as a regular packet (shouldn't happen normally,
                    # but handle gracefully if peer sends it before we process connect)
                    pass

                elif packet.get("type") == "ping":
                    # Reply with pong
                    pong = {"type": "pong", "timestamp": packet.get("timestamp")}
                    data = json.dumps(pong).encode("utf-8")
                    stored = self._connections.get(peer_id)
                    if stored:
                        stored[1].write(struct.pack("!I", len(data)) + data)
                        await stored[1].drain()
                elif packet.get("type") == "pong":
                    self._last_pong[peer_id] = time.time()
        except Exception as e:
            logger.debug(f"Read loop for {peer_id} ended: {e}")
        finally:
            stored = self._connections.get(peer_id)
            if stored and stored[0] is reader:
                self._connections.pop(peer_id, None)
                self._last_pong.pop(peer_id, None)
                self._schedule_reconnect(peer_id)

    async def _read_packet(self, reader: asyncio.StreamReader) -> Optional[Dict]:
        """Read a single length-prefixed JSON packet."""
        try:
            header = await reader.readexactly(4)
            length = struct.unpack("!I", header)[0]
            # Sanity check on length to prevent memory issues
            if length > 10 * 1024 * 1024: # 10MB max
                logger.error(f"Packet too large: {length} bytes, dropping connection")
                return None
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