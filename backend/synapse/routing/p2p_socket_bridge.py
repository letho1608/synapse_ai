import asyncio
import json
import struct
import socket
import time
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from synapse.routing.event_router import EventRouter, Event
from synapse.helpers import DEBUG, get_all_local_ips as _get_all_local_ips

PACKET_TYPE_EVENT = 0x01
PACKET_TYPE_HANDSHAKE = 0x02
PACKET_TYPE_PING = 0x03
PACKET_TYPE_PONG = 0x04
PACKET_TYPE_ERROR = 0x05
# Reserved packet types (checkpoint control plane, never implemented on
# the wire). Kept registered so the handshake caps stay stable across peers.
PACKET_TYPE_CKPT_REQUEST = 0x06
PACKET_TYPE_CKPT_META = 0x07
PACKET_TYPE_CKPT_CHUNK = 0x08

# Wire-protocol versioning (T0.3, additive-only policy):
# - BRIDGE_VERSION bumps ONLY on breaking framing changes. A peer whose major
#   version differs must NOT form a distributed pipeline with us (fail closed
#   at the node layer via is_compatible(); the bridge itself stays tolerant).
# - features are capability flags; receivers ignore unknown flags.
# - New packet types register here with the minimum bridge version that
#   understands them; older peers reply error/unsupported, never silent-drop.
BRIDGE_VERSION = 1
FEATURE_SNP1 = "snp1"            # SYNP binary tensor payloads (codec.py)
FEATURE_CKPT_JSON1 = "ckpt-json1"  # JSON checkpoint P2P pull (fault_tolerance.py)
FEATURE_Q8ROW1 = "q8row1"        # per-row INT8 tensor payloads (codec int8_row)
DEFAULT_FEATURES = (FEATURE_SNP1, FEATURE_CKPT_JSON1, FEATURE_Q8ROW1)
PACKET_MIN_VERSION = {
    PACKET_TYPE_EVENT: 1,
    PACKET_TYPE_HANDSHAKE: 1,
    PACKET_TYPE_PING: 1,
    PACKET_TYPE_PONG: 1,
    PACKET_TYPE_ERROR: 1,
    PACKET_TYPE_CKPT_REQUEST: 1,
    PACKET_TYPE_CKPT_META: 1,
    PACKET_TYPE_CKPT_CHUNK: 1,
}

# Default ceiling for a single framed JSON packet on the event bridge.
# Raised from 10 MiB (which silently dropped frames for ~5 MB tensors after
# JSON+hex expansion) to a configurable 256 MiB so legitimate large frames
# are delivered; a frame that still exceeds the cap gets an explicit
# error packet back to the sender instead of a silent disconnect.
DEFAULT_MAX_FRAME_SIZE = 256 * 1024 * 1024


class P2PSocketBridge:
    """
    P2PSocketBridge provides a real TCP transport for the EventRouter.
    It allows multiple processes to share a common event-driven mesh.
    """
    def __init__(
        self, node_id: str, port: int, event_router: EventRouter,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        caps_hash: Optional[str] = None,
        features: Optional[List[str]] = None,
        ssl_server_ctx=None,
        ssl_client_ctx=None,
        tls_fp: Optional[str] = None,
        bind_host: str = "0.0.0.0",
    ):
        self.node_id = node_id
        self.port = port
        self.event_router = event_router
        self.max_frame_size = max_frame_size
        # Version/capability advertisement (T0.3). caps_hash is the sha256 of
        # the canonical DeviceCapabilities subset used by partitioning; it is
        # wired by the node layer (None until then) and never required.
        self.caps_hash = caps_hash
        self.features = list(features) if features is not None else list(DEFAULT_FEATURES)
        # TLS staging (Item 5): None = plaintext, byte-identical to before.
        # tls_fp carries the server-cert pin once TLS is on (else None).
        self._ssl_server_ctx = ssl_server_ctx
        self._ssl_client_ctx = ssl_client_ctx
        self.tls_fp = tls_fp
        # Set by the node once its binary tensor transport is listening;
        # advertised in the handshake so peers can reach it.
        self.tcp_port: Optional[int] = None
        # C-W3: bind address for the P2P listener (default all interfaces).
        self.bind_host = bind_host or "0.0.0.0"
        self._last_errors: Dict[str, Dict] = {}  # peer_id -> last error packet received
        # peer_id -> decoded handshake dict (versions/features/caps). Basis for
        # mixed-version fail-closed decisions at the node layer.
        self._peer_handshakes: Dict[str, Dict[str, Any]] = {}
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
        try:
            self._server = await asyncio.start_server(
                self._handle_incoming_connection, self.bind_host, self.port,
                ssl=self._ssl_server_ctx,
            )
        except OSError as e:
            # Port busy (usually a stale backend). One clear line + an
            # actionable error instead of a bare traceback downstream.
            from synapse.helpers import check_ports_free, format_port_conflicts
            conflicts = check_ports_free({"p2p": self.port})
            msg = format_port_conflicts(conflicts) if conflicts else (
                "Cannot bind P2P port %s on %s: %s" % (self.port, self.bind_host, e))
            logger.error(msg)
            self._running = False
            raise RuntimeError(msg) from e

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
        if peer_id == self.node_id:
            return True

        if peer_id in self._connections:
            _, writer = self._connections[peer_id]
            if not writer.is_closing():
                return True
            else:
                self._connections.pop(peer_id, None)

        if peer_id in self._temp_id_map:
            self._temp_id_map.pop(peer_id)
            self._connections.pop(peer_id, None)

        try:
            _host, _port_str = address.split(":")
            resolved_ip = socket.gethostbyname(_host)
            target_port = int(_port_str)
            if resolved_ip in _get_all_local_ips() and target_port == self.port:
                logger.debug(f"Skipping self-connection to {address} (local IP + same port)")
                return False
        except Exception:
            pass

        try:
            host, port_str = address.split(":")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(port_str),
                                        ssl=self._ssl_client_ctx),
                timeout=timeout
            )
            if self._ssl_client_ctx is not None:
                from synapse.transport.tls_wrap import check_peer_pin
                try:
                    check_peer_pin(writer)
                except Exception as e:
                    logger.error(f"TLS pin check failed for {peer_id} at {address}: {e}")
                    self._close_writer(writer)
                    return False
            self._enable_nodelay(writer)

            await self._send_handshake(writer, PACKET_TYPE_HANDSHAKE)

            try:
                handshake_packet = await asyncio.wait_for(self._read_packet(reader), timeout=timeout)
                if not handshake_packet or handshake_packet.get("type") != "handshake":
                    logger.warning(f"Invalid or missing handshake from {peer_id} at {address}")
                    self._close_writer(writer)
                    return False
                actual_peer_id = handshake_packet.get("node_id")
            except asyncio.TimeoutError:
                logger.warning(f"Handshake timeout for {peer_id} at {address}")
                self._close_writer(writer)
                return False
            except Exception as e:
                logger.error(f"Error during handshake with {peer_id} at {address}: {e}")
                self._close_writer(writer)
                return False

            if actual_peer_id:
                self.record_peer_handshake(actual_peer_id, handshake_packet or {})

            if actual_peer_id == self.node_id:
                logger.debug(f"Handshake resolved to self ({self.node_id}), closing self-connection to {address}")
                self._close_writer(writer)
                return False

            stale_temps = [tid for tid, rid in list(self._temp_id_map.items())
                           if rid == actual_peer_id or self._peer_addresses.get(tid) == address]
            for stale_temp in stale_temps:
                self._temp_id_map.pop(stale_temp, None)
                if stale_temp in self._connections:
                    old_r, old_w = self._connections.pop(stale_temp)
                    self._close_writer(old_w)

            if self.node_id < actual_peer_id:
                if actual_peer_id in self._connections:
                    _, old_writer = self._connections[actual_peer_id]
                    self._close_writer(old_writer)
                self._connections[actual_peer_id] = (reader, writer)
                self._peer_addresses[actual_peer_id] = address
                self._tasks.append(asyncio.create_task(self._read_loop(actual_peer_id, reader)))
                self._mark_peer_alive(actual_peer_id)
                logger.info(f"Connected to peer (outgoing): {actual_peer_id} at {address}")
            else:
                if actual_peer_id in self._connections:
                    self._close_writer(writer)
                    await self._close_writer_wait(writer)
                    self._mark_peer_alive(actual_peer_id)
                    logger.info(f"Connected to peer (incoming): {actual_peer_id} at {address}")
                else:
                    self._connections[actual_peer_id] = (reader, writer)
                    self._peer_addresses[actual_peer_id] = address
                    self._tasks.append(asyncio.create_task(self._read_loop(actual_peer_id, reader)))
                    self._mark_peer_alive(actual_peer_id)
                    logger.info(f"Connected to peer (outgoing, single-dir): {actual_peer_id} at {address}")

            return True
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id} at {address}: {e}")
            return False

    def _close_writer(self, writer: asyncio.StreamWriter):
        """Close a writer without awaiting wait_closed."""
        if writer and not writer.is_closing():
            writer.close()

    async def _close_writer_wait(self, writer: asyncio.StreamWriter):
        """Close a writer and wait for it to close."""
        if writer and not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

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
                    writer.write(struct.pack("!IB", len(data), PACKET_TYPE_PING) + data)
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

    def _enable_nodelay(self, writer) -> None:
        """Disable Nagle on a bridge socket (T1.2).

        The binary TCP transport sets TCP_NODELAY everywhere, but the live
        bridge never did — small control frames could stall ~40ms behind
        ACKs. One-liner, best-effort, never raises.
        """
        try:
            sock = writer.get_extra_info("socket") if writer else None
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    async def _handle_incoming_connection(self, reader, writer):
        """Handle a new incoming peer connection."""
        addr = writer.get_extra_info('peername')
        logger.debug(f"Incoming connection from {addr}")
        self._enable_nodelay(writer)

        try:
            # Peek the frame header ourselves so raw latency probes (which do
            # not speak the framed protocol) can be answered with an echo.
            header = await asyncio.wait_for(reader.readexactly(5), timeout=5.0)
            length = struct.unpack("!I", header[:4])[0]

            # LatencyProber compatibility: a probe is a bare PROBE_SIZE-byte
            # blob whose leading 4 bytes are the top of a packed "!dQ"
            # timestamp — always a huge number, never a valid frame length
            # (way past the configured cap). Echo the whole blob back so the
            # prober's RTT and bandwidth measurement succeed against the
            # production listener (previously the probe was dropped as
            # "Packet too large" and bandwidth stayed at its hardcoded default).
            if length <= 0 or length > self.max_frame_size:
                try:
                    from synapse.topology.latency_matrix import LatencyProber
                    probe_size = LatencyProber.PROBE_SIZE
                except Exception:
                    probe_size = 1024
                rest = await asyncio.wait_for(
                    reader.readexactly(probe_size - len(header)), timeout=5.0
                )
                writer.write(header + rest)
                await writer.drain()
                self._close_writer(writer)
                return

            payload = await asyncio.wait_for(reader.readexactly(length), timeout=5.0)
            peer_info = json.loads(payload.decode("utf-8"))
            if not peer_info or peer_info.get("type") != "handshake":
                logger.warning(f"Invalid handshake from {addr}")
                self._close_writer(writer)
                return

            peer_id = peer_info.get("node_id")
            if not peer_id:
                self._close_writer(writer)
                return

            # T0.3: record versions/features (old peers simply lack the keys).
            self.record_peer_handshake(peer_id, peer_info)

            for temp_id, future in list(self._pending_handshake_futures.items()):
                if not future.done() and (self._temp_id_map.get(temp_id) == peer_id
                                           or self._peer_addresses.get(temp_id) == f"{addr[0]}:{addr[1]}"):
                    future.set_result(peer_id)

            if self.node_id < peer_id:
                if peer_id not in self._connections:
                    self._connections[peer_id] = (reader, writer)
                    self._tasks.append(asyncio.create_task(self._read_loop(peer_id, reader)))
                    logger.debug(f"Peer {peer_id} connected (incoming, temporary) from {addr}")
                await self._send_handshake(writer, PACKET_TYPE_HANDSHAKE)
                return

            if peer_id in self._connections:
                logger.debug(f"Peer {peer_id} already connected. Closing old (outgoing) connection.")
                _, old_writer = self._connections[peer_id]
                self._close_writer(old_writer)

            self._connections[peer_id] = (reader, writer)
            self._tasks.append(asyncio.create_task(self._read_loop(peer_id, reader)))
            self._mark_peer_alive(peer_id)
            logger.info(f"Peer {peer_id} connected from {addr}")

            await self._send_handshake(writer, PACKET_TYPE_HANDSHAKE)

        except Exception as e:
            logger.error(f"Error handling incoming connection from {addr}: {e}")
            self._close_writer(writer)

    async def _on_local_event(self, event: Event):
        """React to events published on the local EventRouter."""
        # CRITICAL FIX: Only broadcast events that originated locally (from this node's own code
        # or from this node's EventRouter.publish with origin=self.node_id).
        # Events received from remote peers arrive via _read_loop with origin=remote_node_id,
        # and those MUST NOT be re-broadcast (that would create infinite loops).
        if event.origin != "local" and event.origin != self.node_id:
            if DEBUG >= 3:
                logger.debug(f"SKIP broadcast topic={event.topic} origin={event.origin} (remote event, not re-broadcasting)")
            return

        if DEBUG >= 2:
            logger.debug(f"Broadcasting topic={event.topic} origin={event.origin}")

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
        data = json.dumps(packet, default=self._json_serialize).encode('utf-8')

        disconnected = []
        for peer_id, (_, writer) in self._connections.items():
            try:
                writer.write(struct.pack("!IB", len(data), PACKET_TYPE_EVENT) + data)
                await writer.drain()
            except Exception as e:
                logger.warning(f"Lost connection to peer {peer_id}: {e}")
                disconnected.append(peer_id)

        for peer_id in disconnected:
            self._connections.pop(peer_id, None)
            self._last_pong.pop(peer_id, None)
            self._schedule_reconnect(peer_id)

    @staticmethod
    def _json_serialize(obj):
        """Custom JSON serializer for numpy arrays and bytes."""
        import numpy as np

        if isinstance(obj, np.ndarray):
            return {
                "__np__": True,
                "data": obj.tobytes().hex(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, bytes):
            # Tag raw bytes so the receiver can restore the exact bytes object.
            # Plain obj.hex() is NOT reversible: the receiver would see a hex
            # string and feed it to decode_tensor() (which expects bytes),
            # silently dropping every binary tensor sent between nodes.
            return {"__bytes__": True, "data": obj.hex()}
        # Topology payloads carry DeviceCapabilities/DeviceFlops objects; the
        # topology broadcast failed on every attempt without this.
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if hasattr(obj, "model_dump") and callable(obj.model_dump):  # pydantic
            return obj.model_dump()
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def _read_loop(self, peer_id: str, reader: asyncio.StreamReader):
        """Read packets from a specific peer stream."""
        try:
            while self._running:
                try:
                    stored = self._connections.get(peer_id)
                    writer = stored[1] if stored else None
                    packet, pkt_type = await self._read_packet_with_type(reader, writer)
                    if packet is None:
                        break

                    if pkt_type == PACKET_TYPE_ERROR:
                        self._last_errors[peer_id] = packet
                        logger.warning(
                            f"Peer {peer_id} rejected a frame: {packet.get('reason')} "
                            f"(size={packet.get('size')}, max={packet.get('max_frame_size')})"
                        )
                    elif pkt_type == PACKET_TYPE_PING:
                        pong = {"type": "pong", "timestamp": packet.get("timestamp")}
                        data = json.dumps(pong).encode("utf-8")
                        stored = self._connections.get(peer_id)
                        if stored:
                            stored[1].write(struct.pack("!IB", len(data), PACKET_TYPE_PONG) + data)
                            await stored[1].drain()
                    elif pkt_type == PACKET_TYPE_PONG:
                        self._last_pong[peer_id] = time.time()
                    elif pkt_type == PACKET_TYPE_EVENT:
                        topic = packet.get("topic")
                        data = self._json_deserialize(packet.get("data"))
                        origin = packet.get("origin")
                        await self.event_router.publish(topic, data, origin=origin)
                except Exception as e:
                    logger.debug(f"Read loop for {peer_id} ended: {e}")
                    break
        except Exception as e:
            logger.debug(f"Read loop for {peer_id} ended: {e}")
        finally:
            stored = self._connections.get(peer_id)
            if stored and stored[0] is reader:
                self._connections.pop(peer_id, None)
                self._last_pong.pop(peer_id, None)
                self._schedule_reconnect(peer_id)

    @staticmethod
    def _json_deserialize(data):
        """Recursively convert serialized numpy arrays/bytes back from hex."""
        import numpy as np

        if isinstance(data, dict):
            if data.get("__np__"):
                arr = np.frombuffer(bytes.fromhex(data["data"]), dtype=np.dtype(data["dtype"]))
                return arr.reshape(data["shape"])
            if data.get("__bytes__"):
                return bytes.fromhex(data["data"])
            return {k: P2PSocketBridge._json_deserialize(v) for k, v in data.items()}
        if isinstance(data, list):
            return [P2PSocketBridge._json_deserialize(item) for item in data]
        return data

    async def _read_packet(self, reader: asyncio.StreamReader) -> Optional[Dict]:
        """Read a single length-prefixed JSON packet (legacy compatibility)."""
        packet, _ = await self._read_packet_with_type(reader)
        return packet

    async def _read_packet_with_type(
        self, reader: asyncio.StreamReader, writer: Optional[asyncio.StreamWriter] = None
    ) -> Tuple[Optional[Dict], int]:
        """
        Read a single length-prefixed JSON packet with type byte.
        Returns (packet_dict, packet_type).
        Uses [4 bytes length][1 byte type][payload] format.
        """
        try:
            header = await reader.readexactly(5)
            length = struct.unpack("!I", header[:4])[0]
            pkt_type = header[4]
            if length > self.max_frame_size:
                logger.warning(
                    f"Frame too large: {length} bytes exceeds cap "
                    f"{self.max_frame_size} bytes, sending explicit error to sender "
                    f"instead of silently dropping"
                )
                await self._send_error(
                    writer, "frame_too_large", length, self.max_frame_size
                )
                return None, 0
            data = await reader.readexactly(length)
            packet = json.loads(data.decode('utf-8'))
            return packet, pkt_type
        except (asyncio.IncompleteReadError, ConnectionError):
            return None, 0
        except Exception as e:
            logger.error(f"Error reading packet: {e}")
            return None, 0

    async def _send_error(
        self,
        writer: Optional[asyncio.StreamWriter],
        reason: str,
        size: int,
        max_frame_size: int,
    ):
        """Send an explicit error packet back to a peer instead of silently dropping."""
        if writer is None or writer.is_closing():
            return
        try:
            error_pkt = {
                "type": "error",
                "reason": reason,
                "size": size,
                "max_frame_size": max_frame_size,
            }
            data = json.dumps(error_pkt).encode("utf-8")
            writer.write(struct.pack("!IB", len(data), PACKET_TYPE_ERROR) + data)
            await writer.drain()
        except Exception as e:
            logger.debug(f"Failed to send error packet: {e}")

    async def _send_handshake(self, writer: asyncio.StreamWriter, pkt_type: int = PACKET_TYPE_HANDSHAKE):
        """Send identification handshake to a peer.

        Additive-only (T0.3): old peers read node_id/port via .get() and
        ignore the rest; new peers record versions/features/caps for
        mixed-version fail-closed decisions. Never remove a key.
        """
        from synapse.helpers import VERSION as _SYNAPSE_VERSION
        handshake = {
            "type": "handshake",
            "node_id": self.node_id,
            "port": self.port,
            "synapse_version": _SYNAPSE_VERSION,
            "bridge_version": BRIDGE_VERSION,
            "caps_hash": self.caps_hash,
            "features": list(self.features),
            # Reserved for TLS pinning (transport/tls_wrap.py). Advertised
            # from the bridge's configured pin; unenforced while --tls off.
            "tls_fp": self.tls_fp,
            # Port of this node's binary tensor transport (None = bridge only).
            "tcp_port": getattr(self, "tcp_port", None),
        }
        data = json.dumps(handshake).encode('utf-8')
        writer.write(struct.pack("!IB", len(data), pkt_type) + data)
        await writer.drain()

    def record_peer_handshake(self, peer_id: str, info: Dict[str, Any]) -> None:
        """Store a peer's decoded handshake; returns nothing.

        Unknown keys are kept (forward-compat); missing version keys mean a
        pre-versioning peer, which stays compatible (legacy fallback).
        """
        if not isinstance(info, dict):
            return
        self._peer_handshakes[peer_id] = {
            "node_id": info.get("node_id", peer_id),
            "port": info.get("port"),
            "synapse_version": info.get("synapse_version"),
            "bridge_version": info.get("bridge_version"),
            "caps_hash": info.get("caps_hash"),
            "features": list(info.get("features") or []),
            "tls_fp": info.get("tls_fp"),
            "tcp_port": info.get("tcp_port"),
        }
        try:
            bv = self._peer_handshakes[peer_id]["bridge_version"]
            if bv is not None and int(bv) != BRIDGE_VERSION:
                logger.warning(
                    f"Peer {peer_id} bridge_version={bv} != ours={BRIDGE_VERSION}: "
                    "distributed pipeline with this peer must refuse (fail closed)"
                )
        except Exception:
            pass

    def get_peer_handshake(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Return the recorded handshake for a peer, or None if unknown."""
        info = self._peer_handshakes.get(peer_id)
        return dict(info) if info is not None else None

    def is_compatible(self, peer_id: str) -> bool:
        """Fail-closed compatibility: False only on bridge major mismatch.

        Unknown peers and pre-versioning peers (bridge_version None) are
        treated as compatible (legacy fallback); the node layer decides the
        pipeline from this plus feature intersection.
        """
        info = self._peer_handshakes.get(peer_id)
        if not info:
            return True
        bv = info.get("bridge_version")
        if bv is None:
            return True
        try:
            return int(bv) == BRIDGE_VERSION
        except Exception:
            return False

    def negotiated_features(self, peer_id: str) -> List[str]:
        """Feature intersection with a peer (empty when unknown)."""
        info = self._peer_handshakes.get(peer_id) or {}
        theirs = set(info.get("features") or [])
        return sorted(set(self.features) & theirs)