"""
LACP — Direct TCP Transport for Binary Tensor Forwarding

Provides a direct TCP socket transport alongside EventRouter for
high-throughput tensor data. Uses the binary codec from codec.py.

Architecture:
- Each node runs a TCP server on a configurable port
- Outbound connections are pooled per peer
- Tensor data is sent as binary frames (no JSON serialization)
- Control messages (metadata, routing info) still use EventRouter
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple
from collections import deque

from synapse.transport.codec import (
    AsyncTensorForwarder,
    TensorMessage,
    encode_tensor,
    decode_tensor,
)
from synapse.helpers import DEBUG


# Frame format: 4-byte length prefix + binary payload
FRAME_HEADER = struct.Struct("!I")


class TCPServer:
    """Per-node TCP server that receives binary tensor frames."""

    def __init__(self, host: str = "0.0.0.0", port: int = 0):
        self.host = host
        self.port = port
        self._server: Optional[asyncio.AbstractServer] = None
        self._on_frame: Optional[Callable[[bytes], None]] = None

    def set_frame_handler(self, handler: Callable[[bytes], None]):
        self._on_frame = handler

    async def start(self) -> int:
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port,
        )
        for sock in self._server.sockets:
            self.port = sock.getsockname()[1]
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            break
        if DEBUG >= 1:
            print(f"[TCP] Server listening on {self.host}:{self.port}")
        return self.port

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        peer = writer.get_extra_info("peername")
        try:
            sock = writer.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        try:
            while True:
                header = await reader.readexactly(FRAME_HEADER.size)
                length = FRAME_HEADER.unpack(header)[0]
                if length > 100 * 1024 * 1024:
                    if DEBUG >= 1:
                        print(f"[TCP] Frame too large ({length} bytes), dropping")
                    break
                data = await reader.readexactly(length)
                if self._on_frame:
                    self._on_frame(data)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


class TCPClientPool:
    """Pool of outbound TCP connections per peer."""

    def __init__(self, connect_timeout: float = 5.0):
        self._conns: Dict[str, Tuple[asyncio.StreamWriter, float]] = {}
        self._connect_timeout = connect_timeout
        self._lock = asyncio.Lock()

    async def send(self, peer_addr: str, data: bytes) -> bool:
        host, port_str = peer_addr.rsplit(":", 1)
        port = int(port_str)
        writer = await self._get_connection(host, port)
        if writer is None:
            return False
        try:
            frame = FRAME_HEADER.pack(len(data)) + data
            writer.write(frame)
            await writer.drain()
            return True
        except Exception:
            self._conns.pop(peer_addr, None)
            return False

    async def _get_connection(
        self, host: str, port: int
    ) -> Optional[asyncio.StreamWriter]:
        key = f"{host}:{port}"
        entry = self._conns.get(key)
        if entry is not None:
            writer, _ = entry
            if not writer.is_closing():
                return writer
            self._conns.pop(key, None)

        async with self._lock:
            entry = self._conns.get(key)
            if entry is not None:
                writer, _ = entry
                if not writer.is_closing():
                    return writer
            try:
                writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self._connect_timeout,
                )
                writer = writer[1]
                try:
                    sock = writer.get_extra_info("socket")
                    if sock is not None:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                self._conns[key] = (writer, time.time())
                return writer
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                return None

    async def close_all(self):
        for key, (writer, _) in list(self._conns.items()):
            try:
                writer.close()
            except Exception:
                pass
        self._conns.clear()


class TensorTCPTransport:
    """
    Integrates binary TCP transport with AsyncTensorForwarder.

    Usage:
        transport = TensorTCPTransport(node_id)
        await transport.start()

        # Register writers for each peer
        transport.register_peer("peer_addr:port", peer_id)

        # Send tensor via binary TCP
        await transport.send_tensor(request_id, target_stage, tensor, metadata)
    """

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = 0,
    ):
        self.node_id = node_id
        self.server = TCPServer(host, port)
        self.client_pool = TCPClientPool()
        self.forwarder = AsyncTensorForwarder(node_id)
        self._peer_map: Dict[str, str] = {}  # peer_addr -> peer_id

        # Inbound handler
        self._on_tensor: Optional[Callable[[TensorMessage], None]] = None

    def set_tensor_handler(self, handler: Callable[[TensorMessage], None]):
        self._on_tensor = handler

    async def start(self) -> int:
        self.server.set_frame_handler(self._on_frame)
        return await self.server.start()

    async def stop(self):
        await self.server.stop()
        await self.client_pool.close_all()

    def register_peer(self, addr: str, peer_id: str):
        self._peer_map[addr] = peer_id
        self.forwarder.register_writer("__control__", 0, lambda d: self._raw_send(addr, d))

    def _on_frame(self, data: bytes):
        try:
            msg = self.forwarder.receive("__inbound__", 0, data)
            if msg is not None and self._on_tensor:
                self._on_tensor(msg)
        except Exception as e:
            if DEBUG >= 2:
                print(f"[TCP] Error processing frame: {e}")

    async def _raw_send(self, addr: str, data: bytes) -> bool:
        return await self.client_pool.send(addr, data)

    async def send_tensor(
        self,
        request_id: str,
        target_stage: int,
        tensor: Any,
        target_addr: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        if target_addr not in self._peer_map:
            return False

        msg = TensorMessage(
            request_id=request_id,
            orchestrator_id=self.node_id,
            micro_batch_id=metadata.get("micro_batch_id", 0) if metadata else 0,
            source_stage=metadata.get("source_stage", 0) if metadata else 0,
            target_stage=target_stage,
            tensor=tensor,
            metadata=metadata,
        )

        # Register writer for this request if not already
        key = (request_id, target_stage)
        if key not in self.forwarder._writers:
            self.forwarder.register_writer(
                request_id, target_stage,
                lambda d, addr=target_addr: self._raw_send(addr, d),
            )

        return await self.forwarder.send(msg)