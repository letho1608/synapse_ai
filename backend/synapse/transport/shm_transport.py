"""
LACP — Zero-Copy Shared-Memory Transport for Same-Machine Nodes

Same-machine nodes exchange binary tensor frames through
multiprocessing.shared_memory segments instead of TCP sockets:

  - the sender writes the tensor's raw bytes straight into an mmap'd shared
    buffer with a single memcpy — no socket, no OS copy of the payload,
  - the receiver attaches the segment and hands the handler a zero-copy
    numpy view over that buffer (no payload copy on receive),
  - the only non-shared signal is a tiny in-process "segment ready"
    notification; the payload itself always travels through shared memory.

The public API mirrors TensorTCPTransport (register_peer /
set_tensor_handler / start / stop / send_tensor), so a node can select this
transport for same-machine peers and fall back to TCP for remote ones.
Same-process nodes on one machine are linked through a local registry:
attach_local() publishes this transport, and register_peer() automatically
routes notifications to the peer's transport when it is attached.
"""
from __future__ import annotations

import json
import socket
import struct
import uuid
from collections import OrderedDict
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, Optional

import numpy as np

from synapse.transport.codec import TensorMessage
from synapse.helpers import DEBUG

# Segment layout: header(16) | payload bytes | metadata JSON bytes
SHM_MAGIC = b"SYNSHM01"
SHM_HEADER = struct.Struct("!8sII")  # magic(8) + payload_len(4) + meta_len(4)

# Guard against absurd segment sizes (512 MiB max payload).
MAX_SHM_BYTES = 512 * 1024 * 1024


def host_of(addr: str) -> str:
    """Extract the host part of a 'host:port' address."""
    host = addr.rsplit(":", 1)[0] if ":" in addr else addr
    return host


def is_same_machine(addr_a: str, addr_b: str) -> bool:
    """True when two 'host:port' addresses resolve to the same host."""
    ha, hb = host_of(addr_a), host_of(addr_b)
    if ha == hb:
        return True
    local_aliases = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    if ha in local_aliases or hb in local_aliases:
        # Loopback addresses are always the local machine.
        return True
    try:
        return socket.gethostbyname(ha) == socket.gethostbyname(hb)
    except Exception:
        return False


class SharedMemoryTransport:
    """Zero-copy shared-memory transport for same-machine nodes."""

    # In-process registry: node_id -> transport (same-machine deployments).
    _local_registry: Dict[str, "SharedMemoryTransport"] = {}

    # Upper bound on concurrently retained segment handles: beyond this the
    # oldest segments are released so `_active` cannot grow without limit.
    DEFAULT_MAX_ACTIVE_SEGMENTS = 256

    def __init__(
        self,
        node_id: str,
        max_active_segments: int = DEFAULT_MAX_ACTIVE_SEGMENTS,
    ):
        self.node_id = node_id
        self._on_tensor: Optional[Callable[[TensorMessage], None]] = None
        # addr -> peer_id for registered same-machine peers.
        self._peer_ids: Dict[str, str] = {}
        # peer_id -> control notifier(segment_name).
        self._local_notify: Dict[str, Callable[[str], None]] = {}
        # Shared segments kept alive while their numpy views are in use,
        # most-recently-used last; evicts the oldest past the cap.
        self._active: "OrderedDict[str, Any]" = OrderedDict()
        self.max_active_segments = max_active_segments
        # Observability: segment names seen / sent (used by behavioral tests).
        self.segments_sent: list = []
        self.segments_seen: list = []
        self.shm_sends = 0
        self.shm_recvs = 0

    # -- lifecycle -------------------------------------------------------

    def attach_local(self) -> None:
        """Publish this transport so same-process peers can notify it."""
        SharedMemoryTransport._local_registry[self.node_id] = self

    @classmethod
    def detach_local(cls, node_id: str) -> None:
        cls._local_registry.pop(node_id, None)

    def set_tensor_handler(self, handler: Callable[[TensorMessage], None]) -> None:
        self._on_tensor = handler

    async def start(self) -> int:
        self.attach_local()
        return 0  # no socket or port needed

    async def stop(self) -> None:
        SharedMemoryTransport.detach_local(self.node_id)
        self._release_all()

    def _release_all(self) -> None:
        for name, shm in list(self._active.items()):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._active.clear()

    def _track_active(self, name: str, shm: Any) -> None:
        """Retain a segment handle, evicting the oldest when over the cap.

        The cap bounds `_active` (it never shrinks on its own otherwise):
        once `max_active_segments` handles are held, the least-recently-used
        segment is released. Eviction only closes/unlinks the local handle;
        the mapping stays alive while the peer (or an in-flight numpy view)
        still holds a handle to the same segment.
        """
        self._active[name] = shm
        self._active.move_to_end(name)
        while len(self._active) > self.max_active_segments:
            _old_name, old_shm = self._active.popitem(last=False)
            try:
                old_shm.close()
                old_shm.unlink()
            except Exception:
                pass

    def register_peer(self, addr: str, peer_id: str) -> None:
        """Register a same-machine peer.

        If the peer's transport is attached in this process, sends to it are
        routed through shared memory automatically.
        """
        self._peer_ids[addr] = peer_id
        peer_t = SharedMemoryTransport._local_registry.get(peer_id)
        if peer_t is not None:
            self._local_notify[peer_id] = peer_t._deliver

    def unregister_peer(self, addr: str) -> None:
        pid = self._peer_ids.pop(addr, None)
        if pid:
            self._local_notify.pop(pid, None)

    def is_local_peer(self, addr: str) -> bool:
        return addr in self._peer_ids

    # -- sending ---------------------------------------------------------

    async def send_tensor(
        self,
        request_id: str,
        target_stage: int,
        tensor: np.ndarray,
        target_addr: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Write the tensor into a shared segment and notify the local peer.

        Returns False when the target is not a registered same-machine peer
        (the caller should fall back to the TCP transport for remote peers).
        """
        if target_addr not in self._peer_ids:
            return False
        peer_id = self._peer_ids[target_addr]
        notifier = self._local_notify.get(peer_id)
        if notifier is None:
            return False

        payload = np.ascontiguousarray(tensor).tobytes()
        meta = dict(metadata or {})
        # Transport-level fields ride along in the metadata JSON: the receiver
        # reconstructs the TensorMessage from these keys alone.
        meta["request_id"] = request_id
        meta["target_stage"] = int(target_stage)
        meta["_dtype"] = str(tensor.dtype)
        meta["_shape"] = [int(d) for d in tensor.shape]
        meta_bytes = json.dumps(meta).encode("utf-8")

        size = SHM_HEADER.size + len(payload) + len(meta_bytes)
        if size > MAX_SHM_BYTES:
            if DEBUG >= 1:
                print(f"[SHM] Segment too large ({size} bytes), refusing")
            return False

        name = f"synapse_shm_{self.node_id}_{uuid.uuid4().hex}"
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        try:
            SHM_HEADER.pack_into(shm.buf, 0, SHM_MAGIC, len(payload), len(meta_bytes))
            shm.buf[SHM_HEADER.size:SHM_HEADER.size + len(payload)] = payload
            shm.buf[SHM_HEADER.size + len(payload):SHM_HEADER.size + len(payload) + len(meta_bytes)] = meta_bytes
        except Exception:
            shm.close()
            shm.unlink()
            raise

        # Keep the creator's handle open until the transport stops. On Windows
        # the named mapping is destroyed the moment its last handle closes, so
        # closing here (before the receiver attaches) would make the receiver's
        # attach fail with FileNotFoundError. Holding the handle keeps the
        # mapping alive for the segment's whole lifetime; _release_all() on
        # stop() closes (and on POSIX unlinks) every segment on both sides.
        self._track_active(name, shm)
        self.segments_sent.append(name)
        self.shm_sends += 1
        # Control notification only — the payload stays in shared memory.
        notifier(name)
        return True

    # -- receiving -------------------------------------------------------

    def _deliver(self, segment_name: str) -> None:
        """Attach a shared segment written by a local peer and dispatch it.

        The tensor handed to the handler is a zero-copy numpy view over the
        shared buffer; the segment is kept alive (and released on stop) so the
        view stays valid for the transport's lifetime.
        """
        self.segments_seen.append(segment_name)
        try:
            shm = shared_memory.SharedMemory(name=segment_name)
        except FileNotFoundError:
            return
        try:
            magic, payload_len, meta_len = SHM_HEADER.unpack_from(shm.buf, 0)
            if magic != SHM_MAGIC:
                if DEBUG >= 1:
                    print(f"[SHM] Bad magic in {segment_name}")
                return

            payload_start = SHM_HEADER.size
            meta_start = payload_start + payload_len
            meta_bytes = bytes(shm.buf[meta_start:meta_start + meta_len])
            try:
                meta = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}
            except Exception:
                meta = {}
            meta = meta or {}

            try:
                dtype = np.dtype(meta.get("_dtype", "float32"))
                shape = tuple(meta.get("_shape", [payload_len // dtype.itemsize]))
            except Exception:
                dtype, shape = np.dtype("float32"), (payload_len // 4,)

            # Zero-copy receive: numpy view over the shared buffer.
            view = memoryview(shm.buf)[payload_start:payload_start + payload_len]
            tensor = np.ndarray(shape=shape, dtype=dtype, buffer=view)

            # Keep the segment alive while the view is in use.
            self._track_active(segment_name, shm)

            msg = TensorMessage(
                request_id=str(meta.get("request_id", "")),
                orchestrator_id=str(meta.get("orchestrator_id", self.node_id)),
                micro_batch_id=int(meta.get("micro_batch_id", 0)),
                source_stage=int(meta.get("source_stage", 0)),
                target_stage=int(meta.get("target_stage", 0)),
                tensor=tensor,
                metadata=meta,
            )
            self.shm_recvs += 1
            if self._on_tensor is not None:
                self._on_tensor(msg)
        except Exception as e:
            if DEBUG >= 1:
                print(f"[SHM] Error processing segment {segment_name}: {e}")
