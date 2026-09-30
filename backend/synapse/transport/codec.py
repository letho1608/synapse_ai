"""
LACP — Binary Tensor Codec & Async Forwarding Engine

High-performance tensor transport for distributed pipeline parallelism.

Codec:
  - Custom binary header + numpy raw bytes (no JSON, no tolist())
  - Compact 16-byte header: magic(4) + dtype(2) + ndim(2) + reserved(8)
  - Supports zero-copy shared memory for same-machine transfers
  - Optional compression: lz4/zstd for cross-machine high-throughput

Async Forwarder:
  - Non-blocking sends: fire-and-forget, next batch starts immediately
  - Queue-based receive: incoming tensors buffered, processor dequeues
  - Backpressure: flow control via queue depth limits
  - Result routing: (request_id, orchestrator_id, micro_batch_id) tuple
    ensures in-order reassembly at the caller
"""

from __future__ import annotations

import asyncio
import struct
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple
import numpy as np

# ──────────────────────────────────────────────────────────
# Binary Codec
# ──────────────────────────────────────────────────────────

MAGIC = b"SYNP"  # Synapse Protocol magic bytes
HEADER_SIZE = 16  # 4 magic + 2 dtype + 2 ndim + 8 reserved

# ── Chunked transfer (T1.2) ──────────────────────────────────────
# Large SYNP blobs are split into 1 MiB chunks so no single frame ever
# approaches a transport cap. Chunk wire format (after the transport's own
# length prefix): CHUNK_MAGIC(4) + transfer_id(16) + index(U32) +
# total(U32) + flags(U8, FINAL=0x01) + payload_len(U32) + payload.
CHUNK_MAGIC = b"SYNC"
CHUNK_SIZE = 1024 * 1024  # 1 MiB payload per chunk
CHUNK_HEADER = struct.Struct("!16sIIBI")
CHUNK_HEADER_SIZE = CHUNK_HEADER.size  # 29
CHUNK_FLAG_FINAL = 0x01
# Negotiated admission: refuse a transfer whose declared total exceeds this
# instead of buffering unboundedly (DoS guard, H5).
MAX_BLOB_BYTES = 4 * 1024 * 1024 * 1024
CHUNK_TIMEOUT_S = 30.0


class ChunkRefused(Exception):
    """Raised when a chunked transfer is refused (oversize admission)."""


def split_blob(blob: bytes, transfer_id: bytes,
               chunk_size: int = CHUNK_SIZE) -> list:
    """Split a blob into chunk frames. transfer_id must be 16 bytes."""
    if len(transfer_id) != 16:
        raise ValueError("transfer_id must be 16 bytes")
    if not blob:
        raise ValueError("cannot chunk an empty blob")
    total = (len(blob) + chunk_size - 1) // chunk_size
    if len(blob) > MAX_BLOB_BYTES:
        raise ChunkRefused("blob %d bytes exceeds admission %d" % (len(blob), MAX_BLOB_BYTES))
    frames = []
    for idx in range(total):
        part = blob[idx * chunk_size:(idx + 1) * chunk_size]
        flags = CHUNK_FLAG_FINAL if idx == total - 1 else 0
        frames.append(CHUNK_MAGIC + CHUNK_HEADER.pack(
            transfer_id, idx, total, flags, len(part)) + part)
    return frames


def parse_chunk_frame(frame: bytes):
    """Parse one chunk frame -> (transfer_id, idx, total, is_final, payload).

    Raises ValueError when the frame is not a chunk frame.
    """
    if len(frame) < 4 + CHUNK_HEADER_SIZE or frame[:4] != CHUNK_MAGIC:
        raise ValueError("not a chunk frame")
    transfer_id, idx, total, flags, payload_len = CHUNK_HEADER.unpack(
        frame[4:4 + CHUNK_HEADER_SIZE])
    payload = frame[4 + CHUNK_HEADER_SIZE:]
    if len(payload) != payload_len:
        raise ValueError("chunk payload length mismatch (%d != %d)"
                         % (len(payload), payload_len))
    if total <= 0 or idx >= total:
        raise ValueError("bad chunk index/total (%d/%d)" % (idx, total))
    if total * CHUNK_SIZE > MAX_BLOB_BYTES + CHUNK_SIZE:
        raise ChunkRefused("declared transfer exceeds admission")
    return transfer_id, idx, total, bool(flags & CHUNK_FLAG_FINAL), payload


class ChunkReassembler:
    """Reassemble chunked blobs. One instance per receiver.

    Missing-chunk timeout discards the buffer (never stalls the pipeline);
    oversize declarations are refused up front (never pre-allocated).
    """

    def __init__(self, timeout_s: float = CHUNK_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._buf: dict = {}  # transfer_id -> {"total":int,"parts":{int:bytes},"ts":float}

    def add(self, transfer_id: bytes, idx: int, total: int,
            _is_final: bool, payload: bytes):
        """Add a chunk. Returns the complete blob when all parts arrived."""
        now = time.time()
        self.purge(now)
        entry = self._buf.get(transfer_id)
        if entry is None:
            if total * CHUNK_SIZE > MAX_BLOB_BYTES + CHUNK_SIZE:
                raise ChunkRefused("declared transfer exceeds admission")
            entry = {"total": total, "parts": {}, "ts": now}
            self._buf[transfer_id] = entry
        if entry["total"] != total:
            return None  # inconsistent retransmit; ignore
        if idx not in entry["parts"]:
            entry["parts"][idx] = payload
            entry["ts"] = now
        if len(entry["parts"]) == entry["total"]:
            ordered = b"".join(entry["parts"][i] for i in range(entry["total"]))
            del self._buf[transfer_id]
            return ordered
        return None

    def purge(self, now: Optional[float] = None) -> int:
        """Discard timed-out transfers. Returns count discarded."""
        now = now if now is not None else time.time()
        stale = [tid for tid, e in self._buf.items()
                 if now - e["ts"] > self.timeout_s]
        for tid in stale:
            del self._buf[tid]
        return len(stale)

    @property
    def pending(self) -> int:
        return len(self._buf)

# dtype encoding table
DTYPE_MAP = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 1,
    np.dtype("int32"): 2,
    np.dtype("int64"): 3,
    np.dtype("int8"): 4,
    np.dtype("uint8"): 5,
    np.dtype("bool"): 6,
}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


Q_MODE_INT8_ROW = "int8_row"


# Largest-|x| entries per row kept exactly (float32) instead of quantized.
# LLM hidden states carry a few huge "massive activation" channels; with a
# plain per-row scale they flatten every other channel (measured on
# SmolLM2-135M, 2-stage split, teacher-forced: top-1 agreement 0.953 /
# KL 8.8e-3 without outliers vs 0.99+ / ~1e-4 with 4-16 kept).
Q_OUTLIERS_PER_ROW = 8


def _quantize_rows_int8(tensor: np.ndarray, k: int = Q_OUTLIERS_PER_ROW):
    """Symmetric INT8, one float32 scale per row of the last axis, with the
    k largest-magnitude entries of each row carried exactly.

    Returns (q, payload_tail, k): q int8 in the input shape; payload_tail =
    scales f32[rows] + outlier indices u32[rows, k] + values f32[rows, k].
    """
    x = np.asarray(tensor, dtype=np.float32)
    rows = x.reshape(1, -1) if x.ndim <= 1 else x.reshape(-1, x.shape[-1])
    n_rows, width = rows.shape
    k = max(0, min(int(k), width))
    rows = rows.copy()
    if k and n_rows:
        idx = np.argpartition(-np.abs(rows), k - 1, axis=1)[:, :k].astype(np.uint32)
        vals = np.take_along_axis(rows, idx.astype(np.intp), 1).astype(np.float32)
        np.put_along_axis(rows, idx.astype(np.intp), 0.0, 1)
    else:
        idx = np.zeros((n_rows, 0), dtype=np.uint32)
        vals = np.zeros((n_rows, 0), dtype=np.float32)
    scales = (np.abs(rows).max(axis=1) / 127.0).astype(np.float32) if rows.size else \
        np.ones(n_rows, dtype=np.float32)
    scales[~(scales > 0)] = 1.0  # all-zero (or NaN) rows
    q = np.rint(rows / scales[:, None]).clip(-127, 127).astype(np.int8)
    tail = scales.tobytes() + idx.tobytes() + vals.tobytes()
    return q.reshape(x.shape), tail, k


def _dequantize_rows_int8(q: np.ndarray, tail: bytes, k: int, shape, orig_dtype: str) -> np.ndarray:
    count = q.size
    width = shape[-1] if len(shape) >= 1 and shape[-1] else max(count, 1)
    n_rows = 1 if len(shape) <= 1 else (count // width if width else 0)
    if count == 0 or n_rows == 0:
        return np.zeros(shape, dtype=np.dtype(orig_dtype))
    scales = np.frombuffer(tail, dtype=np.float32, count=n_rows)
    off = 4 * n_rows
    idx = np.frombuffer(tail, dtype=np.uint32, count=n_rows * k, offset=off).reshape(n_rows, k)
    off += 4 * n_rows * k
    vals = np.frombuffer(tail, dtype=np.float32, count=n_rows * k, offset=off).reshape(n_rows, k)
    rows = q.astype(np.float32).reshape(n_rows, -1) * scales[:, None]
    if k:
        np.put_along_axis(rows, idx.astype(np.intp), vals, 1)
    return rows.reshape(shape).astype(np.dtype(orig_dtype))


def encode_tensor(
    tensor: np.ndarray,
    metadata: Optional[Dict] = None,
    compress: bool = False,
    compress_level: int = 1,
    quantize_int8: bool = False,
) -> bytes:
    """
    Encode a numpy tensor into a compact binary format.
    Supports optional INT8 quantization for transmission.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(tensor)}")

    target_tensor = tensor
    q_tail: Optional[bytes] = None
    q_k = 0

    # INT8 wire quantization, one symmetric scale per row (per token for
    # [..., seq, hidden] activations) plus exact outliers: a single
    # per-tensor scale let one outlier flush every other value to 0
    # (teacher-forced top-1 agreement ~1-6% on a real model). Rounded, not
    # truncated. Scales/outliers travel after the int8 data (see _q_mode).
    # Per-row quantization is deterministic per row, so prefix rows stay
    # byte-identical across decode steps (downstream KV reuse keeps working).
    if quantize_int8 and tensor.dtype in (np.float32, np.float16, np.float64):
        target_tensor, q_tail, q_k = _quantize_rows_int8(tensor)

    dtype_code = DTYPE_MAP.get(target_tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported dtype: {target_tensor.dtype}")

    ndim = len(target_tensor.shape)
    if ndim > 65535:
        raise ValueError(f"Too many dimensions: {ndim}")

    raw_data = target_tensor.tobytes()
    if q_tail is not None:
        raw_data += q_tail
    uncompressed_size = len(raw_data)

    if compress and uncompressed_size > 4096:
        try:
            import lz4.frame
            compressed_data = lz4.frame.compress(raw_data, compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
            is_compressed_flag = 2 # 2 means LZ4
        except ImportError:
            compressed_data = zlib.compress(raw_data, compress_level)
            is_compressed_flag = 1 # 1 means zlib
            
        compressed_size = len(compressed_data)
        if compressed_size < uncompressed_size:
            data_payload = compressed_data
        else:
            data_payload = raw_data
            compressed_size = uncompressed_size
            is_compressed_flag = 0
    else:
        data_payload = raw_data
        compressed_size = uncompressed_size
        is_compressed_flag = 0

    shape_bytes = b"".join(struct.pack("!H", dim) for dim in target_tensor.shape)

    # Combine metadata with quantization info
    meta_dict = metadata.copy() if metadata else {}
    if q_tail is not None:
        meta_dict["_q_mode"] = Q_MODE_INT8_ROW
        meta_dict["_q_k"] = q_k
        meta_dict["_orig_dtype"] = str(tensor.dtype)

    if meta_dict:
        import json
        meta_json = json.dumps(meta_dict).encode("utf-8")
        if len(meta_json) > 0xFFFF:
            # Never prune: dropping request_id/shard silently misroutes.
            raise ValueError(f"Tensor metadata too large: {len(meta_json)} bytes")
        meta_bytes = struct.pack("!H", len(meta_json)) + meta_json # Use H (2 bytes) for meta len
    else:
        meta_bytes = struct.pack("!H", 0)

    # Header now uses 2 bits for compression flag (0=none, 1=zlib, 2=lz4)
    header = struct.pack(
        "!4sHHII",
        MAGIC,
        dtype_code,
        ndim,
        uncompressed_size,
        (compressed_size << 2) | is_compressed_flag,
    )

    return header + shape_bytes + meta_bytes + data_payload


def decode_tensor(data: bytes) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Decode a binary tensor back into a numpy array.
    Automatically de-quantizes if q-info is present.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Data too short: {len(data)} bytes")

    magic, dtype_code, ndim, uncompressed_size, compr_field = struct.unpack(
        "!4sHHII", data[:HEADER_SIZE]
    )
    is_compressed = compr_field & 3
    compressed_size = compr_field >> 2

    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r}")

    dtype_obj = DTYPE_REVERSE.get(dtype_code)
    if dtype_obj is None:
        raise ValueError(f"Unknown dtype code: {dtype_code}")

    shape_offset = HEADER_SIZE
    shape = []
    for _ in range(ndim):
        dim = struct.unpack("!H", data[shape_offset:shape_offset + 2])[0]
        shape.append(dim)
        shape_offset += 2

    meta_len = struct.unpack("!H", data[shape_offset:shape_offset + 2])[0]
    meta_offset = shape_offset + 2
    metadata = None
    if meta_len > 0:
        import json
        meta_json = data[meta_offset:meta_offset + meta_len]
        try:
            metadata = json.loads(meta_json.decode("utf-8"))
        except Exception:
            metadata = None

    data_offset = meta_offset + meta_len
    raw_data = data[data_offset:]

    if is_compressed == 1:
        raw_data = zlib.decompress(raw_data)
        if len(raw_data) != uncompressed_size:
            # Maybe it's partial data or wrong size?
            pass
    elif is_compressed == 2:
        import lz4.frame
        raw_data = lz4.frame.decompress(raw_data)
        if len(raw_data) != uncompressed_size:
            pass

    if metadata and metadata.get("_q_mode") == Q_MODE_INT8_ROW:
        count = int(np.prod(shape)) if shape else 1
        q = np.frombuffer(raw_data[:count], dtype=np.int8)
        return _dequantize_rows_int8(q, raw_data[count:], int(metadata.get("_q_k", 0)),
                                     tuple(shape), metadata.get("_orig_dtype", "float32")), metadata

    tensor = np.frombuffer(raw_data, dtype=dtype_obj).reshape(shape)

    # Legacy per-tensor INT8 (peers from before _q_mode).
    if metadata and "_q_scale" in metadata:
        scale = metadata["_q_scale"]
        orig_dtype = metadata.get("_orig_dtype", "float16")
        tensor = (tensor.astype(np.float32) * scale).astype(np.dtype(orig_dtype))
        
    return tensor, metadata


# ──────────────────────────────────────────────────────────
# Async Forwarder
# ──────────────────────────────────────────────────────────


@dataclass
class TensorMessage:
    """A tensor message in-flight through the transport layer."""
    request_id: str
    orchestrator_id: str
    micro_batch_id: int
    source_stage: int
    target_stage: int
    tensor: np.ndarray
    metadata: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)
    quantize: bool = False


class AsyncTensorForwarder:
    """
    Non-blocking tensor forwarding for pipeline stages.

    Features:
    - Fire-and-forget send: doesn't block the compute pipeline
    - Receive queue: buffered inbound tensors, dequeued by processor
    - Backpressure: queue depth limit prevents memory exhaustion
    - Ordered delivery: same-path tensors arrive in order (TCP guarantee)
    """

    def __init__(
        self,
        node_id: str,
        max_queue_depth: int = 32,
        compress_threshold_bytes: int = 65536,  # 64KB
    ):
        self.node_id = node_id
        self.max_queue_depth = max_queue_depth
        self.compress_threshold_bytes = compress_threshold_bytes

        # Outbound: write function per (request_id, target_stage)
        self._writers: Dict[Tuple[str, int], callable] = {}

        # Inbound: queue per (request_id, source_stage)
        self._queues: Dict[Tuple[str, int], Deque[TensorMessage]] = {}

        # Backpressure semaphores
        self._send_semaphore = asyncio.Semaphore(max_queue_depth)

        # Ordered reassembly buffers
        self._reassembly: Dict[str, Dict[int, TensorMessage]] = {}
        self._next_batch_id: Dict[str, int] = {}

    def register_writer(
        self, request_id: str, stage_id: int,
        writer_fn: callable,
    ):
        """Register a writer for sending tensors to a specific stage."""
        self._writers[(request_id, stage_id)] = writer_fn
        if (request_id, stage_id) not in self._queues:
            self._queues[(request_id, stage_id)] = deque()

    def register_receiver(
        self, request_id: str, stage_id: int,
    ):
        """Ensure a receive queue exists for a stage."""
        key = (request_id, stage_id)
        if key not in self._queues:
            self._queues[key] = deque()

    async def send(
        self, message: TensorMessage,
    ) -> bool:
        """
        Non-blocking send. Returns True if queued for send, False if
        backpressure prevents sending (caller should retry or drop).
        """
        key = (message.request_id, message.target_stage)
        writer = self._writers.get(key)
        if not writer:
            return False

        # Acquire backpressure semaphore
        try:
            await asyncio.wait_for(
                self._send_semaphore.acquire(), timeout=0.5
            )
        except asyncio.TimeoutError:
            return False

        try:
            # Encode tensor with optional INT8 quantization
            compress = (
                message.tensor.nbytes > self.compress_threshold_bytes
            )
            encoded = encode_tensor(
                message.tensor,
                metadata={
                    "request_id": message.request_id,
                    "orchestrator_id": message.orchestrator_id,
                    "micro_batch_id": message.micro_batch_id,
                    "source_stage": message.source_stage,
                    "target_stage": message.target_stage,
                    **(message.metadata or {}),
                },
                compress=compress,
                quantize_int8=message.quantize,
            )

            # Fire-and-forget write (don't await - let it run in background)
            asyncio.create_task(self._do_send(writer, encoded))
            return True
        except Exception:
            self._send_semaphore.release()
            return False

    async def _do_send(self, writer_fn, data: bytes):
        """Background task for actual network send."""
        try:
            await writer_fn(data)
        finally:
            self._send_semaphore.release()

    @staticmethod
    def decode_message(data: bytes, request_id: str = "") -> Optional[TensorMessage]:
        """Decode one packet into a TensorMessage, without any reordering."""
        try:
            tensor, metadata = decode_tensor(data)
        except (ValueError, struct.error):
            return None
        if not metadata:
            return None
        return TensorMessage(
            request_id=metadata.get("request_id", request_id),
            orchestrator_id=metadata.get("orchestrator_id", ""),
            micro_batch_id=metadata.get("micro_batch_id", 0),
            source_stage=metadata.get("source_stage", 0),
            target_stage=metadata.get("target_stage", 0),
            tensor=tensor,
            metadata=metadata,
        )

    def receive(
        self, request_id: str, source_stage: int, data: bytes,
    ) -> Optional[TensorMessage]:
        """
        Process an incoming tensor packet. Returns a TensorMessage if
        this is the next expected micro-batch in sequence, or None if
        it was buffered for later reassembly.
        """
        try:
            tensor, metadata = decode_tensor(data)
        except (ValueError, struct.error):
            return None

        if not metadata:
            return None

        msg = TensorMessage(
            request_id=metadata.get("request_id", request_id),
            orchestrator_id=metadata.get("orchestrator_id", ""),
            micro_batch_id=metadata.get("micro_batch_id", 0),
            source_stage=metadata.get("source_stage", 0),
            target_stage=metadata.get("target_stage", 0),
            tensor=tensor,
            metadata=metadata,
        )

        # Buffer for ordered reassembly
        if request_id not in self._reassembly:
            self._reassembly[request_id] = {}
            self._next_batch_id[request_id] = 0

        self._reassembly[request_id][msg.micro_batch_id] = msg

        # Return the next in-order batch if available
        next_id = self._next_batch_id[request_id]
        if next_id in self._reassembly[request_id]:
            result = self._reassembly[request_id].pop(next_id)
            self._next_batch_id[request_id] = next_id + 1
            return result

        return None

    def flush_request(self, request_id: str):
        """Clean up all state for a completed request."""
        self._reassembly.pop(request_id, None)
        self._next_batch_id.pop(request_id, None)
        # Clean up writer/queue registrations
        keys_to_remove = [
            k for k in self._writers if k[0] == request_id
        ]
        for k in keys_to_remove:
            self._writers.pop(k, None)
            self._queues.pop(k, None)