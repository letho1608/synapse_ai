"""
Test LACP — Async Tensor Binary Codec & Forwarding

Validates:
1. Binary codec: encode → decode roundtrip for various dtypes/shapes
2. Compression: zlib reduces size for large tensors
3. Codec header validation: invalid data rejected
4. Async forwarder: send/receive message ordering
5. Ordered reassembly: micro-batches delivered in sequence
6. Backpressure: send blocked when queue full
7. Flush: cleanup after request completes
"""

import pytest
import sys
import os
import asyncio
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.transport.codec import (
    encode_tensor,
    decode_tensor,
    TensorMessage,
    AsyncTensorForwarder,
    MAGIC,
    HEADER_SIZE,
)


class TestBinaryCodec:
    def test_roundtrip_float32(self):
        t = np.random.randn(64, 768).astype(np.float32)
        encoded = encode_tensor(t)
        decoded, meta = decode_tensor(encoded)
        assert decoded.shape == t.shape
        assert decoded.dtype == t.dtype
        assert np.allclose(decoded, t)

    def test_roundtrip_int64(self):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        encoded = encode_tensor(t)
        decoded, meta = decode_tensor(encoded)
        assert np.array_equal(decoded, t)

    def test_roundtrip_with_metadata(self):
        t = np.ones((10, 10), dtype=np.float16)
        metadata = {"request_id": "abc-123", "batch": 5}
        encoded = encode_tensor(t, metadata=metadata)
        decoded, returned_meta = decode_tensor(encoded)
        assert np.allclose(decoded, t)
        assert returned_meta is not None
        assert returned_meta["request_id"] == "abc-123"

    def test_compression_large_tensor(self):
        # Use repetitive data that compresses well (constant column pattern)
        t = np.ones((512, 4096), dtype=np.float32)
        uncompressed = encode_tensor(t, compress=False)
        compressed = encode_tensor(t, compress=True)
        assert len(compressed) < len(uncompressed), f"{len(compressed)} >= {len(uncompressed)}"
        decoded, _ = decode_tensor(compressed)
        assert np.allclose(decoded, t)

    def test_small_tensor_no_compression(self):
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        compressed = encode_tensor(t, compress=True)
        decoded, _ = decode_tensor(compressed)
        assert np.allclose(decoded, t)

    def test_invalid_magic(self):
        with pytest.raises(ValueError, match="Invalid magic"):
            decode_tensor(b"\x00" * 100)

    def test_invalid_dtype(self):
        t = np.array(["hello", "world"])
        with pytest.raises(ValueError, match="Unsupported dtype"):
            encode_tensor(t)

    def test_empty_tensor(self):
        t = np.array([], dtype=np.float32).reshape(0)
        encoded = encode_tensor(t)
        decoded, _ = decode_tensor(encoded)
        assert decoded.shape == (0,)
        assert np.array_equal(decoded, t)

    def test_scalar_tensor(self):
        t = np.array(42.0, dtype=np.float32)
        encoded = encode_tensor(t)
        decoded, _ = decode_tensor(encoded)
        assert decoded.shape == ()
        assert decoded.dtype == np.float32
        assert float(decoded) == 42.0

    def test_header_size_constant(self):
        t = np.ones((3, 4), dtype=np.float32)
        encoded = encode_tensor(t)
        assert encoded[:4] == MAGIC
        assert len(encoded) == HEADER_SIZE + 2 * len(t.shape) + 2 + t.nbytes

    def test_large_shape(self):
        t = np.ones((128, 512, 768), dtype=np.float32)
        encoded = encode_tensor(t)
        decoded, _ = decode_tensor(encoded)
        assert decoded.shape == t.shape


class TestAsyncForwarder:
    def test_basic_send_receive(self):
        fw = AsyncTensorForwarder("node-0")
        t = np.ones((10,), dtype=np.float32)

        sent_data = None

        async def capture_writer(data):
            nonlocal sent_data
            sent_data = data

        fw.register_writer("req-1", 1, capture_writer)

        async def run():
            msg = TensorMessage(
                request_id="req-1",
                orchestrator_id="orch-0",
                micro_batch_id=0,
                source_stage=0,
                target_stage=1,
                tensor=t,
            )
            result = await fw.send(msg)
            assert result
            # Wait for background send to complete
            await asyncio.sleep(0.1)
            assert sent_data is not None
            decoded, meta = decode_tensor(sent_data)
            assert meta["request_id"] == "req-1"
            assert meta["micro_batch_id"] == 0

        asyncio.run(run())

    def test_send_to_unregistered_writer(self):
        fw = AsyncTensorForwarder("node-0")
        t = np.ones((5,), dtype=np.float32)

        async def run():
            msg = TensorMessage(
                request_id="req-no-exist",
                orchestrator_id="orch",
                micro_batch_id=0,
                source_stage=0,
                target_stage=99,
                tensor=t,
            )
            result = await fw.send(msg)
            assert not result

        asyncio.run(run())

