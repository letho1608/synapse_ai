"""
Behavioral test: SYNAPSE_TLS also encrypts the chunked TCP tensor path.

Before: TLS wrapped only the event bridge; the chunked-TCP fast path (default
on) still shipped tensors in plaintext even with SYNAPSE_TLS=require. Now
TensorTCPTransport takes the same server/client contexts: a TLS client
delivers bit-exact over TLS, and a plaintext client cannot inject frames
into a TLS server.

Port 59421/59422: unused by any other test module.
"""
import asyncio

import numpy as np
import pytest

from synapse.transport.codec import encode_tensor
from synapse.transport.tcp_transport import TensorTCPTransport
from synapse.transport.tls_wrap import (
    build_client_context,
    build_server_context,
    gen_self_signed_cert,
)


def _ctxs(tmp_path):
    try:
        cert_pem, key_pem = gen_self_signed_cert()
    except ImportError:
        pytest.skip("cryptography package not installed")
    cert_f, key_f = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_f.write_bytes(cert_pem)
    key_f.write_bytes(key_pem)
    return (build_server_context(str(cert_f), str(key_f)),
            build_client_context(cafile=str(cert_f)))


def _blob(arr, rid):
    return encode_tensor(arr, metadata={
        "request_id": rid, "orchestrator_id": "cli",
        "micro_batch_id": 0, "source_stage": 0, "target_stage": 1})


async def _wait(received, n=1):
    for _ in range(50):
        if len(received) >= n:
            return
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_chunked_tensor_over_tls(tmp_path):
    srv_ctx, cli_ctx = _ctxs(tmp_path)
    srv = TensorTCPTransport("srv", port=59421, ssl_server_ctx=srv_ctx)
    cli = TensorTCPTransport("cli", ssl_client_ctx=cli_ctx)
    received = []
    srv.set_tensor_handler(received.append)
    await srv.start()
    try:
        arr = np.arange(3 * 1024 * 1024, dtype=np.float32).reshape(3, 1024, 1024)
        assert await cli.send_blob_chunked("127.0.0.1:59421", _blob(arr, "r-tls"))
        await _wait(received)
        assert received and np.array_equal(received[0].tensor, arr)
        writer, _ = next(iter(cli.client_pool._conns.values()))
        assert writer.get_extra_info("ssl_object") is not None, "not actually TLS"
    finally:
        await cli.stop()
        await srv.stop()


@pytest.mark.asyncio
async def test_plaintext_client_cannot_inject_into_tls_server(tmp_path):
    srv_ctx, _ = _ctxs(tmp_path)
    srv = TensorTCPTransport("srv", port=59422, ssl_server_ctx=srv_ctx)
    plain = TensorTCPTransport("plain")
    received = []
    srv.set_tensor_handler(received.append)
    await srv.start()
    try:
        arr = np.ones((4, 4), dtype=np.float32)
        await plain.send_blob_chunked("127.0.0.1:59422", _blob(arr, "r-plain"))
        await asyncio.sleep(1.0)
        assert received == []
    finally:
        await plain.stop()
        await srv.stop()
