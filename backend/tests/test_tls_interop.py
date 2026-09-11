"""
Behavioral test: TLS interop on loopback (Item 5 / C4, H5).

Generates a throwaway self-signed cert (cryptography lib), brings up two
bridges with real server/client SSL contexts, and asserts: handshake with
versions both directions, event delivery, and BOTH writers actually wrapped
in TLS (ssl_object present) plus the advertised tls_fp pin recorded.

Ports 59511/59512: unused by any other test module in this repo.
"""
import asyncio
import ssl

import pytest

from synapse.routing.event_router import EventRouter
from synapse.routing.p2p_socket_bridge import P2PSocketBridge
from synapse.transport.tls_wrap import (
    build_client_context,
    build_server_context,
    fingerprint_bytes,
    gen_self_signed_cert,
)


async def _mk(nid, port, srv_ctx, cli_ctx, tls_fp):
    er = EventRouter()
    try:
        await er.start()
    except Exception:
        pass
    br = P2PSocketBridge(nid, port, er, ssl_server_ctx=srv_ctx,
                         ssl_client_ctx=cli_ctx, tls_fp=tls_fp)
    task = asyncio.create_task(br.start())
    await asyncio.sleep(0.3)
    return er, br, task


async def _close(br, task):
    try:
        await br.stop()
    except Exception:
        pass
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


@pytest.mark.asyncio
async def test_tls_pair_handshake_event_and_pin(tmp_path):
    try:
        cert_pem, key_pem = gen_self_signed_cert()
    except ImportError:
        pytest.skip("cryptography package not installed")
    cert_f = tmp_path / "cert.pem"
    key_f = tmp_path / "key.pem"
    cert_f.write_bytes(cert_pem)
    key_f.write_bytes(key_pem)
    fp = fingerprint_bytes(cert_pem)
    assert len(fp) == 64

    srv_ctx = build_server_context(str(cert_f), str(key_f))
    cli_ctx = build_client_context(cafile=str(cert_f))

    _, a, ta = await _mk("tls-a", 59511, srv_ctx, cli_ctx, fp)
    _, b, tb = await _mk("tls-b", 59512, srv_ctx, cli_ctx, fp)
    try:
        assert await a.connect_to_peer("tls-b", "127.0.0.1:59512")
        await asyncio.sleep(0.5)

        ha = a.get_peer_handshake("tls-b")
        hb = b.get_peer_handshake("tls-a")
        assert ha and hb, "no handshake over TLS"
        assert ha["tls_fp"] == fp and hb["tls_fp"] == fp

        _, wa = a._connections["tls-b"]
        _, wb = b._connections["tls-a"]
        assert wa.get_extra_info("ssl_object") is not None, "A->B not TLS"
        assert wb.get_extra_info("ssl_object") is not None, "B->A not TLS"
        assert isinstance(wa.get_extra_info("ssl_object"), ssl.SSLObject)

        received = []
        b.event_router.subscribe("synapse/test/tls", received.append)
        await a.event_router.publish("synapse/test/tls", {"secret": 7},
                                     origin="local")
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.1)
        assert received and received[0].data == {"secret": 7}
    finally:
        await _close(a, ta)
        await _close(b, tb)
