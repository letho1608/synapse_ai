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


@pytest.mark.asyncio
async def test_pins_authenticate_the_peer_without_a_ca(tmp_path, monkeypatch):
    """No CA: the client context cannot verify the chain, so the pin must.
    Before, SYNAPSE_TLS without a CA encrypted to WHOEVER answered (the
    advertised tls_fp was never checked). A wrong pin refuses the link on
    both the event bridge and the chunked TCP tensor path."""
    from synapse.transport.tls_wrap import cert_fingerprint_pem
    from synapse.transport.tcp_transport import TCPClientPool, TCPServer
    try:
        cert_pem, key_pem = gen_self_signed_cert()
    except ImportError:
        pytest.skip("cryptography package not installed")
    cert_f, key_f = tmp_path / "c.pem", tmp_path / "k.pem"
    cert_f.write_bytes(cert_pem)
    key_f.write_bytes(key_pem)
    pin = cert_fingerprint_pem(cert_pem)
    srv_ctx = build_server_context(str(cert_f), str(key_f))
    cli_ctx = build_client_context(cafile=None)  # no CA: pins only

    _, a, ta = await _mk("pin-a", 59513, srv_ctx, cli_ctx, pin)
    _, b, tb = await _mk("pin-b", 59514, srv_ctx, cli_ctx, pin)
    try:
        monkeypatch.setenv("SYNAPSE_TLS_PINS", "0" * 64)
        assert not await a.connect_to_peer("pin-b", "127.0.0.1:59514")
        monkeypatch.setenv("SYNAPSE_TLS_PINS", pin)
        assert await a.connect_to_peer("pin-b", "127.0.0.1:59514")
    finally:
        await _close(a, ta)
        await _close(b, tb)

    srv = TCPServer(host="127.0.0.1", port=59515, ssl_ctx=srv_ctx)
    got = []
    srv.set_frame_handler(got.append)
    await srv.start()
    try:
        monkeypatch.setenv("SYNAPSE_TLS_PINS", "f" * 64)
        assert await TCPClientPool(ssl_ctx=cli_ctx).send("127.0.0.1:59515", b"x") is False
        monkeypatch.setenv("SYNAPSE_TLS_PINS", pin)
        assert await TCPClientPool(ssl_ctx=cli_ctx).send("127.0.0.1:59515", b"hello") is True
    finally:
        await srv.stop()


def test_require_mode_refuses_unauthenticated_tls(tmp_path, monkeypatch):
    from synapse.transport.tls_wrap import configure_from_env
    cert_pem, key_pem = gen_self_signed_cert()
    (tmp_path / "c.pem").write_bytes(cert_pem)
    (tmp_path / "k.pem").write_bytes(key_pem)
    monkeypatch.setenv("SYNAPSE_TLS", "require")
    monkeypatch.setenv("SYNAPSE_TLS_CERT", str(tmp_path / "c.pem"))
    monkeypatch.setenv("SYNAPSE_TLS_KEY", str(tmp_path / "k.pem"))
    monkeypatch.delenv("SYNAPSE_TLS_CA", raising=False)
    monkeypatch.delenv("SYNAPSE_TLS_PINS", raising=False)
    with pytest.raises(ValueError, match="CA or SYNAPSE_TLS_PINS|SYNAPSE_TLS_CA"):
        configure_from_env("n1")
    monkeypatch.setenv("SYNAPSE_TLS_PINS", "a" * 64)
    srv, cli, pin = configure_from_env("n1")
    assert srv is not None and cli is not None and len(pin) == 64


def test_selfsigned_mode_generates_and_reuses_a_node_cert(tmp_path, monkeypatch):
    from synapse.transport import tls_wrap
    monkeypatch.setenv("SYNAPSE_TLS", "selfsigned")
    for k in ("SYNAPSE_TLS_CERT", "SYNAPSE_TLS_KEY", "SYNAPSE_TLS_CA", "SYNAPSE_TLS_PINS"):
        monkeypatch.delenv(k, raising=False)
    orig = tls_wrap.ensure_self_signed
    monkeypatch.setattr(tls_wrap, "ensure_self_signed",
                        lambda nid, directory=None: orig(nid, str(tmp_path)))
    _, _, pin1 = tls_wrap.configure_from_env("node/a")
    _, _, pin2 = tls_wrap.configure_from_env("node/a")
    assert pin1 == pin2 and len(pin1) == 64  # generated once, then reused
    assert sorted(p.name for p in tmp_path.iterdir()) == ["node_a.cert.pem", "node_a.key.pem"]


@pytest.mark.asyncio
async def test_ca_mode_is_mutual_tls(tmp_path):
    """With a CA both ends authenticate: a client without a CA-signed
    certificate is refused by the server (bridge)."""
    from synapse.transport.tls_wrap import enable_mutual, cert_fingerprint_pem
    cert_pem, key_pem = gen_self_signed_cert()
    cert_f, key_f = tmp_path / "c.pem", tmp_path / "k.pem"
    cert_f.write_bytes(cert_pem)
    key_f.write_bytes(key_pem)
    pin = cert_fingerprint_pem(cert_pem)

    def ctxs(with_client_cert=True):
        srv = build_server_context(str(cert_f), str(key_f))
        cli = build_client_context(cafile=str(cert_f))
        if with_client_cert:
            enable_mutual(srv, cli, str(cert_f), str(key_f), str(cert_f))
        else:
            enable_mutual(srv, build_client_context(cafile=str(cert_f)), str(cert_f), str(key_f), str(cert_f))
        return srv, cli

    srv_a, cli_a = ctxs()
    srv_b, _ = ctxs()
    _, a, ta = await _mk("mtls-a", 59516, srv_a, cli_a, pin)
    _, b, tb = await _mk("mtls-b", 59517, srv_b, cli_a, pin)
    try:
        assert await a.connect_to_peer("mtls-b", "127.0.0.1:59517")
    finally:
        await _close(a, ta)
        await _close(b, tb)

    _, cli_nocert = ctxs(with_client_cert=False)
    srv_c, _ = ctxs()
    _, c, tc = await _mk("mtls-c", 59518, srv_c, cli_nocert, pin)
    _, d, td = await _mk("mtls-d", 59519, srv_c, cli_nocert, pin)
    try:
        assert not await c.connect_to_peer("mtls-d", "127.0.0.1:59519")
    finally:
        await _close(c, tc)
        await _close(d, td)
