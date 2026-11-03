"""
Pairing: one code per cluster instead of certs + pins copied per machine.

Two machines given the same code (each its own node key, cert issued
separately) connect over mutual TLS; a machine with a different code is
refused; a machine without any code stays plaintext unless told otherwise.
"""
import pytest

from test_tls_interop import _close, _free_port, _mk


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """Never touch the real ~/.synapse."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SYNAPSE_TLS", raising=False)
    monkeypatch.delenv("SYNAPSE_CLUSTER_CODE", raising=False)
    for k in ("SYNAPSE_TLS_CERT", "SYNAPSE_TLS_KEY", "SYNAPSE_TLS_CA", "SYNAPSE_TLS_PINS"):
        monkeypatch.delenv(k, raising=False)


def test_code_format_and_normalization():
    from synapse.transport.tls_wrap import new_cluster_code, normalize_code
    code = new_cluster_code()
    assert len(code) == 39 and code.count("-") == 7
    assert normalize_code(code.lower().replace("-", " ")) == code
    assert new_cluster_code() != code
    with pytest.raises(ValueError):
        normalize_code("ABCD-1234")
    with pytest.raises(ValueError):
        normalize_code("O" * 32)  # O/0/1/I are not in the alphabet


def test_mode_follows_the_saved_code(monkeypatch):
    from synapse.transport import tls_wrap
    assert tls_wrap.tls_enabled() == "off"
    assert tls_wrap.configure_from_env("n1") == (None, None, None)
    tls_wrap.save_cluster_code(tls_wrap.new_cluster_code())
    assert tls_wrap.tls_enabled() == "cluster"
    monkeypatch.setenv("SYNAPSE_TLS", "off")  # explicit off wins
    assert tls_wrap.tls_enabled() == "off"


def test_same_code_same_ca_on_every_machine(tmp_path):
    from synapse.transport.tls_wrap import ensure_cluster_identity, new_cluster_code
    from cryptography import x509
    code = new_cluster_code()
    c1, _, ca1 = ensure_cluster_identity("a", code, str(tmp_path / "m1"))
    c2, _, ca2 = ensure_cluster_identity("b", code, str(tmp_path / "m2"))
    load = lambda p: x509.load_pem_x509_certificate(open(p, "rb").read())
    assert load(ca1).public_key() == load(ca2).public_key()
    assert load(c1).public_key() != load(c2).public_key(), "node keys must be per machine"
    # reused while valid, reissued when the code changes
    first = open(c1, "rb").read()
    again, _, _ = ensure_cluster_identity("a", code, str(tmp_path / "m1"))
    assert open(again, "rb").read() == first
    other, _, _ = ensure_cluster_identity("a", new_cluster_code(), str(tmp_path / "m1"))
    assert open(other, "rb").read() != first


def _node_ctxs(monkeypatch, node_id, code):
    from synapse.transport.tls_wrap import configure_from_env
    monkeypatch.setenv("SYNAPSE_CLUSTER_CODE", code)
    srv, cli, fp = configure_from_env(node_id)
    monkeypatch.delenv("SYNAPSE_CLUSTER_CODE")
    return srv, cli, fp


@pytest.mark.asyncio
async def test_paired_nodes_connect_and_strangers_are_refused(monkeypatch):
    from synapse.transport.tls_wrap import new_cluster_code
    code, other = new_cluster_code(), new_cluster_code()
    srv_a, cli_a, fp_a = _node_ctxs(monkeypatch, "pair-a", code)
    srv_b, cli_b, fp_b = _node_ctxs(monkeypatch, "pair-b", code)
    srv_x, cli_x, fp_x = _node_ctxs(monkeypatch, "pair-x", other)

    _, a, ta = await _mk("pair-a", _free_port(), srv_a, cli_a, fp_a)
    pb = _free_port()
    _, b, tb = await _mk("pair-b", pb, srv_b, cli_b, fp_b)
    px = _free_port()
    _, x, tx = await _mk("pair-x", px, srv_x, cli_x, fp_x)
    try:
        assert await a.connect_to_peer("pair-b", f"127.0.0.1:{pb}")
        # wrong code, both directions
        assert not await x.connect_to_peer("pair-b", f"127.0.0.1:{pb}")
        assert not await a.connect_to_peer("pair-x", f"127.0.0.1:{px}")
    finally:
        for br, t in ((a, ta), (b, tb), (x, tx)):
            await _close(br, t)


@pytest.mark.asyncio
async def test_pairing_api_create_join_forget():
    """The web UI pairs machines through /v1/cluster/pairing (was `synapse pair`)."""
    from aiohttp.test_utils import make_mocked_request
    from synapse.api.chatgpt_api import ChatGPTAPI
    from synapse.transport import tls_wrap
    import json as _json

    api = ChatGPTAPI.__new__(ChatGPTAPI)  # handlers only need log_activity
    api.log_activity = lambda *a, **k: None

    async def post(body):
        req = make_mocked_request("POST", "/v1/cluster/pairing")
        async def _json_body():
            return body
        req.json = _json_body
        r = await api.handle_post_pairing(req)
        return r.status, _json.loads(r.body)

    async def get():
        r = await api.handle_get_pairing(make_mocked_request("GET", "/v1/cluster/pairing"))
        return _json.loads(r.body)

    assert (await get())["paired"] is False
    status, body = await post({"action": "create"})
    assert status == 200 and body["code"] == tls_wrap.cluster_code()
    st = await get()
    assert st["paired"] and st["tls_mode"] == "cluster" and st["restart_needed"]
    other = tls_wrap.new_cluster_code()
    status, _ = await post({"action": "join", "code": other.lower()})
    assert status == 200 and tls_wrap.cluster_code() == other
    status, body = await post({"action": "join", "code": "nope"})
    assert status == 400 and not body["success"]
    status, _ = await post({"action": "forget"})
    assert status == 200 and tls_wrap.cluster_code() is None
    status, _ = await post({"action": "bogus"})
    assert status == 400
