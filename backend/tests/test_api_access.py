"""API access policy: admin routes are local-only without a key; with
SYNAPSE_API_KEY every non-liveness route needs the bearer token."""
import pytest

from synapse.api.chatgpt_api import check_access

LAN = "192.168.1.20"


def test_no_key_chat_open_to_lan():
    assert check_access("POST", "/v1/chat/completions", LAN, "", "", None) is None
    assert check_access("GET", "/v1/models", LAN, "", "", None) is None
    assert check_access("GET", "/v1/distributed/status", LAN, "", "", None) is None


def test_no_key_admin_routes_local_only():
    for method, path in [("POST", "/quit"), ("POST", "/v1/models/delete"),
                         ("POST", "/v1/settings"), ("POST", "/v1/datasets/delete"),
                         ("GET", "/v1/logs/terminal"), ("POST", "/v1/models/pull")]:
        status, _ = check_access(method, path, LAN, "", "", None)
        assert status == 403, path
        assert check_access(method, path, "127.0.0.1", "", "", None) is None
        assert check_access(method, path, "::1", "", "", None) is None


def test_unknown_remote_is_not_loopback():
    assert check_access("POST", "/quit", None, "", "", None)[0] == 403
    assert check_access("POST", "/quit", "not-an-ip", "", "", None)[0] == 403


def test_key_required_everywhere_but_liveness():
    key = "s3cret"
    assert check_access("GET", "/healthcheck", LAN, "", "", key) is None
    assert check_access("GET", "/", LAN, "", "", key) is None
    # even from localhost, and on the non-/v1 aliases (/quit was unprotected)
    assert check_access("POST", "/quit", "127.0.0.1", "", "", key)[0] == 401
    assert check_access("POST", "/chat/completions", LAN, "", "", key)[0] == 401
    assert check_access("POST", "/v1/chat/completions", LAN, "Bearer nope", "", key)[0] == 403
    assert check_access("POST", "/v1/chat/completions", LAN, "Bearer s3cret", "", key) is None
    # websocket can't set headers from a browser: ?token=
    assert check_access("GET", "/v1/ws/metrics", LAN, "", "s3cret", key) is None
    assert check_access("GET", "/v1/ws/metrics", LAN, "", "", key)[0] == 401


def test_preflight_always_allowed():
    assert check_access("OPTIONS", "/quit", LAN, "", "", "k") is None
    assert check_access("OPTIONS", "/quit", LAN, "", "", None) is None


def test_web_ui_page_and_assets_open_even_with_a_key():
    """The UI page and its static files load without a key (the UI asks for
    the key itself); its admin calls still need one."""
    for path in ("/", "/assets/index-abc.js", "/favicon.svg"):
        assert check_access("GET", path, LAN, "", "", "k") is None, path
        assert check_access("GET", path, LAN, "", "", None) is None, path
    assert check_access("POST", "/assets/x.js", LAN, "", "", "k")[0] == 401
    assert check_access("POST", "/v1/cluster/pairing", LAN, "", "", None)[0] == 403
    assert check_access("GET", "/v1/system/stats", LAN, "", "", None) is None  # read-only monitoring


@pytest.mark.asyncio
async def test_root_serves_built_ui_and_confines_assets(tmp_path, monkeypatch):
    from aiohttp.test_utils import make_mocked_request
    from aiohttp.web_exceptions import HTTPNotFound
    from synapse.api import chatgpt_api as C
    api = C.ChatGPTAPI.__new__(C.ChatGPTAPI)
    monkeypatch.delenv("SYNAPSE_HEADLESS", raising=False)
    monkeypatch.setattr(C, "FRONTEND_DIST", tmp_path / "dist")
    r = await api.handle_root(make_mocked_request("GET", "/"))
    assert "Web UI not built" in r.text  # fallback page says how to build
    (tmp_path / "dist" / "assets").mkdir(parents=True)
    (tmp_path / "dist" / "index.html").write_text("<div id=app></div>")
    (tmp_path / "dist" / "assets" / "a.js").write_text("x")
    (tmp_path / "secret.txt").write_text("no")
    r = await api.handle_root(make_mocked_request("GET", "/"))
    assert str(r._path).endswith("index.html")
    ok = make_mocked_request("GET", "/assets/a.js", match_info={"path": "a.js"})
    assert str((await api.handle_ui_asset(ok))._path).endswith("a.js")
    bad = make_mocked_request("GET", "/assets/../../secret.txt", match_info={"path": "../../secret.txt"})
    with pytest.raises(HTTPNotFound):
        await api.handle_ui_asset(bad)
