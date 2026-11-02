"""API access policy: admin routes are local-only without a key; with
SYNAPSE_API_KEY every non-liveness route needs the bearer token."""
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
