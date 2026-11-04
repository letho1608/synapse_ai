"""Per-request metrics for the web UI's live charts (/v1/metrics/requests)."""
import asyncio
import json
import time
from collections import deque

from aiohttp.test_utils import make_mocked_request

from synapse.api.chatgpt_api import ChatGPTAPI, check_access


def _api():
    api = ChatGPTAPI.__new__(ChatGPTAPI)  # the recorder and handler need only the buffer
    api._request_metrics = deque(maxlen=200)
    return api


def test_record_computes_ttft_and_decode_rate():
    api = _api()
    t0 = time.time() - 2.0
    api._record_request("abcdef1234", "m", True, t0, t0 + 0.5, 16, None)
    m = api._request_metrics[-1]
    assert m["id"] == "abcdef12" and m["tokens"] == 16 and m["stream"] is True
    assert abs(m["ttft_s"] - 0.5) < 0.01
    # 15 tokens after the first over ~1.5 s
    assert 9.0 < m["tok_s"] < 11.0
    # a failed request has no rate and carries its error
    api._record_request("x", "m", False, t0, None, 0, "boom")
    assert api._request_metrics[-1]["tok_s"] is None and api._request_metrics[-1]["error"] == "boom"


def test_handler_filters_by_since():
    api = _api()
    now = time.time()
    for i in range(3):
        api._request_metrics.append({"id": str(i), "at": now - 10 + i})

    async def get(q):
        r = await api.handle_get_request_metrics(make_mocked_request("GET", f"/v1/metrics/requests{q}"))
        return [m["id"] for m in json.loads(r.text)["data"]]

    assert asyncio.run(get("")) == ["0", "1", "2"]
    assert asyncio.run(get(f"?since={now - 9.5}")) == ["1", "2"]
    assert asyncio.run(get("?since=bad")) == ["0", "1", "2"]


def test_metrics_route_is_read_only_public():
    # the UI opened from another machine reads it without a key, like the other monitoring routes
    assert check_access("GET", "/v1/metrics/requests", "192.168.1.20", "", "", None) is None


def test_quit_stops_the_node_then_exits_zero(monkeypatch):
    """/quit used to await shutdown(..., self.node.server); Node has no
    `server`, so the call raised and the node never stopped."""
    import synapse.api.chatgpt_api as mod

    class FakeNode:
        stopped = False

        async def stop(self):
            FakeNode.stopped = True

    exits = []
    monkeypatch.setattr(mod.os, "_exit", lambda code: exits.append(code))
    api = ChatGPTAPI.__new__(ChatGPTAPI)
    api.node = FakeNode()

    async def run():
        r = await api.handle_quit(make_mocked_request("POST", "/quit"))
        assert r.status == 200
        await api._quit_task

    asyncio.run(run())
    assert FakeNode.stopped and exits == [0]
