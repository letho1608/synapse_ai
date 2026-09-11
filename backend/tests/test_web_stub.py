"""
Behavioral tests: JSONL logging contract + GET / headless stub (T2.1 / M4).

- emit() writes exactly one JSON object per call (stdout discipline).
- redact() removes dashed/underscored/camel secrets.
- parse_stream() rejects non-object lines (scripts can trust stdout).
- handle_root returns 204 under SYNAPSE_HEADLESS=1, real page otherwise.
"""
import io
import json
from contextlib import redirect_stdout

import pytest

from synapse.logging_json import emit, parse_stream, redact


def test_emit_single_json_object_per_call():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rec = emit("INFO", "serve", "hello", {"a": 1}, node_id="n1")
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0]) == rec
    assert rec["level"] == "INFO" and rec["cmd"] == "serve"
    assert rec["fields"] == {"a": 1} and rec["node_id"] == "n1"
    assert isinstance(rec["ts"], float)


def test_emit_normalizes_level_and_redacts_fields():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rec = emit("BOGUS", "c", "m", {"tailscale-api-key": "SECRET", "x": 1})
    assert rec["level"] == "INFO"
    assert rec["fields"] == {"tailscale-api-key": "***", "x": 1}
    assert "SECRET" not in buf.getvalue()


def test_parse_stream_strict():
    recs = parse_stream('{"a": 1}\n\n{"b": 2}\n')
    assert recs == [{"a": 1}, {"b": 2}]
    with pytest.raises(ValueError):
        parse_stream('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError):
        parse_stream('[1, 2]\n')


def test_redact_variants():
    assert redact({"apiKey": "s", "a": 1}) == {"apiKey": "***", "a": 1}
    assert redact({"auth_Token": "s"}) == {"auth_Token": "***"}
    assert redact([{"password": "s"}]) == [{"password": "***"}]
    assert redact("plain") == "plain"


@pytest.mark.asyncio
async def test_handle_root_headless_stub_and_real_page(monkeypatch, tmp_path):
    from aiohttp import web
    from synapse.api.chatgpt_api import ChatGPTAPI
    api = ChatGPTAPI.__new__(ChatGPTAPI)
    monkeypatch.setenv("SYNAPSE_HEADLESS", "1")
    resp = await api.handle_root(None)
    assert isinstance(resp, web.Response)
    assert resp.status == 204
    # Real page branch without a full server: point static_dir at a dir
    # containing index.html (mirrors the production fallback chain).
    monkeypatch.delenv("SYNAPSE_HEADLESS", raising=False)
    (tmp_path / "index.html").write_text("<html>hi</html>", encoding="utf-8")
    api.static_dir = tmp_path
    resp2 = await api.handle_root(None)
    assert resp2.status == 200
