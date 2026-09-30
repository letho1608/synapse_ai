"""
POST /v1/settings optimization knobs are validated, APPLIED and persisted.

Found 2026-09-30: load_quantization, lookahead_window and
transport_quantization were stored on the API object and never reached the
node, the engine or the transport router; lacp_priority / lacp_max_ping
were written to node attributes that did not exist. Now each knob drives
the real setting (node.lookahead_window, SYNAPSE_WEIGHT_QUANT,
SYNAPSE_TRANSPORT_INT8, node.partition_objective) and round-trips through
settings.json.
"""
import json

import pytest
from aiohttp.test_utils import make_mocked_request

from synapse.api.chatgpt_api import ChatGPTAPI
from test_api_endpoints import MockNode


def _post(body):
    req = make_mocked_request("POST", "/v1/settings")

    async def _json():
        return body
    req.json = _json
    return req


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "auto")
    monkeypatch.setenv("SYNAPSE_TRANSPORT_INT8", "1")
    node = MockNode()
    node.lookahead_window = 3
    node.partition_objective = None
    a = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")
    a._settings_path = tmp_path / "settings.json"
    return a


@pytest.mark.asyncio
async def test_settings_are_applied_and_persisted(api, monkeypatch):
    import os
    resp = await api.handle_post_settings(_post({
        "lookahead_window": 5, "load_quantization": "int8",
        "transport_quantization": "none", "lacp_priority": "latency"}))
    assert resp.status == 200, resp.text
    assert api.node.lookahead_window == 5
    assert os.environ["SYNAPSE_WEIGHT_QUANT"] == "int8"
    assert os.environ["SYNAPSE_TRANSPORT_INT8"] == "0"
    assert api.node.partition_objective == "latency"
    from synapse.transport.router import should_quantize
    assert not should_quantize("float32", target_id="peer", self_id="me")

    got = json.loads((await api.handle_get_settings(make_mocked_request("GET", "/v1/settings"))).text)
    assert got["lookahead_window"] == 5 and got["load_quantization"] == "int8"
    assert got["transport_quantization"] == "none" and got["lacp_priority"] == "latency"

    saved = json.loads(api._settings_path.read_text(encoding="utf-8"))
    assert saved["lookahead_window"] == 5 and saved["transport_quantization"] == "none"

    # A fresh API (restart) re-applies the saved knobs.
    monkeypatch.setenv("SYNAPSE_TRANSPORT_INT8", "1")
    node2 = MockNode()
    node2.lookahead_window = 3
    node2.partition_objective = None
    api2 = ChatGPTAPI.__new__(ChatGPTAPI)
    api2.node = node2
    api2._settings_path = api._settings_path
    api2.default_model = ""
    api2._load_settings()
    assert node2.lookahead_window == 5 and os.environ["SYNAPSE_TRANSPORT_INT8"] == "0"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"lookahead_window": 99}, {"lookahead_window": "x"}, {"load_quantization": "fp8"},
    {"transport_quantization": "bf16"}, {"lacp_priority": "fast"}])
async def test_invalid_settings_are_rejected_without_side_effects(api, bad):
    import os
    resp = await api.handle_post_settings(_post(bad))
    assert resp.status == 400
    assert api.node.lookahead_window == 3
    assert os.environ["SYNAPSE_WEIGHT_QUANT"] == "auto"
    assert os.environ["SYNAPSE_TRANSPORT_INT8"] == "1"


@pytest.mark.asyncio
async def test_models_ready_means_cached_locally(api, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, fn: "/x/config.json" if "SmolLM2-135M" in repo else None)
    data = json.loads((await api.handle_get_models(make_mocked_request("GET", "/v1/models"))).text)["data"]
    ready = {m["id"] for m in data if m["ready"]}
    assert ready and all("SmolLM2-135M" in m for m in ready)
    assert any(not m["ready"] for m in data)


@pytest.mark.asyncio
async def test_chat_model_default_alias_resolves_to_server_model(api):
    """`synapse benchmark` without --model sends model="default"; it used to
    get HTTP 400 (model_not_found)."""
    api.default_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    api.inference_engine_classname = "PyTorchHFInferenceEngine"
    seen = {}

    async def fake_process_prompt(base_shard, prompt, request_id=None, **kw):
        seen["model"] = base_shard.model_id
        raise RuntimeError("stop here")
    api.node.process_prompt = fake_process_prompt
    req = make_mocked_request("POST", "/v1/chat/completions")

    async def _json():
        return {"model": "default", "stream": False, "max_tokens": 2,
                "messages": [{"role": "user", "content": "hi"}]}
    req.json = _json
    try:
        resp = await api.handle_post_chat_completions(req)
        assert resp.status != 400 or "model_not_found" not in resp.text
    except Exception:
        pass
    assert seen.get("model") == "HuggingFaceTB/SmolLM2-135M-Instruct"
