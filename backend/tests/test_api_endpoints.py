import pytest
from synapse.api.chatgpt_api import ChatGPTAPI
from synapse.topology.partitioning_strategy import Partition
from typing import Optional, List
from unittest.mock import MagicMock
from aiohttp.test_utils import make_mocked_request, TestClient, TestServer
import json
import tempfile
from pathlib import Path

class MockNode:
    def __init__(self):
        self.id = "mock_node_1"
        self.topology = MagicMock()
        self.topology.nodes = {}
        self.peers = []
        self.authoritative_partitions: Optional[List[Partition]] = None
        self._current_metrics = MagicMock()
        self._current_metrics.cpu_util_pct = 10.0
        self._current_metrics.ram_util_pct = 20.0
        self._current_metrics.ram_used_mb = 1024
        self._current_metrics.ram_total_mb = 4096
        self._current_metrics.gpu_vram_used_mb = 512
        self._current_metrics.gpu_power_w = 50.0
        self._current_metrics.gpu_temp_c = 45.0
        self.on_token = MagicMock()

    def _compute_lacp_partition(self):
        return []

    def _get_authoritative_partitions(self, base_shard=None):
        return self.authoritative_partitions

@pytest.mark.asyncio
async def test_get_system_info():
    node = MockNode()
    api = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")
    req = make_mocked_request('GET', '/v1/system/info')
    resp = await api.handle_get_system_info(req)
    assert resp.status == 200
    data = json.loads(resp.text)
    assert 'hardware' in data
    assert 'cpu' in data['hardware']
    assert 'ram_total_gb' in data['hardware']

@pytest.mark.asyncio
async def test_get_system_stats():
    node = MockNode()
    api = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")
    req = make_mocked_request('GET', '/v1/system/stats')
    resp = await api.handle_get_system_stats(req)
    assert resp.status == 200
    data = json.loads(resp.text)
    assert 'cpu' in data
    assert 'memory' in data
    assert 'uptime' in data


@pytest.mark.asyncio
async def test_root_route_serves_real_page():
    """GET / is registered unconditionally and serves a real HTML page."""
    node = MockNode()
    api = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")
    client = TestClient(TestServer(api.app))
    await client.start_server()
    try:
        resp = await client.get('/')
        assert resp.status == 200
        assert 'text/html' in resp.headers.get('Content-Type', '')
        body = await resp.text()
        assert '<html' in body.lower()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_root_serves_builtin_page_when_ui_not_built(tmp_path, monkeypatch):
    """Without frontend/dist, GET / is the built-in page (with build hint);
    constructing the API must not create any directory for it."""
    from synapse.api import chatgpt_api as C
    monkeypatch.setattr(C, "FRONTEND_DIST", tmp_path / "frontend" / "dist")
    monkeypatch.delenv("SYNAPSE_HEADLESS", raising=False)
    node = MockNode()
    api = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")
    assert not (tmp_path / "frontend").exists()
    req = make_mocked_request('GET', '/')
    resp = await api.handle_root(req)
    assert resp.status == 200
    assert '<html' in resp.text.lower()
    assert 'Synapse' in resp.text and 'npm --prefix frontend' in resp.text


@pytest.mark.asyncio
async def test_distributed_status_uses_authoritative_partition():
    node = MockNode()
    node.authoritative_partitions = [
        Partition("mock_node_1", 0.0, 0.5),
        Partition("peer", 0.5, 1.0),
    ]
    api = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")

    resp = await api.handle_get_distributed_status(make_mocked_request('GET', '/v1/distributed/status'))
    data = json.loads(resp.text or "{}")

    assert resp.status == 200
    assert data["partitions_count"] == 2
    assert data["multi_node_inference"] is True


@pytest.mark.asyncio
async def test_distributed_status_falls_back_to_peer_count_before_topology_converges():
    node = MockNode()
    node.peers = [MagicMock()]
    api = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")

    resp = await api.handle_get_distributed_status(make_mocked_request('GET', '/v1/distributed/status'))
    data = json.loads(resp.text or "{}")

    assert resp.status == 200
    assert data["partitions_count"] == 2
    assert data["multi_node_inference"] is True


@pytest.mark.asyncio
async def test_models_search_route_works(monkeypatch):
    """/v1/models/search used to 500 on every call: synapse.model_discovery
    could not even be imported (List/Dict/Any used without typing import),
    which also broke pulling any HF repo id outside the built-in list."""
    import io
    import urllib.request

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return io.BytesIO(json.dumps([{"modelId": "org/tiny", "downloads": 3}]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    api = ChatGPTAPI(node=MockNode(), inference_engine_classname="PyTorchEngine")
    client = TestClient(TestServer(api.app))
    await client.start_server()
    try:
        resp = await client.get('/v1/models/search?q=tiny llm')
        assert resp.status == 200, await resp.text()
        data = (await resp.json())["data"]
        assert data and data[0]["name"] == "org/tiny"
        assert "search=tiny%20llm" in seen["url"] and seen["timeout"]
    finally:
        await client.close()
