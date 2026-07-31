import pytest
from synapse.api.chatgpt_api import ChatGPTAPI
from unittest.mock import MagicMock
from aiohttp.test_utils import make_mocked_request, TestClient, TestServer
import json
import tempfile
from pathlib import Path

class MockNode:
    def __init__(self):
        self.id = "mock_node_1"
        self._current_metrics = MagicMock()
        self._current_metrics.cpu_util_pct = 10.0
        self._current_metrics.ram_util_pct = 20.0
        self._current_metrics.ram_used_mb = 1024
        self._current_metrics.ram_total_mb = 4096
        self._current_metrics.gpu_vram_used_mb = 512
        self._current_metrics.gpu_power_w = 50.0
        self._current_metrics.gpu_temp_c = 45.0
        self.on_token = MagicMock()

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
async def test_root_falls_back_to_bundled_page_when_frontend_missing():
    """Even with no on-disk frontend, GET / returns the bundled page."""
    node = MockNode()
    api = ChatGPTAPI(node=node, inference_engine_classname="PyTorchEngine")
    api.static_dir = Path(tempfile.mkdtemp())  # empty dir -> no index.html
    req = make_mocked_request('GET', '/')
    resp = await api.handle_root(req)
    assert resp.status == 200
    assert '<html' in resp.text.lower()
    assert 'Synapse' in resp.text
