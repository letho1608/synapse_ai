"""Download progress reaches the node's own map and /v1/download/progress.

HFShardDownloader.on_progress used to have observers (main.py's broadcast)
but nothing ever triggered it, so the Model page's progress bar was always
empty. Pulls now report files done / total while they run.
"""
import asyncio
import json
import sys
import types

from aiohttp.test_utils import make_mocked_request

from synapse.api.chatgpt_api import ChatGPTAPI
from synapse.inference.partial_loader import PartialModel
from synapse.inference.pytorch_hf_engine import HFShardDownloader
from synapse.orchestration.node import Node
from synapse.routing import EventRouter
from synapse.routing.libp2p_node import Libp2pNode

from test_b3b4b5_wiring import _Engine


def _node_and_api():
    router = EventRouter()
    node = Node(_id="node-a", event_router=router, libp2p_node=Libp2pNode("node-a", 0, router),
                inference_engine=_Engine(), shard_downloader=HFShardDownloader())
    api = ChatGPTAPI.__new__(ChatGPTAPI)
    api.node = node
    api.inference_engine_classname = "PyTorchHFInferenceEngine"
    return node, api


async def _progress_json(api):
    r = await api.handle_get_download_progress(make_mocked_request("GET", "/v1/download/progress"))
    return json.loads(r.text)["progress"]


def test_progress_from_a_worker_thread_lands_in_the_node_map():
    node, api = _node_and_api()

    async def run():
        progress = api._download_reporter("org/model", "org/model")
        # pulls run in asyncio.to_thread: the reporter must hop back to the loop
        await asyncio.to_thread(progress, "in_progress", 2, 5)
        await asyncio.sleep(0)
        mid = await _progress_json(api)
        await asyncio.to_thread(progress, "complete", 5, 5)
        await asyncio.sleep(0)
        return mid, await _progress_json(api)

    mid, done = asyncio.run(run())
    assert mid["node-a"]["repo_id"] == "org/model"
    assert (mid["node-a"]["status"], mid["node-a"]["downloaded_files"], mid["node-a"]["total_files"]) == ("in_progress", 2, 5)
    assert done["node-a"]["status"] == "complete"


def test_full_repo_pull_reports_snapshot_file_count(monkeypatch):
    seen = []

    def fake_snapshot_download(repo_id, tqdm_class=None, **kw):
        bar = tqdm_class(total=3, disable=True)
        for _ in range(3):
            bar.update(1)
        bar.close()

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    ChatGPTAPI._pull_full_repo("org/model", lambda s, d, t: seen.append((s, d, t)))
    assert seen[:3] == [("in_progress", 1, 3), ("in_progress", 2, 3), ("in_progress", 3, 3)]
    assert seen[-1][0] == "complete"


def test_prefetch_reports_each_file():
    pm = PartialModel.__new__(PartialModel)
    pm.remote = {"b.safetensors"}  # read over HTTP ranges at load: counted, not fetched
    fetched, ticks = [], []
    pm.files_for = lambda s, e, f, l: ["a.safetensors", "b.safetensors", "c.safetensors"]
    pm._file = fetched.append
    files = pm.prefetch(0, 3, True, True, on_file=lambda d, t: ticks.append((d, t)))
    assert files == ["a.safetensors", "b.safetensors", "c.safetensors"]
    assert fetched == ["a.safetensors", "c.safetensors"]
    assert ticks == [(1, 3), (2, 3), (3, 3)]
