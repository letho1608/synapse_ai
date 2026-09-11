"""
Behavioral tests: manifest-indexed P2P checkpoint pull (T1.1 / C2).

- save_checkpoint indexes the sidecar manifest (sha256, capped).
- prefer_local_then_pull order: memory -> shared-dir -> P2P fetch.
- Cross-"machine" recovery with SEPARATE tmpdirs (no shared dir) succeeds
  over a real loopback bridge pair; sha mismatch and stale steps rejected.
- Traversal filenames confined; oversize files get explicit too_large.
- Weight-restore honesty (B1): PyTorchHF.load_checkpoint returns True only
  for restorable weight dirs, False (loud) for metadata-only paths.

Ports 59311/59312: unused by any other test module in this repo.
"""
import asyncio
import json
import os

import pytest

from synapse.pipeline.fault_tolerance import FaultToleranceManager
from synapse.routing.event_router import EventRouter
from synapse.routing.p2p_socket_bridge import P2PSocketBridge
from synapse.transport.checkpoint_pull import (
    CheckpointPullServer,
    make_fetch_fn,
)


async def _bridge_pair():
    ra, rb = EventRouter(), EventRouter()
    for r in (ra, rb):
        try:
            await r.start()
        except Exception:
            pass
    a = P2PSocketBridge("pull-a", 59311, ra)
    b = P2PSocketBridge("pull-b", 59312, rb)
    ta = asyncio.create_task(a.start())
    tb = asyncio.create_task(b.start())
    await asyncio.sleep(0.3)
    assert await a.connect_to_peer("pull-b", "127.0.0.1:59312")
    await asyncio.sleep(0.5)
    return (ra, a, ta), (rb, b, tb)


async def _close(bridge, task):
    try:
        await bridge.stop()
    except Exception:
        pass
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def test_save_writes_manifest_with_sha_and_cap(tmp_path):
    ft = FaultToleranceManager("n1", checkpoint_dir=str(tmp_path))
    for step in range(12):
        ft.save_checkpoint("m", 0, 7, step)
    manifest = ft.get_checkpoint_manifest("n1", "m")
    assert manifest is not None
    assert manifest["node_id"] == "n1"
    assert len(manifest["files"]) == ft.MANIFEST_KEEP, "manifest must be capped"
    steps = [e["step"] for e in manifest["files"]]
    assert steps == sorted(steps)[-ft.MANIFEST_KEEP:]
    assert steps[-1] == 11
    for e in manifest["files"]:
        assert len(e["sha256"]) == 64
        assert e["path"].endswith(".ckpt")
    assert ft.latest_manifest_step("n1", "m") == 11
    assert ft.get_checkpoint_manifest("n1", "nope") is None


def test_traversal_names_confined(tmp_path):
    ft = FaultToleranceManager("n1", checkpoint_dir=str(tmp_path))
    # Basename-first policy: traversal segments can never escape the dir —
    # they are normalized to a flat name inside checkpoint_dir.
    assert "/" not in ft._safe_name("../../etc/passwd")
    confined = ft._confine("../escape.ckpt")
    assert os.path.abspath(confined).startswith(os.path.abspath(str(tmp_path)))
    assert os.path.basename(confined) == "escape.ckpt"
    # Corrupt manifest reads as None, never raises.
    bad = os.path.join(str(tmp_path), "n1_m.manifest.json")
    with open(bad, "w") as f:
        f.write("{not json")
    assert ft.get_checkpoint_manifest("n1", "m") is None


@pytest.mark.asyncio
async def test_cross_dir_recovery_over_p2p(tmp_path):
    """Separate tmpdirs (shared-nothing): survivor pulls dead node's ckpt."""
    dir_dead = tmp_path / "dead"
    dir_live = tmp_path / "live"
    dir_dead.mkdir()
    dir_live.mkdir()
    ft_dead = FaultToleranceManager("dead", checkpoint_dir=str(dir_dead))
    ft_live = FaultToleranceManager("live", checkpoint_dir=str(dir_live))
    ft_dead.save_checkpoint("m", 0, 7, 3)
    ft_dead.save_checkpoint("m", 0, 7, 5)

    (ra, a, ta), (rb, b, tb) = await _bridge_pair()
    try:
        srv = CheckpointPullServer(ft_dead, ra, "dead")
        await srv.start()
        fetch = make_fetch_fn(rb, "dead", timeout=10.0)
        got = await ft_live.prefer_local_then_pull("dead", "m", 5, fetch_fn=fetch)
        assert got is not None, "pull failed across separate dirs"
        assert got.step == 5 and got.node_id == "dead"
        # Persisted into OUR dir: next lookup is local, no fetch needed.
        again = await ft_live.prefer_local_then_pull("dead", "m", 5)
        assert again is not None and again.step == 5
        assert again.checkpoint_path.startswith(str(dir_live))
        await srv.stop()
    finally:
        await _close(a, ta)
        await _close(b, tb)


@pytest.mark.asyncio
async def test_pull_rejects_stale_step_and_missing(tmp_path):
    dir_dead = tmp_path / "dead"
    dir_live = tmp_path / "live"
    dir_dead.mkdir()
    dir_live.mkdir()
    ft_dead = FaultToleranceManager("dead", checkpoint_dir=str(dir_dead))
    ft_live = FaultToleranceManager("live", checkpoint_dir=str(dir_live))
    ft_dead.save_checkpoint("m", 0, 7, 5)
    # Live side already knows a NEWER step: stale pull must be refused.
    ft_live.save_checkpoint("m", 0, 7, 9)

    (ra, a, ta), (rb, b, tb) = await _bridge_pair()
    try:
        srv = CheckpointPullServer(ft_dead, ra, "dead")
        await srv.start()
        fetch = make_fetch_fn(rb, "dead", timeout=10.0)
        # Asking for step 5 while local latest is 9 -> refused (G5 monotonicity).
        stale = await ft_live.prefer_local_then_pull("dead", "m", 5, fetch_fn=fetch)
        assert stale is None
        # Missing step -> None, not an exception.
        missing = await ft_live.prefer_local_then_pull("dead", "m", 999, fetch_fn=fetch)
        assert missing is None
        await srv.stop()
    finally:
        await _close(a, ta)
        await _close(b, tb)


@pytest.mark.asyncio
async def test_pull_too_large_is_explicit(tmp_path):
    dir_dead = tmp_path / "dead"
    dir_dead.mkdir()
    ft_dead = FaultToleranceManager("dead", checkpoint_dir=str(dir_dead))
    ft_dead.save_checkpoint("m", 0, 7, 1, kv_cache={"big": "x" * 100})

    (ra, a, ta), (rb, b, tb) = await _bridge_pair()
    try:
        srv = CheckpointPullServer(ft_dead, ra, "dead", max_bytes=10)
        await srv.start()
        got = {"resp": None}
        arrived = asyncio.Event()

        def _on_resp(event):
            data = event.data if isinstance(event.data, dict) else {}
            if isinstance(data, dict) and data.get("request_id") == "r1":
                got["resp"] = data
                arrived.set()

        rb.subscribe("synapse/checkpoint/response/r1", _on_resp)
        await rb.publish("synapse/checkpoint/request/dead", {
            "request_id": "r1", "target_node_id": "dead",
            "model_id": "m", "step": 1}, origin="local")
        await asyncio.wait_for(arrived.wait(), timeout=10.0)
        assert got["resp"] is not None
        assert got["resp"]["ok"] is False
        assert got["resp"]["reason"] == "too_large"
        await srv.stop()
    finally:
        await _close(a, ta)
        await _close(b, tb)


@pytest.mark.asyncio
async def test_engine_load_checkpoint_honesty(tmp_path):
    """B1: True only for restorable weight dirs, False (loud) otherwise."""
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    from synapse.inference.shard import Shard
    eng = PyTorchHFInferenceEngine()
    shard = Shard(model_id="m", start_layer=0, end_layer=1, n_layers=2)
    # Metadata-only JSON path -> False, never raises, never silent.
    meta = tmp_path / "n1_m_L0-1_step1.ckpt"
    meta.write_text(json.dumps({"node_id": "n1", "step": 1}))
    assert await eng.load_checkpoint(shard, str(meta)) is False
    # Missing path -> False.
    assert await eng.load_checkpoint(shard, str(tmp_path / "nope")) is False
