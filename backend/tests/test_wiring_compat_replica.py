"""
Two pieces that existed but were never connected:

1. Handshake compatibility. bridge.is_compatible() (bridge major version)
   was only called by tests; the README said the node refuses mixed-version
   pipelines but nothing did. Now an incompatible peer never gets layers
   (placement and failover plans skip it) and gets no checkpoint replicas.
2. Checkpoint replication. A training checkpoint (JSON + shard weights)
   lived only on the machine that wrote it, so it died with that machine
   (the old "pull from the dead node" path could never work). Now each new
   checkpoint is shipped to the pipeline successor, which stores it under
   the writer's name: load_latest_checkpoint(dead_node) on the survivor
   finds it, weights included.
"""
import asyncio
import os

import numpy as np
import pytest

from synapse.inference.shard import Shard
from synapse.pipeline.fault_tolerance import FaultToleranceManager
import test_measured_placement as _placement  # tests/ is on sys.path


def _write_ckpt(ft, tmp_path, payload=b"layer-weights-\x00\x01\x02" * 100):
    wpath = os.path.join(ft.checkpoint_dir, "A_m_L0-4.weights.pt")
    with open(wpath, "wb") as f:
        f.write(payload)
    return ft.save_checkpoint("m", 0, 4, step=3, weights_path=wpath, weights_step=3), payload


class TestReplicaRoundTrip:
    def test_survivor_recovers_dead_nodes_checkpoint_with_weights(self, tmp_path):
        a = FaultToleranceManager("A", str(tmp_path / "a"))
        b = FaultToleranceManager("B", str(tmp_path / "b"))
        path, payload = _write_ckpt(a, tmp_path)
        meta, weights = a.export_replica(path)
        stored = b.import_replica(meta, weights)
        assert stored and os.path.dirname(stored) == os.path.abspath(str(tmp_path / "b"))
        got = b.load_latest_checkpoint("A")  # A is dead; B only has the replica
        assert got is not None and got.node_id == "A" and got.step == 3
        assert got.weights_path.startswith(os.path.abspath(str(tmp_path / "b")))
        with open(got.weights_path, "rb") as f:
            assert f.read() == payload

    def test_tampered_or_hostile_replicas_are_rejected(self, tmp_path):
        a = FaultToleranceManager("A", str(tmp_path / "a"))
        b = FaultToleranceManager("B", str(tmp_path / "b"))
        path, _ = _write_ckpt(a, tmp_path)
        meta, weights = a.export_replica(path)
        assert b.import_replica(meta, weights[:-1] + b"X") is None       # weights tampered
        assert b.import_replica(dict(meta, ckpt_json=meta["ckpt_json"] + " "), weights) is None
        assert b.import_replica(dict(meta, from_node="B"), weights) is None  # claims to be ours
        evil = dict(meta, ckpt_name="../../escape.ckpt", weights_name="../../escape.pt")
        stored = b.import_replica(evil, weights)
        assert stored is not None
        assert os.path.dirname(stored) == os.path.abspath(str(tmp_path / "b"))
        assert not (tmp_path / "escape.ckpt").exists()


class TestNodeWiring:
    def test_incompatible_peer_gets_no_layers(self, monkeypatch):
        node = _placement.TestNodeWiring._node(monkeypatch, ("A", "B", "C"))

        class Bridge:
            def is_compatible(self, nid):
                return nid != "B"
        node.libp2p_node._bridge = Bridge()
        parts = node._compute_lacp_partition(Shard("m", 0, 29, 30))
        ids = [p.node_id for p in parts]
        assert "B" not in ids and set(ids) == {"A", "C"}
        cur = [("A", Shard("m", 0, 14, 30)), ("C", Shard("m", 15, 29, 30))]
        new = node._replan_excluding(Shard("m", 0, 29, 30), cur, {"C"})
        assert new and all(n != "B" for n, _ in new)

    @pytest.mark.asyncio
    async def test_checkpoint_reaches_the_successor_over_the_mesh(self, monkeypatch, tmp_path):
        from synapse.topology.partitioning_strategy import Partition
        a = _placement.TestNodeWiring._node(monkeypatch, ("A", "B"))
        b = _placement.TestNodeWiring._node(monkeypatch, ("B", "A"))
        b.event_router = a.event_router  # one in-process mesh
        a.event_router.subscribe("synapse/checkpoint/replica/B", b.on_p2p_ckpt_replica)
        a.fault_tolerance = FaultToleranceManager("A", str(tmp_path / "a"))
        b.fault_tolerance = FaultToleranceManager("B", str(tmp_path / "b"))
        a._authoritative_partitions = [Partition("A", 0.0, 0.5), Partition("B", 0.5, 1.0)]
        path, payload = _write_ckpt(a.fault_tolerance, tmp_path)
        await a.event_router.start()
        try:
            assert await a._replicate_checkpoint(path) == "B"
            for _ in range(50):
                if b.fault_tolerance.load_latest_checkpoint("A"):
                    break
                await asyncio.sleep(0.05)
        finally:
            await a.event_router.stop()
        got = b.fault_tolerance.load_latest_checkpoint("A")
        assert got is not None and got.step == 3
        with open(got.weights_path, "rb") as f:
            assert f.read() == payload

    @pytest.mark.asyncio
    async def test_replication_can_be_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNAPSE_CKPT_REPLICATE", "0")
        a = _placement.TestNodeWiring._node(monkeypatch, ("A", "B"))
        a.fault_tolerance = FaultToleranceManager("A", str(tmp_path / "a"))
        path, _ = _write_ckpt(a.fault_tolerance, tmp_path)
        assert await a._replicate_checkpoint(path) is None
