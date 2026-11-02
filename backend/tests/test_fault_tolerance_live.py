"""
Test: Fault tolerance heartbeat detection and rebalance wiring.

Verifies the fixes for:
- B2: check_timeout() is called in _monitor_loop (not orphaned)
- B3: pickle replaced with JSON-safe serialization
- M2: HEARTBEAT_TIMEOUT matches README 5s claim

All tests are behavioral: they run the shipped monitor loop, write real
checkpoint files, and drive the production Node.check_fault_tolerance() path.
"""
import asyncio
import json
import os
import tempfile
import time

import numpy as np
import pytest


class TestFaultToleranceLive:
    """Tests that exercise the real FaultToleranceManager on its shipped path."""

    def test_heartbeat_timeout_matches_readme(self):
        """HEARTBEAT_TIMEOUT must be 5.0 as claimed in README."""
        from synapse.pipeline.fault_tolerance import FaultToleranceManager
        assert FaultToleranceManager.HEARTBEAT_TIMEOUT == 5.0, (
            f"HEARTBEAT_TIMEOUT is {FaultToleranceManager.HEARTBEAT_TIMEOUT}, expected 5.0"
        )

    def test_heartbeat_interval_constant(self):
        """HEARTBEAT_INTERVAL must be 2.0 for 5s timeout detection."""
        from synapse.pipeline.fault_tolerance import FaultToleranceManager
        assert FaultToleranceManager.HEARTBEAT_INTERVAL == 2.0

    @pytest.mark.asyncio
    async def test_monitor_loop_detects_timeout(self):
        """The REAL _monitor_loop task detects a node whose heartbeat expired."""
        from synapse.pipeline.fault_tolerance import FaultToleranceManager

        ft = FaultToleranceManager(node_id="test_orch")
        ft.register_node("node2")
        ft.record_heartbeat("node2")
        ft.HEARTBEAT_INTERVAL = 0.1
        ft.HEARTBEAT_TIMEOUT = 1.0
        # 6s-old heartbeat exceeds the 1s test timeout.
        ft._health["node2"].last_heartbeat = time.time() - 6.0

        await ft.start()
        try:
            for _ in range(50):
                if not ft._health["node2"].is_alive:
                    break
                await asyncio.sleep(0.1)
            assert not ft._health["node2"].is_alive, (
                "monitor loop never timed out the stale node"
            )
            assert "node2" in ft.get_dead_nodes()
        finally:
            await ft.stop()

    @pytest.mark.asyncio
    async def test_monitor_loop_sets_pending_rebalance(self):
        """The shipped monitor loop calls check_timeout and flags the dead node
        for rebalance (behavioral — no source inspection)."""
        from synapse.pipeline.fault_tolerance import FaultToleranceManager

        ft = FaultToleranceManager(node_id="test_orch")
        ft.register_node("node2")
        ft.record_heartbeat("node2")
        ft.HEARTBEAT_INTERVAL = 0.1
        ft.HEARTBEAT_TIMEOUT = 1.0
        ft._health["node2"].last_heartbeat = time.time() - 6.0

        await ft.start()
        try:
            for _ in range(50):
                if ft._pending_rebalance == "node2":
                    break
                await asyncio.sleep(0.1)
            assert ft._pending_rebalance == "node2", (
                "monitor loop did not flag the dead node for rebalance"
            )
        finally:
            await ft.stop()

    @pytest.mark.asyncio
    async def test_check_timeout_detects_dead_node(self):
        """Verify check_timeout() sets is_alive=False on timeout."""
        from synapse.pipeline.fault_tolerance import (
            FaultToleranceManager,
            NodeHealth,
        )

        # Direct test: check a NodeHealth with old heartbeat
        h = NodeHealth(node_id="test_node")
        h.record_heartbeat()
        assert h.is_alive is True

        # Simulate old heartbeat by tampering with last_heartbeat
        h.last_heartbeat = time.time() - 10.0  # 10s old

        # check_timeout should detect this
        timed_out = h.check_timeout(timeout_s=5.0)
        assert timed_out is True
        assert h.is_alive is False
        assert h.consecutive_misses >= 1

    @pytest.mark.asyncio
    async def test_save_checkpoint_writes_plain_json(self):
        """Checkpoint files are written as plain JSON text (no pickle, no
        binary protocol markers) — verified by reading the real file bytes."""
        from synapse.pipeline.fault_tolerance import FaultToleranceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            ft = FaultToleranceManager(node_id="test_node", checkpoint_dir=tmpdir)
            path = ft.save_checkpoint(
                model_id="test-model",
                start_layer=0,
                end_layer=7,
                step=1,
                kv_cache={"layer_0": [1, 2, 3]},
            )

            with open(path, "rb") as f:
                raw = f.read()
            # Pickle protocol frames start with 0x80; JSON never does.
            assert b"\x80" not in raw, "checkpoint contains pickle protocol bytes"
            text = raw.decode("utf-8")
            data = json.loads(text)
            assert data["model_id"] == "test-model"
            assert data["node_id"] == "test_node"
            assert data["kv_cache"] == {"layer_0": [1, 2, 3]}

            # And it loads back through the shipped loader.
            loaded = ft.load_checkpoint("test-model", 0, 7, 1)
            assert loaded is not None
            assert loaded.start_layer == 0
            assert loaded.end_layer == 7

    @pytest.mark.asyncio
    async def test_checkpoint_json_roundtrip(self):
        """Verify checkpoint save/load roundtrip works with JSON serialization."""
        from synapse.pipeline.fault_tolerance import FaultToleranceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            ft = FaultToleranceManager(node_id="test_node", checkpoint_dir=tmpdir)

            # Save a checkpoint
            path = ft.save_checkpoint(
                model_id="test-model",
                start_layer=0,
                end_layer=7,
                step=1,
                kv_cache={"layer_0": [1, 2, 3]},
            )

            # Verify file is valid JSON
            with open(path, "r") as f:
                data = json.load(f)
            assert data["model_id"] == "test-model"
            assert data["start_layer"] == 0
            assert data["end_layer"] == 7
            assert data["node_id"] == "test_node"

            # Load it back
            loaded = ft.load_checkpoint("test-model", 0, 7, 1)
            assert loaded is not None
            assert loaded.model_id == "test-model"
            assert loaded.start_layer == 0
            assert loaded.end_layer == 7

    @pytest.mark.asyncio
    async def test_rebalance_after_failure_wired_in_node(self):
        """Node.check_fault_tolerance() invokes rebalance_after_failure() and
        consumes the result (behavioral — drives the real production method)."""
        from synapse.inference.inference_engine import InferenceEngine
        from synapse.orchestration.node import Node
        from synapse.pipeline.fault_tolerance import FaultToleranceManager
        from synapse.routing import EventRouter
        from synapse.routing.libp2p_node import Libp2pNode
        from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
        from synapse.topology.topology import Topology

        class _DummyEngine(InferenceEngine):
            async def encode(self, shard, prompt):
                return np.array([])

            async def sample(self, x):
                return np.array([1])

            async def decode(self, shard, tokens):
                return ""

            async def infer_tensor(self, rid, s, t, state=None):
                return np.array([]), {}

            async def load_checkpoint(self, s, p):
                pass

        def _dc(vram_mb, fp16_tflops):
            return DeviceCapabilities(
                model="Test-GPU", chip="Test-Chip", memory=vram_mb,
                flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
                cpu_cores=4, disk_gb=100, system_ram_mb=8192,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            router = EventRouter()
            libp2p = Libp2pNode("master", 59999, router)
            node = Node(
                _id="master",
                event_router=router,
                libp2p_node=libp2p,
                inference_engine=_DummyEngine(),
                shard_downloader=None,
            )
            node.tflops_proportional = False
            topo = Topology()
            topo.update_node("node-a", _dc(4096, 2.0))
            topo.update_node("node-b", _dc(6144, 3.0))
            node.topology = topo

            ft = FaultToleranceManager("master", checkpoint_dir=tmpdir)
            ft.register_node("node-a")
            ft.register_node("node-b")
            ft._health["node-b"].is_alive = False
            ft._health["node-b"].is_blacklisted = True
            node.fault_tolerance = ft
            node._distributed_pipeline = object()  # truthy: distributed mode on

            await node.check_fault_tolerance()

            assert node._last_rebalance_result is not None, (
                "check_fault_tolerance() did not consume a rebalance result"
            )
            result = node._last_rebalance_result
            assert result.success
            # node-b dropped, survivors get the full layer range.
            assert "node-b" not in result.new_assignments

    @pytest.mark.asyncio
    async def test_rebalance_applies_recovery_checkpoint_to_survivor(self):
        """Cross-node recovery hands the dead node's on-disk JSON checkpoint to
        the survivor's engine via load_checkpoint() — the recovery path must
        resume state from disk, not just print that it found one (behavioral:
        drives Node.check_fault_tolerance() end to end with a real ckpt file)."""
        from synapse.inference.inference_engine import InferenceEngine
        from synapse.orchestration.node import Node
        from synapse.pipeline.fault_tolerance import FaultToleranceManager
        from synapse.routing import EventRouter
        from synapse.routing.libp2p_node import Libp2pNode
        from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
        from synapse.topology.topology import Topology

        class _SpyEngine(InferenceEngine):
            def __init__(self):
                self.loaded = []

            async def encode(self, shard, prompt):
                return np.array([])

            async def sample(self, x):
                return np.array([1])

            async def decode(self, shard, tokens):
                return ""

            async def infer_tensor(self, rid, s, t, state=None):
                return np.array([]), {}

            async def load_checkpoint(self, shard, path):
                self.loaded.append((shard, path))

        def _dc(vram_mb, fp16_tflops):
            return DeviceCapabilities(
                model="Test-GPU", chip="Test-Chip", memory=vram_mb,
                flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
                cpu_cores=4, disk_gb=100, system_ram_mb=8192,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            router = EventRouter()
            libp2p = Libp2pNode("master", 60001, router)
            engine = _SpyEngine()
            node = Node(
                _id="master",
                event_router=router,
                libp2p_node=libp2p,
                inference_engine=engine,
                shard_downloader=None,
            )
            node.tflops_proportional = False
            topo = Topology()
            topo.update_node("node-a", _dc(4096, 2.0))
            topo.update_node("node-b", _dc(6144, 3.0))
            node.topology = topo

            ft = FaultToleranceManager("master", checkpoint_dir=tmpdir)
            ft.register_node("node-a")
            ft.register_node("node-b")
            # node-b dies; its checkpoint lives on disk only (a different
            # process wrote it, so master's in-memory store never held it).
            ft._health["node-b"].is_alive = False
            ft._health["node-b"].is_blacklisted = True
            node.fault_tolerance = ft
            node._distributed_pipeline = object()  # truthy: distributed mode on

            # Write node-b's checkpoint to disk the way node-b itself would
            # have: node-prefixed JSON in the shared checkpoint directory.
            dead_writer = FaultToleranceManager("node-b", checkpoint_dir=tmpdir)
            path = dead_writer.save_checkpoint(
                model_id="qwen",
                start_layer=0,
                end_layer=31,
                step=7,
                kv_cache={"layer_5": np.zeros((2, 4))},
            )

            await node.check_fault_tolerance()

            result = node._last_rebalance_result
            assert result is not None and result.success
            assert result.recovered_from_checkpoint, (
                "rebalance did not recover node-b's on-disk checkpoint"
            )
            assert result.recovery_target_node == "node-a", (
                f"expected survivor node-a, got {result.recovery_target_node}"
            )
            assert len(engine.loaded) == 1, (
                f"engine.load_checkpoint called {len(engine.loaded)} times, expected 1"
            )
            shard, ckpt_path = engine.loaded[0]
            assert (shard.start_layer, shard.end_layer) == (0, 31), (
                f"survivor should take over the dead node's range, got "
                f"{shard.start_layer}-{shard.end_layer}"
            )
            assert os.path.basename(ckpt_path) == os.path.basename(path), (
                f"engine got {ckpt_path!r}, expected the on-disk ckpt {path!r}"
            )
            assert os.path.isfile(ckpt_path), "load_checkpoint path must exist on disk"

    @pytest.mark.asyncio
    async def test_distributed_ddp_step_writes_json_checkpoint(self, monkeypatch, tmp_path):
        """A distributed training step persists a real JSON checkpoint to the
        shared checkpoint directory — C-H1: production code writes checkpoints
        after every training step, so cross-node recovery restores real
        on-disk state (behavioral: drives train_step_distributed end to end)."""
        from synapse.inference.inference_engine import InferenceEngine
        from synapse.inference.shard import Shard
        from synapse.orchestration.node import Node
        from synapse.pipeline.gradient_sync import DistributedTrainer
        from synapse.routing import EventRouter
        from synapse.routing.libp2p_node import Libp2pNode
        from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
        from synapse.topology.topology import Topology

        class _CkptEngine(InferenceEngine):
            async def encode(self, shard, prompt):
                return np.array([])

            async def sample(self, x):
                return np.array([1])

            async def decode(self, shard, tokens):
                return ""

            async def infer_tensor(self, rid, s, t, state=None):
                return np.array([]), {}

            async def load_checkpoint(self, s, p):
                pass

            async def train(self, request_id, shard, example, target, length,
                            backgrad=None, loss="forward", sync_gradients=False):
                return 1.5, {"w": np.array([0.5, -0.25], dtype=np.float32)}

            async def apply_gradient_updates(self, shard, updates):
                pass

        def _dc(vram_mb, fp16_tflops):
            return DeviceCapabilities(
                model="Test-GPU", chip="Test-Chip", memory=vram_mb,
                flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
                cpu_cores=4, disk_gb=100, system_ram_mb=8192,
            )

        router = EventRouter()
        libp2p = Libp2pNode("node-a", 59999, router)
        node = Node(
            _id="node-a",
            event_router=router,
            libp2p_node=libp2p,
            inference_engine=_CkptEngine(),
            shard_downloader=None,
        )
        node.tflops_proportional = False
        topo = Topology()
        topo.update_node("node-a", _dc(4096, 2.0))
        topo.update_node("node-b", _dc(6144, 3.0))
        node.topology = topo

        # Point the (lazily created) fault-tolerance manager at tmp_path.
        node._lazy_init_fault_tolerance()
        node.fault_tolerance.checkpoint_dir = str(tmp_path)

        # No real ring I/O in the test.
        async def fake_step(self, local_gradients, local_loss, send_fn=None, recv_fn=None):
            return {"w": np.zeros(2, dtype=np.float32)}

        monkeypatch.setattr(DistributedTrainer, "step", fake_step)

        ext = node.distributed_pipeline
        input_ids = np.array([[1, 2, 3], [4, 5, 6]])
        labels = np.array([[2, 3, 4], [5, 6, 7]])
        loss = await ext.train_step_distributed(
            Shard(model_id="qwen", start_layer=0, end_layer=31, n_layers=32),
            input_ids, labels, request_id="req-ckpt-ddp",
        )
        assert loss == pytest.approx(1.5)

        # A real JSON checkpoint file now exists in the checkpoint directory.
        files = [p for p in tmp_path.iterdir() if p.suffix == ".ckpt"]
        assert len(files) == 1, (
            f"expected one checkpoint file, got {[f.name for f in tmp_path.iterdir()]}"
        )
        raw = files[0].read_bytes()
        assert b"\x80" not in raw, "checkpoint contains pickle protocol bytes"
        data = json.loads(raw.decode("utf-8"))
        assert data["model_id"] == "qwen"
        assert data["node_id"] == "node-a"
        assert data["step"] == 1
        # The shard range is node-a's actual LACP partition slice (first in
        # ring order), i.e. real on-disk state for this node's shard.
        assert 0 <= data["start_layer"] <= data["end_layer"] <= 31

