"""
Behavioral tests: cross-node checkpoint recovery.

Verifies the fixes for:
- FT-cross-node: a checkpoint saved by a *different* node (on the shared
  checkpoint dir) is found by load_latest_checkpoint() during recovery — the
  orchestrator's in-memory dict can never hold another node's state, so
  recovery must fall back to disk.
- FT-consume: rebalance_after_failure() returns the recovered checkpoint in a
  proper RebalanceResult field (not a bogus "_recovery_checkpoint" key stuffed
  into the assignments dict) and stages it for the surviving node that takes
  over the dead node's shard range; Node.check_fault_tolerance() consumes the
  result instead of discarding it.
"""
import os
import tempfile

import numpy as np
import pytest

from synapse.inference.inference_engine import InferenceEngine
from synapse.orchestration.node import Node
from synapse.pipeline.fault_tolerance import FaultToleranceManager, ShardCheckpoint
from synapse.routing import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.topology.topology import Topology


def _dc(vram_mb: int, fp16_tflops: float) -> DeviceCapabilities:
    return DeviceCapabilities(
        model="Test-GPU",
        chip="Test-Chip",
        memory=vram_mb,
        flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
        cpu_cores=4,
        disk_gb=100,
        system_ram_mb=8192,
    )


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


def _dead_node_checkpoint(tmpdir, model_id="qwen", start=16, end=31,
                          step=42, kv_cache=None, optimizer_state=None):
    """Simulate the dead node saving its own state to the shared dir."""
    dead = FaultToleranceManager("node-b", checkpoint_dir=tmpdir)
    return dead.save_checkpoint(
        model_id=model_id,
        start_layer=start,
        end_layer=end,
        step=step,
        kv_cache=kv_cache or {"layer_16": [1, 2, 3]},
        optimizer_state=optimizer_state or {"opt": {"lr": 0.001}},
    )


class TestCrossNodeRecovery:
    def test_save_checkpoint_filename_includes_node_id(self):
        """Filenames must be node-prefixed so shards from different nodes do not collide."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _dead_node_checkpoint(tmpdir)
            assert os.path.basename(path) == "node-b_qwen_L16-31_step42.ckpt", (
                f"unexpected filename: {os.path.basename(path)}"
            )

    def test_load_latest_checkpoint_falls_back_to_disk(self):
        """A different manager (the orchestrator) recovers a dead node's on-disk state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _dead_node_checkpoint(
                tmpdir,
                kv_cache={"layer_16": [7, 8, 9]},
                optimizer_state={"opt": {"lr": 0.01}},
            )
            orch = FaultToleranceManager("master", checkpoint_dir=tmpdir)
            # Orchestrator's in-memory dict has nothing for node-b:
            assert orch._checkpoints.get("node-b") is None

            ckpt = orch.load_latest_checkpoint("node-b")
            assert ckpt is not None, "cross-node disk recovery returned None"
            assert ckpt.node_id == "node-b"
            assert ckpt.step == 42
            assert ckpt.start_layer == 16 and ckpt.end_layer == 31
            assert ckpt.kv_cache == {"layer_16": [7, 8, 9]}
            assert ckpt.optimizer_state == {"opt": {"lr": 0.01}}

    def test_load_checkpoint_finds_node_prefixed_file(self):
        """The specific loader still finds checkpoints saved with the node prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _dead_node_checkpoint(tmpdir)
            orch = FaultToleranceManager("master", checkpoint_dir=tmpdir)
            loaded = orch.load_checkpoint("qwen", 16, 31, step=42)
            assert loaded is not None
            assert loaded.node_id == "node-b"
            assert loaded.step == 42

    @pytest.mark.asyncio
    async def test_rebalance_recovers_checkpoint_from_disk(self):
        """rebalance_after_failure recovers the dead node's checkpoint from disk,
        returns it in a proper field with a recovery target, and stages it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _dead_node_checkpoint(tmpdir, kv_cache={"layer_16": "state"})

            ft = FaultToleranceManager("master", checkpoint_dir=tmpdir)
            ft.register_node("node-a")
            ft.register_node("node-b")

            async def fake_lacp(alive_nodes, total_layers):
                return {"node-a": (0, total_layers - 1)}

            result = await ft.rebalance_after_failure(
                "node-b",
                {"node-a": (0, 15), "node-b": (16, 31)},
                total_layers=32,
                lacp_partition_fn=fake_lacp,
            )

            assert result.success
            assert result.recovered_from_checkpoint is True
            assert "_recovery_checkpoint" not in result.new_assignments, (
                "checkpoint must not be stuffed into the assignments dict"
            )
            assert result.recovery_checkpoint is not None
            assert result.recovery_checkpoint.node_id == "node-b"
            assert result.recovery_checkpoint.kv_cache == {"layer_16": "state"}
            # node-a absorbs node-b's shard range (16..31) -> it is the target
            assert result.recovery_target_node == "node-a"
            # Recovered state is staged for the survivor
            staged = ft.get_recovery_checkpoint("node-a")
            assert staged is not None
            assert staged.step == 42
            assert staged.optimizer_state == {"opt": {"lr": 0.001}}

    @pytest.mark.asyncio
    async def test_rebalance_no_checkpoint_is_not_recovery(self):
        """Without any on-disk state, recovery flags stay False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ft = FaultToleranceManager("master", checkpoint_dir=tmpdir)
            ft.register_node("node-a")
            ft.register_node("node-b")

            async def fake_lacp(alive_nodes, total_layers):
                return {"node-a": (0, total_layers - 1)}

            result = await ft.rebalance_after_failure(
                "node-b",
                {"node-a": (0, 15), "node-b": (16, 31)},
                total_layers=32,
                lacp_partition_fn=fake_lacp,
            )

            assert result.success
            assert result.recovered_from_checkpoint is False
            assert result.recovery_checkpoint is None
            assert ft.get_recovery_checkpoint("node-a") is None


class TestNodeCheckFaultToleranceConsumesRecovery:
    """End-to-end: the production Node.check_fault_tolerance() path consumes the
    recovered cross-node checkpoint instead of discarding the rebalance result."""

    @staticmethod
    def _node_with_topology(node_id="master"):
        router = EventRouter()
        libp2p = Libp2pNode(node_id, 59999, router)
        node = Node(
            _id=node_id,
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
        return node

    @pytest.mark.asyncio
    async def test_check_fault_tolerance_recovers_and_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # The dead node left its state on the shared checkpoint dir.
            _dead_node_checkpoint(tmpdir, kv_cache={"layer_16": [4, 5, 6]})

            node = self._node_with_topology()
            ft = FaultToleranceManager("master", checkpoint_dir=tmpdir)
            ft.register_node("node-a")
            ft.register_node("node-b")
            # node-b is blacklisted/dead (as detected by the heartbeat monitor)
            ft._health["node-b"].is_alive = False
            ft._health["node-b"].is_blacklisted = True
            node.fault_tolerance = ft
            node._distributed_pipeline = object()  # truthy: distributed mode on

            await node.check_fault_tolerance()

            # The rebalance result is consumed (not discarded)...
            assert node._last_rebalance_result is not None
            result = node._last_rebalance_result
            assert result.success
            assert result.recovered_from_checkpoint is True
            assert result.recovery_checkpoint is not None
            assert result.recovery_checkpoint.node_id == "node-b"
            assert result.recovery_checkpoint.kv_cache == {"layer_16": [4, 5, 6]}
            # ...and the recovered state is staged for the surviving node that
            # takes over node-b's shard range.
            assert result.recovery_target_node == "node-a"
            staged = ft.get_recovery_checkpoint("node-a")
            assert staged is not None
            assert staged.step == 42
            assert staged.optimizer_state == {"opt": {"lr": 0.001}}
            # The dead node no longer appears in the new assignments.
            assert "node-b" not in result.new_assignments
