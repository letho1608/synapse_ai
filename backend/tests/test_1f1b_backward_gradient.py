"""
Test: real 1F1B backward phase (A2).

The backward phase of the shipped 1F1B schedule (train_step_distributed
with schedule="1f1b") must produce real gradient values from the engine —
no longer the None/np.nan placeholder — and hand them off between stages
through the pipeline send path (no longer a noop_send), in reverse stage
order: stage k's backward consumes stage k+1's input-gradient as backgrad.

Behavioral: drives the real PipelineScheduler + PipelineStage objects
through DistributedTrainer.train_1f1b(). The engine stub implements the
documented engine contract — back_gradient returns a genuine deterministic
function of its input — so the test can verify the exact gradient values
that flowed through the handoff chain.
"""
import numpy as np
import pytest

from synapse.inference.inference_engine import InferenceEngine
from synapse.inference.shard import Shard
from synapse.orchestration.node import Node
from synapse.pipeline.gradient_sync import DistributedTrainer
from synapse.routing import EventRouter
from synapse.routing.libp2p_node import Libp2pNode
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.topology.topology import Topology


def _dc(vram_mb, fp16_tflops):
    return DeviceCapabilities(
        model="Test-GPU", chip="Test-Chip", memory=vram_mb,
        flops=DeviceFlops(fp32=fp16_tflops / 2, fp16=fp16_tflops, int8=fp16_tflops * 2),
        cpu_cores=4, disk_gb=100, system_ram_mb=8192,
    )


def _grad_of(example):
    """The engine contract's deterministic gradient of loss w.r.t. input."""
    return (example.astype(np.float32) + 1.0) * 2.0


class _GradEngine(InferenceEngine):
    """Implements the documented engine contract: back_gradient returns a
    genuine gradient (a deterministic function of its input) and records
    every backward call so the test can verify the reverse-order handoff."""

    def __init__(self):
        self.backward_calls = []  # {shard_start, example, backgrad}
        self.apply_calls = []

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
        if loss == "back_gradient":
            self.backward_calls.append({
                "shard_start": shard.start_layer,
                "example": example.copy(),
                "backgrad": None if backgrad is None else backgrad.copy(),
            })
            return None, {"__input_grad__": _grad_of(example)}
        return 1.5, {"w": np.array([0.5, -0.25], dtype=np.float32)}

    async def apply_gradient_updates(self, shard, updates):
        self.apply_calls.append(updates)


def _node_with_topology(n_nodes=2):
    router = EventRouter()
    libp2p = Libp2pNode("node-a", 59999, router)
    engine = _GradEngine()
    node = Node(
        _id="node-a",
        event_router=router,
        libp2p_node=libp2p,
        inference_engine=engine,
        shard_downloader=None,
    )
    node.tflops_proportional = False
    topo = Topology()
    for i in range(n_nodes):
        topo.update_node(f"node-{chr(ord('a') + i)}", _dc(4096 + i * 2048, 2.0 + i))
    node.topology = topo
    return node, engine


class Test1F1BBackwardGradient:
    @pytest.mark.asyncio
    async def test_backward_produces_real_gradients_and_handoff(self, monkeypatch):
        """A2: back_gradient returns real gradient values and the stage send
        path carries them upstream, consumed as backgrad by the upstream
        stage (reverse-order handoff through the real scheduler/stages)."""
        node, engine = _node_with_topology(n_nodes=2)
        ext = node.distributed_pipeline

        step_calls = {}

        async def fake_step(self, local_gradients, local_loss, send_fn=None, recv_fn=None):
            step_calls["grads"] = local_gradients
            step_calls["loss"] = local_loss
            return {"w": np.zeros(2, dtype=np.float32)}

        monkeypatch.setattr(DistributedTrainer, "step", fake_step)

        input_ids = np.array([[1, 2, 3], [4, 5, 6]])
        labels = np.array([[2, 3, 4], [5, 6, 7]])
        loss = await ext.train_step_distributed(
            Shard(model_id="qwen", start_layer=0, end_layer=31, n_layers=32),
            input_ids, labels,
            request_id="req-a2-2stage",
            schedule="1f1b",
        )

        # 1. Loss still comes from the last stage's forwards.
        assert loss == pytest.approx(1.5)
        assert "w" in step_calls["grads"]
        assert len(engine.apply_calls) == 1

        # 2. The backward phase ran for every stage and every micro-batch
        #    (2 stages x 2 micro-batches = 4 backward calls).
        bwd = engine.backward_calls
        assert len(bwd) == 4, f"expected 4 backward calls, got {len(bwd)}"

        # Identify stages by their shard range (no assumption on values).
        starts = sorted({c["shard_start"] for c in bwd})
        assert len(starts) == 2, f"expected 2 stages, got {starts}"
        first_start, last_start = starts[0], starts[1]
        last_calls = [c for c in bwd if c["shard_start"] == last_start]
        first_calls = [c for c in bwd if c["shard_start"] == first_start]
        assert len(last_calls) == 2 and len(first_calls) == 2

        # 3. The last stage owns the loss: it has no downstream gradient.
        assert all(c["backgrad"] is None for c in last_calls), (
            "last stage received a backgrad, but it owns the loss"
        )

        # 4. Every non-last stage consumed the downstream stage's real
        #    input-gradient as its backgrad (reverse-order handoff), for
        #    every micro-batch slice.
        for c in first_calls:
            m = 0 if np.array_equal(c["example"], input_ids[0:1]) else 1
            expected = _grad_of(input_ids[m:m + 1])
            assert c["backgrad"] is not None, "non-last stage got no backgrad"
            assert np.allclose(c["backgrad"], expected), (
                f"stage {c['shard_start']} backgrad is not the downstream "
                "stage's input-gradient for the same micro-batch"
            )

        # 5. Handoff occurred through the stage send path (was a noop_send):
        #    the non-last stage sent its input-gradient upstream for every mb.
        assert sorted(ext._last_handoff_sends) == [(0, 0), (0, 1)], (
            f"send path did not carry backward handoffs: {ext._last_handoff_sends}"
        )

    @pytest.mark.asyncio
    async def test_three_stage_backward_handoff_chain(self, monkeypatch):
        """A2: with 3 stages the reverse-order chain holds end to end —
        every non-last stage receives the downstream input-gradient, and the
        send path carries handoffs from both intermediate stages."""
        node, engine = _node_with_topology(n_nodes=3)
        ext = node.distributed_pipeline

        async def fake_step(self, local_gradients, local_loss, send_fn=None, recv_fn=None):
            return {"w": np.zeros(2, dtype=np.float32)}

        monkeypatch.setattr(DistributedTrainer, "step", fake_step)

        input_ids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        loss = await ext.train_step_distributed(
            Shard(model_id="qwen", start_layer=0, end_layer=31, n_layers=32),
            input_ids, labels,
            request_id="req-a2-3stage",
            schedule="1f1b",
        )
        assert loss == pytest.approx(1.5)

        bwd = engine.backward_calls
        # 3 stages x 3 micro-batches = 9 backward calls.
        assert len(bwd) == 9, f"expected 9 backward calls, got {len(bwd)}"

        starts = sorted({c["shard_start"] for c in bwd})
        assert len(starts) == 3
        first_start, mid_start, last_start = starts

        # Last stage: no downstream gradient.
        assert all(c["backgrad"] is None for c in bwd if c["shard_start"] == last_start)

        # Every non-last stage got the downstream stage's input-gradient.
        for c in bwd:
            if c["shard_start"] == last_start:
                continue
            m = int(np.argmax([np.array_equal(c["example"], input_ids[m:m + 1])
                               for m in range(3)]))
            assert c["backgrad"] is not None
            assert np.allclose(c["backgrad"], _grad_of(input_ids[m:m + 1])), (
                f"stage {c['shard_start']} backgrad is not the downstream "
                "stage's input-gradient"
            )

        # Send path carried handoffs from both non-last stages for every mb.
        assert sorted(ext._last_handoff_sends) == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        ], f"send path handoffs missing: {ext._last_handoff_sends}"
