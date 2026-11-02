"""
Behavioral tests: production routing of distributed training.

- train_step_distributed() keeps DDP (full replicas + ring all-reduce) as its
  default schedule;
- schedule="1f1b" on a single node is a plain training step;
- the production caller (chatgpt_api.py) selects schedule="1f1b" for
  multi-node training, which runs the real pipeline (each node trains its own
  layers; see test_pipeline_training.py for the cross-node gate).
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


class _TrainDummyEngine(InferenceEngine):
    """Records engine.train calls; returns a fixed loss and gradient."""

    def __init__(self):
        self.train_calls = []
        self.apply_calls = []
        self._grads = {"w": np.array([0.5, -0.25], dtype=np.float32)}

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
        rows = example.shape[0] if example.ndim >= 1 else 1
        self.train_calls.append({
            "loss": loss,
            "sync": sync_gradients,
            "rows": rows,
            "backgrad": backgrad is not None,
        })
        return 1.5, self._grads

    async def apply_gradient_updates(self, shard, updates):
        self.apply_calls.append(updates)


def _node_with_topology(own_id="node-a", n_nodes=2):
    router = EventRouter()
    libp2p = Libp2pNode(own_id, 59999, router)
    engine = _TrainDummyEngine()
    node = Node(
        _id=own_id,
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


class Test1F1BProductionTraining:
    @pytest.mark.asyncio
    async def test_schedule_param_defaults_to_ddp(self):
        """train_step_distributed() keeps its DDP default (backward compatible)."""
        node, engine = _node_with_topology(n_nodes=2)
        ext = node.distributed_pipeline
        # Replace the ring all-reduce step so no real network wait happens.
        calls = {}

        async def fake_step(local_gradients, local_loss, send_fn=None, recv_fn=None):
            calls["grads"] = local_gradients
            calls["loss"] = local_loss
            return {}

        monkeypatch_step = None
        orig_step = DistributedTrainer.step

        async def patched_step(self, *a, **k):
            return await fake_step(*a, **k)

        try:
            DistributedTrainer.step = patched_step
            input_ids = np.array([[1, 2, 3], [4, 5, 6]])
            labels = np.array([[2, 3, 4], [5, 6, 7]])
            loss = await ext.train_step_distributed(
                Shard(model_id="qwen", start_layer=0, end_layer=31, n_layers=32),
                input_ids, labels,
                request_id="req-ddp",
            )
        finally:
            DistributedTrainer.step = orig_step

        # DDP path: engine.train called with sync_gradients=True, no 1F1B schedule
        assert len(engine.train_calls) == 1
        assert engine.train_calls[0]["sync"] is True
        assert loss == pytest.approx(1.5)
        assert calls["loss"] == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_1f1b_single_node_falls_back_to_plain_train(self, monkeypatch):
        """With a single partition, schedule="1f1b" runs a plain training step."""
        node, engine = _node_with_topology(n_nodes=1)
        ext = node.distributed_pipeline

        async def no_pipeline(*a, **k):
            raise AssertionError("a single node must not run the cross-node pipeline")

        monkeypatch.setattr(ext.pp_trainer, "train_step", no_pipeline)

        input_ids = np.array([[1, 2, 3], [4, 5, 6]])
        labels = np.array([[2, 3, 4], [5, 6, 7]])
        loss = await ext.train_step_distributed(
            Shard(model_id="qwen", start_layer=0, end_layer=31, n_layers=32),
            input_ids, labels,
            request_id="req-1node",
            schedule="1f1b",
        )

        assert loss == pytest.approx(1.5)
        assert len(engine.train_calls) == 1
        assert engine.train_calls[0]["sync"] is False

    @pytest.mark.asyncio
    async def test_production_training_routes_to_1f1b_extension(self, monkeypatch):
        """chatgpt_api's multi-node training reaches DistributedPipelineExtension
        .train_step_distributed(schedule="1f1b") — NOT the single-node
        enqueue_example fallback. Drives the real production method
        (_run_training_job_distributed) end to end with a multi-node cluster."""
        import synapse.api.chatgpt_api as capi
        from synapse.api.chatgpt_api import ChatGPTAPI

        node, engine = _node_with_topology(n_nodes=2)
        node.peers = [object()]  # world_size = 2 → multi-node training branch

        class _FakeTokenizer:
            pad_token = "[PAD]"
            eos_token = "</s>"
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, texts, **kwargs):
                n = len(texts)
                ids = np.full((n, 8), 1, dtype=np.int64)
                return {
                    "input_ids": ids,
                    "attention_mask": np.ones((n, 8), dtype=np.int64),
                }

        engine.tokenizer = _FakeTokenizer()

        async def _noop_ensure_shard(shard):
            return None

        engine.ensure_shard = _noop_ensure_shard

        # The multi-node branch must call train_step_distributed and must NOT
        # fall back to the single-node enqueue_example path.
        calls = {}

        async def fake_train_step(self, base_shard, input_ids, labels,
                                  request_id=None, learning_rate=1e-4,
                                  schedule="ddp"):
            calls["schedule"] = schedule
            return 1.5

        monkeypatch.setattr(
            "synapse.pipeline.node_extension.DistributedPipelineExtension.train_step_distributed",
            fake_train_step,
        )

        async def fail_enqueue(*args, **kwargs):
            raise AssertionError(
                "single-node enqueue_example must not run for multi-node training"
            )

        monkeypatch.setattr(node, "enqueue_example", fail_enqueue)

        # Stub the data pipeline: the production loop must run real tokenization
        # and one real step through the 1F1B extension.
        async def _fake_load(path):
            return [{"text": "hello world"}]

        async def _fake_preprocess(raw, **kw):
            return {
                "processed_samples": [{"text": "hello world"}],
                "original_count": 1, "empty_removed": 0, "short_removed": 0,
                "duplicate_removed": 0, "validation_errors": 0,
                "valid_count": 1, "format_detected": "jsonl",
            }

        monkeypatch.setattr(capi, "load_raw_data_async", _fake_load)
        monkeypatch.setattr(capi, "preprocess_async", _fake_preprocess)
        monkeypatch.setattr(
            capi, "format_sample_to_text",
            lambda sample, tokenizer: "hello world",
        )
        monkeypatch.setattr(
            capi, "build_base_shard",
            lambda model, engine_classname: Shard(
                model_id="qwen", start_layer=0, end_layer=31, n_layers=32,
            ),
        )

        api = ChatGPTAPI(node=node, inference_engine_classname="DummyEngine")
        job_id = "job-1f1b-prod"
        api._training_job = {"job_id": job_id}

        def _fake_resolve(path_str, name_only_ok=True):
            from pathlib import Path
            return Path("fake.jsonl")

        monkeypatch.setattr(api, "_resolve_dataset_path", _fake_resolve)

        job = {
            "job_id": job_id, "model": "qwen", "dataset": "fake.jsonl",
            "epochs": 1, "max_steps": 1, "batch_size": 1, "max_length": 32,
            "min_chars": 1, "max_chars": 100000, "skip_dedup": True,
            "save_every": 0, "learning_rate": "1e-5",
        }

        await api._run_training_job_distributed(job)

        assert job["status"] == "completed", (
            f"training job failed: {job.get('error')}"
        )
        assert calls.get("schedule") == "1f1b", (
            f"production training did not select 1F1B: {calls}"
        )
        assert job.get("loss") == pytest.approx(1.5)
