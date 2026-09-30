"""
Behavioral tests: failover restores a dead node's REAL shard weights.

Before: distributed training wrote JSON metadata only, so the rebalance
handed engine.load_checkpoint() a .ckpt JSON path and every recovery was
degraded (weights never restored). Now:
- engine.save_shard_weights() persists only layers [start, end];
- the training checkpoint hook writes that file every N steps and records
  weights_path/weights_step in the JSON (round-trips through the disk loader);
- load_checkpoint(shard, weights_path) copies exactly that range into the
  survivor's loaded model (weights hash changes back to the saved one, other
  layers untouched), and refuses partial coverage / wrong model loudly.

Uses the cached tiny-random GPT-2 on CPU (local_files_only, no network).
"""
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
from synapse.inference.shard import Shard
from synapse.pipeline.fault_tolerance import FaultToleranceManager
from synapse.pipeline.node_extension import DistributedPipelineExtension

MODEL_ID = "peft-internal-testing/tiny-random-GPT2LMHeadModel"


@pytest.fixture(autouse=True)
def _no_broken_directml(monkeypatch):
    # Same env guard as test_seq_kv_parity (DirectML DLL mismatch on this box).
    import sys
    monkeypatch.setitem(sys.modules, "torch_directml", None)


def _layer_hash(engine, i):
    h = hashlib.sha256()
    for k, v in sorted(engine._get_transformer_components()["layers"][i].state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


async def _loaded(shard):
    engine = PyTorchHFInferenceEngine()
    await engine.ensure_shard(shard)  # loads the shard (encode loads only the tokenizer)
    assert engine._get_transformer_components() is not None
    return engine


def _perturb(engine, i):
    import torch
    with torch.no_grad():
        for p in engine._get_transformer_components()["layers"][i].parameters():
            p.add_(1.0)


@pytest.mark.asyncio
async def test_survivor_restores_dead_range_weights(tmp_path):
    full = Shard(MODEL_ID, 0, 4, 5)
    dead_range = Shard(MODEL_ID, 2, 4, 5)
    dead = await _loaded(full)
    _perturb(dead, 3)  # "trained" state that only the dead node had
    saved = {i: _layer_hash(dead, i) for i in range(5)}
    path = str(tmp_path / ("dead_L2-4.weights.pt"))
    assert dead.save_shard_weights(dead_range, path)

    survivor = await _loaded(full)
    before0 = _layer_hash(survivor, 0)
    assert _layer_hash(survivor, 3) != saved[3]

    assert await survivor.load_checkpoint(dead_range, path) is True
    for i in (2, 3, 4):
        assert _layer_hash(survivor, i) == saved[i], i
    assert _layer_hash(survivor, 0) == before0, "layers outside the range untouched"

    # Restored model computes the same logits as the dead node's.
    ids = np.asarray(await dead.encode(full, "hello world"), dtype=np.int64).reshape(-1)
    a, _ = await dead.infer_tensor("r-dead", full, ids)
    b, _ = await survivor.infer_tensor("r-surv", full, ids)
    assert np.allclose(a, b)


@pytest.mark.asyncio
async def test_restore_refuses_partial_or_mismatch(tmp_path):
    full = Shard(MODEL_ID, 0, 4, 5)
    engine = await _loaded(full)
    path = str(tmp_path / "part.weights.pt")
    assert engine.save_shard_weights(Shard(MODEL_ID, 3, 4, 5), path)
    # Asking for 2-4 when only 3-4 were saved: refused, nothing applied.
    assert await engine.load_checkpoint(Shard(MODEL_ID, 2, 4, 5), path) is False
    # Different model id: refused.
    assert await engine.load_checkpoint(Shard("other/model", 3, 4, 5), path) is False
    # Metadata-only JSON path stays a degraded (False) restore.
    ckpt = tmp_path / "x.ckpt"
    ckpt.write_text("{}")
    assert await engine.load_checkpoint(full, str(ckpt)) is False


@pytest.mark.asyncio
async def test_training_hook_records_weights_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPSE_CKPT_WEIGHTS_EVERY", "3")
    full = Shard(MODEL_ID, 0, 4, 5)
    engine = await _loaded(full)
    ft = FaultToleranceManager("node-a", checkpoint_dir=str(tmp_path))
    node = SimpleNamespace(
        id="node-a", inference_engine=engine, fault_tolerance=ft,
        _lazy_init_fault_tolerance=lambda: None,
        get_current_shard=lambda base: Shard(MODEL_ID, 1, 2, 5),
    )
    ext = DistributedPipelineExtension.__new__(DistributedPipelineExtension)
    ext._node = node
    ext._checkpoint_steps = {}
    ext._weights_saved = {}

    written = []
    for step in range(1, 5):
        ext._save_shard_checkpoint(full)
        ck = FaultToleranceManager("node-b", checkpoint_dir=str(tmp_path)) \
            .load_latest_checkpoint("node-a")
        written.append((ck.step, ck.weights_step))
        assert ck.weights_path and os.path.isfile(ck.weights_path)
    # Weights written at step 1 and step 3; steps 2/4 point at the newest.
    assert written == [(1, 1), (2, 1), (3, 3), (4, 3)]

    # A survivor restores that range from the recorded path.
    survivor = await _loaded(full)
    _perturb(survivor, 1)
    assert await survivor.load_checkpoint(Shard(MODEL_ID, 1, 2, 5), ck.weights_path)
    assert _layer_hash(survivor, 1) == _layer_hash(engine, 1)


def test_hf_repo_id_checkpoint_roundtrip(tmp_path):
    """The default model id contains '/'; checkpoints used to fail to write
    (path pointed into a nonexistent subdirectory) for every training step."""
    ft = FaultToleranceManager("n1", checkpoint_dir=str(tmp_path))
    path = ft.save_checkpoint("HuggingFaceTB/SmolLM2-135M-Instruct", 0, 3, step=7)
    assert os.path.dirname(path) == str(tmp_path) and os.path.isfile(path)
    other = FaultToleranceManager("n2", checkpoint_dir=str(tmp_path))
    ck = other.load_checkpoint("HuggingFaceTB/SmolLM2-135M-Instruct", 0, 3, 7)
    assert ck is not None and ck.step == 7
    # Node "n1" must not pick up node "n1_x"'s files via the shared prefix.
    FaultToleranceManager("n1_x", checkpoint_dir=str(tmp_path)).save_checkpoint("m", 0, 1, step=99)
    assert other.load_latest_checkpoint("n1").step == 7
