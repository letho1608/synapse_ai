"""
Behavioral tests: pipeline-parallel training where every node trains only
the layers it serves.

Found 2026-11-02: every training step called _ensure_full() (the WHOLE model
on every node) and the "1F1B" path ran all stages inside one process, each
re-running the full forward; no activation or gradient ever crossed a node
boundary. Now each node keeps its layer range, activations go forward and
d loss / d input goes back between nodes (pp_training.PipelineTrainer over
an injected send), and every stage steps its own parameters.

The gate: one pipeline step equals one step of the full HF model (same
micro-batch averaging, same global-norm clipping, same AdamW), parameter by
parameter, and a tied embedding split over the first and last stage stays
identical in both copies.
"""
import asyncio
import json
import sys

import numpy as np
import pytest
import torch

TINY = ("peft-internal-testing/tiny-random-GPT2LMHeadModel", 5)
SMOL = ("HuggingFaceTB/SmolLM2-135M-Instruct", 30)


def _cached(model_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


MODELS = [pytest.param(TINY, id="tiny-gpt2"),
          pytest.param(SMOL, id="smollm2",
                       marks=pytest.mark.skipif(not _cached(SMOL[0]), reason="SmolLM2 not cached"))]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SYNAPSE_PARTIAL_LOAD", "1")
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "none")


def _engine(deterministic=True):
    from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
    e = PyTorchHFInferenceEngine()
    if deterministic:
        e._stage_modules = lambda shard: []  # keep eval mode: no dropout, exact compare
    return e


def _cuts(n, k):
    b = np.linspace(0, n, k + 1).astype(int)
    return [(int(b[i]), int(b[i + 1]) - 1) for i in range(k)]


class _Cluster:
    """N nodes in one process; messages are JSON-round-tripped like the wire."""

    def __init__(self, engines, timeout=120.0):
        from synapse.pipeline.pp_training import PipelineTrainer
        self.trainers = {}
        self.sent = []
        for i, eng in enumerate(engines):
            nid = f"node-{i}"
            self.trainers[nid] = PipelineTrainer(nid, eng, self._send_from(nid), timeout=timeout)

    def _send_from(self, src):
        async def send(target, tensor, meta):
            self.sent.append((src, target, meta["op"]))
            arr = np.array(tensor, copy=True)
            m = json.loads(json.dumps(meta))
            asyncio.get_running_loop().create_task(self.trainers[target].on_message(arr, m))
        return send


def _reference_step(model_id, ids, labels, m, lr, max_norm=1.0):
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                               attn_implementation="eager").eval()
    losses = []
    for idx in np.array_split(np.arange(ids.shape[0]), m):
        x = torch.from_numpy(ids[idx])
        y = torch.from_numpy(labels[idx])
        lg = ref(x).logits
        loss = torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, lg.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        losses.append(float(loss))
    for p in ref.parameters():
        if p.grad is not None:
            p.grad.mul_(1.0 / m)
    torch.nn.utils.clip_grad_norm_(ref.parameters(), max_norm)
    opt = torch.optim.AdamW(ref.parameters(), lr=lr, weight_decay=0.01)
    opt.step()
    return ref, float(np.mean(losses))


def _ref_name(ref, name):
    """Map a stage parameter name to the full model's parameter."""
    params = dict(ref.named_parameters())
    if name.startswith("layers."):
        rest = name[len("layers."):]
        for prefix in ("model.layers.", "transformer.h."):
            if prefix + rest in params:
                return params[prefix + rest]
    if name.startswith("embed."):
        return ref.get_input_embeddings().weight
    if name.startswith("pos_embed."):
        return ref.transformer.wpe.weight
    if name.startswith("head."):
        return ref.get_output_embeddings().weight
    if name.startswith("norm."):
        for k in ("model.norm.", "transformer.ln_f."):
            if k + name[len("norm."):] in params:
                return params[k + name[len("norm."):]]
    raise KeyError(name)


def _batch(model_id, rows=4, length=12):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    text = "The quick brown fox jumps over the lazy dog while the cat sleeps on the warm mat"
    base = tok.encode(text)[:length]
    ids = np.array([np.roll(base, r) for r in range(rows)], dtype=np.int64)
    labels = ids.copy()
    labels[1, -3:] = -100  # padding positions are ignored
    return ids, labels


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MODELS)
async def test_pipeline_step_equals_full_model_step(spec):
    from synapse.inference.shard import Shard
    model_id, n = spec
    ids, labels = _batch(model_id)
    lr, m = 1e-3, 2
    engines = [_engine() for _ in range(3)]
    plan = [(f"node-{i}", Shard(model_id, a, b, n)) for i, (a, b) in enumerate(_cuts(n, 3))]
    cluster = _Cluster(engines)

    loss = await cluster.trainers["node-0"].train_step(plan, ids, labels, lr=lr, num_micro_batches=m)
    ref, ref_loss = _reference_step(model_id, ids, labels, m, lr)

    assert loss == pytest.approx(ref_loss, rel=1e-5)
    worst, compared = 0.0, 0
    for eng, (_, shard) in zip(engines, plan):
        assert eng._model is None, "a stage loaded the whole model"
        assert eng.loaded_layers() == list(range(shard.start_layer, shard.end_layer + 1))
        for name, p in eng._stage_params(shard):
            r = _ref_name(ref, name).detach()
            worst = max(worst, float((p.detach() - r).abs().max()))
            compared += 1
    assert compared > 10
    assert worst < 1e-5, f"pipeline step diverged from the full-model step by {worst}"
    # Activations and gradients really crossed between the nodes.
    hops = {(s, t, op) for s, t, op in cluster.sent}
    assert ("node-0", "node-1", "fwd") in hops and ("node-1", "node-2", "fwd") in hops
    assert ("node-2", "node-1", "bwd") in hops and ("node-1", "node-0", "bwd") in hops


@pytest.mark.asyncio
@pytest.mark.skipif(not _cached(SMOL[0]), reason="SmolLM2 not cached")
async def test_tied_embedding_copies_stay_identical_and_loss_drops():
    """SmolLM2 ties its head to the input embedding. Split over two nodes,
    each holds a copy; after several steps both copies are still equal and
    the training loss went down."""
    from synapse.inference.shard import Shard
    model_id, n = SMOL
    ids, labels = _batch(model_id)
    engines = [_engine(), _engine()]
    plan = [("node-0", Shard(model_id, 0, 14, n)), ("node-1", Shard(model_id, 15, 29, n))]
    cluster = _Cluster(engines)
    losses =[await cluster.trainers["node-0"].train_step(plan, ids, labels, lr=5e-4)
              for _ in range(4)]
    emb = engines[0]._pm.embed.weight
    head = engines[1]._pm.lm_head.weight
    assert emb.data_ptr() != head.data_ptr(), "the two stages must hold separate copies"
    assert torch.equal(emb, head), "tied embedding copies drifted apart"
    assert losses[-1] < losses[0], losses


@pytest.mark.asyncio
async def test_stage_error_fails_the_step_fast_and_drops_graphs():
    from synapse.inference.shard import Shard
    model_id, n = TINY
    ids, labels = _batch(model_id, rows=2)
    engines = [_engine(), _engine()]
    plan = [("node-0", Shard(model_id, 0, 1, n)), ("node-1", Shard(model_id, 2, 4, n))]

    async def broken(*a, **k):
        raise RuntimeError("disk on fire")
    engines[1].stage_forward = broken
    cluster = _Cluster(engines, timeout=30.0)
    with pytest.raises(RuntimeError, match="disk on fire"):
        await cluster.trainers["node-0"].train_step(plan, ids, labels)
    await asyncio.sleep(0.05)  # let the drop messages land
    assert engines[0]._pp_graphs == {}, "kept graphs of a failed step"


@pytest.mark.asyncio
async def test_single_node_train_uses_served_layers_not_a_full_copy():
    from synapse.inference.shard import Shard
    model_id, n = TINY
    ids, labels = _batch(model_id, rows=2)
    eng = _engine()
    shard = Shard(model_id, 0, n - 1, n)
    loss, grads = await eng.train("r", shard, ids, labels, np.ones(2))
    assert loss is not None and grads is None
    assert eng._model is None and eng._pm is not None, "train() loaded a second, full model"
    # A partial shard must go through the pipeline.
    with pytest.raises(RuntimeError, match="PipelineTrainer"):
        await _engine().train("r", Shard(model_id, 0, 1, n), ids, labels, np.ones(2))


@pytest.mark.asyncio
async def test_quantized_weights_refuse_training(monkeypatch):
    from synapse.inference.shard import Shard
    monkeypatch.setenv("SYNAPSE_WEIGHT_QUANT", "int8")
    model_id, n = TINY
    with pytest.raises(RuntimeError, match="quantized"):
        await _engine().stage_train_prepare(Shard(model_id, 0, n - 1, n))


@pytest.mark.asyncio
async def test_two_nodes_train_through_their_real_message_paths(tmp_path, monkeypatch):
    """Production wiring: train_step_distributed(schedule="1f1b") on a real
    Node plans over the topology, each node trains only its own range, and
    every hop is encoded with the tensor codec and delivered through the
    receiving node's _on_transport_tensor (only the socket is faked). Each
    stage checkpoints its layers after the step."""
    from types import SimpleNamespace
    from synapse.inference.shard import Shard
    from synapse.orchestration.node import Node
    from synapse.routing import EventRouter
    from synapse.routing.libp2p_node import Libp2pNode
    from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
    from synapse.topology.topology import Topology
    from synapse.transport.codec import decode_tensor

    monkeypatch.setenv("SYNAPSE_CKPT_REPLICATE", "0")
    model_id, n = TINY
    caps = DeviceCapabilities(model="t", chip="t", memory=4096,
                              flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
                              cpu_cores=4, disk_gb=100, system_ram_mb=8192)
    nodes = {}
    for nid in ("node-a", "node-b"):
        router = EventRouter()
        node = Node(_id=nid, event_router=router, libp2p_node=Libp2pNode(nid, 0, router),
                    inference_engine=_engine(deterministic=False), shard_downloader=None)
        node.tflops_proportional = True
        topo = Topology()
        topo.update_node("node-a", caps)
        topo.update_node("node-b", caps)
        node.topology = topo
        node._lazy_init_fault_tolerance()
        node.fault_tolerance.checkpoint_dir = str(tmp_path / nid)
        (tmp_path / nid).mkdir()
        nodes[nid] = node

    hops = []

    def wire(src):
        async def send_blob(target, blob):
            hops.append((src, target, decode_tensor(blob)[1]["op"]))
            tensor, meta = decode_tensor(blob)
            await nodes[target]._on_transport_tensor(SimpleNamespace(tensor=tensor, metadata=meta), "tcp")
            return True
        return send_blob

    for nid, node in nodes.items():
        node._send_blob_tcp = wire(nid)

    base = Shard(model_id, 0, n - 1, n)
    ids, labels = _batch(model_id, rows=2)
    ext = nodes["node-a"].distributed_pipeline
    plan = ext._get_partition_assignments(base)
    assert [nid for nid, _ in plan] == ["node-a", "node-b"]
    loss = await ext.train_step_distributed(base, ids, labels, schedule="1f1b", learning_rate=1e-3)

    assert np.isfinite(loss) and loss > 0
    for nid, shard in plan:
        eng = nodes[nid].inference_engine
        assert eng._model is None
        assert eng.loaded_layers() == list(range(shard.start_layer, shard.end_layer + 1))
    ops = {(s, t, op) for s, t, op in hops}
    assert ("node-a", "node-b", "fwd") in ops and ("node-b", "node-a", "bwd") in ops
    assert ("node-b", "node-a", "stepped") in ops
    # Both stages wrote a checkpoint for their own range.
    for nid in nodes:
        assert any(p.suffix == ".json" for p in (tmp_path / nid).iterdir()), nid
