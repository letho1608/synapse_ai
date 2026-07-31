"""
Test: PyTorchHF engine back_gradient returns real gradients (A2).

The engine's backward phase must no longer be a no-op returning (None, None):
it computes a real gradient of the loss w.r.t. the stage input via torch
autograd (captured on the embedding output) plus per-parameter gradients,
and seeds the backward from the downstream stage's gradient when it is
shape-compatible with the model output.

Behavioral: drives the shipped PyTorchHFInferenceEngine.train() path with a
real (tiny) torch model injected — the same autograd math the engine runs
on a full Hugging Face model, without a network download.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine
from synapse.inference.shard import Shard


class _TinyModel(torch.nn.Module):
    """Minimal model exposing the HF interface the engine's backward path
    relies on: forward() returns logits and get_input_embeddings() returns
    the embedding module whose output gradient is the stage input-gradient."""

    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))

    def get_input_embeddings(self):
        return self.emb


def _make_engine():
    engine = PyTorchHFInferenceEngine()
    engine._model = _TinyModel()
    engine._model_id = "tiny"
    engine._device = "cpu"
    return engine


class TestPyTorchHFBackGradient:
    @pytest.mark.asyncio
    async def test_back_gradient_returns_real_input_gradient(self):
        engine = _make_engine()
        shard = Shard(model_id="tiny", start_layer=0, end_layer=31, n_layers=32)
        example = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        target = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int64)

        loss, grads = await engine.train(
            "req", shard, example, target, np.ones(2), loss="back_gradient",
        )

        assert loss is None  # backward phase: gradient feedback, not a loss
        assert grads is not None, "back_gradient returned no gradients"
        ig = grads.get("__input_grad__")
        assert ig is not None, "missing input-gradient in back_gradient result"
        assert ig.shape == (2, 3, 8), f"input-grad shape {ig.shape}"
        assert np.isfinite(ig).all()
        assert np.abs(ig).sum() > 0.0, "input-gradient is all zeros"
        # Real per-parameter gradients too (not just the input gradient).
        assert any(k != "__input_grad__" for k in grads), "no parameter gradients"

    @pytest.mark.asyncio
    async def test_back_gradient_seeds_from_downstream_grad(self):
        """A shape-compatible downstream gradient seeds the backward pass
        (real external-gradient backprop) instead of the local loss."""
        engine = _make_engine()
        shard = Shard(model_id="tiny", start_layer=0, end_layer=31, n_layers=32)
        example = np.array([[1, 2, 3]], dtype=np.int64)
        target = np.array([[2, 3, 4]], dtype=np.int64)
        backgrad = np.ones((1, 3, 16), dtype=np.float32)  # matches logits shape

        loss, grads = await engine.train(
            "req", shard, example, target, np.ones(1),
            backgrad=backgrad, loss="back_gradient",
        )

        assert loss is None
        assert grads is not None, "back_gradient returned no gradients"
        ig = grads["__input_grad__"]
        assert ig.shape == (1, 3, 8), f"input-grad shape {ig.shape}"
        assert np.isfinite(ig).all()
        assert np.abs(ig).sum() > 0.0, "seeded input-gradient is all zeros"

    @pytest.mark.asyncio
    async def test_back_gradient_returns_none_without_model(self):
        """No model loaded -> honest (None, None), never a fabricated value.
        Driven through the shipped _back_gradient math directly so the load
        path (which needs a real HF id) is not exercised."""
        engine = PyTorchHFInferenceEngine()
        engine._model = None
        engine._model_id = None
        engine._device = "cpu"
        shard = Shard(model_id="tiny", start_layer=0, end_layer=31, n_layers=32)

        loss, grads = await engine._back_gradient(
            "req", shard,
            np.array([[1, 2, 3]], dtype=np.int64),
            np.array([[2, 3, 4]], dtype=np.int64),
            np.ones(1), None,
        )
        assert loss is None
        assert grads is None
