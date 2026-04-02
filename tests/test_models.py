"""
Unit tests: Shard, model registry, build_base_shard, build_full_shard, get_repo.
Chay: pytest tests/test_models.py -v
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_shard_basic():
    """Shard: model_id, layers, to_dict."""
    from synapse.inference.shard import Shard
    s = Shard("qwen2.5:1.5b", 0, 27, 28)
    assert s.model_id == "qwen2.5:1.5b"
    assert s.start_layer == 0
    assert s.end_layer == 27
    assert s.n_layers == 28
    assert s.get_layer_count() == 28
    d = s.to_dict()
    assert d["model_id"] == "qwen2.5:1.5b" and d["n_layers"] == 28
    print("  [OK] test_shard_basic")


def test_shard_first_last():
    """Shard: is_first_layer, is_last_layer."""
    from synapse.inference.shard import Shard
    s = Shard("m", 0, 9, 10)
    assert s.is_first_layer() is True
    assert s.is_last_layer() is True
    s2 = Shard("m", 2, 5, 10)
    assert s2.is_first_layer() is False
    assert s2.is_last_layer() is False
    print("  [OK] test_shard_first_last")


def test_shard_overlap():
    """Shard: overlaps."""
    from synapse.inference.shard import Shard, shards_overlap
    a = Shard("m", 0, 5, 10)
    b = Shard("m", 4, 9, 10)
    assert a.overlaps(b) is True
    c = Shard("m", 6, 9, 10)
    assert a.overlaps(c) is False
    d = Shard("other", 0, 5, 10)
    assert a.overlaps(d) is False
    assert shards_overlap(a, b) is True
    print("  [OK] test_shard_overlap")


def test_shard_from_dict():
    """Shard.from_dict."""
    from synapse.inference.shard import Shard
    d = {"model_id": "x", "start_layer": 0, "end_layer": 11, "n_layers": 12}
    s = Shard.from_dict(d)
    assert s.model_id == "x" and s.n_layers == 12
    print("  [OK] test_shard_from_dict")


def test_build_base_and_full_shard():
    """build_base_shard, build_full_shard sau khi dang ky PyTorch models."""
    from synapse.inference.inference_engine import get_inference_engine
    from synapse.models import build_base_shard, build_full_shard, get_repo, list_models
    get_inference_engine("pytorch", None)
    base = build_base_shard("qwen2.5:1.5b", "PyTorchHFInferenceEngine")
    assert base is not None
    assert base.start_layer == 0 and base.n_layers == 28
    full = build_full_shard("qwen2.5:1.5b", "PyTorchHFInferenceEngine")
    assert full is not None
    assert full.end_layer == full.n_layers - 1
    assert get_repo("qwen2.5:1.5b", "PyTorchHFInferenceEngine") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "qwen2.5:1.5b" in list_models()
    assert build_base_shard("__none__", "PyTorchHFInferenceEngine") is None
    assert build_full_shard("__none__", "PyTorchHFInferenceEngine") is None
    print("  [OK] test_build_base_and_full_shard")


def test_get_model_info():
    """get_model_info, get_pretty_name (sau khi register)."""
    from synapse.inference.inference_engine import get_inference_engine
    from synapse.models import get_model_info, get_pretty_name
    get_inference_engine("pytorch", None)
    info = get_model_info("qwen2.5:1.5b")
    assert info is not None
    assert "layers" in info and info["layers"] == 28
    assert get_pretty_name("qwen2.5:1.5b") is None or isinstance(get_pretty_name("qwen2.5:1.5b"), str)
    assert get_model_info("__none__") is None
    assert get_pretty_name("__none__") is None
    print("  [OK] test_get_model_info")
