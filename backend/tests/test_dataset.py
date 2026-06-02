"""
Unit tests: SimpleDataset (synapse.data.simple_dataset), load_dataset (synapse.train.dataset).
Chay: pytest tests/test_dataset.py -v
"""
import sys
import os
import tempfile
import json
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_simple_dataset_jsonl():
    """SimpleDataset load JSONL file."""
    try:
        from synapse.data.simple_dataset import SimpleDataset
    except ImportError:
        import pytest
        pytest.skip("synapse.data.simple_dataset not available (no synapse.data package)")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write('{"text": "hello"}\n')
        f.write('{"text": "world"}\n')
        path = f.name
    try:
        ds = SimpleDataset(path)
        assert len(ds) == 2
        assert ds[0]["text"] == "hello"
        assert ds[1]["text"] == "world"
    finally:
        os.unlink(path)
    print("  [OK] test_simple_dataset_jsonl")


def test_simple_dataset_json():
    """SimpleDataset load JSON array file."""
    try:
        from synapse.data.simple_dataset import SimpleDataset
    except ImportError:
        import pytest
        pytest.skip("synapse.data.simple_dataset not available")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([{"id": 1}, {"id": 2}], f)
        path = f.name
    try:
        ds = SimpleDataset(path)
        assert len(ds) == 2
        assert ds[0]["id"] == 1
    finally:
        os.unlink(path)
    print("  [OK] test_simple_dataset_json")


def test_simple_dataset_not_found():
    """SimpleDataset file khong ton tai -> FileNotFoundError."""
    try:
        from synapse.data.simple_dataset import SimpleDataset
    except ImportError:
        import pytest
        pytest.skip("synapse.data.simple_dataset not available")
    import pytest
    with pytest.raises(FileNotFoundError):
        SimpleDataset("/nonexistent/file.jsonl")
    print("  [OK] test_simple_dataset_not_found")


def test_train_dataset_load():
    """synapse.train.dataset.load_dataset voi thu muc co train/valid/test.jsonl."""
    from synapse.train.dataset import load_dataset, Dataset
    data_dir = os.path.join(ROOT, "synapse", "train", "data", "lora")
    if not os.path.isdir(data_dir):
        import pytest
        pytest.skip("synapse/train/data/lora not found")
    train, valid, test = load_dataset(data_dir)
    assert train is not None and len(train) >= 0
    assert valid is not None
    assert test is not None
    if len(train) > 0:
        sample = train[0]
        assert isinstance(sample, dict) or hasattr(sample, "__getitem__")
    print("  [OK] test_train_dataset_load")


def test_train_dataset_batch_with_lengths():
    """batch_with_lengths, compose."""
    from synapse.train.dataset import batch_with_lengths
    import numpy as np
    tokens = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    x, y, lengths = batch_with_lengths(tokens)
    assert x.shape[0] == 3
    assert lengths.shape[0] == 3
    print("  [OK] test_train_dataset_batch_with_lengths")
