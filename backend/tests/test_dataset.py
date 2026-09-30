"""
Unit tests: load_dataset / batching (synapse.train.dataset).
Chay: pytest tests/test_dataset.py -v
"""
import sys
import os
import tempfile
import json
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


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
