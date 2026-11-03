"""
Pytest fixtures: ROOT, TEST_PORT, session.
Chay: pytest tests/ -v
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TEST_PORT = 59999


import pytest


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path_factory, monkeypatch):
    """Fault-tolerance checkpoints go to a per-test temp dir, never into the
    user's home (they used to land in the HF hub cache root)."""
    if "SYNAPSE_CHECKPOINT_DIR" not in os.environ:
        monkeypatch.setenv("SYNAPSE_CHECKPOINT_DIR", str(tmp_path_factory.mktemp("ckpt")))
