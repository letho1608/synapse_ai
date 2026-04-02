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


def pytest_configure(config):
    """Pytest hook."""
    pass
