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
def _console_ready_immediately(request, monkeypatch):
    """Console /benchmark + /client wait for backend readiness in prod.

    Tests must never block on that (default 120s): pretend ready, unless
    the test is marked @pytest.mark.real_wait (its own patch then wins).
    """
    if request.node.get_closest_marker("real_wait"):
        return
    try:
        from synapse import console as _C
    except Exception:
        return
    monkeypatch.setattr(_C, "_wait_ready", lambda *a, **k: True)
