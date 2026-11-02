"""The live-cluster harness refuses loopback runs: N processes on one
machine must not pass for an N-machine measurement (C1 gate)."""
import os
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(__file__), "benchmark_live_cluster.py")


def _run(*args):
    return subprocess.run([sys.executable, HARNESS, *args], capture_output=True, text=True, timeout=120)


def test_loopback_only_run_is_refused():
    r = _run("--api-urls", "http://127.0.0.1:52415,http://localhost:52416", "--model", "m", "--out", os.devnull)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "loopback" in (r.stdout + r.stderr).lower()
