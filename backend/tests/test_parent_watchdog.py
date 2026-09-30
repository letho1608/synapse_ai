"""
Behavioral tests: the backend never outlives a killed launcher.

main.py / `serve` / `console` spawn backend/synapse/main.py as a child. A hard
kill of the launcher (TerminateProcess, SIGKILL) runs no cleanup, and the
orphaned backend kept holding its API/node ports (seen as a flaky
test_system::test_api_full and a zombie node for users). Launchers now pass
SYNAPSE_PARENT_PID and the backend exits when that process is gone.
"""
import os
import subprocess
import sys
import threading
import time

import psutil

from synapse.helpers import PARENT_PID_ENV, start_parent_watchdog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_watchdog_fires_when_parent_dies(monkeypatch):
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setenv(PARENT_PID_ENV, str(parent.pid))
        fired = threading.Event()
        t = start_parent_watchdog(interval_s=0.05, _exit=lambda code: fired.set(),
                                  _own_start=time.time() + 60)
        assert t is not None
        time.sleep(0.3)
        assert not fired.is_set(), "must not fire while the parent lives"
        parent.kill()
        parent.wait()
        assert fired.wait(5), "watchdog did not notice the dead parent"
    finally:
        if parent.poll() is None:
            parent.kill()


def test_watchdog_inert_without_valid_parent(monkeypatch):
    monkeypatch.delenv(PARENT_PID_ENV, raising=False)
    assert start_parent_watchdog(_exit=lambda c: None) is None
    monkeypatch.setenv(PARENT_PID_ENV, "not-a-pid")
    assert start_parent_watchdog(_exit=lambda c: None) is None


def test_backend_exits_after_launcher_hard_kill():
    port = 59731
    env = dict(os.environ)
    env.pop(PARENT_PID_ENV, None)
    launcher = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "main.py"), "--chatgpt-api-port", str(port),
         "--disable-tui"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    backend = None
    try:
        deadline = time.time() + 60
        while time.time() < deadline and backend is None:
            kids = psutil.Process(launcher.pid).children()
            backend = next((k for k in kids if "synapse" in " ".join(k.cmdline())), None)
            time.sleep(0.2)
        assert backend is not None, "launcher never spawned the backend"
        launcher.kill()  # hard kill: no finally/atexit in the launcher runs
        launcher.wait()
        gone, alive = psutil.wait_procs([backend], timeout=20)
        assert backend in gone, "backend outlived its killed launcher (orphan)"
    finally:
        if launcher.poll() is None:
            launcher.kill()
        if backend is not None and backend.is_running():
            backend.kill()


def test_watchdog_exits_if_parent_already_gone(monkeypatch):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    monkeypatch.setenv(PARENT_PID_ENV, str(dead.pid))
    fired = threading.Event()
    assert start_parent_watchdog(interval_s=0.05, _exit=lambda c: fired.set(),
                                 _own_start=time.time() + 60) is not None
    assert fired.wait(5)


def test_recycled_pid_is_not_the_launcher(monkeypatch):
    """A live process with the launcher's PID but created after us is a
    recycled PID -> treated as the launcher being gone."""
    young = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setenv(PARENT_PID_ENV, str(young.pid))
        fired = threading.Event()
        start_parent_watchdog(interval_s=0.05, _exit=lambda c: fired.set(),
                              _own_start=time.time() - 3600)
        assert fired.wait(5)
    finally:
        young.kill()
