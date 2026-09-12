"""
Behavioral tests for Ctrl+C cancellation (no source inspection).

- wait_interruptible: done path + cancelled path kill the real proc.
- !shell: cancel event kills a real sleeping process (rc 130).
- _stream_chat: hanging server + abort -> Cancelled, fast.
- benchmark: pre-set event stops the harness child, no hang.
- abort_inflight: registered conns get closed.
"""
import argparse
import socket
import sys
import threading
import time

import pytest

from synapse import cli
from synapse import console as C


def _sleep_cmd(secs=30):
    return '%s -c "import time; time.sleep(%d)"' % (sys.executable, secs)


def test_wait_interruptible_done():
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    outcome, rc = cli.wait_interruptible(p, None)
    assert (outcome, rc) == ("done", 0)


def test_wait_interruptible_cancel_kills_real_proc():
    import subprocess
    import threading
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    ev = threading.Event()
    ev.set()
    t0 = time.monotonic()
    outcome, _ = cli.wait_interruptible(p, ev)
    assert outcome == "cancelled"
    assert time.monotonic() - t0 < 15
    assert p.poll() is not None  # really dead, no orphan


def test_run_shell_cancel_kills_real_proc():
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t0 = time.monotonic()
    rc = C.run_shell(_sleep_cmd(), timeout=60, cancel_event=ev)
    assert rc == 130
    assert time.monotonic() - t0 < 20
    assert not C._ACTIVE_PROCS  # registry cleaned, nothing leaked


def test_abort_inflight_closes_registered_conns():
    closed = []

    class _FakeConn:
        def close(self):
            closed.append(True)

    cli._track(_FakeConn())
    cli.abort_inflight()
    assert closed == [True]
    assert not cli._INFLIGHT


def test_stream_chat_cancelled_on_hanging_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def _serve():
        try:
            c, _ = srv.accept()
            time.sleep(15)  # accept, never reply
            try:
                c.close()
            except Exception:
                pass
        except Exception:
            pass

    threading.Thread(target=_serve, daemon=True).start()
    ev = threading.Event()
    outcome = {}

    def _run():
        try:
            cli._stream_chat("http://127.0.0.1:%d" % port, "m", "hi",
                             None, 30, cancel_event=ev)
            outcome["rc"] = "noraise"
        except cli.Cancelled:
            outcome["rc"] = "cancelled"
        except Exception as e:
            outcome["rc"] = "other: %r" % (e,)

    w = threading.Thread(target=_run, daemon=True)
    w.start()
    time.sleep(0.5)
    ev.set()
    cli.abort_inflight()
    w.join(timeout=15)
    srv.close()
    assert outcome.get("rc") == "cancelled", outcome


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.terminated or self.killed:
            self.returncode = -1 if self.returncode is None else self.returncode
            return self.returncode
        raise TimeoutError("still running")

    def communicate(self, timeout=None):
        return ("", "")


def test_benchmark_cancel_stops_child(monkeypatch, capsys):
    made = {}

    def _fake_popen(*a, **k):
        made["p"] = _FakeProc()
        return made["p"]

    monkeypatch.setattr(cli.subprocess, "Popen", _fake_popen)
    ev = threading.Event()
    ev.set()
    ns = argparse.Namespace(port=1, model=None, prompt=None, max_tokens=1,
                            reps=1, failover=False, json=False, cancel_event=ev)
    t0 = time.monotonic()
    assert cli.cmd_benchmark(ns) == cli.EXIT_OK
    assert time.monotonic() - t0 < 15
    assert made["p"].terminated
    assert "ngắt benchmark" in capsys.readouterr().out


def test_benchmark_pipe_output_prints_at_end(monkeypatch, capsys):
    class _QuickProc:
        def poll(self):
            return 0

        @property
        def returncode(self):
            return 0

        def communicate(self, timeout=None):
            return ("harness-line\n", "")

    captured = {}

    def _fake_popen(cmd, **kw):
        captured.update(kw)
        return _QuickProc()

    monkeypatch.setattr(cli.subprocess, "Popen", _fake_popen)
    ns = argparse.Namespace(port=1, model="m", prompt=None, max_tokens=1,
                            reps=1, failover=False, json=False,
                            cancel_event=None, pipe_output=True)
    assert cli.cmd_benchmark(ns) == cli.EXIT_OK
    assert captured.get("stdout") == cli.subprocess.PIPE  # captured, not inherited
    assert "harness-line" in capsys.readouterr().out


def test_request_cancel_sets_event_and_kills_procs(monkeypatch):
    ctx = C.ConsoleCtx(port=1, defaults_file=None)
    ctx.job_event = threading.Event()
    killed = []
    monkeypatch.setattr(cli, "abort_inflight", lambda: killed.append("http"))

    class _P:
        def terminate(self):
            killed.append("proc")

    with C._ACTIVE_PROCS_LOCK:
        C._ACTIVE_PROCS.add(_P())
    try:
        C.request_cancel(ctx)
    finally:
        with C._ACTIVE_PROCS_LOCK:
            C._ACTIVE_PROCS.clear()
    assert ctx.job_event.is_set()
    assert "http" in killed and "proc" in killed
