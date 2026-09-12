"""
Behavioral tests: interactive console shell (interface only, no agent).

- parse_slash: blank/slash/quoted/unknown/bad-quote.
- dispatch routes to the REAL command functions (monkeypatched to record,
  never network): status/topology/models/benchmark/client/help/quit.
- run_console boots a FAKE backend subprocess, streams its logs prefixed
  as [node], serves /help + /quit from scripted input, stops cleanly.
- Plain text without slash does nothing (no guessing, no model calls).
"""
import argparse
import io
from contextlib import redirect_stdout

import pytest

from synapse import console as C


def test_parse_slash():
    assert C.parse_slash("") == (None, None)
    assert C.parse_slash("   ") == (None, None)
    assert C.parse_slash("/status") == ("status", [])
    assert C.parse_slash("/status --json --port 5555") == ("status", ["--json", "--port", "5555"])
    assert C.parse_slash("/QUIT") == ("quit", [])
    assert C.parse_slash("/client hello world") == ("client", ["hello", "world"])
    assert C.parse_slash('/client "hello world"') == ("client", ["hello world"])
    assert C.parse_slash("hello?") == ("unknown", "hello?")
    assert C.parse_slash("/unclosed \"x") == ("unknown", '/unclosed "x')


def test_dispatch_routes_and_quits(monkeypatch):
    calls = []

    def rec(name):
        def _fn(ns):
            calls.append((name, ns))
            return 0
        return _fn

    monkeypatch.setattr(C, "cmd_status", rec("status"))
    monkeypatch.setattr(C, "cmd_topology", rec("topology"))
    monkeypatch.setattr(C, "cmd_models", rec("models"))
    monkeypatch.setattr(C, "cmd_benchmark", rec("benchmark"))
    monkeypatch.setattr(C, "cmd_client", rec("client"))
    monkeypatch.setattr(C, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(C, "_wait_ready", lambda *a, **k: True)
    d = {"port": 52415}
    assert C.dispatch("status", [], d) == (True, 0)
    assert C.dispatch("topology", ["--json"], d) == (True, 0)
    assert C.dispatch("models", ["pull", "m1"], d) == (True, 0)
    assert C.dispatch("benchmark", ["--reps", "2"], d) == (True, 0)
    assert C.dispatch("client", ["hi", "there"], d) == (True, 0)
    assert C.dispatch("help", [], d) == (True, 0)
    assert C.dispatch("quit", [], d) == (False, 0)
    assert C.dispatch("exit", [], d) == (False, 0)
    assert C.dispatch("frobnicate", [], d) == (True, 2)  # unknown: stays, no action
    names = [c[0] for c in calls]
    assert names == ["status", "topology", "models", "benchmark", "client"]
    status_ns = calls[0][1]
    assert status_ns.port == 52415 and status_ns.json is False
    topo_ns = calls[1][1]
    assert topo_ns.json is True
    models_ns = calls[2][1]
    assert (models_ns.action, models_ns.model_id) == ("pull", "m1")
    bench_ns = calls[3][1]
    assert bench_ns.reps == 2
    assert bench_ns.model is None and bench_ns.pipe_output is False
    client_ns = calls[4][1]
    assert client_ns.prompt == "hi there"


def test_dispatch_models_validates_action(capsys):
    assert C.dispatch("models", ["frobnicate"], {"port": 1}) == (True, 2)
    assert "usage: /models" in capsys.readouterr().out


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


class _FakeProc:
    def __init__(self, lines):
        self.stdout = _FakeStdout(lines)
        self.pid = 1234
        self.terminated = False
        self.waited = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0

    @property
    def returncode(self):
        return 0


def _serve_ns(**kw):
    base = dict(model="m1", node_id=None, node_port=None, discovery_module=None,
                config=None, join=None, wait_for_peers=None, partition_method=None,
                ilp_time_limit_seconds=None, temperature=None, chat_port=None,
                tui=False, no_tui=True, headless=False, log_format="text")
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_console_boots_streams_and_quits(monkeypatch, capsys):
    procs = []

    def _fake_popen(cmd, **kw):
        assert cmd[0].endswith("python") or "python" in cmd[0]
        assert "main.py" in cmd[1]
        p = _FakeProc(["[test] backend booted\n"])
        procs.append((cmd, kw, p))
        return p

    monkeypatch.setattr(C.subprocess, "Popen", _fake_popen)
    inputs = iter(["/help", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    code = C.run_console(_serve_ns())
    assert code == 0
    out = capsys.readouterr().out
    assert "[node] [test] backend booted" in out
    assert "/status" in out  # help text shown
    assert procs and procs[0][2].terminated and procs[0][2].waited


def test_run_console_plain_text_chats(monkeypatch, capsys):
    def _fake_popen(cmd, **kw):
        return _FakeProc([])

    monkeypatch.setattr(C.subprocess, "Popen", _fake_popen)
    calls = []
    monkeypatch.setattr(C, "cmd_client",
                        lambda ns: calls.append(ns.prompt) or 0)
    inputs = iter(["hello can you hear me?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    code = C.run_console(_serve_ns())
    assert code == 0
    assert calls == ["hello can you hear me?"]  # plain text -> chat
    capsys.readouterr()


def test_run_console_backend_start_failure(monkeypatch, capsys):
    def _boom(cmd, **kw):
        raise OSError("no python here")

    monkeypatch.setattr(C.subprocess, "Popen", _boom)
    code = C.run_console(_serve_ns())
    assert code == 1
    assert "cannot start backend" in capsys.readouterr().err


def test_console_main_parses_flags(monkeypatch):
    import synapse.cli as cli
    called = {}

    def _fake_run(args):
        called.update(vars(args))
        return 7

    monkeypatch.setattr(C, "run_console", _fake_run)
    code = C.main(["--model", "m1", "--chat-port", "5555", "--headless"])
    assert code == 7
    assert called["model"] == "m1" and called["chat_port"] == 5555
    assert called["headless"] is True and called["no_tui"] is True
    # cli + root dispatch reach the console entry
    seen = {}

    def _fake_console(argv):
        seen["argv"] = argv
        return 3

    monkeypatch.setattr("synapse.console.main", _fake_console)
    assert cli.main(["console", "--model", "m1"]) == 3
    assert seen["argv"] == ["--model", "m1"]


def test_dispatch_set_config_unset(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(C, "confirm", lambda *a, **k: True)
    fp = str(tmp_path / "d.json")
    d = {"port": 1, "defaults_file": fp}
    assert C.dispatch("set", ["model", "m1"], d) == (True, 0)
    assert C.dispatch("set", ["node-port", "abc"], d) == (True, 1)
    assert C.dispatch("set", ["nope", "x"], d) == (True, 1)
    assert C.dispatch("set", ["model"], d) == (True, 2)  # usage, no crash
    assert C.dispatch("config", [], d) == (True, 0)
    out = capsys.readouterr().out
    assert "[ok] 'model' = 'm1'" in out
    assert "[fail]" in out
    assert "model" in out and "[file]" in out and "[builtin]" in out
    assert C.dispatch("unset", ["model"], d) == (True, 0)
    assert C.dispatch("unset", [], d) == (True, 2)  # usage, no crash
    assert C.dispatch("unset", ["nope"], d) == (True, 1)
    out2 = capsys.readouterr().out
    assert "cleared" in out2
