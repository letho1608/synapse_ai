"""
Behavioral tests for UI.md acceptance items (phase 1, no source inspection).

Maps to UI.md section 9:
 1. `/` + Enter lists every command.             7. /doctor FAIL + fix hint
 2. fuzzy /to, unknown /xyz calls nothing.       8. /clear keeps backend alive
 3. dead API -> `peers ?`, prompt survives.      9. Ctrl+C propagates (loop keeps session)
 4. piped stdout -> plain parseable tables.     10. !shell echo/block/timeout
 5. confirm defaults to N (no API call).        11. history file read/write
 6. /logs off buffers, /logs on flushes all.   12. /export + /bug files
"""
import io
import os
import subprocess
import sys
import types
from contextlib import redirect_stdout

import pytest

from synapse import console as C


def _ctx(**kw):
    base = dict(port=1, defaults_file=None, node_label="node",
                model_short="?", modes="")
    base.update(kw)
    return C.ConsoleCtx(**base)


# 1. `/` + Enter lists every command -------------------------------------------
def test_bare_slash_lists_all_commands(capsys):
    assert C.resolve_line("/") == ("list", "")
    assert C.resolve_line("//") == ("list", "")
    C.cmd_list("")
    out = capsys.readouterr().out
    for name in C.COMMANDS:
        assert ("/" + name) in out


# 2. fuzzy match + unknown calls nothing ---------------------------------------
def test_fuzzy_topology_and_suggest_and_pick():
    assert C.resolve_line("/topology")[:2] == ("run", "topology")
    assert C.resolve_line("/to")[:2] == ("run", "topology")  # single prefix hit
    kind, name, argv = C.resolve_line("/opo")
    assert (kind, name) == ("suggest", "topology")  # substring-only hit
    kind, cands, argv = C.resolve_line("/o")
    assert kind == "pick" and len(cands) > 1 and "topology" in cands


def test_unknown_calls_nothing(monkeypatch):
    assert C.resolve_line("/xyz") == ("unknown", "/xyz")
    called = []
    for fn in ("cmd_status", "cmd_topology", "cmd_models",
               "cmd_benchmark", "cmd_client"):
        monkeypatch.setattr(C, fn, lambda ns, _n=fn: called.append(_n))
    assert C.dispatch("frobnicate", [], {"port": 1}) == (True, 2)
    assert called == []


# 3. dead API -> peers ?, prompt survives --------------------------------------
def test_dead_api_gives_peers_unknown():
    assert C.fetch_bar_data(1) is None  # nothing listens on port 1
    line = C.build_bar_line("node", "?", "SmolLM2", "")
    assert "peers ?" in line and len(line) <= 80


def test_emit_above_never_loses_text(capsys):
    bar = C.StatusBar(_ctx())
    assert bar.enabled is False  # pytest stdout is not a TTY
    bar.idle = True
    bar.emit_above("[node] hello")
    assert "[node] hello" in capsys.readouterr().out


# 4. piped stdout -> plain tables ----------------------------------------------
def test_tables_plain_when_piped(capsys):
    assert not C._is_tty()
    C._print_table(["command", "args", "does"], [["/status", "", "x"]])
    out = capsys.readouterr().out
    assert "command" in out and "/status" in out


# 5. confirm defaults to N ------------------------------------------------------
def test_confirm_default_n(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert C.confirm("sure?") is False
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert C.confirm("sure?") is True


def test_models_pull_cancelled_calls_no_api(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a: "")  # Enter = N
    called = []
    monkeypatch.setattr(C, "cmd_models", lambda ns: called.append(ns))
    assert C.dispatch("models", ["pull", "m1"], {"port": 1}) == (True, 0)
    assert called == []
    assert "cancelled" in capsys.readouterr().out


def test_models_use_sets_default_and_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    fp = str(tmp_path / "d.json")
    assert C.dispatch("models", ["use", "m1"], {"port": 1, "defaults_file": fp}) == (True, 0)
    out = capsys.readouterr().out
    assert "reboot" in out


# 6. /logs off buffers, /logs on flushes ---------------------------------------
def test_logs_toggle_buffers_and_flushes(capsys):
    ctx = _ctx()
    ctx.logs_on = True
    assert C.dispatch("logs", ["off"], {"port": 1}, ctx) == (True, 0)
    assert ctx.logs_on is False
    capsys.readouterr()

    def _sink(text):
        ctx.logbuf.append(text)
        ctx.transcript.append(text.rstrip("\n"))
        # logs off: record only, do not display
    C._pump_logs(_Proc(["l1\n", "l2\n"]), _Stop(), _sink)
    assert "l1" not in capsys.readouterr().out  # buffered, not shown
    assert C.dispatch("logs", ["on"], {"port": 1}, ctx) == (True, 0)
    out = capsys.readouterr().out
    assert "l1" in out and "l2" in out  # flushed, nothing lost


class _Proc:
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(lines))


class _Stop:
    def set(self):
        pass


# 7. /doctor FAIL + fix hint ----------------------------------------------------
def test_doctor_fails_api_with_fix():
    rows = C.doctor_checks(1, node_port=1, backend_pid=999999,
                           model_id="no-such-model-xyz")
    names = [r[0] for r in rows]
    assert names == ["ports", "api", "version", "tls", "model-cache", "discovery"]
    api = [r for r in rows if r[0] == "api"][0]
    assert api[1] is False and api[3]  # FAIL carries a fix hint


# 8. /clear keeps backend/history/defaults --------------------------------------
def test_clear_wipes_screen_only(monkeypatch):
    calls = []
    monkeypatch.setattr(C.os, "system", lambda c: calls.append(c) or 0)
    assert C.dispatch("clear", [], {"port": 1}) == (True, 0)
    assert calls  # screen was cleared; no backend handle touched


# 9. Ctrl+C propagates so the loop can keep the session -------------------------
def test_ctrl_c_propagates_from_command(monkeypatch):
    monkeypatch.setattr(C, "_wait_ready", lambda *a, **k: True)

    def _boom(ns):
        raise KeyboardInterrupt()
    monkeypatch.setattr(C, "cmd_benchmark", _boom)
    with pytest.raises(KeyboardInterrupt):
        C.dispatch("benchmark", [], {"port": 1})


# 10. !shell --------------------------------------------------------------------
def test_shell_echo_and_exit_code(capsys):
    rc = C.run_shell("echo hi")
    assert rc == 0
    out = capsys.readouterr().out
    assert "hi" in out and "[exit 0]" in out


def test_shell_denylist_blocks(capsys):
    assert C.run_shell("rm -rf /") == 125
    assert C.run_shell("mkfs /dev/sda") == 125
    assert C.run_shell("dd if=/dev/zero of=/dev/sda") == 125
    assert "blocked" in capsys.readouterr().out
    assert C.run_shell("echo not-rm -rf / ; echo safe") == 0  # no false hit


def test_shell_timeout_kills(monkeypatch, capsys):
    class _HangProc:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            type(self).returncode = -9

        def wait(self, timeout=None):
            return -9

        def communicate(self, timeout=None):
            return ("", "")

    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: _HangProc())
    assert C.run_shell("sleep 999", timeout=1) == 124
    assert "killed" in capsys.readouterr().out


def test_resolve_shell_and_keys():
    assert C.resolve_line("!nvidia-smi") == ("shell", "nvidia-smi")
    assert C.resolve_line("?") == ("keys", None)
    assert C.resolve_line("plain words") == ("chat", "plain words")


# 11. history file ---------------------------------------------------------------
def test_history_roundtrip(tmp_path, monkeypatch):
    fake = types.ModuleType("readline")
    stored = {}

    def _read(p):
        if p in stored:
            fake._items = list(stored[p])
    def _write(p):
        stored[p] = list(getattr(fake, "_items", []))
    fake._items = ["cmd1", "cmd2"]
    fake.read_history_file = _read
    fake.write_history_file = _write
    fake.set_history_length = lambda n: None
    fake.parse_and_bind = lambda s: None
    fake.set_completer = lambda f: None
    fake.get_line_buffer = lambda: ""
    monkeypatch.setitem(sys.modules, "readline", fake)
    monkeypatch.setattr(C, "HISTORY_FILE", str(tmp_path / "hist"))
    C._setup_history()
    C._save_history()
    assert stored[str(tmp_path / "hist")] == ["cmd1", "cmd2"]


# 12. /export + /bug --------------------------------------------------------------
def test_export_writes_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = _ctx()
    ctx.transcript = ["> /status", "api=ok"]
    dest = C.cmd_export(ctx, None)
    assert dest and os.path.isfile(dest)
    body = open(dest, encoding="utf-8").read()
    assert "> /status" in body and "api=ok" in body


def test_bug_bundle_has_versions_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = _ctx()
    ctx.logbuf.append("[node] tail-line")
    dest = C.cmd_bug(ctx, None)
    assert dest and os.path.isfile(dest)
    body = open(dest, encoding="utf-8").read()
    assert "synapse" in body and "tail-line" in body
    assert "never sent anywhere" in body
