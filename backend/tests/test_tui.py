"""
Behavioral tests for the full-screen TUI (display layer only, no app run).

The engine (resolve/dispatch/shell/doctor/export) is tested in
test_console*.py; here we test the TUI wiring: palette values, the
overlay list and the stdout tee that feeds /export.
"""
import io

import pytest

from synapse import console as C
from synapse import tui as T


def _ctx(**kw):
    base = dict(port=1, defaults_file=None, node_label="node",
                model_short="SmolLM2", modes="")
    base.update(kw)
    return C.ConsoleCtx(**base)


def test_palette_lists_every_command():
    vals = T.pick_values()
    assert [v for v, _ in vals] == list(C.COMMANDS)
    assert all(label.startswith("/") for _, label in vals)


def test_tee_forwards_and_captures(capsys):
    real = io.StringIO()
    tee = T.Tee(real)
    assert tee.isatty() is False  # transcript stays parseable plain text
    tee.write("ab")
    tee.writelines(["c\n"])
    tee.flush()
    assert real.getvalue() == "abc\n"
    assert tee.text == "abc\n"


def test_exec_command_captures_and_keeps(monkeypatch):
    monkeypatch.setattr(C, "cmd_status", lambda ns: print("api=ok") or 0)
    keep, rc, out = T.exec_command("status", [], {"port": 1}, _ctx())
    assert (keep, rc) == (True, 0) and "api=ok" in out


def test_exec_command_quit_stops_loop():
    keep, rc, _ = T.exec_command("quit", [], {"port": 1}, _ctx())
    assert (keep, rc) == (False, 0)


def test_exec_command_ctrl_c_keeps_session(monkeypatch, capsys):
    def _boom(ns):
        raise KeyboardInterrupt()
    monkeypatch.setattr(C, "cmd_benchmark", _boom)
    keep, rc, out = T.exec_command("benchmark", [], {"port": 1}, _ctx())
    assert (keep, rc) == (True, 130)
    assert "session kept" in capsys.readouterr().out


def test_overlay_list_filter_navigate_render():
    items = [(n, "/%-10s  %s" % (n, C.COMMANDS[n][1])) for n in C.COMMANDS]
    lst = T.OverlayList(items)
    assert lst.current() == "status"
    lst.filter = "topo"  # the TUI's filter box writes this (_on_filter)
    assert lst.current() == "topology"
    assert any(r.startswith(">") and "topology" in r for r in lst.render_rows())
    lst2 = T.OverlayList(items)
    lst2.move(1)
    assert lst2.current() == "topology"
    lst2.move(-1)
    assert lst2.current() == "status"
    lst3 = T.OverlayList(items)
    lst3.filter = "zzz-no-match"
    assert lst3.current() is None
    assert lst3.render_rows() == ["  (no match)"]


def test_model_items_from_catalog():
    items = T.load_model_items()
    assert len(items) > 100  # real catalog, filterable
    assert all("/" in v for v, _ in items)


def test_leader_seqs_documented():
    seqs = dict(T.LEADER_SEQS)
    assert seqs == {"c-x c-c": "quit", "c-x p": "palette", "c-x m": "models",
                    "c-x t": "theme", "c-x ?": "help"}
    assert "c-x p" in T.LEADER_HINT


def test_theme_command_lists_and_sets(tmp_path, monkeypatch, capsys):
    import synapse.theme as TH
    monkeypatch.setattr(TH, "THEME_DIR", str(tmp_path))
    monkeypatch.setattr(TH, "SELECTION_FILE", str(tmp_path / "theme.json"))
    assert C.dispatch("theme", [], {"port": 1}) == (True, 0)
    out = capsys.readouterr().out
    assert "synapse" in out and "ember-dark" in out
    assert C.dispatch("theme", ["ember-light"], {"port": 1}) == (True, 0)
    assert "saved" in capsys.readouterr().out
    assert C.dispatch("theme", ["nope"], {"port": 1}) == (True, 1)
    assert "fail" in capsys.readouterr().out


def test_line_style_semantics():
    assert T.line_style("> /status") == "echo"
    assert T.line_style("[node] booted") == "node"
    assert T.line_style("[node] FAIL: x") == "node"  # log wins over fail
    assert T.line_style("[fail] bad key") == "err"
    assert T.line_style("ports  FAIL  fix: ...") == "err"
    assert T.line_style("[ok] saved") == "ok"
    assert T.line_style("api  PASS  detail") == "ok"
    assert T.line_style("(ngắt — phiên còn)") == "warn"
    assert T.line_style("[console] running") == "muted"
    assert T.line_style("› /status") == "echo"
    assert T.line_style("plain output") == ""
    assert T.line_style("") == ""


def test_highlighter_applies_class():
    from prompt_toolkit.layout.processors import Transformation

    class _TI:
        def unpack(self):
            return (None, None, 0, None, [("", "> /status"), ("", " end")],
                    None, None)

    out = T.TranscriptHighlighter().apply_transformation(_TI())
    assert isinstance(out, Transformation)
    assert out.fragments == [("class:echo", "> /status end")]

    class _TI2:
        def unpack(self):
            return (None, None, 0, None, [("", "plain")], None, None)

    out2 = T.TranscriptHighlighter().apply_transformation(_TI2())
    assert out2.fragments == [("", "plain")]


def test_slash_frag_rules():
    assert T.slash_frag("/") == ""
    assert T.slash_frag("/to") == "to"
    assert T.slash_frag("/a b") is None  # args started -> hide
    assert T.slash_frag("plain") is None  # not a slash context
    assert T.slash_frag("") is None


def test_slash_items_full_on_bare_slash():
    names = [n for n, _ in T.slash_items("")]
    assert names == list(C.COMMANDS)
    assert [n for n, _ in T.slash_items("topo")] == ["topology"]
    assert T.slash_items("zzz-no-match") == []


def test_slash_complete_replaces_current_line():
    assert T.slash_complete("/to", 3, "topology") == ("/topology ", 10)
    assert T.slash_complete("/", 1, "status") == ("/status ", 8)
    # multiline: only the current line is replaced
    text, cur = T.slash_complete("ab\n/mo", 6, "models")
    assert (text, cur) == ("ab\n/models ", 11)


async def _wait_until(cond, timeout=5.0):
    import asyncio as _aio
    t0 = _aio.get_event_loop().time()
    while not cond():
        if _aio.get_event_loop().time() - t0 > timeout:
            raise AssertionError("timeout waiting for condition")
        await _aio.sleep(0.05)


def test_headless_slash_menu_typing_tab_enter(monkeypatch, tmp_path):
    import asyncio
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setattr(T, "HISTORY_FILE", str(tmp_path / "hist"))

    async def _drive():
        with create_pipe_input() as inp:
            args = type("A", (), {})()
            ctx = C.ConsoleCtx(port=1, defaults_file=None)
            proc = type("P", (), {"poll": lambda self: None,
                                  "pid": 0})()
            ui = T._build_ui(args, ctx, proc, None, None,
                             app_input=inp, app_output=DummyOutput())
            seen = {}

            async def _driver():
                inp.send_text("/")
                await _wait_until(lambda: ui.state.slash_visible)
                seen["menu"] = ui.slash_buf.text
                inp.send_text("topo")
                await _wait_until(
                    lambda: [n for n, _ in ui.state.slash_items] == ["topology"])
                inp.send_text("\t")  # Tab inserts the selection
                await _wait_until(lambda: ui.input_buf.text == "/topology ")
                assert not ui.state.slash_visible  # trailing space hides it
                inp.send_text("\r")  # Enter submits
                await _wait_until(lambda: ui.input_buf.text == "")
                seen["echo"] = True
                ui.app.exit(result=0)

            asyncio.ensure_future(_driver())
            await ui.app.run_async()
            return seen

    seen = asyncio.run(_drive())
    assert "/status" in seen["menu"]  # bare / shows the full box
    assert seen["echo"]


def test_build_ui_accepts_no_proc_yet():
    # TUI opens first now; the backend boots behind it (state.proc set later).
    from prompt_toolkit.output import DummyOutput
    args = type("A", (), {})()
    ctx = C.ConsoleCtx(port=1, defaults_file=None)
    ui = T._build_ui(args, ctx, None, None, None, app_output=DummyOutput())
    assert ui.state.proc is None


def test_spawn_backend_failure_raises():
    with __import__("pytest").raises(Exception):
        T._spawn_backend(["definitely-not-a-real-binary-xyz"], {})


def test_footer_and_collapse():
    assert T._footer(0, 12.34) == "  ● done · 12.3s"
    assert T._footer(3, 0.06) == "  × exit 3 · 0.1s"
    short = "a\nb"
    assert T._collapse(short) == short
    long_text = "\n".join("l%d" % i for i in range(40))
    out = T._collapse(long_text)
    assert "l0" in out and "l39" in out and "gập" in out
    assert len(out.split("\n")) < 40


def test_panel_text_sections():
    T._bar_cache.update(ts=0.0)  # force fresh probe (dead port -> fallback)
    ctx = C.ConsoleCtx(port=1, defaults_file=None, node_label="n1",
                       model_short="SmolLM2", modes="[adv]")
    ctx.logbuf.append("[node] hello\n")
    st = T.TuiState()
    body = T._panel_text(ctx, st)
    assert "cluster" in body and "n1" in body and "SmolLM2" in body
    assert "jobs" in body and "(trống)" in body
    assert "logs" in body and "hello" in body
    st.job = {"label": "/benchmark", "t0": 0.0}
    body2 = T._panel_text(ctx, st)
    assert "/benchmark" in body2
    ctx.logs_on = False
    assert "/logs on" in T._panel_text(ctx, st)


def test_headless_chat_routes_to_client(monkeypatch, tmp_path):
    import asyncio
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setattr(T, "HISTORY_FILE", str(tmp_path / "hist"))
    monkeypatch.setattr(C, "_wait_ready", lambda *a, **k: True)
    calls = []
    monkeypatch.setattr(C, "cmd_client",
                        lambda ns: calls.append(ns.prompt) or 0)

    async def _drive():
        with create_pipe_input() as inp:
            args = type("A", (), {})()
            ctx = C.ConsoleCtx(port=1, defaults_file=None)
            proc = type("P", (), {"poll": lambda self: None,
                                  "pid": 0})()
            loop = asyncio.get_event_loop()
            ui = T._build_ui(args, ctx, proc, None, loop,
                             app_input=inp, app_output=DummyOutput())
            seen = {}

            async def _driver():
                for ch in "mấy node?":
                    inp.send_text(ch)
                await _wait_until(lambda: ui.input_buf.text == "mấy node?")
                inp.send_text("\r")
                await _wait_until(lambda: len(calls) == 1)
                seen["prompt"] = calls[0]
                await asyncio.sleep(1.5)  # let the worker finish + footer
                ui.app.exit(result=0)

            asyncio.ensure_future(_driver())
            await ui.app.run_async()
            return seen

    seen = asyncio.run(_drive())
    assert seen["prompt"] == "mấy node?"  # plain text -> chat


def test_panel_truncates_to_width():
    ctx = C.ConsoleCtx(port=1, defaults_file=None)
    ctx.logbuf.append("[node] " + "x" * 200 + "\n")
    body = T._panel_text(ctx, T.TuiState())
    assert max(len(ln) for ln in body.split("\n")) <= 38


def test_logs_on_panel_mode_skips_dump(capsys):
    ctx = C.ConsoleCtx(port=1, defaults_file=None)
    ctx.has_panel = True
    ctx.logs_on = False
    ctx.logbuf.append("buffered-line\n")
    assert C.dispatch("logs", ["on"], {"port": 1}, ctx) == (True, 0)
    out = capsys.readouterr().out
    assert "panel streaming" in out
    assert "buffered-line" not in out  # stays in the panel, not transcript


def test_benchmark_uses_effective_model_and_pipes_in_tui(monkeypatch):
    monkeypatch.setattr(C, "_wait_ready", lambda *a, **k: True)
    seen = {}

    def _rec(ns):
        seen.update(vars(ns))
        return 0

    monkeypatch.setattr(C, "cmd_benchmark", _rec)
    ctx = C.ConsoleCtx(port=1, defaults_file=None, model_id="m-good")
    ctx.has_panel = True
    assert C.dispatch("benchmark", [], {"port": 1}, ctx) == (True, 0)
    assert seen["model"] == "m-good"
    assert seen["pipe_output"] is True


@pytest.mark.real_wait
def test_wait_ready_ok_cancel_timeout(monkeypatch, capsys):
    import threading
    monkeypatch.setattr(C, "http_get_json",
                        lambda *a, **k: {"inference_ready": True})
    assert C._wait_ready(1) is True
    ev = threading.Event()
    ev.set()
    assert C._wait_ready(1, ev, timeout=30) is False
    assert "ngắt" in capsys.readouterr().out
    monkeypatch.setattr(C, "http_get_json", lambda *a, **k: (_ for _ in ()).throw(
        Exception("refused")))
    assert C._wait_ready(1, None, timeout=0) is False
    assert "chưa sẵn sàng" in capsys.readouterr().out


def test_benchmark_waits_for_ready(monkeypatch):
    monkeypatch.setattr(C, "_wait_ready", lambda *a, **k: False)
    assert C.dispatch("benchmark", [], {"port": 1},
                      C.ConsoleCtx(port=1, defaults_file=None)) == (True, 3)


def test_want_tui_logic(monkeypatch):
    ns = {"line": False}
    args = type("A", (), ns)()
    # no TTY (pytest pipes) -> line mode
    assert C._want_tui(args) is False
    # forced line mode wins even on a TTY
    monkeypatch.setattr(C, "_is_tty", lambda: True)
    args2 = type("A", (), {"line": True})()
    assert C._want_tui(args2) is False
    monkeypatch.setenv("SYNAPSE_LINE", "1")
    assert C._want_tui(args) is False
    monkeypatch.delenv("SYNAPSE_LINE")
    assert C._want_tui(args) is True  # TTY + prompt_toolkit present


def test_exec_command_quiet_mode_writes_nothing_to_the_terminal(monkeypatch, capsys):
    monkeypatch.setattr(C, "cmd_status", lambda ns: print("api=ok") or 0)
    keep, rc, out = T.exec_command("status", [], {"port": 1}, _ctx(), echo=False)
    assert (keep, rc) == (True, 0) and "api=ok" in out
    assert capsys.readouterr().out == ""
