"""
Full-screen console (prompt_toolkit custom Application). UI.md phase 2.

This is the part that LOOKS like the reference repos:
- scrollback transcript on top, multiline input box at the bottom,
- pinned status toolbar + leader cheatsheet under the input,
- fuzzy `/slash` completion popup as you type (Claude Code style),
- float overlays: palette (ctrl+p), model picker (leader m), theme picker
  (leader t), help (leader ?), confirm dialogs (OpenCode style),
- ctrl+x leader sequences (OpenCode keybinds.leader),
- [node] logs stream above without breaking the input (queue + drain),
- theme colors from synapse.theme (synapse / ember-dark / ember-light).

The engine is untouched: every command runs the same deterministic
dispatch in synapse.console, in a worker thread, with output teed to
memory and appended to the transcript afterwards. Only the loop thread
ever touches the UI (pump -> queue -> drain task).

Honest limits (same as UI.md):
- Ctrl+C during a running command cannot kill the worker thread; it
  prints a note and the command runs to completion.
- lone ctrl+x does nothing (sequences only); prompt_toolkit FileHistory
  shares ~/.synapse/console_history with readline (cosmetic '+' cross-mode).
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
from functools import partial

from synapse import console as C
from synapse import theme as T
from synapse.cli import Cancelled as _Cancelled

from prompt_toolkit.layout.processors import Processor, Transformation

HISTORY_FILE = C.HISTORY_FILE
PROMPT = C.PROMPT
TRANSCRIPT_CAP = 60000

LEADER_SEQS = (
    ("c-x c-c", "quit"),
    ("c-x p", "palette"),
    ("c-x m", "models"),
    ("c-x t", "theme"),
    ("c-x ?", "help"),
)
LEADER_HINT = "  ".join("%s %s" % s for s in LEADER_SEQS)


# -- pure helpers (tested) -----------------------------------------------------

_bar_cache = {"node": "node", "peers": "?", "ts": 0.0}


def _bar_data(ctx):
    """Cached (node, peers), refreshed at most every 5s. Never raises."""
    try:
        now = time.monotonic()
        if now - _bar_cache["ts"] > 5:
            data = C.fetch_bar_data(ctx.port)
            _bar_cache["node"] = (data.get("node") if data else None) or ctx.node_label
            _bar_cache["peers"] = data.get("peers") if data else "?"
            _bar_cache["ts"] = now
        return _bar_cache["node"], _bar_cache["peers"]
    except Exception:
        return ctx.node_label, "?"


def toolbar_line(ctx) -> str:
    """Pinned bottom-bar text. Never raises."""
    try:
        node, peers = _bar_data(ctx)
        return C.build_bar_line(node, peers, ctx.model_short, ctx.modes)
    except Exception:
        return "node | peers ?"


def _panel_text(ctx, state, log_tail=8, width=38):
    """Right ops panel content (pure, tested). Lines fit the panel width."""
    node, peers = _bar_data(ctx)
    lines = ["cluster", "  node    %s" % node, "  peers   %s" % peers,
             "  model   %s" % ctx.model_short]
    if ctx.modes:
        lines.append("  modes   %s" % ctx.modes)
    lines.append("jobs")
    job = getattr(state, "job", None)
    if job:
        dt = max(0.0, time.monotonic() - (job.get("t0") or time.monotonic()))
        lines.append("  • %s  %.0fs" % (str(job.get("label", "?"))[:width - 10], dt))
    else:
        lines.append("  (trống)")
    lines.append("logs")
    if not ctx.logs_on:
        lines.append("  (tắt — /logs on để xem)")
    else:
        tail = list(ctx.logbuf)[-log_tail:]
        if not tail:
            lines.append("  …")
        lines += ["  " + ln.rstrip("\n")[:width - 2] for ln in tail]
    return "\n".join(lines)


def _spin():
    return SPINNER[int(time.monotonic() * 6) % len(SPINNER)]


def panel_fragments(ctx, state, width=38, log_tail=10):
    """Colored right-panel content: CLUSTER / JOBS / LOGS sections."""
    node, peers = _bar_data(ctx)
    online = peers not in ("?", None)
    out = []

    def nl():
        out.append(("", "\n"))

    def section(title):
        if out:
            nl()
        out.append(("class:section", title))
        nl()

    def kv(key, val, cls="class:value"):
        out.append(("class:key", "  %-7s" % key))
        out.append((cls, _pad(str(val), max(1, width - 9)).rstrip()))
        nl()

    section("CLUSTER")
    out.append(("class:dot.ok" if online else "class:dot.off", "  " + (G_OK if online else "○") + " "))
    out.append(("class:value", str(node)))
    out.append(("class:key", "  this node" if online else "  api down"))
    nl()
    kv("peers", peers)
    kv("model", ctx.model_short)
    if ctx.modes:
        kv("modes", ctx.modes)

    section("JOBS")
    job = getattr(state, "job", None)
    if job:
        dt = max(0.0, time.monotonic() - (job.get("t0") or time.monotonic()))
        out.append(("class:dot.busy", "  %s " % _spin()))
        out.append(("class:value", _pad(str(job.get("label", "?")), width - 12).rstrip()))
        out.append(("class:key", "  %ds" % dt))
        nl()
    else:
        out.append(("class:key", "  idle"))
        nl()

    section("LOGS")
    if not ctx.logs_on:
        out.append(("class:key", "  off — /logs on"))
        nl()
    else:
        tail = list(ctx.logbuf)[-log_tail:]
        if not tail:
            out.append(("class:key", "  …"))
            nl()
        for ln in tail:
            ln = ln.rstrip("\n")
            if ln.startswith("[node] "):
                ln = ln[7:]
            out.append(("class:node", "  " + _pad(ln, width - 2).rstrip()))
            nl()
    if out and out[-1] == ("", "\n"):
        out.pop()
    return out


def status_fragments(ctx, state, cols):
    """One-line bottom bar: state pill + node facts left, key hints right."""
    node, peers = _bar_data(ctx)
    left = [("class:status.logo", " synapse "), ("class:status", " ")]
    if getattr(state, "job", None):
        dt = time.monotonic() - (state.job.get("t0") or time.monotonic())
        left += [("class:status.busy", "%s running %ds" % (_spin(), dt))]
    elif peers in ("?", None):
        left += [("class:status.err", "○ api down")]
    else:
        left += [("class:status.ok", G_OK + " online")]
    for key, val in (("node", node), ("peers", peers), ("model", ctx.model_short),
                     ("mode", ctx.modes)):
        if val in ("", None):
            continue
        left += [("class:status.key", "  %s " % key), ("class:status.value", str(val))]
    hints = [("ctrl+p", "commands"), ("ctrl+x ?", "keys"), ("ctrl+c", "quit")]
    right = []
    for i, (k, d) in enumerate(hints):
        right += [("class:status.hintkey", ("  " if i else "") + k),
                  ("class:status.hint", " " + d)]
    right.append(("class:status", " "))
    from prompt_toolkit.utils import get_cwidth
    lw = sum(get_cwidth(t) for _, t in left)
    rw = sum(get_cwidth(t) for _, t in right)
    if lw + rw + 1 > cols:
        right = []
        rw = 0
    return left + [("class:status", " " * max(0, cols - lw - rw))] + right


def menu_fragments(rows, index, name_width=11, max_rows=10, more_hint="keep typing"):
    """Selectable rows [(name, desc)] -> highlighted fragments."""
    out = []
    if not rows:
        return [("class:mrow.more", "  (no match)")]
    start = (index // max_rows) * max_rows
    for i, (name, desc) in enumerate(rows[start:start + max_rows]):
        sel = start + i == index % len(rows)
        pfx = "mrow.sel" if sel else "mrow"
        out.append(("class:" + pfx, (G_CARD + " ") if sel else "  "))
        out.append(("class:" + ("mrow.sel.name" if sel else "mrow.name"), _pad(name, name_width)))
        out.append(("class:" + ("mrow.sel.desc" if sel else "mrow.desc"), " " + desc))
        out.append(("", "\n"))
    if len(rows) > max_rows:
        out.append(("class:mrow.more", "  … %d more (%s)" % (len(rows) - max_rows, more_hint)))
    elif out:
        out.pop()
    return out


def build_completer():
    """Fuzzy /slash completer with one-line descriptions."""
    from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
    words = ["/" + n for n in C.COMMANDS]
    meta = {"/" + n: desc for n, (_, desc) in C.COMMANDS.items()}
    return FuzzyCompleter(WordCompleter(words, meta_dict=meta))


def pick_values():
    """(value, label) rows for the palette (all commands)."""
    return [(n, "/%-10s  %s" % (n, C.COMMANDS[n][1])) for n in C.COMMANDS]


def slash_frag(line_before_cursor):
    """Fragment after a leading `/` on the current line, or None.

    None = the slash menu must stay hidden (not a slash context, or
    args already started — same hide rules as OpenCode's autocomplete).
    """
    if not line_before_cursor.startswith("/"):
        return None
    frag = line_before_cursor[1:]
    if " " in frag or "\t" in frag or "\n" in frag:
        return None
    return frag


def slash_items(frag):
    """[(name, label)] for the slash menu. Empty frag = full list."""
    return [(n, "/%-10s  %s" % (n, C.COMMANDS[n][1]))
            for n in C.suggest_commands(frag or "")]


def slash_complete(text, cursor, name):
    """Replace the current line's `/frag` with `/name `.

    Returns (new_text, new_cursor). Pure string surgery (tested).
    """
    doc_before = text[:cursor]
    line_start = doc_before.rfind("\n") + 1
    new_line = "/" + name + " "
    new_text = text[:line_start] + new_line + text[cursor:]
    return new_text, line_start + len(new_line)


# Glyphs present in BOTH Cascadia Mono (Windows Terminal default) and
# Consolas (legacy console): ✔ ✘ ❯ ◆ render as tofu boxes in Consolas.
G_ECHO = "›"
G_CARD = "▸"
G_OK = "●"
G_ERR = "×"
G_GUTTER = "│"
SPINNER = ("·", "•", "●", "•")
CARD_INDENT = "  " + G_GUTTER + " "


def _footer(rc, dt):
    """Command card footer: `  ● done · 12.3s` / `  × exit N · 0.1s`."""
    if rc == 0:
        return "  %s done · %.1fs" % (G_OK, dt)
    return "  %s exit %s · %.1fs" % (G_ERR, rc, dt)


def card_header(label, clock=None):
    """`▸ /status  · 14:02:11` — the time part is styled muted."""
    clock = clock or time.strftime("%H:%M:%S")
    return "%s %s  · %s" % (G_CARD, label, clock)


def card_body(text):
    """Indent command output under a muted gutter so cards read as blocks."""
    lines = (text or "").split("\n")
    return "\n".join(CARD_INDENT + ln if ln.strip() else "  " + G_GUTTER
                     for ln in lines)


def reply_body(text):
    """Chat reply: `● ` on the first line, aligned indent after (Claude style)."""
    lines = (text or "").strip("\n").split("\n")
    if not lines or not "".join(lines).strip():
        return ""
    return "\n".join((G_OK + " " if i == 0 else "  ") + ln
                     for i, ln in enumerate(lines))


def style_line(text):
    """[(class, text)] fragments for one transcript line (pure, tested).

    Characters are never added or removed: the fragments concatenate back
    to `text` exactly (the buffer's cursor/selection math depends on it).
    """
    s = text or ""
    if s.startswith(CARD_INDENT) or s == "  " + G_GUTTER:
        rest = s[len(CARD_INDENT):]
        cls = line_style(rest)
        return [("class:gutter", s[:len(CARD_INDENT)])] + (
            [("class:" + cls if cls else "", rest)] if rest else [])
    if s.startswith(G_CARD + " "):
        head, sep, clock = s.partition("  · ")
        out = [("class:card.head", head)]
        if sep:
            out.append(("class:card.time", sep + clock))
        return out
    if s.startswith("  " + G_OK + " "):
        return [("class:card.ok", s[:4]), ("class:card.meta", s[4:])]
    if s.startswith("  " + G_ERR + " "):
        return [("class:card.err", s)]
    if s.startswith(G_OK + " "):
        return [("class:reply.dot", s[:2]), ("class:reply", s[2:])]
    if s[:1] in ("╭", "╰") or (s.startswith("│") and s.endswith("│")):
        return _banner_fragments(s)
    if s.startswith(G_ECHO + " "):
        return [("class:echo", s)]
    cls = line_style(s)
    return [("class:" + cls if cls else "", s)]


def _banner_fragments(s):
    if s[:1] in ("╭", "╰"):
        i = s.find(" synapse ")
        if i < 0:
            return [("class:banner.border", s)]
        j = i + len(" synapse ")
        return [("class:banner.border", s[:i]), ("class:banner.title", s[i:j]),
                ("class:banner.border", s[j:])]
    inner = s[1:-1]
    out = [("class:banner.border", "│")]
    stripped = inner.lstrip()
    lead = inner[:len(inner) - len(stripped)]
    if stripped.startswith(G_OK + " "):
        out += [("", lead), ("class:dot.ok", G_OK), ("class:banner.title", stripped[1:])]
    else:
        key, sp, val = stripped.partition("  ")
        out += [("", lead), ("class:banner.key", key + sp), ("", val)]
    out.append(("class:banner.border", "│"))
    return out


def _pad(text, width):
    """Pad/trim to a display width (CJK/emoji aware via prompt_toolkit)."""
    from prompt_toolkit.utils import get_cwidth
    out, w = "", 0
    for ch in text:
        cw = get_cwidth(ch)
        if w + cw > width:
            break
        out += ch
        w += cw
    return out + " " * (width - w)


def banner_lines(version, model, port, theme_name, width=64):
    """Rounded welcome card (plain text; colored by style_line)."""
    inner = width - 2
    rows = [
        "  %s console %s · deterministic, no agent" % (G_OK, version),
        "",
        "    model   %s" % model,
        "    api     127.0.0.1:%s  ·  theme %s" % (port, theme_name),
        "    try     /status   /topology   /help",
        "    keys    ctrl+p commands · ctrl+x ? keys · ctrl+c quit",
    ]
    title = " synapse "
    top = "╭─" + title + "─" * (inner - 1 - len(title)) + "╮"
    body = ["│" + _pad(r, inner) + "│" for r in rows]
    return "\n".join([top] + body + ["╰" + "─" * inner + "╯"])


def _collapse(text, limit=30):
    """Long outputs collapse to head + tail so the transcript stays scannable."""
    lines = (text or "").split("\n")
    if len(lines) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return "\n".join(
        lines[:head]
        + ["  … %d dòng đã gập (mở /export để xem đủ) …" % (len(lines) - limit)]
        + lines[-tail:])


def load_model_items():
    """[(model_id, label)] from the catalog. Best effort, [] on error."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "config", "hf_models.json"),
                  encoding="utf-8") as f:
            catalog = json.load(f)
    except Exception:
        return []
    items = []
    for m in catalog if isinstance(catalog, list) else []:
        if not isinstance(m, dict) or not m.get("name"):
            continue
        label = "%s  (%s)" % (m["name"], m.get("parameter_count", "?"))
        items.append((m["name"], label))
    return items


def line_style(text):
    """Semantic class for one transcript line (theme-colored in TUI).

    The line form (see UI.md): `>` echo, `[node]` logs, `[console]`
    lifecycle, `[shell]`/`[export]`/`[bug]`/`[logs]` notes, `[ok]`/`[fail]`
    results, plain command output. Returns "" for default text.
    """
    s = text or ""
    if s.startswith("> ") or s.startswith(G_ECHO + " "):
        return "echo"
    if s.startswith("[node]"):
        return "node"
    if "[fail]" in s or "FAIL" in s or s.startswith("(lỗi") or "Traceback" in s:
        return "err"
    if "[ok]" in s or "PASS" in s:
        return "ok"
    if "(ngắt" in s or "(đang ngắt" in s or "cancelled" in s:
        return "warn"
    if s.startswith("[console]"):
        return "muted"
    return ""


class TranscriptHighlighter(Processor):
    """BufferControl input_processor: theme-color each line via line_style."""

    def apply_transformation(self, transformation_input):
        (bc, doc, lineno, s2d, fragments, _a, _b) = transformation_input.unpack()
        text = "".join(f[1] for f in fragments)
        styled = style_line(text)
        if all(not c for c, _ in styled):
            return Transformation(fragments)
        return Transformation(styled)


class Tee:
    """stdout wrapper: live to the terminal AND captured for /export.

    isatty() is False on purpose so rich tables render as parseable
    plain text inside the transcript.
    """

    def __init__(self, real):
        self.real = real
        self.parts = []

    def write(self, s):
        self.real.write(s)
        self.parts.append(s)
        return len(s)

    def writelines(self, lines):
        for ln in lines:
            self.write(ln)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    @property
    def text(self):
        return "".join(self.parts)


class _NullWriter:
    def write(self, s):
        return len(s)

    def writelines(self, lines):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


def exec_command(name, argv, defaults, ctx):
    """Run one dispatch with output teed. Returns (keep, rc, text).

    KeyboardInterrupt (Ctrl+C during the command) keeps the session.
    """
    tee = Tee(sys.stdout)
    old = sys.stdout
    sys.stdout = tee
    try:
        keep, rc = C.dispatch(name, argv, defaults, ctx)
    except KeyboardInterrupt:
        keep, rc = True, 130
        print("\n(interrupted — session kept, /quit to leave)")
    finally:
        sys.stdout = old
    return keep, rc, tee.text


def exec_quiet(name, argv, defaults, ctx):
    """Same as exec_command but nothing reaches the real terminal."""
    tee = Tee(_NullWriter())
    old = sys.stdout
    sys.stdout = tee
    try:
        keep, rc = C.dispatch(name, argv, defaults, ctx)
    except KeyboardInterrupt:
        keep, rc = True, 130
        tee.write("\n(interrupted — session kept, /quit to leave)\n")
    finally:
        sys.stdout = old
    return keep, rc, tee.text


class OverlayList:
    """Filterable selectable list backing palette/picker/theme overlays."""

    def __init__(self, items):
        self.items = list(items)  # [(value, label)]
        self.filter = ""
        self.index = 0

    def filtered(self):
        f = self.filter.lower()
        if not f:
            return list(self.items)
        starts = [it for it in self.items if it[1].lower().startswith(f)]
        rest = [it for it in self.items
                if f in it[1].lower() and it not in starts]
        return starts + rest

    def move(self, delta):
        n = max(1, len(self.filtered()))
        self.index = (self.index + delta) % n

    def type_char(self, ch):
        self.filter += ch
        self.index = 0

    def backspace(self):
        self.filter = self.filter[:-1]
        self.index = 0

    def current(self):
        rows = self.filtered()
        if not rows:
            return None
        return rows[self.index % len(rows)][0]

    def render_rows(self, max_rows=9):
        rows = self.filtered()
        if not rows:
            return ["  (no match)"]
        start = max(0, (self.index // max_rows) * max_rows)
        out = []
        for i, (_, label) in enumerate(rows[start:start + max_rows]):
            mark = ">" if start + i == self.index % len(rows) else " "
            out.append("%s %s" % (mark, label))
        if len(rows) > max_rows:
            out.append("  … %d more (keep typing to filter)" % (len(rows) - max_rows))
        return out


class TuiState:
    """Mutable session state (loop thread owns UI fields)."""

    def __init__(self):
        self.overlay = None  # None | palette | picker | theme | help | confirm
        self.lists = {}      # kind -> OverlayList
        self.confirm_prompt = ""
        self.confirm_event = None
        self.confirm_result = False
        self.running = False
        self.exit_code = 0
        self.logs_flush = False
        self.slash_visible = False
        self.slash_items = []
        self.slash_index = 0
        self.just_submitted = False
        self.proc = None  # backend subprocess (set at boot, may lag TUI)
        self.job = None  # {"label", "t0"} of the running command, if any


# -- application ---------------------------------------------------------------

def _build_ui(args, ctx, proc, log_q, loop=None, app_input=None, app_output=None):
    """Build the full-screen app and its loop-thread helpers.

    Returns a SimpleNamespace(app, state, input_buf, slash_buf, say,
    submit_text, ...). Pump/drain/run live in _tui_main; this split lets
    headless tests drive real keystrokes through the app (pipe input +
    DummyOutput) without a backend or a terminal.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import (BufferControl, Dimension, Float,
                                       FloatContainer, FormattedTextControl,
                                       HSplit, Layout, VSplit, Window)
    from prompt_toolkit.layout.containers import ConditionalContainer
    from prompt_toolkit.layout.processors import (BeforeInput,
                                                  HighlightMatchingBracketProcessor)
    from prompt_toolkit.styles import Style

    state = TuiState()
    state.proc = proc
    ctx.has_panel = True  # right ops panel: logs live here, not transcript

    # -- buffers ------------------------------------------------------------
    transcript_buf = Buffer(read_only=True)
    # NOTE: no completer on purpose — the dedicated slash menu below owns
    # Tab/Enter completion (a built-in popup would double-render).
    input_buf = Buffer(multiline=True, history=FileHistory(HISTORY_FILE))
    filter_buf = Buffer(multiline=False)
    list_buf = Buffer(read_only=True)
    slash_buf = Buffer(read_only=True)
    panel_buf = Buffer(read_only=True)
    help_buf = Buffer(read_only=True)
    confirm_buf = Buffer(read_only=True)

    def _render_panel():
        panel_buf.set_document(Document(_panel_text(ctx, state)),
                               bypass_readonly=True)

    def say(text):
        """Append text to the transcript (loop thread only)."""
        if not text:
            return
        doc = transcript_buf.document
        # Breathing room: a blank line before each new turn/card.
        if doc.text and (text.startswith(G_ECHO + " ") or text.startswith(G_CARD + " ")):
            text = "\n" + text
        new = (doc.text + ("\n" if doc.text else "") + text.rstrip("\n"))
        transcript_buf.set_document(Document(new[-TRANSCRIPT_CAP:]),
                                    bypass_readonly=True)
        transcript_buf.cursor_position = len(transcript_buf.document.text)

    def transcript_print(fn, *a, **k):
        """Run a print-based helper, routing its output into the transcript."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*a, **k)
        say(buf.getvalue().rstrip("\n"))

    # -- overlays ------------------------------------------------------------
    def overlay_open(kind, items=None, title=""):
        state.overlay = kind
        if items is not None:
            state.lists[kind] = OverlayList(items)
        else:
            state.lists.setdefault(kind, OverlayList([]))
        state.lists[kind].filter = ""
        state.lists[kind].index = 0
        filter_buf.reset()
        _render_list()
        app.layout.focus(filter_win)

    def overlay_close():
        state.overlay = None
        app.layout.focus(input_win)

    def _render_list():
        lst = state.lists.get(state.overlay or "")
        if lst is None:
            return
        title = {"palette": "palette — commands",
                 "picker": "models — Enter sets default + reboot",
                 "theme": "theme — Enter applies now"}[state.overlay]
        lines = [title, "filter: " + lst.filter] + lst.render_rows()
        list_buf.set_document(Document("\n".join(lines)), bypass_readonly=True)

    def overlay_choose():
        lst = state.lists.get(state.overlay or "")
        val = lst.current() if lst else None
        kind = state.overlay
        overlay_close()
        if val is None:
            return
        if kind == "palette":
            submit_text("/" + val)
        elif kind == "picker":
            submit_text("/models use " + val)
        elif kind == "theme":
            ok, msg = T.set_theme(val)
            say(("[ok] " if ok else "[fail] ") + msg)
            if ok:
                try:
                    app.style = Style.from_dict(T.build_style())
                    app.invalidate()
                except Exception:
                    pass

    # -- command execution ----------------------------------------------------
    def tui_confirm(prompt):
        """C.confirm replacement: float dialog, resolved on the loop."""
        ev = threading.Event()
        state.confirm_prompt = prompt
        state.confirm_event = ev
        state.confirm_result = False

        def _show():
            confirm_buf.set_document(
                Document(prompt + "\n\n[y]es / [n]o  (default N)"),
                bypass_readonly=True)
            state.overlay = "confirm"
            app.layout.focus(confirm_win)
            app.invalidate()
        loop.call_soon_threadsafe(_show)
        ev.wait()
        state.overlay = None
        return state.confirm_result

    def submit_text(text):
        if state.running or state.overlay:
            return
        text = " ".join((text or "").split())
        if not text:
            return
        input_buf.reset()
        state.just_submitted = True
        act = C.resolve_line(text)
        kind = act[0]
        if kind == "blank":
            return
        if kind == "keys":
            say(G_ECHO + " " + text)
            transcript_print(C.cmd_keys)
            return
        if kind == "shell":
            app.create_background_task(_run_shell(act[1]))
            return
        if kind == "list":
            say(G_ECHO + " " + text)
            transcript_print(C.cmd_list, act[1])
            return
        if kind == "suggest":
            say(G_ECHO + " " + text)
            say("did you mean /%s? (retype the full command)" % act[1])
            return
        if kind == "pick":
            say(G_ECHO + " " + text)
            overlay_open("palette",
                         [(n, "/%-10s  %s" % (n, C.COMMANDS[n][1])) for n in act[1]])
            return
        if kind == "unknown":
            say(G_ECHO + " " + text)
            say("unknown command — type /help")
            return
        if kind == "chat":
            # chat-first: plain text goes to the model like /client.
            say(G_ECHO + " " + text)
            app.create_background_task(_run_cmd("client", [act[1]],
                                                header=False))
            return
        # ("run", name, argv)
        if act[1] == "clear":
            _clear_view()
            return
        app.create_background_task(_run_cmd(act[1], act[2]))

    # -- slash menu (OpenCode-style box above the input) --------------------
    # Own overlay: shows on a bare `/` too, filters per keystroke,
    # Tab/Enter inserts `/name `, Esc hides. This replaces the generic
    # completion popup (which cannot show descriptions in a themed box).
    def slash_update():
        doc = input_buf.document
        lines = doc.text.split("\n")
        row = min(doc.cursor_position_row, len(lines) - 1)
        line_before = lines[row][:doc.cursor_position_col]
        frag = slash_frag(line_before)
        if frag is None or state.overlay is not None or state.running:
            state.slash_visible = False
            state.slash_items = []
            return
        items = slash_items(frag)
        if not items:
            state.slash_visible = False
            state.slash_items = []
            return
        if [n for n, _ in items] != [n for n, _ in state.slash_items]:
            state.slash_index = 0
        state.slash_items = items
        state.slash_visible = True
        _render_slash()

    def _render_slash():
        rows = []
        for i, (_, label) in enumerate(state.slash_items[:10]):
            mark = ">" if i == state.slash_index % max(1, len(state.slash_items)) else " "
            rows.append("%s %s" % (mark, label))
        if len(state.slash_items) > 10:
            rows.append("  … %d more (keep typing)" % (len(state.slash_items) - 10))
        slash_buf.set_document(Document("\n".join(rows)), bypass_readonly=True)

    def slash_complete_sel():
        if not state.slash_visible or not state.slash_items:
            return False
        name = state.slash_items[state.slash_index % len(state.slash_items)][0]
        doc = input_buf.document
        new_text, new_cursor = slash_complete(doc.text, doc.cursor_position, name)
        input_buf.set_document(Document(new_text, new_cursor))
        return True

    def slash_move(delta):
        if not state.slash_items:
            return
        state.slash_index = (state.slash_index + delta) % len(state.slash_items)
        _render_slash()

    @input_buf.on_text_changed.add_handler
    def _on_input_changed(_buf):
        # Swallow the stray newline when app-level Enter and the control-level
        # Enter handler both fire on submit (both always run in prompt_toolkit).
        if state.just_submitted:
            state.just_submitted = False
            if input_buf.text.startswith("\n"):
                input_buf.set_document(Document(input_buf.text.lstrip("\n")))
                return
        slash_update()

    async def _run_cmd(name, argv, header=True):
        label = "/" + name + (" " + " ".join(argv) if argv else "")
        state.running = True
        state.job = {"label": label, "t0": time.monotonic()}
        t0 = time.monotonic()
        if header:
            say(card_header(label))
        defaults = {"port": ctx.port, "defaults_file": ctx.defaults_file}
        old_confirm = C.confirm
        C.confirm = tui_confirm
        try:
            keep, rc, out = await loop.run_in_executor(
                None, partial(exec_quiet, name, argv, defaults, ctx))
        except _Cancelled:
            keep, rc, out = True, 130, "(ngắt — phiên còn)"
        except Exception as e:
            keep, rc, out = True, 1, "(lỗi: %s — phiên còn)" % e
        finally:
            C.confirm = old_confirm
            state.running = False
            state.job = None
        dt = time.monotonic() - t0
        ctx.transcript.append(out.rstrip("\n"))
        if header:
            if out.strip():
                say(card_body(_collapse(out.rstrip("\n"))))
            say(_footer(rc, dt))
        else:
            # chat reply: `● text`; a footer only when something went wrong
            say(reply_body(_collapse(out.rstrip("\n"))) or "  (no reply)")
            if rc != 0:
                say(_footer(rc, dt))
        if not keep:
            state.exit_code = 0
            app.exit(result=0)
            return
        p = state.proc
        if p is not None and p.poll() is not None:
            say("[console] backend exited (code %d)" % p.returncode)
            state.exit_code = 0 if p.returncode == 0 else 1
            app.exit(result=state.exit_code)

    async def _run_shell(cmd):
        state.running = True
        state.job = {"label": "!" + cmd, "t0": time.monotonic()}
        t0 = time.monotonic()
        say(card_header("!" + cmd))
        ctx.job_event = threading.Event()
        try:
            rc, out = await loop.run_in_executor(
                None, partial(_shell_quiet, cmd, ctx.job_event))
        except Exception as e:
            rc, out = 127, "(lỗi: %s — phiên còn)" % e
        finally:
            state.running = False
            state.job = None
            ctx.job_event = None
        ctx.transcript.append(out.rstrip("\n"))
        if out.strip():
            say(card_body(_collapse(out.rstrip("\n"))))
        say(_footer(rc, time.monotonic() - t0))

    def _shell_quiet(cmd, cancel_event=None):
        tee = Tee(_NullWriter())
        old = sys.stdout
        sys.stdout = tee
        try:
            rc = C.run_shell(cmd, cancel_event=cancel_event)
        finally:
            sys.stdout = old
        return rc, tee.text

    def _clear_view():
        try:
            app.renderer.clear()
        except Exception:
            pass
        say("(view cleared — backend, history and defaults untouched)")

    # -- key bindings ----------------------------------------------------------
    kb = KeyBindings()
    overlay_open_f = Condition(lambda: state.overlay is not None)
    input_focused = Condition(lambda: app.layout.has_focus(input_win))
    slash_f = Condition(lambda: state.slash_visible and state.overlay is None
                        and not state.running
                        and app.layout.has_focus(input_win))

    @kb.add("enter", filter=input_focused & ~overlay_open_f & ~slash_f, eager=True)
    def _(event):
        submit_text(input_buf.text)

    @kb.add("enter", filter=slash_f, eager=True)
    def _(event):
        # Menu open: Enter inserts the selection (OpenCode behavior),
        # Enter again submits.
        slash_complete_sel()

    @kb.add("tab", filter=input_focused & ~overlay_open_f, eager=True)
    def _(event):
        if state.slash_visible:
            slash_complete_sel()
        else:
            slash_update()

    @kb.add("escape", filter=slash_f, eager=True)
    def _(event):
        state.slash_visible = False
        app.invalidate()

    @kb.add("up", filter=slash_f, eager=True)
    def _(event):
        slash_move(-1)
        app.invalidate()

    @kb.add("down", filter=slash_f, eager=True)
    def _(event):
        slash_move(1)
        app.invalidate()

    @kb.add("escape", "enter", filter=input_focused, eager=True)
    @kb.add("c-j", filter=input_focused, eager=True)
    def _(event):
        input_buf.insert_text("\n")

    @kb.add("c-c", filter=~overlay_open_f, eager=True)
    def _(event):
        if state.running:
            C.request_cancel(ctx)
            say("(đang ngắt lệnh…)")
        else:
            app.exit(result=0)

    @kb.add("c-d", filter=input_focused & ~overlay_open_f, eager=True)
    def _(event):
        app.exit(result=0)

    @kb.add("c-l", eager=True)
    def _(event):
        try:
            app.renderer.clear()
        except Exception:
            pass
        app.invalidate()

    @kb.add("c-p", eager=True)
    def _(event):
        if state.overlay or state.running:
            return
        overlay_open("palette", pick_values())

    @kb.add("c-x", "c-c", eager=True)
    def _(event):
        app.exit(result=0)

    @kb.add("c-x", "p", eager=True)
    def _(event):
        if state.overlay or state.running:
            return
        overlay_open("palette", pick_values())

    @kb.add("c-x", "m", eager=True)
    def _(event):
        if state.overlay or state.running:
            return
        overlay_open("picker", [(i, l) for i, l in load_model_items()])

    @kb.add("c-x", "t", eager=True)
    def _(event):
        if state.overlay or state.running:
            return
        overlay_open("theme", [(n, n) for n in T.list_themes()])

    @kb.add("c-x", "?", eager=True)
    def _(event):
        if state.overlay or state.running:
            return
        help_buf.set_document(Document(_help_text()), bypass_readonly=True)
        state.overlay = "help"
        app.layout.focus(filter_win)

    @kb.add("escape", filter=overlay_open_f, eager=True)
    def _(event):
        if state.overlay == "confirm" and state.confirm_event is not None:
            state.confirm_result = False
            state.confirm_event.set()
        overlay_close()

    @kb.add("up", filter=overlay_open_f, eager=True)
    def _(event):
        lst = state.lists.get(state.overlay or "")
        if lst is not None:
            lst.move(-1)
            _render_list()

    @kb.add("down", filter=overlay_open_f, eager=True)
    def _(event):
        lst = state.lists.get(state.overlay or "")
        if lst is not None:
            lst.move(1)
            _render_list()

    # y/n only while the confirm dialog is up
    @kb.add("y", filter=Condition(lambda: state.overlay == "confirm"), eager=True)
    def _(event):
        if state.confirm_event is not None:
            state.confirm_result = True
            state.confirm_event.set()
        overlay_close()

    @kb.add("n", filter=Condition(lambda: state.overlay == "confirm"), eager=True)
    def _(event):
        if state.confirm_event is not None:
            state.confirm_result = False
            state.confirm_event.set()
        overlay_close()

    @filter_buf.on_text_changed.add_handler
    def _on_filter(_buf):
        lst = state.lists.get(state.overlay or "")
        if lst is not None:
            lst.filter = filter_buf.text
            lst.index = 0
            _render_list()

    @kb.add("enter", filter=Condition(
        lambda: state.overlay in ("palette", "picker", "theme")
        and app.layout.has_focus(filter_win)), eager=True)
    def _(event):
        overlay_choose()

    # -- layout ------------------------------------------------------------------
    def _box(body, title=None, active=None, width=None, height=None, pad=1):
        """Rounded frame (╭╮╰╯). `active()` switches to the accent border."""
        def border():
            return "class:box.border.active" if (active and active()) else "class:box.border"
        top = [Window(width=1, height=1, char="╭", style=border),
               Window(width=1, height=1, char="─", style=border)]
        if title is not None:
            top.append(Window(FormattedTextControl(
                lambda: [("class:box.title", " %s " % (title() if callable(title) else title))]),
                height=1, dont_extend_width=True))
        top += [Window(height=1, char="─", style=border),
                Window(width=1, height=1, char="╮", style=border)]
        side = lambda: Window(width=1, char="│", style=border)
        mid = VSplit([side(), Window(width=pad), body, Window(width=pad), side()])
        bottom = VSplit([Window(width=1, height=1, char="╰", style=border),
                         Window(height=1, char="─", style=border),
                         Window(width=1, height=1, char="╯", style=border)])
        return HSplit([VSplit(top, height=1), mid, VSplit([bottom], height=1)],
                      width=width, height=height)

    class _Placeholder(Processor):
        def apply_transformation(self, ti):
            if ti.lineno == 0 and not ti.document.text:
                hint = ("running… ctrl+c to stop" if state.running else
                        "Ask the model, or type / for commands")
                return Transformation(list(ti.fragments) + [("class:placeholder", hint)])
            return Transformation(ti.fragments)

    input_focused_f = Condition(lambda: app.layout.has_focus(input_win))
    input_win = Window(
        BufferControl(input_buf,
                      input_processors=[BeforeInput(G_ECHO + " ", style="class:prompt"),
                                        _Placeholder(),
                                        HighlightMatchingBracketProcessor()],
                      include_default_input_processors=False),
        # Exactly as tall as the text (1..8 lines): the transcript keeps the
        # spare space instead of an empty stretched input box.
        height=lambda: Dimension.exact(min(8, max(1, input_buf.document.line_count))),
        wrap_lines=True, style="class:input")
    transcript_win = Window(BufferControl(
        transcript_buf, input_processors=[TranscriptHighlighter()]),
        wrap_lines=True)
    slash_win = Window(
        FormattedTextControl(lambda: menu_fragments(
            [("/" + n, C.COMMANDS[n][1]) for n, _ in state.slash_items],
            state.slash_index)),
        height=lambda: Dimension(preferred=min(11, max(1, len(state.slash_items)
                                                        + (1 if len(state.slash_items) > 10 else 0)))),
        wrap_lines=False)

    filter_win = Window(BufferControl(
        filter_buf, input_processors=[BeforeInput(G_ECHO + " ", style="class:prompt")]),
        height=Dimension.exact(1), dont_extend_height=True)

    def _overlay_rows():
        lst = state.lists.get(state.overlay or "")
        if lst is None:
            return []
        import re as _re
        rows = []
        for val, label in lst.filtered():
            parts = _re.split(r"\s{2,}", str(label).strip(), maxsplit=1)
            name = parts[0]
            desc = parts[1] if len(parts) > 1 else ""
            if state.overlay == "theme" and val == T.current_name():
                desc = (desc + " current").strip()
            rows.append((name, desc))
        idx = lst.index % max(1, len(rows))
        return menu_fragments(rows, idx, name_width=28 if state.overlay == "picker" else 12,
                              max_rows=12)

    list_win = Window(FormattedTextControl(_overlay_rows), wrap_lines=False)
    help_win = Window(FormattedTextControl(help_fragments), wrap_lines=False)
    confirm_win = Window(FormattedTextControl(lambda: [
        ("class:value", state.confirm_prompt or ""), ("", "\n\n"),
        ("class:mrow.sel.name", " y "), ("class:key", " yes    "),
        ("class:mrow.name", " n "), ("class:key", " no    "),
        ("class:key", "esc cancels · default no")]), wrap_lines=True)

    def _overlay_title():
        return {"palette": "commands", "picker": "models  ·  Enter = set default + reboot",
                "theme": "theme  ·  Enter applies now", "help": "keys",
                "confirm": "confirm"}.get(state.overlay or "", "")

    def _overlay_body(kind):
        if kind in ("palette", "picker", "theme"):
            return HSplit([filter_win,
                           Window(height=1, char="─", style="class:box.border"),
                           list_win])
        if kind == "help":
            return help_win
        return confirm_win

    def _float(kind, w=72, h=18):
        return Float(
            content=ConditionalContainer(
                _box(_overlay_body(kind), title=_overlay_title,
                     active=lambda: True, width=w, height=h),
                filter=Condition(lambda k=kind: state.overlay == k)),
            top=2)

    def _term_cols():
        try:
            size = app.output.get_size()
            return getattr(size, "columns", getattr(size, "cols", 200))
        except Exception:
            return 200

    panel_win = Window(FormattedTextControl(lambda: panel_fragments(ctx, state, width=36)),
                       wrap_lines=False)
    status_win = Window(FormattedTextControl(
        lambda: status_fragments(ctx, state, _term_cols())),
        height=1, style="class:status")

    body = FloatContainer(
        HSplit([
            VSplit([
                VSplit([Window(width=1), transcript_win, Window(width=1)]),
                ConditionalContainer(
                    _box(panel_win, title="ops", width=40),
                    filter=Condition(lambda: _term_cols() >= 100)),
            ]),
            ConditionalContainer(
                _box(slash_win, title="commands  ·  tab/enter insert  ·  esc hide",
                     active=lambda: True),
                filter=Condition(lambda: state.slash_visible)),
            _box(input_win, active=lambda: input_focused_f() and not state.running),
            status_win,
        ]),
        floats=[
            _float("palette"),
            _float("picker", w=84),
            _float("theme", w=48, h=12),
            _float("help", h=22),
            _float("confirm", w=60, h=7),
        ],
    )

    app = Application(
        layout=Layout(body, focused_element=input_win),
        key_bindings=kb,
        style=Style.from_dict(T.build_style()),
        full_screen=True,
        input=app_input,
        output=app_output,
    )

    def _live_style():
        app.style = Style.from_dict(T.build_style())
        app.invalidate()
    ctx.apply_style = _live_style

    from types import SimpleNamespace as _NS
    return _NS(app=app, state=state, input_buf=input_buf,
               slash_buf=slash_buf, say=say, submit_text=submit_text,
               slash_update=slash_update, render_panel=_render_panel,
               slash_complete_sel=slash_complete_sel)


async def _tui_main(args, ctx, proc=None, boot=None):
    """Full-screen session. Either proc (already booted) or boot=(cmd, env)
    to spawn the backend in the background after the TUI is on screen."""
    import asyncio as _asyncio
    log_q = queue.Queue()
    loop = _asyncio.get_running_loop()
    ui = _build_ui(args, ctx, proc, log_q, loop)
    app, state, say = ui.app, ui.state, ui.say

    # -- log pump ---------------------------------------------------------------
    pump_state = {"stop": None}

    def _start_pump(p):
        # NOTE: logs NEVER go to the transcript buffer (they would bury
        # chat and command output). They live in ctx.logbuf (right panel
        # + /export) only.
        def _sink(line):
            ctx.logbuf.append(line if line.endswith("\n") else line + "\n")
            ctx.transcript.append(line.rstrip("\n"))
        stop_event = threading.Event()
        pump_state["stop"] = stop_event
        pump = threading.Thread(target=C._pump_logs, args=(p, stop_event, _sink),
                                daemon=True)
        pump.start()
        say("[console] node running (pid %d). Logs stream above as [node]." % p.pid)

    async def _drain():
        last_bar = 0.0
        while True:
            await asyncio.sleep(0.1)
            moved = False
            try:
                ui.render_panel()
                moved = True
            except Exception:
                pass
            now = time.monotonic()
            if moved or now - last_bar > 5:
                last_bar = now
                try:
                    app.invalidate()
                except Exception:
                    pass

    async def _boot_task(boot):
        """Spawn the backend behind the already-visible TUI."""
        cmd, env = boot
        say("[console] starting node…")
        try:
            p = await loop.run_in_executor(
                None, partial(_spawn_backend, cmd, env))
        except Exception as e:
            say("[console] cannot start backend: %s" % (e,))
            state.exit_code = 1
            app.exit(result=1)
            return
        state.proc = p
        ctx.backend_pid = p.pid
        ctx.backend_proc = p
        _start_pump(p)

    # -- run -------------------------------------------------------------------------
    say(_banner_text(args, ctx))
    tasks = [asyncio.ensure_future(_drain())]
    if proc is not None:
        _start_pump(proc)
    elif boot is not None:
        tasks.append(asyncio.ensure_future(_boot_task(boot)))
    try:
        await app.run_async()
    finally:
        for t in tasks:
            t.cancel()
    return state.exit_code


def _banner_text(args, ctx):
    from synapse.helpers import VERSION
    return banner_lines(VERSION, getattr(ctx, "model_id", None) or "?",
                        getattr(ctx, "port", "?"), T.current_name())


HELP_GROUPS = (
    ("INPUT", (("enter", "send (plain text chats, /cmd runs)"),
               ("alt+enter  ctrl+j", "new line"),
               ("up / down", "history (saved across sessions)"),
               ("/", "command menu · tab/enter insert · esc hide"),
               ("!cmd", "run a local shell command"))),
    ("OVERLAYS", (("ctrl+p", "command palette"),
                  ("ctrl+x m", "model picker (sets default, reboot)"),
                  ("ctrl+x t", "theme picker (applies now)"),
                  ("ctrl+x ?", "this help"),
                  ("type · up/down · esc", "filter · move · close"))),
    ("SESSION", (("ctrl+c", "stop the running command (or quit when idle)"),
                 ("ctrl+x ctrl+c  ctrl+d", "quit (stops the node)"),
                 ("ctrl+l", "redraw the screen"))),
)


def help_fragments(key_width=24):
    """Grouped, colored key reference for the help overlay (pure)."""
    out = []
    for gi, (title, rows) in enumerate(HELP_GROUPS):
        if gi:
            out.append(("", "\n"))
        out += [("class:section", title), ("", "\n")]
        for k, d in rows:
            out += [("class:mrow.name", "  " + _pad(k, key_width)),
                    ("class:value", d), ("", "\n")]
    if out and out[-1] == ("", "\n"):
        out.pop()
    return out


def _help_text():
    rows = [(k, v) for k, v in C.KEYS_HELP]
    rows += [(s, d) for s, d in LEADER_SEQS]
    rows += [("alt+enter / ctrl+j", "newline in the input box"),
             ("enter", "send (clears the box)"),
             ("tab/enter in slash menu", "insert the highlighted /command"),
             ("esc in slash menu", "hide the menu"),
             ("up/down in overlay", "move selection"),
             ("typing in overlay", "filters the list"),
             ("esc", "close overlay")]
    width = max(len(k) for k, _ in rows)
    return "\n".join("  %-*s  %s" % (width, k, v) for k, v in rows)


def _spawn_backend(cmd, env):
    """Blocking Popen for the boot task (runs in an executor). May raise."""
    import subprocess as _sp
    return _sp.Popen(cmd, env=env, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                     text=True, bufsize=1)


def run_tui(args, ctx, proc=None, boot=None):
    """Enter the full-screen console. Returns the exit code.

    proc = already-booted backend; boot = (cmd, env) to spawn it behind
    the visible TUI. run_console passes boot so the theme shows first.
    """
    code = asyncio.run(_tui_main(args, ctx, proc, boot))
    C._save_history()
    print("[console] stopping node...")
    target = proc or getattr(ctx, "backend_proc", None)
    if target is not None:
        C._stop_node(target)
    return code
