"""
Interactive console (Claude-style SHELL only, no agent). Implements UI.md phase 1.

`synapse console` boots the node (same backend argv as `serve`) in a
background subprocess, streams its logs above a pinned bottom status bar,
and offers a `/slash` REPL on top of the existing deterministic CLI commands.

Rules (interface, not intelligence):
- Every output line is a real command result or a real node log line.
  Nothing is generated, guessed, or summarized by a model.
- Plain text without a leading slash is sent to the model as chat.
  A bad /command prints the hint, never acts.
- Ctrl+C once interrupts the running command and keeps the session;
  Ctrl+C at an empty prompt (or /quit) stops the node and exits 0.
- History (up/down) persists in ~/.synapse/console_history.
- Status bar: pinned bottom line via plain ANSI codes on a real TTY;
  falls back to a status line printed before each prompt on dumb pipes.
"""
from __future__ import annotations

import argparse
import datetime
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import redirect_stdout

from synapse.cli import (
    DEFAULT_PORT,
    Cancelled,
    CliHttpError,
    abort_inflight as _abort_http,
    api_base,
    cmd_benchmark,
    cmd_client,
    cmd_models,
    cmd_status,
    cmd_topology,
    http_get_json,
    translate_serve,
    wait_interruptible,
    BACKEND_MAIN,
    _backend_env,
)

PROMPT = "synapse> "
HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".synapse", "console_history")
HISTORY_LENGTH = 500
LOGBUF_SIZE = 200
BAR_TICK_SECONDS = 5
SHELL_TIMEOUT = 30

# name -> (arg usage, one-line description). Order = listing order.
COMMANDS = {
    "status": ("[--json] [--port N]", "node + cluster health"),
    "topology": ("[--json]", "partition + latency map"),
    "models": ("[list|pull|rm|use] [id]", "model inventory (use = set default + reboot)"),
    "benchmark": ("[--max-tokens N] [--reps N] [--failover]", "measure the live cluster"),
    "client": ("<text...>", "one-shot chat with the mesh"),
    "export": ("[file]", "write transcript to a markdown file"),
    "bug": ("", "write a diagnostics bundle to a file (never sends)"),
    "set": ("<key> <value>", "save a default (applies next boot)"),
    "config": ("", "effective values + where each comes from"),
    "unset": ("<key>", "clear a saved default"),
    "logs": ("[on|off]", "toggle [node] log streaming (buffered, never lost)"),
    "clear": ("", "clear the screen (backend/history/defaults untouched)"),
    "doctor": ("", "machine checkup: ports, API, versions, TLS, cache"),
    "theme": ("[name]", "color theme (synapse, ember-dark, ember-light)"),
    "help": ("", "this command table"),
    "quit": ("", "stop the node and leave (alias: /exit)"),
    "exit": ("", "same as /quit"),
}

KEYS_HELP = [
    ("Enter", "send the command"),
    ("Alt+Enter / Ctrl+J", "newline in the input box (TUI)"),
    ("Up/Down", "command history (saved to file)"),
    ("Tab", "complete slash commands"),
    ("?", "this shortcut table (empty prompt)"),
    ("Ctrl+R", "reverse history search (readline)"),
    ("Ctrl+C (in command)", "interrupt the running command, keep session"),
    ("Ctrl+C (at prompt) / /quit", "stop the node, exit 0"),
    ("Ctrl+L", "terminal-native redraw (use /clear to wipe)"),
    ("Ctrl+P", "command palette (TUI)"),
    ("Ctrl+X P/M/T/?", "palette / models / theme / help (TUI leader)"),
    ("Ctrl+X Ctrl+C", "quit (TUI leader)"),
    ("Esc", "close overlay (TUI); readline cannot catch a lone Esc — use Ctrl+C instead"),
]

HELP_TEXT = """synapse console — chat-first (deterministic, no agent):
  <plain text>                     chat with the mesh (like /client)
%s
Shortcuts: type ? at an empty prompt. Shell escape: !<cmd>.""" % "\n".join(
    "  /%-10s %-32s %s" % (n, u, d) for n, (u, d) in COMMANDS.items()
)


def parse_slash(line: str):
    """Parse one REPL line -> (name, argv) or (None, None) for blank,
    or ("unknown", line) for non-slash input."""
    text = (line or "").strip()
    if not text:
        return None, None
    if not text.startswith("/"):
        return "unknown", text
    try:
        parts = shlex.split(text[1:])
    except ValueError:
        return "unknown", text
    if not parts:
        return None, None
    return parts[0].lower(), parts[1:]


def suggest_commands(frag: str):
    """Fuzzy (case-insensitive substring) match over command names.

    startswith hits first, then other substring hits, alphabetical.
    """
    frag = (frag or "").lower()
    if not frag:
        return list(COMMANDS)
    starts = [n for n in COMMANDS if n.startswith(frag)]
    rest = [n for n in COMMANDS if frag in n and n not in starts]
    return starts + sorted(rest)


def resolve_line(line: str):
    """Pure REPL line classifier (no I/O). Returns an action tuple:

    ("blank", None) | ("keys", None) | ("shell", cmd) |
    ("list", frag) | ("run", name, argv) |
    ("suggest", name, argv) | ("pick", [names], argv) |
    ("chat", text) | ("unknown", text)

    Plain text (no leading slash) is CHAT (sent to the model like
    /client); only a bad `/frag` is unknown.
    """
    text = (line or "").strip()
    if not text:
        return ("blank", None)
    if text == "?":
        return ("keys", None)
    if text.startswith("!"):
        cmd = text[1:].strip()
        return ("shell", cmd) if cmd else ("unknown", text)
    if text.startswith("//"):
        return ("list", text[2:].strip().lower())
    if text == "/":
        return ("list", "")
    if text.startswith("/"):
        name, argv = parse_slash(text)
        if name == "unknown" or name is None:
            return ("blank", None) if name is None else ("unknown", text)
        if name in COMMANDS:
            return ("run", name, argv)
        starts = [n for n in COMMANDS if n.startswith(name)]
        if len(starts) == 1:
            return ("run", starts[0], argv)  # single prefix hit runs directly
        cands = suggest_commands(name)
        if len(cands) == 1:
            return ("suggest", cands[0], argv)
        if cands:
            return ("pick", cands, argv)
        return ("unknown", text)
    return ("chat", text)


def confirm(prompt: str) -> bool:
    """Hard y/N question, default N. Returns True only on y/yes."""
    try:
        ans = input("%s [y/N] " % prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def _is_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _ansi_ok() -> bool:
    if not _is_tty():
        return False
    term = (os.getenv("TERM") or "").lower()
    return term != "dumb"


def _print_table(headers, rows):
    """rich table on a TTY, aligned plain text otherwise (pipe-safe)."""
    rows = [[str(c) for c in r] for r in rows]
    if _is_tty():
        try:
            from rich.table import Table
            from rich.console import Console
            t = Table()
            for h in headers:
                t.add_column(h)
            for r in rows:
                t.add_row(*r)
            Console().print(t)
            return
        except Exception:
            pass
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*headers))
    for r in rows:
        print(fmt.format(*r))


def _setup_history():
    try:
        import readline
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        try:
            readline.read_history_file(HISTORY_FILE)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        try:
            readline.set_history_length(HISTORY_LENGTH)
        except Exception:
            pass
        try:
            readline.parse_and_bind("tab: complete")
            readline.set_completer(_complete)
        except Exception:
            pass
    except Exception:
        pass


def _save_history():
    try:
        import readline
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass


def _complete(text, state):
    try:
        import readline
        buf = readline.get_line_buffer()
    except Exception:
        return None
    if not buf.startswith("/"):
        return None
    opts = ["/" + n for n in COMMANDS if ("/" + n).startswith(text)]
    try:
        return opts[state]
    except IndexError:
        return None


def _ns(**kw):
    """argparse-like namespace for direct command calls (testable)."""
    base = {"port": None, "json": False, "headless": False,
            "log_format": "text", "model": None, "prompt": None,
            "interactive": False, "temperature": None, "timeout": 300,
            "action": "list", "model_id": None, "max_tokens": 64,
            "reps": 5, "failover": False, "cancel_event": None,
            "pipe_output": False}
    base.update(kw)
    return argparse.Namespace(**base)


class ConsoleCtx:
    """Runtime state for the interactive session (transcript, log buffer)."""

    def __init__(self, port, defaults_file, node_label="node",
                 model_short="?", modes="", backend_pid=None, node_port=None,
                 model_id=None):
        self.port = port
        self.defaults_file = defaults_file
        self.node_label = node_label
        self.model_short = model_short
        self.modes = modes
        self.backend_pid = backend_pid
        self.node_port = node_port
        self.model_id = model_id
        self.transcript = []
        self.logbuf = deque(maxlen=LOGBUF_SIZE)
        self.logs_on = True
        self.apply_style = None  # TUI sets this for live /theme apply
        self.job_event = None  # threading.Event of the running command (Ctrl+C)


# Subprocesses started by !shell, tracked so Ctrl+C can kill them.
_ACTIVE_PROCS = set()
_ACTIVE_PROCS_LOCK = threading.Lock()


def request_cancel(ctx=None):
    """Ctrl+C: stop the running console job. Thread-safe, never raises.

    Sets the job event, closes in-flight HTTP connections, and kills
    !shell subprocesses. At most one command runs at a time, so the
    globals below are unambiguous.
    """
    try:
        if ctx is not None and getattr(ctx, "job_event", None) is not None:
            ctx.job_event.set()
    except Exception:
        pass
    try:
        # late import so tests/tools can patch synapse.cli.abort_inflight
        from synapse.cli import abort_inflight
        abort_inflight()
    except Exception:
        pass
    try:
        with _ACTIVE_PROCS_LOCK:
            procs = list(_ACTIVE_PROCS)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
    except Exception:
        pass


def _shell_blocked(cmd: str):
    """Minimal hard denylist. Returns the reason or None."""
    s = (cmd or "").strip()
    if re.match(r"rm\s+.*-[A-Za-z]*r[A-Za-z]*f", s) and \
            re.search(r"(^|\s)/(\*?(\s|$))", s):
        return "rm -rf /"
    m = re.match(r"(mkfs|dd)(\s|$)", s)
    if m:
        return m.group(1)
    return None


def run_shell(cmd: str, timeout: int = SHELL_TIMEOUT, cancel_event=None):
    """Run a local shell command, print output inline. Returns exit code.

    125 = blocked by denylist, 124 = timeout (killed),
    130 = cancelled by Ctrl+C (killed), 127 = spawn error.
    """
    blocked = _shell_blocked(cmd)
    if blocked:
        print("[shell] blocked hard: %s (minimal denylist)" % blocked)
        return 125
    try:
        proc = subprocess.Popen(cmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
    except Exception as e:
        print("[shell] spawn error: %s" % e)
        return 127
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.add(proc)
    try:
        outcome, _rc = wait_interruptible(proc, cancel_event, timeout=timeout)
    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:
            pass
        outcome = "cancelled"
    finally:
        with _ACTIVE_PROCS_LOCK:
            _ACTIVE_PROCS.discard(proc)
    if outcome == "cancelled":
        try:
            proc.kill()
        except Exception:
            pass
        print("[shell] ngắt bởi Ctrl+C — đã kill")
        return 130
    if outcome == "timeout":
        print("[shell] exceeded %ds — killed" % timeout)
        return 124
    try:
        out, err = proc.communicate(timeout=5)
    except Exception:
        out, err = "", ""
    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
    if err:
        sys.stdout.write(err if err.endswith("\n") else err + "\n")
    print("[exit %d]" % proc.returncode)
    return proc.returncode


def _short_model(model_id):
    if not model_id:
        return "?"
    short = str(model_id).split("/")[-1]
    return short[:20]


def fetch_bar_data(port, timeout=2):
    """Best-effort status probe for the status bar. Never raises.

    Returns dict(node, peers) or None when the API is unreachable.
    """
    base = api_base(port)
    try:
        health = http_get_json(base, "/healthcheck", timeout=timeout)
    except Exception:
        return None
    node = None
    if isinstance(health, dict):
        node = health.get("node_id") or health.get("node")
    peers = "?"
    try:
        dist = http_get_json(base, "/v1/distributed/status", timeout=timeout)
        if isinstance(dist, dict):
            for key in ("peers", "nodes", "members", "cluster", "hosts"):
                v = dist.get(key)
                if isinstance(v, (list, tuple, dict, set)):
                    peers = str(len(v))
                    break
                if isinstance(v, int):
                    peers = str(v)
                    break
    except Exception:
        pass
    return {"node": node, "peers": peers}


def build_bar_line(node_label, peers, model_short, modes):
    line = "%s | peers %s | %s" % (node_label, peers, model_short)
    if modes:
        line += " | " + modes
    return line[:80]


class StatusBar:
    """Pinned bottom status line via plain ANSI (no alt-screen, no new deps).

    Screen order bottom-up is [status][prompt]. Log lines are inserted
    ABOVE the bar; the prompt line (with the readline buffer) is re-echoed
    after each insert. Display may flicker on very dumb terminals but the
    typed buffer is never lost (readline owns it). Non-TTY/TERM=dumb:
    disabled — the loop prints a plain status line before each prompt.
    """

    def __init__(self, ctx: ConsoleCtx):
        self.ctx = ctx
        self.enabled = _ansi_ok()
        self.idle = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.line = ""
        self.refresh_data()

    def refresh_data(self):
        data = fetch_bar_data(self.ctx.port)
        node = (data.get("node") if data else None) or self.ctx.node_label
        peers = data.get("peers") if data else "?"
        with self._lock:
            self.line = build_bar_line(node, peers, self.ctx.model_short,
                                       self.ctx.modes)

    def print_bar(self):
        """Fresh last line (cursor is on an empty line, e.g. after a command)."""
        if not self.enabled:
            return
        with self._lock:
            sys.stdout.write(self.line + "\n")
            sys.stdout.flush()

    def _prompt_echo(self):
        try:
            import readline
            buf = readline.get_line_buffer()
        except Exception:
            buf = ""
        sys.stdout.write(PROMPT + buf)
        sys.stdout.flush()

    def emit_above(self, text):
        """Insert log lines above the bar; re-echo bar + prompt buffer."""
        if not self.enabled or not self.idle:
            sys.stdout.write(text if text.endswith("\n") else text + "\n")
            sys.stdout.flush()
            return
        with self._lock:
            sys.stdout.write("\r\x1b[2K")
            for ln in str(text).splitlines() or [""]:
                sys.stdout.write(ln + "\n")
            sys.stdout.write(self.line + "\n")
            sys.stdout.write("\r\x1b[2K")
            self._prompt_echo_locked()

    def _prompt_echo_locked(self):
        try:
            import readline
            buf = readline.get_line_buffer()
        except Exception:
            buf = ""
        sys.stdout.write(PROMPT + buf)
        sys.stdout.flush()

    def redraw_if_idle(self):
        """5s tick: rewrite the bar in place without scrolling."""
        if not self.enabled or not self.idle:
            return
        with self._lock:
            sys.stdout.write("\x1b[s\x1b[1B\r\x1b[2K" + self.line +
                             "\x1b[u\r\x1b[2K")
            self._prompt_echo_locked()

    def start_ticker(self):
        if not self.enabled:
            return
        def _tick():
            while not self._stop.wait(BAR_TICK_SECONDS):
                try:
                    self.refresh_data()
                    self.redraw_if_idle()
                except Exception:
                    pass
        self._thread = threading.Thread(target=_tick, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


def _command_table_rows(frag=""):
    names = suggest_commands(frag) if frag else list(COMMANDS)
    return [("/" + n, COMMANDS[n][0], COMMANDS[n][1]) for n in names]


def cmd_list(frag=""):
    rows = _command_table_rows(frag)
    if not rows:
        print("no command matches — type /help")
        return
    _print_table(["command", "args", "does"], rows)


def cmd_keys():
    _print_table(["key", "does"], KEYS_HELP)


def _default_export_path():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(os.getcwd(), "synapse-export-%s.md" % ts)


def cmd_export(ctx: ConsoleCtx, path=None):
    dest = path or _default_export_path()
    try:
        with open(dest, "w", encoding="utf-8") as f:
            f.write("# synapse console export %s\n\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("- node: %s | model: %s | api port: %s\n\n" % (
                ctx.node_label, ctx.model_short, ctx.port))
            f.write("## transcript\n\n```\n")
            for ln in ctx.transcript:
                f.write(ln if ln.endswith("\n") else ln + "\n")
            f.write("```\n")
    except Exception as e:
        print("[export] cannot write %s: %s" % (dest, e))
        return None
    print("[export] wrote %s (%d lines)" % (dest, len(ctx.transcript)))
    return dest


def _bug_versions():
    from synapse.helpers import VERSION
    return {"synapse": VERSION, "python": sys.version.split()[0],
            "platform": sys.platform}


def cmd_bug(ctx: ConsoleCtx, path=None):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path or os.path.join(os.getcwd(), "synapse-bug-%s.md" % ts)
    try:
        status_doc = fetch_bar_data(ctx.port) or {"api": "unreachable"}
    except Exception as e:
        status_doc = {"api": "error: %s" % e}
    lines = [
        "# synapse bug bundle %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "_This bundle is never sent anywhere automatically — attach it yourself._",
        "",
        "## versions",
        "",
        "```",
    ]
    lines += ["%s: %s" % kv for kv in sorted(_bug_versions().items())]
    lines += ["```", "", "## status probe", "", "```", str(status_doc), "```",
              "", "## ports", "",
              "```",
              "api=%s p2p=%s backend_pid=%s" % (ctx.port, ctx.node_port,
                                               ctx.backend_pid),
              "```", "", "## last %d log lines" % LOGBUF_SIZE, "", "```"]
    lines += list(ctx.logbuf)
    lines += ["```", ""]
    try:
        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(l if l.endswith("\n") else l for l in lines))
    except Exception as e:
        print("[bug] cannot write %s: %s" % (dest, e))
        return None
    print("[bug] wrote %s (NOT sent anywhere — attach it yourself)" % dest)
    return dest


def doctor_checks(port, node_port=None, backend_pid=None, model_id=None):
    """Machine checkup. Returns [(name, ok, detail, fix)]. Never raises."""
    rows = []
    # 1. ports bindable (own backend pid counts as PASS — it holds them)
    wanted = {"api": port}
    if node_port:
        wanted["p2p"] = node_port
    try:
        from synapse.helpers import check_ports_free
        conflicts = check_ports_free(wanted)
        real = [c for c in conflicts
                if not (backend_pid and str(backend_pid) in str(c.get("holder", "")))]
        if not real:
            rows.append(("ports", True,
                         "api=%s p2p=%s free-or-held-by-this-node" % (
                             port, node_port),
                         ""))
        else:
            c = real[0]
            rows.append(("ports", False,
                         "%s:%s held by %s" % (c.get("name"), c.get("port"),
                                               c.get("holder", "?")),
                         "stop the holder or pick another port"))
    except Exception as e:
        rows.append(("ports", False, "probe error: %s" % e,
                     "re-run /doctor"))
    # 2. local API answers
    try:
        http_get_json(api_base(port), "/healthcheck", timeout=5)
        rows.append(("api", True, "GET /healthcheck ok", ""))
    except Exception as e:
        rows.append(("api", False, "unreachable: %s" % e,
                     "is the node still running? check [node] logs"))
    # 3. handshake version agreement
    try:
        from synapse.helpers import VERSION
        info = http_get_json(api_base(port), "/v1/system/info", timeout=5)
        backend_v = info.get("version") if isinstance(info, dict) else None
        if backend_v == VERSION:
            rows.append(("version", True, "console=%s backend=%s" % (VERSION, backend_v), ""))
        else:
            rows.append(("version", False,
                         "console=%s backend=%s" % (VERSION, backend_v),
                         "restart both from the same checkout"))
    except Exception as e:
        rows.append(("version", False, "unknown: %s" % e,
                     "fix the api check first"))
    # 4. TLS mode + cert files
    tls_mode = (os.getenv("SYNAPSE_TLS", "off") or "off").strip().lower()
    if tls_mode == "off":
        rows.append(("tls", True, "off (default, trusted LAN)", ""))
    else:
        cert, key = os.getenv("SYNAPSE_TLS_CERT", ""), os.getenv("SYNAPSE_TLS_KEY", "")
        if cert and key and os.path.isfile(cert) and os.path.isfile(key):
            rows.append(("tls", True, "%s, cert+key present" % tls_mode, ""))
        else:
            rows.append(("tls", False, "%s but cert/key missing" % tls_mode,
                         "set SYNAPSE_TLS_CERT and SYNAPSE_TLS_KEY to files"))
    # 5. default model in local cache (else first request downloads)
    mid = model_id or "?"
    slug = "models--" + str(mid).replace("/", "--")
    roots = [os.getenv("HF_HOME", ""),
             os.getenv("HUGGINGFACE_HUB_CACHE", ""),
             os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
    hit = any(r and os.path.isdir(os.path.join(r, slug)) for r in roots)
    if hit:
        rows.append(("model-cache", True, "%s cached" % mid, ""))
    else:
        rows.append(("model-cache", False, "%s not in local cache" % mid,
                     "run /models pull %s before serving" % mid))
    # 6. discovery module + tailscale CLI
    try:
        from synapse.cli_defaults import effective
        disc = (effective({}, None).get("discovery-module") if True else None) or "udp"
    except Exception:
        disc = "udp"
    if disc == "tailscale":
        if shutil.which("tailscale"):
            rows.append(("discovery", True, "tailscale module + CLI present", ""))
        else:
            rows.append(("discovery", False, "module=tailscale but no CLI",
                         "install tailscale or /set discovery-module udp"))
    elif disc == "udp":
        rows.append(("discovery", True, "udp (stdlib, always available)", ""))
    else:
        rows.append(("discovery", True, "%s (manual config)" % disc, ""))
    return rows


def cmd_doctor(ctx: ConsoleCtx):
    rows = doctor_checks(ctx.port, ctx.node_port, ctx.backend_pid,
                         getattr(ctx, "model_id", None) or ctx.model_short)
    table = [(n, "PASS" if ok else "FAIL",
              detail + ("" if ok and not fix else (" | fix: " + fix if fix else "")))
             for n, ok, detail, fix in rows]
    _print_table(["check", "result", "detail"], table)


def dispatch(name, argv, defaults, ctx=None):
    """Run one slash command. Returns True to keep looping, False to quit."""
    port = defaults.get("port") or DEFAULT_PORT
    rest = [a for a in argv if not a.startswith("--")]
    # One cancellable job at a time: Ctrl+C sets this event (see
    # request_cancel); long commands poll it or die with closed conns.
    job = threading.Event()
    if ctx is not None:
        ctx.job_event = job
    try:
        return _dispatch_inner(name, argv, defaults, ctx, port, rest, job)
    finally:
        if ctx is not None:
            ctx.job_event = None


def _wait_ready(port, cancel_event=None, timeout=120):
    """Wait until the backend API reports inference_ready.

    Returns True when ready, False on timeout/cancel (message printed).
    Ctrl+C sets cancel_event from another thread — polled every second.
    """
    import time as _t
    deadline = _t.monotonic() + timeout
    waited = False
    while True:
        if cancel_event is not None and cancel_event.is_set():
            print("(ngắt — phiên còn)")
            return False
        try:
            health = http_get_json(api_base(port), "/healthcheck", timeout=5)
            if isinstance(health, dict) and health.get("inference_ready"):
                return True
        except Exception:
            pass
        if _t.monotonic() >= deadline:
            print("[fail] backend chưa sẵn sàng sau %ds "
                  "(đang tải model? xem panel logs, rồi thử lại)" % timeout)
            return False
        if not waited:
            print("[console] chờ backend sẵn sàng… (Ctrl+C để ngắt)")
            waited = True
        _t.sleep(1)


def _dispatch_inner(name, argv, defaults, ctx, port, rest, job):
    """Returns (keep_looping, exit_code)."""
    if name in ("quit", "exit"):
        return False, 0
    if name == "help":
        cmd_list()
        print("shortcuts: type ? at an empty prompt | shell: !<cmd>")
        return True, 0
    if name == "status":
        rc = cmd_status(_ns(port=port, json="--json" in argv, cancel_event=job))
        return True, rc
    if name == "topology":
        rc = cmd_topology(_ns(port=port, json="--json" in argv, cancel_event=job))
        return True, rc
    if name == "models":
        action, model_id = "list", None
        if rest:
            action = rest[0]
        if len(rest) > 1:
            model_id = rest[1]
        if action not in ("list", "pull", "rm", "use"):
            print("usage: /models [list|pull|rm|use] [id]")
            return True, 2
        if action in ("pull", "rm", "use") and not model_id:
            print("usage: /models %s MODEL_ID" % action)
            return True, 2
        if action in ("pull", "rm", "use"):
            if not confirm("/models %s %s — apply?" % (action, model_id or "")):
                print("(cancelled — default N)")
                return True, 0
        if action == "use":
            from synapse.cli_defaults import set_key
            ok, msg = set_key("model", model_id,
                              defaults.get("defaults_file"))
            print(("[ok] " if ok else "[fail] ") + str(msg))
            print("model switches on node reboot only (no hot-swap) — "
                  "restart serve/join to take effect.")
            return True, 0 if ok else 1
        rc = cmd_models(_ns(port=port, json="--json" in argv, action=action,
                            model_id=model_id, cancel_event=job))
        return True, rc
    if name == "benchmark":
        # Effective model (not "default"): the backend 400s unknown names.
        eff_model = None
        if ctx is not None:
            eff_model = getattr(ctx, "model_id", None)
        if not _wait_ready(port, job):
            return True, 3
        kw = {"port": port, "json": "--json" in argv,
              "failover": "--failover" in argv, "model": eff_model,
              "prompt": None, "max_tokens": 64, "reps": 5,
              "cancel_event": job,
              "pipe_output": bool(ctx is not None
                                  and getattr(ctx, "has_panel", False))}
        i = 0
        while i < len(argv):
            if argv[i] == "--max-tokens" and i + 1 < len(argv):
                kw["max_tokens"] = int(argv[i + 1])
                i += 2
            elif argv[i] == "--reps" and i + 1 < len(argv):
                kw["reps"] = int(argv[i + 1])
                i += 2
            elif argv[i] == "--model" and i + 1 < len(argv):
                kw["model"] = argv[i + 1]
                i += 2
            else:
                i += 1
        rc = cmd_benchmark(_ns(**kw))
        return True, rc
    if name == "client":
        if not rest:
            print("usage: /client <text...>")
            return True, 2
        if not _wait_ready(port, job):
            return True, 3
        rc = cmd_client(_ns(port=port, model=None, prompt=" ".join(rest),
                            interactive=False, temperature=None, timeout=300,
                            json=False, cancel_event=job))
        return True, rc
    if name == "set":
        from synapse.cli_defaults import set_key
        if len(rest) < 2:
            print("usage: /set <key> <value...>  (keys: model, node-id, node-port, discovery-module, config, wait-for-peers, partition-method, temperature, chat-port, ilp-time-limit-seconds)")
            return True, 2
        ok, msg = set_key(rest[0], " ".join(rest[1:]), defaults.get("defaults_file"))
        print(("[ok] " if ok else "[fail] ") + str(msg))
        return True, 0 if ok else 1
    if name == "unset":
        from synapse.cli_defaults import unset_key
        if len(rest) != 1:
            print("usage: /unset <key>")
            return True, 2
        from synapse.cli_defaults import effective
        try:
            cur = effective({}, defaults.get("defaults_file")).get(rest[0])
        except Exception:
            cur = None
        if cur is not None and not confirm("/unset %s (now %r) — clear?" % (rest[0], cur)):
            print("(cancelled — default N)")
            return True, 0
        ok, msg = unset_key(rest[0], defaults.get("defaults_file"))
        print(("[ok] " if ok else "[fail] ") + str(msg))
        return True, 0 if ok else 1
    if name == "config":
        from synapse.cli_defaults import (
            VALID_KEYS, describe_source, effective)
        eff = effective({}, defaults.get("defaults_file"))
        rows = [(k, repr(eff.get(k)),
                 "[%s]" % describe_source(k, {}, defaults.get("defaults_file")))
                for k in sorted(VALID_KEYS)]
        _print_table(["key", "value", "source"], rows)
        return True, 0
    if name == "logs":
        if ctx is None:
            print("(only inside an interactive console)")
            return True, 2
        if rest and rest[0] not in ("on", "off"):
            print("usage: /logs [on|off]")
            return True, 2
        want = rest[0] == "on" if rest else (not ctx.logs_on)
        if want and not ctx.logs_on:
            ctx.logs_on = True
            if getattr(ctx, "has_panel", False):
                # TUI: logs live in the right panel, never in the transcript.
                print("[logs] on — panel streaming")
            else:
                for ln in list(ctx.logbuf):
                    sys.stdout.write(ln if ln.endswith("\n") else ln + "\n")
                sys.stdout.flush()
                print("[logs] on — buffer flushed, streaming")
        elif not want and ctx.logs_on:
            ctx.logs_on = False
            print("[logs] off — buffering (nothing lost, /logs on to flush)")
        else:
            print("[logs] already %s" % ("on" if ctx.logs_on else "off"))
        return True, 0
    if name == "clear":
        if os.name == "nt":
            os.system("cls")
        else:
            if os.system("clear") != 0:
                print("\n" * 50)
        return True, 0
    if name == "doctor":
        if ctx is None:
            print("(only inside an interactive console)")
            return True, 2
        cmd_doctor(ctx)
        return True, 0
    if name == "export":
        if ctx is None:
            print("(only inside an interactive console)")
            return True, 2
        dest = cmd_export(ctx, rest[0] if rest else None)
        return True, 0 if dest else 1
    if name == "bug":
        if ctx is None:
            print("(only inside an interactive console)")
            return True, 2
        dest = cmd_bug(ctx)
        return True, 0 if dest else 1
    if name == "theme":
        from synapse import theme as _theme
        if not rest:
            _print_table(["theme", "note"],
                         [(n, "current" if n == _theme.current_name() else "")
                          for n in _theme.list_themes()])
            return True, 0
        ok, msg = _theme.set_theme(rest[0])
        print(("[ok] " if ok else "[fail] ") + str(msg))
        if ok and ctx is not None and getattr(ctx, "apply_style", None):
            try:
                ctx.apply_style()
            except Exception:
                pass
        return True, 0 if ok else 1
    print("unknown command '/%s' — type /help" % name)
    return True, 2


def _pump_logs(proc, stop_event, sink=None):
    """Stream backend stdout, prefixed. Ends with proc.

    sink(line) receives each "[node] ..." line when given (console mode:
    records transcript + ring buffer + bar-aware display). Without a sink
    the lines go straight to stdout (legacy behavior).
    """
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            text = "[node] " + line
            if sink is not None:
                try:
                    sink(text)
                except Exception:
                    pass
            else:
                sys.stdout.write(text)
                sys.stdout.flush()
    except Exception:
        pass
    finally:
        stop_event.set()


def _stop_node(proc):
    """Terminate the backend subprocess. Never raises."""
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _want_tui(args) -> bool:
    """Full-screen TUI when possible; line mode otherwise.

    Forced off by --line, SYNAPSE_LINE=1, missing prompt_toolkit,
    or a non-TTY stdout (pipes, tests, dumb terminals).
    """
    if os.getenv("SYNAPSE_LINE", "") == "1":
        return False
    if getattr(args, "line", False):
        return False
    if not _is_tty():
        return False
    try:
        import prompt_toolkit  # noqa: F401
    except Exception:
        return False
    return True


def card_header_line(label):
    """Line-mode card header (same glyphs as the TUI, font-safe)."""
    return "▸ %s" % label


def card_footer(rc, dt):
    """Line-mode card footer: `● done · 1.2s` / `× exit 3 · 0.1s`.

    ✔/✘ were used before; Consolas (legacy console font) has neither and
    drew tofu boxes.
    """
    return ("● done · %.1fs" % dt) if rc == 0 else ("× exit %s · %.1fs" % (rc, dt))


def _banner(args, eff_model):
    from synapse.helpers import VERSION
    p2p = args.node_port if args.node_port is not None else "backend default"
    rows = [
        "  ● synapse console %s · deterministic, no agent" % VERSION,
        "    model   %s" % eff_model,
        "    api     127.0.0.1:%s  ·  p2p %s" % (args.chat_port or DEFAULT_PORT, p2p),
        "    try     /status   /topology   /help   ·   /quit exits cleanly",
    ]
    width = max(60, max(len(r) for r in rows) + 2)
    print("╭" + "─" * width + "╮")
    for r in rows:
        print("│" + r.ljust(width) + "│")
    print("╰" + "─" * width + "╯")


def run_console(args):
    """Boot the node (like serve) + slash REPL. Returns exit code."""
    _setup_history()
    from synapse.cli_defaults import apply_to_namespace, effective
    from synapse.helpers import VERSION  # noqa: F401 (banner via _banner)
    eff = effective({}, None)
    eff_model = getattr(args, "model", None) or eff.get("model") or "?"
    use_tui = _want_tui(args)
    if not use_tui:
        _banner(args, eff_model)

    argv = translate_serve(args)
    cmd = [sys.executable, BACKEND_MAIN] + argv
    env = _backend_env(headless=args.headless)

    modes = "".join([
        "[adv]" if (os.getenv("SYNAPSE_AUTOSCALE", "") in ("advise", "on")) else "",
        "[tls]" if (os.getenv("SYNAPSE_TLS", "off") != "off") else "",
        "[headless]" if getattr(args, "headless", False) else "",
    ])
    ctx = ConsoleCtx(
        port=args.chat_port or DEFAULT_PORT,
        defaults_file=os.getenv("SYNAPSE_DEFAULTS_PATH") or None,
        node_label=getattr(args, "node_id", None) or "node",
        model_short=_short_model(eff_model),
        modes=modes,
        backend_pid=None,
        node_port=getattr(args, "node_port", None),
    )
    try:
        from synapse.cli_defaults import defaults_path
        if ctx.defaults_file is None:
            ctx.defaults_file = str(defaults_path())
    except Exception:
        pass
    ctx.model_id = eff_model

    if use_tui:
        try:
            from synapse.tui import run_tui
            return run_tui(args, ctx, None, boot=(cmd, env))
        except Exception as e:
            print("[console] TUI unavailable (%s) — line mode" % e)

    print("[console] starting node: %s" % " ".join(cmd[1:]))
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
    except Exception as e:
        print("[console] cannot start backend: %s" % (e,), file=sys.stderr)
        return 1
    ctx.backend_pid = proc.pid

    bar = StatusBar(ctx)
    bar.start_ticker()

    def _sink(line):
        ctx.logbuf.append(line)
        ctx.transcript.append(line.rstrip("\n"))
        if ctx.logs_on:
            bar.emit_above(line.rstrip("\n"))
    stop_event = threading.Event()
    pump = threading.Thread(target=_pump_logs, args=(proc, stop_event, _sink),
                            daemon=True)
    pump.start()
    print("[console] node running (pid %d). Logs stream above as [node]." % proc.pid)
    bar.print_bar()

    defaults = {"port": args.chat_port or DEFAULT_PORT,
                "defaults_file": ctx.defaults_file}
    code = 0
    pending = None  # (candidates, argv) after an ambiguous /frag
    try:
        while True:
            if not bar.enabled:
                sys.stdout.write(bar.line + "\n")
                sys.stdout.flush()
            bar.idle = True
            try:
                line = input(PROMPT)
            except KeyboardInterrupt:
                print()
                break  # Ctrl+C at prompt: quit cleanly
            except EOFError:
                break
            finally:
                bar.idle = False
            stripped = (line or "").strip()
            ctx.transcript.append("> " + line.rstrip("\n"))
            # pending numbered choice from an earlier ambiguous /frag
            if pending is not None:
                cands, pargv = pending
                if stripped.isdigit() and 1 <= int(stripped) <= len(cands):
                    pending = None
                    buf = io.StringIO()
                    label = "/" + cands[int(stripped) - 1] + (
                        " " + " ".join(pargv) if pargv else "")
                    t0 = time.monotonic()
                    print(card_header_line(label))
                    try:
                        with redirect_stdout(buf):
                            keep, rc = dispatch(cands[int(stripped) - 1], pargv,
                                                defaults, ctx)
                    except KeyboardInterrupt:
                        request_cancel(ctx)
                        keep, rc = True, 130
                        print("\n(interrupted — session kept, /quit to leave)")
                    out = buf.getvalue()
                    sys.stdout.write(out)
                    ctx.transcript.append(out.rstrip("\n"))
                    print(card_footer(rc, time.monotonic() - t0))
                    if not keep:
                        break
                    bar.refresh_data()
                    bar.print_bar()
                    continue
                pending = None  # anything else drops the pick list
            act = resolve_line(line)
            kind = act[0]
            if kind == "blank":
                continue
            if kind == "keys":
                cmd_keys()
                continue
            if kind == "shell":
                buf = io.StringIO()
                ctx.job_event = threading.Event()
                t0 = time.monotonic()
                print(card_header_line("!" + act[1]))
                try:
                    with redirect_stdout(buf):
                        rc = run_shell(act[1], cancel_event=ctx.job_event)
                except KeyboardInterrupt:
                    request_cancel(ctx)
                    rc = 130
                    print("\n(interrupted — session kept, /quit to leave)")
                finally:
                    ctx.job_event = None
                out = buf.getvalue()
                sys.stdout.write(out)
                ctx.transcript.append(out.rstrip("\n"))
                print(card_footer(rc, time.monotonic() - t0))
                bar.refresh_data()
                bar.print_bar()
                continue
            if kind == "list":
                cmd_list(act[1])
                continue
            if kind == "suggest":
                print("did you mean /%s? (retype the full command)" % act[1])
                continue
            if kind == "pick":
                _print_table(["#", "command", "does"],
                             [(str(i + 1), "/" + n, COMMANDS[n][1])
                              for i, n in enumerate(act[1])])
                print("type the number to run (anything else cancels)")
                pending = (act[1], act[2])
                continue
            if kind == "unknown":
                print("unknown command '%s' — type /help" % line.strip())
                continue
            if kind == "chat":
                # chat-first: plain text goes to the model like /client.
                act = ("run", "client", [act[1]])
                kind = "run"
            # ("run", name, argv)
            buf = io.StringIO()
            label = "/" + act[1] + (" " + " ".join(act[2]) if act[2] else "")
            t0 = time.monotonic()
            print(card_header_line(label))
            try:
                with redirect_stdout(buf):
                    keep, rc = dispatch(act[1], act[2], defaults, ctx)
            except KeyboardInterrupt:
                request_cancel(ctx)  # kill benchmark child, close HTTP conns
                keep, rc = True, 130
                print("\n(interrupted — session kept, /quit to leave)")
            out = buf.getvalue()
            sys.stdout.write(out)
            if out.strip():
                ctx.transcript.append(out.rstrip("\n"))
            print(card_footer(rc, time.monotonic() - t0))
            if not keep:
                break
            if proc.poll() is not None:
                print("[console] backend exited (code %d)" % proc.returncode)
                code = 0 if proc.returncode == 0 else 1
                break
            bar.refresh_data()
            bar.print_bar()
    finally:
        bar.stop()
        _save_history()
        print("[console] stopping node...")
        _stop_node(proc)
    return code


def main(argv=None):
    """Entry for `synapse console`: serve-equivalent flags + REPL."""
    p = argparse.ArgumentParser(prog="synapse console")
    p.add_argument("--model", default=None)
    p.add_argument("--node-id", default=None)
    p.add_argument("--node-port", type=int, default=None)
    p.add_argument("--discovery-module", choices=("udp", "tailscale", "manual"), default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--join", default=None)
    p.add_argument("--wait-for-peers", type=int, default=None)
    p.add_argument("--partition-method", choices=("lrm", "ilp"), default=None)
    p.add_argument("--ilp-time-limit-seconds", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--chat-port", type=int, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--log-format", choices=("text", "json"), default="text")
    p.add_argument("--line", action="store_true",
                   help="force line-mode REPL instead of the full-screen TUI")
    args = p.parse_args(argv)
    args.tui = False
    args.no_tui = True
    # Zero-flag boot uses the same per-machine defaults file as serve.
    from synapse.cli_defaults import apply_to_namespace
    apply_to_namespace(args)
    return run_console(args)
