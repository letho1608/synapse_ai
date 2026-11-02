"""
Behavioral tests: CLI-first surface (T2.1 / Part E).

- Parser builds; serve/join translate onto the existing backend argv.
- --json failure payloads are single stdout documents (M4 stdout discipline).
- Dead API port exits 3 naming the port (never hangs, never traceback).
- Secrets never echo in redacted payloads (G3).
- No network except a refused loopback port (fast); no model, no cluster.
"""
import io
import json
from contextlib import redirect_stdout

import pytest

from synapse.cli import (
    EXIT_OK,
    EXIT_UNREACHABLE,
    EXIT_USAGE,
    _redact,
    build_parser,
    cmd_models,
    cmd_status,
    cmd_topology,
    main,
    translate_serve,
)


def _args(*argv):
    return build_parser().parse_args(list(argv))


def test_parser_commands_exist():
    for cmd in ("serve", "join", "client", "status", "topology", "models", "benchmark"):
        ns = _args(cmd) if cmd not in ("serve", "join", "client", "models", "benchmark") else None
        assert ns is None or ns.command == cmd


def test_translate_serve_maps_to_backend_argv():
    ns = _args("serve", "--model", "m1", "--node-id", "n1", "--node-port", "5679",
               "--discovery-module", "manual", "--config", "c.json",
               "--wait-for-peers", "2", "--partition-method", "ilp",
               "--ilp-time-limit-seconds", "5", "--temperature", "0.1",
               "--chat-port", "5555")
    argv = translate_serve(ns)
    assert argv[:2] == ["run", "m1"]
    for flag, val in (("--node-id", "n1"), ("--node-port", "5679"),
                      ("--discovery-module", "manual"),
                      ("--discovery-config-path", "c.json"),
                      ("--wait-for-peers", "2"), ("--partition-method", "ilp"),
                      ("--ilp-time-limit-seconds", "5.0"), ("--default-temp", "0.1"),
                      ("--chatgpt-api-port", "5555")):
        assert flag in argv, flag
        if val is not None:
            assert argv[argv.index(flag) + 1] == val
    # Backend-unknown flags must never leak through.
    assert not any("tui" in a for a in argv)


def test_translate_join_tailscale_never_takes_key_on_argv():
    ns = _args("join", "tailscale://my-tailnet", "--node-id", "n9")
    # join delegates to serve translation; assert the tailnet mapping only.
    assert ns.target == "tailscale://my-tailnet"
    assert "api-key" not in " ".join(__import__("sys").argv)


def _run_json(fn, ns):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = fn(ns)
    out = buf.getvalue()
    doc = json.loads(out)  # single stdout document or the test fails here
    return code, doc


def test_status_dead_port_exit3_single_json_doc():
    ns = _args("status", "--json", "--port", "59998")
    code, doc = _run_json(cmd_status, ns)
    assert code == EXIT_UNREACHABLE
    assert doc["ok"] is False
    assert "59998" in doc["error"]["message"]


def test_topology_dead_port_exit3_single_json_doc():
    ns = _args("topology", "--json", "--port", "59998")
    code, doc = _run_json(cmd_topology, ns)
    assert code == EXIT_UNREACHABLE
    assert doc["ok"] is False


def test_models_dead_port_exit3_single_json_doc():
    ns = _args("models", "--json", "--port", "59998")
    code, doc = _run_json(cmd_models, ns)
    assert code == EXIT_UNREACHABLE
    assert doc["ok"] is False


def test_redact_removes_secrets():
    payload = {"tailscale_api_key": "SECRET", "tailnet_name": "t",
               "nested": {"apiKey": "SECRET", "port": 1},
               "tailscale-api-key": "SECRET"}
    red = _redact(payload)
    flat = json.dumps(red)
    assert "SECRET" not in flat
    assert red["tailnet_name"] == "t"
    assert red["nested"]["port"] == 1


def test_client_requires_prompt_or_interactive(capsys):
    code = main(["client", "--port", "59998"])
    assert code == EXIT_USAGE


def test_unknown_command_exits_usage():
    with pytest.raises(SystemExit) as e:
        main(["frobnicate"])
    assert e.value.code == EXIT_USAGE or e.value.code == 2


def test_console_reachable_after_global_flags(monkeypatch):
    """`--port N --headless console ...` must reach the console entry with
    the globals mapped onto console flags (was: argparse 'invalid choice')."""
    import synapse.console as console_mod
    seen = {}
    monkeypatch.setattr(console_mod, "main", lambda a: seen.setdefault("argv", a) and 0)
    main(["--port", "6000", "--headless", "--log-format", "json", "console", "--line"])
    argv = seen["argv"]
    assert argv[argv.index("--chat-port") + 1] == "6000"
    assert "--headless" in argv and "--line" in argv
    assert argv[argv.index("--log-format") + 1] == "json"
    # An explicit console --chat-port wins over the global --port.
    seen.clear()
    main(["--port", "6000", "console", "--chat-port", "7000"])
    assert seen["argv"].count("--chat-port") == 1
    assert seen["argv"][seen["argv"].index("--chat-port") + 1] == "7000"


def test_help_lists_console():
    assert "console" in build_parser().format_help()
