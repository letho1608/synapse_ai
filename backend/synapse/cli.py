"""
Synapse CLI-first terminal surface (T2.1 / PLAN.md Part E).

Thin wrappers only — no serving, transport, partition, or engine logic here:
- serve/join: translate flags and supervise the existing backend entry point
  (backend/synapse/main.py) in a subprocess, preserving the restart-on-42
  contract from the root launcher.
- client/status/topology/models/benchmark: urllib over the local HTTP API.
  Read commands print exactly one JSON document on stdout under --json
  (logs and hints go to stderr); exit codes 0/1/2/3/4 per contract.

Explicitly NOT here (see PLAN.md Part E): dashboard, mobile, MoA, plugins,
media generation. --headless disables the TUI only; the GET / liveness stub
and --log-format json for backend logs land with the T2 API slice (dirty
files, tracked as pending — a warning is printed, never silent).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3
EXIT_SERVER = 4

DEFAULT_PORT = 52415
BACKEND_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

COMMANDS = ("serve", "join", "client", "status", "topology", "models", "benchmark")


class CliHttpError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _err(msg):
    print("synapse: %s" % msg, file=sys.stderr)


def _emit_json(doc):
    sys.stdout.write(json.dumps(doc, indent=2) + "\n")
    sys.stdout.flush()


def _fail_json(message, code):
    _emit_json({"ok": False, "error": {"code": code, "message": message}})


def api_base(port):
    return "http://127.0.0.1:%d" % port


def http_get_json(base, path, timeout=15):
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise CliHttpError(EXIT_SERVER, "HTTP %s on %s" % (e.code, url))
    except Exception as e:
        raise CliHttpError(EXIT_UNREACHABLE, "cannot reach %s (%s)" % (url, e))


def http_post_json(base, path, payload, timeout=600):
    url = base.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return json.loads(raw)
            except Exception:
                return {"ok": True, "raw": raw}
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        raise CliHttpError(EXIT_SERVER, "HTTP %s on %s %s" % (e.code, url, detail))
    except Exception as e:
        raise CliHttpError(EXIT_UNREACHABLE, "cannot reach %s (%s)" % (url, e))


from synapse.logging_json import emit as _emit_log
from synapse.logging_json import redact as _redact


def _port_parent():
    """Shared --port accepted before AND after the subcommand.

    default=None on both levels so whichever position the user picks wins;
    main() resolves None -> DEFAULT_PORT.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--port", type=int, default=argparse.SUPPRESS,
                   help="local API port (default %d)" % DEFAULT_PORT)
    return p


def build_parser():
    p = argparse.ArgumentParser(prog="synapse",
                                description="Synapse AI cluster CLI (CLI-first, Part E)")
    p.add_argument("--log-format", choices=("text", "json"), default="text",
                   help="log encoding (json = JSONL on stdout for serve)")
    p.add_argument("--headless", action="store_true",
                   help="disable TUI; API/logs/healthcheck unaffected")
    p.add_argument("--port", type=int, default=None,
                   help="local API port to query (default %d)" % DEFAULT_PORT)
    sub = p.add_subparsers(dest="command", required=True)
    port_parent = _port_parent()

    s = sub.add_parser("serve", help="run a node in the foreground")
    s.add_argument("--model", default=None)
    s.add_argument("--node-id", default=None)
    s.add_argument("--node-port", type=int, default=None)
    s.add_argument("--discovery-module", choices=("udp", "tailscale", "manual"), default=None)
    s.add_argument("--config", default=None, help="cluster config path (manual discovery)")
    s.add_argument("--join", default=None, help="config path or tailscale://<tailnet> (key via TAILSCALE_API_KEY env only)")
    s.add_argument("--wait-for-peers", type=int, default=None)
    s.add_argument("--partition-method", choices=("lrm", "ilp"), default=None)
    s.add_argument("--ilp-time-limit-seconds", type=float, default=None)
    s.add_argument("--temperature", type=float, default=None)
    s.add_argument("--chat-port", type=int, default=None)
    tui = s.add_mutually_exclusive_group()
    tui.add_argument("--tui", action="store_true", help="opt into rich topology TUI (needs TTY)")
    tui.add_argument("--no-tui", action="store_true")

    j = sub.add_parser("join", help="join a mesh without serving a model shard")
    j.add_argument("target", help="cluster config path or tailscale://<tailnet>")
    j.add_argument("--node-id", default=None)
    j.add_argument("--node-port", type=int, default=None)

    c = sub.add_parser("client", parents=[port_parent], help="chat with the mesh (POST /v1/chat/completions)")
    c.add_argument("--model", default=None)
    c.add_argument("--prompt", default=None)
    c.add_argument("--interactive", action="store_true")
    c.add_argument("--temperature", type=float, default=None)
    c.add_argument("--timeout", type=int, default=300)
    c.add_argument("--json", action="store_true", help="single completion document on stdout")

    for name, help_text in (("status", "node + cluster health"),
                            ("topology", "partition + latency view")):
        r = sub.add_parser(name, parents=[port_parent], help=help_text)
        r.add_argument("--json", action="store_true")

    m = sub.add_parser("models", parents=[port_parent], help="model inventory")
    m.add_argument("--json", action="store_true")
    m.add_argument("action", nargs="?", choices=("list", "pull", "rm"), default="list")
    m.add_argument("model_id", nargs="?")

    b = sub.add_parser("benchmark", parents=[port_parent], help="measure the live cluster (C1 metrics)")
    b.add_argument("--model", default=None)
    b.add_argument("--prompt", default=None)
    b.add_argument("--max-tokens", type=int, default=64)
    b.add_argument("--reps", type=int, default=5)
    b.add_argument("--json", action="store_true")
    b.add_argument("--failover", action="store_true")
    return p


def _backend_env(headless=False):
    env = dict(os.environ)
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = backend_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if headless:
        # T2.1: narrows GET / to a 204 stub (handle_root); API/logs/healthcheck+/v1/* unaffected.
        env["SYNAPSE_HEADLESS"] = "1"
    return env


def translate_serve(args):
    """Map serve flags onto the existing backend argv (single source of truth:
    backend/synapse/main.py). Unknown future flags must extend THIS table."""
    argv = ["run"]
    if args.model:
        argv.append(args.model)
    if args.node_id:
        argv += ["--node-id", args.node_id]
    if args.node_port:
        argv += ["--node-port", str(args.node_port)]
    disc = args.discovery_module
    cfg = args.config or (args.join if args.join and not args.join.startswith("tailscale://") else None)
    if args.join and args.join.startswith("tailscale://"):
        disc = disc or "tailscale"
        tailnet = args.join[len("tailscale://"):].split("?")[0].split("/")[0]
        argv += ["--tailnet-name", tailnet]
    if disc:
        argv += ["--discovery-module", disc]
    if cfg:
        argv += ["--discovery-config-path", cfg]
    if args.wait_for_peers is not None:
        argv += ["--wait-for-peers", str(args.wait_for_peers)]
    if args.partition_method:
        argv += ["--partition-method", args.partition_method]
    if args.ilp_time_limit_seconds is not None:
        argv += ["--ilp-time-limit-seconds", str(args.ilp_time_limit_seconds)]
    if args.temperature is not None:
        argv += ["--default-temp", str(args.temperature)]
    if args.chat_port:
        argv += ["--chatgpt-api-port", str(args.chat_port)]
    if args.tui:
        argv += ["--no-disable-tui"]
    elif args.no_tui:
        argv += ["--disable-tui"]
    return argv


def cmd_serve(args):
    if args.log_format == "json":
        _err("note: --log-format json for backend logs lands with the T2 API slice (M4); launcher lines below are JSON")
    argv = translate_serve(args)
    cmd = [sys.executable, BACKEND_MAIN] + argv
    while True:
        if args.log_format == "json":
            _emit_log("INFO", "serve", "starting backend", {"argv": argv})
        else:
            print("[INFO] Starting backend: %s" % " ".join(cmd[1:]))
            print("[INFO] Press Ctrl+C to stop")
        try:
            p = subprocess.Popen(cmd, env=_backend_env(headless=args.headless))
            p.wait()
        except KeyboardInterrupt:
            print("\n[INFO] Shutdown requested by user.", file=sys.stderr)
            return EXIT_OK
        if p.returncode == 42:
            print("[INFO] Auto-restarting to apply new settings...", file=sys.stderr)
            continue
        return EXIT_OK if p.returncode == 0 else EXIT_FAIL


def cmd_join(args):
    if args.target.startswith("tailscale://"):
        tailnet = args.target[len("tailscale://"):].split("?")[0].split("/")[0]
        if "TAILSCALE_API_KEY" not in os.environ and "--tailscale-api-key" not in sys.argv:
            _err("using TAILSCALE_API_KEY from environment (never pass keys on argv)")
        ns = argparse.Namespace(model=None, node_id=args.node_id, node_port=args.node_port,
                                discovery_module="tailscale", config=None,
                                join=args.target, wait_for_peers=None,
                                partition_method=None, ilp_time_limit_seconds=None,
                                temperature=None, chat_port=None, tui=False, no_tui=True,
                                headless=args.headless, log_format=args.log_format)
        _ = tailnet
    else:
        ns = argparse.Namespace(model=None, node_id=args.node_id, node_port=args.node_port,
                                discovery_module="manual", config=args.target,
                                join=None, wait_for_peers=None,
                                partition_method=None, ilp_time_limit_seconds=None,
                                temperature=None, chat_port=None, tui=False, no_tui=True,
                                headless=args.headless, log_format=args.log_format)
    return cmd_serve(ns)


def _stream_chat(base, model, prompt, temperature, timeout):
    """Streaming chat; returns (text, ttft_s, elapsed_s). Tolerant parser."""
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0 if temperature is None else temperature,
                       "stream": True}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    first = None
    parts = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                text = ""
                try:
                    ch = obj["choices"][0]
                    d = ch.get("delta", {}) or {}
                    text = d.get("content") or ch.get("text") or ""
                    if not text and isinstance(ch.get("message"), dict):
                        text = ch["message"].get("content") or ""
                except Exception:
                    text = obj.get("text") or ""
                if text and first is None:
                    first = time.time()
                parts.append(text)
    except urllib.error.HTTPError as e:
        raise CliHttpError(EXIT_SERVER, "HTTP %s on chat completions" % e.code)
    except Exception as e:
        raise CliHttpError(EXIT_UNREACHABLE, "chat request failed (%s)" % e)
    return "".join(parts), ((first - t0) if first else None), time.time() - t0


def cmd_client(args):
    base = api_base(args.port)
    if args.interactive:
        print("synapse interactive (empty line quits). model=%s" % (args.model or "server default"))
        while True:
            try:
                prompt = input("> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            try:
                text, _, _ = _stream_chat(base, args.model, prompt, args.temperature, args.timeout)
                print(text)
            except CliHttpError as e:
                _err(str(e))
                return e.code
        return EXIT_OK
    if not args.prompt:
        _err("client needs --prompt TEXT or --interactive")
        return EXIT_USAGE
    try:
        text, ttft, elapsed = _stream_chat(base, args.model, args.prompt, args.temperature, args.timeout)
    except CliHttpError as e:
        if args.json:
            _fail_json(str(e), e.code)
        else:
            _err(str(e))
        return e.code
    if args.json:
        _emit_json({"ok": True, "model": args.model, "content": text,
                    "ttft_s": ttft, "elapsed_s": elapsed})
    else:
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
    return EXIT_OK


def cmd_status(args):
    base = api_base(args.port)
    try:
        health = http_get_json(base, "/healthcheck")
        dist = http_get_json(base, "/v1/distributed/status")
        info = http_get_json(base, "/v1/system/info")
        resources = http_get_json(base, "/v1/cluster/resources")
    except CliHttpError as e:
        if args.json:
            _fail_json(str(e), e.code)
        else:
            _err(str(e))
        return e.code
    doc = _redact({"ok": True, "health": health, "distributed": dist,
                   "system": info, "cluster_resources": resources})
    if args.json:
        _emit_json(doc)
        return EXIT_OK
    inf_ready = health.get("inference_ready") if isinstance(health, dict) else None
    parts = dist.get("partitions_count") if isinstance(dist, dict) else None
    print("api=%s inference_ready=%s partitions=%s" % (base, inf_ready, parts))
    return EXIT_OK


def cmd_topology(args):
    base = api_base(args.port)
    try:
        topo = http_get_json(base, "/v1/topology")
    except CliHttpError as e:
        if args.json:
            _fail_json(str(e), e.code)
        else:
            _err(str(e))
        return e.code
    nodes = topo.get("nodes", {}) if isinstance(topo, dict) else {}
    if args.json:
        _emit_json(_redact({"ok": True, "topology": topo}))
        return EXIT_OK
    print("nodes=%d" % len(nodes))
    for nid in sorted(nodes):
        print("  - %s" % nid)
    return EXIT_OK


def cmd_models(args):
    base = api_base(args.port)
    action = args.action
    if action == "pull":
        if not args.model_id:
            _err("models pull needs MODEL_ID")
            return EXIT_USAGE
        try:
            res = http_post_json(base, "/v1/models/pull", {"model_id": args.model_id})
        except CliHttpError as e:
            _err(str(e))
            return e.code
        print(json.dumps(res)[:500])
        return EXIT_OK
    if action == "rm":
        if not args.model_id:
            _err("models rm needs MODEL_ID")
            return EXIT_USAGE
        try:
            res = http_post_json(base, "/v1/models/delete", {"model_id": args.model_id})
        except CliHttpError as e:
            _err(str(e))
            return e.code
        print(json.dumps(res)[:500])
        return EXIT_OK
    try:
        models = http_get_json(base, "/v1/models")
    except CliHttpError as e:
        if args.json:
            _fail_json(str(e), e.code)
        else:
            _err(str(e))
        return e.code
    if args.json:
        _emit_json(_redact({"ok": True, "models": models}))
        return EXIT_OK
    items = models.get("data", models) if isinstance(models, dict) else models
    if isinstance(items, list):
        for m in items:
            print("  - %s" % (m.get("id", m) if isinstance(m, dict) else m))
    else:
        print(json.dumps(models)[:1000])
    return EXIT_OK


def cmd_benchmark(args):
    import tempfile
    harness = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "tests", "benchmark_live_cluster.py")
    harness = os.path.abspath(harness)
    out = tempfile.mktemp(suffix=".jsonl", prefix="synapse-bench-")
    cmd = [sys.executable, harness, "--api-urls", "http://127.0.0.1:%d" % args.port,
           "--model", args.model or "default", "--max-tokens", str(args.max_tokens),
           "--reps", str(args.reps), "--out", out, "--allow-loopback"]
    if args.prompt:
        cmd += ["--model", args.model or "default"]
    if args.failover:
        cmd += ["--failover"]
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        return EXIT_OK
    if args.json:
        try:
            with open(out, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            summaries = [l for l in lines if l.get("type") == "summary"]
            _emit_json(summaries[-1] if summaries else {"ok": rc == 0, "out": out})
        except Exception as e:
            _fail_json("cannot read harness output: %s" % e, EXIT_FAIL)
            return EXIT_FAIL
    return EXIT_OK if rc == 0 else EXIT_FAIL


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "port", None) is None:
        args.port = DEFAULT_PORT
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "join":
        return cmd_join(args)
    if args.command == "client":
        return cmd_client(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "topology":
        return cmd_topology(args)
    if args.command == "models":
        return cmd_models(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)
    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
