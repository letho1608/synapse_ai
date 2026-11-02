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
import http.client
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
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


class Cancelled(Exception):
    """A console job was cancelled (Ctrl+C). Never a real failure."""


# -- cancellable transport ------------------------------------------------------
# Every HTTP connection the CLI opens is tracked here so a console Ctrl+C
# can close it from another thread (unblocks a stuck read at once).
# Safe because the console runs at most one command at a time.
_INFLIGHT = set()
_INFLIGHT_LOCK = threading.Lock()


def _track(conn):
    with _INFLIGHT_LOCK:
        _INFLIGHT.add(conn)
    return conn


def _untrack(conn):
    with _INFLIGHT_LOCK:
        _INFLIGHT.discard(conn)


def abort_inflight():
    """Close every in-flight HTTP connection. Thread-safe, never raises."""
    with _INFLIGHT_LOCK:
        conns = list(_INFLIGHT)
        _INFLIGHT.clear()
    for c in conns:
        try:
            c.close()
        except Exception:
            pass


def _split(base, path):
    u = urllib.parse.urlsplit(base.rstrip("/") + path)
    return u


def _conn_for(u, timeout):
    if u.scheme == "https":
        return http.client.HTTPSConnection(u.hostname, u.port or 443,
                                           timeout=timeout)
    return http.client.HTTPConnection(u.hostname, u.port or 80,
                                      timeout=timeout)


def wait_interruptible(proc, cancel_event=None, timeout=None, poll=0.05):
    """Wait for proc; kill it when cancel_event fires or timeout hits.

    Returns ("done", rc) | ("cancelled", rc) | ("timeout", rc).
    Never raises on the happy path; KeyboardInterrupt propagates.
    """
    import time as _t
    deadline = None if timeout is None else _t.monotonic() + timeout
    while True:
        rc = proc.poll()
        if rc is not None:
            return "done", rc
        if cancel_event is not None and cancel_event.is_set():
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                rc = proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    rc = proc.wait(timeout=5)
                except Exception:
                    rc = None
            return "cancelled", rc
        if deadline is not None and _t.monotonic() >= deadline:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                rc = proc.wait(timeout=5)
            except Exception:
                rc = None
            return "timeout", rc
        _t.sleep(poll)


def _err(msg):
    print("synapse: %s" % msg, file=sys.stderr)


def _emit_json(doc):
    sys.stdout.write(json.dumps(doc, indent=2) + "\n")
    sys.stdout.flush()


def _fail_json(message, code):
    _emit_json({"ok": False, "error": {"code": code, "message": message}})


def api_base(port):
    return "http://127.0.0.1:%d" % port


def http_get_json(base, path, timeout=15, cancel_event=None):
    # NOTE: plain http.client (no redirect/proxy handling like urllib).
    # Our API is local HTTP with no redirects, so this is equivalent here.
    u = _split(base, path)
    url = base.rstrip("/") + path
    conn = _track(_conn_for(u, timeout))
    try:
        conn.request("GET", u.path or "/", headers={"Accept": "application/json"})
        r = conn.getresponse()
        status = r.status
        raw = r.read().decode("utf-8", "replace")
    except Exception as e:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("cancelled")
        raise CliHttpError(EXIT_UNREACHABLE, "cannot reach %s (%s)" % (url, e))
    finally:
        _untrack(conn)
        try:
            conn.close()
        except Exception:
            pass
    if status >= 400:
        raise CliHttpError(EXIT_SERVER, "HTTP %s on %s" % (status, url))
    return json.loads(raw)


def http_post_json(base, path, payload, timeout=600, cancel_event=None):
    u = _split(base, path)
    url = base.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    conn = _track(_conn_for(u, timeout))
    try:
        conn.request("POST", u.path or "/",
                     body=body, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        status = r.status
        raw = r.read().decode("utf-8", "replace")
    except Exception as e:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("cancelled")
        raise CliHttpError(EXIT_UNREACHABLE, "cannot reach %s (%s)" % (url, e))
    finally:
        _untrack(conn)
        try:
            conn.close()
        except Exception:
            pass
    if status >= 400:
        raise CliHttpError(EXIT_SERVER, "HTTP %s on %s %s" % (status, url, raw[:500]))
    try:
        return json.loads(raw)
    except Exception:
        return {"ok": True, "raw": raw}


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

    # Listed for --help only: main() routes `console` to synapse.console
    # before argparse ever parses it.
    sub.add_parser("console", add_help=False,
                   help="interactive shell: boot a node + slash-command TUI "
                        "(see `console --help`)")

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

    pr = sub.add_parser("pair", help="pairing code: machines sharing it talk mutual TLS")
    pr.add_argument("--code", default=None, help="join an existing cluster with its code")
    pr.add_argument("--show", action="store_true", help="print this machine's code")
    pr.add_argument("--forget", action="store_true", help="remove the code (plaintext again)")

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
    # Backend exits if this launcher is killed (no orphan holding ports).
    from synapse.helpers import PARENT_PID_ENV
    env[PARENT_PID_ENV] = str(os.getpid())
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
                                temperature=None, chat_port=None,
                                headless=args.headless, log_format=args.log_format)
        _ = tailnet
    else:
        ns = argparse.Namespace(model=None, node_id=args.node_id, node_port=args.node_port,
                                discovery_module="manual", config=args.target,
                                join=None, wait_for_peers=None,
                                partition_method=None, ilp_time_limit_seconds=None,
                                temperature=None, chat_port=None,
                                headless=args.headless, log_format=args.log_format)
    return cmd_serve(ns)


def _stream_chat(base, model, prompt, temperature, timeout, cancel_event=None):
    """Streaming chat; returns (text, ttft_s, elapsed_s). Tolerant parser."""
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0 if temperature is None else temperature,
                       "stream": True}).encode("utf-8")
    base.rstrip("/") + "/v1/chat/completions"
    u = _split(base, "/v1/chat/completions")
    t0 = time.time()
    first = None
    parts = []
    conn = _track(_conn_for(u, timeout))
    try:
        conn.request("POST", u.path or "/",
                     body=body, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status >= 400:
            raise CliHttpError(EXIT_SERVER, "HTTP %s on chat completions" % r.status)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled("cancelled")
            raw = r.readline()
            if not raw:
                break
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
    except Cancelled:
        raise
    except CliHttpError:
        raise
    except Exception as e:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("cancelled")
        raise CliHttpError(EXIT_UNREACHABLE, "chat request failed (%s)" % e)
    finally:
        _untrack(conn)
        try:
            conn.close()
        except Exception:
            pass
    return "".join(parts), ((first - t0) if first else None), time.time() - t0


def cmd_client(args):
    base = api_base(args.port)
    cancel = getattr(args, "cancel_event", None)
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
                text, _, _ = _stream_chat(base, args.model, prompt, args.temperature,
                                          args.timeout, cancel_event=cancel)
                print(text)
            except Cancelled:
                print("(ngắt — phiên còn)")
            except CliHttpError as e:
                _err(str(e))
                return e.code
        return EXIT_OK
    if not args.prompt:
        _err("client needs --prompt TEXT or --interactive")
        return EXIT_USAGE
    try:
        text, ttft, elapsed = _stream_chat(base, args.model, args.prompt, args.temperature,
                                           args.timeout, cancel_event=cancel)
    except Cancelled:
        print("(ngắt — phiên còn)")
        return EXIT_OK
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
    cancel = getattr(args, "cancel_event", None)
    try:
        health = http_get_json(base, "/healthcheck", cancel_event=cancel)
        dist = http_get_json(base, "/v1/distributed/status", cancel_event=cancel)
        info = http_get_json(base, "/v1/system/info", cancel_event=cancel)
        resources = http_get_json(base, "/v1/cluster/resources", cancel_event=cancel)
    except Cancelled:
        print("(ngắt — phiên còn)")
        return EXIT_OK
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
    cancel = getattr(args, "cancel_event", None)
    try:
        topo = http_get_json(base, "/v1/topology", cancel_event=cancel)
    except Cancelled:
        print("(ngắt — phiên còn)")
        return EXIT_OK
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
    cancel = getattr(args, "cancel_event", None)
    action = args.action
    if action == "pull":
        if not args.model_id:
            _err("models pull needs MODEL_ID")
            return EXIT_USAGE
        try:
            res = http_post_json(base, "/v1/models/pull", {"model_id": args.model_id},
                                 cancel_event=cancel)
        except Cancelled:
            # client-side stop only; the server may still finish the pull.
            print("(ngắt — server có thể vẫn pull nốt phía sau)")
            return EXIT_OK
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
            res = http_post_json(base, "/v1/models/delete", {"model_id": args.model_id},
                                 cancel_event=cancel)
        except Cancelled:
            print("(ngắt — phiên còn)")
            return EXIT_OK
        except CliHttpError as e:
            _err(str(e))
            return e.code
        print(json.dumps(res)[:500])
        return EXIT_OK
    try:
        models = http_get_json(base, "/v1/models", cancel_event=cancel)
    except Cancelled:
        print("(ngắt — phiên còn)")
        return EXIT_OK
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
        cmd += ["--prompt", args.prompt]
    if args.failover:
        cmd += ["--failover"]
    # Console TUI: a child inheriting the terminal would paint over the
    # full-screen app. Pipe it and print at the end instead (goes through
    # the Tee into the transcript card). Line mode keeps live streaming.
    pipe = bool(getattr(args, "pipe_output", False))
    popen_kw = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                "text": True} if pipe else {}
    try:
        proc = subprocess.Popen(cmd, **popen_kw)
    except KeyboardInterrupt:
        print("(ngắt benchmark — phiên còn)")
        return EXIT_OK
    except Exception as e:
        _err("cannot start benchmark harness (%s)" % e)
        return EXIT_FAIL
    try:
        outcome, rc = wait_interruptible(
            proc, getattr(args, "cancel_event", None))
        if pipe:
            try:
                out, _ = proc.communicate(timeout=5)
            except Exception:
                out = ""
            if out.strip():
                print(out.rstrip("\n"))
    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:
            pass
        print("(ngắt benchmark — phiên còn)")
        return EXIT_OK
    if outcome == "cancelled":
        print("(ngắt benchmark — phiên còn)")
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


def _split_console(argv):
    """Return console argv when `console` follows only global flags, else None.

    Global flags map onto console's own: --headless / --log-format pass
    through, --port N becomes --chat-port N unless the console args set it.
    """
    i, glob = 0, []
    while i < len(argv):
        a = argv[i]
        if a == "console":
            rest = list(argv[i + 1:])
            out = []
            for j, g in enumerate(glob):
                if g == "--port":
                    if "--chat-port" not in rest:
                        out += ["--chat-port", glob[j + 1]]
                elif g == "--log-format":
                    out += [g, glob[j + 1]]
                elif g == "--headless":
                    out.append(g)
            return out + rest
        if a in ("--port", "--log-format") and i + 1 < len(argv):
            glob += [a, argv[i + 1]]
            i += 2
            continue
        if a == "--headless":
            glob.append(a)
            i += 1
            continue
        return None
    return None


def cmd_pair(args):
    """Create, join, show or forget this machine's pairing code."""
    from synapse.transport import tls_wrap
    if args.forget:
        try:
            os.remove(tls_wrap.code_path())
            print("pairing code removed: this node talks plaintext again (restart it)")
        except FileNotFoundError:
            print("no pairing code on this machine")
        return EXIT_OK
    try:
        current = tls_wrap.cluster_code()
    except ValueError as e:
        current = None
        _err("saved pairing code is malformed (%s); replacing it" % e)
    if args.show:
        if not current:
            _err("no pairing code yet: run `synapse pair`")
            return EXIT_FAIL
        print(current)
        return EXIT_OK
    if args.code:
        try:
            code = tls_wrap.normalize_code(args.code)
        except ValueError as e:
            _err(str(e))
            return EXIT_USAGE
        tls_wrap.save_cluster_code(code)
        print("paired: this machine joins the cluster of code %s...  (restart the node)" % code[:9])
        return EXIT_OK
    if current:
        print(current)
        print("(existing code; give it to the other machines: synapse pair --code <code>)")
        return EXIT_OK
    code = tls_wrap.new_cluster_code()
    tls_wrap.save_cluster_code(code)
    print(code)
    print("on every other machine run:  synapse pair --code %s" % code)
    print("nodes with the code talk mutual TLS; nodes without it cannot connect")
    return EXIT_OK


def main(argv=None):
    # Console is a separate entry (serve flags + REPL); route before argparse
    # so `synapse console ...` never collides with subcommand parsing.
    console_argv = _split_console(list(sys.argv[1:] if argv is None else argv))
    if console_argv is not None:
        from synapse.console import main as console_main
        return console_main(console_argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "port", None) is None:
        args.port = DEFAULT_PORT
    if args.command in ("serve", "join"):
        # Zero-flag boot: per-machine defaults file fills the gaps
        # (explicit flags always win). See synapse/cli_defaults.py.
        from synapse.cli_defaults import apply_to_namespace
        apply_to_namespace(args)
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
    if args.command == "pair":
        return cmd_pair(args)
    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
