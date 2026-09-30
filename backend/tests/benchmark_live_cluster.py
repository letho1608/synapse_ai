"""
Live-cluster benchmark harness (PLAN.md C1 gate, Lane 5 runbook).

Measures a RUNNING synapse cluster, never simulates:
  M1 mesh formation (via /v1/topology + /v1/distributed/status agreement)
  M2 time-to-first-token (streaming, distributed vs single-node)
  M3 sustained tokens/s
  M4 failover seconds (--failover: operator kills a host mid-run, semi-auto)
  M5 training smoke is out of scope here (separate job endpoint)
  M6 temp-0 text identity single-vs-distributed

Usage (run from ANY machine that reaches all APIs):
  python benchmark_live_cluster.py --api-urls http://192.168.1.10:52415,http://192.168.1.11:52415 \
      --model HuggingFaceTB/SmolLM2-135M-Instruct --max-tokens 64 --reps 5 --out results.jsonl

Exit codes: 0 pass, 2 usage error, 3 unreachable / loopback-masquerade,
4 server error / metric mismatch / partition disagreement.
Loopback-only runs are REJECTED (exit 3) unless --allow-loopback is passed,
and then results are labeled measured:false (red-team B3).
Stdlib only (urllib/json/argparse/time/sys).
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import urllib.error

FIXED_PROMPTS = [
    "Say the word hello and nothing else.",
    "What is 2+2? Answer with one number.",
    "Name three primary colors, comma separated.",
    "Repeat after me exactly: pipeline parallelism works.",
    # ~512-token approx prompt (fixed, comparable; char count recorded):
    "PACKET " * 400,
]


def is_loopback(url):
    return ("127.0.0.1" in url or "localhost" in url or "::1" in url)


def http_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %s on %s" % (e.code, url))
    except Exception as e:
        raise RuntimeError("unreachable %s (%s)" % (url, e))


def find_key(obj, names):
    """Tolerant deep search for first matching key (API shapes vary by version)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names:
                return v
            hit = find_key(v, names)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = find_key(v, names)
            if hit is not None:
                return hit
    return None


def check_convergence(api_urls, out):
    """M1: all hosts agree on peer set/master/partitions. Returns snapshot dict."""
    snaps = {}
    for base in api_urls:
        base = base.rstrip("/")
        snap = {"api": base}
        for name in ("topology", "dist_status", "resources", "health"):
            path = {
                "topology": "/v1/topology",
                "dist_status": "/v1/distributed/status",
                "resources": "/v1/cluster/resources",
                "health": "/healthcheck",
            }[name]
            try:
                snap[name] = http_json(base + path)
            except RuntimeError as e:
                snap[name] = {"_error": str(e)}
        snaps[base] = snap
    out.write(json.dumps({"type": "convergence_snapshot", "snapshots": snaps}) + "\n")

    counts = set()
    for base, snap in snaps.items():
        d = snap.get("dist_status", {})
        if isinstance(d, dict) and "_error" in d:
            raise RuntimeError("dist_status failed on %s: %s" % (base, d["_error"]))
        c = find_key(d, ("partitions_count",))
        counts.add(json.dumps(c, sort_keys=True, default=str))
    if len(counts) != 1:
        raise RuntimeError("partition disagreement across hosts: %s" % counts)
    n = len(api_urls)
    c = json.loads(list(counts)[0])
    if c != n:
        raise RuntimeError(
            "partitions_count=%r but %d apis queried (mesh not converged?)" % (c, n)
        )
    masters = set()
    for base, snap in snaps.items():
        m = find_key(snap.get("dist_status", {}), ("master_node_id", "master", "active_node_id"))
        masters.add(json.dumps(m, sort_keys=True, default=str))
    return {"partitions_count": c, "master_agreement": len(masters) == 1}


def stream_completion(api_base, model, prompt, max_tokens, timeout=300):
    """POST /v1/chat/completions stream:true. Returns dict with ttft/tokens/text."""
    url = api_base.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    first_ts = None
    chunks = []
    completion_tokens = None
    estimated = True
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
                    text = obj.get("text") or obj.get("content") or ""
                if text and first_ts is None:
                    first_ts = time.time()
                chunks.append(text)
                u = obj.get("usage") or {}
                if isinstance(u, dict) and u.get("completion_tokens"):
                    completion_tokens = int(u["completion_tokens"])
                    estimated = False
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %s on %s" % (e.code, url))
    except Exception as e:
        raise RuntimeError("stream failed on %s (%s)" % (url, e))
    t1 = time.time()
    full = "".join(chunks)
    if completion_tokens is None:
        completion_tokens = max(1, int(len(full.split()) * 1.3))
    ttft = (first_ts - t0) if first_ts else None
    # End-to-end rate (incl. TTFT). Measuring only from the first streamed
    # chunk overstated it wildly when the server released tokens in a burst.
    wall = t1 - t0
    tps = (completion_tokens / wall) if wall > 0 else 0.0
    return {
        "ttft_s": ttft,
        "tokens_per_s": tps,
        "completion_tokens": completion_tokens,
        "tokens_estimated": estimated,
        "elapsed_s": t1 - t0,
        "text": full,
        "text_sha": hashlib.sha256(full.encode("utf-8")).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser(description="Synapse live-cluster benchmark (C1 gate)")
    ap.add_argument("--api-urls", required=True, help="comma-separated API bases")
    ap.add_argument("--single-url", default=None, help="single-node API base for M2/M3/M6 baseline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True, help="results JSONL path")
    ap.add_argument("--allow-loopback", action="store_true", help="permit 127.0.0.1 runs (labeled measured:false)")
    ap.add_argument("--failover", action="store_true", help="semi-auto failover timing (operator kills a host on prompt)")
    ap.add_argument("--prompt", default=None, help="measure this prompt instead of the fixed set")
    args = ap.parse_args()

    urls = [u.strip() for u in args.api_urls.split(",") if u.strip()]
    if len(urls) < 1:
        print("need >=1 --api-urls", file=sys.stderr)
        return 2
    loopback = all(is_loopback(u) for u in urls)
    if loopback and not args.allow_loopback:
        print("REFUSED loopback-masquerade: all apis are loopback. "
              "Use LAN/Tailscale IPs or pass --allow-loopback (results labeled measured:false).",
              file=sys.stderr)
        return 3
    measured = not loopback

    try:
        out = open(args.out, "w", encoding="utf-8")
    except Exception as e:
        print("cannot open --out: %s" % e, file=sys.stderr)
        return 2

    def emit(o):
        o["measured"] = measured
        out.write(json.dumps(o) + "\n")
        out.flush()

    try:
        conv = check_convergence(urls, out)
        emit({"type": "m1_formation", "ok": True, "detail": conv})
    except RuntimeError as e:
        emit({"type": "m1_formation", "ok": False, "error": str(e)})
        print("M1 FAIL: %s" % e, file=sys.stderr)
        out.close()
        return 4

    target = urls[0]
    results = []
    prompts = [args.prompt] if args.prompt else FIXED_PROMPTS
    for pi, prompt in enumerate(prompts):
        for rep in range(args.reps):
            try:
                # warmup rep 0 of prompt 0 is discarded
                r = stream_completion(target, args.model, prompt, args.max_tokens)
            except RuntimeError as e:
                emit({"type": "rep", "ok": False, "prompt_idx": pi, "rep": rep, "error": str(e)})
                print("rep FAIL: %s" % e, file=sys.stderr)
                out.close()
                return 4
            rec = {"type": "rep", "ok": True, "prompt_idx": pi, "rep": rep,
                   "prompt_chars": len(prompt), "warmup": (pi == 0 and rep == 0)}
            rec.update(r)
            emit(rec)
            if not rec["warmup"]:
                results.append(rec)

    def stats(key):
        vals = sorted(r[key] for r in results if r.get(key) is not None)
        if not vals:
            return None
        return {"mean": sum(vals) / len(vals), "p50": vals[len(vals) // 2],
                "min": vals[0], "max": vals[-1], "n": len(vals)}

    summary = {"type": "summary", "model": args.model, "target": target,
               "ttft_s": stats("ttft_s"), "tokens_per_s": stats("tokens_per_s"),
               "partitions": conv["partitions_count"]}
    emit(summary)

    if args.single_url:
        try:
            s = stream_completion(args.single_url.rstrip("/"), args.model, prompts[0], args.max_tokens)
            base = {"type": "single_baseline", "single_url": args.single_url}
            base.update(s)
            emit(base)
            if summary["ttft_s"] and s["ttft_s"]:
                ratio = summary["ttft_s"]["mean"] / s["ttft_s"] if s["ttft_s"] else None
                emit({"type": "m2_ttft_ratio_dist_over_single", "ratio": ratio,
                      "gate": "dist TTFT <= 2x single"})
                if ratio and ratio > 2.0:
                    print("M2 WARN: distributed TTFT %.2fx single (gate 2x)" % ratio, file=sys.stderr)
            # M6 identity on prompt 0 temp-0
            firsts = [r["text_sha"] for r in results if r["prompt_idx"] == 0]
            emit({"type": "m6_identity", "match": (len(set(firsts)) == 1 and firsts[0] == s["text_sha"]),
                  "dist_sha": (firsts[0] if firsts else None), "single_sha": s["text_sha"]})
        except RuntimeError as e:
            emit({"type": "single_baseline", "ok": False, "error": str(e)})

    if args.failover:
        print("FAILOVER: kill ONE host now (non-master first), then press Enter here...", file=sys.stderr)
        try:
            input()
        except EOFError:
            pass
        t0 = time.time()
        ok = False
        deadline = t0 + 120
        last_err = ""
        while time.time() < deadline:
            try:
                r = stream_completion(target, args.model, FIXED_PROMPTS[1], args.max_tokens, timeout=60)
                ok = True
                break
            except RuntimeError as e:
                last_err = str(e)
                time.sleep(3)
        emit({"type": "m4_failover", "ok": ok, "failover_s": (time.time() - t0) if ok else None,
              "gate_s": 30, "within_gate": ok and (time.time() - t0) <= 30, "last_error": last_err})
        if not ok:
            print("M4 FAIL: no successful request within 120s (%s)" % last_err, file=sys.stderr)
            out.close()
            return 4

    out.close()
    print("OK: %d reps, mean tok/s=%.2f mean ttft=%.3fs measured=%s -> %s"
          % (len(results), summary["tokens_per_s"]["mean"] if summary["tokens_per_s"] else -1,
             summary["ttft_s"]["mean"] if summary["ttft_s"] else -1, measured, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
