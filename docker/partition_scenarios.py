"""Network partitions on a 3-container cluster (docker/docker-compose.yml, profile "three").

  python docker/partition_scenarios.py

Cuts one node off from the other two (netem loss 100% on its interface), then
heals the link. Reports, per node, which master it follows and how many nodes
hold layers, whether each side still answers a chat, and how long the cluster
takes after healing to agree on ONE master with all three nodes holding layers.

Status is read with `docker exec` inside each container: a cut node's API is
unreachable from the host too.
"""
import json
import subprocess
import sys
import time

COMPOSE = ["docker", "compose", "-f", "docker/docker-compose.yml", "--profile", "three"]
NODES = {"A": "synapse-a", "B": "synapse-b", "C": "synapse-c"}
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

PROBE = r"""
import json, sys, urllib.request
def get(p):
    with urllib.request.urlopen("http://127.0.0.1:52415" + p, timeout=8) as r:
        return json.loads(r.read())
try:
    d = get("/v1/distributed/status")
    print(json.dumps({"master": d.get("master_id"), "parts": d.get("partitions_count")}))
except Exception as e:
    print(json.dumps({"error": type(e).__name__}))
"""

CHAT = r"""
import json, sys, time, urllib.request
body = json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "What is the capital of France?"}],
                   "max_tokens": 12, "temperature": 0}).encode()
t = time.time()
try:
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:52415/v1/chat/completions", body,
                                {"Content-Type": "application/json"}), timeout=150) as r:
        d = json.loads(r.read())
    print(json.dumps({"s": round(time.time() - t, 1), "text": d["choices"][0]["message"]["content"]}))
except Exception as e:
    print(json.dumps({"s": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"[:120]}))
"""


def sh(*args, check=True, timeout=None):
    return subprocess.run(list(args), capture_output=True, text=True, check=check, timeout=timeout).stdout


def status(node):
    try:
        out = sh("docker", "exec", NODES[node], "python", "-c", PROBE, check=False, timeout=20).strip()
        return json.loads(out.splitlines()[-1]) if out else {"error": "no output"}
    except Exception as e:
        return {"error": type(e).__name__}


def chat(node):
    out = sh("docker", "exec", NODES[node], "python", "-c", CHAT, MODEL, check=False, timeout=200).strip()
    return json.loads(out.splitlines()[-1]) if out else {"error": "no output"}


def snapshot():
    return {n: status(n) for n in NODES}


def fmt(snap):
    return "  ".join(f"{n}: master={s.get('master', '?')} parts={s.get('parts', '?')}" if "error" not in s
                     else f"{n}: {s['error']}" for n, s in snap.items())


def cut(node):
    sh("docker", "exec", NODES[node], "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "loss", "100%")


def heal(node):
    sh("docker", "exec", NODES[node], "tc", "qdisc", "del", "dev", "eth0", "root", check=False)


def converged(snap):
    masters = {s.get("master") for s in snap.values()}
    return len(masters) == 1 and None not in masters and all(s.get("parts") == 3 for s in snap.values())


def wait_converged(limit, label):
    t0 = time.time()
    stable = 0
    snap = {}
    while time.time() - t0 < limit:
        snap = snapshot()
        stable = stable + 1 if converged(snap) else 0
        if stable >= 3:  # three agreeing polls in a row
            print(f"   {label}: converged after {time.time() - t0:.0f}s -> {fmt(snap)}", flush=True)
            return True, time.time() - t0
        time.sleep(2)
    print(f"   {label}: NOT converged after {limit}s -> {fmt(snap)}", flush=True)
    return False, None


def log_lines(since):
    out = []
    for n, c in NODES.items():
        for line in sh("docker", "logs", "--since", since, c, check=False).splitlines():
            if any(k in line for k in ("[FAULT]", "[FAILOVER]", "electing new Master", "recognizes", "still Master")):
                out.append(f"{n} | {line.strip()[:170]}")
    return out


def scenario(name, isolate):
    print(f"\n== {name}: cut {isolate} off from the others", flush=True)
    since = sh("docker", "exec", "synapse-a", "date", "-u", "+%Y-%m-%dT%H:%M:%SZ").strip()
    print("   before:", fmt(snapshot()), flush=True)
    cut(isolate)
    others = [n for n in NODES if n != isolate]
    time.sleep(45)  # heartbeat timeouts + master grace period
    print("   split (45s):", fmt(snapshot()), flush=True)
    for n in (others[0], isolate):
        print(f"   chat on {n} while split:", chat(n), flush=True)
    heal(isolate)
    ok, secs = wait_converged(240, "after heal")
    print(f"   chat on {isolate} after heal:", chat(isolate), flush=True)
    for l in log_lines(since)[-14:]:
        print("    |", l)
    return ok, secs


def main():
    print("starting 3 containers ...", flush=True)
    sh(*COMPOSE, "down", check=False)
    sh(*COMPOSE, "up", "-d")
    ok, secs = wait_converged(300, "initial formation")
    if not ok:
        sys.exit("cluster never formed with 3 nodes")
    results = {}
    results["non-master cut"] = scenario("1. non-master node isolated", "C")
    master = snapshot()["A"].get("master") or "A"
    results["master cut"] = scenario(f"2. master node ({master}) isolated", master)
    print("\n== summary")
    for k, (ok, secs) in results.items():
        print(f"   {k}: {'converged in %.0fs' % secs if ok else 'DID NOT CONVERGE'}")


if __name__ == "__main__":
    main()
