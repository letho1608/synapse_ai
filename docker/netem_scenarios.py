"""Measure a 2-container cluster (docker/docker-compose.yml) under network conditions.

  python docker/netem_scenarios.py            # fresh start, then every scenario

Prints tok/s per request plus the node's [decode]/[FAULT]/[FAILOVER] lines, so a
slow request shows whether steps timed out, a node was declared dead, or neither.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

COMPOSE = ["docker", "compose", "-f", "docker/docker-compose.yml"]
API = "http://127.0.0.1:52416"  # node A
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
QUESTIONS = ["What is the capital of France?", "List three fruits.", "Name two planets."]


def sh(*args, check=True):
    return subprocess.run(list(args), capture_output=True, text=True, check=check).stdout


def status():
    with urllib.request.urlopen(API + "/v1/distributed/status", timeout=5) as r:
        return json.loads(r.read())


def chat(q, n=40, timeout=300):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": q}],
                       "max_tokens": n, "temperature": 0}).encode()
    req = urllib.request.Request(API + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        k = d["usage"]["completion_tokens"]
        dt = time.time() - t
        return f"{dt:6.1f}s {k:3d} tok {k / dt:5.1f} tok/s {d['choices'][0]['message']['content'][:40]!r}"
    except Exception as e:
        return f"{time.time() - t:6.1f}s ERROR {type(e).__name__}: {e}"


def node_lines(since):
    out = sh("docker", "logs", "--since", since, "synapse-a", check=False)
    out += sh("docker", "logs", "--since", since, "synapse-b", check=False)
    return [l for l in out.splitlines() if any(k in l for k in ("[decode]", "[FAULT]", "[FAILOVER]", "[TCP] Cannot"))]


def netem(node, rule=None):
    sh("docker", "exec", node, "tc", "qdisc", "del", "dev", "eth0", "root", check=False)
    if rule:
        sh("docker", "exec", node, "tc", "qdisc", "add", "dev", "eth0", "root", "netem", *rule.split())


def scenario(name, rule=None, requests=2, n=40):
    netem("synapse-b", rule)
    since = sh("docker", "exec", "synapse-a", "date", "-u", "+%Y-%m-%dT%H:%M:%SZ").strip()
    print(f"\n== {name} ({rule or 'clean'})", flush=True)
    for i in range(requests):
        print("  ", chat(QUESTIONS[i % len(QUESTIONS)], n), flush=True)
    for l in node_lines(since):
        print("    |", l[:200])
    netem("synapse-b")


def outage(seconds=3):
    since = sh("docker", "exec", "synapse-a", "date", "-u", "+%Y-%m-%dT%H:%M:%SZ").strip()
    print(f"\n== link down {seconds}s in the middle of a 128-token answer", flush=True)

    def cut():
        time.sleep(2)
        netem("synapse-b", "loss 100%")
        time.sleep(seconds)
        netem("synapse-b")

    t = threading.Thread(target=cut)
    t.start()
    print("  ", chat("Write a short story about a robot learning to paint.", 128), flush=True)
    t.join()
    print("   after:", chat(QUESTIONS[0]), flush=True)
    for l in node_lines(since):
        print("    |", l[:200])


def main():
    print("restarting both containers ...", flush=True)
    sh(*COMPOSE, "down", check=False)
    sh(*COMPOSE, "up", "-d")
    t0 = time.time()
    while True:
        try:
            if status().get("partitions_count") == 2:
                break
        except Exception:
            pass
        if time.time() - t0 > 240:
            sys.exit("cluster never formed:\n" + sh("docker", "logs", "--tail", "40", "synapse-a", check=False))
        time.sleep(1)
    print(f"cluster formed after {time.time() - t0:.0f}s", flush=True)

    # The slow period seen on two real machines came right after the cluster formed.
    since = sh("docker", "exec", "synapse-a", "date", "-u", "+%Y-%m-%dT%H:%M:%SZ").strip()
    print("\n== right after formation, one request every few seconds for ~2 min", flush=True)
    end = time.time() + 120
    i = 0
    while time.time() < end:
        print(f"   t+{time.time() - t0:4.0f}s", chat(QUESTIONS[i % len(QUESTIONS)]), flush=True)
        i += 1
        time.sleep(3)
    for l in node_lines(since):
        print("    |", l[:200])

    scenario("latency", "delay 50ms 10ms")
    scenario("packet loss", "loss 5%")
    scenario("packet loss", "loss 20%")
    outage(3)
    outage(10)


if __name__ == "__main__":
    main()
