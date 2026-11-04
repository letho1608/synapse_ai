"""
Live failover: three real backend processes, one stage process is KILLED
while a streamed answer is being generated, and the answer still completes
with exactly the text the healthy cluster produced (greedy).

Also checks that measured placement took over once every node had gossiped
its layer profile ([PLACEMENT] log line on the master).

Same-machine only (real processes, sockets and heartbeats; not a real
network). Skipped unless the default model is in the local HF cache.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

import pytest

from _ports import free_ports

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# (node id, node port, API port): free ports picked by the OS, see _ports.py
_P = free_ports(6)
NODES = [('foA', _P[0], _P[3]), ('foB', _P[1], _P[4]), ('foC', _P[2], _P[5])]
MESSAGES = [
    {"role": "system", "content": "You are a helpful AI assistant named SmolLM, trained by Hugging Face"},
    {"role": "user", "content": "List five fruits, one per line."}]


def _cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)
    except Exception:
        return False


def _get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, body, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _stream(url, body, on_chunk, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
            except Exception:
                continue
            text += delta
            on_chunk(text)
    return text


@pytest.mark.skipif(not _cached(), reason=f"{MODEL} not in local HF cache")
def test_killing_a_stage_mid_answer_keeps_the_answer():
    tmp = tempfile.mkdtemp(prefix="synapse-failover-")
    cfg = {"peers": {nid: {"address": "127.0.0.1", "port": port, "device_capabilities": {
        "model": nid, "chip": "cpu", "memory": 8000,
        "flops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0}}} for nid, port, _ in NODES}}
    cfg_path = os.path.join(tmp, "cluster.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    env = dict(os.environ, HF_HUB_OFFLINE="1", DEBUG="1", PYTHONIOENCODING="utf-8")
    env.pop("SYNAPSE_PARENT_PID", None)
    procs, logs = {}, {}
    body = {"model": MODEL, "max_tokens": 40, "temperature": 0, "messages": MESSAGES}
    try:
        for nid, port, api in NODES:
            logs[nid] = os.path.join(tmp, f"{nid}.log")
            procs[nid] = subprocess.Popen(
                [sys.executable, os.path.join(ROOT, "main.py"), "--node-id", nid,
                 "--node-port", str(port), "--discovery-module", "manual",
                 "--discovery-config-path", cfg_path, "--chatgpt-api-port", str(api),
                 "--disable-tui"],
                cwd=ROOT, env=env, stdout=open(logs[nid], "w", encoding="utf-8"),
                stderr=subprocess.STDOUT)
        api_a = NODES[0][2]
        base = f"http://127.0.0.1:{api_a}"
        deadline = time.time() + 150
        while time.time() < deadline:
            try:
                if _get(f"{base}/v1/distributed/status").get("partitions_count") == 3:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            pytest.fail("cluster never formed 3 partitions")
        time.sleep(3)
        # Warm-up request: every node loads its layers and measures them;
        # then wait for a gossip round so the measured placement applies.
        _post(f"{base}/v1/chat/completions", dict(body, max_tokens=4))
        time.sleep(22)
        healthy = _post(f"{base}/v1/chat/completions", body)["choices"][0]["message"]["content"]
        assert healthy.strip()

        # The master (any node) computes the placement and broadcasts it.
        placements = []
        for nid, _, _ in NODES:
            with open(logs[nid], encoding="utf-8", errors="replace") as f:
                placements += re.findall(r"\[PLACEMENT\] throughput: (\[.*?\])", f.read())
        assert placements, "measured placement never ran"
        stages = re.findall(r"\('(\w+)', (\d+), (\d+)\)", placements[-1])
        victims = [nid for nid, _, _ in stages if nid != "foA"]
        assert victims, stages
        victim = victims[-1]

        killed = {}

        def on_chunk(text):
            if not killed and len(text) >= 8:
                procs[victim].kill()
                killed["at"] = len(text)

        recovered = _stream(f"{base}/v1/chat/completions", body, on_chunk)
    finally:
        for p in procs.values():
            p.kill()
        for p in procs.values():
            p.wait(timeout=20)

    assert killed, "the stream never produced text before the kill"
    with open(logs["foA"], encoding="utf-8", errors="replace") as f:
        log_a = f.read()
    dead = re.findall(r"\[FAILOVER\] dead=\[([^\]]*)\]", log_a)
    assert dead, "orchestrator never failed over"
    assert f"'{victim}'" in dead[0], (victim, dead)
    assert recovered == healthy, (recovered, healthy)
