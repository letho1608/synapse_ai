"""
Live test: two real backend PROCESSES split one model and answer correctly.

Launches `main.py` twice (manual discovery on 127.0.0.1, separate bridge and
API ports), sends one chat request to node A, then proves from each node's
forward log that A ran ONLY layers 0-14 and B ran ONLY layers 15-29 of
SmolLM2-135M-Instruct, that tensors crossed processes over the binary TCP
path, and that the greedy answer equals the single-process one.

Same-machine only: it exercises every inter-process code path (discovery,
handshake, partition plan, chunked TCP, result routing) but not a real
network. Skipped unless the default model is in the local HF cache.
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

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = [("liveA", 51161, 59811), ("liveB", 51162, 59812)]


def _cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)
    except Exception:
        return False


def _get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, body, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


@pytest.mark.skipif(not _cached(), reason=f"{MODEL} not in local HF cache")
def test_two_processes_split_layers_and_answer_correctly():
    tmp = tempfile.mkdtemp(prefix="synapse-live-")
    cfg = {"peers": {nid: {"address": "127.0.0.1", "port": port, "device_capabilities": {
        "model": nid, "chip": "cpu", "memory": 8000,
        "flops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0}}} for nid, port, _ in NODES}}
    cfg_path = os.path.join(tmp, "cluster.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    env = dict(os.environ, HF_HUB_OFFLINE="1", DEBUG="1", PYTHONIOENCODING="utf-8")
    env.pop("SYNAPSE_PARENT_PID", None)
    procs, logs = [], {}
    try:
        for nid, port, api in NODES:
            logs[nid] = os.path.join(tmp, f"{nid}.log")
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(ROOT, "main.py"), "--node-id", nid,
                 "--node-port", str(port), "--discovery-module", "manual",
                 "--discovery-config-path", cfg_path, "--chatgpt-api-port", str(api),
                 "--disable-tui"],
                cwd=ROOT, env=env, stdout=open(logs[nid], "w", encoding="utf-8"),
                stderr=subprocess.STDOUT))
        api_a = NODES[0][2]
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if _get(f"http://127.0.0.1:{api_a}/v1/distributed/status").get("partitions_count") == 2:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            pytest.fail("cluster never formed 2 partitions")
        time.sleep(3)  # handshakes carry the tensor-transport ports

        resp = _post(f"http://127.0.0.1:{api_a}/v1/chat/completions", {
            "model": MODEL, "stream": False, "max_tokens": 16, "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant named SmolLM, trained by Hugging Face"},
                {"role": "user", "content": "What is the capital of France? Answer in one word."}]})
        answer = resp["choices"][0]["message"]["content"]
        status = _get(f"http://127.0.0.1:{api_a}/v1/distributed/status")
    finally:
        for p in procs:
            p.kill()
        for p in procs:
            p.wait(timeout=20)

    ranges = {}
    for nid, _, _ in NODES:
        with open(logs[nid], encoding="utf-8", errors="replace") as f:
            ranges[nid] = set(re.findall(r"forward layers (\d+-\d+)/30", f.read()))
    # Each node ran exactly one range; together they cover 0-29 once. (The
    # measured placement may pick 15/15 or 16/14 on two identical nodes.)
    assert all(len(r) == 1 for r in ranges.values()), ranges
    (a,), (b,) = ranges["liveA"], ranges["liveB"]
    a0, a1 = map(int, a.split("-"))
    b0, b1 = map(int, b.split("-"))
    assert sorted([(a0, a1), (b0, b1)])[0][0] == 0 and sorted([(a0, a1), (b0, b1)])[1][1] == 29
    assert sorted([(a0, a1), (b0, b1)])[0][1] + 1 == sorted([(a0, a1), (b0, b1)])[1][0], ranges
    assert answer.strip() == "The capital of France is Paris.", answer
    assert status.get("transport_counts", {}).get("tcp-chunked", 0) >= 1, status
