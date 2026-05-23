"""
test_cluster_partition_verify.py
=================================
Chia đôi máy tính thành 2 node ảo — chạy model + server thật.
Mục đích: kiểm tra giao diện cluster (2 node hiển thị) và dữ liệu phân tán.

Cách chạy:  python tests/test_cluster_partition_verify.py
Sau đó mở:  http://127.0.0.1:52417  (Node-A, master)
            http://127.0.0.1:52418  (Node-B, worker)
Nhấn Enter để dừng.
"""

import sys
import os
import time
import json
import subprocess
import urllib.request
import urllib.error
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NODE_A_ID       = "NODE-MASTER-GPU"
NODE_A_PORT     = 5678
NODE_A_API_PORT = 52417

NODE_B_ID       = "NODE-WORKER-CPU"
NODE_B_PORT     = 5679
NODE_B_API_PORT = 52418

LIGHTEST_MODEL = "qwen2.5:0.5b"

MOCK_STRONG = {
    "model": "Node-A (RTX 4090)", "chip": "NVIDIA RTX 4090", "memory": 24576,
    "flops": {"fp32": 82.0, "fp16": 165.0, "int8": 330.0},
    "cpu_cores": 16, "system_ram_mb": 64000, "gpu_count": 1, "total_gpu_vram_mb": 24576
}
MOCK_WEAK = {
    "model": "Node-B (CPU Only)", "chip": "Intel Core i5", "memory": 8192,
    "flops": {"fp32": 3.0, "fp16": 6.0, "int8": 12.0},
    "cpu_cores": 4, "system_ram_mb": 16384, "gpu_count": 0, "total_gpu_vram_mb": 0
}

CONFIG = {
    "peers": {
        NODE_A_ID: {
            "address": "127.0.0.1", "port": NODE_A_PORT,
            "device_capabilities": MOCK_STRONG
        },
        NODE_B_ID: {
            "address": "127.0.0.1", "port": NODE_B_PORT,
            "device_capabilities": MOCK_WEAK
        }
    }
}


def _write_config(config_path: str) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"  Config: {config_path}")


def _start_node(node_id: str, p2p_port: int, api_port: int,
                log_path: str, config_path: str, env_extra: dict = None):
    cmd = [
        sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synapse", "main.py"),
        "--default-model",         LIGHTEST_MODEL,
        "--node-id",               node_id,
        "--node-port",             str(p2p_port),
        "--chatgpt-api-port",      str(api_port),
        "--discovery-module",      "manual",
        "--discovery-config-path", config_path,
    ]
    log_fh = open(log_path, "w", encoding="utf-8")
    proc_env = {
        **os.environ,
        "DEBUG": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    }
    if env_extra:
        proc_env.update(env_extra)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=proc_env)
    return proc, log_fh


def _wait_for_api(base_url: str, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/healthcheck")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _api_get(url: str, timeout: int = 5) -> dict:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"_error": str(e)}


def run_cluster_test():
    print(f"\n{'='*60}")
    print(f"  SYNAPSE AI — 2-NODE CLUSTER VERIFICATION")
    print(f"  Model: {LIGHTEST_MODEL}")
    print(f"{'='*60}\n")

    tmpdir = tempfile.mkdtemp(prefix="synapse2_")
    config_path = os.path.join(tmpdir, "cluster_config.json")
    _write_config(config_path)

    proc_a = proc_b = None
    log_a_fh = log_b_fh = None
    all_ok = True

    try:
        # ===== START NODES =====
        print("[1] Starting Node-A (GPU-STRONG, master candidate)...")
        proc_a, log_a_fh = _start_node(
            NODE_A_ID, NODE_A_PORT, NODE_A_API_PORT,
            os.path.join(tmpdir, "node_a.log"), config_path,
            {"SYNAPSE_MOCK_CAPABILITIES": json.dumps(MOCK_STRONG)}
        )

        print("[2] Starting Node-B (CPU-WEAK, worker)...")
        proc_b, log_b_fh = _start_node(
            NODE_B_ID, NODE_B_PORT, NODE_B_API_PORT,
            os.path.join(tmpdir, "node_b.log"), config_path,
            {"SYNAPSE_MOCK_CAPABILITIES": json.dumps(MOCK_WEAK)}
        )

        # ===== WAIT FOR API =====
        print("[3] Waiting for servers to start...", end=" ", flush=True)
        a_ready = _wait_for_api(f"http://127.0.0.1:{NODE_A_API_PORT}")
        b_ready = _wait_for_api(f"http://127.0.0.1:{NODE_B_API_PORT}")
        print()
        print(f"  Node-A API: {'OK' if a_ready else 'FAIL'}")
        print(f"  Node-B API: {'OK' if b_ready else 'FAIL'}")
        if not a_ready or not b_ready:
            print("  Check logs:", os.path.join(tmpdir, "node_a.log"),
                  "and", os.path.join(tmpdir, "node_b.log"))
            raise RuntimeError("Servers failed to start")

        # ===== WAIT FOR CLUSTER FORMATION =====
        print("\n[4] Waiting for cluster formation (mesh + election)...")
        nodes_data = {}
        master_id = None
        for i in range(30):
            time.sleep(3)
            topo = _api_get(f"http://127.0.0.1:{NODE_A_API_PORT}/v1/topology")
            if "_error" not in topo:
                nodes_data = topo.get("nodes", {})
                master_id = topo.get("active_node_id")
                if len(nodes_data) >= 2 and master_id:
                    break
            print(f"  Waiting... ({i+1}/30)", end="\r")

        print()
        if len(nodes_data) >= 2:
            print(f"  Cluster formed: {len(nodes_data)} nodes, Master: {master_id}")
            for nid, info in nodes_data.items():
                mem = info.get("memory", 0)
                flops = info.get("flops", {})
                fp16 = flops.get("fp16", 0) if isinstance(flops, dict) else 0
                status = info.get("status", "?")
                print(f"    - {nid}: {mem}MB VRAM, {fp16} TFLOPS fp16, status={status}")
        else:
            print(f"  FAIL: Only {len(nodes_data)} nodes visible (need 2)")
            print(f"  Topology: {json.dumps(nodes_data, indent=2)[:300]}")
            all_ok = False

        # ===== CHECK DISTRIBUTED STATUS =====
        print("\n[5] Checking distributed status...")
        dist = _api_get(f"http://127.0.0.1:{NODE_A_API_PORT}/v1/distributed/status")
        if "_error" not in dist:
            print(f"  {json.dumps(dist, indent=2)}")
            if dist.get("partitions_count", 0) >= 2:
                print("  [OK] Data is partitioned across nodes!")
            else:
                print("  [--] No partitions yet (send a chat request first)")
        else:
            print(f"  GET /v1/distributed/status: {dist['_error']}")

        # ===== HOLD FOR UI INSPECTION =====
        print(f"\n{'='*60}")
        print(f"  SERVERS RUNNING — Open in browser:")
        print(f"  Node-A (master): http://127.0.0.1:{NODE_A_API_PORT}")
        print(f"  Node-B (worker): http://127.0.0.1:{NODE_B_API_PORT}")
        print(f"\n  Check Cluster page for both nodes.")
        print(f"  Send a chat to see data partitioning.")
        print(f"{'='*60}")
        print("\nPress Enter to stop servers...", end="", flush=True)
        input()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    finally:
        print("\nShutting down servers...")
        for p, fh, name in [(proc_a, log_a_fh, "Node-A"), (proc_b, log_b_fh, "Node-B")]:
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(5)
                except Exception:
                    p.kill()
                    p.wait(3)
            if fh:
                fh.close()
            print(f"  {name} stopped")
        print("Done.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    run_cluster_test()
