"""
test_cluster.py - Distributed Synapse AI Test (3 Nodes with GPU Support)
=========================================================================
This script automates:
  1. Checking GPU availability and configuring device
  2. Starting 3 nodes (NODE-A: Orchestrator, NODE-B/C: Workers)
  3. Waiting for cluster formation and topology sync
  4. Sending inference request and verifying distributed processing
  5. Checking node logs for actual work distribution
  6. Cleaning up all nodes
"""

import sys
import os
import time
import subprocess
import argparse
import json
import signal
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# GPU availability check
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEM_MB = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    else:
        GPU_NAME = None
        GPU_MEM_MB = 0
except ImportError:
    HAS_GPU = False
    GPU_NAME = None
    GPU_MEM_MB = 0

# Project root path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Node definitions
NODE_A_ID          = "NODE-A"
NODE_A_GRPC_PORT   = 50051
NODE_A_API_PORT    = 52415

NODE_B_ID          = "NODE-B"
NODE_B_GRPC_PORT   = 50052
NODE_B_API_PORT    = 52416

NODE_C_ID          = "NODE-C"
NODE_C_GRPC_PORT   = 50053
NODE_C_API_PORT    = 52417

DEFAULT_STARTUP_TIMEOUT = 300
DEFAULT_INFERENCE_TIMEOUT = 600


def _find_llm_model() -> str | None:
    """Scan Hugging Face cache to find the most suitable LLM."""
    from synapse.loading import get_models_dir
    search_paths = [
        get_models_dir(),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    found = []
    for p in search_paths:
        if not p.exists():
            continue
        for item in p.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if name.startswith("models--"):
                parts = name.split("--")
                if len(parts) >= 3:
                    found.append("/".join(parts[1:]))
            elif not name.startswith("."):
                found.append(name.replace("--", "/"))

    # Filter out non-LLM models
    llm = [m for m in found if not any(x in m.lower() for x in ["clip", "tokenizer", "embed"])]
    if not llm:
        return None
    # Prioritize Qwen / Llama
    prio = [m for m in llm if any(x in m.lower() for x in ["qwen", "llama"])]
    return prio[0] if prio else llm[0]


def _pick_available_port(preferred_port: int) -> int:
    """Use preferred port if available; otherwise pick a random free port."""
    import socket

    def _try_bind(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    if _try_bind(preferred_port):
        return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return int(s.getsockname()[1])


def _make_manual_config(log_dir: str,
                        node_a_grpc_port: int,
                        node_b_grpc_port: int,
                        node_c_grpc_port: int) -> tuple[str, str, str]:
    """
    Create 3 JSON config files for Manual Discovery with GPU/CPU capabilities.
    """
    if HAS_GPU and GPU_NAME:
        hw = {
            "model": GPU_NAME,
            "chip": GPU_NAME,
            "memory": GPU_MEM_MB,
            "flops": {"fp32": 10.0, "fp16": 20.0, "int8": 40.0}
        }
        print(f"  Using GPU: {GPU_NAME} ({GPU_MEM_MB}MB)")
    else:
        print("  WARNING: No GPU available, using CPU fallback")
        hw = {
            "model": "CPU",
            "chip": "CPU",
            "memory": 16384,
            "flops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0}
        }

    peers = {
        NODE_A_ID: {"address": "127.0.0.1", "port": node_a_grpc_port, "device_capabilities": hw},
        NODE_B_ID: {"address": "127.0.0.1", "port": node_b_grpc_port, "device_capabilities": hw},
        NODE_C_ID: {"address": "127.0.0.1", "port": node_c_grpc_port, "device_capabilities": hw},
    }

    config_a = {"peers": peers}
    config_b = {"peers": peers}
    config_c = {"peers": peers}

    path_a = os.path.join(log_dir, "config_node_a.json")
    path_b = os.path.join(log_dir, "config_node_b.json")
    path_c = os.path.join(log_dir, "config_node_c.json")

    for path, config in [(path_a, config_a), (path_b, config_b), (path_c, config_c)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    return path_a, path_b, path_c


def _start_node(node_id: str, grpc_port: int, api_port: int, model_name: str,
                log_file_path: str, config_path: str) -> tuple[subprocess.Popen, object]:
    """Start a Synapse node via subprocess using Manual Discovery."""
    cmd = [
        sys.executable, os.path.join(ROOT, "main.py"),
        "--default-model",          model_name,
        "--node-id",                node_id,
        "--node-port",              str(grpc_port),
        "--chatgpt-api-port",       str(api_port),
        "--discovery-module",       "manual",
        "--discovery-config-path",  config_path,
    ]
    log_fh = open(log_file_path, "w", encoding="utf-8")
    proc_env = {
        **os.environ,
        "SYNAPSE_DEBUG": "2",
        "PYTHONIOENCODING": "utf-8",
    }
    if HAS_GPU:
        proc_env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=proc_env,
    )
    return proc, log_fh


def _wait_for_api(base_url: str, timeout: int = 90) -> bool:
    """Wait for /healthcheck to return 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/healthcheck", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        except TimeoutError:
            pass
        except Exception:
            pass
        time.sleep(2)
    return False


def _http_get_json(url: str, timeout: int) -> dict | None:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            payload = resp.read().decode("utf-8", errors="replace")
            return json.loads(payload)
    except Exception:
        return None


def _http_post_json(url: str, payload: dict, timeout: int) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return resp.status, text
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", errors="replace")
        except Exception:
            text = str(e)
        return e.code, text
    except Exception as e:
        return 0, str(e)


def _wait_for_tcp_port(host: str, port: int, timeout: int = 90) -> bool:
    """Wait for TCP port to accept connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except OSError:
            time.sleep(1)
    return False


def _stop_process(proc: subprocess.Popen, log_fh):
    """Stop subprocess and close log file."""
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if log_fh:
        try:
            log_fh.close()
        except Exception:
            pass


def _check_topology(api_port: int) -> dict | None:
    """Get topology info from API."""
    return _http_get_json(f"http://127.0.0.1:{api_port}/v1/topology", timeout=5)


def _check_distributed_logs(log_paths: list[str]) -> bool:
    """Verify logs show actual distributed processing across nodes."""
    distributed_keywords = ["partition", "shard", "distributed", "inference", "forward", "layer"]
    all_active = True
    print("\n  Checking node logs for distributed activity:")
    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"    {os.path.basename(log_path)}: MISSING")
            all_active = False
            continue
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
            has_activity = any(kw in content for kw in distributed_keywords)
            status = "ACTIVE" if has_activity else "INACTIVE"
            print(f"    {os.path.basename(log_path)}: {status}")
            if not has_activity:
                all_active = False
    return all_active


def _send_chat(api_port: int, prompt: str, model_name: str,
               timeout: int = 900) -> tuple[str, float]:
    """Send chat request and return response."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 20,
        "temperature": 0.5,
    }
    t0 = time.perf_counter()
    status_code, text = _http_post_json(
        f"http://127.0.0.1:{api_port}/v1/chat/completions", payload, timeout=timeout
    )
    elapsed = time.perf_counter() - t0
    if status_code == 200:
        data = json.loads(text)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, elapsed
    else:
        return f"[HTTP {status_code}] {text[:200]}", elapsed


def run_distributed_test(model_name: str, prompt: str,
                         startup_timeout: int, inference_timeout: int):
    # Pick available ports for all nodes
    node_a_grpc_port = _pick_available_port(NODE_A_GRPC_PORT)
    node_a_api_port = _pick_available_port(NODE_A_API_PORT)
    node_b_grpc_port = _pick_available_port(NODE_B_GRPC_PORT)
    node_b_api_port = _pick_available_port(NODE_B_API_PORT)
    node_c_grpc_port = _pick_available_port(NODE_C_GRPC_PORT)
    node_c_api_port = _pick_available_port(NODE_C_API_PORT)

    # Display test dashboard
    header = (
        f"\n{'='*60}\n"
        f"{'SYNAPSE AI - 3-NODE DISTRIBUTED CLUSTER TEST'.center(60)}\n"
        f"{'='*60}\n"
        f" Model       : {model_name}\n"
        f" Prompt      : \"{prompt}\"\n"
        f" Device      : {'GPU (' + GPU_NAME + ')' if HAS_GPU else 'CPU (fallback)'}\n"
        f" NODE-A      : gRPC={node_a_grpc_port}, API={node_a_api_port} (Orchestrator)\n"
        f" NODE-B      : gRPC={node_b_grpc_port}, API={node_b_api_port} (Worker)\n"
        f" NODE-C      : gRPC={node_c_grpc_port}, API={node_c_api_port} (Worker)\n"
        f" Discovery   : Manual (Static JSON)\n"
        f"{'='*60}"
    )
    print(header)

    # Initialize process handles
    proc_a = proc_b = proc_c = None
    log_a_fh = log_b_fh = log_c_fh = None
    temp_dir = os.path.join(ROOT, "test_logs")
    os.makedirs(temp_dir, exist_ok=True)

    log_a = os.path.join(temp_dir, "node_a.log")
    log_b = os.path.join(temp_dir, "node_b.log")
    log_c = os.path.join(temp_dir, "node_c.log")

    # Create config files for all nodes
    print(f"\nTest logs at: {temp_dir}")
    config_a_path, config_b_path, config_c_path = _make_manual_config(
        temp_dir,
        node_a_grpc_port=node_a_grpc_port,
        node_b_grpc_port=node_b_grpc_port,
        node_c_grpc_port=node_c_grpc_port,
    )

    try:
        # Step 1: Start worker nodes (B and C)
        print(f"\n[Step 1/6] Starting worker nodes...")
        proc_b, log_b_fh = _start_node(
            NODE_B_ID, node_b_grpc_port, node_b_api_port, model_name, log_b, config_b_path
        )
        proc_c, log_c_fh = _start_node(
            NODE_C_ID, node_c_grpc_port, node_c_api_port, model_name, log_c, config_c_path
        )

        # Wait for worker gRPC ports
        print("  Waiting for worker gRPC servers...")
        if not _wait_for_tcp_port("127.0.0.1", node_b_grpc_port, timeout=startup_timeout):
            print(f"  ERROR: NODE-B gRPC failed to start on port {node_b_grpc_port}")
            return False
        if not _wait_for_tcp_port("127.0.0.1", node_c_grpc_port, timeout=startup_timeout):
            print(f"  ERROR: NODE-C gRPC failed to start on port {node_c_grpc_port}")
            return False

        # Step 2: Start orchestrator node (A)
        print(f"\n[Step 2/6] Starting orchestrator node (NODE-A)...")
        proc_a, log_a_fh = _start_node(
            NODE_A_ID, node_a_grpc_port, node_a_api_port, model_name, log_a, config_a_path
        )

        # Step 3: Wait for all API servers
        print(f"\n[Step 3/6] Waiting for HTTP API servers...")
        url_a = f"http://127.0.0.1:{node_a_api_port}"
        url_b = f"http://127.0.0.1:{node_b_api_port}"
        url_c = f"http://127.0.0.1:{node_c_api_port}"

        ok_a = _wait_for_api(url_a, timeout=startup_timeout)
        ok_b = _wait_for_api(url_b, timeout=startup_timeout)
        ok_c = _wait_for_api(url_c, timeout=startup_timeout)

        if not all([ok_a, ok_b, ok_c]):
            failed = []
            if not ok_a: failed.append("NODE-A")
            if not ok_b: failed.append("NODE-B")
            if not ok_c: failed.append("NODE-C")
            print(f"  ERROR: API servers failed to start: {', '.join(failed)}")
            return False

        print("  All API servers ready!")
        time.sleep(5)  # Allow API stabilization

        # Step 4: Wait for cluster topology sync
        print(f"\n[Step 4/6] Synchronizing LACP Topology...")
        topology_timeout = 60
        topo_deadline = time.time() + topology_timeout
        while time.time() < topo_deadline:
            topo = _check_topology(node_a_api_port)
            if topo:
                nodes = topo.get("nodes", {}) if isinstance(topo, dict) else {}
                print(f"  Current cluster size: {len(nodes)} nodes")
                if len(nodes) >= 3:
                    print(f"  3-node cluster formed successfully!")
                    print(f"  Waiting 30s for device initialization...")
                    time.sleep(30)
                    break
            time.sleep(3)
        else:
            print(f"  ERROR: Failed to form 3-node cluster")
            return False

        # Step 5: Execute distributed inference
        print(f"\n[Step 5/6] Executing Distributed Inference...")
        print(f"  Prompt: \"{prompt}\"")
        response, elapsed = _send_chat(
            node_a_api_port, prompt, model_name, timeout=inference_timeout
        )

        # Print inference results
        print(f"\n" + "-" * 60)
        print(f"  DISTRIBUTED INFERENCE RESULT SUMMARY".center(60))
        print("-" * 60)

        if response.startswith("[HTTP"):
            print(f"  STATUS   : FAILED")
            print(f"  ERROR    : {response}")
            success = False
        else:
            print(f"  STATUS   : SUCCESS")
            print(f"  LATENCY  : {elapsed:.2f}s")
            try:
                print(f"  RESPONSE : {response}")
            except UnicodeEncodeError:
                import sys
                print("  RESPONSE : ", end="")
                sys.stdout.buffer.write(response.encode('utf-8'))
                sys.stdout.buffer.write(b'\n')
                sys.stdout.buffer.flush()

            word_count = len(response.split())
            print(f"  ESTIMATE : ~{word_count} words ({word_count/elapsed:.2f} words/sec)")
            success = True

        print("-" * 60)

        # Step 6: Verify distributed processing
        print(f"\n[Step 6/6] Verifying distributed processing...")
        log_paths = [log_a, log_b, log_c]
        is_distributed = _check_distributed_logs(log_paths)
        if is_distributed:
            print("  [OK] Cluster is ACTIVELY distributed across all 3 nodes")
        else:
            print("  [WARN] Cluster may not be fully distributed")
        return success and is_distributed

    except KeyboardInterrupt:
        print("\n  Warn: Test cancelled by user.")
        return False
    except Exception as e:
        print(f"\n  ERROR: Unexpected failure: {e}")
        return False
    finally:
        print("\n  Cleaning up environment...")
        _stop_process(proc_a, log_a_fh)
        _stop_process(proc_b, log_b_fh)
        _stop_process(proc_c, log_c_fh)
        print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse AI 3-Node Distributed Test")
    parser.add_argument("--model", type=str, default=None, help="Model name to use")
    parser.add_argument("--prompt", type=str, default="Xin chào", help="Test prompt")
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT)
    parser.add_argument("--inference-timeout", type=int, default=DEFAULT_INFERENCE_TIMEOUT)
    args = parser.parse_args()

    model = args.model
    if not model:
        print("Scanning for available models...")
        model = _find_llm_model()
        if model:
            try:
                from synapse.model_list import resolve_hf_id
                model = resolve_hf_id(model)
            except Exception:
                pass
            print(f"  Found: {model}")
        else:
            print("  ERROR: No model found.")
            sys.exit(1)

    success = run_distributed_test(
        model_name        = model,
        prompt            = args.prompt,
        startup_timeout   = args.startup_timeout,
        inference_timeout = args.inference_timeout,
    )
    sys.exit(0 if success else 1)
