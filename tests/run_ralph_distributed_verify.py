"""
SYNAPSE AI - DISTRIBUTED SYSTEM VERIFICATION SCRIPT
====================================================
This script runs a FULL end-to-end distributed test within a single machine
by launching two Python subprocesses that communicate over real TCP sockets.
It validates ALL critical fixes:

1. P2P TCP mesh formation (handshake, heartbeat, tiebreaker)
2. Event routing across processes (cross-process publish/subscribe via TCP bridge)
3. Binary tensor serialization (tobytes/frombuffer with hex encoding in bridge)
4. ManualDiscovery health checks + topology collection via RPC
5. Master election with weighted voting
6. Authoritative topology/partition broadcast from master node
7. Orchestrator_id routing for result delivery

This script CAN be run on a single machine for verification:
  python tests/run_ralph_distributed_verify.py

It will start two subprocess nodes, form a mesh, and verify all operations.
"""

import asyncio
import subprocess
import sys
import os
import time
import json
import tempfile
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_multimachine_distributed.py")


def run_node(args_list, timeout=120):
    """Run a node as a subprocess and capture output."""
    proc = subprocess.Popen(
        [sys.executable, TEST_SCRIPT] + args_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    class Result:
        def __init__(self):
            self.output_lines = []
            self.success = None

    result = Result()

    def reader():
        for line in iter(proc.stdout.readline, ''):
            result.output_lines.append(line.rstrip())
            print(f"  [SUB] {line.rstrip()}")

    import threading
    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        proc.wait(timeout=timeout)
        output = '\n'.join(result.output_lines)
        result.success = "PASS" in output and "FAIL" not in output
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        result.success = False

    return result


async def main():
    print("=" * 60)
    print("SYNAPSE AI - DISTRIBUTED SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Script: {TEST_SCRIPT}")
    print()

    PORT_A = 5990
    PORT_B = 5991

    # Phase 1: Test simultaneous P2P connectivity (direct mode server/client)
    print("=" * 60)
    print("PHASE 1: Direct P2P TCP Connectivity (simultaneous)") 
    print("=" * 60)

    # Start server in background
    print("Starting server process...")
    server_proc = subprocess.Popen(
        [sys.executable, TEST_SCRIPT, "--role", "server", "--port", str(PORT_A)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    time.sleep(3)

    print("Running client...")
    client = run_node(["--role", "client", "--port", str(PORT_B), "--target", f"127.0.0.1:{PORT_A}"], timeout=30)

    server_proc.kill()
    server_out, _ = server_proc.communicate()

    print()
    print(f"  Client: {'PASSED' if client.success else 'FAILED'}")

    if not client.success:
        print("\n  *** Phase 1 FAILED - stopping ***")
        sys.exit(1)

    # Phase 2: Test full cluster with config-based mode (both sides)
    print()
    print("=" * 60)
    print("PHASE 2: Full Cluster Mode (Discovery + Election + Topology)")
    print("=" * 60)

    # Create shared config for multi-machine test
    config = {
        "peers": {
            "MACHINE-A": {
                "address": "127.0.0.1",
                "port": PORT_A + 100,
                "device_capabilities": {
                    "model": "Primary(GPU)", "chip": "RTX 3080", "memory": 10240,
                    "flops": {"fp32": 30.0, "fp16": 60.0, "int8": 120.0},
                    "cpu_cores": 8, "system_ram_mb": 32000, "gpu_count": 1, "total_gpu_vram_mb": 10240
                }
            },
            "MACHINE-B": {
                "address": "127.0.0.1",
                "port": PORT_B + 100,
                "device_capabilities": {
                    "model": "Secondary(CPU)", "chip": "Core i5", "memory": 8192,
                    "flops": {"fp32": 3.0, "fp16": 6.0, "int8": 12.0},
                    "cpu_cores": 4, "system_ram_mb": 16384, "gpu_count": 0, "total_gpu_vram_mb": 0
                }
            }
        }
    }
    tmpdir = tempfile.mkdtemp(prefix="synapse_ralph_")
    config_path = os.path.join(tmpdir, "cluster_config_multimachine.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Run both sides concurrently
    import threading
    results = {}

    def run_config_node(name, node_id, port):
        args_list = [
            "--config-based", "--port", str(port), "--node-id", node_id,
            "--peer-host", "127.0.0.1", "--peer-port", str(PORT_B + 100 if node_id == "MACHINE-A" else PORT_A + 100)
        ]
        proc = subprocess.Popen(
            [sys.executable, TEST_SCRIPT] + args_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output_lines = []
        for line in iter(proc.stdout.readline, ''):
            output_lines.append(line.rstrip())
            print(f"  [{name}] {line.rstrip()}")
        proc.wait()
        output = '\n'.join(output_lines)
        results[name] = "PASS" in output and "FAIL" not in output

    print("\nStarting both nodes simultaneously...")
    t_a = threading.Thread(target=run_config_node, args=("A", "MACHINE-A", PORT_A + 100))
    t_b = threading.Thread(target=run_config_node, args=("B", "MACHINE-B", PORT_B + 100))
    t_a.start()
    t_b.start()
    t_a.join(timeout=60)
    t_b.join(timeout=60)

    print(f"\n  Machine-A: {'PASSED' if results.get('A', False) else 'FAILED'}")
    print(f"  Machine-B: {'PASSED' if results.get('B', False) else 'FAILED'}")

    # Phase 3: Verify existing tests still pass
    print()
    print("=" * 60)
    print("PHASE 3: Regression Tests")
    print("=" * 60)

    print("Running test_simultaneous...")
    r1 = subprocess.run([sys.executable, "-m", "pytest", "tests/test_simultaneous.py", "-v"],
                        capture_output=True, text=True, timeout=60,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(r1.stdout[-300:] if len(r1.stdout) > 300 else r1.stdout)

    print("\nRunning test_cluster_inprocess via direct execution...")
    r2 = subprocess.run([sys.executable, "tests/test_cluster_inprocess.py"],
                        capture_output=True, text=True, timeout=60,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(r2.stdout[-500:] if len(r2.stdout) > 500 else r2.stdout)

    phase1_ok = client.success if hasattr(client, 'success') else True
    phase2_ok = results.get('A', False) and results.get('B', False)
    phase3_ok = "PASSED" in r1.stdout and "PASS" in r2.stdout

    # Cleanup
    try:
        os.unlink(config_path)
        os.rmdir(tmpdir)
    except Exception:
        pass

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Phase 1 (Direct P2P):         {'PASSED' if phase1_ok else 'FAILED'}")
    print(f"  Phase 2 (Full Cluster):       {'PASSED' if phase2_ok else 'FAILED'}")
    print(f"  Phase 3 (Regression Tests):   {'PASSED' if phase3_ok else 'FAILED'}")

    all_ok = phase1_ok and phase2_ok and phase3_ok
    print(f"\n  OVERALL: {'DONE - ALL PASSED' if all_ok else 'NOT_DONE - SOME FAILED'}")

    return all_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)