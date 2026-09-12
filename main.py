import os
import sys
import subprocess
import time
import argparse
import traceback

# Add backend directory to Python path so we can import from it
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

# Terminal logs capturing needs to be done here
from synapse.terminal_log import install as install_terminal_log

def _check_gpu_environment() -> None:
    """
    Check GPU environment and print status using the core device_capabilities module.
    """
    try:
        from synapse.topology.device_capabilities import SystemSpecs, GpuBackend
        
        specs = SystemSpecs.detect()
        
        print("\n[GPU-SUMMARY]")
        
        if specs.has_gpu and specs.gpu_count > 0:
            print(f"  1) Physical Hardware GPU: YES (Count: {specs.gpu_count})")
            for i, gpu in enumerate(specs.gpus):
                name = gpu.name or "Unknown GPU"
                vram_info = f"{gpu.vram_gb:.1f} GB" if gpu.vram_gb else "Unknown VRAM"
                unified_mark = " [Unified Memory]" if gpu.unified_memory else ""
                print(f"     - [{i}] {name} ({vram_info}){unified_mark}")
        else:
            print("  1) Physical Hardware GPU: NO")

        # PyTorch check
        try:
            import torch
            pytorch_installed = True
        except ImportError:
            pytorch_installed = False

        print(f"  2) PyTorch Installed: {'YES' if pytorch_installed else 'NO'}")
        
        backend_label = specs.backend.value if hasattr(specs.backend, 'value') else str(specs.backend)
        print(f"  3) Detected Backend API: {backend_label}")

        conclusion = "CPU"
        if specs.has_gpu and specs.backend not in (GpuBackend.CPU_ARM, GpuBackend.CPU_X86):
            conclusion = f"{backend_label.upper()} GPU"
        
        print(f"  => CONCLUSION: System will priority run on {conclusion}.\n")
        
    except Exception as e:
        print(f"[GPU] Error checking hardware => REASON: {e}")
        print("  => CONCLUSION: System will run on CPU.\n")

def run():
    # C-W11: --no-frontend was a documented no-op; it now maps to headless
    # (TUI off + GET / stub) with a one-release deprecation warning.
    if "--no-frontend" in sys.argv:
        print("[DEPRECATION] --no-frontend is renamed to the headless mode "
              "(CLI: serve --headless, backend: --headless / SYNAPSE_HEADLESS=1).",
              file=sys.stderr)
        os.environ["SYNAPSE_HEADLESS"] = "1"
    # T2.1 CLI-first: subcommands route to synapse.cli; everything else keeps
    # the legacy passthrough below (incl. --no-frontend no-op).
    # Global flags may precede the subcommand (either order works).
    _cli_cmds = ("serve", "join", "client", "status", "topology", "models",
                 "benchmark", "console")
    _goes_to_cli = (len(sys.argv) > 1 and sys.argv[1] in _cli_cmds
                    or len(sys.argv) > 2 and sys.argv[1] in (
                        "--port", "--log-format", "--headless")
                    and any(a in _cli_cmds for a in sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        from synapse.helpers import VERSION
        print("synapse %s" % VERSION)
        return
    if _goes_to_cli:
        from synapse.cli import main as cli_main
        # Normalize global-first order onto the canonical subcommand-first
        # form argparse understands (parents accept both, this is just for
        # the legacy argv[1] check above on re-entry — no-op otherwise).
        sys.exit(cli_main(sys.argv[1:]))
    parser = argparse.ArgumentParser(description="Universal Synapse Launcher")
    # Retained for CLI compatibility: the web UI is a backend-served static
    # dashboard (no Node.js dev server), so this flag is a no-op.
    parser.add_argument("--no-frontend", action="store_true", help="No-op: the web UI is served by the backend (no Node.js dependency)")

    # Parse known args, pass the rest to the backend core
    args, unknown_args = parser.parse_known_args()

    install_terminal_log()
    _check_gpu_environment()

    try:
        while True:
            print("\n[INFO] Starting Backend System...")
            print("[INFO] Press Ctrl+C to stop\n")
            
            script_path = os.path.join(BACKEND_DIR, "synapse", "main.py")
            cmd = [sys.executable, script_path] + unknown_args
            p = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=BACKEND_DIR))
            p.wait()
            
            if p.returncode == 42:
                print("\n[INFO] Auto-restarting system to apply new settings...\n")
                continue
            else:
                break
                
    except KeyboardInterrupt:
        print("\n\n[INFO] Shutdown requested by user.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        print("\n[INFO] Exiting...")

if __name__ == "__main__":
    run()
