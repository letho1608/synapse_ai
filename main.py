import os
import sys
import subprocess
import time
import argparse
import traceback

# Add backend directory to Python path so we can import from it
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
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
    parser = argparse.ArgumentParser(description="Universal Synapse Launcher")
    parser.add_argument("--no-frontend", action="store_true", help="Do not launch the web frontend")
    
    # Parse known args, pass the rest to the backend core
    args, unknown_args = parser.parse_known_args()

    install_terminal_log()
    _check_gpu_environment()

    frontend_proc = None
    if not args.no_frontend:
        print("[INFO] Starting Frontend Dev Server...")
        # Only start frontend if we are on Windows, we use shell=True and npm command
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        try:
            # Install dependencies first if node_modules doesn't exist
            if not os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
                print("[INFO] Installing Frontend dependencies...")
                subprocess.run([npm_cmd, "install"], cwd=FRONTEND_DIR)
            
            # Start dev server
            frontend_proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=FRONTEND_DIR)
        except Exception as e:
            print(f"[ERROR] Failed to start frontend: {e}")

    try:
        while True:
            print("\n[INFO] Starting Backend System...")
            print("[INFO] Press Ctrl+C to stop both Frontend and Backend\n")
            
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
        if frontend_proc is not None:
            print("[INFO] Terminating Frontend Server...")
            try:
                # Terminate the frontend process tree gracefully
                if os.name == 'nt':
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)], capture_output=True)
                else:
                    frontend_proc.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    run()
