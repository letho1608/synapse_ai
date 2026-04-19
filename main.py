
"""
Synapse AI Entry Point
"""
import os
import sys
import asyncio
import traceback
import warnings
import subprocess

# Suppress pynvml deprecation warning (torch uses pynvml; recommend nvidia-ml-py)
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*nvidia-ml-py.*", category=FutureWarning)


def _detect_nvidia_gpus_via_nvidia_smi() -> list[str]:
    """
    Use nvidia-smi (if available) to detect NVIDIA GPUs.
    Returns list of GPU names.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not (out.stdout or "").strip():
            return []
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return names
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def _check_gpu_environment() -> None:
    """
    Check GPU environment and print status:
    1. NVIDIA GPU presence
    2. PyTorch build type (CPU-only / GPU+CUDA)
    3. CUDA availability in PyTorch
    Conclusion: Will use CPU or GPU.
    """
    nvidia_gpus = _detect_nvidia_gpus_via_nvidia_smi()
    has_nvidia_gpu = len(nvidia_gpus) > 0

    # Status variables for summary
    torch_installed = False
    torch_has_cuda_module = False
    torch_cuda_available = False
    pytorch_mode = "not_installed"

    # 1. PyTorch status
    try:
        import torch  # type: ignore
        torch_installed = True
    except Exception:
        print("\n[GPU-SUMMARY]")
        print(f"  1) Physical NVIDIA GPU: {'YES' if has_nvidia_gpu else 'NO'}")
        print("  2) PyTorch: NOT INSTALLED")
        print("  3) CUDA in PyTorch: NO (PyTorch not installed)")
        print("  => CONCLUSION: System will run on CPU.")
        return

    # 3. PyTorch CPU-only or GPU build?
    torch_has_cuda_module = bool(getattr(torch, 'cuda', None))
    if not torch_has_cuda_module:
        pytorch_mode = "cpu_only"
        print("\n[GPU-SUMMARY]")
        print(f"  1) Physical NVIDIA GPU: {'YES' if has_nvidia_gpu else 'NO'}")
        print("  2) PyTorch: INSTALLED (CPU-only, NO CUDA support)")
        print("  3) CUDA in PyTorch: NO (no torch.cuda)")
        print("  => CONCLUSION: System will run on CPU.")
        return

    pytorch_mode = "gpu_build"

    # 4. CUDA/driver in PyTorch
    try:
        torch_cuda_available = bool(torch.cuda.is_available())
        if not torch_cuda_available:
            print("\n[GPU-SUMMARY]")
            print(f"  1) Physical NVIDIA GPU: {'YES' if has_nvidia_gpu else 'NO'}")
            print("  2) PyTorch: INSTALLED (GPU/CUDA build)")
            print("  3) CUDA in PyTorch: NOT AVAILABLE (torch.cuda.is_available() == False)")
            print("  => CONCLUSION: System will run on CPU.")
            return

        device_count = torch.cuda.device_count()
        if device_count <= 0:
            print("\n[GPU-SUMMARY]")
            print(f"  1) Physical NVIDIA GPU: {'YES' if has_nvidia_gpu else 'NO'}")
            print("  2) PyTorch: INSTALLED (GPU/CUDA build)")
            print("  3) CUDA in PyTorch: YES but no GPUs listed (device_count == 0)")
            print("  => CONCLUSION: System will run on CPU.")
            return

        torch_gpu_names = []
        for i in range(device_count):
            try:
                name = torch.cuda.get_device_name(i)
                torch_gpu_names.append(name or f"GPU{i}")
            except Exception:
                torch_gpu_names.append(f"GPU{i}")

        print("\n[GPU-SUMMARY]")
        print(f"  1) Physical NVIDIA GPU: {'YES' if has_nvidia_gpu else 'NO'}")
        print("  2) PyTorch: INSTALLED (GPU/CUDA build)")
        print("  3) CUDA in PyTorch: YES (torch.cuda.is_available() == True)")
        print("  => CONCLUSION: System will priority run on GPU.")
    except Exception as e:
        print(f"[GPU] Error checking CUDA with PyTorch => REASON: {e}")
        print("       System will run on CPU. Check NVIDIA driver, CUDA and PyTorch GPU version.")
        print("\n[GPU-SUMMARY]")
        print(f"  1) Physical NVIDIA GPU: {'YES' if has_nvidia_gpu else 'NO'}")
        print(f"  2) PyTorch: {'INSTALLED' if torch_installed else 'NOT INSTALLED'} (GPU/CUDA build)")
        print("  3) CUDA in PyTorch: ERROR DURING CHECK")
        print("  => CONCLUSION: System will run on CPU.")


def run():
    """Main entry point"""
    # Capture stdout/stderr to show logs in dashboard
    from synapse.terminal_log import install as install_terminal_log
    install_terminal_log()

    # Check GPU / PyTorch environment on startup
    _check_gpu_environment()

    try:
        # Import synapse main
        from synapse.main import main, configure_event_loop
        
        # Configure event loop
        loop = configure_event_loop()
        
        print("\n[INFO] Starting system...")
        print("[INFO] Press Ctrl+C to stop\n")
        
        # Run main
        loop.run_until_complete(main())
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Shutdown requested by user.")
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        print("Please ensure all dependencies are installed.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        print("\n[INFO] Exiting...")

if __name__ == "__main__":
    run()
