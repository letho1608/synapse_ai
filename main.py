
"""
Synapse AI Entry Point
"""
import os
import sys
import asyncio
import traceback
import warnings
import subprocess
import torch 
# Suppress pynvml deprecation warning (torch uses pynvml; recommend nvidia-ml-py)
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*nvidia-ml-py.*", category=FutureWarning)


def _detect_nvidia_gpus_via_nvidia_smi() -> list[str]:
    """
    Dùng nvidia-smi (nếu có) để phát hiện rõ có GPU NVIDIA hay không.
    Trả về list tên GPU (có thể rỗng nếu không có/không truy cập được).
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
    Kiểm tra chi tiết môi trường GPU và in ra 3 thông tin rõ ràng:
    1. Có GPU NVIDIA hay không
    2. Đang dùng bản PyTorch loại nào (chưa cài / CPU-only / GPU+CUDA)
    3. CUDA trong PyTorch có khả dụng không
    Cuối cùng KẾT LUẬN: chạy bằng CPU hay GPU.
    """
    nvidia_gpus = _detect_nvidia_gpus_via_nvidia_smi()
    has_nvidia_gpu = len(nvidia_gpus) > 0

    # Biến trạng thái để in summary cuối cùng
    torch_installed = False
    torch_has_cuda_module = False
    torch_cuda_available = False
    will_use_gpu = False
    pytorch_mode = "chưa_cài"

    # 1. Trạng thái PyTorch
    try:
        import torch  # type: ignore
        torch_installed = True
    except Exception:
        print("\n[GPU-SUMMARY]")
        print(f"  1) Có GPU NVIDIA vật lý: {'CÓ' if has_nvidia_gpu else 'KHÔNG'}")
        print("  2) PyTorch: CHƯA CÀI")
        print("  3) CUDA trong PyTorch: KHÔNG (do chưa cài PyTorch)")
        print("  => KẾT LUẬN: Hệ thống sẽ chạy bằng CPU.")
        return

    # 3. PyTorch CPU-only hay GPU build?
    torch_has_cuda_module = bool(getattr(torch, 'cuda', None))
    if not torch_has_cuda_module:
        pytorch_mode = "cpu_only"
        print("\n[GPU-SUMMARY]")
        print(f"  1) Có GPU NVIDIA vật lý: {'CÓ' if has_nvidia_gpu else 'KHÔNG'}")
        print("  2) PyTorch: ĐÃ CÀI (bản CPU-only, KHÔNG hỗ trợ CUDA)")
        print("  3) CUDA trong PyTorch: KHÔNG (không có torch.cuda)")
        print("  => KẾT LUẬN: Hệ thống sẽ chạy bằng CPU.")
        return

    pytorch_mode = "gpu_build"

    # 4. CUDA/driver trong PyTorch
    try:
        torch_cuda_available = bool(torch.cuda.is_available())
        if not torch_cuda_available:
            print("\n[GPU-SUMMARY]")
            print(f"  1) Có GPU NVIDIA vật lý: {'CÓ' if has_nvidia_gpu else 'KHÔNG'}")
            print("  2) PyTorch: ĐÃ CÀI (bản GPU/CUDA)")
            print("  3) CUDA trong PyTorch: KHÔNG KHẢ DỤNG (torch.cuda.is_available() == False)")
            print("  => KẾT LUẬN: Hệ thống sẽ chạy bằng CPU.")
            return

        device_count = torch.cuda.device_count()
        if device_count <= 0:
            print("\n[GPU-SUMMARY]")
            print(f"  1) Có GPU NVIDIA vật lý: {'CÓ' if has_nvidia_gpu else 'KHÔNG'}")
            print("  2) PyTorch: ĐÃ CÀI (bản GPU/CUDA)")
            print("  3) CUDA trong PyTorch: CÓ nhưng không liệt kê được GPU (device_count == 0)")
            print("  => KẾT LUẬN: Hệ thống sẽ chạy bằng CPU.")
            return

        torch_gpu_names = []
        for i in range(device_count):
            try:
                name = torch.cuda.get_device_name(i)
                torch_gpu_names.append(name or f"GPU{i}")
            except Exception:
                torch_gpu_names.append(f"GPU{i}")

        print("\n[GPU-SUMMARY]")
        print(f"  1) Có GPU NVIDIA vật lý: {'CÓ' if has_nvidia_gpu else 'KHÔNG'}")
        print("  2) PyTorch: ĐÃ CÀI (bản GPU/CUDA)")
        print("  3) CUDA trong PyTorch: CÓ (torch.cuda.is_available() == True)")
        print("  => KẾT LUẬN: Hệ thống sẽ ƯU TIÊN chạy bằng GPU.")
    except Exception as e:
        print(f"[GPU] Lỗi khi kiểm tra CUDA với PyTorch => NGUYÊN NHÂN: {e}")
        print("       Hệ thống sẽ chạy trên CPU. Kiểm tra lại driver NVIDIA, CUDA và bản PyTorch GPU.")
        print("\n[GPU-SUMMARY]")
        print(f"  1) Có GPU NVIDIA vật lý: {'CÓ' if has_nvidia_gpu else 'KHÔNG'}")
        print(f"  2) PyTorch: {'ĐÃ CÀI' if torch_installed else 'CHƯA CÀI'} (bản GPU/CUDA)")
        print("  3) CUDA trong PyTorch: LỖI KHI KIỂM TRA")
        print("  => KẾT LUẬN: Hệ thống sẽ chạy bằng CPU.")


def run():
    """Main entry point"""
    # Capture stdout/stderr ngay từ đầu để dashboard hiển thị log như terminal
    from synapse.terminal_log import install as install_terminal_log
    install_terminal_log()

    # Kiểm tra môi trường GPU / PyTorch ngay khi khởi động
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
