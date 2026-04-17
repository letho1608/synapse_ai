"""
System hardware detection — CPU, RAM, GPU (NVIDIA, AMD, Intel, Apple Silicon, Ascend).
Ported from llmit/llmfit_core/hardware.py and integrated directly into synapse.
Replaces the old windows_device_capabilities() logic with a full cross-platform implementation.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel
from synapse import DEBUG

TFLOPS = 1.00


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GpuBackend(Enum):
    CUDA = "CUDA"
    METAL = "Metal"
    ROCM = "ROCm"
    VULKAN = "Vulkan"
    SYCL = "SYCL"
    CPU_ARM = "CPU (ARM)"
    CPU_X86 = "CPU (x86)"
    ASCEND = "NPU (Ascend)"

    @property
    def label(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Pydantic models (kept for synapse compatibility)
# ---------------------------------------------------------------------------

class DeviceFlops(BaseModel):
    # units of TFLOPS
    fp32: float
    fp16: float
    int8: float

    def __str__(self):
        return f"fp32: {self.fp32 / TFLOPS:.2f} TFLOPS, fp16: {self.fp16 / TFLOPS:.2f} TFLOPS, int8: {self.int8 / TFLOPS:.2f} TFLOPS"

    def to_dict(self):
        return self.model_dump()


class DeviceCapabilities(BaseModel):
    model: str
    chip: str
    memory: int  # GPU VRAM (MB) when NVIDIA present, else system RAM (MB)
    flops: DeviceFlops
    cpu_cores: int = 0
    disk_gb: int = 0
    system_ram_mb: int = 0
    # Extended fields from llmit
    gpu_backend: str = "Unknown"
    unified_memory: bool = False
    gpu_count: int = 0
    total_gpu_vram_mb: int = 0
    # Network fields
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    warmup_throughput: float = 0.0
    # Real-time metrics
    cpu_usage_pct: float = 0.0
    gpu_usage_pct: float = 0.0
    ram_used_mb: int = 0
    gpu_memory_used_mb: int = 0

    def __str__(self):
        return (
            f"Model: {self.model}. Chip: {self.chip}. "
            f"Memory: {self.memory}MB. System RAM: {self.system_ram_mb}MB. "
            f"Flops: {self.flops}. CPU: {self.cpu_cores} cores. "
            f"Disk: {self.disk_gb} GB. Backend: {self.gpu_backend}. "
            f"Net: {self.bandwidth_mbps} Mbps / {self.latency_ms} ms. "
            f"Warmup: {self.warmup_throughput:.2f} samples/s"
        )

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.flops, dict):
            self.flops = DeviceFlops(**self.flops)

    def to_dict(self):
        return {
            "model": self.model,
            "chip": self.chip,
            "memory": self.memory,
            "flops": self.flops.to_dict(),
            "cpu_cores": self.cpu_cores,
            "disk_gb": self.disk_gb,
            "system_ram_mb": self.system_ram_mb,
            "gpu_backend": self.gpu_backend,
            "unified_memory": self.unified_memory,
            "gpu_count": self.gpu_count,
            "total_gpu_vram_mb": self.total_gpu_vram_mb,
            "bandwidth_mbps": self.bandwidth_mbps,
            "latency_ms": self.latency_ms,
            "warmup_throughput": self.warmup_throughput,
            "cpu_usage_pct": self.cpu_usage_pct,
            "gpu_usage_pct": self.gpu_usage_pct,
            "ram_used_mb": self.ram_used_mb,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
        }


UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(
    model="Unknown Model",
    chip="Unknown Chip",
    memory=0,
    flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    cpu_cores=0,
    disk_gb=0,
    system_ram_mb=0,
    bandwidth_mbps=0.0,
    latency_ms=0.0,
    warmup_throughput=0.0,
)


# ---------------------------------------------------------------------------
# Internal GPU detection data classes
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    """Information about a single detected GPU."""
    name: str
    vram_gb: Optional[float]
    backend: GpuBackend
    count: int = 1
    unified_memory: bool = False


@dataclass
class SystemSpecs:
    """Detected system hardware specifications."""
    total_ram_gb: float
    available_ram_gb: float
    total_cpu_cores: int
    cpu_name: str
    has_gpu: bool
    gpu_vram_gb: Optional[float]
    total_gpu_vram_gb: Optional[float]
    gpu_name: Optional[str]
    gpu_count: int
    unified_memory: bool
    backend: GpuBackend
    gpus: list = field(default_factory=list)

    @classmethod
    def detect(cls) -> "SystemSpecs":
        """Auto-detect system hardware specs."""
        import psutil

        total_ram_bytes = psutil.virtual_memory().total
        available_ram_bytes = psutil.virtual_memory().available
        total_ram_gb = total_ram_bytes / (1024.0 ** 3)
        available_ram_gb = available_ram_bytes / (1024.0 ** 3)

        if available_ram_gb <= 0 and total_ram_gb > 0:
            available_ram_gb = total_ram_gb * 0.8

        total_cpu_cores = psutil.cpu_count(logical=True) or 1
        cpu_name = _get_cpu_name()

        gpus = _detect_all_gpus(total_ram_gb, cpu_name)

        primary = gpus[0] if gpus else None
        has_gpu = len(gpus) > 0
        gpu_vram_gb = primary.vram_gb if primary else None
        total_gpu_vram_gb = (
            primary.vram_gb * primary.count if primary and primary.vram_gb else None
        )
        gpu_name = primary.name if primary else None
        gpu_count = primary.count if primary else 0
        unified_memory = primary.unified_memory if primary else False

        cpu_backend = (
            GpuBackend.CPU_ARM
            if platform.machine().lower() in ("arm64", "aarch64")
            or "apple" in cpu_name.lower()
            else GpuBackend.CPU_X86
        )
        backend = primary.backend if primary else cpu_backend

        return cls(
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            total_cpu_cores=total_cpu_cores,
            cpu_name=cpu_name,
            has_gpu=has_gpu,
            gpu_vram_gb=gpu_vram_gb,
            total_gpu_vram_gb=total_gpu_vram_gb,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            unified_memory=unified_memory,
            backend=backend,
            gpus=gpus,
        )


# ---------------------------------------------------------------------------
# Main device_capabilities function (synapse entry point)
# ---------------------------------------------------------------------------

async def device_capabilities() -> DeviceCapabilities:
    """
    Detect hardware capabilities. This replaces the old windows_device_capabilities().
    Uses the full cross-platform detection logic ported from llmit.
    """
    import psutil

    try:
        specs = SystemSpecs.detect()
    except Exception as e:
        if DEBUG >= 1:
            print(f"[device_capabilities] Hardware detection error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to minimal info
        import psutil
        cpu_cores = psutil.cpu_count(logical=True) or 0
        system_ram_mb = psutil.virtual_memory().total // 2**20
        return DeviceCapabilities(
            model="Unknown Device",
            chip="Unknown Chip",
            memory=system_ram_mb,
            flops=DeviceFlops(fp32=0, fp16=0, int8=0),
            cpu_cores=cpu_cores,
            disk_gb=0,
            system_ram_mb=system_ram_mb,
        )

    # Build TFLOPS from GPU name if possible
    gpu_name = specs.gpu_name or ""
    flops = _lookup_flops(gpu_name)

    # Memory in MB
    vram_mb = int((specs.gpu_vram_gb or 0) * 1024)
    total_vram_mb = int((specs.total_gpu_vram_gb or 0) * 1024)
    system_ram_mb = int(specs.total_ram_gb * 1024)

    cpu_cores = specs.total_cpu_cores

    try:
        root = "C:\\" if sys.platform == "win32" else "/"
        disk_gb = psutil.disk_usage(root).total // (1024**3)
    except Exception:
        disk_gb = 0

    # Model name: prefer GPU name, fallback to CPU
    if specs.has_gpu and gpu_name:
        model_name = f"{platform.system()} ({gpu_name})"
        chip = gpu_name
        memory = vram_mb if vram_mb > 0 else system_ram_mb
    else:
        model_name = f"{platform.system()} (CPU: {specs.cpu_name})"
        chip = specs.cpu_name
        memory = system_ram_mb

    if DEBUG >= 1:
        print(f"[device_capabilities] Detected: {model_name}, VRAM={vram_mb}MB, RAM={system_ram_mb}MB, Backend={specs.backend.label}")

    return DeviceCapabilities(
        model=model_name,
        chip=chip,
        memory=memory,
        flops=flops,
        cpu_cores=cpu_cores,
        disk_gb=disk_gb,
        system_ram_mb=system_ram_mb,
        gpu_backend=specs.backend.label,
        unified_memory=specs.unified_memory,
        gpu_count=specs.gpu_count,
        total_gpu_vram_mb=total_vram_mb,
    )


# ---------------------------------------------------------------------------
# TFLOPS lookup from GPU name (kept from synapse for backward compatibility)
# ---------------------------------------------------------------------------

CHIP_FLOPS = {
    # RTX 50 series
    "NVIDIA GEFORCE RTX 5090": DeviceFlops(fp32=218.0*TFLOPS, fp16=436.0*TFLOPS, int8=872.0*TFLOPS),
    "NVIDIA GEFORCE RTX 5080": DeviceFlops(fp32=137.0*TFLOPS, fp16=274.0*TFLOPS, int8=548.0*TFLOPS),
    "NVIDIA GEFORCE RTX 5070 TI": DeviceFlops(fp32=92.0*TFLOPS, fp16=184.0*TFLOPS, int8=368.0*TFLOPS),
    "NVIDIA GEFORCE RTX 5070": DeviceFlops(fp32=61.0*TFLOPS, fp16=122.0*TFLOPS, int8=244.0*TFLOPS),
    # RTX 40 series
    "NVIDIA GEFORCE RTX 4090": DeviceFlops(fp32=82.58*TFLOPS, fp16=165.16*TFLOPS, int8=330.32*TFLOPS),
    "NVIDIA GEFORCE RTX 4080 SUPER": DeviceFlops(fp32=52.0*TFLOPS, fp16=104.0*TFLOPS, int8=208.0*TFLOPS),
    "NVIDIA GEFORCE RTX 4080": DeviceFlops(fp32=48.74*TFLOPS, fp16=97.48*TFLOPS, int8=194.96*TFLOPS),
    "NVIDIA GEFORCE RTX 4070 TI SUPER": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
    "NVIDIA GEFORCE RTX 4070 TI": DeviceFlops(fp32=39.43*TFLOPS, fp16=78.86*TFLOPS, int8=157.72*TFLOPS),
    "NVIDIA GEFORCE RTX 4070 SUPER": DeviceFlops(fp32=30.0*TFLOPS, fp16=60.0*TFLOPS, int8=120.0*TFLOPS),
    "NVIDIA GEFORCE RTX 4070": DeviceFlops(fp32=29.0*TFLOPS, fp16=58.0*TFLOPS, int8=116.0*TFLOPS),
    "NVIDIA GEFORCE RTX 4060 TI 16GB": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
    "NVIDIA GEFORCE RTX 4060 TI": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
    "NVIDIA GEFORCE RTX 4060": DeviceFlops(fp32=15.1*TFLOPS, fp16=30.2*TFLOPS, int8=60.4*TFLOPS),
    # RTX 30 series
    "NVIDIA GEFORCE RTX 3090 TI": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
    "NVIDIA GEFORCE RTX 3090": DeviceFlops(fp32=35.6*TFLOPS, fp16=71.2*TFLOPS, int8=142.4*TFLOPS),
    "NVIDIA GEFORCE RTX 3080 TI": DeviceFlops(fp32=34.1*TFLOPS, fp16=68.2*TFLOPS, int8=136.4*TFLOPS),
    "NVIDIA GEFORCE RTX 3080 (12 GB)": DeviceFlops(fp32=30.6*TFLOPS, fp16=61.2*TFLOPS, int8=122.4*TFLOPS),
    "NVIDIA GEFORCE RTX 3080 (10 GB)": DeviceFlops(fp32=29.8*TFLOPS, fp16=59.6*TFLOPS, int8=119.2*TFLOPS),
    "NVIDIA GEFORCE RTX 3070 TI": DeviceFlops(fp32=21.8*TFLOPS, fp16=43.6*TFLOPS, int8=87.2*TFLOPS),
    "NVIDIA GEFORCE RTX 3070": DeviceFlops(fp32=20.3*TFLOPS, fp16=40.6*TFLOPS, int8=81.2*TFLOPS),
    "NVIDIA GEFORCE RTX 3060 TI": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
    "NVIDIA GEFORCE RTX 3060": DeviceFlops(fp32=13.0*TFLOPS, fp16=26.0*TFLOPS, int8=52.0*TFLOPS),
    "NVIDIA GEFORCE RTX 3050 TI LAPTOP GPU": DeviceFlops(fp32=7.2*TFLOPS, fp16=14.4*TFLOPS, int8=28.8*TFLOPS),
    "NVIDIA GEFORCE RTX 3050 TI": DeviceFlops(fp32=7.2*TFLOPS, fp16=14.4*TFLOPS, int8=28.8*TFLOPS),
    "NVIDIA GEFORCE RTX 3050": DeviceFlops(fp32=9.11*TFLOPS, fp16=18.22*TFLOPS, int8=36.44*TFLOPS),
    # RTX 20 series
    "NVIDIA TITAN RTX": DeviceFlops(fp32=16.31*TFLOPS, fp16=32.62*TFLOPS, int8=65.24*TFLOPS),
    "NVIDIA GEFORCE RTX 2080 TI": DeviceFlops(fp32=13.45*TFLOPS, fp16=26.9*TFLOPS, int8=40.28*TFLOPS),
    "NVIDIA GEFORCE RTX 2080 SUPER": DeviceFlops(fp32=11.15*TFLOPS, fp16=22.30*TFLOPS, int8=44.60*TFLOPS),
    "NVIDIA GEFORCE RTX 2080": DeviceFlops(fp32=10.07*TFLOPS, fp16=20.14*TFLOPS, int8=40.28*TFLOPS),
    "NVIDIA GEFORCE RTX 2070 SUPER": DeviceFlops(fp32=9.06*TFLOPS, fp16=18.12*TFLOPS, int8=36.24*TFLOPS),
    "NVIDIA GEFORCE RTX 2070": DeviceFlops(fp32=7.46*TFLOPS, fp16=14.93*TFLOPS, int8=29.86*TFLOPS),
    "NVIDIA GEFORCE RTX 2060 SUPER": DeviceFlops(fp32=7.2*TFLOPS, fp16=14.4*TFLOPS, int8=28.8*TFLOPS),
    "NVIDIA GEFORCE RTX 2060": DeviceFlops(fp32=6.45*TFLOPS, fp16=12.9*TFLOPS, int8=25.8*TFLOPS),
    # GTX
    "NVIDIA GEFORCE GTX 1080 TI": DeviceFlops(fp32=11.34*TFLOPS, fp16=0.177*TFLOPS, int8=45.36*TFLOPS),
    "NVIDIA GEFORCE GTX 1080": DeviceFlops(fp32=8.873*TFLOPS, fp16=0.138*TFLOPS, int8=35.492*TFLOPS),
    "NVIDIA GEFORCE GTX 1070": DeviceFlops(fp32=6.463*TFLOPS, fp16=0.101*TFLOPS, int8=25.852*TFLOPS),
    "NVIDIA GEFORCE GTX 1660 TI": DeviceFlops(fp32=4.8*TFLOPS, fp16=9.6*TFLOPS, int8=19.2*TFLOPS),
    "NVIDIA GEFORCE GTX 1050 TI": DeviceFlops(fp32=2.0*TFLOPS, fp16=4.0*TFLOPS, int8=8.0*TFLOPS),
    # Server / Workstation GPU
    "NVIDIA A100 80GB SXM": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
    "NVIDIA A100 80GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
    "NVIDIA A100 40GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
    "NVIDIA RTX A6000": DeviceFlops(fp32=38.71*TFLOPS, fp16=38.71*TFLOPS, int8=154.84*TFLOPS),
    "NVIDIA RTX A5000": DeviceFlops(fp32=27.8*TFLOPS, fp16=27.8*TFLOPS, int8=111.2*TFLOPS),
    "NVIDIA RTX A4000": DeviceFlops(fp32=19.17*TFLOPS, fp16=19.17*TFLOPS, int8=76.68*TFLOPS),
    "NVIDIA RTX A2000": DeviceFlops(fp32=7.99*TFLOPS, fp16=7.99*TFLOPS, int8=31.91*TFLOPS),
}


def _normalize_gpu_name(raw: Any) -> str:
    """Normalize GPU name: decode bytes, uppercase, strip, collapse spaces."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    s = str(raw).upper().strip()
    return " ".join(s.split())


def _lookup_flops(gpu_name: str) -> DeviceFlops:
    """Lookup TFLOPS from GPU name: exact match then partial match fallback."""
    name = _normalize_gpu_name(gpu_name)
    if not name:
        return DeviceFlops(fp32=0, fp16=0, int8=0)
    if name in CHIP_FLOPS:
        return CHIP_FLOPS[name]
    if name.endswith("GB"):
        name = name.rsplit(" ", 1)[0].strip()
    if name in CHIP_FLOPS:
        return CHIP_FLOPS[name]
    for key in sorted(CHIP_FLOPS.keys(), key=len, reverse=True):
        if key in name or name.startswith(key) or name in key:
            return CHIP_FLOPS[key]
    return DeviceFlops(fp32=0, fp16=0, int8=0)


# ---------------------------------------------------------------------------
# Internal helpers ported from llmit/llmfit_core/hardware.py
# ---------------------------------------------------------------------------

def _get_cpu_name() -> str:
    """Get CPU brand name."""
    try:
        import cpuinfo  # type: ignore
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", "Unknown CPU")
    except Exception:
        pass

    cpu = platform.processor()
    if cpu:
        return cpu

    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    return "Unknown CPU"


def _run_command(cmd: list, timeout: int = 10) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def _detect_all_gpus(total_ram_gb: float, cpu_name: str) -> list:
    """Detect all GPUs across all vendors. Returns sorted by VRAM descending."""
    gpus: list = []

    # NVIDIA GPUs via nvidia-smi
    nvidia = _detect_nvidia_gpus()
    if not nvidia:
        nvidia_sysfs = _detect_nvidia_gpu_sysfs_info()
        if nvidia_sysfs:
            gpus.append(nvidia_sysfs)
    else:
        gpus.extend(nvidia)

    # AMD GPUs via rocm-smi or sysfs
    amd = _detect_amd_gpu_rocm_info()
    if amd:
        gpus.append(amd)
    else:
        amd = _detect_amd_gpu_sysfs_info()
        if amd:
            gpus.append(amd)

    # Windows WMI
    for wmi_gpu in _detect_gpu_windows_info():
        dominated = any(
            existing.name.lower() in wmi_gpu.name.lower()
            or wmi_gpu.name.lower() in existing.name.lower()
            for existing in gpus
        )
        if not dominated:
            gpus.append(wmi_gpu)

    # AMD unified memory APUs
    if _is_amd_unified_memory_apu(cpu_name):
        amd_idx = None
        for i, g in enumerate(gpus):
            lower = g.name.lower()
            if "amd" in lower or "radeon" in lower:
                amd_idx = i
                break
        if amd_idx is not None:
            gpus[amd_idx].unified_memory = True
            gpus[amd_idx].vram_gb = total_ram_gb
        else:
            gpus.append(GpuInfo(
                name=f"{cpu_name} (integrated)",
                vram_gb=total_ram_gb,
                backend=GpuBackend.VULKAN,
                count=1,
                unified_memory=True,
            ))

    # Intel Arc via sysfs
    intel_vram = _detect_intel_gpu()
    if intel_vram is not None:
        already_found = any("intel" in g.name.lower() for g in gpus)
        if not already_found:
            gpus.append(GpuInfo(
                name="Intel Arc",
                vram_gb=intel_vram,
                backend=GpuBackend.SYCL,
                count=1,
                unified_memory=False,
            ))

    # Apple Silicon (unified memory)
    apple_vram = _detect_apple_gpu(total_ram_gb)
    if apple_vram is not None:
        name = cpu_name if "apple" in cpu_name.lower() else "Apple Silicon"
        gpus.append(GpuInfo(
            name=name,
            vram_gb=apple_vram,
            backend=GpuBackend.METAL,
            count=1,
            unified_memory=True,
        ))

    # Ascend NPUs
    ascend = _detect_ascend_npus()
    if ascend:
        gpus.extend(ascend)

    # Sort by VRAM descending
    gpus.sort(key=lambda g: g.vram_gb or 0.0, reverse=True)
    return gpus


def _detect_nvidia_gpus() -> list:
    """Detect NVIDIA GPUs via nvidia-smi."""
    result = _try_nvidia_smi_with_addressing_mode()
    if result is not None:
        return result
    text = _run_command([
        "nvidia-smi", "--query-gpu=memory.total,name",
        "--format=csv,noheader,nounits",
    ])
    if not text:
        return []
    return _parse_nvidia_smi_list(text)


def _try_nvidia_smi_with_addressing_mode() -> Optional[list]:
    text = _run_command([
        "nvidia-smi", "--query-gpu=addressing_mode,memory.total,name",
        "--format=csv,noheader,nounits",
    ])
    if text is None:
        return None
    return _parse_nvidia_smi_extended(text)


def _parse_nvidia_smi_extended(text: str) -> list:
    grouped: OrderedDict = OrderedDict()
    total_ram_gb = _read_proc_meminfo_total_gb()

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 2)
        if len(parts) < 3:
            continue

        addr_mode = parts[0].strip()
        is_unified = addr_mode.upper() == "ATS"
        name = parts[2].strip() or "NVIDIA GPU"

        try:
            parsed_vram_mb = float(parts[1].strip())
        except ValueError:
            parsed_vram_mb = 0.0

        if parsed_vram_mb > 0:
            vram_mb = parsed_vram_mb
        elif is_unified and total_ram_gb:
            vram_mb = total_ram_gb * 1024.0
        else:
            vram_mb = estimate_vram_from_name(name) * 1024.0

        if name not in grouped:
            grouped[name] = (0, 0.0, False)
        count, max_vram, was_unified = grouped[name]
        grouped[name] = (count + 1, max(max_vram, vram_mb), was_unified or is_unified)

    if not grouped:
        return []

    return [
        GpuInfo(
            name=name,
            vram_gb=vram_mb / 1024.0 if vram_mb > 0 else None,
            backend=GpuBackend.CUDA,
            count=count,
            unified_memory=is_unified,
        )
        for name, (count, vram_mb, is_unified) in grouped.items()
    ]


def _parse_nvidia_smi_list(text: str) -> list:
    grouped: OrderedDict = OrderedDict()

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "NVIDIA GPU"

        try:
            parsed_vram_mb = float(parts[0].strip())
        except ValueError:
            parsed_vram_mb = 0.0

        vram_mb = parsed_vram_mb if parsed_vram_mb > 0 else estimate_vram_from_name(name) * 1024.0

        if name not in grouped:
            grouped[name] = (0, 0.0)
        count, max_vram = grouped[name]
        grouped[name] = (count + 1, max(max_vram, vram_mb))

    if not grouped:
        return []

    return [
        GpuInfo(
            name=name,
            vram_gb=vram_mb / 1024.0 if vram_mb > 0 else None,
            backend=GpuBackend.CUDA,
            count=count,
            unified_memory=False,
        )
        for name, (count, vram_mb) in grouped.items()
    ]


def _detect_nvidia_gpu_sysfs_info() -> Optional[GpuInfo]:
    if sys.platform != "linux":
        return None
    try:
        entries = os.listdir("/sys/class/drm")
    except OSError:
        return None

    gpu_count = 0
    total_vram_bytes = 0
    backend = GpuBackend.VULKAN

    for fname in entries:
        if not fname.startswith("card") or "-" in fname:
            continue
        device_path = f"/sys/class/drm/{fname}/device"
        vendor_path = f"{device_path}/vendor"
        try:
            vendor = open(vendor_path).read().strip()
        except OSError:
            continue
        if vendor != "0x10de":
            continue
        gpu_count += 1
        try:
            vram_str = open(f"{device_path}/mem_info_vram_total").read().strip()
            vram_bytes = int(vram_str)
            if vram_bytes > 0:
                total_vram_bytes = max(total_vram_bytes, vram_bytes)
        except (OSError, ValueError):
            pass
        try:
            uevent = open(f"{device_path}/uevent").read()
            for line in uevent.splitlines():
                if line.startswith("DRIVER=") and line.split("=", 1)[1].lower() == "nvidia":
                    backend = GpuBackend.CUDA
        except OSError:
            pass

    if gpu_count == 0:
        return None

    name = "NVIDIA GPU"
    vram_gb = total_vram_bytes / (1024.0 ** 3) if total_vram_bytes > 0 else None
    if vram_gb is None:
        est = estimate_vram_from_name(name)
        if est > 0:
            vram_gb = est
    return GpuInfo(name=name, vram_gb=vram_gb, backend=backend, count=gpu_count, unified_memory=False)


def _detect_amd_gpu_rocm_info() -> Optional[GpuInfo]:
    text = _run_command(["rocm-smi", "--showmeminfo", "vram"])
    if text is None:
        return None

    per_gpu_vram_bytes: list = []
    gpu_count = 0
    for line in text.splitlines():
        lower = line.lower()
        if "total" in lower and "used" not in lower:
            nums = [int(w) for w in line.split() if w.isdigit()]
            if nums and nums[-1] > 0:
                per_gpu_vram_bytes.append(nums[-1])
                gpu_count += 1

    if gpu_count == 0:
        gpu_count = 1

    gpu_name = None
    name_text = _run_command(["rocm-smi", "--showproductname"])
    if name_text:
        for line in name_text.splitlines():
            lower = line.lower()
            if ("card series" in lower or "card model" in lower) and ":" in line:
                val = line.split(":", 1)[1].strip()
                if val:
                    gpu_name = val
                    break

    name = gpu_name or "AMD GPU"
    max_vram = max(per_gpu_vram_bytes) if per_gpu_vram_bytes else 0
    if max_vram > 0:
        vram_gb: Optional[float] = max_vram / (1024.0 ** 3)
    else:
        est = estimate_vram_from_name(name)
        vram_gb = est if est > 0 else None

    return GpuInfo(name=name, vram_gb=vram_gb, backend=GpuBackend.ROCM, count=gpu_count, unified_memory=False)


def _detect_amd_gpu_sysfs_info() -> Optional[GpuInfo]:
    if sys.platform != "linux":
        return None
    try:
        entries = os.listdir("/sys/class/drm")
    except OSError:
        return None

    for fname in entries:
        if not fname.startswith("card") or "-" in fname:
            continue
        device_path = f"/sys/class/drm/{fname}/device"
        try:
            vendor = open(f"{device_path}/vendor").read().strip()
        except OSError:
            continue
        if vendor != "0x1002":
            continue
        vram_gb = None
        try:
            vram_str = open(f"{device_path}/mem_info_vram_total").read().strip()
            vram_bytes = int(vram_str)
            if vram_bytes > 0:
                vram_gb = vram_bytes / (1024.0 ** 3)
        except (OSError, ValueError):
            pass
        name = "AMD GPU"
        if vram_gb is None:
            est = estimate_vram_from_name(name)
            if est > 0:
                vram_gb = est
        return GpuInfo(name=name, vram_gb=vram_gb, backend=GpuBackend.VULKAN, count=1, unified_memory=False)

    return None


def _detect_gpu_windows_info() -> list:
    if sys.platform != "win32":
        return []

    text = _run_command([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | "
        "ForEach-Object { $_.Name + '|' + $_.AdapterRAM }",
    ], timeout=15)
    if text:
        gpus = _parse_windows_gpu_list(text)
        if gpus:
            return gpus

    text = _run_command([
        "wmic", "path", "win32_VideoController", "get",
        "Name,AdapterRAM", "/format:csv",
    ], timeout=15)
    if text:
        return _parse_windows_wmic_list(text)
    return []


def _parse_windows_gpu_list(text: str) -> list:
    gpus: list = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 1)
        name = parts[0].strip()
        try:
            raw_vram = int(parts[1].strip()) if len(parts) > 1 else 0
        except ValueError:
            raw_vram = 0

        lower = name.lower()
        if any(kw in lower for kw in ("microsoft", "basic", "virtual")) or not name:
            continue

        backend = _infer_gpu_backend(name)
        vram_gb = _resolve_wmi_vram(raw_vram, name)
        gpus.append(GpuInfo(name=name, vram_gb=vram_gb, backend=backend, count=1, unified_memory=False))
    return gpus


def _parse_windows_wmic_list(text: str) -> list:
    gpus: list = []
    lines = text.strip().splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                raw_vram = int(parts[1].strip())
            except ValueError:
                raw_vram = 0
            name = ",".join(parts[2:]).strip()
            lower = name.lower()
            if any(kw in lower for kw in ("microsoft", "basic", "virtual")):
                continue
            backend = _infer_gpu_backend(name)
            vram_gb = _resolve_wmi_vram(raw_vram, name)
            gpus.append(GpuInfo(name=name, vram_gb=vram_gb, backend=backend, count=1, unified_memory=False))
    return gpus


def _resolve_wmi_vram(raw_bytes: int, name: str) -> Optional[float]:
    """WMI AdapterRAM is 32-bit capped at ~4 GB. Estimate from name if needed."""
    vram_gb = raw_bytes / (1024.0 ** 3)
    if vram_gb < 0.1 or (vram_gb <= 4.1 and estimate_vram_from_name(name) > 4.1):
        estimated = estimate_vram_from_name(name)
        if estimated > 0:
            vram_gb = estimated
    return vram_gb if vram_gb > 0 else None


def _infer_gpu_backend(name: str) -> GpuBackend:
    lower = name.lower()
    if any(kw in lower for kw in ("nvidia", "geforce", "quadro", "tesla", "rtx")):
        return GpuBackend.CUDA
    elif any(kw in lower for kw in ("amd", "radeon", "ati")):
        return GpuBackend.VULKAN
    elif any(kw in lower for kw in ("intel", "arc")):
        return GpuBackend.SYCL
    return GpuBackend.VULKAN


def _detect_intel_gpu() -> Optional[float]:
    if sys.platform != "linux":
        return None
    try:
        entries = os.listdir("/sys/class/drm")
    except OSError:
        return None
    for fname in entries:
        device_path = f"/sys/class/drm/{fname}/device"
        try:
            vendor = open(f"{device_path}/vendor").read().strip()
        except OSError:
            continue
        if vendor != "0x8086":
            continue
        try:
            vram_str = open(f"{device_path}/mem_info_vram_total").read().strip()
            vram_bytes = int(vram_str)
            if vram_bytes > 0:
                return vram_bytes / (1024.0 ** 3)
        except (OSError, ValueError):
            pass
    return None


def _detect_apple_gpu(total_ram_gb: float) -> Optional[float]:
    if sys.platform != "darwin":
        return None
    text = _run_command(["system_profiler", "SPDisplaysDataType"])
    if not text:
        return None
    is_apple = any(
        "apple m" in line.lower() or "apple gpu" in line.lower()
        for line in text.splitlines()
    )
    return total_ram_gb if is_apple else None


def _detect_ascend_npus() -> list:
    list_text = _run_command(["npu-smi", "info", "-l"])
    if not list_text:
        return []

    ids = []
    for line in list_text.splitlines():
        if "NPU ID" in line and ":" in line:
            npu_id = line.split(":")[-1].strip()
            ids.append(npu_id)

    if not ids:
        return []

    infos: list = []
    for npu_id in ids:
        mem_text = _run_command(["npu-smi", "info", "-t", "memory", "-i", npu_id])
        mem_mb = 0
        if mem_text:
            for line in mem_text.splitlines():
                if "HBM Capacity" in line and ":" in line:
                    val = line.split(":")[-1].strip().split()[0]
                    try:
                        mem_mb = int(val)
                    except ValueError:
                        pass
                    break
        infos.append(GpuInfo(
            name="Ascend NPU",
            vram_gb=mem_mb / 1024.0,
            backend=GpuBackend.ASCEND,
            count=1,
            unified_memory=False,
        ))
    return infos


def _is_amd_unified_memory_apu(cpu_name: str) -> bool:
    return "ryzen ai" in cpu_name.lower()


def _read_proc_meminfo_total_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except (OSError, ValueError):
        pass
    return None


def estimate_vram_from_name(name: str) -> float:
    """Fallback VRAM estimation from GPU model name (ported from llmit)."""
    lower = name.lower()
    # RTX 50 series
    if "5090" in lower: return 32.0
    if "5080" in lower: return 16.0
    if "5070 ti" in lower: return 16.0
    if "5070" in lower: return 12.0
    if "5060 ti" in lower: return 16.0
    if "5060" in lower: return 8.0
    # RTX 40 series
    if "4090" in lower: return 24.0
    if "4080" in lower: return 16.0
    if "4070 ti" in lower: return 12.0
    if "4070" in lower: return 12.0
    if "4060 ti" in lower: return 16.0
    if "4060" in lower: return 8.0
    # RTX 30 series
    if "3090" in lower: return 24.0
    if "3080 ti" in lower: return 12.0
    if "3080" in lower: return 10.0
    if "3070" in lower: return 8.0
    if "3060 ti" in lower: return 8.0
    if "3060" in lower: return 12.0
    # Data center
    if "h100" in lower: return 80.0
    if "a100" in lower: return 80.0
    if "l40" in lower: return 48.0
    if "a10" in lower: return 24.0
    if "t4" in lower: return 16.0
    # AMD RX 7000 series
    if "7900 xtx" in lower: return 24.0
    if "7900" in lower: return 20.0
    if "7800" in lower: return 16.0
    if "7700" in lower: return 12.0
    if "7600" in lower: return 8.0
    # AMD RX 6000 series
    if "6950" in lower: return 16.0
    if "6900" in lower: return 16.0
    if "6800" in lower: return 16.0
    if "6700" in lower: return 12.0
    if "6600" in lower: return 8.0
    if "6500" in lower: return 4.0
    # Generic fallbacks
    if "rtx" in lower: return 8.0
    if "gtx" in lower: return 4.0
    if "rx " in lower or "radeon" in lower: return 8.0
    return 0.0
