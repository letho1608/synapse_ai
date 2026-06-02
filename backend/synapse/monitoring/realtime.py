"""
Realtime system resource monitor.

Periodically samples GPU utilization, VRAM usage, GPU power draw,
CPU load, and system RAM pressure across NVIDIA / AMD / Apple / CPU-only backends.

Provides the latest snapshot without blocking the main event loop and
feeds into:
  - ElectionManager dynamic weight (already existed but inline)
  - Resource scoring penalty (gpu_util_pct, ram_used_mb)
  - Dashboard /metrics and /system/stats endpoints
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class NodeMetrics:
    """Snapshot of live system resources (measured, NOT static capabilities)."""

    timestamp: float = 0.0

    # GPU
    gpu_util_pct: float = 0.0
    gpu_vram_used_mb: float = 0.0
    gpu_vram_total_mb: float = 0.0
    gpu_power_w: float = 0.0
    gpu_power_limit_w: float = 0.0
    gpu_temp_c: float = 0.0

    # CPU
    cpu_util_pct: float = 0.0

    # System RAM
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_util_pct: float = 0.0

    @property
    def vram_pressure(self) -> float:
        """0.0 = empty, 1.0 = full."""
        if self.gpu_vram_total_mb > 0:
            return self.gpu_vram_used_mb / self.gpu_vram_total_mb
        return 0.0

    @property
    def gpu_available_mb(self) -> float:
        return max(0.0, self.gpu_vram_total_mb - self.gpu_vram_used_mb)


class RealtimeMonitor:
    """Async-compatible cross-platform system resource sampler.

    Uses nvidia-smi when available, falls back to psutil for CPU/RAM only.
    All sampling is done in a background thread so the event loop is never
    blocked by a slow subprocess call.
    """

    SAMPLE_INTERVAL: float = 3.0

    def __init__(self):
        self._latest: NodeMetrics = NodeMetrics()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ── public API ──────────────────────────────────────────────

    @property
    def latest(self) -> NodeMetrics:
        return self._latest

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def sample_now(self) -> NodeMetrics:
        """Force an immediate sample (runs blocking work in thread)."""
        metrics = await asyncio.to_thread(_sample_blocking)
        self._latest = metrics
        return metrics

    # ── internal loop ───────────────────────────────────────────

    async def _sample_loop(self):
        while self._running:
            try:
                self._latest = await asyncio.to_thread(_sample_blocking)
            except Exception:
                pass
            await asyncio.sleep(self.SAMPLE_INTERVAL)


# ── shared sampler singleton ────────────────────────────────────

_monitor: Optional[RealtimeMonitor] = None


def get_monitor() -> RealtimeMonitor:
    """Lazy singleton.  Call start() on first use."""
    global _monitor
    if _monitor is None:
        _monitor = RealtimeMonitor()
    return _monitor


async def ensure_monitor() -> RealtimeMonitor:
    m = get_monitor()
    if not m._running:
        await m.start()
    return m


# ── blocking sampler (runs in thread) ───────────────────────────

def _sample_blocking() -> NodeMetrics:
    m = NodeMetrics(timestamp=time.time())

    _sample_nvidia_smi(m)
    _sample_cpu_ram(m)

    return m


def _sample_nvidia_smi(m: NodeMetrics) -> None:
    if not _has_nvidia_smi():
        return

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return

        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            try:
                gpu_util = _parse_float(parts[0])
                vram_used = _parse_float(parts[1])
                vram_total = _parse_float(parts[2])
                power_w = _parse_float(parts[3])
                power_limit = _parse_float(parts[4])
                temp_c = _parse_float(parts[5])
            except ValueError:
                continue

            m.gpu_util_pct = max(m.gpu_util_pct, gpu_util)
            m.gpu_vram_used_mb += vram_used
            m.gpu_vram_total_mb += vram_total
            m.gpu_power_w += power_w
            m.gpu_power_limit_w += power_limit
            m.gpu_temp_c = max(m.gpu_temp_c, temp_c)

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        pass


def _sample_cpu_ram(m: NodeMetrics) -> None:
    if psutil is None:
        return
    try:
        m.cpu_util_pct = psutil.cpu_percent(interval=0.05)
    except Exception:
        pass
    try:
        mem = psutil.virtual_memory()
        m.ram_total_mb = mem.total / (1024 * 1024)
        m.ram_used_mb = mem.used / (1024 * 1024)
        m.ram_util_pct = mem.percent
    except Exception:
        pass


def _has_nvidia_smi() -> bool:
    return bool(shutil.which("nvidia-smi"))


def _parse_float(val: str) -> float:
    if not val or val in ("[N/A]", "[Unknown Error]", "[Not Supported]"):
        return 0.0
    return float(val.replace("%", "").replace("W", "").replace("C", "").strip())