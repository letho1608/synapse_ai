"""
LACP — Latency-Aware Compute Pooling Resource Weight Engine

Computes a unified resource score for each node based on:
- GPU VRAM (primary for layer capacity)
- TFLOPS (primary for compute speed)
- CPU cores (for pre/post processing parallelism)
- System RAM (for KV cache offloading, embedding storage)
- Disk I/O throughput (for model weight streaming)
- Network bandwidth (for tensor forwarding throughput)
- Network latency to master (for pipeline stage assignment)
- Power envelope (thermal/power-aware weighting)

All resources are normalized to 0.0-1.0 range and combined via a
configurable weighted sum to produce a final "compute capacity" score.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# ──────────────────────────────────────────────────────────
# Resource measurement
# ──────────────────────────────────────────────────────────


@dataclass
class NodeResources:
    """Measured (or provided) resources for a single node."""

    # GPU
    gpu_name: str = ""
    gpu_vram_mb: float = 0.0
    gpu_tflops_fp16: float = 0.0
    gpu_tflops_fp32: float = 0.0
    gpu_count: int = 0
    gpu_util_pct: float = 0.0  # current GPU utilization 0-100

    # CPU
    cpu_cores: int = 0
    cpu_model: str = ""
    cpu_util_pct: float = 0.0

    # Memory
    system_ram_mb: float = 0.0
    system_ram_used_mb: float = 0.0

    # Disk
    disk_read_mbps: float = 0.0   # sequential read MB/s
    disk_write_mbps: float = 0.0  # sequential write MB/s

    # Network (filled by latency matrix probing)
    bandwidth_mbps: float = 0.0   # measured to master
    latency_ms: float = 0.0       # measured to master
    jitter_ms: float = 0.0        # latency variance

    # Power
    power_limit_w: float = 0.0    # TDP/TGP in watts
    current_power_w: float = 0.0  # current draw

    @classmethod
    def from_device_capabilities(cls, dc) -> "NodeResources":
        """Convert existing DeviceCapabilities to NodeResources."""
        flops = dc.flops if hasattr(dc, "flops") else {}
        # DeviceFlops is a pydantic BaseModel, access as attributes not dict
        if hasattr(flops, "fp16"):
            fp16_val = flops.fp16
        elif isinstance(flops, dict):
            fp16_val = flops.get("fp16", 0)
        else:
            fp16_val = 0

        if hasattr(flops, "fp32"):
            fp32_val = flops.fp32
        elif isinstance(flops, dict):
            fp32_val = flops.get("fp32", 0)
        else:
            fp32_val = 0

        # Measure disk I/O, power, and other resources from live system
        disk_metrics = _measure_disk_io()
        power_metrics = _measure_power()
        bandwidth_metrics = _measure_dummy_bandwidth()  # Overwritten by latency probing later

        return cls(
            gpu_name=getattr(dc, "chip", "") or getattr(dc, "model", ""),
            gpu_vram_mb=float(getattr(dc, "memory", 0)),
            gpu_tflops_fp16=float(fp16_val),
            gpu_tflops_fp32=float(fp32_val),
            cpu_cores=int(getattr(dc, "cpu_cores", 0) or 0),
            system_ram_mb=float(getattr(dc, "system_ram_mb", 0) or 0),
            disk_read_mbps=disk_metrics.get("read_mbps", 0.0),
            disk_write_mbps=disk_metrics.get("write_mbps", 0.0),
            bandwidth_mbps=bandwidth_metrics.get("bandwidth_mbps", 0.0),
            power_limit_w=power_metrics.get("power_limit_w", 0.0),
            current_power_w=power_metrics.get("current_power_w", 0.0),
        )


# ──────────────────────────────────────────────────────────
# Live resource measurement helpers
# ──────────────────────────────────────────────────────────


def _measure_disk_io() -> Dict[str, float]:
    """Measure disk I/O throughput using psutil.
    
    Returns dict with read_mbps and write_mbps (MB/s).
    Falls back to 0 values if psutil is unavailable or measurement fails.
    """
    result = {"read_mbps": 0.0, "write_mbps": 0.0}
    if not HAS_PSUTIL:
        return result
    try:
        # Sample disk I/O counters twice with a short interval
        before = psutil.disk_io_counters()
        if before is None:
            return result
        time.sleep(0.5)
        after = psutil.disk_io_counters()
        if after is None:
            return result
        
        elapsed = 0.5  # sleep duration
        # Convert bytes to MB, divide by elapsed seconds
        read_bytes = after.read_bytes - before.read_bytes
        write_bytes = after.write_bytes - before.write_bytes
        result["read_mbps"] = (read_bytes / (1024 * 1024)) / elapsed
        result["write_mbps"] = (write_bytes / (1024 * 1024)) / elapsed
    except Exception:
        pass
    return result


def _measure_power() -> Dict[str, float]:
    """Measure GPU power using pynvml, or system power via psutil.
    
    Returns dict with power_limit_w (TDP) and current_power_w.
    Falls back to 0 values if no measurement is possible.
    """
    result = {"power_limit_w": 0.0, "current_power_w": 0.0}
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    result["power_limit_w"] = power_limit / 1000.0  # milliwatts to watts
                except Exception:
                    pass
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
                    result["current_power_w"] = power_usage / 1000.0  # milliwatts to watts
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    elif HAS_PSUTIL:
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                result["current_power_w"] = battery.power_plugged * 50.0  # rough estimate
                result["power_limit_w"] = result["current_power_w"] * 1.2
        except Exception:
            pass
    return result


def _measure_dummy_bandwidth() -> Dict[str, float]:
    """Return a placeholder bandwidth value.
    
    Real bandwidth is measured during latency probing (latency_matrix.py)
    and populated into NodeResources.bandwidth_mbps afterward.
    This value is a reasonable default for single-node operation.
    """
    return {"bandwidth_mbps": 1000.0}  # 1 Gbps default, overwritten by probing


def apply_measured_bandwidth(
    nodes_resources: Dict[str, NodeResources],
    latency_matrix: Optional["LatencyMatrix"],
    self_node_id: str = "",
) -> None:
    """
    Copy measured network bandwidth from the latency matrix into each node's
    NodeResources so composite scores (LACP partitioning + master election)
    reflect real network capacity instead of the 1000 Mbps placeholder.

    Uses the entry from self_node_id -> node when available (bandwidth to the
    master is the figure NodeResources.bandwidth_mbps documents), otherwise
    falls back to any matrix entry touching the node.
    """
    if not latency_matrix:
        return
    for nid in nodes_resources:
        entry = None
        if self_node_id:
            entry = latency_matrix.get(self_node_id, nid)
        if entry is None or entry.bandwidth_mbps <= 0:
            for f, to_entries in latency_matrix.entries.items():
                e = to_entries.get(nid)
                if e and e.bandwidth_mbps > 0:
                    entry = e
                    break
        if entry and entry.bandwidth_mbps > 0:
            nodes_resources[nid].bandwidth_mbps = float(entry.bandwidth_mbps)


# ──────────────────────────────────────────────────────────
# Resource normalization → 0.0-1.0 scores
# ──────────────────────────────────────────────────────────


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to [0, 1] range. Returns 0 if max_val == 0."""
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


@dataclass
class ResourceWeights:
    """
    Configurable weights for the LACP composite score.
    Sums to 1.0. Adjust per deployment scenario.
    """
    vram: float = 0.25
    tflops: float = 0.25
    cpu: float = 0.10
    ram: float = 0.10
    disk: float = 0.05
    bandwidth: float = 0.10
    latency: float = 0.10
    power_headroom: float = 0.05

    def __post_init__(self):
        total = sum([
            self.vram, self.tflops, self.cpu, self.ram,
            self.disk, self.bandwidth, self.latency, self.power_headroom
        ])
        if abs(total - 1.0) > 0.001:
            scale = 1.0 / total
            self.vram *= scale
            self.tflops *= scale
            self.cpu *= scale
            self.ram *= scale
            self.disk *= scale
            self.bandwidth *= scale
            self.latency *= scale
            self.power_headroom *= scale


@dataclass
class NormalizedScores:
    """0.0-1.0 normalized resource scores for a single node."""
    vram: float = 0.0
    tflops: float = 0.0
    cpu: float = 0.0
    ram: float = 0.0
    disk: float = 0.0
    bandwidth: float = 0.0
    latency: float = 0.0   # inverted: 1.0 = lowest latency
    power_headroom: float = 0.0


class ResourceScorer:
    """
    Normalizes all node resources against the cluster's range and
    produces a weighted composite score for each node.
    """

    def __init__(self, weights: ResourceWeights = None):
        self.weights = weights or ResourceWeights()

    def compute_scores(
        self, nodes: Dict[str, NodeResources]
    ) -> Dict[str, Tuple[float, NormalizedScores]]:
        """
        Returns {node_id: (composite_score, NormalizedScores)}.
        Composite score drives layer allocation and master election.
        """
        if not nodes:
            return {}

        ids = list(nodes.keys())
        resources = [nodes[nid] for nid in ids]

        # Collect ranges for normalization
        def _range(getter):
            vals = [getter(r) for r in resources]
            return min(vals), max(vals)

        vram_min, vram_max = _range(lambda r: r.gpu_vram_mb)
        tflops_min, tflops_max = _range(lambda r: r.gpu_tflops_fp16 + r.gpu_tflops_fp32)
        cpu_min, cpu_max = _range(lambda r: r.cpu_cores)
        ram_min, ram_max = _range(lambda r: r.system_ram_mb)
        disk_min, disk_max = _range(lambda r: r.disk_read_mbps + r.disk_write_mbps)
        bw_min, bw_max = _range(lambda r: r.bandwidth_mbps)
        lat_min, lat_max = _range(lambda r: r.latency_ms)
        pwr_min, pwr_max = _range(lambda r: r.power_limit_w - r.current_power_w)

        # True only when every node reports identical values on every resource
        # dimension. In that case normalization collapses to 0 for everyone and
        # the raw-utilization formula below is the only differentiator. When
        # the cluster is heterogeneous, a node that is the min on every
        # dimension genuinely has zero normalized resources and must NOT be
        # boosted by the identical-cluster fallback (that would let the weakest
        # node win master election).
        cluster_identical = (
            vram_min == vram_max
            and tflops_min == tflops_max
            and cpu_min == cpu_max
            and ram_min == ram_max
            and disk_min == disk_max
            and bw_min == bw_max
            and lat_min == lat_max
            and pwr_min == pwr_max
        )

        results = {}
        for nid, r in zip(ids, resources):
            tflops = r.gpu_tflops_fp16 + r.gpu_tflops_fp32
            disk = r.disk_read_mbps + r.disk_write_mbps
            pwr_headroom = r.power_limit_w - r.current_power_w

            ns = NormalizedScores(
                vram=normalize(r.gpu_vram_mb, vram_min, vram_max),
                tflops=normalize(tflops, tflops_min, tflops_max),
                cpu=normalize(r.cpu_cores, cpu_min, cpu_max),
                ram=normalize(r.system_ram_mb, ram_min, ram_max),
                disk=normalize(disk, disk_min, disk_max),
                bandwidth=normalize(r.bandwidth_mbps, bw_min, bw_max),
                latency=1.0 - normalize(r.latency_ms, lat_min, lat_max),
                power_headroom=normalize(pwr_headroom, pwr_min, pwr_max),
            )

            # CRITICAL: GPU utilization and RAM pressure are applied as
            # multiplicative penalties on the normalized scores. When all nodes
            # have identical base resources (normalize returns 0), we still
            # differentiate by applying the penalty to the composite score directly.
            gpu_util_penalty = max(0.1, 1.0 - (r.gpu_util_pct / 100.0) * 0.9)

            if r.system_ram_mb > 0:
                ram_pressure = r.system_ram_used_mb / r.system_ram_mb
                ram_penalty = max(0.1, 1.0 - ram_pressure * 0.9)
            else:
                ram_penalty = 1.0

            base_composite = (
                self.weights.vram * ns.vram
                + self.weights.tflops * ns.tflops
                + self.weights.cpu * ns.cpu
                + self.weights.ram * ns.ram
                + self.weights.disk * ns.disk
                + self.weights.bandwidth * ns.bandwidth
                + self.weights.latency * ns.latency
                + self.weights.power_headroom * ns.power_headroom
            )

            # Apply utilization penalties proportionally to the weights.
            # The raw-utilization formula is used only when the whole cluster
            # is identical (base_composite == 0 for everyone); otherwise a node
            # with zero normalized resources keeps its true (low) score.
            if base_composite > 0 or not cluster_identical:
                composite = base_composite * (
                    1.0
                    + self.weights.tflops * (gpu_util_penalty - 1.0)
                    + self.weights.ram * (ram_penalty - 1.0)
                )
            else:
                # All resources identical — differentiate by utilization alone
                composite = (
                    self.weights.tflops * gpu_util_penalty
                    + self.weights.ram * ram_penalty
                    + (self.weights.cpu + self.weights.disk + self.weights.bandwidth
                       + self.weights.latency + self.weights.power_headroom + self.weights.vram) * 0.5
                )

            results[nid] = (composite, ns)

        return results

    def layer_allocation(
        self,
        scores: Dict[str, Tuple[float, NormalizedScores]],
        total_layers: int,
    ) -> Dict[str, int]:
        """
        Allocate layers to nodes based on composite scores using
        the Largest Remainder Method (integer-optimal distribution).

        Returns {node_id: layer_count}.
        """
        if not scores:
            return {}

        weights = {nid: sc[0] for nid, sc in scores.items()}
        total_weight = sum(weights.values())

        if total_weight == 0:
            # Equal distribution fallback
            n = len(scores)
            base = total_layers // n
            rem = total_layers % n
            ids = list(scores.keys())
            alloc = {nid: base for nid in ids}
            for i in range(rem):
                alloc[ids[i]] += 1
            return alloc

        # Largest Remainder Method
        shares = {nid: (w / total_weight) * total_layers for nid, w in weights.items()}
        allocated = {nid: int(s) for nid, s in shares.items()}
        remainder = total_layers - sum(allocated.values())

        fractions = {nid: shares[nid] - allocated[nid] for nid in shares}
        sorted_ids = sorted(fractions.keys(), key=lambda k: fractions[k], reverse=True)

        for i in range(remainder):
            allocated[sorted_ids[i]] += 1

        return allocated