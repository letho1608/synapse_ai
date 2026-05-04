"""
Test LACP 3.0 — Multi-Resource Weighting Engine

Validates:
1. Resource normalization produces correct 0.0-1.0 ranges
2. Composite scoring correctly weights resources
3. Layer allocation via Largest Remainder is integer-optimal
4. GPU utilization penalty reduces effective compute score
5. RAM pressure penalty reduces effective memory score
6. Multiple clusters get proportional layer allocation
7. Edge cases: empty nodes, single node, zero resources
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.lacp.resource_weight import (
    NodeResources,
    ResourceWeights,
    NormalizedScores,
    ResourceScorer,
    normalize,
)


class TestNormalize:
    def test_basic(self):
        assert normalize(0, 0, 100) == 0.0
        assert normalize(50, 0, 100) == 0.5
        assert normalize(100, 0, 100) == 1.0

    def test_clamp(self):
        assert normalize(-10, 0, 100) == 0.0
        assert normalize(200, 0, 100) == 1.0

    def test_zero_range(self):
        assert normalize(50, 100, 100) == 0.0


class TestNodeResources:
    def test_from_dict(self):
        from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
        dc = DeviceCapabilities(
            model="RTX 4090",
            chip="AD102",
            memory=24576,
            flops=DeviceFlops(fp32=82.0, fp16=165.0, int8=330.0),
            cpu_cores=16,
            system_ram_mb=64000,
            gpu_count=1,
            total_gpu_vram_mb=24576,
        )
        nr = NodeResources.from_device_capabilities(dc)
        assert nr.gpu_vram_mb == 24576.0
        assert nr.gpu_tflops_fp16 == 165.0
        assert nr.cpu_cores == 16
        assert nr.system_ram_mb == 64000.0

    def test_from_minimal_dict(self):
        from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
        dc = DeviceCapabilities(
            model="CPU Only",
            chip="Core i3",
            memory=8192,
            flops=DeviceFlops(fp32=1.0, fp16=1.0, int8=1.0),
            cpu_cores=4,
            system_ram_mb=8192,
            gpu_count=0,
            total_gpu_vram_mb=0,
        )
        nr = NodeResources.from_device_capabilities(dc)
        assert nr.gpu_vram_mb == 8192.0
        assert nr.gpu_tflops_fp16 == 1.0


class TestResourceScorer:
    def _make_nodes(self):
        return {
            "gpu-strong": NodeResources(
                gpu_name="RTX 4090", gpu_vram_mb=24576,
                gpu_tflops_fp16=165, gpu_tflops_fp32=82,
                cpu_cores=16, system_ram_mb=64000,
                gpu_count=1, disk_read_mbps=5000, disk_write_mbps=4000,
                bandwidth_mbps=1000, latency_ms=0.5, power_limit_w=450,
                current_power_w=100,
            ),
            "gpu-mid": NodeResources(
                gpu_name="RTX 3080", gpu_vram_mb=10240,
                gpu_tflops_fp16=60, gpu_tflops_fp32=30,
                cpu_cores=8, system_ram_mb=32000,
                gpu_count=1, disk_read_mbps=3000, disk_write_mbps=2500,
                bandwidth_mbps=800, latency_ms=1.0, power_limit_w=320,
                current_power_w=80,
            ),
            "cpu-only": NodeResources(
                gpu_name="", gpu_vram_mb=0,
                gpu_tflops_fp16=0, gpu_tflops_fp32=0,
                cpu_cores=4, system_ram_mb=8192,
                gpu_count=0, disk_read_mbps=500, disk_write_mbps=400,
                bandwidth_mbps=100, latency_ms=5.0, power_limit_w=65,
                current_power_w=30,
            ),
        }

    def test_composite_scores_ordering(self):
        scorer = ResourceScorer()
        scores = scorer.compute_scores(self._make_nodes())

        assert len(scores) == 3
        _, ns_strong = scores["gpu-strong"]
        _, ns_mid = scores["gpu-mid"]
        _, ns_weak = scores["cpu-only"]

        # Strongest node should have highest all resource scores
        assert ns_strong.vram >= ns_mid.vram >= ns_weak.vram
        assert ns_strong.tflops >= ns_mid.tflops >= ns_weak.tflops
        assert ns_strong.cpu >= ns_mid.cpu >= ns_weak.cpu
        assert ns_strong.ram >= ns_mid.ram >= ns_weak.ram

    def test_composite_score_ranges(self):
        scorer = ResourceScorer()
        scores = scorer.compute_scores(self._make_nodes())

        for nid, (composite, ns) in scores.items():
            assert 0.0 <= composite <= 1.0, f"{nid}: {composite}"

    def test_gpu_utilization_penalty(self):
        """Busy GPU should get lower effective compute score."""
        nodes = {
            "idle": NodeResources(
                gpu_name="RTX 3080", gpu_vram_mb=10240,
                gpu_tflops_fp16=60, gpu_tflops_fp32=30,
                cpu_cores=8, system_ram_mb=32000,
                gpu_util_pct=0.0,
            ),
            "busy": NodeResources(
                gpu_name="RTX 3080", gpu_vram_mb=10240,
                gpu_tflops_fp16=60, gpu_tflops_fp32=30,
                cpu_cores=8, system_ram_mb=32000,
                gpu_util_pct=95.0,
            ),
        }
        scorer = ResourceScorer()
        scores = scorer.compute_scores(nodes)

        comp_idle, _ = scores["idle"]
        comp_busy, _ = scores["busy"]

        assert comp_idle > comp_busy, "Idle GPU should score higher than busy GPU"

    def test_ram_pressure_penalty(self):
        """Node with high RAM usage should score lower."""
        nodes = {
            "free": NodeResources(
                gpu_vram_mb=10240, gpu_tflops_fp16=60, gpu_tflops_fp32=30,
                cpu_cores=8, system_ram_mb=32000, system_ram_used_mb=1000,
            ),
            "tight": NodeResources(
                gpu_vram_mb=10240, gpu_tflops_fp16=60, gpu_tflops_fp32=30,
                cpu_cores=8, system_ram_mb=32000, system_ram_used_mb=31000,
            ),
        }
        scorer = ResourceScorer()
        scores = scorer.compute_scores(nodes)

        comp_free, _ = scores["free"]
        comp_tight, _ = scores["tight"]

        assert comp_free > comp_tight, "Free RAM node should score higher than tight RAM node"

    def test_empty_nodes(self):
        scorer = ResourceScorer()
        scores = scorer.compute_scores({})
        assert len(scores) == 0

    def test_single_node(self):
        nodes = {
            "only": NodeResources(
                gpu_vram_mb=24576, gpu_tflops_fp16=165, gpu_tflops_fp32=82,
                cpu_cores=16, system_ram_mb=64000,
            ),
        }
        scorer = ResourceScorer()
        scores = scorer.compute_scores(nodes)
        assert len(scores) == 1
        comp, ns = scores["only"]
        # Single node: all normalizations are 0 (min==max), but composite should exist
        assert isinstance(comp, float)
        assert isinstance(ns, NormalizedScores)


class TestLayerAllocation:
    def _make_nodes(self):
        return {
            "a": NodeResources(
                gpu_name="A100", gpu_vram_mb=81920,
                gpu_tflops_fp16=312, gpu_tflops_fp32=156,
                cpu_cores=32, system_ram_mb=256000,
            ),
            "b": NodeResources(
                gpu_name="RTX 4090", gpu_vram_mb=24576,
                gpu_tflops_fp16=165, gpu_tflops_fp32=82,
                cpu_cores=16, system_ram_mb=64000,
            ),
            "c": NodeResources(
                gpu_name="RTX 3080", gpu_vram_mb=10240,
                gpu_tflops_fp16=60, gpu_tflops_fp32=30,
                cpu_cores=8, system_ram_mb=32000,
            ),
            "d": NodeResources(
                gpu_vram_mb=8192, gpu_tflops_fp16=6, gpu_tflops_fp32=3,
                cpu_cores=4, system_ram_mb=8192,
            ),
        }

    def test_total_layer_count(self):
        scorer = ResourceScorer()
        scores = scorer.compute_scores(self._make_nodes())
        alloc = scorer.layer_allocation(scores, 32)
        assert sum(alloc.values()) == 32

    def test_allocation_ordering(self):
        """Nodes with higher weight get more layers."""
        scorer = ResourceScorer()
        scores = scorer.compute_scores(self._make_nodes())
        alloc = scorer.layer_allocation(scores, 32)

        # Strongest should get most layers
        assert alloc["a"] >= alloc["b"] >= alloc["c"] >= alloc["d"]

    def test_zero_total_weight(self):
        """When all weights are zero, layers are distributed equally."""
        scores = {
            "x": (0.0, NormalizedScores()),
            "y": (0.0, NormalizedScores()),
            "z": (0.0, NormalizedScores()),
        }
        scorer = ResourceScorer()
        alloc = scorer.layer_allocation(scores, 30)
        assert sum(alloc.values()) == 30
        assert all(8 <= v <= 12 for v in alloc.values())

    def test_large_model_200_layers(self):
        scorer = ResourceScorer()
        scores = scorer.compute_scores(self._make_nodes())
        alloc = scorer.layer_allocation(scores, 200)
        assert sum(alloc.values()) == 200
        assert alloc["a"] > 0
        assert alloc["d"] > 0  # even weakest should get some layers

    def test_single_node_gets_all(self):
        scorer = ResourceScorer()
        scores = scorer.compute_scores({
            "only": NodeResources(
                gpu_vram_mb=24576, gpu_tflops_fp16=165, gpu_tflops_fp32=82,
                cpu_cores=16, system_ram_mb=64000,
            ),
        })
        alloc = scorer.layer_allocation(scores, 32)
        assert sum(alloc.values()) == 32
        assert alloc["only"] == 32

    def test_no_negative_allocations(self):
        scorer = ResourceScorer()
        scores = scorer.compute_scores(self._make_nodes())
        alloc = scorer.layer_allocation(scores, 16)
        assert all(v >= 0 for v in alloc.values())


class TestResourceWeights:
    def test_default_weights_sum_to_one(self):
        w = ResourceWeights()
        total = sum([w.vram, w.tflops, w.cpu, w.ram, w.disk, w.bandwidth, w.latency, w.power_headroom])
        assert abs(total - 1.0) < 0.001

    def test_custom_weights_auto_normalize(self):
        w = ResourceWeights(vram=1.0, tflops=1.0, cpu=0, ram=0, disk=0, bandwidth=0, latency=0, power_headroom=0)
        total = sum([w.vram, w.tflops, w.cpu, w.ram, w.disk, w.bandwidth, w.latency, w.power_headroom])
        assert abs(total - 1.0) < 0.001
        assert w.vram == 0.5
        assert w.tflops == 0.5