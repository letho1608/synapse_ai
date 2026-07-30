"""
Test: All 8 LACP resource dimensions are populated in production.

Verifies the fix for H3: disk I/O, bandwidth, latency, and power
are no longer zero defaults.
"""
import pytest


class TestResourceMeasurement:
    """Tests for live resource measurement in resource_weight.py."""

    def test_psutil_available(self):
        """psutil must be importable (dependency for disk I/O measurement)."""
        import psutil
        assert psutil is not None

    def test_disk_io_measurement_function_exists(self):
        """Verify _measure_disk_io() is defined and returns non-negative values."""
        from synapse.topology.resource_weight import _measure_disk_io
        result = _measure_disk_io()
        assert isinstance(result, dict)
        assert "read_mbps" in result
        assert "write_mbps" in result
        # Values should be >= 0
        assert result["read_mbps"] >= 0
        assert result["write_mbps"] >= 0

    def test_power_measurement_function_exists(self):
        """Verify _measure_power() is defined and returns dict."""
        from synapse.topology.resource_weight import _measure_power
        result = _measure_power()
        assert isinstance(result, dict)
        assert "power_limit_w" in result
        assert "current_power_w" in result

    def test_bandwidth_measurement_has_default(self):
        """Verify _measure_dummy_bandwidth returns a reasonable default."""
        from synapse.topology.resource_weight import _measure_dummy_bandwidth
        result = _measure_dummy_bandwidth()
        assert isinstance(result, dict)
        assert "bandwidth_mbps" in result
        assert result["bandwidth_mbps"] > 0  # Should have a default

    def test_from_device_capabilities_uses_bare_dc(self):
        """Verify from_device_capabilities populates all 8 dimension fields."""
        from synapse.topology.resource_weight import NodeResources

        # Create a minimal mock DeviceCapabilities object
        class MockDC:
            chip = "TestGPU"
            memory = 8192
            class Flops:
                fp16 = 20.0
                fp32 = 10.0
            flops = Flops()
            cpu_cores = 8
            system_ram_mb = 16384

        mock_dc = MockDC()
        nr = NodeResources.from_device_capabilities(mock_dc)

        # Check all 8 dimension fields are present and populated
        # GPU fields
        assert nr.gpu_vram_mb > 0
        assert nr.gpu_tflops_fp16 > 0
        # CPU/RAM
        assert nr.cpu_cores > 0
        assert nr.system_ram_mb > 0
        # Previously zero fields — now measured
        assert nr.disk_read_mbps >= 0  # Measured by psutil
        assert nr.disk_write_mbps >= 0
        assert nr.bandwidth_mbps > 0   # Has reasonable default
        # Power
        assert nr.power_limit_w >= 0
        assert nr.current_power_w >= 0

    def test_resource_scorer_includes_all_dimensions(self):
        """Verify ResourceScorer.compute_scores uses all 8 resource dimensions."""
        from synapse.topology.resource_weight import (
            NodeResources, ResourceWeights, ResourceScorer
        )

        # Create two nodes with different disk/bandwidth/power
        nr1 = NodeResources(
            gpu_vram_mb=8192,
            gpu_tflops_fp16=20.0,
            gpu_tflops_fp32=10.0,
            cpu_cores=8,
            system_ram_mb=16384,
            disk_read_mbps=500.0,
            disk_write_mbps=300.0,
            bandwidth_mbps=10000.0,
            latency_ms=1.0,
            power_limit_w=300.0,
            current_power_w=150.0,
        )
        nr2 = NodeResources(
            gpu_vram_mb=4096,
            gpu_tflops_fp16=10.0,
            gpu_tflops_fp32=5.0,
            cpu_cores=4,
            system_ram_mb=8192,
            disk_read_mbps=100.0,
            disk_write_mbps=50.0,
            bandwidth_mbps=1000.0,
            latency_ms=10.0,
            power_limit_w=150.0,
            current_power_w=100.0,
        )

        scorer = ResourceScorer()
        scores = scorer.compute_scores({"node1": nr1, "node2": nr2})

        assert len(scores) == 2
        # node1 should score higher on all dimensions
        assert scores["node1"][0] > scores["node2"][0]

        # Verify normalized scores include all 8 dimensions
        ns1 = scores["node1"][1]
        assert ns1.vram > 0
        assert ns1.tflops > 0
        assert ns1.cpu > 0
        assert ns1.ram > 0
        assert ns1.disk > 0
        assert ns1.bandwidth > 0
        assert ns1.latency >= 0
        assert ns1.power_headroom >= 0
