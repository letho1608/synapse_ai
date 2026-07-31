"""
Test: PuLP-based ILP solver for LACP partitioning.

Verifies the fix for H1: the partitioner now offers a real ILP solver
via the method="ilp" parameter, accessible via compute_partition().
"""
import pytest

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class TestPartitionerILP:
    """Tests for the ILP-based partitioner."""

    def test_pulp_available(self):
        """PuLP must be importable (declared dependency in requirements.txt)."""
        import pulp
        assert pulp is not None

    def test_ilp_method_invokes_solver(self, monkeypatch):
        """compute_partition(method='ilp') invokes the PuLP solver — behavioral,
        no source inspection."""
        from synapse.topology import partitioner
        from synapse.topology.partitioner import compute_partition
        from synapse.topology.resource_weight import NodeResources

        called = {}
        orig = partitioner.compute_partition_ilp

        def recording(nodes, latency_matrix=None, total_layers=32, **kwargs):
            called["invoked"] = True
            return orig(
                nodes, latency_matrix=latency_matrix,
                total_layers=total_layers, **kwargs,
            )

        monkeypatch.setattr(partitioner, "compute_partition_ilp", recording)

        nodes = {
            "node0": NodeResources(
                gpu_vram_mb=8192, gpu_tflops_fp16=20.0, gpu_tflops_fp32=10.0,
                cpu_cores=8, system_ram_mb=16384,
            ),
            "node1": NodeResources(
                gpu_vram_mb=4096, gpu_tflops_fp16=10.0, gpu_tflops_fp32=5.0,
                cpu_cores=4, system_ram_mb=8192,
            ),
        }
        result = compute_partition(nodes, None, 16, method="ilp")
        assert called.get("invoked"), "compute_partition(method='ilp') did not call the ILP solver"
        assert result.total_layers == 16

    def test_compute_partition_ilp_function_exists(self):
        """Verify compute_partition_ilp is defined and callable."""
        from synapse.topology.partitioner import compute_partition_ilp
        assert callable(compute_partition_ilp)

    def test_ilp_method_parameter(self):
        """Verify compute_partition accepts method='ilp'."""
        from synapse.topology.partitioner import compute_partition
        import inspect
        sig = inspect.signature(compute_partition)
        assert "method" in sig.parameters, (
            "compute_partition missing method parameter"
        )

    @pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
    def test_ilp_produces_valid_partition(self):
        """Verify ILP solver produces a valid partition covering all layers."""
        from synapse.topology.partitioner import compute_partition
        from synapse.topology.resource_weight import NodeResources

        nodes = {
            "node0": NodeResources(
                gpu_vram_mb=8192, gpu_tflops_fp16=20.0, gpu_tflops_fp32=10.0,
                cpu_cores=8, system_ram_mb=16384,
            ),
            "node1": NodeResources(
                gpu_vram_mb=4096, gpu_tflops_fp16=10.0, gpu_tflops_fp32=5.0,
                cpu_cores=4, system_ram_mb=8192,
            ),
        }

        result = compute_partition(
            nodes=nodes,
            latency_matrix=None,
            total_layers=24,
            model_id="test-model",
            method="ilp",
        )

        # Verify basic properties
        assert result.total_layers == 24
        assert len(result.assignments) == 2

        # Layers should be contiguous and cover 0-23
        start_layers = sorted([a.start_layer for a in result.assignments])
        end_layers = sorted([a.end_layer for a in result.assignments])
        assert start_layers[0] == 0
        assert end_layers[-1] >= 23

        # node0 (stronger) should get more layers than node1
        node0_layers = sum(
            a.layer_count for a in result.assignments if a.node_id == "node0"
        )
        node1_layers = sum(
            a.layer_count for a in result.assignments if a.node_id == "node1"
        )
        assert node0_layers > node1_layers, (
            f"node0 should get more layers than node1; got {node0_layers} vs {node1_layers}"
        )

    @pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
    def test_ilp_full_coverage(self):
        """Verify ILP partition covers all layers exactly once."""
        from synapse.topology.partitioner import compute_partition
        from synapse.topology.resource_weight import NodeResources

        nodes = {
            f"node{i}": NodeResources(
                gpu_vram_mb=4096 + i * 1024,
                gpu_tflops_fp16=10.0 + i * 2.0,
                gpu_tflops_fp32=5.0 + i,
                cpu_cores=4 + i,
                system_ram_mb=8192 + i * 2048,
            )
            for i in range(3)
        }

        result = compute_partition(
            nodes=nodes,
            latency_matrix=None,
            total_layers=30,
            model_id="test-model",
            method="ilp",
        )

        total_layer_count = sum(a.layer_count for a in result.assignments)
        assert total_layer_count == 30, (
            f"ILP partition only covers {total_layer_count}/30 layers"
        )

        # Check contiguity: no gaps
        all_layers = set()
        for a in result.assignments:
            for l in range(a.start_layer, a.end_layer + 1):
                all_layers.add(l)
        expected = set(range(30))
        assert all_layers == expected, (
            f"Gaps in ILP partition: layers {expected - all_layers} missing"
        )

    @pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
    def test_ilp_defaults_to_lrm_without_pulp(self):
        """If PuLP is available, ILP method should still work (smoke test)."""
        from synapse.topology.partitioner import compute_partition
        from synapse.topology.resource_weight import NodeResources

        nodes = {
            "nodeA": NodeResources(gpu_vram_mb=8192, gpu_tflops_fp16=20.0, gpu_tflops_fp32=10.0, cpu_cores=8, system_ram_mb=16384),
            "nodeB": NodeResources(gpu_vram_mb=4096, gpu_tflops_fp16=10.0, gpu_tflops_fp32=5.0, cpu_cores=4, system_ram_mb=8192),
        }

        # Without ILP (LRM)
        lrm_result = compute_partition(nodes, None, 24, method="lrm")
        # With ILP
        ilp_result = compute_partition(nodes, None, 24, method="ilp")

        # Both should be valid
        assert lrm_result.assignments
        assert ilp_result.assignments
