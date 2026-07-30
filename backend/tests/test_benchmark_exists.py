"""
Test: Benchmark script exists and is importable.
"""
import os
import sys


def test_benchmark_script_exists():
    """benchmark_cluster_speedup.py must exist on disk."""
    path = os.path.join(
        os.path.dirname(__file__),
        "benchmark_cluster_speedup.py"
    )
    assert os.path.exists(path), f"Benchmark script not found at {path}"


def test_benchmark_importable():
    """benchmark_cluster_speedup must be importable as a module."""
    # Add backend/tests to path
    tests_dir = os.path.dirname(__file__)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import benchmark_cluster_speedup
    assert benchmark_cluster_speedup is not None
    assert hasattr(benchmark_cluster_speedup, "compute_speedup")


def test_compute_speedup_basics():
    """Verify compute_speedup returns reasonable values."""
    from benchmark_cluster_speedup import compute_speedup
    
    # Single node: speedup should be 1.0
    s, p, ratio = compute_speedup(1)
    assert ratio >= 1.0
    
    # 2 nodes: speedup should be > 1.0
    s2, p2, ratio2 = compute_speedup(2)
    assert ratio2 > 1.0
    
    # More nodes should give more speedup
    _, _, ratio4 = compute_speedup(4)
    assert ratio4 > ratio2, "4-node speedup should exceed 2-node speedup"


def test_compute_speedup_efficiency_bounds():
    """Verify speedup efficiency stays in reasonable bounds."""
    from benchmark_cluster_speedup import compute_speedup
    
    # With ideal conditions, efficiency should be high
    for nodes in [1, 2, 4, 8]:
        _, _, speedup = compute_speedup(
            num_nodes=nodes,
            stage_time_ms=10.0,  # Large stage time relative to overhead
            network_latency_ms=0.1,
            serialization_ms=0.05,
            num_micro_batches=16,
        )
        efficiency = speedup / nodes
        assert efficiency > 0.5, (
            f"Efficiency too low for {nodes} nodes: {efficiency:.1%}"
        )


def test_benchmark_table_generated():
    """Verify benchmark_table() produces non-empty output."""
    from benchmark_cluster_speedup import benchmark_table
    table = benchmark_table()
    assert table
    assert "Speedup" in table
    assert "100" in table  # Should include 100-node row
