"""
Test LACP (Latency-Aware Collaborative Partitioning) system.
Chạy: python -m pytest tests/test_lacp.py -v
"""
import sys
import os
import time
import json
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# --- Test 1: Import LACP modules ---

def test_import_lacp_modules():
    """Import tất cả LACP 3.0 modules."""
    from synapse.topology.resource_weight import NodeResources, ResourceWeights, ResourceScorer, NormalizedScores
    from synapse.topology.latency_matrix import LatencyMatrix, LatencyProber, HierarchicalClusterer, Cluster
    from synapse.topology.partitioner import compute_partition, LACPPartition, ShardAssignment
    from synapse.pipeline.scheduler import PipelineScheduler, PipelineSchedule, PipelineSlot, MicroBatchOp
    from synapse.pipeline.gradient_sync import RingAllReducer, DistributedTrainer, GradientShard
    from synapse.pipeline.fault_tolerance import FaultToleranceManager, NodeHealth, ShardCheckpoint
    from synapse.transport.codec import encode_tensor, decode_tensor, AsyncTensorForwarder, MAGIC
    print(" [OK] test_import_lacp_modules")


# --- Test 2: NodeResources ---

def test_node_resources():
    """Test NodeResources dataclass (LACP 3.0)."""
    from synapse.topology.resource_weight import NodeResources
    
    r = NodeResources(
        gpu_name="RTX 4090", gpu_vram_mb=24576, gpu_tflops_fp16=165,
        cpu_cores=16, system_ram_mb=64000, bandwidth_mbps=10000,
        latency_ms=1.0
    )
    assert r.gpu_vram_mb == 24576
    assert r.gpu_tflops_fp16 == 165
    assert r.cpu_cores == 16
    print(" [OK] test_node_resources")


# --- Test 3: Cluster ---

def test_lacp3_cluster():
    """Test Cluster dataclass (LACP 3.0)."""
    from synapse.topology.latency_matrix import Cluster
    
    cluster = Cluster(cluster_id=1, nodes=["A", "B", "C"], centroid_latency_ms=15.0, intra_latency_ms=5.0)
    assert cluster.cluster_id == 1
    assert len(cluster.nodes) == 3
    assert "A" in cluster.nodes
    assert "D" not in cluster.nodes
    print(" [OK] test_lacp3_cluster")


# --- Test 4: LACP 3.0 HierarchicalClusterer ---

def test_hierarchical_clusterer():
    """Test LACP 3.0 HierarchicalClusterer clustering."""
    from synapse.topology.latency_matrix import HierarchicalClusterer, LatencyMatrix, LatencyEntry
    
    matrix = LatencyMatrix(
        nodes=["A", "B", "C"],
        entries={
            "A": {
                "A": LatencyEntry(from_node="A", to_node="A", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
                "B": LatencyEntry(from_node="A", to_node="B", latency_ms=15, jitter_ms=2, packet_loss_pct=0),
                "C": LatencyEntry(from_node="A", to_node="C", latency_ms=200, jitter_ms=10, packet_loss_pct=0.01),
            },
            "B": {
                "A": LatencyEntry(from_node="B", to_node="A", latency_ms=15, jitter_ms=2, packet_loss_pct=0),
                "B": LatencyEntry(from_node="B", to_node="B", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
                "C": LatencyEntry(from_node="B", to_node="C", latency_ms=180, jitter_ms=8, packet_loss_pct=0.01),
            },
            "C": {
                "A": LatencyEntry(from_node="C", to_node="A", latency_ms=200, jitter_ms=10, packet_loss_pct=0.01),
                "B": LatencyEntry(from_node="C", to_node="B", latency_ms=180, jitter_ms=8, packet_loss_pct=0.01),
                "C": LatencyEntry(from_node="C", to_node="C", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
            },
        }
    )
    
    clusterer = HierarchicalClusterer(max_intra_latency_ms=50.0)
    clusters = clusterer.cluster(matrix)
    
    assert len(clusters) >= 1
    print(" [OK] test_hierarchical_clusterer")


def test_hierarchical_clusterer_fallback():
    """Test fallback clustering with 2 nodes."""
    from synapse.topology.latency_matrix import HierarchicalClusterer, LatencyMatrix, LatencyEntry
    
    matrix = LatencyMatrix(
        nodes=["A", "B"],
        entries={
            "A": {
                "A": LatencyEntry(from_node="A", to_node="A", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
                "B": LatencyEntry(from_node="A", to_node="B", latency_ms=15, jitter_ms=1, packet_loss_pct=0),
            },
            "B": {
                "A": LatencyEntry(from_node="B", to_node="A", latency_ms=15, jitter_ms=1, packet_loss_pct=0),
                "B": LatencyEntry(from_node="B", to_node="B", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
            },
        }
    )
    
    clusterer = HierarchicalClusterer()
    clusters = clusterer.cluster(matrix)
    assert len(clusters) >= 1
    print(" [OK] test_hierarchical_clusterer_fallback")


# --- Test 5: ResourceScorer ---

def test_resource_scorer():
    """Test LACP 3.0 ResourceScorer."""
    from synapse.topology.resource_weight import ResourceScorer, NodeResources, ResourceWeights
    
    nodes = {
        "A": NodeResources(gpu_name="RTX 4090", gpu_vram_mb=24576, gpu_tflops_fp16=165, cpu_cores=16, system_ram_mb=64000),
        "B": NodeResources(gpu_name="RTX 3060", gpu_vram_mb=6144, gpu_tflops_fp16=26, cpu_cores=4, system_ram_mb=16000),
    }
    
    scorer = ResourceScorer(ResourceWeights())
    scores = scorer.compute_scores(nodes)
    
    assert len(scores) == 2
    assert "A" in scores
    assert "B" in scores
    assert scores["A"][0] > scores["B"][0]  # RTX 4090 should score higher
    
    # Layer allocation
    alloc = scorer.layer_allocation(scores, 32)
    assert sum(alloc.values()) == 32
    print(" [OK] test_resource_scorer")


# --- Test 6: compute_partition (LACP 3.0 entry point) ---

def test_compute_partition():
    """Test LACP 3.0 compute_partition with 2 nodes."""
    from synapse.topology.partitioner import compute_partition
    from synapse.topology.resource_weight import NodeResources
    
    nodes = {
        "A": NodeResources(gpu_name="RTX 4090", gpu_vram_mb=24576, gpu_tflops_fp16=165, cpu_cores=16, system_ram_mb=64000),
        "B": NodeResources(gpu_name="RTX 3060", gpu_vram_mb=6144, gpu_tflops_fp16=26, cpu_cores=4, system_ram_mb=16000),
    }
    
    result = compute_partition(nodes=nodes, latency_matrix=None, total_layers=32, model_id="test")
    
    assert len(result.assignments) == 2
    assert result.total_layers == 32
    assert sum(a.layer_count for a in result.assignments) == 32
    
    # RTX 4090 should get more layers
    a_assignment = result.get_assignment("A")
    b_assignment = result.get_assignment("B")
    assert a_assignment.layer_count > b_assignment.layer_count
    
    print(" [OK] test_compute_partition")


# --- Test 7: PipelineScheduler 1F1B ---

def test_pipeline_scheduler_1f1b():
    """Test 1F1B pipeline schedule generation."""
    from synapse.pipeline.scheduler import PipelineScheduler, MicroBatchOp
    
    scheduler = PipelineScheduler(num_micro_batches=4)
    schedule = scheduler.generate_1f1b_schedule(num_stages=3, num_micro_batches=4)
    
    assert schedule.num_stages == 3
    assert schedule.num_micro_batches == 4
    assert schedule.warmup_steps == 2
    assert schedule.cooldown_steps == 2
    
    # Check forward + backward slot counts
    fwd = sum(1 for s in schedule.slots if s.op == MicroBatchOp.FORWARD)
    bwd = sum(1 for s in schedule.slots if s.op == MicroBatchOp.BACKWARD)
    assert fwd == 12  # 3 stages × 4 micro-batches
    assert bwd == 12
    
    print(" [OK] test_pipeline_scheduler_1f1b")


def test_inference_schedule():
    """Test forward-only inference pipeline schedule."""
    from synapse.pipeline.scheduler import PipelineScheduler, MicroBatchOp
    
    scheduler = PipelineScheduler(num_micro_batches=4)
    schedule = scheduler.generate_inference_schedule(num_stages=3, num_micro_batches=4)
    
    assert schedule.num_stages == 3
    assert all(s.op == MicroBatchOp.FORWARD for s in schedule.slots)
    
    print(" [OK] test_inference_schedule")


# --- Test 8: Binary Codec ---

def test_encode_decode_tensor():
    """Test binary tensor encode/decode roundtrip."""
    from synapse.transport.codec import encode_tensor, decode_tensor, MAGIC
    import numpy as np
    
    tensor = np.random.rand(3, 4, 5).astype(np.float32)
    encoded = encode_tensor(tensor)
    
    assert encoded[:4] == MAGIC
    
    decoded, meta = decode_tensor(encoded)
    assert np.array_equal(tensor, decoded)
    assert meta is None
    
    print(" [OK] test_encode_decode_tensor")


def test_encode_decode_with_metadata():
    """Test binary codec with metadata."""
    from synapse.transport.codec import encode_tensor, decode_tensor
    import numpy as np
    
    tensor = np.array([[1, 2, 3]]).astype(np.int32)
    meta = {"request_id": "req-123", "micro_batch_id": 5}
    encoded = encode_tensor(tensor, metadata=meta)
    
    decoded, decoded_meta = decode_tensor(encoded)
    assert np.array_equal(tensor, decoded)
    assert decoded_meta == {"request_id": "req-123", "micro_batch_id": 5}
    
    print(" [OK] test_encode_decode_with_metadata")


def test_compress_tensor():
    """Test compressed tensor encoding."""
    from synapse.transport.codec import encode_tensor, decode_tensor
    import numpy as np
    
    # Large tensor that should benefit from compression
    tensor = np.random.rand(128, 128).astype(np.float32)  # 64KB
    encoded = encode_tensor(tensor, compress=True, compress_level=1)
    
    decoded, _ = decode_tensor(encoded)
    assert np.array_equal(tensor, decoded)
    
    print(" [OK] test_compress_tensor")


# --- Test 9: Ring All-Reduce ---

def test_ring_all_reduce():
    """Test RingAllReducer logic."""
    from synapse.pipeline.gradient_sync import RingAllReducer
    import numpy as np
    
    reducer = RingAllReducer(num_nodes=3, ring_order=["A", "B", "C"])
    
    assert reducer.get_predecessor("A") == "C"
    assert reducer.get_successor("A") == "B"
    assert reducer.get_predecessor("B") == "A"
    assert reducer.get_successor("C") == "A"
    
    # Accumulate gradients
    g1 = {"w": np.array([1.0, 2.0])}
    g2 = {"w": np.array([3.0, 4.0])}
    
    reducer.accumulate("A", g1)
    reducer.accumulate("A", g2)
    result = reducer.get_accumulated("A")
    
    assert np.array_equal(result["w"], np.array([4.0, 6.0]))
    
    print(" [OK] test_ring_all_reduce")


# --- Test 10: FaultToleranceManager ---

def test_fault_tolerance_manager():
    """Test FaultToleranceManager heartbeat and detection."""
    from synapse.pipeline.fault_tolerance import FaultToleranceManager, NodeHealth
    import time
    
    health = NodeHealth(node_id="node-1")
    health.record_heartbeat()
    assert health.is_alive
    assert not health.is_blacklisted
    
    # Simulate timeout
    health.last_heartbeat = 0  # Very old
    assert health.check_timeout(timeout_s=10.0)
    assert not health.is_alive
    
    # 3 consecutive timeouts → blacklisted
    health.check_timeout(timeout_s=10.0)
    health.check_timeout(timeout_s=10.0)
    assert health.is_blacklisted
    
    print(" [OK] test_fault_tolerance_manager")


def test_checkpoint_save_load():
    """Test ShardCheckpoint save and load."""
    from synapse.pipeline.fault_tolerance import FaultToleranceManager, ShardCheckpoint
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ft = FaultToleranceManager("test-node", checkpoint_dir=tmpdir)
        
        path = ft.save_checkpoint(
            model_id="test-model",
            start_layer=0,
            end_layer=7,
            step=100,
            kv_cache={"layer_0": "state"},
        )
        
        assert os.path.exists(path)
        
        loaded = ft.load_checkpoint("test-model", 0, 7, 100)
        assert loaded is not None
        assert loaded.step == 100
        assert loaded.kv_cache == {"layer_0": "state"}
    
    print(" [OK] test_checkpoint_save_load")


# --- Test 11: AsyncTensorForwarder ---

def test_async_forwarder_create():
    """Test AsyncTensorForwarder initialization."""
    from synapse.transport.codec import AsyncTensorForwarder
    
    fw = AsyncTensorForwarder(node_id="node-A", max_queue_depth=16)
    assert fw.node_id == "node-A"
    assert fw.max_queue_depth == 16
    
    print(" [OK] test_async_forwarder_create")


# --- Test 12: Integration - Full LACP 3.0 flow ---

def test_lacp3_full_integration():
    """Test full LACP 3.0 flow: resource scoring → cluster → partition → pipeline."""
    from synapse.topology.partitioner import compute_partition
    from synapse.topology.resource_weight import NodeResources
    from synapse.topology.latency_matrix import LatencyMatrix, LatencyEntry
    from synapse.pipeline.scheduler import PipelineScheduler
    
    nodes = {
        "VN-A": NodeResources(gpu_name="RTX 4090", gpu_vram_mb=24576, gpu_tflops_fp16=165, cpu_cores=16, system_ram_mb=64000),
        "VN-B": NodeResources(gpu_name="RTX 4080", gpu_vram_mb=16384, gpu_tflops_fp16=104, cpu_cores=12, system_ram_mb=32000),
        "US-C": NodeResources(gpu_name="A100", gpu_vram_mb=40960, gpu_tflops_fp16=312, cpu_cores=32, system_ram_mb=256000),
    }
    
    # Latency matrix shows VN-A and VN-B are close (5ms), US-C is far (200ms)
    lm = LatencyMatrix(
        nodes=["VN-A", "VN-B", "US-C"],
        entries={
            "VN-A": {
                "VN-A": LatencyEntry(from_node="VN-A", to_node="VN-A", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
                "VN-B": LatencyEntry(from_node="VN-A", to_node="VN-B", latency_ms=5, jitter_ms=1, packet_loss_pct=0),
                "US-C": LatencyEntry(from_node="VN-A", to_node="US-C", latency_ms=200, jitter_ms=15, packet_loss_pct=0.01),
            },
            "VN-B": {
                "VN-A": LatencyEntry(from_node="VN-B", to_node="VN-A", latency_ms=5, jitter_ms=1, packet_loss_pct=0),
                "VN-B": LatencyEntry(from_node="VN-B", to_node="VN-B", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
                "US-C": LatencyEntry(from_node="VN-B", to_node="US-C", latency_ms=195, jitter_ms=14, packet_loss_pct=0.01),
            },
            "US-C": {
                "VN-A": LatencyEntry(from_node="US-C", to_node="VN-A", latency_ms=200, jitter_ms=15, packet_loss_pct=0.01),
                "VN-B": LatencyEntry(from_node="US-C", to_node="VN-B", latency_ms=195, jitter_ms=14, packet_loss_pct=0.01),
                "US-C": LatencyEntry(from_node="US-C", to_node="US-C", latency_ms=0, jitter_ms=0, packet_loss_pct=0),
            },
        }
    )
    
    result = compute_partition(nodes=nodes, latency_matrix=lm, total_layers=32, model_id="test")
    
    assert len(result.assignments) == 3
    assert sum(a.layer_count for a in result.assignments) == 32
    
    # VN-A and VN-B should be in same cluster (low latency)
    assert len(result.clusters) >= 1
    
    # Generate pipeline schedule from partition
    scheduler = PipelineScheduler(num_micro_batches=4)
    stages = result.get_pipeline_stages()
    assert len(stages) >= 1
    
    schedule = scheduler.generate_inference_schedule(
        num_stages=len(result.assignments),
        num_micro_batches=4,
    )
    assert schedule.num_stages == len(result.assignments)
    
    print(" [OK] test_lacp3_full_integration")


# --- Test 13: Verify LargestRemainder removed ---

def test_largest_remainder_removed():
    """Verify LargestRemainderPartitioningStrategy is removed (LACP 3.0 replaces LACP 2.0)."""
    import os
    assert not os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "synapse", "topology", "largest_remainder.py")
    ), "LargestRemainderPartitioningStrategy should be deleted"
    
    # Verify LACP 3.0 is available
    from synapse.topology.partitioner import compute_partition
    assert compute_partition is not None
    
    print(" [OK] test_largest_remainder_removed")


# --- Run all tests ---

if __name__ == "__main__":
    print("=" * 60)
    print("Running LACP System Tests")
    print("=" * 60)
    
    tests = [
        test_import_lacp_modules,
        test_node_resources,
        test_lacp3_cluster,
        test_hierarchical_clusterer,
        test_hierarchical_clusterer_fallback,
        test_resource_scorer,
        test_compute_partition,
        test_pipeline_scheduler_1f1b,
        test_inference_schedule,
        test_encode_decode_tensor,
        test_encode_decode_with_metadata,
        test_compress_tensor,
        test_ring_all_reduce,
        test_fault_tolerance_manager,
        test_checkpoint_save_load,
        test_async_forwarder_create,
        test_lacp3_full_integration,
        test_largest_remainder_removed,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f" [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)