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
    """Import tất cả LACP modules."""
    from synapse.topology.lacp_partitioning import LACPPartitioningStrategy
    from synapse.topology.latency_clustering import HierarchicalClusterer, MachineCluster
    from synapse.topology.ilp_partitioner import ILPPartitioner, LayerProfile
    from synapse.networking.latency_probing import LatencyProber, LatencyCache
    from synapse.resources import (
        CPURAMKVCacheManager, DiskEmbeddingsCacheManager,
        MultiGPUManager, PreloadManager, ResourceManagerRegistry
    )
    print(" [OK] test_import_lacp_modules")


# --- Test 2: LayerProfile ---

def test_layer_profile():
    """Test LayerProfile dataclass."""
    from synapse.topology.ilp_partitioner import LayerProfile
    
    profile = LayerProfile(layer_id=0, memory_mb=4000, compute_flops=1e12)
    assert profile.layer_id == 0
    assert profile.memory_mb == 4000
    assert profile.compute_flops == 1e12
    print(" [OK] test_layer_profile")


# --- Test 3: MachineCluster ---

def test_machine_cluster():
    """Test MachineCluster dataclass."""
    from synapse.topology.latency_clustering import MachineCluster
    
    cluster = MachineCluster(id=1, machine_ids=["A", "B", "C"], avg_latency=15.0)
    assert cluster.id == 1
    assert len(cluster.machine_ids) == 3
    assert cluster.contains("A")
    assert not cluster.contains("D")
    print(" [OK] test_machine_cluster")


# --- Test 4: HierarchicalClusterer ---

def test_hierarchical_clusterer():
    """Test HierarchicalClusterer clustering."""
    from synapse.topology.latency_clustering import HierarchicalClusterer
    
    # Latency matrix với 3 machines
    # A-B: 15ms, A-C: 200ms, B-C: 180ms
    latency_matrix = {
        "A": {"A": 0, "B": 15, "C": 200},
        "B": {"A": 15, "B": 0, "C": 180},
        "C": {"A": 200, "B": 180, "C": 0}
    }
    
    clusterer = HierarchicalClusterer(default_threshold_ms=50.0)
    clusters = clusterer.cluster(latency_matrix, threshold_ms=50.0)
    
    # Với threshold 50ms, A và B nên cùng cluster (15ms), C riêng
    assert len(clusters) >= 1
    
    # Test are_in_same_cluster
    assert clusterer.are_in_same_cluster("A", "B", clusters)
    print(" [OK] test_hierarchical_clusterer")


def test_hierarchical_clusterer_fallback():
    """Test fallback clustering khi scipy không available."""
    from synapse.topology.latency_clustering import HierarchicalClusterer, SCIPY_AVAILABLE
    
    latency_matrix = {
        "A": {"A": 0, "B": 15},
        "B": {"A": 15, "B": 0}
    }
    
    clusterer = HierarchicalClusterer()
    clusters = clusterer.cluster(latency_matrix)
    
    # Nếu scipy available, nó sẽ cluster thành 1 (vì 15ms < threshold)
    # Nếu scipy không available, fallback sẽ tạo 2 clusters (mỗi machine 1 cluster)
    if SCIPY_AVAILABLE:
        # scipy available - có thể cluster thành 1 hoặc 2 tùy threshold
        assert len(clusters) >= 1
    else:
        # scipy not available - fallback
        assert len(clusters) == 2
    print(" [OK] test_hierarchical_clusterer_fallback")


# --- Test 5: ILPPartitioner ---

def test_ilp_partitioner_greedy_fallback():
    """Test ILPPartitioner với greedy fallback."""
    from synapse.topology.ilp_partitioner import ILPPartitioner, LayerProfile
    from synapse.topology.topology import Topology
    from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
    
    # Tạo topology với 2 machines
    topo = Topology()
    topo.update_node("A", DeviceCapabilities(
        model="GPU A", chip="RTX 4090", memory=24000,
        flops=DeviceFlops(fp32=82, fp16=165, int8=330)
    ))
    topo.update_node("B", DeviceCapabilities(
        model="GPU B", chip="RTX 3060", memory=6000,
        flops=DeviceFlops(fp32=13, fp16=26, int8=52)
    ))
    
    # Tạo layer profiles (32 layers, mỗi layer 4GB)
    layer_profiles = [
        LayerProfile(layer_id=i, memory_mb=4000, compute_flops=1e12)
        for i in range(32)
    ]
    
    latency_matrix = {
        "A": {"A": 0, "B": 15},
        "B": {"A": 15, "B": 0}
    }
    
    partitioner = ILPPartitioner(timeout=1)  # Short timeout
    partitions = partitioner.find_optimal(
        topology=topo,
        latency_matrix=latency_matrix,
        clusters=[],  # Empty clusters
        layer_profiles=layer_profiles
    )
    
    # Phải có partitions
    assert len(partitions) >= 1
    
    # Mỗi partition phải có node_id hợp lệ
    for p in partitions:
        assert p.node_id in ["A", "B"]
        assert 0 <= p.start < p.end <= 1.0
    
    print(" [OK] test_ilp_partitioner_greedy_fallback")


# --- Test 6: LACPPartitioningStrategy ---

def test_lacp_partitioning_strategy_init():
    """Test LACPPartitioningStrategy initialization."""
    from synapse.topology.lacp_partitioning import LACPPartitioningStrategy
    
    strategy = LACPPartitioningStrategy(
        latency_cache_file="test_latency_cache.json",
        cluster_threshold_ms=50.0,
        ilp_timeout=60
    )
    
    assert strategy.latency_cache_file == "test_latency_cache.json"
    assert strategy.cluster_threshold_ms == 50.0
    assert strategy.ilp_timeout == 60
    
    print(" [OK] test_lacp_partitioning_strategy_init")


def test_lacp_partitioning_strategy_partition():
    """Test LACPPartitioningStrategy.partition()."""
    from synapse.topology.lacp_partitioning import LACPPartitioningStrategy
    from synapse.topology.topology import Topology
    from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
    from synapse.topology.ilp_partitioner import LayerProfile
    
    # Tạo topology
    topo = Topology()
    topo.update_node("A", DeviceCapabilities(
        model="GPU A", chip="RTX 4090", memory=24000,
        flops=DeviceFlops(fp32=82, fp16=165, int8=330)
    ))
    topo.update_node("B", DeviceCapabilities(
        model="GPU B", chip="RTX 3060", memory=6000,
        flops=DeviceFlops(fp32=13, fp16=26, int8=52)
    ))
    
    # Tạo strategy với custom layer profiles
    strategy = LACPPartitioningStrategy(ilp_timeout=1)
    strategy.set_layer_profiles([
        LayerProfile(layer_id=i, memory_mb=4000, compute_flops=1e12)
        for i in range(32)
    ])
    
    # Gọi partition (sẽ dùng greedy fallback vì không có peers)
    partitions = strategy.partition(topo)
    
    assert len(partitions) >= 1
    for p in partitions:
        assert p.node_id in ["A", "B"]
        assert 0 <= p.start < p.end <= 1.0
    
    print(" [OK] test_lacp_partitioning_strategy_partition")


# --- Test 7: LatencyProber ---

def test_latency_prober_init():
    """Test LatencyProber initialization."""
    from synapse.networking.latency_probing import LatencyProber
    
    prober = LatencyProber(
        cache_file="test_cache.json",
        probe_count=5,
        timeout_ms=3000
    )
    
    assert prober.cache_file == Path("test_cache.json")
    assert prober.probe_count == 5
    assert prober.timeout_ms == 3000
    
    print(" [OK] test_latency_prober_init")


def test_latency_prober_cache():
    """Test LatencyProber cache operations."""
    from synapse.networking.latency_probing import LatencyProber, LatencyCache
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.json"
        prober = LatencyProber(cache_file=str(cache_file))
        
        # Initial state - no cache
        assert prober._needs_update()
        
        # Save cache manually
        cache = LatencyCache(
            version="1.0",
            timestamp=time.time(),
            matrix={"A": {"B": 15}, "B": {"A": 15}}
        )
        prober._save_cache(cache)
        
        # Should not need update now
        prober._cache = None  # Reset cache
        assert not prober._needs_update()
        
        # Get latency
        prober._cache = cache
        lat = prober.get_latency("A", "B")
        assert lat == 15
        
        lat_default = prober.get_latency("A", "C")  # Not in cache
        assert lat_default == 200.0  # Default value
        
        # Clear cache
        prober.clear_cache()
        assert prober._cache is None
        
    print(" [OK] test_latency_prober_cache")


# --- Test 8: CPU RAM KV Cache ---

def test_cpu_ram_cache_basic():
    """Test CPURAMCache basic operations."""
    from synapse.resources.cpu_ram_kv_cache import CPURAMCache
    import numpy as np
    
    cache = CPURAMCache(max_size_mb=10)  # 10MB
    
    # Test empty cache
    assert not cache.has("key1")
    assert cache.get("key1") is None
    assert cache.usage() == 0.0
    
    # Add item
    data = np.random.rand(100, 100).astype(np.float32)  # ~40KB
    cache.set("key1", data)
    
    assert cache.has("key1")
    assert cache.get("key1") is not None
    assert cache.usage() > 0
    
    # Update item
    data2 = np.random.rand(50, 50).astype(np.float32)  # ~10KB
    cache.set("key1", data2)
    assert cache.has("key1")
    
    # Clear
    cache.clear()
    assert not cache.has("key1")
    assert cache.usage() == 0.0
    
    print(" [OK] test_cpu_ram_cache_basic")


def test_cpu_ram_cache_eviction():
    """Test CPURAMCache LRU eviction."""
    from synapse.resources.cpu_ram_kv_cache import CPURAMCache
    import numpy as np
    
    cache = CPURAMCache(max_size_mb=1)  # 1MB very small
    
    # Add items until eviction (each ~40KB, 1MB can hold ~25 items)
    for i in range(30):  # Add 30 items to ensure eviction
        data = np.random.rand(100, 100).astype(np.float32)  # ~40KB
        cache.set(f"key{i}", data)
    
    # Oldest items should be evicted (key0, key1, etc.)
    # With 30 items of ~40KB each = ~1.2MB for 1MB cache, some should be evicted
    assert not cache.has("key0") or not cache.has("key1"), "At least some oldest items should be evicted"
    
    # Most recent items should still be there
    assert cache.has("key29"), "Most recent item should still be in cache"
    
    print(" [OK] test_cpu_ram_cache_eviction")


def test_cpu_ram_kv_cache_manager():
    """Test CPURAMKVCacheManager."""
    from synapse.resources.cpu_ram_kv_cache import CPURAMKVCacheManager
    import numpy as np
    
    manager = CPURAMKVCacheManager(max_cpu_ram_gb=1, gpu_threshold=0.9)
    
    data = np.random.rand(100, 100).astype(np.float32)
    
    # Test set_with_overflow (no GPU, so goes to CPU)
    manager.set_with_overflow("key1", data, gpu_usage=0.95)
    assert manager.cpu_cache.has("key1")
    
    # Test get
    result = manager.get("key1")
    assert result is not None
    
    # Test has
    assert manager.has("key1")
    assert not manager.has("nonexistent")
    
    print(" [OK] test_cpu_ram_kv_cache_manager")


# --- Test 9: Disk Embeddings Cache ---

def test_disk_embeddings_cache():
    """Test DiskEmbeddingsCache."""
    from synapse.resources.disk_embeddings_cache import DiskEmbeddingsCache
    import numpy as np
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskEmbeddingsCache(tmpdir, max_size_gb=1)
        
        # Test has/get on empty cache
        assert not cache.has("prompt1")
        assert cache.get("prompt1") is None
        
        # Add embeddings
        embeddings = np.random.rand(768).astype(np.float32)
        cache.set("prompt1", embeddings)
        
        assert cache.has("prompt1")
        result = cache.get("prompt1")
        assert result is not None
        assert np.array_equal(result, embeddings)
        
        # Clear
        cache.clear()
        assert not cache.has("prompt1")
    
    print(" [OK] test_disk_embeddings_cache")


# --- Test 10: Resource Manager Registry ---

def test_resource_manager_registry():
    """Test ResourceManagerRegistry."""
    from synapse.resources import ResourceManagerRegistry
    
    registry = ResourceManagerRegistry()
    
    # List managers
    managers = registry.list_managers()
    assert "cpu_ram_kv" in managers
    assert "disk_cache" in managers
    assert "preload" in managers
    
    # Get managers
    cpu_ram = registry.get_cpu_ram_kv()
    disk_cache = registry.get_disk_cache()
    preload = registry.get_preload()
    
    assert cpu_ram is not None
    assert disk_cache is not None
    assert preload is not None
    
    # Multi-GPU should be None until configured
    multi_gpu = registry.get_multi_gpu()
    assert multi_gpu is None
    
    # Setup multi-GPU
    registry.setup_multi_gpu([0, 1])
    multi_gpu = registry.get_multi_gpu()
    assert multi_gpu is not None
    
    print(" [OK] test_resource_manager_registry")


# --- Test 11: MultiGPUManager ---

def test_multi_gpu_manager_init():
    """Test MultiGPUManager initialization."""
    from synapse.resources.multi_gpu_manager import MultiGPUManager
    
    # Single GPU
    manager = MultiGPUManager([0])
    assert manager.num_gpus == 1
    assert manager.tensor_parallel is False
    
    # Multi-GPU
    manager = MultiGPUManager([0, 1], tensor_parallel=True)
    assert manager.num_gpus == 2
    assert manager.tensor_parallel is True
    
    print(" [OK] test_multi_gpu_manager_init")


# --- Test 12: PreloadManager ---

def test_preload_manager_init():
    """Test PreloadManager initialization."""
    from synapse.resources.preload_strategy import PreloadManager
    
    manager = PreloadManager()
    
    # Should start empty
    assert len(manager.preload_strategies) == 0
    
    # Get non-existent strategy
    assert manager.get_strategy("model1") is None
    
    # Clear all
    manager.clear_all()
    assert len(manager.preload_strategies) == 0
    
    print(" [OK] test_preload_manager_init")


# --- Test 13: Integration - Full LACP flow ---

def test_lacp_full_integration():
    """Test full LACP flow: topology -> clustering -> partitioning."""
    from synapse.topology.lacp_partitioning import LACPPartitioningStrategy
    from synapse.topology.topology import Topology
    from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
    from synapse.topology.ilp_partitioner import LayerProfile
    
    # Create topology with 3 machines
    topo = Topology()
    topo.update_node("VN-A", DeviceCapabilities(
        model="VN GPU A", chip="RTX 4090", memory=24000,
        flops=DeviceFlops(fp32=82, fp16=165, int8=330)
    ))
    topo.update_node("VN-B", DeviceCapabilities(
        model="VN GPU B", chip="RTX 4080", memory=16000,
        flops=DeviceFlops(fp32=52, fp16=104, int8=208)
    ))
    topo.update_node("US-C", DeviceCapabilities(
        model="US GPU C", chip="A100", memory=40000,
        flops=DeviceFlops(fp32=19, fp16=312, int8=624)
    ))
    
    # Create LACP strategy
    strategy = LACPPartitioningStrategy(
        latency_cache_file="test_integration_cache.json",
        cluster_threshold_ms=100.0,
        ilp_timeout=5
    )
    
    # Set layer profiles (simulate 32-layer model)
    layer_profiles = [
        LayerProfile(layer_id=i, memory_mb=4000, compute_flops=1e12)
        for i in range(32)
    ]
    strategy.set_layer_profiles(layer_profiles)
    
    # Partition
    partitions = strategy.partition(topo)
    
    # Verify partitions
    assert len(partitions) >= 1
    total_coverage = sum(p.end - p.start for p in partitions)
    assert abs(total_coverage - 1.0) < 0.01  # Should cover ~100%
    
    # Verify all nodes in partitions are valid
    for p in partitions:
        assert p.node_id in ["VN-A", "VN-B", "US-C"]
        assert 0 <= p.start < p.end <= 1.0
    
    # Get cached data
    latency_matrix = strategy.get_latency_matrix()
    clusters = strategy.get_clusters()

    # Latency matrix should now have data (fixed: uses topology.nodes instead of non-existent topology.peers)
    assert len(latency_matrix) == 3  # 3 nodes in topology
    assert "VN-A" in latency_matrix
    assert "VN-B" in latency_matrix
    assert "US-C" in latency_matrix
    
    # Clean up
    import os
    if os.path.exists("test_integration_cache.json"):
        os.remove("test_integration_cache.json")
    
    print(" [OK] test_lacp_full_integration")


# --- Test 14: Verify RingMemoryWeighted removed ---

def test_ring_memory_weighted_removed():
    """Verify RingMemoryWeightedPartitioningStrategy is removed."""
    try:
        from synapse.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
        assert False, "RingMemoryWeightedPartitioningStrategy should be removed"
    except ImportError:
        pass  # Expected
    
    # Verify LACP is the default
    from synapse.topology.lacp_partitioning import LACPPartitioningStrategy
    strategy = LACPPartitioningStrategy()
    assert strategy is not None
    
    print(" [OK] test_ring_memory_weighted_removed")


# --- Run all tests ---

if __name__ == "__main__":
    print("=" * 60)
    print("Running LACP System Tests")
    print("=" * 60)
    
    tests = [
        test_import_lacp_modules,
        test_layer_profile,
        test_machine_cluster,
        test_hierarchical_clusterer,
        test_hierarchical_clusterer_fallback,
        test_ilp_partitioner_greedy_fallback,
        test_lacp_partitioning_strategy_init,
        test_lacp_partitioning_strategy_partition,
        test_latency_prober_init,
        test_latency_prober_cache,
        test_cpu_ram_cache_basic,
        test_cpu_ram_cache_eviction,
        test_cpu_ram_kv_cache_manager,
        test_disk_embeddings_cache,
        test_resource_manager_registry,
        test_multi_gpu_manager_init,
        test_preload_manager_init,
        test_lacp_full_integration,
        test_ring_memory_weighted_removed,
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