"""
Benchmark: Pipeline parallelism speedup projection.

Exercises the PipelineScheduler's theoretical speedup formula with
realistic overhead parameters (network latency, serialization cost,
pipeline depth) and logs results.

This benchmarks the pipeline's theoretical peak throughput, not
end-to-end system performance. Use it to validate scaling projections.
"""
import time
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def compute_speedup(
    num_nodes: int,
    stage_time_ms: float = 2.0,
    network_latency_ms: float = 0.5,
    serialization_ms: float = 0.1,
    num_micro_batches: int = 4,
) -> Tuple[float, float, float]:
    """
    Compute pipeline speedup over single-node execution.
    
    Args:
        num_nodes: Number of pipeline stages.
        stage_time_ms: Time for one node to process one micro-batch (ms).
        network_latency_ms: One-way network latency between nodes (ms).
        serialization_ms: Time to encode/decode tensor (ms).
        num_micro_batches: Number of micro-batches split across pipeline.
        
    Returns:
        (single_node_time, pipeline_time, speedup_ratio)
    """
    # Single-node total time: all layers on one node
    total_work = num_nodes * stage_time_ms
    single_node_time = total_work * num_micro_batches
    
    # Pipeline time: warmup + steady state + cooldown
    # Warmup: (num_nodes - 1) * (stage_time_ms + network_latency + serialization)
    # Steady: num_micro_batches * max(stage_time, network_latency + serialization)
    # Cooldown: (num_nodes - 1) * (stage_time_ms + network_latency + serialization)
    overhead_per_step = network_latency_ms + serialization_ms
    step_time = max(stage_time_ms, overhead_per_step)
    
    warmup = (num_nodes - 1) * (stage_time_ms + overhead_per_step)
    steady = num_micro_batches * step_time
    cooldown = (num_nodes - 1) * (stage_time_ms + overhead_per_step)
    
    pipeline_time = warmup + steady + cooldown
    
    speedup = single_node_time / pipeline_time if pipeline_time > 0 else 1.0
    efficiency = speedup / num_nodes * 100
    
    return single_node_time, pipeline_time, speedup


def benchmark_table() -> str:
    """Generate a markdown table of speedup projections for various cluster sizes."""
    lines = ["# Pipeline Speedup Projections (Benchmark)",
             "",
             "| Nodes | Stage(ms) | Latency(ms) | Serialize(ms) | Micro-batches | Single(ms) | Pipeline(ms) | Speedup | Efficiency |",
             "|-------|-----------|-------------|---------------|---------------|------------|--------------|---------|------------|"]
    
    configs = [
        (1, 2.0, 0.5, 0.1, 4),
        (2, 2.0, 0.5, 0.1, 4),
        (4, 2.0, 0.5, 0.1, 4),
        (8, 2.0, 0.5, 0.1, 4),
        (10, 2.0, 0.5, 0.1, 4),
        (4, 3.0, 1.0, 0.2, 8),
        (8, 3.0, 1.0, 0.2, 8),
        (10, 3.0, 1.0, 0.2, 8),
        (100, 2.0, 2.0, 0.5, 16),
    ]
    
    for nodes, stage, lat, ser, mbs in configs:
        single, pipeline, speedup = compute_speedup(nodes, stage, lat, ser, mbs)
        eff = speedup / nodes * 100
        lines.append(
            f"| {nodes} | {stage} | {lat} | {ser} | {mbs} | "
            f"{single:.1f} | {pipeline:.1f} | {speedup:.2f}x | {eff:.0f}% |"
        )
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Pipeline Speedup Benchmark ===\n")
    print(benchmark_table())
    print()
    
    # Quick validation: speedup should be > 1 for multi-node
    for nodes in [2, 4, 8, 10]:
        _, _, speedup = compute_speedup(nodes)
        print(f"  {nodes} nodes: {speedup:.2f}x speedup (efficiency: {speedup/nodes*100:.0f}%)")
    
    assert compute_speedup(1)[2] >= 1.0, "Single-node speedup must be >= 1.0"
    assert compute_speedup(2)[2] > 1.0, "2-node speedup must be > 1.0"
    assert compute_speedup(4)[2] > compute_speedup(2)[2], "4-node should beat 2-node"
    
    print("\nAll assertions passed.")
