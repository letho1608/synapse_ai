"""
Test LACP partitioning with 3 nodes for real model sizes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.topology.partitioner import compute_partition
from synapse.topology.resource_weight import NodeResources, ResourceScorer, ResourceWeights
from synapse.topology.latency_matrix import LatencyMatrix, LatencyEntry

nodes = {
    'NODE-A': NodeResources(gpu_name='RTX 4090', gpu_vram_mb=24576, gpu_tflops_fp16=165, cpu_cores=16, system_ram_mb=64000, bandwidth_mbps=1000, latency_ms=1),
    'NODE-B': NodeResources(gpu_name='RTX 3060', gpu_vram_mb=12288, gpu_tflops_fp16=26, cpu_cores=8, system_ram_mb=32000, bandwidth_mbps=500, latency_ms=5),
    'NODE-C': NodeResources(gpu_name='CPU only', gpu_vram_mb=0, gpu_tflops_fp16=4, cpu_cores=4, system_ram_mb=16000, bandwidth_mbps=100, latency_ms=10),
}

lm = LatencyMatrix(
    nodes=['NODE-A', 'NODE-B', 'NODE-C'],
    entries={
        'NODE-A': {
            'A': LatencyEntry('A','A',0,0,0),
            'B': LatencyEntry('A','B',5,1,0),
            'C': LatencyEntry('A','C',200,15,0.01),
        },
        'NODE-B': {
            'A': LatencyEntry('B','A',5,1,0),
            'B': LatencyEntry('B','B',0,0,0),
            'C': LatencyEntry('B','C',195,14,0.01),
        },
        'NODE-C': {
            'A': LatencyEntry('C','A',200,15,0.01),
            'B': LatencyEntry('C','B',195,14,0.01),
            'C': LatencyEntry('C','C',0,0,0),
        },
    }
)

models = [
    ('Qwen2.5-1.5B (28 layers)', 28),
    ('Llama-3.2-3B (32 layers)', 32),
    ('Mistral-7B (32 layers)', 32),
    ('Qwen2.5-7B (28 layers)', 28),
    ('Llama-3.1-8B (32 layers)', 32),
]

print('=' * 85)
print('LAYOUT: 3 node phan tan (pipeline parallelism)')
print('=' * 85)

for name, total_layers in models:
    result = compute_partition(nodes=nodes, latency_matrix=lm, total_layers=total_layers, model_id='test')
    print(f'\n--- {name} ---')
    for a in result.assignments:
        pct = (a.layer_count / total_layers) * 100
        print(f'  {a.node_id:<10} layers={a.layer_count:<3} ({pct:5.1f}%) range=[{a.start_layer:2}-{a.end_layer:2}]')

print()
print('=' * 85)
print('RESOURCE SCORES:')
print('=' * 85)
scorer = ResourceScorer(ResourceWeights())
scores = scorer.compute_scores(nodes)
from dataclasses import fields as dc_fields
for nid, (score, breakdown) in sorted(scores.items()):
    print(f'  {nid:<10} score={score:.4f}')
    for f in dc_fields(breakdown):
        v = getattr(breakdown, f.name)
        if v != 0:
            print(f'            {f.name}={v:.3f}')

print()
print('=' * 85)
print('CLUSTERS:')
print('=' * 85)
for name, total_layers in models:
    result = compute_partition(nodes=nodes, latency_matrix=lm, total_layers=total_layers, model_id='test')
    for i, c in enumerate(result.clusters):
        print(f'  {name:<25} Cluster {i}: {c.nodes} (latency={c.intra_latency_ms:.1f}ms)')
