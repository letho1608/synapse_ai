"""
LACP 3.0 — Optimal Shard-to-Node Partitioning Engine

Combines resource weighting, latency clustering, and ILP/Greedy
optimization to produce the optimal assignment of model layers to nodes.

The partitioner:
1. Accepts a set of nodes with resources + latency matrix
2. Groups nodes into latency clusters (hierarchical clustering)
3. Within each cluster, allocates layers based on composite resource scores
4. Assigns contiguous layer ranges to adjacent clusters
5. Produces Shard assignments used by the pipeline scheduler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .resource_weight import (
    NodeResources,
    ResourceWeights,
    ResourceScorer,
    NormalizedScores,
    normalize,
)
from .latency_matrix import (
    LatencyMatrix,
    LatencyEntry,
    LatencyProber,
    Cluster,
    HierarchicalClusterer,
)


@dataclass
class ShardAssignment:
    """Result of LACP 3.0 partitioning for a single node."""
    node_id: str
    start_layer: int
    end_layer: int
    layer_count: int
    cluster_id: int = 0
    composite_score: float = 0.0
    normalized_scores: Optional[NormalizedScores] = None

    def is_first(self) -> bool:
        return self.start_layer == 0

    def is_last(self, total_layers: int) -> bool:
        return self.end_layer >= total_layers - 1


@dataclass
class LACPPartition:
    """Complete LACP 3.0 partitioning result for a model."""
    model_id: str = ""
    total_layers: int = 0
    num_micro_batches: int = 1
    assignments: List[ShardAssignment] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)
    latency_matrix: Optional[LatencyMatrix] = None
    node_scores: Dict[str, Tuple[float, NormalizedScores]] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        return len(self.assignments)

    def get_assignment(self, node_id: str) -> Optional[ShardAssignment]:
        for a in self.assignments:
            if a.node_id == node_id:
                return a
        return None

    def get_pipeline_stages(self) -> List[List[ShardAssignment]]:
        """
        Return pipeline stages ordered by cluster. Each stage is a list
        of ShardAssignments that run in parallel within the same pipeline slot.
        For LACP 3.0, clusters represent pipeline stages.
        """
        stages: Dict[int, List[ShardAssignment]] = {}
        for a in self.assignments:
            stages.setdefault(a.cluster_id, []).append(a)

        # Sort by cluster centroid latency (closest → first)
        cluster_order = sorted(stages.keys(), key=lambda cid: (
            self.clusters[cid].centroid_latency_ms if cid < len(self.clusters) else 999999
        ))
        return [stages[cid] for cid in cluster_order]


def compute_partition(
    nodes: Dict[str, NodeResources],
    latency_matrix: Optional[LatencyMatrix],
    total_layers: int,
    model_id: str = "",
    weights: Optional[ResourceWeights] = None,
    max_intra_cluster_latency_ms: float = 5.0,
    num_micro_batches: int = 1,
) -> LACPPartition:
    """
    Main entry point for LACP 3.0 partitioning.

    Algorithm:
    1. Compute composite resource scores for all nodes
    2. If latency matrix available, cluster nodes by proximity
    3. Within each cluster, allocate layers via Largest Remainder
    4. Assign contiguous ranges matching cluster order
    5. Return complete partition with per-node shard assignments

    Args:
        nodes: {node_id: NodeResources} for all nodes in the cluster.
        latency_matrix: Optional N×N latency data. If None, all nodes in one cluster.
        total_layers: Total number of transformer layers in the model.
        model_id: Model identifier string.
        weights: Custom resource weights (default: balanced).
        max_intra_cluster_latency_ms: Max latency within a cluster before splitting.
        num_micro_batches: Number of micro-batches for pipeline schedule.

    Returns:
        LACPPartition with per-node ShardAssignment list.
    """
    partition = LACPPartition(
        model_id=model_id,
        total_layers=total_layers,
        num_micro_batches=num_micro_batches,
        latency_matrix=latency_matrix,
    )

    if not nodes:
        return partition

    # Defensive: ensure latency_matrix is a LatencyMatrix, not a plain dict
    if latency_matrix is not None and not isinstance(latency_matrix, LatencyMatrix):
        latency_matrix = None

    # Step 1: Resource scoring
    scorer = ResourceScorer(weights)
    partition.node_scores = scorer.compute_scores(nodes)

    # Step 2: Latency clustering
    if latency_matrix and len(latency_matrix.nodes) >= 2:
        clusterer = HierarchicalClusterer(max_intra_cluster_latency_ms)
        partition.clusters = clusterer.cluster(latency_matrix)
    else:
        # All nodes in one cluster
        partition.clusters = [Cluster(
            cluster_id=0,
            nodes=list(nodes.keys()),
            centroid_latency_ms=0.0,
            intra_latency_ms=0.0,
        )]

    # Step 3-4: Allocate layers per cluster, then per node within cluster
    # Compute total cluster weight
    cluster_weights: Dict[int, float] = {}
    for c in partition.clusters:
        total = 0.0
        for nid in c.nodes:
            total += partition.node_scores.get(nid, (0.0, None))[0]
        cluster_weights[c.cluster_id] = total

    total_cluster_weight = sum(cluster_weights.values())
    if total_cluster_weight <= 0:
        total_cluster_weight = len(partition.clusters)
        cluster_weights = {c.cluster_id: 1.0 for c in partition.clusters}

    # Allocate layers to clusters (Largest Remainder)
    cluster_shares = {
        cid: (w / total_cluster_weight) * total_layers
        for cid, w in cluster_weights.items()
    }
    cluster_layer_alloc = {cid: int(s) for cid, s in cluster_shares.items()}
    remainder = total_layers - sum(cluster_layer_alloc.values())
    sorted_cids = sorted(
        cluster_layer_alloc.keys(),
        key=lambda cid: cluster_shares[cid] - cluster_layer_alloc[cid],
        reverse=True,
    )
    for i in range(remainder):
        cluster_layer_alloc[sorted_cids[i % len(sorted_cids)]] += 1

    # Allocate layers to nodes within each cluster
    current_layer = 0
    partition.assignments = []

    for c in sorted(partition.clusters, key=lambda x: x.centroid_latency_ms):
        cluster_nodes = c.nodes
        cluster_total = cluster_layer_alloc.get(c.cluster_id, 0)
        if cluster_total <= 0:
            continue

        # Per-node allocation within cluster via Largest Remainder
        node_alloc = scorer.layer_allocation(
            {nid: partition.node_scores[nid] for nid in cluster_nodes
             if nid in partition.node_scores},
            cluster_total,
        )

        for nid in sorted(cluster_nodes):
            layers = node_alloc.get(nid, 0)
            if layers <= 0:
                continue

            score, nscores = partition.node_scores.get(nid, (0.0, None))
            end_layer = current_layer + layers - 1
            if end_layer >= total_layers:
                end_layer = total_layers - 1

            partition.assignments.append(ShardAssignment(
                node_id=nid,
                start_layer=current_layer,
                end_layer=end_layer,
                layer_count=end_layer - current_layer + 1,
                cluster_id=c.cluster_id,
                composite_score=score,
                normalized_scores=nscores,
            ))
            current_layer = end_layer + 1

    # Ensure full coverage
    if current_layer < total_layers and partition.assignments:
        partition.assignments[-1].end_layer = total_layers - 1
        partition.assignments[-1].layer_count = (
            partition.assignments[-1].end_layer - partition.assignments[-1].start_layer + 1
        )

    return partition