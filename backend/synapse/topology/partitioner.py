"""
LACP — Optimal Shard-to-Node Partitioning Engine

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

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

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
    """Result of LACP partitioning for a single node."""
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
    """Complete LACP partitioning result for a model."""
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
        For LACP, clusters represent pipeline stages.
        """
        stages: Dict[int, List[ShardAssignment]] = {}
        for a in self.assignments:
            stages.setdefault(a.cluster_id, []).append(a)

        # Sort by cluster centroid latency (closest → first)
        cluster_order = sorted(stages.keys(), key=lambda cid: (
            self.clusters[cid].centroid_latency_ms if cid < len(self.clusters) else 999999
        ))
        return [stages[cid] for cid in cluster_order]


def compute_partition_ilp(
    nodes: Dict[str, NodeResources],
    latency_matrix: Optional[LatencyMatrix],
    total_layers: int,
    model_id: str = "",
    weights: Optional[ResourceWeights] = None,
    num_micro_batches: int = 1,
) -> LACPPartition:
    """
    ILP-based optimal partitioner using PuLP.
    
    Formulation:
    - Decision variable x_{i,l}: binary, 1 if node i processes layer l
    - Constraint: each layer assigned to exactly one node
    - Constraint: each node gets contiguous layers (enforced via ordering)
    - Objective: minimize max weighted-latency across all node assignments
    
    Falls back to Largest Remainder if PuLP is not available.
    
    Args:
        nodes: {node_id: NodeResources} for all nodes in the cluster.
        latency_matrix: Optional N×N latency data.
        total_layers: Total number of transformer layers.
        model_id: Model identifier string.
        weights: Custom resource weights.
        num_micro_batches: Number of micro-batches.
        
    Returns:
        LACPPartition with ILP-optimized ShardAssignment list.
    """
    if not HAS_PULP:
        # Fall back to LRM if PuLP not available
        return compute_partition(
            nodes=nodes,
            latency_matrix=latency_matrix,
            total_layers=total_layers,
            model_id=model_id,
            weights=weights,
            num_micro_batches=num_micro_batches,
        )
    
    partition = LACPPartition(
        model_id=model_id,
        total_layers=total_layers,
        num_micro_batches=num_micro_batches,
        latency_matrix=latency_matrix,
    )
    
    if not nodes:
        return partition
    
    # Score all nodes
    scorer = ResourceScorer(weights)
    scores = scorer.compute_scores(nodes)
    node_ids = list(nodes.keys())
    num_nodes = len(node_ids)
    
    if num_nodes == 0 or total_layers <= 0:
        return partition
    
    # Build PuLP problem
    prob = pulp.LpProblem("LACP_Partitioning", pulp.LpMinimize)
    
    # x[i][l] = 1 if node i gets layer l
    x = {}
    for i, nid in enumerate(node_ids):
        for l in range(total_layers):
            x[(i, l)] = pulp.LpVariable(f"x_{i}_{l}", cat=pulp.LpBinary)
    
    # y[i][l] = 1 if node i starts at layer l (for contiguity)
    y = {}
    for i in range(num_nodes):
        for l in range(total_layers):
            y[(i, l)] = pulp.LpVariable(f"y_{i}_{l}", cat=pulp.LpBinary)
    
    # z[i][j] = 1 if node i precedes node j in pipeline order
    z = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                z[(i, j)] = pulp.LpVariable(f"z_{i}_{j}", cat=pulp.LpBinary)
    
    # Constraint 1: Each layer assigned to exactly one node
    for l in range(total_layers):
        prob += pulp.lpSum([x[(i, l)] for i in range(num_nodes)]) == 1
    
    # Constraint 2: Each node has exactly one start position (or 0 if no layers)
    for i in range(num_nodes):
        prob += pulp.lpSum([y[(i, l)] for l in range(total_layers)]) <= 1
    
    # Constraint 3: First layer must start somewhere
    prob += pulp.lpSum([y[(i, 0)] for i in range(num_nodes)]) == 1
    
    # Constraint 4: y[i][l] == 1 iff x[i][l] == 1 and (l==0 or x[i][l-1]==0)
    for i in range(num_nodes):
        for l in range(total_layers):
            if l == 0:
                prob += y[(i, l)] >= x[(i, l)]
                prob += y[(i, l)] <= x[(i, l)]
            else:
                # y[i][l] ≤ x[i][l]
                prob += y[(i, l)] <= x[(i, l)]
                # y[i][l] ≤ 1 - x[i][l-1]
                prob += y[(i, l)] <= 1 - x[(i, l-1)]
                # y[i][l] ≥ x[i][l] - x[i][l-1]
                prob += y[(i, l)] >= x[(i, l)] - x[(i, l-1)]

    # NOTE: contiguity needs no separate constraints. Every run of layers has
    # exactly one start (Constraint 4), each node may start at most one run
    # (Constraint 2), and layer 0 starts exactly one run (Constraint 3), so
    # each node's assigned layers form one contiguous range. The former
    # "Contiguity via ordered positions" block was removed: both of its
    # constraints were tautologies (l*(1-y) + l*y == l and
    # x <= 1 - y + x), so they never constrained anything.

    # Simplify: enforce ordering of node start positions
    # Use a big-M formulation for pipeline order
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # z[i][j] + z[j][i] <= 1
                prob += z[(i, j)] + z[(j, i)] <= 1
    
    # Transitivity
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_nodes):
                if i != j and j != k and i != k:
                    prob += z[(i, j)] + z[(j, k)] - 1 <= z[(i, k)]
    
    # At least (num_nodes-1) of the z pairs
    prob += pulp.lpSum([z[(i, j)] for i in range(num_nodes) for j in range(num_nodes) if i != j]) >= num_nodes - 1
    
    # Link x to ordering: if node i starts at layer l1 and node j starts at l2 and l1 < l2, then z[i][j] = 1
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                for l1 in range(total_layers):
                    for l2 in range(total_layers):
                        if l1 < l2:
                            prob += z[(i, j)] >= y[(i, l1)] + y[(j, l2)] - 1
    
    # Objective: minimize max latency-weighted load
    # Create variable for max load
    max_load = pulp.LpVariable("max_load", lowBound=0)
    
    for i, nid in enumerate(node_ids):
        score = scores.get(nid, (1.0, None))[0]
        if score <= 0:
            score = 1.0
        layers_on_node = pulp.lpSum([x[(i, l)] for l in range(total_layers)])
        prob += max_load >= layers_on_node / score
    
    prob += max_load
    
    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        # Solver failed or the problem was infeasible: fall back to the
        # Largest Remainder method so the cluster still gets a valid
        # contiguous partition instead of a garbage/incomplete one.
        return compute_partition(
            nodes=nodes,
            latency_matrix=latency_matrix,
            total_layers=total_layers,
            model_id=model_id,
            weights=weights,
            num_micro_batches=num_micro_batches,
        )
    
    # Extract solution
    # Determine which node gets which layer range
    node_layers: Dict[str, List[int]] = {nid: [] for nid in node_ids}
    for i, nid in enumerate(node_ids):
        for l in range(total_layers):
            if pulp.value(x[(i, l)]) == 1:
                node_layers[nid].append(l)
    
    # Determine pipeline order from y
    start_positions = []
    for i, nid in enumerate(node_ids):
        for l in range(total_layers):
            if pulp.value(y[(i, l)]) == 1:
                start_positions.append((l, i, nid))
    
    start_positions.sort(key=lambda t: t[0])  # Sort by start layer
    
    # Build pipeline ordering
    ordered_nodes = [t[2] for t in start_positions]
    
    # Ensure all nodes with layers are included
    assigned = set(ordered_nodes)
    for nid in node_ids:
        if node_layers[nid] and nid not in assigned:
            ordered_nodes.append(nid)
            if node_layers[nid]:
                start_positions.append((min(node_layers[nid]), 0, nid))
    
    # Build assignments
    if not ordered_nodes:
        ordered_nodes = node_ids
    
    partition.node_scores = scores
    
    # One cluster for ILP (latency clustering happens inside the ILP weight)
    partition.clusters = [Cluster(
        cluster_id=0,
        nodes=list(nodes.keys()),
        centroid_latency_ms=0.0,
        intra_latency_ms=0.0,
    )]
    
    current_layer = 0
    for nid in ordered_nodes:
        layers = node_layers.get(nid, [])
        if not layers:
            continue
        start_l = min(layers)
        end_l = max(layers)
        # Ensure no gaps
        if start_l > current_layer:
            # Fill gap with next node
            start_l = current_layer
        score, nscores = scores.get(nid, (0.0, None))
        partition.assignments.append(ShardAssignment(
            node_id=nid,
            start_layer=start_l,
            end_layer=end_l,
            layer_count=end_l - start_l + 1,
            cluster_id=0,
            composite_score=score,
            normalized_scores=nscores,
        ))
        current_layer = end_l + 1
    
    # Ensure full coverage
    if partition.assignments and partition.assignments[-1].end_layer < total_layers - 1:
        partition.assignments[-1].end_layer = total_layers - 1
        partition.assignments[-1].layer_count = partition.assignments[-1].end_layer - partition.assignments[-1].start_layer + 1
    
    return partition


def compute_partition(
    nodes: Dict[str, NodeResources],
    latency_matrix: Optional[LatencyMatrix],
    total_layers: int,
    model_id: str = "",
    weights: Optional[ResourceWeights] = None,
    max_intra_cluster_latency_ms: float = 5.0,
    num_micro_batches: int = 1,
    method: str = "lrm",
) -> LACPPartition:
    """
    Main entry point for LACP partitioning.
    Supports two methods:
    - "lrm": Largest Remainder Method (greedy proportional, fast)
    - "ilp": PuLP-based Integer Linear Programming (optimal, slower)

    Algorithm:
    1. Compute composite resource scores for all nodes
    2. If latency matrix available, cluster nodes by proximity
    3. Within each cluster, allocate layers via chosen method
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
    # Route to ILP solver if requested
    if method == "ilp":
        return compute_partition_ilp(
            nodes=nodes,
            latency_matrix=latency_matrix,
            total_layers=total_layers,
            model_id=model_id,
            weights=weights,
            num_micro_batches=num_micro_batches,
        )
    
    partition = LACPPartition(
        model_id=model_id,
        total_layers=total_layers,
        num_micro_batches=num_micro_batches,
        latency_matrix=latency_matrix,
    )

    if not nodes:
        return partition

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