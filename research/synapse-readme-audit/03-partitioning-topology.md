# Lane 3: Partitioning Engine & Topology — Audit Report

**Files audited:**
- `backend/synapse/topology/resource_weight.py`
- `backend/synapse/topology/latency_matrix.py`
- `backend/synapse/topology/partitioner.py`
- `backend/synapse/topology/partitioning_strategy.py`
- `backend/synapse/topology/device_capabilities.py`
- `backend/synapse/topology/topology.py`
- `backend/synapse/networking/latency_probing.py`
- `backend/synapse/orchestration/node.py`
- `backend/synapse/monitoring/realtime.py`
- `backend/synapse/pipeline/scheduler.py`

---

## Claim 1: Multi-resource LACP — "weighs VRAM + TFLOPS + CPU + RAM + disk I/O + bandwidth for every node"

**Verdict: PARTIALLY**

**Evidence:**

The `NodeResources` dataclass (`resource_weight.py:40-66`) defines fields for all claimed resources:
```python
gpu_vram_mb: float         # VRAM (line 41)
gpu_tflops_fp16: float     # TFLOPS (line 42)
cpu_cores: int              # CPU (line 48)
system_ram_mb: float        # RAM (line 51)
disk_read_mbps: float       # Disk I/O (line 55)
disk_write_mbps: float      # Disk I/O (line 56)
bandwidth_mbps: float       # Network bandwidth (line 59)
latency_ms: float           # Latency (line 60)
power_limit_w: float        # Power envelope (line 64)
```

The `ResourceWeights` dataclass (`resource_weight.py:110-138`) assigns each a configurable weight, defaulting to:
```python
vram: float = 0.25         # line 113
tflops: float = 0.25       # line 114
cpu: float = 0.10          # line 115
ram: float = 0.10          # line 116
disk: float = 0.05         # line 117
bandwidth: float = 0.10    # line 118
latency: float = 0.10      # line 119
power_headroom: float = 0.05  # line 120
```

The `ResourceScorer.compute_scores()` method (`resource_weight.py:154-248`) normalizes all resources and computes a weighted composite per node.

**Critical gaps:**
1. **Disk I/O** (`disk_read_mbps`, `disk_write_mbps`) — These fields are **never populated in production code**. They only appear in test fixtures (`test_lacp_multiresource.py:85-101`). The `from_device_capabilities()` conversion (line 67-95) and the `_compute_lacp_partition()` in `node.py:1421-1431` both skip disk I/O. Result: always 0.0, scoring weight is dead code.

2. **Bandwidth** (`bandwidth_mbps`) — **Never populated in production code**. `node.py` does not set `nr.bandwidth_mbps`. The `LatencyProber` in `latency_matrix.py:171-182` computes bandwidth during probing, but this result is never fed into `NodeResources`. Result: always 0.0 in practice.

3. **Latency** (`latency_ms`) — Not populated in `from_device_capabilities()`, only populated via live forward latency tracking (`node.py:1464-1467`), which only updates for the local node's perspective.

| Resource | In Schema | Populated in Prod | Functional |
|---|---|---|---|
| VRAM | ✓ | ✓ | ✓ |
| TFLOPS | ✓ | ✓ | ✓ |
| CPU cores | ✓ | ✓ | ✓ |
| RAM | ✓ | ✓ | ✓ |
| Disk I/O | ✓ | **NO** | ✗ |
| Bandwidth | ✓ | **NO** | ✗ |
| Latency | ✓ | Partial | Partial |
| Power | ✓ | Partial (local only) | Partial |

---

## Claim 2: README table showing exactly which resources are weighted and how

**Verdict: PARTIALLY**

**Evidence:**

The README table (lines 69-77) lists:
> | Resource | How It's Weighted |
> | VRAM | Primary factor |
> | TFLOPS | Secondary |
> | CPU cores / RAM | For pre-processing / KV cache offloading |
> | Disk I/O | For model weight streaming |
> | Network bandwidth | Determines which nodes can share adjacent shards |
> | Network latency | Hierarchical clustering |
> | Power efficiency | Low-power nodes get lighter shards |

**Discrepancies:**
1. **"Primary" vs "Secondary"** — The code defaults VRAM=0.25 and TFLOPS=0.25 (equal). Calling VRAM "Primary" and TFLOPS "Secondary" is misleading since they have identical weight.
2. **"CPU cores / RAM" grouped** — The README groups them as one entry but the code assigns separate weights (0.10 each).
3. **Purpose descriptions** — The "How It's Weighted" column describes qualitative purpose, not actual weight coefficients. Only the `ResourceWeights` defaults (lines 113-121) are quantitative.
4. **No explicit mapping** — The README's qualitative descriptions (e.g., "determines max layers per node") are not enforced by code logic — the weight coefficient simply contributes to a dimensionless composite score; layer allocation is proportional to that composite, not VRAM-specific.

---

## Claim 3: "Latency-aware clustering: nodes with <1ms latency get adjacent shards; cross-region nodes get independent shards"

**Verdict: PARTIALLY**

**Evidence:**

The `HierarchicalClusterer` (`latency_matrix.py:232-331`) implements agglomerative hierarchical clustering with a configurable threshold:
```python
# latency_matrix.py:256
def __init__(self, max_intra_latency_ms: float = 5.0):
    self.max_intra_latency_ms = max_intra_latency_ms
```

The `compute_partition()` function (`partitioner.py:100`) defaults to:
```python
max_intra_cluster_latency_ms: float = 5.0,
```

**Discrepancies:**
1. The README says **<1ms** threshold for clustering; the code defaults to **5ms**.
2. The README's specific thresholds (<1ms, >10ms) **appear nowhere in the codebase**.
3. The threshold is configurable in principle, but the documented values are fiction.
4. **Cross-region → independent shards** — This IS verified. The algorithm in `partitioner.py:182-217` allocates contiguous layer ranges per cluster sorted by centroid latency, ensuring separate clusters get separate shard ranges.
5. **Adjacent shards within cluster** — Nodes in the same cluster get contiguous layer ranges in sorted order (`partitioner.py:195-216`), which matches the "adjacent shards" claim.

---

## Claim 4: "Hierarchical clustering — sub-1ms → same cluster; >10ms → separate pipeline stages"

**Verdict: NOT FOUND** (with caveats)

**Evidence:**

The clustering algorithm (`latency_matrix.py:248-330`) does use agglomerative hierarchical clustering. The `get_pipeline_stages()` method (`partitioner.py:70-89`) maps clusters to pipeline stages.

**Discrepancies:**
1. **No hardcoded thresholds** — The code does NOT define or use <1ms or >10ms thresholds anywhere. The default is 5ms.
2. **Complete linkage used** — The algorithm uses **complete linkage** (maximum inter-node latency: `latency_matrix.py:272-279`), not single linkage or average linkage as the docstring suggests. The docstring at line 240-241 says "minimum inter-node latency" (single linkage) but the implementation at line 271-279 computes `max_dist = max(max_dist, d)` (complete linkage). This is a **documentation bug** in the code itself.
3. **Separate pipeline stages** — The concept is structurally verified: `get_pipeline_stages()` (partitioner.py:75-89) groups assignments by `cluster_id` and returns them ordered by centroid latency. Clusters with latencies above the threshold do become separate pipeline stages.

**Severity: Medium.** The thresholds are fictional but the mechanism works; the linkage documentation inconsistency is a minor code quality issue.

---

## Claim 5: "LACP partitioning: master computes optimal shard assignment"

**Verdict: PARTIALLY**

**Evidence:**

The `compute_partition()` function (`partitioner.py:93-224`) does compute shard assignments. The orchestration node calls it via `_compute_lacp_partition()` (`node.py:1412-1441`).

**The "ILP" problem:**
1. The README (line 226) says: `partitioner.py  # ILP optimal assignment`
2. The partitioner docstring (partitioner.py:4) says: `"ILP/Greedy optimization"`
3. The `latency_probing.py:121` comment mentions: `"This ensures the ILP solver has a valid matrix"`
4. **No ILP solver is imported, used, or implemented anywhere in the codebase.** No PuLP, no OR-Tools, no scipy.optimize, no constraint programming library.
5. The actual algorithm is **Largest Remainder Method** (proportional allocation with integer rounding), a greedy algorithm:
   - `layer_allocation()` in `resource_weight.py:252-288`
   - Cluster-to-layer allocation in `partitioner.py:160-178`

The **Largest Remainder Method** is a standard proportional allocation scheme (used in seat distribution in elections). It is deterministic and proportional, but it is **not optimal in any ILP sense** — it does not optimize an objective function subject to constraints. Calling it "ILP optimal" is misleading.

**Does master compute it?** YES — `node.py:1412-1441` and `node.py:357` show the master (via `is_master()` check at line 1357) does run the partition. But it's not ILP.

**Severity: High.** The ILP claim is a significant overstatement. This is code-doc mismatch that inflates technical capability.

---

## Claim 6: "Power efficiency — Low-power nodes get lighter shards"

**Verdict: PARTIALLY**

**Evidence:**

The scoring framework supports power weighting:
- `NodeResources.power_limit_w` and `current_power_w` (lines 64-65)
- `ResourceWeights.power_headroom = 0.05` (line 121)
- `ResourceScorer.compute_scores()` normalizes `power_limit_w - current_power_w` (line 187)

Power data is collected via nvidia-smi in `realtime.py:149-187`:
```python
# realtime.py:175-184
power_w = _parse_float(parts[3])
power_limit = _parse_float(parts[4])
m.gpu_power_w += power_w
m.gpu_power_limit_w += power_limit
```

And populated into `NodeResources` in `node.py:1428`:
```python
nr.current_power_w = m.gpu_power_w
```

**Critical gap:** `power_limit_w` is **never set** on `NodeResources` — only `current_power_w` is populated (node.py:1428). The normalization computes `pwr_headroom = r.power_limit_w - r.current_power_w` (resource_weight.py:193), so with `power_limit_w` always at 0, the headroom is `-current_power_w`, which normalizes to 0 (clamped). This means the power weight is **effectively neutralized** in all cases.

Additionally, power data is only populated for `self.id` (the local node). Remote nodes in the topology will have `power_limit_w=0` and `current_power_w=0`, making cross-node power comparisons meaningless.

**Severity: Medium.** Framework exists, collection exists (partial), but the scoring formula has a bug (`power_limit_w` never set) that nullifies the effect.

---

## Claim 7: Scaling claims — 4 nodes ~2.8x, 10 nodes ~7.2x, 100 nodes ~55x

**Verdict: NOT FOUND**

**Evidence:**

The README (lines 120-128) claims specific speedup numbers:
> | 4 nodes (RTX 4090) | ~2.8x faster | ~3.0x faster |
> | 10 nodes (RTX 4090) | ~7.2x faster | ~7.5x faster |
> | 100 nodes (mixed) | ~55x faster | ~60x faster |

**Code reality:**
1. **No benchmark validates these numbers** — The closest benchmarks are `benchmark_distributed_3node.py` (3-node simulated benchmark) and `benchmark_real_3node.py` (3-node real PyTorch benchmark). Neither produces speedup ratios matching the claims.
2. **`test_cluster_scale.py` does not exist** — The README (line 246) lists this file as `tests/test_cluster_scale.py` but no such file exists anywhere in the repository.
3. **Theoretical formula only** — `PipelineScheduler.speedup_ratio()` (scheduler.py:217-242) computes a theoretical speedup:
   ```python
   single_time = num_micro_batches * single_stage_time_ms
   pipeline_time = bottleneck_ms * (num_stages + num_micro_batches - 1)
   speedup = single_time / pipeline_time
   ```
   This formula assumes ideal conditions (no network overhead, perfect load balance). Actual distributed systems would have significantly lower speedups due to serialization, network transfer, and synchronization costs.
4. **Assumptions not stated** — The README's footnote "Actual speedup depends on model architecture, batch size, network topology, and node homogeneity" acknowledges variability, but the numbers themselves are presented as concrete benchmarks without referencing any empirical test.

**Severity: High.** These are specific numerical claims presented as benchmark results, but they are untestable — no test or benchmark in the repo reproduces them. The referenced test file doesn't exist.

---

## Additional Findings

### A. Project Structure Mismatch
The README (lines 220-248) shows a project structure where LACP lives at `synapse/lacp/`. The actual location is `backend/synapse/topology/`. The tests directory is shown as `tests/` but actually lives at `backend/tests/`. The file `test_cluster_scale.py` listed in the README does not exist.

### B. Hierarchical Clusterer — Algorithm Documentation Bug
The docstring (`latency_matrix.py:239-241`) says:
> "Uses agglomerative (bottom-up) clustering with single-linkage distance metric (minimum inter-node latency)."

But the implementation (latency_matrix.py:271-279) uses **complete linkage** (maximum inter-node latency):
```python
max_dist = max(max_dist, d)
```
This is the opposite of single-linkage.

### C. Two Independent LatencyProber Implementations
- `synapse/topology/latency_matrix.py:128-225` — A TCP-based prober using `asyncio.open_connection`. Implements `probe_all()` with concurrent probing.
- `synapse/networking/latency_probing.py:36-271` — A Tailscale/ping-based prober targeting `PeerHandle`. Its `probe_single()` method is documented as a placeholder (line 157-162: "Ở đây là placeholder").

The networking version at `latency_probing.py:115-143` has a `get_matrix()` method that explicitly falls back to defaults:
```python
# latency_probing.py:128-131
# Default latency: 50ms for same cluster, 200ms for cross-cluster
matrix[peer_id][other_id] = 50.0
```
This means actual network probing is **never performed** in the current code — it always uses placeholder defaults.

### D. Latency Matrix in node.py also uses defaults
In `node.py:1461`, the latency cache initializes to 50ms for all inter-node pairs:
```python
self._latency_cache[nid] = {other: 0.0 if other == nid else 50.0 for other in node_ids}
```
The only dynamic update comes from forward-pass latencies (`_last_forward_latencies`, line 1464), which measures application-level tensor forwarding latency, not network ping latency.

---

## Summary

| Claim | Verdict | Severity |
|---|---|---|
| 1. Multi-resource LACP (8 resource types) | PARTIALLY — All 8 in schema, but disk I/O and bandwidth are never populated; effectively dead code | Medium |
| 2. README table showing resource weighting | PARTIALLY — Table exists but "Primary"/"Secondary" labels misrepresent equal weights | Low |
| 3. <1ms → adjacent shards; cross-region → independent | PARTIALLY — Clustering exists but threshold is 5ms (not <1ms); adjacency claim verified | Medium |
| 4. Hierarchical clustering: sub-1ms / >10ms thresholds | NOT FOUND — Specific thresholds absent; algorithm uses complete linkage despite docstring claiming single-linkage | Medium |
| 5. Master computes optimal shard assignment via ILP | PARTIALLY — Master computes assignment (✓) but "ILP optimal" is false; uses Largest Remainder Method (greedy) | High |
| 6. Power efficiency weighting | PARTIALLY — Framework present and power data collected, but `power_limit_w` never set, nullifying the 0.05 weight | Medium |
| 7. Scaling: 2.8x/7.2x/55x speedup numbers | NOT FOUND — No benchmark validates these; `test_cluster_scale.py` doesn't exist | High |

**Confidence: 7/10**

The codebase has a well-structured partitioning framework with a coherent architecture, but several production code paths are incomplete (disk I/O, bandwidth, power limit) and the README makes specific claims (ILP, exact speedup numbers, sub-1ms thresholds) that are either exaggerated or unsupported. The Largest Remainder Method is a legitimate proportional allocation scheme but calling it "ILP optimal" is inaccurate. The scaling numbers appear fabricated as no empirical benchmark backs them and the referenced test file is missing.

**Ground truth tally:**
- ☑ Facts verified by code: 6
- ☐ Facts contradicted by code: 4 (ILP claim, disk I/O population, bandwidth population, speedup validation)
- ☐ Facts unverifiable (no code): 1 (test_cluster_scale.py)
- ☒ Documentation bugs found: 2 (single-linkage vs complete linkage, project structure path)
