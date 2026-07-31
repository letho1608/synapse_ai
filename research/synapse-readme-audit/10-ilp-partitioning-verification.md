# ILP Partitioning Verification — Lane 3 Analysis

**Auditor:** Specialist Research Agent (Lane 3)  
**Scope:** Verify H1 fix — ILP (Integer Linear Programming) partitioning implementation  
**Commit Examined:** `91ec971` ("Fix all Blocker and High findings from deep-dive audit")  
**Files Examined:**
- `backend/synapse/topology/partitioner.py` (primary)
- `backend/synapse/orchestration/node.py` (production caller)
- `backend/synapse/main.py` (entry point)
- `backend/tests/test_partitioner_ilp.py` (test suite)
- `backend/requirements.txt` (dependency declaration)
- `README.md` (documentation claims)
- `backend/synapse/topology/resource_weight.py` (scorer used by ILP)
- `backend/synapse/networking/latency_probing.py` (ILP reference)
- `backend/synapse/config/settings.json` (configuration)

---

## 1. PuLP Import Check

### Finding ✅: PuLP is properly imported with try/except guard

**File:** `backend/synapse/topology/partitioner.py`, lines 20–24

```python
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
```

The import uses the standard Python pattern: a `try/except ImportError` block that sets a boolean flag `HAS_PULP`. This is correct and mirrors the same pattern used elsewhere in the codebase (e.g., `resource_weight.py` uses the same pattern for `psutil` and `pynvml`).

**Verdict:** PASS — PuLP import guard is correctly implemented.

### Dependency Declaration

**File:** `backend/requirements.txt`, line 50

```
pulp>=2.7.0
```

The PuLP dependency is declared at `>=2.7.0`, which is sufficient for all features used (binary variables, `LpProblem`, `LpVariable`, `lpSum`, `PULP_CBC_CMD`). Version 2.7.0 or higher is available from PyPI.

**Note:** The constraint `>=2.7.0` is permissive — any 2.7.x or higher version is accepted. This is appropriate for a project dependency.

---

## 2. `compute_partition_ilp()` Function — Full Analysis

### 2.1 Function Signature

**File:** `backend/synapse/topology/partitioner.py`, lines 98–106

```python
def compute_partition_ilp(
    nodes: Dict[str, NodeResources],
    latency_matrix: Optional[LatencyMatrix],
    total_layers: int,
    model_id: str = "",
    weights: Optional[ResourceWeights] = None,
    num_micro_batches: int = 1,
) -> LACPPartition:
```

The function accepts the same parameters as `compute_partition()` (minus `max_intra_cluster_latency_ms` and `method`), returns the same `LACPPartition` type, and includes a comprehensive docstring describing the ILP formulation. **Severity: Note** — `latency_matrix` is accepted as a parameter but is **not actually used** in the ILP formulation (the ILP creates a single cluster with all nodes). The latency information is effectively discarded. This means the ILP solver ignores inter-node latency when partitioning.

### 2.2 Fallback When PuLP Missing

**File:** `backend/synapse/topology/partitioner.py`, lines 128–136

```python
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
```

When PuLP is not installed, `compute_partition_ilp()` calls `compute_partition()` **without specifying `method`**, which defaults to `"lrm"`. This is correct fallback behavior.

**Verdict:** PASS — fallback to LRM when PuLP is absent is correctly implemented.

### 2.3 Decision Variables

**File:** `backend/synapse/topology/partitioner.py`, lines 161–178

Three sets of binary variables are defined:

```python
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
```

All three sets use `cat=pulp.LpBinary`, making them proper 0–1 binary variables. This is correct.

**Variable counts for N nodes and L layers:**
- `x`: N × L binary variables
- `y`: N × L binary variables  
- `z`: N × (N−1) binary variables (all ordered pairs of distinct nodes)

**Total binary variables:** 2NL + N(N−1)

For the documented max of 32 nodes and a 70B-parameter model (~80 layers): 2×32×80 + 32×31 = 5,120 + 992 = **6,112 binary variables**. This is within the range that CBC can handle, though solve time may be non-trivial.

### 2.4 Constraint 1: Exact Coverage (Each Layer → Exactly One Node)

**File:** `backend/synapse/topology/partitioner.py`, lines 181–183

```python
# Constraint 1: Each layer assigned to exactly one node
for l in range(total_layers):
    prob += pulp.lpSum([x[(i, l)] for i in range(num_nodes)]) == 1
```

This is correct: for every layer `l`, the sum of `x[i,l]` over all nodes equals 1, meaning exactly one node processes each layer.

### 2.5 Constraint 2: At Most One Start Per Node

**File:** `backend/synapse/topology/partitioner.py`, lines 185–187

```python
# Constraint 2: Each node has exactly one start position (or 0 if no layers)
for i in range(num_nodes):
    prob += pulp.lpSum([y[(i, l)] for l in range(total_layers)]) <= 1
```

This limits each node to at most one start position. Combined with the `y`-to-`x` linkage (Constraint 4), this is the mechanism that enforces contiguity. If a node had two disconnected layer blocks, it would produce two `y=1` positions and violate this constraint.

### 2.6 Constraint 3: First Layer Must Start Somewhere

**File:** `backend/synapse/topology/partitioner.py`, lines 189

```python
# Constraint 3: First layer must start somewhere
prob += pulp.lpSum([y[(i, 0)] for i in range(num_nodes)]) == 1
```

This ensures exactly one node is designated as starting at layer 0. This is correct.

### 2.7 Constraint 4: Link y to x (Start Position Definition)

**File:** `backend/synapse/topology/partitioner.py`, lines 192–202

```python
# Constraint 4: y[i][l] == 1 iff x[i][l] == 1 and (l==0 or x[i][l-1]==0)
for i in range(num_nodes):
    for l in range(total_layers):
        if l == 0:
            prob += y[(i, l)] >= x[(i, l)]
            prob += y[(i, l)] <= x[(i, l)]
        else:
            prob += y[(i, l)] <= x[(i, l)]
            prob += y[(i, l)] <= 1 - x[(i, l-1)]
            prob += y[(i, l)] >= x[(i, l)] - x[(i, l-1)]
```

For `l = 0`: `y[i,0] == x[i,0]` — node i starts at layer 0 if and only if it processes layer 0.

For `l > 0`: The three inequalities implement `y[i,l] = x[i,l] AND (NOT x[i,l-1])`:
- `y[i,l] ≤ x[i,l]`: if y=1 then x[i,l]=1 ✓
- `y[i,l] ≤ 1 − x[i,l-1]`: if y=1 then x[i,l-1]=0 ✓
- `y[i,l] ≥ x[i,l] − x[i,l-1]`: if x[i,l]=1 and x[i,l-1]=0 then y≥1 (so y=1); if x[i,l]=1 and x[i,l-1]=1 then y≥0 ✓

**Verdict:** PASS — this correctly linearizes the logical condition "y[i,l] = 1 iff node i starts at layer l."

### 2.8 Constraint 5: "Contiguity via Ordered Positions" — ⚠️ BROKEN

**File:** `backend/synapse/topology/partitioner.py`, lines 207–215

```python
# Constraint 5: Contiguity via ordered positions
# Each node's layers must be contiguous
M = total_layers
for i in range(num_nodes):
    for l in range(total_layers):
        prob += pulp.lpSum([x[(i, k)] for k in range(l)]) <= l * (1 - y[(i, l)]) + l * y[(i, l)]
        for k in range(l, total_layers):
            prob += x[(i, k)] <= 1 - y[(i, l)] + x[(i, k)]
```

**Finding: BLOCKER — These constraints are algebraically redundant (always satisfied) and do nothing.**

Let's analyze each:

**Line 213:** `sum_{k<l} x[i,k] ≤ l·(1−y[i,l]) + l·y[i,l]`

The right-hand side simplifies: `l·(1−y) + l·y = l − l·y + l·y = l`

So the constraint becomes: `sum_{k<l} x[i,k] ≤ l`

Since the sum of `l` binary variables cannot exceed `l`, this is **always true** regardless of the values of `x` and `y`. The constraint imposes nothing.

The intended constraint was presumably: `sum_{k<l} x[i,k] ≤ l·(1−y[i,l])` — i.e., if `y[i,l]=1` (node starts at l), then no layers before l belong to node i. But the `+ l·y[i,l]` term defeats the purpose.

**Lines 214–215:** `x[i,k] ≤ 1 − y[i,l] + x[i,k]`

This simplifies to `0 ≤ 1 − y[i,l]`, which is always true since `y` is binary (y=0 → 0≤1 ✓; y=1 → 0≤0 ✓). Again, a no-op.

**Why the ILP still works despite this bug:**

Contiguity is actually enforced by the combination of **Constraint 2** (at most one start per node) and **Constraint 4** (definition of start positions). If a node had two disconnected layer blocks (e.g., layers 0–2 and 5–7), that would produce two "start" positions (at layers 0 and 5), violating Constraint 2. Therefore, the broken Constraint 5 does not affect correctness — contiguity is implicitly enforced by Constraints 2+4 alone.

**Severity: Medium** — The bug does not break the ILP, but the code is misleading. The comments claim these constraints enforce contiguity, but they are dead code. A reader (or future maintainer) would be confused. These constraints should either be corrected (with the proper Big-M formulation) or removed entirely.

### 2.9 Ordering Constraints (z Variables)

**File:** `backend/synapse/topology/partitioner.py`, lines 220–242

```python
# Antisymmetry
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            prob += z[(i, j)] + z[(j, i)] <= 1

# Transitivity
for i in range(num_nodes):
    for j in range(num_nodes):
        for k in range(num_nodes):
            if i != j and j != k and i != k:
                prob += z[(i, j)] + z[(j, k)] - 1 <= z[(i, k)]

# At least (num_nodes-1) ordering relations
prob += pulp.lpSum([z[(i, j)] for ...]) >= num_nodes - 1

# Link y to z
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            for l1 in range(total_layers):
                for l2 in range(total_layers):
                    if l1 < l2:
                        prob += z[(i, j)] >= y[(i, l1)] + y[(j, l2)] - 1
```

These constraints define a total order over all nodes that is consistent with their start positions.

**Finding: HIGH — The y→z link has O(N²·L²) constraints, which is intractable for realistic workloads.**

For N=32 nodes and L=80 layers: approximately 32×31×80×79/2 ≈ **3.1 million constraints** just for this single link. This will cause extreme model construction and solve times.

A more efficient formulation would be:
- Use the `z` variables only for nodes that actually receive layers.
- Use a smaller number of ordering constraints (e.g., only requiring that start positions are sorted).

Alternatively, the entire `z` variable machinery could be dropped in favor of simply requiring that the start positions correspond to a permutation of node indices, which can be enforced more cheaply.

**Severity: High** — The quadratic-quartic blowup makes the ILP impractical for the documented 32-node maximum. In practice, the solver will spend most of its time processing redundant ordering constraints rather than optimizing the objective.

### 2.10 Objective Function

**File:** `backend/synapse/topology/partitioner.py`, lines 246–258

```python
max_load = pulp.LpVariable("max_load", lowBound=0)

for i, nid in enumerate(node_ids):
    score = scores.get(nid, (1.0, None))[0]
    if score <= 0:
        score = 1.0
    layers_on_node = pulp.lpSum([x[(i, l)] for l in range(total_layers)])
    prob += max_load >= layers_on_node / score

prob += max_load
```

The objective minimizes `max_load`, which is bounded below by each node's layer-count-to-score ratio. This is a **minimax formulation** (also called "bottleneck" or "Chebyshev" objective): it minimizes the maximum normalized load across all nodes. This is correct for achieving balanced partitioning.

**Verdict:** PASS — the minimax objective is correctly formulated.

However, note that:
- `prob += max_load` at line 258 adds the variable itself to the objective. In PuLP, this makes `max_load` the minimization target. This is correct.
- **Finding: Note** — The default solver (`PULP_CBC_CMD`) is invoked with `msg=False`, suppressing solver output. No time limit or optimality gap is set. For a problem with thousands of binary variables, the solver could take arbitrarily long on larger instances.

### 2.11 Solve and Solution Extraction

**File:** `backend/synapse/topology/partitioner.py`, lines 259–329

The solution is extracted by:
1. Reading `x[i,l]` values to build per-node layer lists
2. Reading `y[i,l]` values to determine start positions
3. Sorting nodes by start position
4. Building `ShardAssignment` objects

**Finding: Note — The extraction code has defensive gap-filling that could mask solver errors.**

At lines 304–312:
```python
current_layer = 0
for nid in ordered_nodes:
    layers = node_layers.get(nid, [])
    if not layers:
        continue
    start_l = min(layers)
    end_l = max(layers)
    if start_l > current_layer:
        start_l = current_layer
```

If the solver produces a gap (a layer assigned to no node), the extraction code silently fills it by adjusting `start_l`. While the ILP constraints should prevent gaps, this defensive code could mask a real solver error.

At lines 326–329:
```python
if partition.assignments and partition.assignments[-1].end_layer < total_layers - 1:
    partition.assignments[-1].end_layer = total_layers - 1
```

Similarly, the last node is extended to cover any unassigned trailing layers. This is defensive but could mask an incorrect solution.

**Severity: Low** — These defensive fixes don't affect correctness when the ILP works, but they make silent data corruption possible on solver failure.

### 2.12 Latency Matrix Not Used

**File:** `backend/synapse/topology/partitioner.py`, lines 295–303

```python
# One cluster for ILP (latency clustering happens inside the ILP weight)
partition.clusters = [Cluster(
    cluster_id=0,
    nodes=list(nodes.keys()),
    centroid_latency_ms=0.0,
    intra_latency_ms=0.0,
)]
```

The ILP partitioner ignores the `latency_matrix` parameter entirely, creating a single cluster with all nodes and zero latencies. The comment claims "latency clustering happens inside the ILP weight" but no latency data appears in the objective function or any constraint. The `latency_matrix` argument is accepted but never referenced after the parameter signature.

**Severity: Medium** — The documentation implies the ILP is latency-aware, but it is not. The objective only considers resource scores (VRAM, TFLOPS, etc.), not inter-node latency. The LRM path does use latency clustering via `HierarchicalClusterer`, making it more latency-aware than the ILP path in this regard.

---

## 3. `method="ilp"` Routing in `compute_partition()`

### Findings ✅: Routing is correctly implemented

**File:** `backend/synapse/topology/partitioner.py`, lines 334–378

```python
def compute_partition(
    ...
    method: str = "lrm",
) -> LACPPartition:
```

The function has a `method` parameter defaulting to `"lrm"`. Lines 370–378 route to the ILP solver when `method == "ilp"`:

```python
if method == "ilp":
    return compute_partition_ilp(
        nodes=nodes,
        latency_matrix=latency_matrix,
        total_layers=total_layers,
        model_id=model_id,
        weights=weights,
        num_micro_batches=num_micro_batches,
    )
```

**Verdict:** PASS — Routing is correct and complete. The `ilp` and `lrm` paths are properly separated.

---

## 4. Fallback Behavior (PuLP Not Installed)

### Finding ✅: Fallback works correctly

**File:** `backend/synapse/topology/partitioner.py`, lines 128–136

As analyzed in §2.2, when `HAS_PULP` is `False`, the ILP function calls `compute_partition()` (which defaults to LRM). This is correct.

**Note:** The fallback creates an infinite-recursion-adjacent design: `compute_partition(method="ilp")` calls `compute_partition_ilp()`, which calls `compute_partition()` again. However, since `compute_partition_ilp()` calls `compute_partition()` without a `method` argument (defaulting to `"lrm"`), there is no recursion risk.

### Finding: Note — The fallback is inefficient.

When PuLP is absent, every call to `compute_partition_ilp()` first:
1. Creates an empty `LACPPartition` object (line 139)
2. Computes all resource scores (lines 150–151)
3. Checks `if not nodes` and `if num_nodes == 0` (lines 154–156)
4. Then falls back to `compute_partition()`, which recomputes everything

The overhead is negligible for a single call, but the early return should ideally be placed before the scoring work.

---

## 5. Production Wiring

### Finding: BLOCKER — The ILP path is NOT reachable from production code

**File:** `backend/synapse/orchestration/node.py`

Two call sites exist:

**Line 1433 (main partitioning):**
```python
lacp_result = compute_partition(
    nodes=nodes_resources,
    latency_matrix=latency_matrix,
    total_layers=num_layers,
    model_id=model_id,
)
# No method="ilp" — uses default "lrm"
```

**Lines 401–408 (repartition after failure):**
```python
result = compute_partition(
    nodes=nodes_resources,
    total_layers=total_layers,
    model_id="",
)
# No method="ilp" — uses default "lrm"
```

Both call sites use the default `method="lrm"`. There is:

1. **No configuration option** to enable ILP — `settings.json` contains only `default_model` and `discovery_module`
2. **No CLI flag** to enable ILP — `backend/synapse/main.py` line 121 references `compute_partition()` with no mention of `method`
3. **No environment variable** check for ILP
4. **No automatic switching** to ILP based on cluster size or other criteria

The entire ILP code path is accessible **only via direct API calls** or **tests**. A production user running `python main.py` will never trigger the ILP solver.

**Severity: Blocker** — The H1 finding was that "ILP optimal assignment" was claimed but no ILP code existed. The fix adds the code, but the claim remains unfulfilled because the ILP code is never invoked in production. The user must modify source code to use it.

**File:** `backend/synapse/main.py`, line 121

```python
# LACP multi-resource partitioning is built into Node via compute_partition()
```

This comment does not mention ILP or the `method` parameter. It does not help an operator discover that ILP exists.

---

## 6. README Consistency

### Finding ✅: README correctly documents both methods

**File:** `README.md`

Line 75:
```
- **`method="ilp"`** — Integer Linear Programming via PuLP: binary decision variables `x_{i,l}` with exact-coverage + contiguity constraints, minimizing max latency-weighted load. Optimal but slower; use for small clusters (≤32 nodes) or when shard balance is critical
```

Line 75 (same paragraph):
```
- **`method="lrm"`** (default) — Largest Remainder Method: fast greedy proportional allocation, ideal for large clusters
```

Line 199:
```
4. **LACP** → master computes optimal shard assignment via Largest Remainder Method (`method="lrm"`, fast) or ILP (`method="ilp"`, optimal via PuLP binary variables `x_{i,l}`)
```

The README accurately describes both methods, their trade-offs, and the recommended cluster size for ILP (≤32 nodes). The variable naming `x_{i,l}` matches the actual code.

**Finding: Note** — The README does not mention that ILP is not enabled in production by default. A reader who follows the Quick Start guide will never see ILP in action.

---

## 7. Test Suite Analysis

**File:** `backend/tests/test_partitioner_ilp.py` (151 lines, 7 tests)

### Tests present:

| # | Test | Coverage | Requires PuLP |
|---|------|----------|--------------|
| 1 | `test_pulp_available` | PuLP importable | No |
| 2 | `test_pulp_imported_in_partitioner` | `import pulp` in source | No |
| 3 | `test_compute_partition_ilp_function_exists` | Function callable | No |
| 4 | `test_ilp_method_parameter` | `method` param exists | No |
| 5 | `test_ilp_produces_valid_partition` | 2 nodes, 24 layers, checks coverage + node0 > node1 | Yes |
| 6 | `test_ilp_full_coverage` | 3 nodes, 30 layers, exact-set comparison | Yes |
| 7 | `test_ilp_defaults_to_lrm_without_pulp` | Both methods produce valid partitions | Yes |

### Test quality assessment:

**Strengths:**
- Tests 5 and 6 verify basic correctness: full coverage, no gaps, stronger node gets more layers.
- Tests 1–4 verify structural properties without needing PuLP installed.
- Proper use of `@pytest.mark.skipif(not HAS_PULP, ...)` for PuLP-dependent tests.

**Weaknesses:**
- **No optimality test:** Test 5 checks that node0 (stronger) gets more layers than node1, but doesn't verify the assignment is actually optimal (e.g., by comparing against a brute-force optimum for a tiny case).
- **No contiguity stress test:** Test 6 checks coverage via set comparison (`all_layers == expected`), which catches gaps but not non-contiguous assignments. Since each assignment already reports contiguous `start_layer`–`end_layer`, if the solver returned non-contiguous layers, the extraction code would produce overlapping or wrong ranges. The test doesn't verify that `start_layer ≤ end_layer` or that the assignments don't overlap.
- **No large-cluster test:** The tests use 2 or 3 nodes with 24–30 layers. No test exercises the 32-node / 80-layer case that the README documents, nor measures solve time.
- **No solver-status check:** The test never checks `pulp.LpStatus[prob.status]` to confirm the solver found an optimal (or even feasible) solution. It trusts that `assignments` is non-empty.
- **Test 7 name mismatch:** `test_ilp_defaults_to_lrm_without_pulp` — the name suggests testing LRM fallback, but the code runs ILP with PuLP available and only asserts both are valid. The fallback behavior (without PuLP) is not actually tested because it requires PuLP to be absent.

**Severity: Medium** — Tests validate basic functionality but miss optimality, solver-status checking, and large-scale behavior.

---

## 8. Summary of All Findings

| # | Severity | Finding | File:Line |
|---|----------|---------|-----------|
| 1 | **Blocker** | ILP path is not wired into production code — `node.py` calls `compute_partition()` without `method="ilp"` at both call sites. Users must modify source code to use ILP. | `node.py:401,1433` |
| 2 | **High** | The y→z ordering link creates O(N²·L²) constraints (~3.1M for 32 nodes, 80 layers), making the model intractable for documented cluster sizes. | `partitioner.py:237–242` |
| 3 | **Medium** | Constraint 5 ("contiguity via ordered positions") is algebraically redundant — both sub-constraints simplify to always-true. Comments are misleading. | `partitioner.py:213–215` |
| 4 | **Medium** | The ILP path ignores the `latency_matrix` parameter entirely, creating a single dummy cluster. LRM is more latency-aware because it uses `HierarchicalClusterer`. | `partitioner.py:295–303` |
| 5 | **Low** | Defensive gap-filling in the extraction code (lines 310–311, 326–329) can silently mask solver errors by patching coverage gaps. | `partitioner.py:310,326` |
| 6 | **Note** | Tests validate basic coverage but never check solver status, optimality, or large-cluster performance. Fallback test name is misleading. | `test_partitioner_ilp.py:134–150` |
| 7 | **Note** | No solver timeout or optimality gap is set — the ILP could run arbitrarily long on larger instances. | `partitioner.py:259` |
| 8 | **Note** | The fallback in `compute_partition_ilp()` computes scores before discovering PuLP is absent (minor redundant work). | `partitioner.py:128–136` |

---

## 9. Executive Summary

Phase 1 identified **H1 (High):** the codebase claimed ILP-based partitioning but used only the Largest Remainder Method (greedy proportional allocation). Commit `91ec971` implements a genuine PuLP-based ILP solver (`compute_partition_ilp`) with binary decision variables, exact-coverage and contiguity constraints, and a minimax load-balancing objective. The implementation is structurally correct: PuLP is imported with a proper try/except guard, the `method="ilp"` parameter routes correctly through `compute_partition()`, and the fallback to LRM when PuLP is absent works. The README accurately documents both methods and their trade-offs.

However, two critical issues remain. **First and most severe:** the ILP solver is never called in production. Both call sites in `node.py` use the default `method="lrm"`, with no configuration option, CLI flag, or automatic switching to enable ILP. The fix adds the code but leaves it unreachable — the same practical effect as the original H1 finding from a user's perspective. **Second:** the ordering constraints linking start positions to pipeline order are formulated with O(N²·L²) constraints, producing ~3.1 million constraints for the documented 32-node maximum, making large instances intractable. Additionally, the ILP ignores the latency matrix entirely, the contiguity-enforcement constraints are mathematically redundant, and the tests never verify solver status or optimality.

The ILP implementation is real and correct in its core formulation, but is **incomplete** as a resolution to H1 because it is not production-wired and would perform poorly at documented scale.

---

## 10. Top 5 Most Concerning Findings

1. **ILP path unreachable in production** — Both call sites in `node.py` (lines 401, 1433) omit `method="ilp"`. The H1 claim remains unfulfilled in practice.
2. **O(N²·L²) ordering constraints** — The y→z link at `partitioner.py:237–242` generates ~3.1M constraints for 32 nodes × 80 layers, making scale-up impractical.
3. **Broken contiguity constraints** — Constraint 5 at `partitioner.py:213–215` is algebraically always-true; the comments promise contiguity but provide dead code.
4. **Latency matrix ignored by ILP** — The ILP path creates a single dummy cluster with zero latency; the LRM path actually uses latency clustering via `HierarchicalClusterer`.
5. **No solver-status verification** — The code never checks `LpStatus` after `solve()`, and tests never validate optimality or even feasibility of the solution.

---

## 11. Confidence Rating

**Confidence: 8/10**

**Ground-truth tally:**
- ✅ PuLP import with try/except guard — **verified** (partitioner.py:20–24)
- ✅ `compute_partition_ilp()` exists with correct signature — **verified** (partitioner.py:98–106)
- ✅ Binary decision variables — **verified** (partitioner.py:165,171,178)
- ✅ Exact-coverage constraint (each layer → one node) — **verified** (partitioner.py:182–183)
- ✅ Contiguity enforced via start-position constraints — **verified** (partitioner.py:185–202)
- ✅ Minimax objective minimizes max load/score — **verified** (partitioner.py:246–258)
- ✅ `prob.solve()` called — **verified** (partitioner.py:259)
- ✅ `method` parameter on `compute_partition()` — **verified** (partitioner.py:342)
- ✅ Routing to ILP when `method="ilp"` — **verified** (partitioner.py:370–378)
- ✅ Fallback to LRM when PuLP absent — **verified** (partitioner.py:128–136)
- ✅ README documents both methods — **verified** (README.md:75,199)
- ❌ ILP wired into production — **NOT VERIFIED** — `node.py` never passes `method="ilp"` (node.py:401,1433)
- ❌ Constraint 5 is correct — **FALSE** — algebraically redundant (partitioner.py:213–215)
- ❌ Solver status checked after `solve()` — **NOT VERIFIED** — no call to `LpStatus` (partitioner.py:259)
- ⚠️ Latency matrix used by ILP — **PARTIALLY** — parameter accepted but ignored (partitioner.py:100 vs 295)
- ⚠️ Large-scale performance — **UNKNOWN** — no tests >3 nodes to verify tractability

**Confidence deduction reasoning:**
- −1 for incomplete production wiring (H1 fix is only half-done)
- −1 for unverified solver behavior (no status check, no optimality validation)
- The core ILP formulation is correct, contiguity is actually enforced (by constraints 2+4 despite broken constraint 5), and the routing/fallback logic is sound. The -2 points reflect that the fix addresses the code-level finding but leaves the user-facing issue unresolved and has potentially severe performance scaling problems.
