# Phase 3 Verification — Lane B: LACP Partitioning, ILP Reachability, Multi-Resource Election & Resource Measurement

**Agent lane:** LACP partitioning, ILP reachability, multi-resource election, resource measurement.
**Scope note:** `node.py`'s partition path (`_build_latency_matrix`, `_compute_lacp_partition`, election wiring, bandwidth probing) belongs to this lane. Fault-tolerance/1F1B/transport/UI claims are covered by lanes C/D.
**Method:** Read-only source audit + independent empirical verification (tests rerun by me, plus two adversarial experiments). No source files were modified.
**Date:** 2026-07-31. All pytest runs were executed from `D:\Code\paper\synapse_ai\backend` with `python -m pytest <files> -v` (Python 3.10.8, pytest 8.2.2, PuLP installed).

---

## 0. Verdict in one paragraph

The four original defects named in the prior audit (`research/synapse-readme-audit/15-executive-briefing.md`) **were genuinely addressed in code**: `_build_latency_matrix()` now returns a real `LatencyMatrix` (not a dict); `compute_partition(method="ilp", ...)` is implemented with a working PuLP solver and non-optimal-status fallback; the `Node` constructor wires a `ResourceScorer` into `ElectionManager` and `set_base_compute_weight()` derives weight from it; and `apply_measured_bandwidth()` propagates matrix bandwidth into `NodeResources.bandwidth_mbps`. All 36 tests in the six suites assigned to this lane pass. **However**, three claims remain materially untrue of shipped code: (1) measured network bandwidth can *never* actually be measured against the production P2P bridge — the probe protocol has no server-side echo handler and my live experiment proves it fails (the 1000 Mbps placeholder therefore stands forever in real deployments); (2) master **stickiness is broken** — I reproduced a **split-brain** (both nodes `is_master()==True`) when a live master's weight drops below a peer's, which the README explicitly claims is prevented; (3) ILP and the entire multi-resource LACP path are **not reachable from any production knob** — `partition_method`/`tflops_proportional` are plain attributes flipped only by tests, and ILP takes ~40 s at 4 nodes / >300 s at 8 nodes, contradicting the README's "≤32 nodes" usability claim. Details with file:line citations below.

---

## 1. Latency-matrix type flow (README: latency-aware clustering, N×N probing matrix) — FIXED

### 1.1 The call chain

`Node._build_latency_matrix()` is defined at `backend/synapse/orchestration/node.py:1606` with an explicit return annotation `Optional[LatencyMatrix]`. It builds the matrix object at:

- `node.py:1670` — `matrix = LatencyMatrix()`
- `node.py:1673–1679` — fills it via `matrix.set(LatencyEntry(from_node=nid, to_node=other, latency_ms=float(latency_ms), bandwidth_mbps=bw))`
- `node.py:1681–1682` — `return matrix`

So the value returned to callers is a `LatencyMatrix` (dataclass with `.nodes`, `.entries`, `.get`, `.set` — `backend/synapse/topology/latency_matrix.py:47–76`), **not a dict**. The pre-fix `AttributeError ("'dict' object has no attribute 'nodes'")` failure mode is gone.

### 1.2 The consumer side

The production path consumes it at `node.py:1588–1597` (inside `_compute_lacp_partition`, the `tflops_proportional=False` branch):

```python
lacp_result = compute_partition(
    nodes=nodes_resources,
    latency_matrix=latency_matrix,
    total_layers=num_layers,
    model_id=model_id,
    method=self.partition_method,
)
```

`compute_partition` at `backend/synapse/topology/partitioner.py:346` declares `latency_matrix: Optional[LatencyMatrix]`, and the code that dereferences it — `partitioner.py:405–410`, `if latency_matrix and len(latency_matrix.nodes) >= 2:` then `clusterer.cluster(latency_matrix)` — is type-correct against `LatencyMatrix`. `HierarchicalClusterer.cluster` (`latency_matrix.py:249–331`) reads `matrix.nodes` and `matrix.entries`, which exist on the dataclass. The fault-tolerance repartition closure (`node.py:455–468`) uses the identical pattern (`latency_matrix = self._build_latency_matrix()` → `apply_measured_bandwidth(...)` → `compute_partition(...)`) and is also type-safe.

### 1.3 Test evidence

`backend/tests/test_multinode_lacp_flow.py` (5/5 passed, 3.6 s): `test_build_latency_matrix_returns_latency_matrix` asserts `isinstance(matrix, LatencyMatrix)` and presence of both directional entries; `test_production_lacp_partition_three_nodes_no_crash` and `..._two_nodes_no_crash` drive `Node._compute_lacp_partition(base_shard)` with `tflops_proportional=False` on 2- and 3-node topologies and assert contiguous, full-coverage partitions with no exception.

**Verdict: TRUE.** The type mismatch that crashed any ≥2-node LACP deployment is fixed, and behavioral tests drive the shipped code to prove it.

### 1.4 Residual issue (Note) — latent-cache staleness on topology growth

`node.py:1615–1619` initializes `self._latency_cache` only once (guarded by `if not hasattr(self, '_latency_cache')`), keyed by the node IDs present on the **first** call. If the cluster later grows (a node joins), the new node's ID is absent from the cache, so the loop at `node.py:1671–1679` produces no matrix rows for it; `LatencyMatrix.nodes` (populated only via `matrix.set`) then omits the new node, and `compute_partition`'s clustering/`assignments` silently excludes it from the partition (it gets no layers). If a node departs, its stale rows *remain* in the cache and the departed ID reappears in `matrix.nodes`, distorting `HierarchicalClusterer` distances (`latency_matrix.py:274–279` uses `distances.get((a,b), 1e9)`). Neither case crashes; both silently degrade partition correctness for dynamic clusters. Severity: **Medium** (multi-resource path only, dynamic membership only).

---

## 2. ILP reachability (README: `method="ilp"` via PuLP binary variables x_{i,l}, exact coverage + contiguity) — PARTIALLY TRUE

### 2.1 The solver is real and correct

`compute_partition(method="ilp", ...)` routes to `compute_partition_ilp` at `partitioner.py:381–391`. The solver (`partitioner.py:98–345`):

- Declares binary `x[i][l]` ("1 if node i processes layer l") at `partitioner.py:161–164`, plus `y[i][l]` start markers (`:169–172`) and `z[i][j]` pipeline-order vars (`:174–177`) — matches the README's "binary decision variables x_{i,l}".
- Exact coverage: `partitioner.py:181–183`, `sum_i x[i][l] == 1` for every layer.
- Contiguity: genuinely enforced, but by constraints 2+4 (`partitioner.py:185–204`: each node has ≤1 start; `y[i][l] == 1 ⟺ x[i][l]==1 and x[i][l-1]==0`). A node's layers therefore form a single contiguous block. The two constraints *labeled* "Contiguity via ordered positions" at `partitioner.py:205–218` are **vacuous as written** (both reduce to `0 ≤ 1 - y[i,l]` or `y[i,l] ≤ 1`; the `M = total_layers` variable at `:208` is unused). Dead constraint code with misleading comments — **Note**, not a correctness failure, since coverage + block-uniqueness still guarantee contiguity.
- Solver status handling: `partitioner.py:259–272` — `status = prob.solve(pulp.PULP_CBC_CMD(msg=False))`; `if pulp.LpStatus[status] != "Optimal":` falls back to the LRM path so the cluster always receives a valid contiguous partition. Handled.
- Solution extraction (`partitioner.py:274–345`) returns `ShardAssignment`s with start/end layers, and patches full coverage (`:339–341`).

Tests `backend/tests/test_partitioner_ilp.py` (7) and `backend/tests/test_ilp_production_flow.py` (4) — **11/11 passed in 26.1 s**. Behavioral checks drive `compute_partition(method="ilp")` and the production `Node._compute_lacp_partition` with `node.partition_method = "ilp"`, asserting stronger-node-bias, exact 0..L−1 coverage, and adjacency.

### 2.2 But it is NOT selectable from any production path

The "config knob" (`node.py:119–122`):

```python
# Partition method for the LACP multi-resource path: "lrm" ... or "ilp" ...
self.partition_method: str = "lrm"
```

- `Node.__init__` (`node.py:35–48`) accepts **no** `partition_method` parameter.
- The shipped entrypoint `backend/synapse/main.py` constructs the Node at `:162–176` and passes only `max_generate_tokens`, `topology_viz`, `default_sample_temperature`. The full argparse block (`main.py:49–92`) has **no** `--partition-method` flag; `settings.json` (`backend/synapse/config/settings.json`) has only `default_model` and `discovery_module`.
- A repo-wide grep for `partition_method` finds exactly three assignments of `"ilp"`: `backend/tests/test_ilp_production_flow.py:85` and `:110`. Nothing in `backend/synapse/` (outside the test file) sets it.

So the *only* way to enable ILP in a real deployment is to edit source or poke a private attribute post-construction. `test_ilp_production_flow.py` itself concedes this: it sets `node.partition_method = "ilp"` directly and asserts the node *passes it through*. The pass-through is real (`node.py:1593`), but there is no user-facing selection mechanism. The README's "master computes ... via LRM (fast) or ILP (optimal)" implies an operator-selectable choice. **Severity: High** (feature is inert in shipped product).

### 2.3 ILP is not practical for the claimed "≤32 nodes"

I timed `compute_partition(method="ilp", total_layers=32)` myself:

| Nodes | Wall time (CBC) |
|-------|-----------------|
| 2 | 2.8 s |
| 3 | 3.7 s |
| 4 | 39.2 s |
| 8 | **killed at 300 s** |

The formulation carries O(n²·L²) y-linking constraints (`partitioner.py:239–244`: nested `l1`/`l2` loops over all layer pairs per ordered node pair) plus O(n³) transitivity constraints (`:228–234`), producing an extremely weak MILP. 4 nodes = ~40 s; 8 nodes > 5 minutes. The README's "use for small clusters (≤32 nodes)" (`README.md:76`) is unsubstantiated — even 4–8 nodes is unusable, and this is a *partition hot path* (see §5). **Severity: High.**

---

## 3. Multi-resource election (README: master elected by LACP composite score; fallback VRAM_GB + FP32_TFLOPS/100; master stickiness) — WIRED, BUT STICKINESS BROKEN

### 3.1 Wiring is present and correct

- `node.py:69` — `self._resource_scorer = ResourceScorer()`
- `node.py:70–73` — `self.election_manager = ElectionManager(self.id, self.event_router, resource_scorer=self._election_resource_scorer)` — the constructor accepts the callable (`election.py:20–28`).
- `node.py:1524–1548` — `_election_resource_scorer(node_id)` builds `NodeResources` for the whole known topology, overlays live metrics for the local node (`gpu_util_pct`, `ram_used_mb`, `current_power_w`, `:1536–1538`), calls `apply_measured_bandwidth`, and returns the LACP composite from `ResourceScorer.compute_scores`.
- `node.py:1551–1560` — `_refresh_election_weight()` computes `vram_gb = memory/1024` and `fp32_flops`, calls `set_base_compute_weight(vram_gb, fp32_flops)`. It runs from `start()` (`:204`) and every `collect_topology()` (`:1461`, i.e. every 2 s).
- `election.py:45–57` — `set_base_compute_weight`: `simple_weight = vram_gb + fp32_flops/100.0`; when a scorer is present, `base_compute_weight = max(simple_weight, multi_weight * 1000.0)`. The scorer path takes precedence (the ×1000 scale is an undocumented hack, but it is what makes the composite dominate). Fallback `VRAM_GB + FP32_TFLOPS/100` matches the README formula exactly.

Tests `test_election_bandwidth_wiring.py` (behavioral, drives real `Node` + real `ElectionManager`) and `test_election_multiresource.py` — **14/14 passed in 8.9 s**, including `test_stronger_node_gets_higher_election_weight` (strong node's `compute_weight` > weak node's) and `test_election_weight_derives_from_composite` (weight ≠ 1.0 default, > simple formula).

### 3.2 Master stickiness is broken — split-brain reproduced

The election loop is `election.py:97–152`. The relevant branch (`:130–134`):

```python
elif self.current_master_id in self.active_nodes:      # current master is ALIVE
    self._master_became_offline_at = None
    if new_master_id != self.current_master_id:
        logger.info(f"Node {self.node_id} recognizes {self.current_master_id} as Master (sticky)")
    self.current_master_id = new_master_id             # <-- reassigns despite the log message!
```

The assignment on `:134` executes whenever the current master is alive **regardless of whether `new_master_id` equals it**. The log even prints the *old* master ID while overwriting it. Combined with the sibling branch `:126–129` (a node that *is* master never demotes itself), any weight shift while the master is alive produces **two nodes that each believe they are master**.

**Empirical reproduction** (two `ElectionManager`s on one `EventRouter`, exactly as wired in `test_election_edge.py` but with a post-start weight swap simulating an LACP composite drop during a load spike):

```
initial masters: A=node-a B=node-a
after weight swap: A=node-a B=node-b
A is_master=True, B is_master=True        # SPLIT BRAIN
```

Log output from the run: `Node node-b recognizes node-a as Master (sticky)` — immediately followed by node-b assigning itself master.

The README claims (`README.md:129–132`): *"The election implements **master stickiness**: a master only loses status when truly offline, preventing flapping during transient AI task load spikes."* Two facts make this especially damaging:

1. The LACP weight is *designed* to fluctuate with load — `_election_resource_scorer` folds in live `gpu_util_pct`/`ram_used_mb` (`node.py:1536–1538`) and `_refresh_election_weight()` recomputes every 2 s (`node.py:1461`). A busy master's composite drops below an idle peer's within one topology cycle — exactly the "transient AI task load spike" scenario the README says stickiness prevents.
2. Split-brain on the orchestrator breaks the master-only actions that *do* exist (`broadcast_authoritative_topology`, `node.py:1745–1753`) and the "is_master" gating of partition broadcast (`node.py:1776–1783`).

The offline-master grace path (`election.py:135–144`, 30 s `MASTER_STICKINESS_GRACE_PERIOD`) does work as claimed, but only for dead masters. **Severity: High.** No test in the suite exercises stickiness under a live-master weight shift (test_election_edge.py only tests tie-breaking, timeout cleanup, and static weights).

---

## 4. Resource measurement + bandwidth propagation (README: VRAM/TFLOPS/CPU/RAM/disk/bandwidth/power weighted; bandwidth measured during latency probing) — PLUMBING FIXED, MEASUREMENT IMPOSSIBLE

### 4.1 What is fixed

- `NodeResources` (`resource_weight.py:46–77`) carries all 8 dimensions; `bandwidth_mbps`/`latency_ms`/`jitter_ms` are real fields (`:71–73`).
- `NodeResources.from_device_capabilities` (`:79–115`) populates disk via `_measure_disk_io()` (psutil counters over a 0.5 s sample, `:126–150`), power via `_measure_power()` (pynvml `nvmlDeviceGetPowerManagementLimit/Usage`, `:152–191`), and bandwidth via `_measure_dummy_bandwidth()` (`:195–203`), whose docstring is honest: *"Real bandwidth is measured during latency probing ... This value is a reasonable default for single-node operation."*
- `apply_measured_bandwidth(nodes_resources, latency_matrix, self_node_id)` (`resource_weight.py:206–235`) copies `matrix.get(self_node_id, nid).bandwidth_mbps` (falling back to any touching entry) into `NodeResources.bandwidth_mbps`. It is invoked in the production partition path (`node.py:1584–1586`), the election scorer (`node.py:1541–1545`), and the fault-tolerance repartition (`node.py:459`).
- `ResourceScorer.compute_scores` (`:297–404`) normalizes all 8 dimensions and weights them (`ResourceWeights`, `:248–274`: vram .25, tflops .25, cpu .10, ram .10, disk .05, bandwidth .10, latency .10, power .05), with utilization/RAM-pressure penalties.

Tests: `test_resource_measurement.py` **6/6 passed (1.4 s)**; `test_election_bandwidth_wiring.py` **14/14 passed**, including `test_production_partition_path_propagates_bandwidth` (fakes a completed probe via `node._measured_bandwidth = {"node-b": 500.0}`, then asserts `compute_partition` receives `NodeResources` with `bandwidth_mbps == 500.0`) and `test_apply_measured_bandwidth_helper`.

### 4.2 What is NOT fixed: the probe can never succeed against the shipped bridge

The measurement mechanism is `LatencyProber.probe` (`latency_matrix.py:127–202`): it opens a raw TCP connection to `peer.addr()` (the peer's P2P node port), writes `struct.pack("!dQ", time.time(), random.getrandbits(64)) + b"\x00"*(1024-16)` raw bytes (`:153–155`), and expects an **unframed 1024-byte echo** (`reader.readexactly(self.PROBE_SIZE)`, `:160`).

What listens on the node port is `P2PSocketBridge` (`routing/p2p_socket_bridge.py:63–79`), whose protocol is `[4-byte length][1-byte type][JSON]` and whose first packet **must** be a handshake (`:285–308`). There is no echo handler anywhere in `backend/synapse/` (verified by grep for the probe format `!dQ`/`PROBE_SIZE`: the only producer/consumer pair is in `latency_matrix.py`; `networking/latency_probing.py` contains a *different, placeholder* `LatencyProber` that fabricates 50 ms/200 ms defaults and never opens a socket).

**My live experiment** (production bridge on a port + `LatencyProber.probe` against it):

```
DEBUG  p2p_socket_bridge:_handle_incoming_connection - Incoming connection from ('127.0.0.1', ...)
ERROR  p2p_socket_bridge:_read_packet_with_type - Packet too large: 1104845579 bytes, dropping connection
WARNING p2p_socket_bridge:_handle_incoming_connection - Invalid handshake from ...
asyncio.exceptions.IncompleteReadError: 0 bytes read on a total of 1024 expected bytes
```

The bridge parses the probe's first 4 bytes as a length (~1.1 GB), drops the connection; the probe's `readexactly(1024)` raises `IncompleteReadError`; in production this is swallowed by `except Exception: pass` at `node.py:1652–1653`. Result: `_measured_bandwidth` stays empty, the matrix entries carry `bandwidth_mbps=0.0` (`node.py:1672–1677`), `apply_measured_bandwidth` requires `> 0` (`resource_weight.py:230–232`) and no-ops, and **`NodeResources.bandwidth_mbps` remains 1000.0 — the hardcoded placeholder — forever, in every real ≥2-node deployment.** The tests that "prove" bandwidth wiring never touch the prober: they set `node._measured_bandwidth` by hand (`test_election_bandwidth_wiring.py:133`, `:157`).

Additionally, even if an echo server existed, the "bandwidth" estimate is a single 1 KB message's `PROBE_SIZE*8/RTT` (`latency_matrix.py:169–175`) — a one-in-flight-message throughput proxy that would report ~16 Mbps on a real 1 Gbps/0.5 ms link. Not a meaningful bandwidth measurement. **Severity: High** (README "Measured during latency probing" is false in production), with a **Medium** sub-note for the methodology.

---

## 5. Multi-node LACP runs without crashing — TRUE *when the path is manually enabled*, but the LACP path is OFF by default with no production switch

`backend/tests/test_multinode_lacp_flow.py` **5/5 passed (3.6 s)**: 2- and 3-node production `Node._compute_lacp_partition` (with `tflops_proportional=False` forced in the test) produces valid contiguous full-coverage partitions; `_build_latency_matrix` returns `LatencyMatrix`; SHM wiring test also passes.

However, `node.py:116`: `self.tflops_proportional = True  # Use TFLOPS-proportional sharding by default (Cake concept)`. A repo-wide grep for `tflops_proportional` shows it is set to `False` **only in tests** (`test_1f1b_training_production.py:93`, `test_election_bandwidth_wiring.py:67`, `test_cross_node_recovery.py:188`, `test_ilp_production_flow.py:65`, `test_multinode_lacp_flow.py:67`). No code in `backend/synapse/` — not `main.py`, not the API, not config — ever flips it. So the shipped default partition path is `TFLOPSProportionalPartitioner` (`node.py:1569–1572`), which **bypasses the latency matrix, resource scoring, and `compute_partition` entirely**.

Consequences:

1. **The README's headline LACP path is inert by default.** Every partition the production system computes today is the simple TFLOPS-proportional allocation. The entire LatencyMatrix/ILP/multi-resource machinery — including the freshly-fixed type flow — only executes if an operator hand-edits `node.tflops_proportional = False`.
2. **The per-forward recompute makes the LACP path a hot-path hazard.** `get_partition_index`/`get_current_shard`/`forward_tensor`/`forward_prompt` call `_compute_lacp_partition` on **every** forward when `_authoritative_partitions` is unset (`node.py:1263–1276`, `:1281–1286`, `:1181`, `:1200`, `:1226`). And `_authoritative_partitions` is never set, because `broadcast_authoritative_partition` (`node.py:1773–1785`) has **no callers** (grep confirms) — the "CRITICAL FIX: master broadcasts authoritative partition" path is dead code. I timed the LACP multi-resource partition at **~1.0 s per call for 2 nodes** (two 0.5 s `_measure_disk_io` sleeps via `from_device_capabilities`, `resource_weight.py:99`/`:133`), called on every tensor hop. **Severity: High** for the enabled-but-unreachable + dead-broadcast combination; the timing issue is Medium (only if someone enables the path).

---

## 6. "No mock-only coverage" — VIOLATED in this lane

README.md:29 states: *"All claims backed by real tests — 213 tests across 28 test modules … **No mock-only coverage**."* In this lane:

- `backend/tests/test_election_multiresource.py`:
  - `test_set_base_compute_weight_uses_scorer` — reads `election.py` source and asserts the string `"self._resource_scorer"` is present and `"compute_scores"` is absent (`:34–41`). **Source-string inspection, not behavior.**
  - `test_election_code_uses_multi_weight_over_simple` — asserts `"multi_weight" in code` (`:46–49`). **Source-string inspection.**
  - `test_election_code_imports_resource_scorer` — asserts `"self._resource_scorer is not None" in code` (`:70–75`). **Source-string inspection.**
  - `test_election_accepts_resource_scorer` — `assert hasattr(ElectionManager, '__init__')` (`:16–21`): trivially true of every Python class. **Meaningless.**
- `backend/tests/test_partitioner_ilp.py::test_pulp_imported_in_partitioner` (`:24–31`) — opens `partitioner.py` on an **absolute hardcoded path** (`r"D:\Code\paper\synapse_ai\backend\synapse\topology\partitioner.py"`) and asserts `"import pulp" in code`. Source-string inspection, and non-portable.

The behavioral tests (multinode_lacp_flow, ilp_production_flow, election_bandwidth_wiring, resource_measurement, the remainder of partitioner_ilp) genuinely drive shipped code, and the election/resource behavior they assert does hold — so this is a coverage-honesty defect, not a signal that the wiring is fake. **Severity: Medium** (directly contradicts a README claim; several "tests" would pass even if the implementation were deleted).

---

## 7. What does NOT work (consolidated)

| # | Finding | Evidence | Severity |
|---|---------|----------|----------|
| 1 | **Network bandwidth is never actually measured in production.** `LatencyProber.probe` sends an unframed 1024-byte blob expecting an echo; `P2PSocketBridge` demands `[len][type][JSON]` + handshake and drops the connection. Empirically reproduced: `Packet too large: 1104845579 bytes` → `IncompleteReadError`. Exception swallowed at `node.py:1652`; `bandwidth_mbps` stays at the 1000.0 placeholder (`resource_weight.py:195–203`) indefinitely. | `latency_matrix.py:136–202`; `p2p_socket_bridge.py:285–308`, `:439–459`; live experiment | **High** (README: "Measured during latency probing") |
| 2 | **Master stickiness broken → split-brain.** `election.py:130–134` re-elects a new master while the current master is alive whenever weights shift; the self-master branch `:126–129` never demotes. Reproduced: `A is_master=True, B is_master=True`. LACP weights shift by design (live `gpu_util_pct`, `node.py:1536–1538`, refreshed every 2 s). | `election.py:120–144`; live experiment | **High** (README: "master only loses status when truly offline") |
| 3 | **ILP is not selectable from production.** `partition_method` defaults `"lrm"` (`node.py:121`); no constructor param, no CLI flag (`main.py:49–92`), no config; only tests assign `"ilp"`. | `node.py:35–48`, `:119–122`; `main.py:162–176`; grep | **High** (README: master computes via "LRM or ILP") |
| 4 | **Multi-resource LACP path is off by default and has no production switch.** `tflops_proportional=True` (`node.py:116`) routes everything through `TFLOPSProportionalPartitioner` (`:1569–1572`); only tests flip it. The broadcast that would make partitions authoritative is dead code (`node.py:1773–1785`, no callers), so `_compute_lacp_partition` re-runs per forward. | grep `tflops_proportional`; grep `broadcast_authoritative_partition` | **High** |
| 5 | **ILP unusable at claimed scale.** 4 nodes ≈ 39 s, 8 nodes > 300 s (killed). README says "≤32 nodes". O(n²L²) constraints (`partitioner.py:239–244`). | my timing runs | **High** |
| 6 | **LACP path per-call cost ~1.0 s (2 nodes)** — 0.5 s disk-I/O sleep per node per `from_device_capabilities` (`resource_weight.py:99`, `:133–137`), invoked per forward and every 2 s in the election scorer (`node.py:1531–1538`). | timing run | Medium |
| 7 | **"No mock-only coverage" claim false in this lane** — source-string tests in `test_election_multiresource.py` (`:34–41`, `:46–49`, `:70–75`) and `test_partitioner_ilp.py:24–31`; one `hasattr` tautology. | test sources | Medium |
| 8 | `_latency_cache` never invalidated on topology change → new nodes invisible to partition; departed nodes linger in matrix. | `node.py:1615–1619`, `:1671–1679` | Medium (dynamic membership only) |
| 9 | ILP "contiguity constraints" at `partitioner.py:205–218` are vacuous (dead); contiguity actually enforced by `:185–204`. `M` unused. | source | Note |
| 10 | `networking/latency_probing.py` ships a second, placeholder `LatencyProber` that fabricates 50 ms/200 ms matrices (`:119–136`) — confusing dead code; the name collision with the real prober is a maintenance hazard. | source | Note |
| 11 | Bandwidth estimate, even with an echo server, would be `8*1024/RTT` for one in-flight message — ~16 Mbps on a real 1 Gbps/0.5 ms link. | `latency_matrix.py:169–175` | Note |

## 8. What now works (confirmed fixed)

1. **Latency-matrix type flow** — `_build_latency_matrix()` returns `LatencyMatrix`; `compute_partition`/`HierarchicalClusterer` consume it type-safely. 2- and 3-node production-path partitions run without the historical AttributeError.
2. **ILP solver** — real PuLP formulation, exact coverage, contiguity (via y/x linking), non-Optimal → LRM fallback.
3. **Election wiring** — `Node` builds `ResourceScorer`, passes `_election_resource_scorer` to `ElectionManager`, `_refresh_election_weight` → `set_base_compute_weight` uses the composite (×1000) with the documented VRAM+FP32_TFLOPS/100 fallback; stronger node gets the higher weight.
4. **Bandwidth propagation plumbing** — `apply_measured_bandwidth` correctly copies matrix bandwidth into `NodeResources.bandwidth_mbps`, and both the partition path and the election scorer call it.
5. **Resource measurement** — disk I/O, power, bandwidth-placeholder all populated into `NodeResources`; scorer differentiates on all 8 dimensions.
6. **No crash on multi-node LACP** — when the LACP path is engaged, partitions are valid and contiguous.

---

## 9. Test results (all rerun by me, 2026-07-31)

| Suite | Result |
|-------|--------|
| `tests/test_ilp_production_flow.py tests/test_partitioner_ilp.py` | **11 passed** in 26.1 s |
| `tests/test_election_multiresource.py tests/test_election_bandwidth_wiring.py` | **14 passed** in 8.9 s |
| `tests/test_resource_measurement.py` | **6 passed** in 1.4 s |
| `tests/test_multinode_lacp_flow.py` | **5 passed** in 3.6 s |
| Adversarial: prober vs. production bridge | **FAILS** (probe rejected, no bandwidth) |
| Adversarial: live-master weight swap | **SPLIT-BRAIN** (both is_master=True) |
| Adversarial: ILP timing | 2.8 s / 3.7 s / 39.2 s / >300 s at 2/3/4/8 nodes |

---

## 10. Executive summary (250 words)

The prior audit's four lane-specific defects are genuinely fixed in code and verified: `_build_latency_matrix()` now yields a `LatencyMatrix` (`node.py:1670–1682`), `compute_partition(method="ilp")` runs a real PuLP solve with fallback (`partitioner.py:259–272`), the Node constructor wires `ResourceScorer` → `ElectionManager` → `set_base_compute_weight` (`node.py:69–73`, `election.py:45–57`), and `apply_measured_bandwidth` propagates matrix bandwidth into `NodeResources` (`resource_weight.py:206–235`). All 36 tests in this lane pass. But four README claims remain untrue of shipped code. First, real bandwidth is never measured: the prober's raw-echo protocol has no counterpart on the production `P2PSocketBridge`, which I verified live (`Packet too large: 1104845579 bytes` → connection dropped); `bandwidth_mbps` stays hardcoded at 1000. Second, master stickiness is broken — when a live master's LACP weight drops (which the composite score does by design under load), non-master nodes immediately elect a replacement while the master keeps itself, producing a reproduced split-brain. Third, ILP is not selectable by any production mechanism (attribute only, flipped by tests), and at 4 nodes it takes ~40 s and at 8 nodes exceeds 5 minutes, contradicting "≤32 nodes". Fourth, the entire multi-resource LACP path is off by default (`tflops_proportional=True`) with no production switch, and the authoritative-partition broadcast is dead code, forcing a ~1 s partition recompute per forward. "No mock-only coverage" is also false in this lane: several election tests assert on source strings.

---

## 11. Top 5 findings (worst first)

1. **Bandwidth measurement is impossible against the shipped bridge** (High) — prober/echo protocol mismatch proven live; `NodeResources.bandwidth_mbps` is permanently the 1000 Mbps placeholder. README "Measured during latency probing" is false.
2. **Master stickiness broken → split-brain** (High) — `election.py:130–134` re-elects while the master is alive; empirically both nodes report `is_master()=True` after a weight shift; README's "master only loses status when truly offline" is false, and composite weights shift under load by design.
3. **LACP multi-resource path unreachable from production** (High) — `tflops_proportional=True` default (`node.py:116`) with no production switch; `broadcast_authoritative_partition` dead code; the freshly-fixed matrix/ILP/election machinery runs only when tests or hand-edits flip attributes.
4. **ILP not production-selectable and not scalable** (High) — no constructor/CLI/config path for `partition_method="ilp"`; 4 nodes ≈ 40 s, 8 nodes > 300 s vs. README "≤32 nodes".
5. **"No mock-only coverage" is false in this lane** (Medium) — source-string inspection tests in `test_election_multiresource.py` and `test_partitioner_ilp.py`, plus a `hasattr(ElectionManager, '__init__')` tautology, pass even if the implementation were deleted.

---

## 12. Confidence

**7 / 10.**

Why not higher: the two decisive adversarial results (probe rejection, split-brain) were reproduced against the *shipped* components, and the reachability/scale findings rest on complete repo greps plus runtime timings, so the negative findings are solid. However, (a) I could not exercise a real multi-host deployment (the probe experiment ran against the real bridge over loopback, which is protocol-faithful but not network-faithful); (b) split-brain was demonstrated with two `ElectionManager`s on one `EventRouter` — the exact topology used by the shipped election tests — but the production effect depends on `_refresh_election_weight`'s 2 s cadence and weight arithmetic I inferred rather than measured end-to-end in a live cluster; (c) the "per-forward recompute" cost assumes the LACP path is enabled, which production never does, so its real-world blast radius is conditional. All positive findings (type flow, ILP validity, election wiring, bandwidth propagation plumbing) are directly verified by driving shipped code plus passing tests. Confidence is docked because several README-vs-code gaps (production switch absence, dead broadcast, placeholder-only bandwidth) are judgment calls about what "production reachable" means rather than hard failures, and lanes A/C/D may surface interactions this lane did not.

**Ground-truth tally:** 36/36 mandated tests pass (independently rerun); 2 independent adversarial experiments fail as predicted by code reading; 5 repo-wide greps corroborate reachability findings; 2 timing experiments corroborate the scale finding. External-checked facts: 12. Model judgment: 3 (severity calibration, "production reachable" interpretation, real-cluster split-brain blast radius).
