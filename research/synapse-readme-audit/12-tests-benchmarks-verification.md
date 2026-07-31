# Lane 5: Tests & Benchmarks — Verification Report

**Agent**: Specialist — Tests & Benchmarks  
**Phase**: 2 verification (commit `91ec971` — "Fix all Blocker and High findings from deep-dive audit")  
**Files Examined**: 27 test/benchmark files + production source files  
**Date**: 2026-07-31

---

## Executive Summary

The Phase 2 commit makes a good-faith effort to bolt on tests for every finding in Phase 1, but the result is structurally hollow in several important ways. **The headline finding H5 (unvalidated scaling speedup claims) is NOT resolved**: the benchmark script (`benchmark_cluster_speedup.py`) exists and runs, but it computes speedup projections that directly contradict the README's claimed numbers (e.g., 1.46× for 10 nodes vs. the claimed 7.2×). The accompanying `test_benchmark_exists.py` tests only file-existence and trivial mathematical properties — it never validates that the benchmark produces numbers consistent with the README. Of the ~175+ test functions across 24 test files, roughly 30–35% are meaningful behavioral tests; the rest are source-code-inspection tests (does the source contain string X?), existence checks (can we import Y?), or trivial mathematical sanity checks. The strongest test files are `test_tcp_nodelay.py` (3 actual socket-option checks), `test_pipeline_1f1b.py` (14 meaningful schedule-verification tests), `test_lacp_multiresource.py` (~19 comprehensive scoring/allocation tests), and `test_async_forwarding.py` (~16 protocol roundtrip tests). The weakest are `test_benchmark_exists.py`, `test_1f1b_production.py`, and `test_decode_pipeline_overlap.py` — all dominated by source-code-inspection rather than behavioral verification. The README contains at least 3 material inaccuracies: test path listing (`tests/` vs `backend/tests/`), test count inflation (claims 12 tests in `test_fault_tolerance_live.py`, actual count is 9), and the speedup table (3–4× discrepancy vs. the benchmark script's own calculations). **Confidence: 7/10**, grounded in thorough file-by-file reading but unable to verify runtime pass/fail without executing the suite.

---

## 1. Benchmark Script: `benchmark_cluster_speedup.py`

**File**: `D:\Code\paper\synapse_ai\backend\tests\benchmark_cluster_speedup.py`  
**Status**: EXISTS — 105 lines, defines `compute_speedup()` and `benchmark_table()`.

### 1.1 What It Actually Computes

The script implements a simple pipeline timing model:

```
single_node_time = (num_nodes * stage_time_ms) * num_micro_batches
pipeline_time = 2*(num_nodes-1)*(stage_time_ms + network_latency + serialization)
               + num_micro_batches * max(stage_time_ms, network_latency + serialization)
speedup = single_node_time / pipeline_time
```

Default parameters: `stage_time_ms=2.0`, `network_latency_ms=0.5`, `serialization_ms=0.1`, `num_micro_batches=4`.

### 1.2 Critical Finding: Benchmark Numbers vs. README Claims

The benchmark's `compute_speedup()` produces speedups that **directly contradict** the README's scaling table:

| Config | Benchmark Speedup | README Claim |
|--------|------------------|--------------|
| 4 nodes (default params) | — | ~2.8× |
| 4 nodes (2.0/0.5/0.1/4) | ~1.37× | — |
| 10 nodes (default params) | ~1.46× | ~7.2× |
| 10 nodes (3.0/1.0/0.2/8) | ~1.75× | — |
| 100 nodes (2.0/2.0/0.5/16) | ~3.44× | ~55× |

**Severity: Blocker** — The benchmark was added specifically to address H5 ("unvalidated scaling speedup claims"), but it does not validate the README claims. A reader running the benchmark would get numbers 3–16× lower than what the README advertises. This is worse than having no benchmark: it creates the appearance of validation while actively undermining the project's claims.

### 1.3 The `speedup_ratio()` Disconnect

The production code (`PipelineScheduler.speedup_ratio()` in `scheduler.py`, line 218) implements a different formula:

```python
# scheduler.py speedup_ratio:
bottleneck_ms = max(per_stage_times_ms)
total_steps = num_stages + num_micro_batches - 1
single_time = num_micro_batches * single_stage_time_ms
pipeline_time = bottleneck_ms * total_steps
speedup = single_time / pipeline_time
```

The benchmark's `compute_speedup()` does **not** call `PipelineScheduler.speedup_ratio()`. It implements an independent formula with different semantics (adds explicit warmup/cooldown terms, adds network overhead inside the max()). The benchmark is disconnected from the actual codebase's speedup model.

**Severity: High** — The benchmark script is the *validation* artifact for H5, but it neither calls the production speedup function nor produces numbers that match the README's table.

### 1.4 Assumptions and Limitations

- All nodes have identical `stage_time_ms` (no heterogeneity).
- Pipeline overhead is symmetric (warmup == cooldown).
- No memory capacity constraints — assumes any node can hold any shard size.
- No gradient synchronization cost for training (inference-only model).
- The "100 node" config uses `num_micro_batches=16` which is unrealistically small for 100 stages (the pipeline bubble dominates at low batch counts).

---

## 2. `test_benchmark_exists.py`

**File**: `D:\Code\paper\synapse_ai\backend\tests\test_benchmark_exists.py`  
**Tests**: 5

### What It Tests

| Test # | Description | Type |
|--------|-------------|------|
| 1 | `benchmark_cluster_speedup.py` exists on disk | File existence |
| 2 | Module is importable | Import check |
| 3 | `compute_speedup(1) >= 1.0`, `compute_speedup(2) > 1.0`, `compute_speedup(4) > compute_speedup(2)` | Trivial math |
| 4 | Efficiency > 50% under favorable (10ms stage, 0.1ms latency) params | Boundary check |
| 5 | `benchmark_table()` is non-empty and contains "Speedup" and "100" | String existence |

### Assessment

**Severity: Medium** — All 5 tests are existence or trivial-mathematical checks. None of them:
- Compare benchmark output to README claims
- Verify the benchmark produces physically realistic speedups (e.g., speedup < num_nodes)
- Test that the benchmark runs as a script (`__main__` block is never exercised)
- Verify that `benchmark_cluster_speedup.compute_speedup()` is consistent with `PipelineScheduler.speedup_ratio()`
- Exercise error paths (division by zero, negative inputs, empty node lists)

The test name says "benchmark speedup projections" but it only checks that the script exists and computes basic math.

---

## 3. Test Quality Assessment — Deep Dives

### 3.1 `test_fault_tolerance_live.py` — 9 tests (README claims 12)

**Severity: Good (overall) / Note (count discrepancy)**

| Test | Assessment |
|------|-----------|
| `test_heartbeat_timeout_matches_readme` | Verifies constant=5.0 ✓ |
| `test_heartbeat_interval_constant` | Verifies constant=2.0 ✓ |
| `test_check_timeout_called_in_monitor_loop` | Source-code-inspection: checks `h.check_timeout` in `_monitor_loop` ✓ |
| `test_check_timeout_detects_dead_node` | Behavioral: creates NodeHealth, tampers heartbeat, asserts timeout ✓ |
| `test_monitor_loop_detects_timeout` | Integration: FaultToleranceManager with modified timeout ✓ |
| `test_save_checkpoint_no_pickle` | Source-code-inspection: checks pickle absent ✓ |
| `test_save_checkpoint_uses_json` | Source-code-inspection: checks json present ✓ |
| `test_checkpoint_json_roundtrip` | Behavioral: save/load JSON checkpoint in temp dir ✓ |
| `test_rebalance_after_failure_wired_in_node` | Source-code-inspection: checks node.py contains rebalance_after_failure ✓ |

**Strengths**: Mixes source-code-inspection (for specific vulnerability fixes) with real behavioral tests. The JSON roundtrip and NodeHealth timeout detection are meaningful.

**Weaknesses**: 
- The README says "12 tests" but only 9 exist. Either the count was rounded up or 3 tests were planned but not written.
- No test for concurrent heartbeat registration (race conditions).
- No test for blacklist behavior during rebalance.
- `test_rebalance_after_failure_wired_in_node` only checks string presence in source code — doesn't verify the function is actually called on failure events.

**Verdict**: Strongest of the Phase 2 test files, but the count inflation is a red flag.

---

### 3.2 `test_tcp_nodelay.py` — 3 tests

**Severity: Good**

| Test | Assessment |
|------|-----------|
| `test_tcp_server_sets_nodelay` | Behavioral: starts TCPServer, checks `getsockopt(TCP_NODELAY) == 1` on listening socket ✓ |
| `test_tcp_connection_handler_sets_nodelay` | Behavioral: connects to server, monkey-patches handler, checks accepted socket's TCP_NODELAY ✓ |
| `test_tcp_client_pool_sets_nodelay` | Behavioral: creates client pool, sends data, checks outbound socket's TCP_NODELAY ✓ |

All three tests directly verify socket options using `getsockopt`. These are genuine behavioral tests. The only minor concern is that they monkey-patch internal methods rather than testing through public APIs.

**Verdict**: Best 3 tests in the entire suite. Concise, meaningful, verifiable.

---

### 3.3 `test_partitioner_ilp.py` — 7 tests

**Severity: Medium**

| Test | Type | Notes |
|------|------|-------|
| `test_pulp_available` | Import check | Trivial |
| `test_pulp_imported_in_partitioner` | Source-code-inspection | Checks "import pulp" in partitioner.py |
| `test_compute_partition_ilp_function_exists` | Existence | Checks callable |
| `test_ilp_method_parameter` | API shape | Checks method param in signature |
| `test_ilp_produces_valid_partition` | **Behavioral** | 2 nodes, 24 layers, verifies coverage + ordering. **Skipped if no PuLP** |
| `test_ilp_full_coverage` | **Behavioral** | 3 nodes, 30 layers, verifies all layers. **Skipped if no PuLP** |
| `test_ilp_defaults_to_lrm_without_pulp` | Behavioral smoke | Runs both methods, checks both produce assignments |

The two behavioral tests (5, 6) that actually exercise the ILP solver are conditionally skipped based on PuLP availability. Since `pulp>=2.7.0` is listed in `requirements.txt`, this is reasonable — but it means the ILP-specific tests only run in environments where the full dependency tree is installed.

**Verdict**: Useful tests but the most important ones are conditional. No test validates ILP *optimality* (e.g., that the solution minimizes max latency-weighted load), only basic coverage.

---

### 3.4 `test_election_multiresource.py` — 6 tests

**Severity: Medium-Low**

| Test | Type | Notes |
|------|------|-------|
| `test_election_accepts_resource_scorer` | Trivial existence | Checks `hasattr(ElectionManager, '__init__')` — always true |
| `test_set_base_compute_weight_uses_scorer` | Source-code-inspection | Checks "self._resource_scorer" in election.py |
| `test_election_code_uses_multi_weight_over_simple` | Source-code-inspection | Checks "multi_weight" in election.py |
| `test_election_fallback_to_vram_flops` | **Behavioral** | Creates ElectionManager, calls set_base_compute_weight, asserts weight=16.2 ✓ |
| `test_election_code_imports_resource_scorer` | Source-code-inspection | Checks "self._resource_scorer is not None" |
| `test_election_with_scorer_gives_higher_weight` | **Behavioral** | Creates ElectionManager with high_scorer, asserts weight > 16.2 ✓ |

**Weaknesses**: Test 1 is genuinely useless (always passes). Tests 2, 3, 5 use source-code-inspection (fragile — change the variable name and they break). The two behavioral tests (4, 6) verify correct — but they use heavy mocking (MagicMock for EventRouter) and test only the weight computation, not the actual election loop behavior.

**Verdict**: The scoring behavior is tested, but the multi-resource election *integration* (does a node with higher multi-resource score actually get elected?) is not tested.

---

### 3.5 `test_1f1b_production.py` — 5 tests

**Severity: Low**

| Test | Type | Notes |
|------|------|-------|
| `test_generate_1f1b_schedule_called_from_production` | **grep test** | Searches source files for "generate_1f1b_schedule" string |
| `test_gradient_sync_imports_scheduler` | Source-code-inspection | Checks "PipelineScheduler" in gradient_sync.py |
| `test_distributed_trainer_has_train_1f1b_method` | Existence | Checks `hasattr(DistributedTrainer, "train_1f1b")` |
| `test_train_1f1b_constructs_schedule` | **Inspects source of train_1f1b** | Reads `inspect.getsource()` and checks for string literals |
| `test_pipeline_stage_accepts_backward_fn` | Behavioral | Creates PipelineStage with backward_fn, asserts it's stored ✓ |

**Severe Weakness**: 4 out of 5 tests use source-code-inspection or existence checks. The only behavioral test (test 5) verifies constructor parameter storage — not that backward_fn is actually called during 1F1B execution.

**`test_train_1f1b_constructs_schedule`** is particularly weak: it reads the *source code* of `train_1f1b` and checks for string literals. This will pass even if the method is completely broken, as long as those strings appear in the source. A trained monkey could pass this test.

**Fatal flaw**: None of these tests actually run `train_1f1b()` with real PipelineStages and verify it produces correct forward/backward ordering.

**Verdict**: This file claims to verify "1F1B schedule wired to production code" but actually verifies only that certain variable names appear in source files. The title is misleading.

---

### 3.6 `test_decode_pipeline_overlap.py` — 4 tests

**Severity: Low**

| Test | Type | Notes |
|------|------|-------|
| `test_decode_loop_uses_multiple_batches` | Source-code-inspection | Looks for "num_decode_batches" in distributed_engine.py |
| `test_decode_loop_fires_before_collect` | **Weak behavioral** | Counts occurrences of send/collect method names in source |
| `test_overlapping_batches_fired` | Mock-heavy integration | Creates controller, monkey-patches 3 methods, asserts tokens > 0 |
| `test_decode_loop_structure_has_overlap` | Source-code-inspection | Checks for two specific for-loop strings in source |

**Test 3 is deceptive**: it sets up an elaborate mock environment with counting-send, counting-collect, fast-collect overrides — but the assertion is only `assert len(tokens) > 0`. The monkey-patching actually replaces the overlap behavior being tested. The test verifies that the mock framework doesn't crash, not that actual pipeline overlap occurs.

None of these tests verify:
- That stage N+1 receives batch M while stage N processes batch M+1
- That the `lookahead_window` parameter actually increases throughput
- That N-gram lookahead guesses are validated correctly

**Verdict**: Of 4 tests, 3 are source-code text searches and 1 is a mock-heavy pass-through test that doesn't verify the overlap claim.

---

### 3.7 Strong Test Files (for completeness)

| File | Tests | Quality | Notes |
|------|-------|---------|-------|
| `test_pipeline_1f1b.py` | 14 | **High** | Genuine behavioral tests for schedule generation, throughput estimation, edge cases |
| `test_lacp_multiresource.py` | ~19 | **High** | Comprehensive normalize/scoring/allocation tests with edge cases |
| `test_async_forwarding.py` | ~16 | **High** | Real protocol roundtrip tests (encode/decode, compression, ordering) |
| `test_gradient_fault.py` | ~16 | **Medium-High** | Ring reduce, node health, checkpoint tests. Some are weak but many are strong |
| `test_distributed_pipeline.py` | ~16 | **Medium** | Mixed quality: some real behavioral, some mock-heavy |
| `test_resource_measurement.py` | 6 | **Medium** | 3 existence checks, 3 behavioral. "All 8 dimensions" claim is verified by existence not accuracy |

---

## 4. README Accuracy Assessment

### 4.1 Test Path
**Claim**: `tests/` directory  
**Reality**: Tests live in `backend/tests/`

**Severity: Low** — The path `tests/` vs `backend/tests/` is a minor inconsistency. The README's project structure diagram correctly shows the tests under `backend/tests/`.

### 4.2 Test Count Discrepancy
**Claim**: `test_fault_tolerance_live.py` has "12 tests"  
**Reality**: 9 test methods

Several possibilities:
- Tests were planned but never written
- The docstring was written optimistically
- README authors rounded up (9 → 12 is unusual rounding)

**Severity: Medium** — This inflates the apparent coverage by 33% for this file. Combined with the "45 dedicated tests" claim, it suggests the test count is loosely approximate.

### 4.3 Speedup Table
The README's scaling table claims:
- 4 nodes: ~2.8× faster
- 10 nodes: ~7.2× faster
- 100 nodes: ~55× faster

The benchmark script produces numbers that are 3–16× lower (see Section 1.2).

**Severity: Blocker** — The benchmark was the *intended fix* for H5 (unvalidated claims), but the benchmark and the claims are in direct conflict. Either the claims are wrong or the benchmark model is wrong — but the existence of the benchmark does nothing to resolve the inconsistency.

### 4.4 Test File Listing Accuracy
The README's project structure lists 11 test files plus the benchmark script. All listed files exist on disk. However, the README misses several older test files:
- `test_cluster.py`, `test_cluster_inprocess.py`, `test_async_forwarding.py` (16 meaningful tests)
- `test_system.py`, `test_dataset.py`, `test_models.py`
- `test_distributed_stress.py` (4 integration tests)
- `test_simultaneous.py`, `test_p2p_minimal.py`, `test_election_edge.py`
- `test_local.py`, `test_inprocess.py`, `test_three_nodes.py`, `test_3node_lacp.py`
- `test_lacp.py` (18 tests)

This is selective listing — only Phase 2 files are shown, hiding the extensive pre-existing test suite.

**Severity: Note** — Not inaccurate, but misleading in its selectivity.

---

## 5. Silent-Bug Category Analysis

### 5.1 Tests That Pass Against Almost No Data
- **`test_partitioner_ilp.py::test_ilp_produces_valid_partition`**: Only 2 nodes. Doesn't test the ILP's ability to handle large N or heterogeneous latency matrices.
- **`test_election_multiresource.py::test_election_with_scorer_gives_higher_weight`**: Only checks numeric score, no actual election.
- **`test_1f1b_production.py::test_train_1f1b_constructs_schedule`**: Reads source, doesn't run anything.

### 5.2 Tests That Test Existence, Not Behavior
- **`test_benchmark_exists.py`**: All 5 tests.
- **`test_fault_tolerance_live.py::test_check_timeout_called_in_monitor_loop`**: Checks string "check_timeout" appears in source.
- **`test_1f1b_production.py`**: 4 of 5 tests are source searches or existence checks.
- **`test_decode_pipeline_overlap.py`**: 3 of 4 tests are source searches.

### 5.3 Acceptance Criteria That Admit Trivially Passing Implementations
- **`test_decode_pipeline_overlap.py::test_overlapping_batches_fired`**: Only asserts `len(tokens) > 0`. A completely broken pipeline that returns a single token via mock would pass.
- **`test_benchmark_exists.py::test_compute_speedup_basics`**: Only checks `ratio4 > ratio2`. A function returning `[1.0, 1.01, 1.02]` would pass.
- **`test_benchmark_exists.py::test_compute_speedup_efficiency_bounds`**: Only checks `efficiency > 0.5`. Would pass even if the model is wildly wrong.

### 5.4 Happy-Path Only Testing
- **`test_partitioner_ilp.py`**: Tests valid partitions but not what happens with invalid inputs (empty node dict, negative layers, duplicate node IDs).
- **`test_tcp_nodelay.py`**: Tests TCP_NODELAY is set but never tests the error path (what if `setsockopt` fails?).
- **`test_fault_tolerance_live.py`**: Tests single node timeout but not concurrent node failures, network partitions, or recovery races.
- **No tests anywhere** inject simulated node crashes during active pipeline execution.

---

## 6. Overall Test Suite Quality

### 6.1 Strengths
1. **`test_tcp_nodelay.py`** (3 tests): Gold standard — each test directly checks a claimed behavior via OS-level socket API.
2. **`test_pipeline_1f1b.py`** (14 tests): Comprehensive schedule verification with real edge cases and throughput estimation.
3. **`test_lacp_multiresource.py`** (~19 tests): Thorough normalization, scoring, and allocation verification including utilization penalties and edge cases.
4. **`test_async_forwarding.py`** (~16 tests): Genuine protocol roundtrip tests with compression, edge cases, and ordering.
5. **`test_gradient_fault.py`** (~16 tests): Mix of strong behavioral and some weak, but overall solid.
6. **Overall breadth**: ~175+ test functions cover many components — more breadth than most hobby projects.

### 6.2 Weaknesses
1. **H5 NOT RESOLVED**: The benchmark script produces numbers that contradict the README. The existence check passes but the validation purpose is defeated.
2. **Source-code-inspection pattern**: ~25% of Phase 2 tests use `open(file).read()` and check for string literals. These tests break on refactoring and don't verify runtime behavior.
3. **Count inflation**: README claims 12 tests in `test_fault_tolerance_live.py`; only 9 exist.
4. **No error-injection testing**: Zero tests simulate network failures, node crashes during pipeline execution, or corrupted data.
5. **Fixture starvation**: Most tests use minimal fixtures (2 nodes, 4 micro-batches) that can't stress real pipeline behavior.
6. **No end-to-end suite**: No test strings together the full pipeline (partition → election → schedule → execute → route) on actual TCP connections.

---

## 7. Findings Register

| ID | Severity | Finding | File |
|----|----------|---------|------|
| T1 | **Blocker** | Benchmark produces speedups (1.46× for 10 nodes) that contradict README claims (7.2×). H5 resolution is a no-op. | `benchmark_cluster_speedup.py` + `README.md` |
| T2 | **Blocker** | Benchmark script does not call `PipelineScheduler.speedup_ratio()` — it's an independent model disconnected from production code. | `benchmark_cluster_speedup.py` |
| T3 | **High** | `test_benchmark_exists.py` only tests file existence and trivial math. Never validates benchmark against README. | `test_benchmark_exists.py` |
| T4 | **High** | README test count inflated: claims 12 tests in `test_fault_tolerance_live.py`, actual count is 9. | `README.md` |
| T5 | **High** | `test_1f1b_production.py` uses source-code-inspection for 4/5 tests. Does not actually run `train_1f1b()`. | `test_1f1b_production.py` |
| T6 | **Medium** | `test_decode_pipeline_overlap.py` uses source-code-inspection for 3/4 tests. The behavioral test is mock-heavy and only asserts `tokens > 0`. | `test_decode_pipeline_overlap.py` |
| T7 | **Medium** | `test_partitioner_ilp.py` skips the 2 most important tests if PuLP is absent (even though it's a declared dependency). | `test_partitioner_ilp.py` |
| T8 | **Medium** | `test_election_multiresource.py::test_election_accepts_resource_scorer` tests `hasattr(ElectionManager, '__init__')` — always passes. | `test_election_multiresource.py` |
| T9 | **Medium** | No error-injection or recovery tests in the entire suite. Node failures during pipeline execution are untested. | All test files |
| T10 | **Note** | README project structure omits ~12 pre-existing test files, showing only Phase 2 additions. | `README.md` |
| T11 | **Note** | Test path in README description says `tests/` but files are at `backend/tests/` (project structure diagram is correct). | `README.md` |
| T12 | **Note** | `bench_log.txt` and `token_output.txt` in the test directory appear to be stale logs (3 lines each from a single test run). | `backend/tests/` |

---

## 8. 250-Word Executive Summary

The Phase 2 commit (`91ec971`) added the benchmark script and test files that README claims resolve H5 (unvalidated scaling speedup claims). **They do not.** The benchmark computes speedup projections 3–16× lower than the README's advertised numbers (e.g., 1.46× for 10 nodes vs. the claimed 7.2×). The accompanying test checks only file existence and trivial math — it never validates the benchmark against the README table. This means H5 remains unresolved: there is still no artifact in the codebase that supports the scaling claims.

Of ~175+ test functions across 24 files, about 30–35% are meaningful behavioral tests. The strongest files (`test_tcp_nodelay.py`, `test_pipeline_1f1b.py`, `test_lacp_multiresource.py`, `test_async_forwarding.py`) contain legitimate socket-option checks, schedule-verification tests, and protocol roundtrip tests. The weakest files (`test_1f1b_production.py`, `test_decode_pipeline_overlap.py`, `test_benchmark_exists.py`) are dominated by source-code-inspection — they grep for string literals rather than exercising runtime behavior. The README inflates the test count by 33% for one file (claims 12, has 9) and contains path inaccuracies.

The test suite has breadth but shallow depth in critical areas. There are zero error-injection or recovery tests. No test strings together the full pipeline with real TCP transport. The benchmark and claims are in direct numerical conflict. The foundation of automated verification is present, but the marquee fix for H5 is hollow.

---

## 9. Top 5 Most Concerning Findings

1. **Blocker**: Benchmark contradicts README speedup claims (1.46× vs 7.2× for 10 nodes). H5 is NOT resolved — the fix creates an artifact that undermines the project rather than validating it.
2. **Blocker**: Benchmark script is disconnected from production code — `compute_speedup()` does not call `PipelineScheduler.speedup_ratio()` and uses a different formula.
3. **High**: `test_1f1b_production.py` — advertised as verifying "1F1B schedule wired to production code" — uses source-code-inspection for 4 of 5 tests and never actually runs `train_1f1b()`.
4. **High**: README test count inflation (claims 12, has 9) for the marquee test file, suggesting the documentation is loosely approximate.
5. **Medium**: Zero error-injection tests in the entire suite. Node crashes, network partitions, and data corruption during pipeline execution are completely untested.

---

## 10. Confidence Rating

### Rating: 7/10

**Ground-truth tally:**
- Files read: 27 (all test files + all benchmark scripts + README + production sources)
- Test functions examined: ~175+ across 24 files
- Source-code assertions verified: ~95 individual `assert` statements checked
- Production code paths traced: partitioner.py, scheduler.py, election.py, gradient_sync.py, fault_tolerance.py, tcp_transport.py, distributed_engine.py, resource_weight.py (8 files)
- README claims checked against reality: 6 specific claims verified (path, test count, speedup table, component list, framework claims, test listing)

**Positive factors (pulling up):**
- Exhaustive file-by-file reading rather than sampling
- Both test code and production source code were cross-referenced
- All README structural claims were verified against actual tree

**Negative factors (pulling down):**
- Unable to run the test suite to verify actual pass/fail rates
- Cannot measure flakiness or runtime performance
- Cannot verify the ILP solver produces mathematically optimal solutions without installing PuLP and running
- 3 test files (test_cluster.py, test_three_nodes.py, test_multimachine_distributed.py) are integration scripts that require multi-node setup — their quality was assessed from code only
- The `frontend` test exclusion claim (169 passing tests minus frontend) could not be independently verified without running the suite

**Bias note**: The confidence would drop to 5/10 if any of the skipped tests (PuLP-dependent ILP tests, integration-only multi-node tests) are found to fail at runtime. The analysis assumes these tests would pass based on code inspection. A runtime verification would increase confidence to 9/10.

---

*End of Lane 5 report. Prepared by Tests & Benchmarks specialist agent.*
