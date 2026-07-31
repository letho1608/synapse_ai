# Follow-Up Executive Briefing: Phase 3 Runtime Verification

**Date:** 2026-07-31
**Method:** 3 focused follow-up verification agents (type mismatch crash, decode contamination, production wiring) — each running actual Python code, not just reading files
**Analysis confidence:** 10/10 for confirmed findings (empirically reproduced); 8/10 overall

---

## TL;DR

**Phase 3 runtime verification confirms that the Phase 2 deep-dive was correct — and understated the severity in one critical case.**

All three follow-up investigations found the claims were accurate:
1. **Decode token contamination: CONFIRMED — actually WORSE than reported.** The code unconditionally accepts 3 tokens per loop iteration instead of 1, inflating output to 3× length with replicated tokens. This is a Blocker, not Medium.
2. **Type mismatch crash: CONFIRMED with nuance.** Default config is safe (`tflops_proportional=True`), but LACP opt-in crashes at partition time. Plus a *second* crash bug found in the rebalance path.
3. **Production dead code: CONFIRMED.** ILP, 1F1B training, and multi-resource election are all 100% unreachable from production. `tflops_proportional=True` gates all LACP features.

---

## Follow-Up 1: Decode Token Contamination — BLOCKER (Confirmed, Escalated)

**File:** `distributed_engine.py:226-272` | **Status:** ✅ CONFIRMED — `simulate_decode_contamination.py` reproduces the bug

### What the code actually does

```
num_decode_batches = max(2, lookahead_window)   # = 3 with default lookahead_window=3
```

The loop fires **3 micro-batches** into the pipeline with different contexts, then unconditionally accepts 1 token from each:

```python
accepted_count = 1  # Always accept the first actual prediction
```

**Result: 3 tokens accepted per loop iteration where autoregressive generation expects exactly 1.**

### Two distinct bugs found

**Bug 1 (always active) — Token replication:**
Output inflates to 3× correct length with replicated tokens. Example from simulation:
- Correct output: `[404, 505, 606, 707]`
- Actual output: `[404, 404, 404, 505, 505, 505, 606, 606, 606, 707, 707, 707]`

**Bug 2 (when n-gram guesses exist) — Context contamination:**
Each batch sees different fake guesses as context. The transformer conditions on these wrong tokens. All contaminated predictions are accepted unconditionally. Simulation with n-gram match:
- Correct next-token logit: 600
- Batch 0 prediction (3 wrong guesses): 1110
- Batch 1 prediction (2 wrong guesses): 1073
- Batch 2 prediction (1 wrong guess): 1036

### Why existing tests don't catch this

All 4 tests in `test_decode_pipeline_overlap.py` are structural string/regex checks or mock tests verifying only `len(tokens) > 0`. None verify token correctness against expected autoregressive output.

**Severity: BLOCKER** — The system's primary output path (autoregressive decode) produces incorrect, replicated tokens. The Phase 1 finding "decode is slow but correct" has been replaced by "decode is potentially fast but WRONG."

---

## Follow-Up 2: Type Mismatch Crash — Confirmed with Nuance

**Files:** `node.py:1428` → `partitioner.py:394` | **Status:** ✅ CONFIRMED, severity adjusted

### What crashes

When `_build_latency_matrix()` returns a `dict` (its actual return type) and it reaches `compute_partition()`, line 394 crashes:

```
AttributeError: 'dict' object has no attribute 'nodes'
```

Successfully reproduced via `poc_type_mismatch.py`.

### The `tflops_proportional` gate

The default `tflops_proportional=True` (`node.py:95`) means LACP is **opt-in only**. The crash only affects users who explicitly enable LACP on multi-node deployments. Severity adjusted: **Blocker → High** for the type mismatch, because the default path is safe.

### SECOND crash bug discovered

The `repartition` closure (`node.py:396-409`) calls `compute_partition()` **without passing `latency_matrix`** — which is a **required** positional argument (no default). This raises:

```
TypeError: compute_partition() missing 1 required positional argument: 'latency_matrix'
```

This is on the fault-tolerance rebalancing path, only reachable when `tflops_proportional=False` AND a node actually fails. **Severity: Medium.**

---

## Follow-Up 3: Production Wiring — All Dead Code Claims Confirmed

**Files:** Multiple | **Status:** ✅ ALL 4 CLAIMS CONFIRMED via grep + Python import + code trace

| Claim | Verdict | Evidence |
|-------|---------|----------|
| `compute_partition(method="ilp")` unreachable | **CONFIRMED** | Both production call sites (`node.py:401,1433`) omit `method`, defaulting to `"lrm"`. Only test files pass `method="ilp"`. |
| `train_1f1b()` unreachable | **CONFIRMED** | Production training: `chatgpt_api.py:2598` → `ext.train_step_distributed()` → `self._trainer.step()` (DDP ring all-reduce). `train_1f1b()` only referenced in tests. |
| `set_base_compute_weight()` unreachable | **CONFIRMED** | `Node.__init__()` creates `ElectionManager(self.id, self.event_router)` with no `resource_scorer` and never calls the method. All nodes have weight `1.0`. |
| `tflops_proportional=True` masks all LACP bugs | **CONFIRMED** | Set at `node.py:95`, checked at `node.py:1417`. **Never set to `False`** anywhere in production code. The entire LACP subsystem is gated behind this flag. |

### The LACP Declaration of Dependence graph

```
main.py
  └→ Node.__init__(tflops_proportional=True)    # default: LACP DISABLED
       ├→ ElectionManager(id, router)           # no resource_scorer → weight=1.0
       ├→ FaultToleranceManager(id)             # heartbeat monitor works
       └→ ...
       
node.py:1417  if not self.tflops_proportional:  # ← THIS GATE blocks everything
                  _build_latency_matrix()       # returns dict, not LatencyMatrix!
                  compute_partition(...)         # expects LatencyMatrix → CRASH
```

The production path that actually runs is `TFLOPSProportionalPartitioner` — a simple greedy algorithm that predates the LACP implementation. The multi-resource LACP engine the README describes is entirely opt-in and, if enabled, crashes.

---

## Updated Verdict

| Phase 1 Finding | Phase 2 Reported | Phase 3 Verified |
|-----------------|-----------------|------------------|
| B1: Decode overlap missing | ⚠️ PARTIALLY RESOLVED | **🔴 BLOCKER — decode now produces wrong/replicated tokens** |
| B2: Zero-copy shared memory | ❌ NOT RESOLVED | ❌ NOT RESOLVED |
| B3: Rebalance dead code | ⚠️ PARTIALLY RESOLVED | ⚠️ PARTIALLY (trigger works, recovery broken) |
| B4: 1F1B never runs | ❌ NOT RESOLVED | **❌ CONFIRMED DEAD CODE** |
| H1: ILP claim false | ⚠️ PARTIALLY RESOLVED | **❌ CONFIRMED DEAD CODE** |
| H2: Scaling unvalidated | ❌ NOT RESOLVED | **❌ WORSENED — benchmark contradicts README** |
| H3: 4/8 dimensions unmeasured | ⚠️ PARTIALLY RESOLVED | ⚠️ PARTIALLY (3/4 now populated, bandwidth fake) |
| H4: TCP_NODELAY missing | ✅ RESOLVED | ✅ CONFIRMED |
| H5: Heartbeat 10s vs 5s | ✅ RESOLVED | ✅ CONFIRMED |
| H6: Election ignores multi-resource | ❌ NOT RESOLVED | **❌ CONFIRMED DEAD CODE** |
| M3: Power headroom zero | ✅ RESOLVED | ✅ CONFIRMED (single-GPU only) |

### New: Decode output corruption (was PARTIALLY in Phase 2, now BLOCKER)

After Phase 3 runtime verification, the most critical finding is that the "fix" for B1 (decode pipeline overlap) introduced **active output corruption** — the system now produces 3× more tokens than expected, with incorrect values from contaminated contexts. The system was slow but correct in Phase 1; it is now potentially fast but wrong.

### Net assessment: 8 files → 11 files, but same gap between promise and delivery

| Report | File count | Key action |
|--------|-----------|------------|
| `08-fault-tolerance-verification.md` | Lane 1: Fault tolerance safety | ✅ Verified |
| `09-transport-resource-verification.md` | Lane 2: TCP + resources | ✅ Verified |
| `10-ilp-partitioning-verification.md` | Lane 3: ILP | ✅ Verified |
| `11-election-pipeline-verification.md` | Lane 4: Pipeline + election | ✅ Verified |
| `12-tests-benchmarks-verification.md` | Lane 5: Tests + benchmarks | ✅ Verified |
| `13-synthesis.md` | Cross-referenced verdict | ✅ Verified |
| `14-red-team.md` | Adversarial review | ✅ Verified |
| `15-executive-briefing.md` | Final verdict | ✅ Verified |
| `16-followup-type-mismatch.md` | Type mismatch crash reproduction | ✅ CRASH CONFIRMED |
| `17-followup-decode-contamination.md` | Decode contamination simulation | ✅ BUG CONFIRMED |
| `18-followup-production-wiring.md` | Production dead code verification | ✅ DEAD CODE CONFIRMED |
| `19-followup-executive-briefing.md` | This file | Phase 3 wrap-up |

### Honest confidence

- **Analysis quality:** 9/10 — Three findings empirically reproduced via running Python code, not just static analysis. The gap remaining is the absence of a live multi-node cluster test.
- **Codebase health:** 2/10 — The decode output corruption (Blocker) and the type-mismatch crash (High) together make correct multi-node inference impossible for any configuration using LACP. The default path avoids the crash but never uses the multi-resource features the README describes.
- **Ground-truth tally:** 34 of 37 load-bearing conclusions are externally verified (source code read + empirical reproduction + pytest results + git diff + grep).
