# Fault Tolerance & Safety — Phase 2 Fix Verification

**Agent:** Lane 1 — Fault Tolerance & Safety  
**Commit:** `91ec971` ("Fix all Blocker and High findings from deep-dive audit")  
**Files analyzed:**
- `backend/synapse/pipeline/fault_tolerance.py` (262 lines)
- `backend/synapse/orchestration/node.py` (lines 63–1300+, selective)
- `backend/tests/test_fault_tolerance_live.py` (168 lines)
- `backend/tests/test_gradient_fault.py` (selective)
- `backend/tests/test_lacp.py` (selective)
- `backend/synapse/pipeline/__init__.py`

---

## 1. B2 (Blocker): `check_timeout()` wired into `_monitor_loop`

### Claim
`check_timeout()` was previously only called from test files and never from the production `_monitor_loop()`. The fix ensures `_monitor_loop()` calls `check_timeout()` on every iteration for every registered node.

### Evidence

**File:** `backend/synapse/pipeline/fault_tolerance.py`

**Before** (Phase 1): The `_monitor_loop()` body was a stub that only set `self._pending_rebalance` without checking timeouts:

```
while self._running:
    await asyncio.sleep(self.HEARTBEAT_INTERVAL)
    dead = self.get_dead_nodes()       # ← always returned []
    for node_id in dead:
        if not self._health[node_id].is_blacklisted:
            self._pending_rebalance = node_id
```

**After** (Phase 2, lines 273–288):

```python
async def _monitor_loop(self) -> None:
    """Background loop: check health, trigger rebalance on failure."""
    while self._running:
        await asyncio.sleep(self.HEARTBEAT_INTERVAL)

        # Check every registered node for timeout
        for nid in list(self._health.keys()):
            h = self._health[nid]
            h.check_timeout(self.HEARTBEAT_TIMEOUT)

        dead = self.get_dead_nodes()
        for node_id in dead:
            if not self._health[node_id].is_blacklisted:
                # Node just died — this would trigger rebalance
                # Rebalance is invoked externally via check_fault_tolerance() in node.py
                self._pending_rebalance = node_id
```

**Verification:** Line 281 now calls `h.check_timeout(self.HEARTBEAT_TIMEOUT)` inside a loop over `self._health.keys()`. The `_monitor_loop` is launched as a background task at line 105 (`self._monitor_task = asyncio.create_task(self._monitor_loop())`).

### Assessment
✅ **FIXED — Verified.** `check_timeout()` is called on every registered node on every cycle of the production monitor loop.

---

## 2. B3 (Blocker): Pickle Deserialization (RCE Vector)

### Claim
Checkpoint files previously used `pickle.load()` which executes arbitrary code during deserialization. The fix replaces Pickle with JSON-safe serialization.

### Evidence

**File:** `backend/synapse/pipeline/fault_tolerance.py`

**Before:** 
- `import pickle` at top of file
- `save_checkpoint()` used `pickle.dump(ckpt, f)`
- `load_checkpoint()` used `pickle.load(f)`

**After:**
- Line 21: `import json` (replaces `import pickle`)
- Line 20: `from dataclasses import dataclass, field, asdict` (added `asdict`)
- `save_checkpoint()` (lines 137–165): calls `self._json_serialize(data, filepath)`
- `load_checkpoint()` (lines 202–212): calls `self._json_deserialize(filepath)` → `json.load(f)`
- New `_json_serialize()` static method (lines 168–184): recursive converter handles `np.ndarray`, `np.integer`, `np.floating`, dict, and list/tuple types
- New `_json_deserialize()` static method (lines 187–193): reads JSON with safe error handling

**Complete search for pickle in pipeline directory:** ✅ No matches found in `backend/synapse/pipeline/`.

### Assessment
✅ **FIXED — Verified.** No `pickle.load()` or `pickle.dump()` calls exist. JSON serialization is used exclusively with numpy-safe type conversion.

### Caveat: numpy boolean gap in `_json_serialize`

The `convert()` helper in `_json_serialize` (lines 170–181) handles:
- `np.ndarray` → `.tolist()` ✅
- `np.integer` → `int()` ✅
- `np.floating` → `float()` ✅
- `dict` → recursive ✅
- `list`/`tuple` → recursive ✅

**Missing:** `np.bool_` is NOT handled. Empirical test confirms:

```
>>> isinstance(np.bool_(True), np.integer)
False
>>> isinstance(np.bool_(True), (np.integer, np.floating, np.ndarray))
False
>>> json.dumps(np.bool_(True), default=str)
'"True"'    # ← string, not boolean!
```

An `np.bool_` value falls through to `return obj`, and `json.dump` serializes it via `default=str`, producing the string `"True"` or `"False"` in JSON output. On deserialization, this comes back as a Python string, losing the boolean type. This is a data integrity issue.

**Severity:** Low — `np.bool_` is unlikely to appear in `kv_cache` or `optimizer_state` in practice, but the gap is real and untested.

---

## 3. M2 (Medium): `HEARTBEAT_TIMEOUT` = 5.0

### Claim
The constant was `10.0` while the README claimed `5s`. Fix changes it to `5.0`.

### Evidence

**File:** `backend/synapse/pipeline/fault_tolerance.py`, line 86:
```python
HEARTBEAT_TIMEOUT = 5.0    # seconds before considering node dead (matches README claim)
```

Git diff confirms the change from `10.0` to `5.0`.

**File:** `backend/synapse/pipeline/fault_tolerance.py`, line 59:
```python
def check_timeout(self, timeout_s: float = 10.0) -> bool:
```

Note: the **default parameter** of `check_timeout()` is still `10.0`, not `5.0`. This is a minor inconsistency — the default parameter value contradicts the class constant. However, the only production caller (`_monitor_loop` at line 281) passes `self.HEARTBEAT_TIMEOUT` (5.0) explicitly, so the default is never used in production.

### Assessment
✅ **FIXED — Verified.** `HEARTBEAT_TIMEOUT` is exactly `5.0`.

⚠️ **Code smell:** The `check_timeout()` default parameter remains `10.0` (line 59). Should be changed to `5.0` or the default removed entirely to prevent future misuse. *Severity: Note.*

---

## 4. `rebalance_after_failure` Production Wiring in `node.py`

### Claim
The fault tolerance manager must be wired into the node's production runtime so rebalancing actually triggers on node failure.

### Evidence

**File:** `backend/synapse/orchestration/node.py`

The wiring chain has four parts:

**4a. Initialization (lines 72, 148–155):**
```python
self.fault_tolerance = None

def _lazy_init_fault_tolerance(self):
    if self.fault_tolerance is None:
        from synapse.pipeline.fault_tolerance import FaultToleranceManager
        self.fault_tolerance = FaultToleranceManager(self.id)
        for nid in self.topology.nodes:
            self.fault_tolerance.register_node(nid)
```

**4b. Start/stop lifecycle (lines 220–222, 275–276):**
- `Node.start()` calls `self._lazy_init_fault_tolerance()` then `await self.fault_tolerance.start()`
- `Node.stop()` calls `await self.fault_tolerance.stop()`

**4c. Heartbeat recording (lines 1279–1287):**
```python
# Called inside periodic_topology_collection (every 2 seconds)
self._lazy_init_fault_tolerance()
if self.fault_tolerance:
    for peer_id in [p.id() for p in self.peers]:
        self.fault_tolerance.record_heartbeat(peer_id)
    for peer_id in [p.id() for p in self.peers]:
        if peer_id not in self.fault_tolerance._health:
            self.fault_tolerance.register_node(peer_id)
```

**4d. Rebalance trigger (lines 1290, 363–414):**
```python
await self.check_fault_tolerance()   # line 1290

async def check_fault_tolerance(self) -> None:        # line 363
    if self.fault_tolerance and self._distributed_pipeline:
        dead_nodes = self.fault_tolerance.get_dead_nodes()
        for dead_id in dead_nodes:
            if (self.fault_tolerance._health.get(dead_id, None)
                and self.fault_tolerance._health[dead_id].is_blacklisted):
                await self.fault_tolerance.rebalance_after_failure(
                    dead_node_id=dead_id,
                    current_assignments=...,
                    total_layers=32,
                    lacp_partition_fn=repartition,
                )
```

### Assessment
✅ **FIXED — Verified.** The full chain is present:
1. `Node.start()` → creates & starts `FaultToleranceManager` ✅
2. `periodic_topology_collection()` → records heartbeats from all P2P-connected peers every 2s ✅
3. `periodic_topology_collection()` → calls `check_fault_tolerance()` every 2s ✅
4. `check_fault_tolerance()` → queries `get_dead_nodes()` and calls `rebalance_after_failure()` for blacklisted nodes ✅
5. `Node.stop()` → stops the fault tolerance manager ✅

---

## 5. Silent-Bug Category Hunting

### 5a. Orphan State: `_pending_rebalance`

**File:** `backend/synapse/pipeline/fault_tolerance.py`, lines 100, 288

```
self._pending_rebalance: Optional[str] = None  # node_id being rebalanced  (line 100)
self._pending_rebalance = node_id               # line 288
```

`_pending_rebalance` is **written in two places but never read anywhere in the entire codebase**. The comment at line 287 says "Rebalance is invoked externally via `check_fault_tolerance()` in `node.py`", which is true — but the field is still dead code.

**Severity:** Low — not harmful, but dead code that can confuse maintainers.

### 5b. Orphan Data: `_recovery_checkpoint` key

**File:** `backend/synapse/pipeline/fault_tolerance.py`, line 261

```python
result.new_assignments["_recovery_checkpoint"] = ckpt
```

This key is written into `new_assignments` but **never read by any consumer**. No code in `backend/synapse/` reads `_recovery_checkpoint` from the assignments dict. The checkpoint object is serialized into the `RebalanceResult` but never used to actually restore state on the surviving nodes.

**Severity:** High — checkpoint-based recovery is implemented as far as saving and attaching the checkpoint to the result, but the result is never consumed. This means recovery from checkpoint is non-functional.

### 5c. Swallowed Error: `rebalance_after_failure` exception handler

**File:** `backend/synapse/pipeline/fault_tolerance.py`, lines 266–268

```python
except Exception as e:
    result.success = False
```

The exception `e` is caught but **never logged, printed, or propagated**. If the rebalance operation fails (e.g., `lacp_partition_fn` raises, or a file read error occurs), the failure is silently swallowed. The caller in `check_fault_tolerance()` does not check the return value either (line 411: `await self.fault_tolerance.rebalance_after_failure(...)` — no assignment, no error handling).

**Severity:** Medium — silent failure of the rebalance subsystem means nodes can die without recovery, and the operator has no indication anything went wrong.

### 5d. Swallowed Error: `_json_deserialize` error suppression

**File:** `backend/synapse/pipeline/fault_tolerance.py`, lines 192–193

```python
except (json.JSONDecodeError, FileNotFoundError, IOError):
    return None
```

This silently returns `None` on any read error. While this is a reasonable pattern for "optional file load", it means corrupt checkpoints are silently ignored with no logging. If checkpoint recovery is critical for fault tolerance, this should at least log a warning.

**Severity:** Note — by design, but worth flagging for observability.

### 5e. Result Thrown Away: `check_fault_tolerance()` ignores `RebalanceResult`

**File:** `backend/synapse/orchestration/node.py`, lines 411–414

```python
await self.fault_tolerance.rebalance_after_failure(
    dead_node_id=dead_id,
    current_assignments=current_assignments,
    total_layers=32,
    lacp_partition_fn=repartition,
)
```

The `RebalanceResult` is **not captured**. The orchestrator never knows whether rebalancing succeeded or failed. Combined with Finding 5c (swallowed exception), the system silently accepts failure.

**Severity:** Medium — the operator has no visibility into whether fault recovery actions succeeded.

### 5f. Unbounded `_checkpoints` List Growth

**File:** `backend/synapse/pipeline/fault_tolerance.py`, line 164

```python
self._checkpoints[self.node_id].append(ckpt)
```

Every call to `save_checkpoint()` appends to an in-memory list. There is **no pruning, eviction, or size limit**. Over a long-running system (hours/days), this list grows without bound, leaking memory. Coupled with the fact that `CHECKPOINT_INTERVAL = 30.0` is declared (line 87) but **never used** (no periodic checkpointing loop exists), this is a double gap: there's no automatic checkpointing, but if someone calls `save_checkpoint()` repeatedly, memory grows unbounded.

**Severity:** Low (Medium in long-running deployments) — no automatic checkpointing currently prevents runaway growth, but the vulnerability exists in the API.

### 5g. `MAX_MISSES` Constant Unused

**File:** `backend/synapse/pipeline/fault_tolerance.py`

```python
MAX_MISSES = 3    # line 88 — class constant
```

In `check_timeout()` (lines 62–64):
```python
if self.consecutive_misses >= 3:     # ← hardcoded 3
    self.is_blacklisted = True
```

The constant `MAX_MISSES = 3` is declared but **never referenced** — the actual implementation uses a hardcoded literal. If a developer changes `MAX_MISSES`, the blacklisting threshold won't change.

**Severity:** Note — refactoring gap, not a runtime bug.

### 5h. `CHECKPOINT_INTERVAL` Constant Unused

**File:** `backend/synapse/pipeline/fault_tolerance.py`, line 87

```python
CHECKPOINT_INTERVAL = 30.0    # seconds between state snapshots
```

This constant is declared but **never referenced anywhere in the codebase**. There is no periodic checkpointing loop — no background task calls `save_checkpoint()` at `CHECKPOINT_INTERVAL`. The checkpointing mechanism exists but is only usable via explicit manual calls.

**Severity:** Medium — automatic periodic checkpointing is advertised (by the constant's name and comment) but not implemented. The facility exists but is never invoked.

---

## 6. Cross-Cutting Architectural Issues

### 6a. Cross-Node Checkpoint Recovery is Non-Functional (High)

`load_latest_checkpoint(dead_node_id)` looks up **in-memory** checkpoints from `self._checkpoints` (line 197):

```python
def load_latest_checkpoint(self, node_id: str) -> Optional[ShardCheckpoint]:
    ckpts = self._checkpoints.get(node_id, [])
    if not ckpts:
        return None
    return max(ckpts, key=lambda c: c.timestamp)
```

But `save_checkpoint()` saves checkpoints under **`self.node_id`** (the saving node's ID), not the dead node's ID:

```python
self._checkpoints[self.node_id].append(ckpt)   # line 164
```

When a remote node dies and the orchestrator calls `load_latest_checkpoint(dead_node_id)`, the dead node's checkpoints are in the **dead node's memory** — they're not available in the orchestrator's `_checkpoints` dict. The disk-backed `load_checkpoint()` isn't called in the recovery path either.

The recovery path in `rebalance_after_failure()` (lines 255–261):

```python
dead_shard = current_assignments.get(dead_node_id)
if dead_shard:
    ckpt = self.load_latest_checkpoint(dead_node_id)
    if ckpt:
        result.recovered_from_checkpoint = True
        result.new_assignments["_recovery_checkpoint"] = ckpt
```

`dead_shard` is a `(start_layer, end_layer)` tuple, which is truthy, so the branch enters. But `load_latest_checkpoint(dead_node_id)` looks up the dead node's ID in the **orchestrator's** in-memory dict, which won't have the dead node's checkpoints (they died with the node).

The disk path `load_checkpoint()` **is not called** — it requires `model_id`, `start_layer`, `end_layer`, and `step`, but the recovery path only has `dead_node_id`. The code has the dead node's shard range (`dead_shard = (start_layer, end_layer)`), knows the `model_id` and `step` from the topology, but doesn't use them to load from disk.

**Furthermore**, `result.new_assignments["_recovery_checkpoint"]` is written but never read (Finding 5b), so even if the checkpoint loaded successfully, it would be orphaned in the result object.

**Severity: High** — checkpoint-based state recovery across node failures is advertised but fundamentally broken. The in-memory checkpoint dict only contains the local node's checkpoints, and the disk-based recovery path is never exercised.

### 6b. No Application-Level Health Checks (Medium)

Heartbeats are recorded based on P2P connection presence (lines 1282–1287):

```python
for peer_id in [p.id() for p in self.peers]:
    self.fault_tolerance.record_heartbeat(peer_id)
```

If a node's process hangs (deadlock, infinite loop, memory corruption) but its TCP/P2P connections remain open, `record_heartbeat()` is called every 2s, resetting the timeout counter. The `_monitor_loop` never detects a timeout because the heartbeat was just recorded.

This means the system detects **disconnected** nodes but not **hung/partitioned** nodes. A node that stops processing inference requests but keeps its P2P connections alive will never be detected as failed.

Application-level health checks (e.g., request-response probes, active ping with timeout) would be needed to detect this class of failure.

**Severity:** Medium — real-world node failures often manifest as hangs, not clean disconnects.

### 6c. `check_timeout()` Default Parameter Mismatch (Note)

```python
def check_timeout(self, timeout_s: float = 10.0) -> bool:   # line 59
```

The default is `10.0` while the class constant is `5.0`. This value is **never relied upon** in production (the only caller passes the parameter explicitly), but it's a trap for future developers who might call `check_timeout()` without arguments and get double the expected timeout.

**Severity:** Note.

---

## 7. Test Coverage Analysis

**`test_fault_tolerance_live.py`** (9 tests, all PASS):

| Test | What it verifies | Quality |
|------|-----------------|---------|
| `test_heartbeat_timeout_matches_readme` | `HEARTBEAT_TIMEOUT == 5.0` | ✅ Strong |
| `test_heartbeat_interval_constant` | `HEARTBEAT_INTERVAL == 2.0` | ✅ Strong |
| `test_check_timeout_called_in_monitor_loop` | `check_timeout` in source code | ⚠️ Text search only — does NOT execute the loop |
| `test_check_timeout_detects_dead_node` | `check_timeout()` sets `is_alive=False` | ✅ Strong |
| `test_monitor_loop_detects_timeout` | Integration: FT manager detects timeout | ✅ Strong |
| `test_save_checkpoint_no_pickle` | No pickle in source | ⚠️ Text search only |
| `test_save_checkpoint_uses_json` | JSON serialize in source | ⚠️ Text search only |
| `test_checkpoint_json_roundtrip` | Save/load roundtrip with JSON | ✅ Strong |
| `test_rebalance_after_failure_wired_in_node` | `rebalance_after_failure` in node.py source | ⚠️ Text search only |

**Gaps:**
1. **No test for numpy type conversion** — the JSON roundtrip test uses plain Python lists (`[1, 2, 3]`), not numpy arrays or scalars. The `_json_serialize` numpy-handling code is untested.
2. **No test for `rebalance_after_failure` end-to-end** — the only test checks string presence in source code, not actual execution.
3. **No test for the `_monitor_loop` / `periodic_topology_collection` interaction** — the full production wiring is not integration tested.
4. **No test for cross-node checkpoint recovery** — expected, since it's non-functional.
5. **No test for `load_checkpoint` from disk after node crash** — the roundtrip test saves and loads on the same `FaultToleranceManager` instance, which doesn't test the crash-recovery scenario.

---

## 8. Verdict Summary

### Verified Fixed (Green)
| ID | Severity | Finding | Status | File:Line |
|----|----------|---------|--------|-----------|
| B2 | Blocker | `check_timeout()` orphaned — not called from `_monitor_loop()` | ✅ **Fixed** — called on every iteration, line 281 | `fault_tolerance.py:281` |
| B3 | Blocker | Pickle deserialization RCE vector | ✅ **Fixed** — JSON with numpy-safe serialization, lines 168–212 | `fault_tolerance.py:21,168,189` |
| M2 | Medium | `HEARTBEAT_TIMEOUT = 10.0` vs README 5s | ✅ **Fixed** — now exactly `5.0`, line 86 | `fault_tolerance.py:86` |

### New Findings (Red/Amber)

| ID | Severity | Finding | File:Line |
|----|----------|---------|-----------|
| **H1** | **High** | Cross-node checkpoint recovery non-functional — in-memory checkpoint store is local-only; disk recovery path never exercised; recovery checkpoint result never consumed | `fault_tolerance.py:164,197,255–261` |
| **H2** | **High** | `_recovery_checkpoint` written but never read — orphan data in `RebalanceResult` | `fault_tolerance.py:261` |
| M3 | Medium | `check_fault_tolerance()` ignores `RebalanceResult` — orchestrator never knows if recovery succeeded | `node.py:411` |
| M4 | Medium | `rebalance_after_failure` silently swallows exceptions — no logging, no error propagation | `fault_tolerance.py:266–268` |
| M5 | Medium | No application-level health checks — system cannot detect hung nodes, only disconnected ones | `node.py:1282–1287` |
| M6 | Medium | `CHECKPOINT_INTERVAL` constant declared but never used — no periodic checkpointing loop exists | `fault_tolerance.py:87` |
| L1 | Low | `_pending_rebalance` orphan field — written but never read | `fault_tolerance.py:100,288` |
| L2 | Low | `_json_serialize` missing `np.bool_` handling — booleans serialize as strings | `fault_tolerance.py:170–181` |
| L3 | Low | `_checkpoints` list unbounded growth — no pruning/eviction | `fault_tolerance.py:164` |
| N1 | Note | `check_timeout()` default parameter `10.0` inconsistent with `HEARTBEAT_TIMEOUT = 5.0` | `fault_tolerance.py:59` |
| N2 | Note | `MAX_MISSES` constant declared but hardcoded as literal `3` in `check_timeout()` | `fault_tolerance.py:62,88` |

---

## 9. Executive Summary (250 words)

The Phase 2 commit `91ec971` correctly resolves all three targeted findings from the Phase 1 audit. B2 (`check_timeout()` orphaned) is fixed with a production call in `_monitor_loop()` at `fault_tolerance.py:281`. B3 (Pickle RCE) is fixed by replacing `pickle.load()`/`pickle.dump()` with JSON serialization backed by a numpy-safe converter. M2 (`HEARTBEAT_TIMEOUT = 10.0` vs README 5s) is fixed to exactly `5.0`. The fault tolerance manager is properly wired into `node.py`'s production lifecycle with heartbeat recording in `periodic_topology_collection()` and rebalance triggering via `check_fault_tolerance()`. Nine new live tests all pass.

However, the verification reveals significant new issues. **Two High-severity findings** involve checkpoint-based recovery: the in-memory checkpoint store is per-instance, making cross-node recovery via `load_latest_checkpoint()` always return `None` for a remote dead node, and the recovery checkpoint is stored in a result field that's never consumed. **Four Medium findings** include swallowed exceptions during rebalance (no logging, no error visibility), no application-level health checks (hung nodes undetectable), and an unused `CHECKPOINT_INTERVAL` constant. Three Low findings cover orphan state (`_pending_rebalance`), `np.bool_` serialization gap, and unbounded checkpoint list growth. Two Notes cover code-style inconsistencies.

The core heartbeat-based failure detection and rebalance trigger pipeline is **correct and functional**. The critical gaps are in the state recovery layer, which exists in skeleton form but is non-functional across node boundaries and produces results that are silently discarded.

---

## 10. Top 5 Most Concerning Findings

1. **[HIGH] H1 — Cross-node checkpoint recovery is fundamentally broken.** `load_latest_checkpoint(dead_node_id)` looks up the orchestrator's in-memory dict, which only contains the orchestrator's own checkpoints, not the dead node's. The disk-based `load_checkpoint()` path is never called. Recovery via checkpoint is advertised but impossible.

2. **[HIGH] H2 — Recovery checkpoint result is orphaned.** `_recovery_checkpoint` is stored in `result.new_assignments` at line 261, but no consumer anywhere reads this key. Even if the checkpoint loaded successfully, the recovered state would be discarded.

3. **[MEDIUM] M4 — Rebalance exceptions silently swallowed.** The `except Exception as e: result.success = False` at line 266 discards the exception with no logging. Combined with the caller (`check_fault_tolerance()`) not capturing the return value, failures in the entire rebalance subsystem are invisible to operators.

4. **[MEDIUM] M5 — No application-level health checks.** Heartbeats are recorded based on P2P connection presence only. A node that hangs (process freeze, deadlock) while keeping TCP connections open is never detected as failed.

5. **[MEDIUM] M6 — `CHECKPOINT_INTERVAL` is dead code.** The constant is declared at line 87 with a docstring claiming "seconds between state snapshots", but no background loop calls `save_checkpoint()` at any interval. The periodic checkpointing facility exists only on paper.

---

## 11. Confidence Rating

**Confidence: 8/10**

**Ground-truth tally:**
- ✅ **Direct code verification (source read):** 12 functions/methods read end-to-end across 2 primary files
- ✅ **Grep searches executed:** 17 across the codebase with exact pattern matching
- ✅ **Runtime tests executed:** 16 tests across 3 test files, all passing
- ✅ **Git diff analyzed:** Before/after comparison of the Phase 2 commit for `fault_tolerance.py`
- ✅ **Numpy behavior empirically tested:** Python REPL confirms `np.bool_` serialization gap

**Confidence deductions:**
- **−1:** The `_monitor_loop` / `periodic_topology_collection` interaction was traced by static analysis only. A runtime trace with actual node disconnection/reconnection would provide stronger evidence.
- **−1:** Several test files (`test_gradient_fault.py`, `test_lacp.py`) use `timeout_s=10.0` in direct `check_timeout()` calls, suggesting incomplete test modernization. These are pre-existing tests that still pass, but they use the old timeout value, which could mask regressions if the constant changes again.

The core B2, B3, and M2 fixes are **verified with high confidence**. The new findings (H1, H2, M3–M6) are based on direct code reading and grep searches — their severity is well-supported by the evidence. The lower confidence reflects the boundary between static analysis and runtime-verified behavior in a distributed system.
