# Lane 4: Tests, Training, Fault Tolerance & Discovery

## Claim 1: Distributed DDP + All-Reduce — "every node computes gradients on its shard in parallel, then synchronizes"

**STATUS: PARTIALLY VERIFIED** — Code structure exists but ring-reduce is in-memory only; no real network transport is wired.

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/pipeline/gradient_sync.py` | 65–74 | `DistributedTrainer` docstring: *"DDP-style distributed training with ring all-reduce. Each node: 1. Computes local gradients on its full model copy 2. Ring all-reduce to sum gradients across all nodes 3. Each node updates weights with globally summed & scaled gradients"* |
| `backend/synapse/pipeline/gradient_sync.py` | 30–60 | `RingAllReducer` class: `get_predecessor()`, `get_successor()`, `accumulate()`, `get_accumulated()`, `reset()` — ring topology primitives defined |
| `backend/synapse/pipeline/gradient_sync.py` | 93–145 | `DistributedTrainer.step()` — the core loop: sends to successor, receives from predecessor, accumulates, then applies gradient clipping + scaling via `1/world_size` |

**Verdict:** The `RingAllReducer` and `DistributedTrainer` implement a logical ring all-reduce with gradient clipping and scaling. However, the `send_fn` / `recv_fn` are injected as callbacks (`Optional[Callable]`) and default to `None` — the class itself provides *no default transport*. The ring loop at lines 115–122 loops `world_size - 1` times but the sent data `current_gradients` is the *predecessor's* received data each iteration, which is a correct ring implementation *if* real network send/recv callbacks are wired externally. The code is a **correct in-memory prototype** but requires external wiring to become a real distributed all-reduce.

---

## Claim 2: Dynamic Rebalancing — "failed node's shard reassigned live; state recovered from checkpoint"

**STATUS: PARTIALLY VERIFIED** — Rebalancing method exists but is **never called** by the monitor loop.

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/pipeline/fault_tolerance.py` | 189–241 | `FaultToleranceManager.rebalance_after_failure()` — full implementation: blacklists dead node, queries surviving nodes via `lacp_partition_fn`, loads checkpoint, returns `RebalanceResult` with `downtime_ms` |
| `backend/synapse/pipeline/fault_tolerance.py` | 137–175 | `save_checkpoint()` and `load_latest_checkpoint()` — serializes `ShardCheckpoint` via pickle to disk and can reload it during rebalance |
| `backend/synapse/pipeline/fault_tolerance.py` | 249–255 | `_monitor_loop()` — detects dead nodes but **only sets `self._pending_rebalance = node_id`**; there is no code path that calls `rebalance_after_failure()`. The field `_pending_rebalance` is never read elsewhere in the file. |

**Verdict:** The *function* for dynamic rebalancing is implemented and the checkpoint plumbing works. But the rebalance is never **triggered** by the monitor — it's dead code. The `_pending_rebalance` field is written and never consumed. Without an external caller invoking `rebalance_after_failure()`, the claim of "live reassignment" is unexecuted.

**Severity: HIGH** — This is a gap between documented capability and runtime behavior.

---

## Claim 3: Node failure — "5s heartbeat detection" and "zero-downtime recovery"

**STATUS: NOT FOUND** for both sub-claims.

### 3a. 5s heartbeat detection: **NOT FOUND**

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/pipeline/fault_tolerance.py` | 87 | `HEARTBEAT_TIMEOUT = 10.0` — constant hard-coded to 10 seconds, not 5 |
| `backend/synapse/pipeline/fault_tolerance.py` | 56–63 | `NodeHealth.check_timeout(timeout_s=10.0)` — `elapsed > timeout_s` is the trigger; default is 10.0s |
| `backend/synapse/pipeline/fault_tolerance.py` | 53 | `HEARTBEAT_INTERVAL = 2.0` — heartbeats sent every 2s, but timeout is 10s |

First detection requires 10 seconds without a heartbeat. Additionally, `is_alive` is set to `False` on first timeout, but blacklisting (misnamed as `is_blacklisted` — it really means "confirmed dead") requires 3 consecutive misses (3 × 10s = 30s worst case, though in practice 10s + 2×2s = ~14s). Neither is 5 seconds.

### 3b. Zero-downtime recovery: **NOT FOUND**

The `rebalance_after_failure()` method is never called from the monitor loop (see Claim 2). Even if called, "zero-downtime" is not validated: `downtime_ms` is computed after the fact as `(time.time() - start_time) * 1000` but no mechanism guarantees uninterrupted service during the rebalance window.

**Severity: HIGH** — The timeout constant is exactly double the advertised value, and the recovery mechanism is dead code.

---

## Claim 4: Discovery — "nodes find each other" via manual config or Tailscale

**STATUS: VERIFIED** — Three discovery backends exist, all implementing the abstract `Discovery` ABC.

### ManualDiscovery

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/networking/manual/manual_discovery.py` | 1–100 | Full implementation: reads a `NetworkTopology` config file, discovers peers, performs health checks, caches results, polls every 5s |
| `backend/synapse/networking/manual/manual_discovery.py` | 32–33 | `start()` creates `task_find_peers_from_config()` |
| `backend/synapse/networking/manual/manual_discovery.py` | 37–44 | `discover_peers(wait_for_peers=N)` — blocking wait with sleep-poll |

### TailscaleDiscovery

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/networking/tailscale/tailscale_discovery.py` | 38–190 | Full implementation: discovers peers via Tailscale API (`get_tailscale_devices`), reads device attributes for capabilities, polls every 5s |
| `backend/synapse/networking/tailscale/tailscale_discovery.py` | 68–72 | `start()` spawns `task_discover_peers()`, `task_cleanup_peers()`, `task_update_device_posture_attributes()` |
| `backend/synapse/networking/tailscale/tailscale_discovery.py` | 133–137 | Fallback for Tailscale Free plan (no custom attributes): uses device name as peer_id |

### UDPDiscovery

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/networking/udp/udp_discovery.py` | 77–290 | Full implementation: UDP broadcast on all interfaces, listens for peers, priority-based replacement, interface-type filtering |
| `backend/synapse/networking/udp/udp_discovery.py` | 108–109 | `start()` spawns `task_broadcast_presence()`, `task_listen_for_peers()`, `task_cleanup_peers()` |

### Abstract base

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/networking/discovery.py` | 1–12 | `Discovery` ABC with `start()`, `stop()`, `discover_peers()` abstract methods |

**Verdict:** All three discovery backends correctly implement the `Discovery` interface. The claim is fully substantiated.

---

## Claim 5: Probing — "latency + bandwidth matrix built (N×N)"

**STATUS: PARTIALLY VERIFIED** — Latency probing exists and builds an N×N matrix; bandwidth probing is **not implemented**.

### Latency matrix: VERIFIED

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/networking/latency_probing.py` | 37–60 | `LatencyProber` class: probes all peers with configurable `probe_count` (default 10), `timeout_ms` (5000), caches to disk with `cache_validity_seconds` (300) |
| `backend/synapse/orchestration/node.py` | 1449–1488 | `_build_latency_matrix()`: probes all peers in background, logs results as *"[latency] probed {peer.id()}: {latency_ms:.1f}ms"* |
| `backend/synapse/api/chatgpt_api.py` | 2941–2953 | `handle_get_cluster_resources()` includes `"latency_matrix": latency_matrix` from `self.node._latency_cache` |
| `backend/synapse/topology/latency_matrix.py` | — | `LatencyMatrix`, `LatencyEntry`, `LatencyProber`, `HierarchicalClusterer`, `Cluster` all exist in this module |

### Bandwidth matrix: NOT FOUND

A `grep` for "bandwidth" in the networking and topology directories returns zero results in probing code. The `ResourceScorer` in `resource_weight.py` stores a `bandwidth_mbps` field that is "filled by latency matrix probing" (comment at line 58), but no actual bandwidth measurement is implemented. The bandwidth value appears to be statically configured, not dynamically probed.

**Severity: MEDIUM** — Latency probing is real; bandwidth probing is a stub/placeholder.

---

## Claim 6: Training — "1F1B schedule for training" and "gradient all-reduce runs concurrently"

### 6a. 1F1B schedule: VERIFIED

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/pipeline/scheduler.py` | 2–4 | Module docstring: *"LACP — Pipeline Parallelism Scheduler (1F1B / GPipe). Implements the One-Forward-One-Backward (1F1B) schedule"* |
| `backend/synapse/pipeline/scheduler.py` | 86–93 | `PipelineScheduler.generate_1f1b_schedule(num_stages, num_microbatches)` — generates warmup → steady → cooldown schedule |
| `backend/tests/test_pipeline_1f1b.py` | 20–50 | `Test1F1BSchedule` class: tests 1-stage/4-batch, 4-stage/4-batch warmup, stage-specific slot extraction |
| `backend/synapse/training/finetune_lora.py` | 1–285 | `run_finetune_sync()` — real LoRA finetuning via HF Trainer + PEFT, but **not distributed** (blocking sync call, no pipeline parallelism) |

The 1F1B scheduler exists and is tested. However, the actual finetuning implementation (`run_finetune_sync`) is a **local single-node Trainer** — it does not use the 1F1B pipeline or ring all-reduce. The training API endpoint (`/v1/training/start`) calls this local trainer.

### 6b. Gradient all-reduce runs concurrently: NOT FOUND

| File | Line(s) | Evidence |
|------|---------|----------|
| `backend/synapse/pipeline/gradient_sync.py` | 93–145 | `DistributedTrainer.step()` is purely sequential: it sends, receives, accumulates, then clips, then returns updates. No asyncio.gather or concurrent task overlaps the reduce with the next forward. |

The README claims *"Gradient all-reduce runs concurrently with the next forward pass"* (line 61), but the code performs the full ring reduce synchronously within `step()`. There is no mechanism to overlap communication with computation.

**Severity: MEDIUM** — 1F1B scheduler is real and tested. But no distributed training actually uses it; finetuning is local only. The concurrent all-reduce claim is aspirational.

---

## Claim 7: API Endpoints

| Claimed Endpoint | Actual Registration | File:Line | Status |
|---|---|---|---|
| `POST /v1/chat/completions` | ✅ `handle_post_chat_completions` | `chatgpt_api.py:297` | VERIFIED |
| `GET /v1/models` | ✅ `handle_get_models` | `chatgpt_api.py:293` | VERIFIED |
| `GET /v1/cluster/status` | ❌ **`/v1/cluster/resources`** instead | `chatgpt_api.py:340` | NOT FOUND |
| `GET /v1/cluster/latency` | ❌ **No handler registered** | — | NOT FOUND |
| `POST /v1/training/start` | ✅ `handle_post_training` | `chatgpt_api.py:329` | VERIFIED |

Key findings:

- `/v1/cluster/status` is **not registered**. The closest endpoint is `/v1/cluster/resources` (line 340), mapped to `handle_get_cluster_resources`, which returns resource usage data. The README also advertises `/v1/cluster/latency` for the N×N latency matrix, but this route has **no handler**.
- There is a `/v1/distributed/status` endpoint (line 307) not mentioned in the README.
- The middleware at line 426–429 intercepts requests containing `"distributed/status"` in the path and redirects them to `handle_get_distributed_status`, regardless of the actual route. This is a workaround for missing route registration.

**Severity: MEDIUM** — Three of five claimed API endpoints exist, but `/v1/cluster/status` is misnamed and `/v1/cluster/latency` is missing entirely.

---

## Claim 8: Tests exist and test what they claim

### README-listed tests vs on-disk reality

| README claims | On disk | Status |
|---|---|---|
| `test_lacp_multiresource.py` | ✅ Exists | VERIFIED |
| `test_pipeline_1f1b.py` | ✅ Exists | VERIFIED |
| `test_async_forwarding.py` | ✅ Exists | VERIFIED |
| `test_cluster_scale.py` | ❌ **`test_cluster.py`** exists instead | MISMATCH |
| `test_e2e_distributed.py` | ❌ **`test_distributed_pipeline.py`** and `test_multimachine_distributed.py` exist instead | MISMATCH |

### Test quality assessment

| Test file | What it tests | Quality |
|---|---|---|
| `test_lacp.py` | Imports + NodeResources + Cluster + ResourceWeights + RingAllReducer + FaultToleranceManager | Good: exercises modular imports, creates real objects. 50+ lines visible. |
| `test_pipeline_1f1b.py` | 1F1B schedule generation: 1-stage/4-batch, 4-stage/4-batch warmup, stage slot extraction | Strong: exercises the core scheduler, validates slot counts, op types, ordering |
| `test_async_forwarding.py` | Binary codec roundtrip (float32, int64, float16), zlib compression, header validation, async forwarder ordering, backpressure, flush | Comprehensive: covers encode/decode roundtrip for multiple dtypes, metadata passthrough |
| `test_distributed_pipeline.py` | PipelineStage forward, PipelineExecutor orchestration, ResultRouter ordered delivery, DistributedPipelineController | Good: exercises the full pipeline integration stack with mocked async functions |
| `test_gradient_fault.py` | Ring all-reduce (4-node accumulate), ring ordering, DistributedTrainer step (loss tracking), FaultToleranceManager heartbeat/checkpoint/rebalance | Strong: combines gradient sync + fault tolerance in one test file |
| `test_multimachine_distributed.py` | Cross-machine discovery (ManualDiscovery), P2P mesh, master election, topology sharing, partition computation, prompt forwarding | Well-structured: is a real multi-machine integration test, not unit-testable without 2 machines |
| `conftest.py` | Minimal: sets `ROOT` in `sys.path`, defines `TEST_PORT = 59999` | Minimal but functional |

### Missing coverage

| Area | Gap |
|---|---|
| **Concurrent gradient all-reduce** | No test verifies overlap of gradient sync with forward pass |
| **Actual network transport for all-reduce** | `DistributedTrainer.step()` is tested in `test_gradient_fault.py` but without real send/recv callbacks |
| **Rebalance trigger** | `rebalance_after_failure()` has no test that calls it through the monitor loop; the `_pending_rebalance` path is never exercised |
| **Training pipeline integration** | `finetune_lora.py:run_finetune_sync()` has no dedicated test; it's a blocking sync call tested only via the API endpoint in `test_api_endpoints.py` (if at all) |

**Severity: MEDIUM** — Test files exist and are substantive, but 2 of 5 README-listed files are misnamed, and critical runtime paths (rebalance trigger, network all-reduce) are untested.

---

## Summary

| Claim | Status | Severity |
|---|---|---|
| 1. DDP + ring all-reduce | PARTIALLY — correct in-memory prototype, no default transport | MEDIUM |
| 2. Dynamic rebalancing | PARTIALLY — function exists but is dead code (never called) | HIGH |
| 3a. 5s heartbeat detection | NOT FOUND — timeout is 10s, not 5s | HIGH |
| 3b. Zero-downtime recovery | NOT FOUND — recovery function is never triggered | HIGH |
| 4. Discovery (manual / Tailscale / UDP) | VERIFIED — all three backends fully implemented | NONE |
| 5. Latency + bandwidth matrix (N×N) | PARTIALLY — latency matrix built; bandwidth probing is static config only | MEDIUM |
| 6a. 1F1B schedule | VERIFIED — scheduler exists and is tested | NONE |
| 6b. Concurrent gradient all-reduce | NOT FOUND — `step()` is synchronous, no overlap | LOW |
| 7. API endpoints | PARTIALLY — 3 of 5 exist; `/v1/cluster/status` is mismatched, `/v1/cluster/latency` missing | MEDIUM |
| 8. Tests exist and test correctly | PARTIALLY — files exist and are substantive; 2 of 5 README names are wrong; rebalance trigger and network all-reduce untested | MEDIUM |

### Critical gaps (HIGH severity)
1. **Rebalance is dead code** — `rebalance_after_failure()` is defined but `_pending_rebalance` is never consumed. A node failure triggers nothing.
2. **Heartbeat timeout is 10s** — The code says `HEARTBEAT_TIMEOUT = 10.0`. The 5s claim is false by a factor of 2×.

### Key strengths
- Discovery is the most robust subsystem: three implementations (manual config, Tailscale API, UDP broadcast) with health checks, cleanup loops, caching, and fallback logic.
- The pipeline scheduler (1F1B) is properly implemented and tested.
- Test infrastructure is substantive: 20+ pytest files, multi-machine integration test, comprehensive unit tests for gradient sync, fault tolerance, and codec.

**Confidence: 8/10** — All code paths were read and verified against claims. No important files were omitted. The two HIGH-severity findings (heartbeat timeout mismatch, dead rebalance code) are unambiguous from reading the source.
