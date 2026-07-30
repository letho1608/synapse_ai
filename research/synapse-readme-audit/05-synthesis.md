# Synthesis: Unified Cross-Referenced Audit of Synapse AI

**Derived from:** Lane 1 (Core Architecture), Lane 2 (Pipeline & Transport), Lane 3 (Partitioning & Topology), Lane 4 (Tests & Resiliency)
**Date:** 2026-07-31
**Method:** All four specialist reports read and cross-referenced; key points verified against source code to resolve contradictions.

---

## 1. Cross-Cutting Themes

### Theme A: Framework Exists but Is Not Wired to Runtime

This is the single most pervasive pattern in the codebase. Across four lanes, the code consistently shows a well-designed **scaffolding** whose runtime wiring is missing, incomplete, or dead:

| Component | Framework Status | Runtime Wiring | Lanes |
|-----------|----------------|----------------|-------|
| 1F1B Schedule Generator (`scheduler.py:89-146`) | Complete, tested | **Called from tests only** — no production code path invokes it | 1, 2, 4 |
| Dynamic Rebalancing (`fault_tolerance.py:189-241`) | Full implementation with checkpoint/restore | **Dead code** — `_pending_rebalance` is written but never consumed | 1, 4 |
| Ring All-Reduce (`gradient_sync.py:93-145`) | Correct in-memory ring algorithm | **No default transport** — `send_fn`/`recv_fn` are `None` with no wiring provided | 4 |
| Distributed Training (`node_extension.py:270-395`) | Train step exists, calls `train()` | **No gradient sync over P2P mesh** — inference engine handles it internally | 1, 4 |
| Fault Tolerance Monitor (`fault_tolerance.py:249-255`) | Detects dead nodes, sets `_pending_rebalance` field | **Never triggers `rebalance_after_failure()`** — recovery path is dead | 4 |

**Impact:** The README describes a fully operational system. In reality, the codebase is a **semi-assembled engine** — many components are built to spec but the drivetrain that connects them to actual execution is missing.

---

### Theme B: README Claims Not Matched by Code Constants

Four README claims directly contradict hard-coded constants in the source:

| README Claim | Code Reality | Factor | Lanes |
|--------------|-------------|--------|-------|
| "5s heartbeat detection" | `HEARTBEAT_TIMEOUT = 10.0` (`fault_tolerance.py:86`) | 2× difference | 4 |
| "Sub-1ms → same cluster" | `max_intra_latency_ms = 5.0` (`partitioner.py:100`) | 5× difference | 3 |
| "msgpack header" | Custom `struct.pack` 16-byte header (`codec.py:131-138`) | Wrong format | 2 |
| "1F1B schedule for training" | 1F1B schedule exists but no training runtime uses it | No execution | 1, 2, 4 |

None of these are "slightly off" — they are all substantial enough to affect system behavior. A heartbeat timeout of 10s instead of 5s doubles detection latency. A clustering threshold of 5ms instead of 1ms merges clusters that should be separate.

---

### Theme C: Multi-Dimensional LACP — Eight Slots, Four Filled

The LACP resource model is the most architecturally polished component in the codebase, but half of its dimensions are never measured:

```
┌─────────────────┬──────────────┬─────────────────────────────────────┐
│ Resource        │ Populated?   │ Notes                               │
├─────────────────┼──────────────┼─────────────────────────────────────┤
│ GPU VRAM        │ Yes          │ From DeviceCapabilities             │
│ GPU TFLOPS      │ Yes          │ From DeviceCapabilities             │
│ CPU cores       │ Yes          │ From DeviceCapabilities             │
│ System RAM      │ Yes          │ From DeviceCapabilities             │
│ Disk I/O        │ **No**       │ Always 0.0; weight 0.05 is dead     │
│ Network bandwidth│ **No**      │ Always 0.0; weight 0.10 is dead     │
│ Network latency │ Partial      │ Populated for self, not remote      │
│ Power           │ Partial      │ current_power_w set; power_limit_w  │
│                 │              │ always 0 ⇒ headroom always negative  │
└─────────────────┴──────────────┴─────────────────────────────────────┘
```

Source: `resource_weight.py:33-66` (schema), `resource_weight.py:68-94` (what's populated), `node.py:1422-1428` (what production code sets).

**Impact:** The resource weighting engine performs min-max normalization across all 8 dimensions. Since disk, bandwidth, and power_headroom normalize to 0, their weights (0.05, 0.10, 0.05 = 20% of the total score) are effectively redistributed to the remaining 5 dimensions proportionally. The system **does not fail** — it silently falls back to a 5-dimensional weighting that behaves differently than documented.

---

### Theme D: Pipeline Parallelism — Prefill Only, Not Decode

Both Lane 1 and Lane 2 independently identified the critical decode-phase gap. The README (line 40-50) shows a diagram of Node 0 processing batch 2 while Node 1 processes batch 1, claiming bubble-free throughput. This diagram **only applies to the prompt prefill phase**:

- **Prefill (prompt processing):** The input is split into micro-batches, staggered across stages, and overlapping occurs. The pipeline schedule works here.
- **Decode (token generation):** Each token traverses the entire pipeline sequentially — send to Node 0, wait for Node N's output, then send next token. No overlapping.

```python
# distributed_engine.py lines 205-270 — decode loop
while not session.is_finished and len(all_generated_tokens) < max_generate_tokens:
    # ... lookahead guesses ...
    candidate_tokens = [all_generated_tokens[-1]] + guesses
    input_tensor = np.array([candidate_tokens], dtype=np.int64)
    
    await self._send_input_to_stage(...)   # Blocking await — fires one, waits
    next_result = await self._collect_result(...)  # Drains pipeline before next
```

**Impact:** The README's headline throughput numbers (333 tok/s vs 143 tok/s) assume pipeline-overlapped decode. Since decode is token-by-token serial, the actual throughput is closer to `1 / sum(stage_times)` — i.e., the "sequential 7ms total" figure the README uses as its comparison baseline.

---

### Theme E: Scaling Claims Are Mathematical Projections, Not Benchmarks

The README (lines 120-128) presents a speedup table:
> 4 nodes: ~2.8x, 10 nodes: ~7.2x, 100 nodes: ~55x

Lanes 1 and 3 independently confirmed: **no benchmark, no simulation, and no test validates these numbers.** The closest code is `PipelineScheduler.speedup_ratio()` (`scheduler.py:217-242`) which computes a theoretical Amdahl's-law formula that assumes:
- Zero network overhead
- Perfect load balance
- No serialization/deserialization cost
- Bubble-free pipeline (which the decode phase does not achieve)

The referenced `tests/test_cluster_scale.py` **does not exist**.

---

## 2. Consolidated Findings (All Severities, Ranked)

### Blocker Items (will prevent the system from working as advertised)

| # | Finding | Severity | Source Lanes | File:Line | Details |
|---|---------|----------|-------------|-----------|---------|
| B1 | **Decode phase is serial token-by-token, not overlapped** | Blocker | 1, 2 | `distributed_engine.py:205-270` | The README's bubble-free throughput applies to prefill only. During token generation, each token waits for the entire pipeline. The 333 tok/s claim is unsupported. |
| B2 | **Zero-copy shared memory transport is entirely unimplemented** | Blocker | 1, 2 | `README.md:108`, `codec.py:9` (docstring only) | Documented as existing. No `mmap`, `SharedMemory`, or `numpy.memmap` usage exists anywhere. Same-machine fast-path claim is false. |
| B3 | **Dynamic rebalancing on node failure is dead code** | Blocker | 1, 4 | `fault_tolerance.py:189-255` | `rebalance_after_failure()` is fully implemented but never called. `_pending_rebalance` is set but never consumed. A node failure triggers nothing. |
| B4 | **1F1B schedule never runs in production** | Blocker | 1, 2, 4 | `scheduler.py:89-146` → test-only usage | The 1F1B schedule generator is correct, but no training or inference code path calls it. Training uses local-only `run_finetune_sync()`. The README's 1F1B training pipeline is aspirational. |

### High Items (will significantly degrade performance or functionality)

| # | Finding | Severity | Source Lanes | File:Line | Details |
|---|---------|----------|-------------|-----------|---------|
| H1 | **"ILP optimal assignment" is factually wrong — uses Largest Remainder Method** | High | 3 | `partitioner.py:4` (docstring), `resource_weight.py:252-288` | No ILP solver (PuLP, OR-Tools, scipy.optimize) exists in the codebase. The actual algorithm is a greedy proportional allocation scheme. |
| H2 | **Scaling table (2.8x / 7.2x / 55x) is unsupported by any benchmark** | High | 1, 3 | `README.md:120-128`; `test_cluster_scale.py` missing | The formula is theoretical (Amdahl-like). No empirical test or simulation in the repo validates these numbers. The referenced test file doesn't exist. |
| H3 | **4 of 8 LACP resource dimensions are never measured in production** | High | 1, 3 | `resource_weight.py:68-94`, `node.py:1422-1428` | Disk I/O, bandwidth, latency (remote), and power_limit always 0.0. The weights for these (20% total) silently redistribute to other dimensions. |
| H4 | **TCP transport does NOT disable Nagle's algorithm** | High | 2 | `tcp_transport.py` — no `TCP_NODELAY` anywhere | README says "TCP with Nagle disabled." Code doesn't set `TCP_NODELAY`. gRPC transport does set it, but gRPC uses protobuf, not the binary codec. |
| H5 | **Heartbeat timeout is 10s, not 5s as claimed** | High | 4 | `fault_tolerance.py:86` | `HEARTBEAT_TIMEOUT = 10.0`. Discovery latency is 2× the advertised value. |
| H6 | **Master election uses VRAM+FLOPs only, not full multi-resource score** | High | 1 | `election.py:40-48` | README says "Elected by multi-resource weight." The actual formula is `vram_gb + fp32_flops / 100`. No disk, bandwidth, CPU, RAM, or power inputs. |

### Medium Items (notable gaps, reduce correctness/transparency)

| # | Finding | Severity | Source Lanes | File:Line | Details |
|---|---------|----------|-------------|-----------|---------|
| M1 | **Hierarchical clustering uses complete linkage despite docstring claiming single-linkage** | Medium | 3 | `latency_matrix.py:271-279` (code) vs `latency_matrix.py:239-241` (docstring) | Opposite linkage strategies. Single-linkage merges by minimum distance; complete-linkage by maximum. Different clustering behavior. |
| M2 | **Clustering threshold is 5ms by default, not <1ms as claimed** | Medium | 3 | `partitioner.py:100` | README says "nodes with <1ms latency get adjacent shards." Code defaults to `max_intra_cluster_latency_ms=5.0`. The documented <1ms and >10ms thresholds do not exist. |
| M3 | **Power headroom weight is always zero due to unset `power_limit_w`** | Medium | 3 | `resource_weight.py:187-193`, `node.py:1428` | `power_headroom = power_limit_w - current_power_w`. Since `power_limit_w` is always 0 and `current_power_w` > 0, headroom normalizes to 0, nullifying the 0.05 weight. |
| M4 | **Bandwidth probing is a placeholder — always returns 50ms default latency** | Medium | 3, 4 | `latency_probing.py:128-131` | The networking `LatencyProber.get_matrix()` falls back to 50ms (same cluster) / 200ms (cross-cluster) defaults. No actual bandwidth measurement. |
| M5 | **API endpoints mismatch: `/v1/cluster/status` → actual `/v1/cluster/resources`; `/v1/cluster/latency` missing** | Medium | 4 | `chatgpt_api.py:340` vs `README.md:190-198` | README lists 5 API endpoints; 3 exist as documented. `/v1/cluster/latency` has no handler. |
| M6 | **Distributed training (DDP) gradient sync has no default transport** | Medium | 4 | `gradient_sync.py:93-145` | `send_fn`/`recv_fn` are `Optional[Callable]` defaulting to `None`. The ring all-reduce is correct in memory but requires external wiring for network transport. |
| M7 | **N-gram lookahead (speculative decoding) exists but is undocumented** | Medium | 1 | `distributed_engine.py:207-270` | A non-trivial optimization (N-gram guess + verify) exists in the distributed engine but is not mentioned in the README. This is a feature that should be advertised. |

### Low / Note Items

| # | Finding | Severity | Source Lanes | File:Line | Details |
|---|---------|----------|-------------|-----------|---------|
| L1 | **`async_forwarder.py` listed in README structure doesn't exist** | Low | 2 | `README.md:231` | The class lives in `codec.py`. Minor documentation structure mismatch. |
| L2 | **README project structure shows `synapse/lacp/` but actual path is `backend/synapse/topology/`** | Low | 1, 3 | `README.md:222-233` | The directory listing is aspirational; the actual module layout uses different paths. |
| L3 | **`test_cluster_scale.py` and `test_e2e_distributed.py` don't exist** | Low | 3, 4 | `README.md:242-246` | Listed as test files; `test_cluster.py` and `test_distributed_pipeline.py` exist instead. |
| L4 | **Concurrent gradient all-reduce (with next forward pass) not implemented** | Low | 4 | `gradient_sync.py:93-145` | The README claims overlaps; `step()` is purely sequential. |

### Resolved Contradictions Between Specialists

The following apparent contradictions were investigated and resolved by re-reading source code:

1. **Lane 1 vs Lane 3 on "Multi-resource LACP Verified"**: Lane 1 classified Claim 3 as **VERIFIED** (engine structure is correct); Lane 3 classified the same claim as **PARTIALLY** (4 of 8 dimensions never populated). **Resolution: Lane 3 is the more accurate classification.** Both reports agree on the facts (4/8 unpopulated), but Lane 1's "Verified" label is misleading given the gap. The unified verdict is **PARTIALLY**.

2. **Lane 1 says "fire-and-forget async send" vs Lane 2 says "blocking await"**: Lane 1 described `AsyncTensorForwarder.send()` (`codec.py:316`, uses `create_task`). Lane 2 described `_send_input_to_stage()` (`distributed_engine.py:290-324`, uses `await event_router.publish`). **Resolution: Both are correct for their specific path.** The transport-layer forwarder is non-blocking; the distributed engine's pipeline input method is blocking. The README describes the transport behavior, but the distributed engine's actual usage is blocking.

3. **Lane 1 says "TCP_NODELAY not checked" vs Lane 2 says "TCP_NODELAY only on gRPC"**: Lane 1 didn't flag this. Lane 2 correctly identified it. **Resolution: Lane 2 is correct.** Re-reading `tcp_transport.py` confirms zero socket option manipulation. The claim is HIGH severity, not mentioned by Lane 1.

---

## 3. README Claims: Verified vs. Aspirational

| README Claim (line) | Specialist Verdict | Unified Verdict | Key Evidence |
|---|---|---|---|
| **"True distributed supercomputing system that pools ALL hardware resources"** (L9-11) | L1: PARTIALLY, L3: PARTIALLY | **PARTIALLY** | Pipeline-parallel architecture (not pooled). 4/8 resource dimensions measured. Architecture is legitimate but overdescribed. |
| **"Pipeline parallelism with micro-batch overlapping"** (L18, L40-50, L80-90) | L1: PARTIALLY (Blocker), L2: PARTIALLY (High) | **PARTIALLY — Prefill only** | Decode phase is serial token-by-token (`distributed_engine.py:205-270`). The README diagram applies to prompt prefill only. |
| **"Multi-resource LACP: VRAM + TFLOPS + CPU + RAM + disk I/O + bandwidth"** (L19, L69-77) | L1: VERIFIED, L3: PARTIALLY | **PARTIALLY** | Schema has 8 fields. Only VRAM, TFLOPS, CPU, RAM are populated in production (`resource_weight.py:68-94`). |
| **"Latency-aware clustering: <1ms → adjacent shards; >10ms → separate"** (L21, L77) | L3: PARTIALLY / NOT FOUND | **PARTIALLY** | Clustering exists and works. Threshold default is 5ms (`partitioner.py:100`), not <1ms. The >10ms threshold is fictional. |
| **"Async non-blocking forwarding: next micro-batch starts immediately"** (L22, L98-111) | L1: VERIFIED, L2: PARTIALLY | **PARTIALLY** | Transport layer supports fire-and-forget (`codec.py:316`). Distributed engine uses blocking await (`distributed_engine.py:290-324`). |
| **"Binary encoding (msgpack header)"** (L98, L107) | L1: VERIFIED, L2: VERIFIED | **VERIFIED** (format differs) | Binary codec exists with 16-byte custom header (`codec.py:131-138`). Uses `struct.pack`, not msgpack. Minor doc inaccuracy. |
| **"Zero-copy shared memory (numpy memmap)"** (L108) | L1: NOT FOUND, L2: NOT FOUND (Blocker) | **NOT FOUND** | Zero implementation. Mentioned in README and codec.py docstring only. No `mmap`/`SharedMemory`/`memmap` usage. |
| **"TCP with Nagle disabled"** (L109) | L2: PARTIALLY (High) | **PARTIALLY — Wrong transport** | TCP_NODELAY set on gRPC (`grpc_server.py:35`), NOT on TCP transport (`tcp_transport.py`). README describes TCP path. |
| **"Optional RDMA for InfiniBand"** (L110) | L1: NOT FOUND, L2: NOT FOUND | **NOT FOUND** | Zero imports, zero modules, zero configuration. |
| **"Result routing with (request_id, orchestrator_id, micro_batch_id, sequence_position)"** (L113-114) | L1: VERIFIED | **VERIFIED** | `RoutedResult` dataclass (`result_router.py:20-30`), dual-topic routing, out-of-order reassembly. |
| **"In-order reassembly — caller sees clean token stream"** (L114) | L1: VERIFIED, L2: VERIFIED | **VERIFIED** | Two-layer reassembly: transport (`codec.py:361-369`) + result (`result_router.py:42-52`). Both tested. |
| **"Master elected by multi-resource weight"** (L32) | L1: VERIFIED | **PARTIALLY (oversold)** | Elected by VRAM+FLOPs only (`election.py:40-48`), not full multi-resource score. Functional but simpler. |
| **"1F1B schedule"** (L55-62) | L1: VERIFIED, L2: VERIFIED, L4: VERIFIED | **PARTIALLY (orphaned)** | Schedule generator exists and is correct (`scheduler.py:89-146`). Never called from any production code path. Test-only. |
| **"Gradient all-reduce runs concurrently with next forward pass"** (L61) | L4: NOT FOUND | **NOT FOUND** | `DistributedTrainer.step()` is synchronous (`gradient_sync.py:93-145`). No overlap mechanism. |
| **"Dynamic rebalancing: failed node's shard reassigned live"** (L25) | L1: PARTIALLY, L4: PARTIALLY (HIGH) | **PARTIALLY (dead code)** | `rebalance_after_failure()` exists (`fault_tolerance.py:189-241`) but never called. |
| **"5s heartbeat detection"** (implied by L25) | L4: NOT FOUND (HIGH) | **NOT FOUND** | `HEARTBEAT_TIMEOUT = 10.0` (`fault_tolerance.py:86`). |
| **"Linear-ish speedup: 2.8x / 7.2x / 55x"** (L120-128) | L1: NOT FOUND, L3: NOT FOUND (HIGH) | **NOT FOUND** | No benchmark validates these. Theoretical formula only (`scheduler.py:217-242`). |
| **"ILP optimal assignment"** (L226) | L3: NOT FOUND (HIGH) | **NOT FOUND** | No ILP library. Uses Largest Remainder Method (greedy) (`resource_weight.py:252-288`). |
| **Discovery (manual / Tailscale / UDP)** (L179-185) | L4: VERIFIED | **VERIFIED** | Three backends, all implement `Discovery` ABC with health checks, polling, cleanup. |
| **API: `/v1/cluster/status`, `/v1/cluster/latency`** (L190-198) | L4: PARTIALLY | **PARTIALLY** | `/v1/cluster/resources` exists (not `status`). `/v1/cluster/latency` has no handler. |

---

## 4. Gaps the Specialists Missed

### Gap 1: No Dynamic Analysis / Runtime Verification

All four specialists performed **static code analysis only**. None:
- Ran the test suite to determine which tests pass/fail
- Started a single-node instance to smoke-test the API
- Built a minimal 2-node setup to verify pipeline execution

**Impact:** Several claims that appear structurally present in code could still fail at runtime due to import errors, configuration bugs, asyncio deadlocks, or missing dependencies. For example, the `_send_input_to_stage()` → `event_router.publish()` path depends on the event router being properly instantiated and subscribed — none of the specialists verified this at runtime. The confidence in all "VERIFIED" verdicts is bounded by the fact that no code was executed.

### Gap 2: Security and Input Validation

None of the specialists examined:
- **Pickle deserialization**: `fault_tolerance.py` uses `pickle.load()` for checkpoint loading, which is a well-known remote code execution vector if checkpoint files are ever exchanged between nodes.
- **No input validation on API endpoints**: The chat completions API accepts arbitrary user input without size limits, injection checks, or authentication guards.
- **No TLS/mTLS**: All inter-node communication (TCP, gRPC, discovery) is unencrypted. The README says "zero-config" which implies no security.
- **No authentication/authorization**: The API endpoints have no auth middleware. Any process on the network can send requests.

While these are typical for a research project, the README's "production-ready" framing makes their absence noteworthy.

### Gap 3: Dependency and Environment Assumptions

No specialist verified:
- Whether `requirements.txt` contains all dependencies needed to run (e.g., PyTorch, CUDA, transformers, grpcio, etc.)
- Whether the code works on the stated platform (Windows — the README shows Windows commands)
- What minimum PyTorch/CUDA version is required
- Whether the code detects missing GPUs gracefully (it fakes device capabilities, but is this path tested?)

### Gap 4: The Training Pipeline Is Almost Entirely Local

None of the specialists explicitly called out that **the actual training path is fully local**. `finetune_lora.py:run_finetune_sync()` calls HuggingFace `Trainer` with PEFT — a standard single-node fine-tuning script. Despite the README's distributed training section, the codebase has no end-to-end distributed training path. The `DistributedTrainer` and ring all-reduce are structurally present but disconnected from any model training loop.

---

## 5. Prioritized Recommendations

| Priority | Action | Impact | File:Line |
|----------|--------|--------|-----------|
| **1** | **Fix decode-phase pipeline parallelism** — Implement micro-batch overlapping for the autoregressive decode loop, or at minimum document that overlapping applies only to prefill. Without this, the headline speedup claims are invalid. | **Critical** — Core architectural promise | `distributed_engine.py:205-270` |
| **2** | **Wire up 1F1B schedule to training path** — Either connect `generate_1f1b_schedule()` to `DistributedTrainer.step()` and the pipeline stages, or remove it from the README. Currently it's misleading. | **Critical** — README accuracy, 3 specialists flagged | `scheduler.py:89-146` → training integration |
| **3** | **Trigger rebalance on node failure** — Add a monitor loop that calls `rebalance_after_failure()` when `_pending_rebalance` is set. Currently the entire fault tolerance subsystem is dead code. | **Critical** — Claimed feature doesn't fire | `fault_tolerance.py:249-255` |
| **4** | **Populate resource measurements** — Implement disk I/O sampling (`psutil.disk_io_counters`), bandwidth estimation (real probe, not RTT guess), and power limit reading from nvidia-smi. These 4 dimensions are structurally supported but zero-valued. | **High** — Restores 20% of LACP weighting | `resource_weight.py:68-94`, `node.py:1422-1428` |
| **5** | **Reduce heartbeat timeout to 5s or document the actual 10s value** — Change `HEARTBEAT_TIMEOUT = 10.0` to 5.0 to match the README, or update the README. | **High** — Claimed vs actual differs by 2× | `fault_tolerance.py:86`, `README.md` |
| **6** | **Implement or remove zero-copy shared memory** — This is a blocker for same-machine performance. Either implement `multiprocessing.shared_memory` or `numpy.memmap`-based transport, or remove the claim. | **High** — Blocker-level gap between doc and code | `README.md:108`, new transport module |
| **7** | **Add TCP_NODELAY to tcp_transport.py** — The TCP transport doesn't disable Nagle. Add `socket.TCP_NODELAY` to the asyncio server/client sockets, or clearly document that gRPC is the actual transport. | **High** — Direct latency penalty for cross-machine traffic | `tcp_transport.py` (server + client creation) |
| **8** | **Remove "ILP" from documentation and comments** — The algorithm uses Largest Remainder Method (greedy proportional allocation). Calling it "ILP optimal" is false and undermines credibility. | **Medium** — Code-doc mismatch | `partitioner.py:4`, `latency_probing.py:121`, `README.md:226` |
| **Bonus** | **Run the existing test suite and fix failures** — Before any refactoring, establish a known-good baseline. Tests exist for codec, scheduler, and fault tolerance — ensure they all pass in the current state. | **Foundational** — Establishes trust | `backend/tests/` |

---

## 6. Combined Confidence

**Confidence: 5/10** — The codebase delivers roughly 40-50% of what the README promises.

**Reasoning:**
- The **transport layer** (binary codec, TCP, result routing, in-order reassembly) is genuinely well-implemented and tested. Approaching production quality.
- The **LACP partitioning engine** (resource weighting, hierarchical clustering, partition computation) is architecturally sound but only half-populated with real data.
- The **discovery subsystem** (manual, Tailscale, UDP) is the most robust component — three full implementations with health checks and cleanup.
- The **core promise** — pipeline-parallel inference with near-linear speedup — is **not delivered**. The decode phase is serial. The speedup numbers are theoretical. The 1F1B pipeline for training is disconnected.
- The **fault tolerance** (rebalancing, heartbeat detection) has a complete implementation that is **never triggered** — it's a museum piece.
- The **API** exposes OpenAI-compatible chat completions, which works for single-node operation, but the cluster monitoring endpoints are incomplete.

**Ground-truth tally:**
- **Load-bearing conclusions that are externally verified (code read / git / tests / cited sources):** 14 of 20 README claims have at least partial code backing that was directly observed.
- **Load-bearing conclusions that rest on model judgment:** 6 of 20 involve severity ratings, runtime behavior analysis, or integration completeness assessment that was inferred from reading code paths without execution.
- **Specific externally verified facts:**
  - `HEARTBEAT_TIMEOUT = 10.0` confirmed at `fault_tolerance.py:86` (code read)
  - `max_intra_latency_ms = 5.0` confirmed at `partitioner.py:100` (code read)
  - No `TCP_NODELAY` in `tcp_transport.py` (grep confirmed zero matches)
  - `from_device_capabilities()` only populates 6 of 10 fields (code read, lines 86-93)
  - `power_limit_w` never set in `node.py` (grep confirmed zero matches)
  - `rebalance_after_failure()` never called from monitor (full path traced in `fault_tolerance.py:249-255`)
  - `generate_1f1b_schedule()` only called from tests (grep confirmed)
  - No ILP library imported anywhere in codebase (grep confirmed)
  - `test_cluster_scale.py` does not exist (filesystem verified)
- **Model judgment applied to:**
  - Whether the decode-phase serial execution materially affects throughput (it mathematically must — pipeline depth × token latency)
  - Whether the resource weighting still functions acceptably with 4 zero-valued dimensions (it runs, but produces different results than documented)
  - Whether missing `power_limit_w` is a bug or intentional (it appears to be an oversight given the rest of the framework)

---

## 7. Load-Bearing Claims Needing Follow-Up

Before the synthesis confidence can be raised above 5/10, these three claims require deeper verification:

### 1. The Prefill-Phase Pipeline Overlap — Is It Actually Working?

The specialists observed that `distributed_engine.py:377-470` calls `generate_inference_schedule()` and runs the staggered schedule, but **none of them ran it**. The prefill overlapping depends on:
- `PipelineExecutor.run_inference()` correctly staggering micro-batches through `process_loop` concurrently
- The async queues not blocking due to backpressure
- The event router correctly distributing tensor messages to downstream stages

**What's needed:** A 2-node integration test with trace logging that measures per-micro-batch timing through the pipeline during prefill. Verify that Node 1 starts processing micro-batch 0 before Node 0 finishes sending all micro-batches. Without this, even the prefill-phase overlapping claim is unsubstantiated at runtime.

### 2. Does the N-gram Lookahead Actually Accelerate Decode?

The `process_distributed_prompt()` method (Lane 1 noted, lines 207-270) implements an N-gram speculative decoding optimization that is **not mentioned in the README at all**. This is the only mechanism that could partially mitigate the serial-decode bottleneck. But:
- It generates "guesses" for future tokens based on an N-gram table — where does this table come from?
- It sends guess batches through the pipeline and verifies them — this does create some parallelism in the decode loop
- Is the confidence threshold for accepting guesses calibrated?
- Does it actually improve throughput, or add overhead?

**What's needed:** An ablation study with lookahead enabled/disabled on a real or simulated multi-node setup. If the lookahead works, this is a major undocumented feature. If it doesn't, it's dead code like the rebalancing.

### 3. Does the System Actually Start and Serve a Request on 2+ Nodes?

This is the **most basic integration test** that no specialist performed:
- Do the discovery mechanisms successfully form a cluster?
- Does master election converge?
- Does the partition computation produce valid (non-overlapping, covering all layers) assignments?
- Does the first micro-batch successfully traverse Node 0 → Node 1 → ... → Node N?
- Does the result reach the orchestrator?

The code exists, imports resolve, and paths are traced, but **no specialist executed a single `pytest` run**. The existing test files (20+ pytest files) may reveal import failures, asyncio timeout bugs, or configuration assumptions that a static analysis cannot detect. Running `pytest backend/tests/ -v` is the single highest-leverage follow-up action.

---

## Appendix: Per-Lane Confidence Reconciliation

| Lane | Reported Confidence | Scope | Specialization Bias |
|------|-------------------|-------|-------------------|
| L1: Core Architecture | 5/10 | Broadest — covered all README claims | Most critical of README gap; caught the decode-phase blocker |
| L2: Pipeline & Transport | 6/10 | Narrow — pipeline, transport, networking | Focused on module structure; caught TCP_NODELAY and shared memory gaps |
| L3: Partitioning & Topology | 7/10 | Narrow — LACP engine, clustering, topology | Most generous; focused on structural correctness; rated partially implemented systems higher |
| L4: Tests & Resiliency | 8/10 | Narrow — tests, fault tolerance, API, discovery | Highest confidence because scope was well-bounded and findings were unambiguous (constants, file existence) |
| **Synthesis** | **5/10** | **All of the above** | Lower confidence reflects cross-cutting findings that compound: orphaned code + missing runtime wiring + unverifiable claims |

The synthesis confidence (5/10) is lower than any individual lane's score because:
- L4's high confidence (8/10) came from their narrow scope (discovery works, tests exist) — these are real strengths but don't offset the core architecture gaps
- The cross-cutting findings (orphaned 1F1B, dead rebalance, serial decode) create **compounding risk**: the system has multiple single points of failure that individually would be "Medium" but together are "Blocker"
- The static-analysis-only limitation binds all lane verdicts — no code was executed
