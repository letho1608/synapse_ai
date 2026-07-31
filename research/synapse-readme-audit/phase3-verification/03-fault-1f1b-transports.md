# Phase 3 Verification — Lane C: Fault Tolerance, 1F1B Wiring, TCP/SHM Transports, Web UI Root Route

**Agent:** Lane C (specialist research agent)
**Date:** 2026-07-31
**Scope:** `synapse_ai` at `D:\Code\paper\synapse_ai`. Research only — no source or test files were modified.
**Method:** Read the shipped code, traced every claimed call path end to end, and ran every designated test suite myself (52 tests, all passing). Prior audit conclusions were treated as untrusted and re-derived from source.

## 0. Verdict in one paragraph

The implementation pass fixed the *mechanics* but not the *wiring*. The heartbeat constants, JSON checkpoint serializer, disk-scan recovery fallback, `rebalance_after_failure()`, `get_recovery_checkpoint()`, TCP_NODELAY on all three `tcp_transport.py` socket sites, the zero-copy SHM transport, and the web-UI root route are all genuinely present, and their tests all pass (52/52 that I ran). But three headline README claims remain **not true of shipped runtime behavior**: (1) the production 1F1B path is unreachable because the API reads a node attribute `_pipeline_extension` that no code ever sets — a dead branch that the "production wiring" test misses because it is source-string inspection; (2) no production code ever writes a FaultToleranceManager checkpoint, and the recovered checkpoint is staged but never applied, so "state recovered from JSON checkpoint" describes a mechanism production never exercises; (3) the TCP and SHM transports are correctly built and peer-registered but are **not on the runtime data plane** — tensor traffic still goes over `EventRouter.publish` → `P2PSocketBridge` (JSON-over-socket, no TCP_NODELAY), so "cross-machine nodes use TCP with Nagle disabled" and "same-machine nodes use zero-copy shared memory" describe components, not the shipped runtime. The web UI root-route claim is fully true.

---

## 1. What I ran

All commands from `D:\Code\paper\synapse_ai\backend` (Python 3.10.8, pytest 8.2.2, `asyncio_mode=auto` from `backend/pytest.ini`):

| Suite | Result |
|---|---|
| `pytest tests/test_cross_node_recovery.py tests/test_fault_tolerance_live.py -v` | **15 passed** (2.97s) |
| `pytest tests/test_1f1b_training_production.py tests/test_1f1b_production.py tests/test_pipeline_1f1b.py -v` | **19 passed** (8.55s) |
| `pytest tests/test_tcp_nodelay.py -v` | **3 passed** (1.11s) |
| `pytest tests/test_shm_transport.py tests/test_multinode_lacp_flow.py -v` | **11 passed** (3.64s) |
| `pytest tests/test_api_endpoints.py -v` | **4 passed** (25.95s) |

52/52 green. Green tests are necessary but not sufficient; several of the most important claims are only "verified" by tests that inspect source strings or bypass the production entry point (Section 7).

---

## 2. Fault tolerance — heartbeat, JSON checkpoint, cross-node recovery

### 2.1 What the README claims

- Line 26: *"Dynamic rebalancing: failed node's shard reassigned live; state recovered from JSON checkpoint (no pickle)"*
- Line 217: *"heartbeat monitoring (2s interval, 5s timeout = `HEARTBEAT_TIMEOUT`) checks every node; dead nodes trigger `check_timeout()` in the monitor loop and `rebalance_after_failure()` repartitions layers among survivors with JSON checkpoint recovery"*

### 2.2 Constants and JSON serialization — VERIFIED TRUE

- `HEARTBEAT_INTERVAL = 2.0` and `HEARTBEAT_TIMEOUT = 5.0` — `backend/synapse/pipeline/fault_tolerance.py:93-94` (exact match to README).
- `save_checkpoint()` writes JSON via `json.dump(sanitized, f, default=str, indent=2)` with a recursive numpy-safe converter (`np.ndarray → tolist`, `np.integer → int`, `np.floating → float`) — `fault_tolerance.py:147-198`. Filenames are node-prefixed: `f"{self.node_id}_{model_id}_L{start}-{end}_step{step}.ckpt"` — `fault_tolerance.py:173-174`. Verified by `test_save_checkpoint_filename_includes_node_id` and the JSON round-trip test.
- No `pickle` anywhere in `fault_tolerance.py` (grep: zero hits). The "no pickle" claim is true.

### 2.3 Cross-node recovery lookup — VERIFIED TRUE (mechanism level)

- `load_latest_checkpoint(node_id)` first checks the in-memory dict, then falls back to a disk scan of `self.checkpoint_dir` for files prefixed `{node_id}_`, reconstructing a `ShardCheckpoint` from JSON — `fault_tolerance.py:209-238`. This is the exact fix the prior audit demanded, and it works: `test_load_latest_checkpoint_falls_back_to_disk` passes with a *different* `FaultToleranceManager` instance finding a dead node's on-disk state.

### 2.4 Rebalance + staging — VERIFIED TRUE (mechanism level)

- `rebalance_after_failure()` loads the dead node's latest checkpoint, picks the surviving node whose new range absorbs the dead shard (`s <= dead_shard[0] <= e`, `fault_tolerance.py:318-322`), returns it in `RebalanceResult.recovery_checkpoint` / `.recovery_target_node`, and stages it in `self._recovery_checkpoints[target]` — `fault_tolerance.py:309-331`. `get_recovery_checkpoint(node_id)` exposes it — `fault_tolerance.py:340-343`.
- `Node.check_fault_tolerance()` calls `rebalance_after_failure()` and stores the result in `self._last_rebalance_result` (no longer discarded) — `backend/synapse/orchestration/node.py:470-491`. The recovered state is therefore staged for the survivor, and the staging is exercised end-to-end by `test_check_fault_tolerance_recovers_and_stages` (which passes).

### 2.5 What is still NOT true about fault tolerance

**HIGH — FT-1: Production never writes a checkpoint, so "state recovered from JSON checkpoint" is vacuous at runtime.**
`FaultToleranceManager.save_checkpoint()` (`fault_tolerance.py:147`) has **zero production callers**. The only calls are in tests (`test_cross_node_recovery.py`, `test_fault_tolerance_live.py`, `test_gradient_fault.py`, `test_lacp.py`). The only production save is `node.py:1019` → `inference_engine.save_checkpoint(shard, path)` (`pytorch_hf_engine.py:586`) — a completely different model-weight/safetensors saver, not the JSON shard-state checkpoint. `CHECKPOINT_INTERVAL = 30.0` (`fault_tolerance.py:94`) is dead — nothing runs a periodic snapshot loop; `_monitor_loop` never saves. Consequence: in a live cluster a dead node leaves nothing on disk, `recovered_from_checkpoint` is always `False`, and the rebalance always runs with `recovery_checkpoint=None`. The recovery *plumbing* is real; the *checkpoint writing* that feeds it does not exist in production.

**HIGH — FT-2: The recovered checkpoint is staged but never consumed to resume work.**
`get_recovery_checkpoint()` has no production caller (grep: tests only). `node.py:482-491` logs the recovery but no code applies the staged kv_cache/optimizer_state to an engine or resumes the pipeline from `step`. So even in the test-invoked scenario, "recovered" state is stored on the manager and never used.

**MEDIUM — FT-3: Cross-node recovery requires a shared filesystem that the README never mentions.**
`get_checkpoints_dir()` resolves to `Path.home()/.cache/huggingface/hub` (`backend/synapse/loading.py:83-87`). On a real multi-machine cluster each node's checkpoint lands on its own disk; the orchestrator scanning its *own* home dir can never find a dead remote node's file. The tests simulate the shared-dir assumption by handing the same `tmpdir` to both managers. No network pull of the dead node's checkpoint exists. The claim "state recovered from JSON checkpoint" therefore holds only for in-process simulated nodes or a cluster with a shared home mount — neither stated in the README.

**MEDIUM — FT-4: Heartbeat is membership-derived, not a real heartbeat protocol.**
`FaultToleranceManager.record_heartbeat()` is fed by `periodic_topology_collection()` for every peer currently in `self.peers` (`node.py:1384-1393`). No heartbeat messages are ever exchanged with the fault-tolerance layer. Detection therefore only works if discovery *removes* the dead peer (manual discovery does health-check and drop dead peers every 5 s — `backend/synapse/networking/manual/manual_discovery.py:58-66`; the socket bridge also drops pong-timeouts). If a discovery source keeps a peer listed (static config without health checks), `record_heartbeat` keeps marking it alive and the node is never blacklisted.

**MEDIUM — FT-5: Rebalance triggers only after 3 consecutive misses and requires prior distributed-pipeline activity.**
`check_timeout()` blacklists only after `consecutive_misses >= 3` (`fault_tolerance.py:59-67`), so end-to-end death→rebalance takes roughly 9–11 s (5 s first timeout + 2×2 s misses), not "within 5 s". Additionally `check_fault_tolerance()` gates the entire rebalance on `self._distributed_pipeline` being truthy (`node.py:439`), which only happens after something accesses the lazy `distributed_pipeline` property (`node.py:169-174`). A node that dies *before* the first distributed request is never rebalanced. The test papers over this with `node._distributed_pipeline = object()` (`test_cross_node_recovery.py:209`).

**LOW — FT-6: `_pending_rebalance` remains orphaned** (written at `fault_tolerance.py:361-362`, never read) — carried over from earlier audits, cosmetic but real.

**Note — FT-7:** `rebalance_after_failure` swallows exceptions with no logging (`fault_tolerance.py:334-336`, `result.success=False`); combined with a caller that only inspects success when a checkpoint was recovered, silent failures are invisible to operators.

Bottom line for fault tolerance: heartbeat constants ✓, JSON ✓, disk-scan lookup ✓, rebalance trigger and staging ✓, but the checkpoint **write side is unplugged** and the **staged state is never applied** — the two halves of "state recovered from JSON checkpoint" are both missing in production.

---

## 3. 1F1B training wiring

### 3.1 What the README claims

Lines 58-65: *"For training, Synapse uses the **1F1B schedule** from GPipe… **Production wiring:** `DistributedTrainer.train_1f1b()` calls `PipelineScheduler.generate_1f1b_schedule()` to coordinate forward and backward passes across all pipeline stages. `PipelineStage.__init__` accepts `backward_fn: Optional[Callable]`, and its `process_loop()` dispatches `backward` operations when the schedule demands it. All verified by behavioral tests that drive the real shipped functions (see `test_1f1b_training_production.py`)."*

### 3.2 Component-level claims — VERIFIED TRUE

- `DistributedTrainer.train_1f1b()` builds a `PipelineScheduler` and calls `generate_1f1b_schedule(num_stages, num_micro_batches, node_ids)` — `backend/synapse/pipeline/gradient_sync.py:163-243` (call at :189).
- `PipelineStage.__init__` accepts `backward_fn: Optional[Callable]` — `backend/synapse/pipeline/stage.py:66`.
- `process_loop()` dispatches on `mb.metadata["op"]` and calls `run_backward()` when `op == "backward"` and `_backward_fn` is set — `stage.py:120-128`; `run_backward()` at `stage.py:141-163`.
- `generate_1f1b_schedule()` produces warmup/steady/cooldown slots with per-stage forward-then-reverse-backward ordering — `backend/synapse/pipeline/scheduler.py:84-147`.
- The extension layer: `train_step_distributed(schedule=...)` routes `"1f1b"` to `train_step_distributed_1f1b()` — `backend/synapse/pipeline/node_extension.py:267-285`; that method builds one `PipelineStage` per partition with `backward_fn`, runs `self._trainer.train_1f1b(...)` — `node_extension.py:412-552`.

All of that is real code, and `test_1f1b_executes_schedule_end_to_end` genuinely drives the schedule through real `PipelineStage` objects with a spy on `generate_1f1b_schedule` (both phases observed, micro-batch slicing asserted).

### 3.3 BLOCKER — 1F1B-1: The production call site is unreachable (typo'd attribute)

The API's multi-node training loop does:

- `backend/synapse/api/chatgpt_api.py:2603` — `ext = getattr(self.node, "_pipeline_extension", None)`
- `chatgpt_api.py:2649-2653` — `if ext is not None and world_size > 1:` → `await ext.train_step_distributed(base_shard, example_np, target_np, request_id=..., learning_rate=lr_val, schedule="1f1b")`

`_pipeline_extension` appears **exactly once in the entire repository** — that one line. It is never assigned anywhere. The node's actual extension property is `distributed_pipeline` (backing field `_distributed_pipeline`, `node.py:169-174`). Therefore `ext` is always `None`, the branch is always false, and multi-node training always falls through to `self.node.enqueue_example(...)` — the legacy single-node path (`chatgpt_api.py:2655-2657`). Neither `schedule="1f1b"` nor the DDP variant of `train_step_distributed` is ever reached from production. The "production can execute 1F1B" claim is **false**.

The test that is supposed to catch exactly this — `test_production_caller_selects_1f1b` (`test_1f1b_training_production.py:233-240`) — reads the API file's source and asserts the literal string `schedule="1f1b"` appears. It passes despite the dead branch. This is precisely the "source-string inspection instead of driving shipped code" failure mode (see Section 7).

### 3.4 HIGH — 1F1B-2: Even if reachable, the "backward" phase is a documented no-op and there is no inter-stage data flow

- The real engine: `pytorch_hf_engine.train(..., loss="back_gradient")` immediately returns `(None, None)` with the log *"train(back_gradient) chua implement cho nhieu node; bo qua"* ("not implemented for multi-node; skipping") — `backend/synapse/inference/pytorch_hf_engine.py:610-625`.
- `node_extension.py:489-502` wraps that call in `try/except: pass` and returns `np.nan`; the last stage's backward also returns `np.nan`.
- The **forward** `fn` is not a forward pass either — it calls the full `inference_engine.train(..., sync_gradients=True)` which computes loss *and gradients* on the engine (`node_extension.py:472-485`). Each stage then does a ring all-reduce once at the end (`node_extension.py:538-545`).
- `train_1f1b` feeds every stage its own row-slice of the *raw input* (`gradient_sync.py:209-222`) and the stage `send_tensor_fn` is `noop_send` (`node_extension.py:505-508`) — stage s+1 never receives stage s's hidden state. There is no activation/gradient handoff between stages.

So the shipped "1F1B" is a schedule-driven *pacing* of per-stage `train()` calls over raw input slices, not pipeline-parallel backpropagation. Gradients come from each stage's independent local forward+backward, then a single DDP-style all-reduce. The schedule coordinates *timing*, not *data flow*. README line 58's "coordinate forward and backward passes across all pipeline stages" is only true at the scheduling level.

**What is honest about this finding:** the code comments say so themselves ("Engines do not yet implement cross-node back_gradient; the schedule still drives the backward phase so the 1F1B flow executes end to end", `node_extension.py:489-490`). The README's *specific* claims about which functions call which are accurate; its *implication* that production training executes 1F1B is not.

### 3.5 LOW — 1F1B-3: `PipelineStage.run_forward/run_backward` read a non-existent attribute

`MicroBatch` has no `step` field (`scheduler.py:47-53`), but `stage.py:107` and `stage.py:156` do `micro_batch.step if hasattr(micro_batch, "step") else 0` — always 0. The step is stored in `mb.metadata["step"]` and never read back. Cosmetic today (nothing consumes `StageResult.step`), but it is a latent bug if anyone starts relying on step.

---

## 4. TCP_NODELAY transport

### 4.1 What the README claims

Line 117-118: *"Cross-machine nodes use **TCP with Nagle disabled** (`TCP_NODELAY=1`) on both server listening socket and all outbound client connections, confirmed in `tcp_transport.py` with 3 dedicated tests."*

### 4.2 Code reality — VERIFIED TRUE (component level)

All three setsockopt sites are present in `backend/synapse/transport/tcp_transport.py`:
- Server listening socket: `sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)` — `tcp_transport.py:57` (inside `TCPServer.start`, note the `break` at :59 means only the first socket if multiple exist — cosmetic).
- Accepted connection: `sock.setsockopt(...)` in `_handle_connection` — `tcp_transport.py:76-79`.
- Outbound client: in `TCPClientPool._get_connection` after `open_connection` — `tcp_transport.py:142-148`.

All 3 tests in `test_tcp_nodelay.py` pass and genuinely exercise real sockets (`getsockopt == 1` asserted on a real listening socket, a real accepted connection, and a real pooled client connection). Caveat: `test_tcp_connection_handler_sets_nodelay` monkeypatches `_handle_connection` to a stub (so the *real* handler's setsockopt is not what the test observes — the value evidently comes from socket-level inheritance on this platform), and `test_tcp_client_pool_sets_nodelay` reads the *server side* of the accepted socket. The shipped code still sets NODELAY in the real handler and client paths, so the claim about `tcp_transport.py` holds.

### 4.3 HIGH — TCP-1: `tcp_transport.py` is not on the runtime data plane

No production code calls `TensorTCPTransport.send_tensor` (grep: only the definition at `tcp_transport.py:220` and tests). The runtime tensor path is:

`node.py forward_tensor` → `encode_tensor(...)` → `event_router.publish(f"synapse/inference/tensor/{target_id}", {"bin": encoded_payload})` — `node.py:1241-1252`; the distributed engine relays identically — `backend/synapse/inference/distributed_engine.py:484-500`.

Cross-process delivery is performed by `P2PSocketBridge` (`backend/synapse/routing/p2p_socket_bridge.py`), which subscribes to `synapse/#` (`:70`), serializes every packet to **JSON** (numpy arrays → hex, `:370-385`), and sets **no TCP_NODELAY anywhere** (zero `setsockopt` calls in the file). So the real cross-machine socket path (a) does not use the binary codec the README's "Async Tensor Forwarding Protocol" diagram describes, and (b) has Nagle enabled. The README's sentence is literally about `tcp_transport.py`, and `tcp_transport.py` does what it says — but "Cross-machine nodes use TCP with Nagle disabled" describes the component, not the shipped traffic.

---

## 5. Zero-copy shared-memory transport

### 5.1 What the README claims

Lines 113-116: *"Same-machine nodes use **zero-copy shared memory**: tensor payloads travel through `multiprocessing.shared_memory` segments (no socket, no OS copy of the payload), and the receiving node hands its handler a zero-copy numpy view over the shared buffer — implemented in `shm_transport.py` and covered by behavioral tests that assert the payload bypasses TCP."*

### 5.2 Code reality — VERIFIED TRUE (transport level)

`SharedMemoryTransport` (`backend/synapse/transport/shm_transport.py`):
- `send_tensor` creates a `multiprocessing.shared_memory.SharedMemory` segment, packs a `SYNSHM01` header + payload bytes + metadata JSON into `shm.buf`, keeps the creator handle alive in `_active`, then calls the in-process notifier — `shm_transport.py:136-193`.
- `_deliver` attaches the segment and builds the tensor as a **zero-copy numpy view**: `np.ndarray(shape=shape, dtype=dtype, buffer=memoryview(shm.buf)[...])` — `shm_transport.py:198-252`.
- `Node.start()` starts it (`node.py:245-258`); `_register_transport_peers` registers same-machine peers via `is_same_machine` (`node.py:1342-1368`); `Node.stop()` releases segments via `shm_transport.stop()` → `_release_all` (`node.py:345-360`, `shm_transport.py:104-112`).

`test_shm_transport.py` (6 tests) genuinely proves the payload bypasses TCP: the receiver sees exactly the segment the sender wrote, and `test_receive_is_zero_copy_view_over_shared_buffer` mutates the received view and observes the change through an independent handle to the same segment. `test_multinode_lacp_flow.py::TestSHMWiring` verifies the registration logic (loopback peer registered on SHM, remote peer on TCP only) and an end-to-end SHM send through the node's transport.

### 5.3 HIGH — SHM-1: The SHM transport is never selected by any runtime send path

Same evidence as TCP-1: zero production call sites for `shm_transport.send_tensor` (grep: definition + tests only). The node registers peers on SHM so that "node-to-node tensor traffic between co-located nodes can bypass TCP entirely" (comment at `node.py:1347-1348`), but the forwarding code in `forward_tensor` and the distributed engine never consults `self.shm_transport` — it always publishes through the event router. So at runtime, same-machine nodes' tensor traffic does **not** travel through shared memory; the SHM transport is exercised only by tests that call it directly. The README claim is about *runtime behavior* ("Same-machine nodes use zero-copy shared memory") and that behavior does not exist in shipped code.

Also worth noting: the SHM wiring test (`test_multinode_lacp_flow.py:141-158`) explicitly *mirrors* the `Node.start()` transport-start block instead of calling `node.start()`, and the in-process-only registry (`shm_transport.py:68-73`, `89-94`) means SHM only ever works between same-process nodes anyway — consistent with the README's "in-process simulated nodes" known limitation, but again not a runtime data path.

**Note — SHM-2:** `_active` on the sender accumulates every segment until `stop()` (`shm_transport.py:185-190`) — unbounded growth on long-lived nodes; on Windows segments cannot be unlinked while attached, which the code acknowledges (`:181-187`). Acceptable for a test-harness transport; a memory hazard if it ever becomes a data path.

---

## 6. Web UI root route

### 6.1 What the README claims

Known Limitations (line 154): *"The web UI is a backend-served static dashboard (no npm dependency) — `GET /` always returns a real page, served by the API process itself."*

### 6.2 Code reality — VERIFIED TRUE

- `self.app.router.add_get("/", self.handle_root)` is registered **unconditionally** in `ChatGPTAPI.__init__` — `backend/synapse/api/chatgpt_api.py:388-393` (no `__compiled__` / frozen / packaging guard anywhere in the file; grep for `__compiled__` returns nothing).
- `handle_root` (`chatgpt_api.py:535-556`) serves `frontend/index.html` if present, else `frontend/dashboard.html`, else a bundled `_DEFAULT_DASHBOARD_HTML` fallback — so `GET /` always returns `200 text/html`.
- `self.static_dir = Path(__file__).parent.parent.parent.parent / "frontend"` resolves to `D:\Code\paper\synapse_ai\frontend`; `frontend/index.html` exists (verified by reading it — a real dashboard that fetches `/v1/system/info` and `/v1/topology`).
- Tests: `test_root_route_serves_real_page` drives the real aiohttp app (`TestClient(TestServer(api.app))`) and asserts `200` + `text/html` + `<html`; `test_root_falls_back_to_bundled_page_when_frontend_missing` asserts the fallback page. Both pass.

**Verdict: the web-UI claim is fully true.** This was the one area where the fix is complete and wired.

---

## 7. "No mock-only coverage" — false; source-string inspection tests are present

README line 27 claims *"No mock-only coverage."* I found at least 5 tests that assert on source text rather than driving shipped code, plus one that re-implements the code under test:

1. `test_fault_tolerance_live.py:37-52` `test_check_timeout_called_in_monitor_loop` — opens `fault_tolerance.py` and asserts the strings `"check_timeout"` and `"h.check_timeout"` appear. It never runs the monitor loop.
2. `test_fault_tolerance_live.py:103-116` `test_save_checkpoint_no_pickle` — asserts `"pickle.dump"`/`"pickle.load"` absent from source text.
3. `test_fault_tolerance_live.py:117-126` `test_save_checkpoint_uses_json` — asserts `"json.dump"`/`"json.load"` present in source text.
4. `test_fault_tolerance_live.py:157-166` `test_rebalance_after_failure_wired_in_node` — opens `node.py` and asserts `"rebalance_after_failure"` and `"check_fault_tolerance"` appear.
5. `test_1f1b_training_production.py:233-240` `test_production_caller_selects_1f1b` — opens `chatgpt_api.py` and asserts the string `schedule="1f1b"` appears. **This is the one that matters**: it is the only test guarding the production wiring claim, and it passes even though the branch it guards is unreachable (Section 3.3) — a textbook false positive.
6. `test_fault_tolerance_live.py:76-101` `test_monitor_loop_detects_timeout` — labeled "Full integration" but re-implements the loop body as a local `fast_monitor()` coroutine instead of running the real `_monitor_loop` task.

Additional reachability gaps (tests that don't cross the production boundary):
- `test_cross_node_recovery.py:209` stubs `node._distributed_pipeline = object()` to satisfy the rebalance gate instead of going through `node.start()`/distributed activity.
- `test_multinode_lacp_flow.py:141-158` mirrors `Node.start()`'s transport block manually.
- `test_1f1b_executes_schedule_end_to_end` monkeypatches `DistributedTrainer.step` (legitimate spy) but never goes through the API; fine in isolation, but it cannot see the `_pipeline_extension` dead branch.

---

## 8. What does NOT work (consolidated)

1. **Production 1F1B training.** `schedule="1f1b"` is never reachable from the API: `chatgpt_api.py:2603` reads a node attribute that nothing sets (`_pipeline_extension`), so multi-node training silently runs the legacy single-node path. (`test_1f1b_training_production.py:233-240` falsely "verifies" it.)
2. **The 1F1B backward phase.** Real engine returns `(None, None)` for `loss="back_gradient"` with an explicit "not implemented" log (`pytorch_hf_engine.py:622-625`); the extension swallows it (`node_extension.py:497-499`). No hidden-state handoff between stages; `send_tensor_fn=noop_send`.
3. **JSON checkpoint writes in production.** `FaultToleranceManager.save_checkpoint` and `CHECKPOINT_INTERVAL` are dead in production; `recovered_from_checkpoint` can never be true in a live cluster.
4. **Recovered-state consumption.** `get_recovery_checkpoint()` / `_last_rebalance_result` are staged and logged but never applied to resume training or inference.
5. **Cross-machine checkpoint recovery.** Depends on a shared filesystem that the README doesn't disclose; no network pull of a dead node's checkpoint exists.
6. **SHM and TCP transports as the data plane.** Neither is called from the runtime send path; tensor traffic goes over `EventRouter` → `P2PSocketBridge` (JSON-over-socket, no TCP_NODELAY).
7. **Rebalance before first distributed request.** Gated on `_distributed_pipeline` being initialized (`node.py:439`).
8. **"Detect dead nodes within 5 s".** Blacklisting needs 3 misses → ~9-11 s to rebalance; and heartbeat is peer-membership-derived, so a discovery source that keeps dead peers listed breaks detection entirely.

## 9. What DOES work (for balance)

- Heartbeat constants exactly match the README (`fault_tolerance.py:93-94`).
- JSON, numpy-safe, node-prefixed checkpoint files; no pickle (`fault_tolerance.py:147-198`).
- Disk-scan fallback in `load_latest_checkpoint` (cross-instance lookup works) (`fault_tolerance.py:209-238`).
- `rebalance_after_failure` stages the recovered checkpoint and `check_fault_tolerance` stores the result (not discarded) (`fault_tolerance.py:309-331`, `node.py:470-491`).
- TCP_NODELAY set on server, accepted, and client sockets in `tcp_transport.py` (3 real-socket tests pass).
- Zero-copy SHM transport is genuinely zero-copy on receive and payload genuinely bypasses TCP when invoked (6 tests pass).
- `GET /` always returns a real backend-served page; `frontend/index.html` exists; no `__compiled__` guard (4 tests pass).
- `train_1f1b` → `generate_1f1b_schedule` → `PipelineStage(backward_fn=...)` → `process_loop` backward dispatch is real, executable code (19 tests pass).

---

## 10. Executive summary (250 words)

I independently re-derived every claim in my lane from source and ran all 52 designated tests (all green). The implementation pass delivered real components but not real production wiring. **Fault tolerance:** heartbeat constants (2 s / 5 s), JSON numpy-safe node-prefixed checkpoints, the disk-scan fallback in `load_latest_checkpoint`, and checkpoint staging in `rebalance_after_failure` all exist and work under test. But no production code ever calls `save_checkpoint` (`CHECKPOINT_INTERVAL` is dead), the recovered checkpoint is staged but never applied, and true cross-machine recovery requires an undisclosed shared filesystem. **1F1B:** `train_1f1b()` genuinely calls `generate_1f1b_schedule()`, and `PipelineStage` genuinely dispatches backward ops — but the production call site at `chatgpt_api.py:2649` is unreachable because `chatgpt_api.py:2603` reads `_pipeline_extension`, an attribute no code ever sets; the one test guarding this is a source-string check that passes anyway, and the real engine's backward phase is an explicit no-op. **Transports:** TCP_NODELAY and zero-copy SHM are correctly implemented in `tcp_transport.py` and `shm_transport.py` with passing real-socket tests, but neither is on the runtime data plane — tensors still flow through `EventRouter` → `P2PSocketBridge` (JSON over sockets, no Nagle disable). **Web UI:** the root route claim is fully true — `GET /` unconditionally serves `frontend/index.html` with a bundled fallback. **"No mock-only coverage" is false:** five tests assert on source strings. Worst finding: the 1F1B production branch is dead code.

## 11. Top 5 findings (worst first)

1. **[Blocker] 1F1B-1 — Production 1F1B path is dead code.** `chatgpt_api.py:2603` reads `node._pipeline_extension`, never set anywhere (one occurrence in the repo); `chatgpt_api.py:2649-2653` is therefore unreachable and multi-node training falls back to the legacy single-node path. The guarding test (`test_1f1b_training_production.py:233-240`) is source-string inspection.
2. **[High] FT-1/FT-2 — No checkpoint writes in production and staged state never consumed.** `FaultToleranceManager.save_checkpoint` has zero production callers; `CHECKPOINT_INTERVAL` unused; `get_recovery_checkpoint` unread. "State recovered from JSON checkpoint" describes plumbing production never exercises.
3. **[High] TCP-1/SHM-1 — Both transports are off the runtime data plane.** No production calls to `tcp_transport.send_tensor` or `shm_transport.send_tensor`; traffic goes over `EventRouter.publish` → `P2PSocketBridge` (JSON, no TCP_NODELAY). README describes components as the runtime.
4. **[High] 1F1B-2 — Backward phase is a documented no-op; no inter-stage activation flow.** `pytorch_hf_engine.py:622-625` returns `(None, None)` for `loss="back_gradient"`; `send_tensor_fn=noop_send`; every stage trains on raw input slices. The "1F1B" is schedule pacing, not pipeline-parallel backprop.
5. **[Medium] FT-3/FT-5 — Recovery/rebuild preconditions are unmet in a real cluster.** Disk-scan recovery needs a shared checkpoint dir (per-machine default is `~/.cache/huggingface/hub`), and rebalance triggers only after 3 blacklist misses and only after `_distributed_pipeline` has been initialized.

Also flagged: "No mock-only coverage" is false (≥5 source-string tests, 1 re-implemented loop).

## 12. Confidence: 7/10

**Ground truth (external-checked): 10 facts.** (1) `_pipeline_extension` appears exactly once in the repo — grep-verified. (2) `FaultToleranceManager.save_checkpoint` callers: tests only — grep-verified. (3) `get_recovery_checkpoint` callers: tests only — grep-verified. (4) `shm_transport.send_tensor` / `tcp_transport.send_tensor` production call sites: none — grep-verified. (5) Constants 2.0/5.0 — read from source. (6) NODELAY setsockopt at 3 sites — read from source. (7) Root route unconditional, `frontend/index.html` exists — read from source. (8) All 52 designated tests pass — executed. (9) `pytorch_hf_engine.py:622-625` back_gradient no-op — read from source. (10) `P2PSocketBridge` has no setsockopt and uses JSON serialization — read from source.

**Model judgment: 3 facts.** (a) Runtime traffic necessarily flows through `EventRouter`/bridge because that is the only cross-process relay and `forward_tensor` only publishes there — strong inference, but I did not run a live multi-process trace. (b) The 9–11 s death-to-rebalance timing and heartbeat-membership semantics — inferred from two loop cadences; I did not kill a live node. (c) Windows platform inheritance explains the patched-handler NODELAY test; on other platforms behavior may differ.

**Uncertainties:** I did not run the full 213-test suite or a live multi-process cluster, and "production wiring" in the README could charitably be read as library-level composition rather than the API call path — under the charitable reading, 1F1B-1 drops from Blocker to High, but the dead attribute remains an objective defect. Everything else is source-grounded, and I found no instance of the shipped runtime calling the two dedicated transports. Hence 7, not higher: the mechanism-level claims are solid, but the headline "wired into production" claims fail on reachability, and the two most severe failures were only visible by tracing callers, which the test suite does not do.
