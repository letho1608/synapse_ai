# Phase 3 — README Truthfulness Verification

**Lane:** README vs. repo facts cross-check (document truthfulness)
**Date:** 2026-07-31
**Scope:** Every claim in `D:\Code\paper\synapse_ai\README.md` checked against (a) the recorded full-suite pytest log, (b) direct execution of the scaling benchmark, (c) grep-verified function/route/constant counts, and (d) file existence. Code behavior itself (decode/inference, LACP partition/election, fault tolerance/1F1B/transports/web UI) is the other three lanes' job; I only cross-check documentation against observable repo facts and flag anything needing their confirmation.

**Method note:** All line numbers below refer to the current README.md as read in this session. Test-function counts were produced with `grep '^\s*(async\s+)?def test_'` over `backend/tests/test_*.py` (213 matches, 28 files). The full suite log at `C:\Users\Admin\AppData\Local\Temp\grok-goal-1857e21bd6f7\implementer\pytest-full2.log` records **213 passed, 4 warnings in 164.42s** and **collected 213 items** across **28 modules** (rootdir `D:\Code\paper\synapse_ai\backend`, pytest.ini). The benchmark was re-executed live (`python tests/benchmark_cluster_speedup.py`, exit 0, "All assertions passed").

---

## 1. Scaling table — EXACT MATCH (verified by live execution)

README "Multi-Node Scaling Characteristics" (lines 137–148) vs. live output of `backend/tests/benchmark_cluster_speedup.py`:

| README row (README lines 140–146) | Benchmark output (live run) | Verdict |
|---|---|---|
| 1 / 2.0 / 0.5 / 0.1 / 4 / **1.00x** / **100%** | 1.00x / 100% | Exact |
| 2 / 2.0 / 0.5 / 0.1 / 4 / **1.21x** / **61%** | 1.21x / 61% | Exact |
| 4 / 2.0 / 0.5 / 0.1 / 4 / **1.36x** / **34%** | 1.36x / 34% | Exact |
| 10 / 2.0 / 0.5 / 0.1 / 4 / **1.46x** / **15%** | 1.46x / 15% | Exact |
| 4 / 3.0 / 1.0 / 0.2 / 8 / **1.95x** / **49%** | 1.95x / 49% | Exact |
| 10 / 3.0 / 1.0 / 0.2 / 8 / **2.41x** / **24%** | 2.41x / 24% | Exact |
| 100 / 2.0 / 2.0 / 0.5 / 16 / **3.44x** / **3%** | 3.44x / 3% | Exact |

**Result:** Every README row is byte-identical to the benchmark output. The README is a strict subset of the benchmark's 9-row table — it omits the two 8-node rows (2.0/0.5/0.1/4 → 1.44x/18% and 3.0/1.0/0.2/8 → 2.32x/29%). A subset with matching values is permitted per the verification criteria; **stated tolerance: none needed — exact match, no rounding differences.** This directly resolves the prior audit's core overclaim (old README's 2.8x/7.2x/55x vs. benchmark's 1.21x/1.36x/1.46x/3.44x): the README now reports the benchmark's honest, modest numbers.

Supporting facts:
- `benchmark_cluster_speedup.py` (lines 62–89) contains the exact config list producing these rows; `test_benchmark_exists.py` (5 tests) asserts the script exists, is importable, `compute_speedup` monotonicity/efficiency bounds, and that the table includes the 100-node row. NOTE: no test asserts README↔benchmark equality — the README's "the README and the benchmark never contradict" (line 137) is a manually-verified fact today, not an enforced invariant. Drift is possible if configs change. **Low/Note.**
- README's model description (line 145: "per-stage compute vs. network latency + serialization") matches `compute_speedup`'s formula (warmup + steady + cooldown; `step_time = max(stage_time, network_latency + serialization)`), and "No single-node baseline speedup is assumed" is consistent with the 1-node row being 1.00x by construction.

---

## 2. Test counts, totals, and paths — EXACT MATCH

### 2.1 "213 tests across 28 test modules" (README line 24–27)

- Independent grep count of `def test_` / `async def test_` across `backend/tests/test_*.py`: **213 matches in exactly 28 files**.
- Recorded pytest log: "collected 213 items", "**213 passed**"; unique modules collected = **28** (enumerated: test_1f1b_production, test_1f1b_training_production, test_api_endpoints, test_async_forwarding, test_benchmark_exists, test_cross_node_recovery, test_dataset, test_decode_pipeline_overlap, test_distributed_pipeline, test_distributed_stress, test_election_bandwidth_wiring, test_election_edge, test_election_multiresource, test_fault_tolerance_live, test_gradient_fault, test_ilp_production_flow, test_lacp, test_lacp_multiresource, test_models, test_multinode_lacp_flow, test_network_config, test_partitioner_ilp, test_pipeline_1f1b, test_resource_measurement, test_shm_transport, test_simultaneous, test_system, test_tcp_nodelay).
- The 9 remaining `test_*.py` files (test_cluster, test_local, test_three_nodes, test_multimachine_distributed, test_inprocess, test_cluster_inprocess, test_p2p_minimal, test_qwen_inference, test_3node_lacp) contain **zero** `def test_` functions — they are executable runner scripts. They are correctly excluded; "28 test modules" is honest. **Verdict: EXACT.**

### 2.2 Per-file counts in the project-structure tree (README lines 296–309)

| README claim | Actual grep count | Verdict |
|---|---|---|
| test_fault_tolerance_live.py — 9 | 9 | Exact |
| test_cross_node_recovery.py — 6 | 6 | Exact |
| test_tcp_nodelay.py — 3 | 3 | Exact |
| test_shm_transport.py — 6 | 6 | Exact |
| test_resource_measurement.py — 6 | 6 | Exact |
| test_partitioner_ilp.py — 7 | 7 | Exact |
| test_ilp_production_flow.py — 4 | 4 | Exact |
| test_election_multiresource.py — 6 | 6 | Exact |
| test_election_bandwidth_wiring.py — 8 | 8 | Exact |
| test_decode_pipeline_overlap.py — 10 | 10 | Exact |
| test_1f1b_production.py — 2 | 2 | Exact |
| test_1f1b_training_production.py — 4 | 4 | Exact |
| test_multinode_lacp_flow.py — 5 | 5 | Exact |
| test_benchmark_exists.py — 5 | 5 | Exact |
| test_api_endpoints.py — 4 | 4 | Exact |

All 15 counted files match. The two uncounted files (test_lacp_multiresource.py = 19, test_pipeline_1f1b.py = 13) have no stated numbers to contradict.

### 2.3 Project-structure tree paths (README lines 250–311)

All tree paths exist on disk (verified via directory listing): `backend/synapse/main.py`, `inference/{distributed_engine,inference_engine,pytorch_hf_engine,shard}.py`, `pipeline/{scheduler,stage,gradient_sync,fault_tolerance,node_extension}.py`, `transport/{codec,tcp_transport,shm_transport,result_router}.py`, `topology/{resource_weight,latency_matrix,partitioner,device_capabilities,topology}.py`, `orchestration/{node,election,tracing}.py`, `networking/{discovery,server}.py`, `networking/{manual,tailscale,udp}/`, `api/`, `routing/`, `training/`, `backend/tests/`, `frontend/`. **Verdict: EXACT.** The prior audit's "tests/ vs backend/tests/" path error is fully fixed — every path in the README now uses `backend/tests/`.

---

## 3. Security / RDMA posture — HONEST (no residual implication)

- README line 118: "**RDMA / InfiniBand transport is planned, not yet implemented**" — correct and explicit.
- README line 154: "**TLS/mTLS between nodes is not implemented.** Intra-cluster tensor traffic (TCP) is unencrypted" — correct; grep found no `ssl`/TLS path in `tcp_transport.py` (plain `socket` connections with `setsockopt(TCP_NODELAY)` only).
- README line 155: "**RDMA/InfiniBand transport is not implemented** — zero-copy shared memory currently covers same-machine nodes only" — correct and cross-consistent with line 118.
- README line 156: "**Multi-node behavior is verified with in-process simulated nodes** driving the real transport/partition/training code; a hardware multi-machine E2E benchmark run is deferred to future work." — **honest.** No fake hardware claims anywhere. The scaling table is explicitly labeled a "simulated timing model" (line 145). The multi-node tests are single-machine (loopback `127.0.0.1`, in-process nodes) and `run_ralph_distributed_verify.py` launches two subprocesses on one machine over real TCP — matching the description.
- README line 158: web UI "backend-served static dashboard (no npm dependency)" — frontend/ contains only `index.html` (no package.json), and `chatgpt_api.py:393` registers `GET /` unconditionally with a bundled-fallback page (see §5).

**Verdict: the security/RDMA posture is now honest.** No wording implies RDMA or TLS exists. This resolves the prior audit's "Optional RDMA support" overclaim.

---

## 4. "LACP 3.0" — NOT in the README; legacy test names only

- `grep -i 'LACP 3.0|LACP3'` across the repo: hits only in `backend/tests/test_lacp.py` — test functions `test_lacp3_cluster` (line 49) and `test_lacp3_full_integration` (line 350) plus their `__main__` references. **The README never uses "LACP 3.0"/"LACP3".**
- `hf_models.json` "v3.0" hits: `speakleash/Bielik-4.5B-v3.0-Instruct` (line 3631) and `speakleash/Bielik-11B-v3.0-Instruct` (line 5679) — HuggingFace model names, unrelated to LACP, as expected.
- **Finding (Note):** two test functions in `test_lacp.py` still carry the old "lacp3" branding. Cosmetic residue from the pre-rewrite era; it neither supports nor contradicts any current README claim, but the names are stale and could confuse future readers into thinking "LACP 3.0" is a current concept.

---

## 5. Other README claims vs. repo facts

| README claim (line) | Repo fact | Verdict |
|---|---|---|
| Heartbeat "2s interval, 5s timeout = HEARTBEAT_TIMEOUT" (215) | `fault_tolerance.py:92–93`: `HEARTBEAT_INTERVAL = 2.0`, `HEARTBEAT_TIMEOUT = 5.0`; monitor loop (line 350/355) sleeps INTERVAL and calls `check_timeout(HEARTBEAT_TIMEOUT)`. `test_fault_tolerance_live.py` asserts both via `test_heartbeat_timeout_matches_readme`/`test_heartbeat_interval_constant` (2 of its 9 tests). | Exact |
| "dead nodes trigger check_timeout() … rebalance_after_failure() … JSON checkpoint recovery" (216) | `check_timeout` (fault_tolerance.py:58), `rebalance_after_failure` (line 269) which loads `load_latest_checkpoint` from disk for the dead node; checkpoints serialized with `json.dump` (`_json_serialize`, line 198) — no pickle. | Exact |
| TCP_NODELAY "on both server listening socket and all outbound client connections, confirmed in tcp_transport.py with 3 dedicated tests" (119) | `tcp_transport.py` sets `TCP_NODELAY` at 3 sites (lines 56, 77, 146) = server socket, accepted-connection handler, client pool. `test_tcp_nodelay.py` has exactly 3 tests (server/accepted/client). | Exact |
| Zero-copy SHM "multiprocessing.shared_memory segments … zero-copy numpy view … implemented in shm_transport.py … tests assert the payload bypasses TCP" (116–117) | `shm_transport.py` exists and implements shared-memory segments with a zero-copy numpy receive view; test module docstring and `test_zero_copy_delivery_between_same_process_peers`/`test_receive_is_zero_copy_view_over_shared_buffer` assert SHM segment usage and buffer views. (Deep behavioral verification = transport lane.) | Supported |
| "GET / always returns a real page, served by the API process itself" (158) | `chatgpt_api.py:393` registers `add_get("/", handle_root)` unconditionally; bundled-fallback page when `frontend/index.html` absent (handle_root, ~line 537); `test_api_endpoints.py` `test_root_route_serves_real_page` (real HTTP GET via TestClient → 200, text/html, `<html`) and `test_root_falls_back_to_bundled_page_when_frontend_missing` both pass in the recorded run. | Exact |
| Election: "ElectionManager.__init__ accepts an optional resource_scorer … falls back to VRAM_GB + FP32_TFLOPS / 100 … master stickiness" (131–133) | `election.py:22` `resource_scorer: Optional[Callable] = None`; `set_base_compute_weight` (line 41) uses scorer when present else VRAM+FLOPs fallback (lines 44–48); stickiness at line 128. `test_election_multiresource.py` (6) and `test_election_bandwidth_wiring.py` (8) cover scorer, fallback, and bandwidth wiring. | Exact |
| 1F1B production wiring "DistributedTrainer.train_1f1b() calls PipelineScheduler.generate_1f1b_schedule() … PipelineStage.__init__ accepts backward_fn … process_loop() dispatches backward … verified by behavioral tests" (65–68) | `gradient_sync.py:163` `train_1f1b`; `scheduler.py:86` `generate_1f1b_schedule`; `stage.py:68` `backward_fn` param and line 123 dispatch of `"backward"` op. Sentence now reads "All verified by behavioral tests that drive the real shipped functions (see test_1f1b_training_production.py)" — the phrase "production-source inspection" is gone (grep: no "inspection" anywhere in README). `test_1f1b_training_production.py` (4 tests, 2 using monkeypatch around real scheduler/stage/trainer objects) passed in the recorded run. | Exact |
| "method='ilp' … use for small clusters (≤32 nodes)" (76); "master computes … via … ILP (method='ilp', optimal via PuLP binary variables x_{i,l})" (213) | PuLP ILP solver exists (`partitioner.py:98` `compute_partition_ilp`, binary vars `x_{i,l}`, coverage + contiguity + minimax, LRM fallback without PuLP). Production wiring: `node.py:121` `partition_method` knob (default "lrm") is passed at both `compute_partition` call sites (lines 460–465, 1588–1593) — ILP is selectable, and `test_ilp_production_flow.py` (4) exercises method="ilp" through the production path. **However, the "≤32 nodes" bound has no code/test basis** (see Finding 3). | Partial — see Finding 3 |
| "213 tests across 28 test modules covering fault tolerance, TCP_NODELAY, resource measurement, ILP partitioning, multi-resource election, decode token verification, pipeline overlap, 1F1B training, cross-node checkpoint recovery, zero-copy shared memory, and the web UI root route. No mock-only coverage." (24–27) | All 11 coverage areas map to real test modules (see §2.1). "No mock-only coverage" is fair: mock/monkeypatch usage exists in only 4 of 28 modules (test_1f1b_training_production: 2, test_election_multiresource: 2, test_api_endpoints: MockNode fixtures, test_decode_pipeline_overlap: mock engine), always driving real shipped code around mocked seams — the suite is not mock-only. | Fair (see Note in §7) |
| Async forwarding "serialize binary (numpy tobytes + msgpack header)" (diagram, 104); "Tensors are encoded as binary blobs (not JSON) using a compact header format" (113) | Prose is correct: `codec.py` = "Custom binary header + numpy raw bytes (no JSON, no tolist())", `MAGIC = b"SYNP"`, 16-byte struct header (magic+dtype+ndim+reserved). **But there is no msgpack in the tensor codec** — see Finding 4. | Finding 4 |
| "Every tensor carries a (request_id, orchestrator_id, micro_batch_id, sequence_position) tuple" (126) | `ResultMessage` dataclass fields: request_id, orchestrator_id, micro_batch_id, tokens, is_finished. `sequence_position` has **zero** occurrences in `backend/` (grep). In-order reassembly itself is real (`result_router.py`, tested by 16 `test_distributed_pipeline.py` reassembly tests). | Finding 2 |
| Quick Start: `python main.py` + `http://127.0.0.1:52415` (175–177); cluster run `python main.py --discovery-module manual --node-id node-gpu-01 --node-port 5678` (205) | Root `main.py` forwards unknown args to `backend/synapse/main.py`, which accepts `--discovery-module` (choices udp/tailscale/manual), `--node-id`, `--node-port` (lines 59–67). Port 52415 is the default API port (`chatgpt_api.py:262,3023`; `main.py:71`). Note: root launcher also attempts to spawn an npm frontend dev server unless `--no-frontend` (see §7). | Valid |
| Testing section: `pytest backend/tests/ -v`; `python backend/tests/run_ralph_distributed_verify.py`; `python backend/tests/test_multimachine_distributed.py --config-based --port 6088 --node-id MACHINE-A --peer-host <OTHER_IP> --peer-port 6089` (236–243) | All paths exist. `run_ralph_distributed_verify.py` is a real 2-subprocess-over-real-TCP verifier. `test_multimachine_distributed.py` argparse accepts `--config-based`, `--port`, `--node-id`, `--peer-host`, `--peer-port` (lines 369–376). | Exact |

---

## 6. Findings — severity-ordered

### Finding 1 — HIGH: README API table documents two endpoints that do not exist
**README lines 226–227:** `GET /v1/cluster/status` ("Per-node resource usage, throughput, shard assignment") and `GET /v1/cluster/latency` ("N×N latency matrix").

**Repo fact:** Neither route is registered anywhere. `chatgpt_api.py` registers `/v1/cluster/resources` (line 387), `/v1/distributed/status` (line 353), and `/v1/topology` (line 351) — the functionality the README describes lives at those paths, not at the documented ones. A client following the README gets a 404. This exact mismatch was flagged by the *previous* audit (`research/synapse-readme-audit/07-executive-briefing.md:129`; `04-tests-resiliency.md:144–145`; `05-synthesis.md:135`) and **survived the README rewrite unchanged** — it is the single most prominent residual overclaim. Fix is trivial (rename rows to `/v1/cluster/resources` and `/v1/topology`, or register the two documented paths).

### Finding 2 — MEDIUM: `sequence_position` tuple field is invented
**README line 126:** "Every tensor carries a `(request_id, orchestrator_id, micro_batch_id, sequence_position)` tuple."

**Repo fact:** `sequence_position` appears nowhere in `backend/` (grep: 0 hits; only README.md and the research/audit docs mention it). The code's routing message carries `request_id`, `orchestrator_id`, `micro_batch_id` (plus `tokens`/`is_finished`); the prior audit determined the functionally equivalent field is `source_stage_id` (`01-core-architecture.md:365`). The substance of the claim — in-order reassembly, clean token stream regardless of micro-batch ordering — is real and tested (result_router.py + 16 distributed_pipeline reassembly tests), so this is a naming fabrication rather than a functional falsehood, but the documented tuple field does not exist in code.

### Finding 3 — MEDIUM: "≤32 nodes" ILP guidance is unbacked and likely optimistic
**README line 76:** "Optimal but slower; use for small clusters (≤32 nodes)…"

**Repo fact:** No code, constant, or test in the repo establishes a 32-node limit for ILP. It is README-authored guidance. Worse, the prior audit's ILP lane computed that the `y→z` ordering link in `compute_partition_ilp` generates **O(N²·L²) constraints ≈ 3.1M for 32 nodes × 80 layers** (`10-ilp-partitioning-verification.md:264,525`), making the documented maximum impractically slow to solve — so the bound as stated implies a usable range the implementation likely cannot reach. Needs the partitioning/ILP lane to confirm the current solver's practical ceiling after the wiring fix; if confirmed, the README's "≤32 nodes" is misleading rather than merely unverified.

### Finding 4 — LOW: "msgpack header" in the Async Tensor Forwarding diagram is a mislabel
**README line 104 (diagram):** "serialize binary (numpy tobytes + msgpack header)".

**Repo fact:** The tensor codec (`transport/codec.py`) uses a custom 16-byte struct header (`MAGIC = b"SYNP"`, magic+dtype+ndim+reserved) — no msgpack. msgpack is used only in `orchestration/node.py:291–301` for node state-sync payloads, a different path. The README's prose (line 113, "binary blobs (not JSON) using a compact header format") is correct; only the diagram's parenthetical is wrong. Relatedly, `msgpack` is **not in `requirements.txt`** (it is currently installed in the environment, but grep shows no requirements entry and it is not a transitive dep of aiohttp/aiohttp-cors), so a fresh `pip install -r requirements.txt` may lack it and break node.py's state-sync import. The Installation section's `pip install -r requirements.txt` claim is therefore only partially supported.

### Finding 5 — LOW: "No mock-only coverage" is a fair but imprecise absolute
**README line 27:** "No mock-only coverage."

**Repo fact:** True in substance — 4 of 28 modules use `monkeypatch`/`MagicMock` (test_1f1b_training_production 2, test_election_multiresource 2, test_api_endpoints MockNode, test_decode_pipeline_overlap mock engine), always exercising real shipped code around mocked seams; the other 24 modules are mock-free. But the sentence reads as "no mocks anywhere," which is not literally accurate. Recommend "no mock-only coverage" → "no module relies on mocks alone" for precision.

### Finding 6 — NOTE: "Run LLMs across hundreds of nodes" is marketing unbacked by the repo
**README line 5.** The scaling table tops out at 100 nodes (3.44x/3%); the repo has no simulation, benchmark, or test beyond 100 nodes. "Hundreds" is aspirational. Not contradicted (nothing says hundreds are *verified*), but unverifiable from repo facts.

### Finding 7 — NOTE: root `main.py` still attempts to launch an npm dev server
**README line 158** says "no npm dependency" for the web UI — true (frontend/ = index.html only; API serves it). But root `main.py:70–83` tries `npm install`/`npm run dev` unless `--no-frontend` is passed; with no package.json present this path prints an error (caught) and continues. The claim itself is accurate; the launcher behavior is inconsistent with it. `test_api_endpoints.py` covers the API-served page, so the *web UI* is genuinely npm-free.

### Finding 8 — NOTE: legacy `test_lacp3_*` test names
`test_lacp.py` still names `test_lacp3_cluster` and `test_lacp3_full_integration` from the pre-rewrite "LACP 3.0" branding. No current README claim depends on this; purely cosmetic/stale naming.

### Finding 9 — NOTE: benchmark↔README equality is not enforced
README line 137 asserts the README and benchmark "never contradict." True today (verified by execution), but `test_benchmark_exists.py` does not diff the README table against `benchmark_table()`. The invariant could silently drift; a test asserting the README rows against the benchmark output would make the claim machine-checkable.

---

## 7. What is still false or unverified

**Still false (confirmed contradictions):**
1. `/v1/cluster/status` and `/v1/cluster/latency` documented as API endpoints but not registered (Finding 1 — the only outright false *feature* claim remaining).
2. `sequence_position` named as a routing tuple field that does not exist in code (Finding 2 — naming-level falsehood; function is real under a different field).
3. "msgpack header" attributed to tensor serialization; the tensor codec uses a custom struct header (Finding 4 — technical mislabel).

**Unverified / at risk (need other lanes):**
4. ILP "≤32 nodes" practicality — prior audit's Lane 3 documented O(N²·L²) ≈ 3.1M constraints at exactly 32 nodes; current practical ceiling after the production-wiring fix needs the ILP/partitioning lane's confirmation (Finding 3).
5. "No OS copy" nuance of the zero-copy SHM claim — sender performs a single memcpy into shared memory (per shm_transport docstring); "no socket, no OS copy of the payload" is consistent, but the transport lane should confirm the receive path hands out a true zero-copy numpy view in all delivery modes.
6. "Master scheduler node … elected by multi-resource weight" (line 32 diagram label) and the LACP composite score wiring — structurally present (election.py + test_election_*); behavioral correctness is the election lane's scope.
7. "Distributed DDP + all-reduce" training claim (line 22) — `gradient_sync.py` ring all-reduce exists and is tested, but distributed-training fidelity is the fault-tolerance/1F1B lane's scope.
8. Absence of TLS — no `ssl` usage in transport code (grep), but the transport lane should confirm no encrypted path exists elsewhere (e.g., grpc).
9. "hundreds of nodes" (line 5) — marketing, not falsifiable from repo facts.

**Explicitly NOT false anymore** (fixed by the rewrite, verified here): the inflated 2.8x/7.2x/55x scaling table (now exact benchmark values); inflated test counts ("12" vs 9, "45 across 7 packages" vs 213/28); wrong `tests/` paths (now `backend/tests/`); "numpy memmap" zero-copy claim with no implementation (now a real `shm_transport.py`); "Optional RDMA support" implying existence (now explicitly planned-not-implemented); "verified via production-source inspection" (now "verified by behavioral tests").

---

## 8. Executive summary (250 words)

The rewritten README is dramatically more honest than the version the prior audit reviewed, and the headline numbers check out. I verified every claim in the README against repo facts: (a) I re-ran `benchmark_cluster_speedup.py` and matched all 7 README scaling-table rows byte-for-byte (1.00x/1.21x/1.36x/1.46x/1.95x/2.41x/3.44x — the README omits the two 8-node rows, an allowed exact subset); (b) I independently counted 213 test functions in exactly 28 `test_*.py` files, and the recorded full-suite log confirms "213 passed"; all 15 per-file counts in the README's tree are exact, and every listed path exists; (c) RDMA/InfiniBand and TLS/mTLS are now honestly stated as not implemented, and multi-node behavior is correctly described as verified on in-process simulated nodes — no fake-hardware claims remain; (d) heartbeat 2s/5s, the 3 TCP_NODELAY tests, `shm_transport.py`, the unconditional `GET /`, election `resource_scorer` + fallback + stickiness, the 1F1B behavioral-test wording, and all three Testing-section commands are backed by real code and passing tests. Residual dishonesty is limited to three items: the API table still documents `/v1/cluster/status` and `/v1/cluster/latency`, which are not registered routes (previously flagged and not fixed — the worst remaining issue); the `sequence_position` tuple field does not exist in code; and the diagram's "msgpack header" mislabels the custom SYNP struct codec (msgpack is also missing from requirements.txt). The "≤32 nodes" ILP bound is unbacked and likely optimistic. Overall: README is now 90%+ accurate; the fixes needed are small and mechanical.

---

## 9. Top 5 findings (worst first)

1. **HIGH — Phantom API endpoints (README 226–227):** `/v1/cluster/status` and `/v1/cluster/latency` are documented but never registered; real equivalents are `/v1/cluster/resources` and `/v1/topology`. Previously flagged by audit 07, not fixed.
2. **MEDIUM — `sequence_position` (README 126):** the 4th routing tuple field doesn't exist in code (functionally `source_stage_id`); in-order reassembly itself is real and tested.
3. **MEDIUM — ILP "≤32 nodes" (README 76):** no code/test basis; prior Lane-3 analysis puts ~3.1M constraints at exactly 32 nodes, so the documented ceiling likely exceeds what the solver can handle — needs ILP-lane confirmation.
4. **LOW — "msgpack header" (README 104):** tensor codec uses a custom SYNP struct header, not msgpack; msgpack only in node state-sync and missing from requirements.txt.
5. **LOW — "No mock-only coverage" (README 27):** fair in substance but 4 of 28 modules do use monkeypatch/MagicMock; phrasing should be "no module relies on mocks alone."

---

## 10. Confidence

**Confidence: 9/10.**

Rationale: the three highest-stakes numeric claims (scaling table, 213/28 test total, per-file counts) were verified by direct execution or independent re-count rather than trusted from the README or the implementer's summary, and all three are exact. The recorded pytest log (`pytest-full2.log`) was cross-checked for both the total ("213 passed") and the module count (28), matching my independent grep. Path existence and route registrations were verified by directory listing and source grep, respectively — the two phantom endpoints, the absent `sequence_position`, and the msgpack mislabel are each confirmed by exhaustive (repo-wide) grep, not by sampling. Deductions from 10: (a) I did not re-run the full 2:44 suite myself — I relied on the recorded log (mitigated: I independently reproduced the function counts and spot-read the passing test files); (b) "no mock-only coverage" and the ILP "≤32 nodes" judgment involve qualitative assessment — the latter explicitly deferred to the ILP lane; (c) the "no OS copy" SHM nuance and TLS-absence are code-behavior claims I only partially substantiated, and the transport lane should confirm. No finding of mine rests on a claim I could not directly observe in the repository or its logs.
