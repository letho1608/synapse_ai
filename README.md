# Synapse AI: LLM inference across your own machines

## What Synapse AI Does

Synapse splits one language model across the machines on your network. Each node loads and runs only a contiguous range of layers, and hidden states flow node to node over TCP. You get two things from this:

- **Capacity:** a model that fits in no single machine runs on several. A node holds only the layers it serves, plus the embedding on the first stage and the norm and head on the last. Weights are read straight from the safetensors checkpoint for that range. Weight-only int8 quantization shrinks every layer when memory is short.
- **Throughput:** many requests run at once. Concurrent decode steps on a node share one batched forward (the weights are read once per batch), and different requests occupy different stages at the same time.

**What it does not do:** it does not make a single request faster than running the model on one machine that can hold it. Every hop adds network latency, and a pipeline over separate machines must pass each token through every stage in turn. exo, Petals and llama.cpp RPC report the same limit, and `research/related-work.md` compares Synapse with these systems in detail. Speculative drafts verified in one pass (below) recover part of that cost when the text repeats itself.

### How it works (per request)

```
 client ─▶ orchestrator (API node): tokenize, choose a stage plan, draft
             │  step = [prompt + verified history + draft]            (token ids)
             ▼
          stage 1 ─▶ stage 2 ─▶ … ─▶ last stage                (hidden states, INT8 per row)
          each stage: reuse its KV cache for the common prefix,
          recompute only the new rows, forward only those rows (+ row_offset)
             │
             ▼  last stage samples one token per draft position (verify_n = draft + 1)
          orchestrator keeps the longest verified prefix + 1 token, streams it,
          and starts the next step
```

- **Measured placement.** Each node measures what one decoder layer costs it, in ms per decode step, taken from real forwards (or from a bandwidth probe before its first forward). It also computes how many layers fit in its memory. Both values ride capability gossip. The master then runs an exact dynamic program over contiguous ranges and node choice (EdgeShard-style), which minimizes either the slowest stage including its incoming hop (`throughput`, the default) or the end-to-end token time (`latency`). A new placement replaces the current one only when it is at least 10% cheaper. If any node has not reported yet, the older TFLOPS-proportional / LACP weighting applies.
- **One-pass draft verification.** A prompt-lookup (n-gram) draft of up to `lookahead_window` tokens is appended to the context. One pipeline pass scores every draft position, and the orchestrator accepts guesses while they match what was sampled. With a deterministic draft, "sample, then accept on match" is exact speculative sampling, so greedy output is unchanged. After a rejection each stage crops its KV cache back to the common prefix. The last `verify_n` rows are always recomputed.
- **Delta rows.** Non-last stages forward only the rows they recomputed, plus `row_offset`, instead of the hidden states for the whole context on every step. A stage that lacks the cached prefix (for example, one that just replaced a dead node) answers `kv_miss`, and the orchestrator resends that step with full hidden states.
- **Failover mid-request.** While it waits for a step, the orchestrator polls stage liveness from the heartbeats. When a stage dies, the orchestrator patches the plan: it gives the dead range to a live node that already holds those layers (a replica, `SYNAPSE_REPLICAS=1`), or else to a neighbouring stage within its memory cap. It then resends the same step, and generation continues with the same tokens.
- **Every request carries its own `partition_plan`,** so every hop runs exactly the layers it was sent and forwards to the owner of the next range.

### Measured on this repository's test machine (CPU, SmolLM2-135M-Instruct)

| What | Result | Test |
|---|---|---|
| Layer-range loading, 2 × 15 layers | each node holds its own layers only (a middle stage has no embedding or head) | `test_partial_load_and_quant.py` |
| Weight-only int8 | top-1 agreement 100%, KL 0.006, decoder layers < 0.3× fp32 bytes (0.41× for the whole shard incl. the fp32 embedding) | `test_int8_weights_keep_answers_and_shrink_memory` |
| Weight-only int4 (asymmetric, group 32) | top-1 51%, KL 0.63 on this 135M model; used only when int8 cannot fit | — |
| Continuous batching, 4 / 8 concurrent requests | 12.8 → 23.3 tok/s / 12.3 → 35.0 tok/s (1.8× / 2.8×); a lone request unchanged (~15 tok/s) | `test_continuous_batching.py` (correctness), `scratchpad` benchmark |
| Distributed decode protocol vs greedy | token-identical, 2–3 stages, INT8 wire, adversarial drafts | `test_verify_protocol_engine.py` |
| 2 real processes | answer equals the single-process one | `test_live_multiprocess_split.py` |
| 3 real processes, one killed while streaming | failover patches the plan and the streamed answer equals the healthy run | `test_live_failover.py` |
| INT8 hidden states on the wire | top-1 100%, KL ≈ 5e-5, 3.6× smaller than fp32 | `test_int8_quality.py` |

These runs use separate processes on one machine. Real-network latency and bandwidth numbers need the pending multi-machine run (C1).

### Also in the box

- Pipeline-parallel training (1F1B schedule, real inter-stage gradients; training loads the whole model on each node).
- Versioned handshakes.
- Optional TLS on the bridge and on the tensor path.
- Shard-weight checkpoints.
- The CLI/TUI console.

---

## Architecture details

### 1F1B Schedule (One-Forward-One-Backward)

For training, Synapse uses the **1F1B schedule** from GPipe:
- Warmup: fill the pipeline with forward passes
- Steady state: alternate 1 forward + 1 backward per step per node
- All nodes stay busy; no idle bubbles after warmup
- Gradient all-reduce runs concurrently with the next forward pass

**Production wiring:** `DistributedTrainer.train_1f1b()` calls `PipelineScheduler.generate_1f1b_schedule()` to coordinate forward and backward passes across all pipeline stages. `PipelineStage.__init__` accepts `backward_fn: Optional[Callable]`, and its `process_loop()` dispatches `backward` operations when the schedule demands it. All verified by behavioral tests that drive the real shipped functions (see `test_1f1b_training_production.py`).

**Honest scope note:** the 1F1B schedule *pacing* is real and executes end to end (every stage receives its forward/backward slots and micro-batch slices are fed through), per-micro-batch gradients are accumulated and ring all-reduced before updates are applied, and — as of the backward-gradient work — the backward phase now produces **real gradients**: the engine's `back_gradient` computes the loss gradient w.r.t. the stage's input via local autograd (no longer a no-op returning `None`), and inter-stage handoff actually flows through the stage send path: each stage's input-gradient is delivered upstream and consumed by the previous stage as its backgrad, so gradients propagate in reverse stage order across node boundaries (verified by `test_1f1b_backward_gradient.py`, which drives the real scheduler and stages).

Still **not** implemented: per-layer shard-cut backprop across nodes — stage *i*'s backward does not consume stage *i*+1's layer activations. Each stage re-runs its own forward pass on its full model slice to compute gradients locally (no activation forwarding/caching between stages), and every stage computes its loss against the same micro-batch labels rather than a true sharded forward/backward split. The DDP-style gradient all-reduce remains the synchronization mechanism for weight updates.

---

## Key Components

### Placement (which node runs which layers)

**Default: measured placement** (`synapse/topology/dp_partition.py`, `layer_profile.py`). It applies as soon as every node has gossiped a layer profile for the model. The master solves an exact DP over contiguous ranges and node selection:
- For ≤ 4 nodes it tries every chain order; above that it builds a nearest-neighbour chain.
- It respects each node's memory cap.
- Its objective is in milliseconds, built from measured ms/layer and the measured ping RTT (halved as the one-way hop cost; 2 ms if unmeasured).
- It is checked against brute force on random small cases (`test_measured_placement.py`).
- `SYNAPSE_PARTITION_OBJECTIVE=throughput|latency`; `SYNAPSE_PARTITIONER=legacy` turns it off.

Until every profile has arrived, the legacy weighting below applies.

#### Legacy LACP weighting (fallback)

Partitioning supports two **methods**:
- **`method="lrm"`** (default) — Largest Remainder Method: fast greedy proportional allocation, ideal for large clusters
- **`method="ilp"`** — Integer Linear Programming via PuLP: binary decision variables `x_{i,l}` with exact-coverage + contiguity constraints, minimizing max latency-weighted load. Optimal but slower; use when shard balance is critical

| Resource | How It's Weighted | Measured Live |
|----------|-------------------|---------------|
| **VRAM** | Primary factor — determines max layers per node | From device capabilities |
| **TFLOPS** | Secondary — faster nodes get proportionally more layers | From device capabilities |
| **CPU cores / RAM** | For pre-processing, tokenization, KV cache offloading | From device capabilities |
| **Disk I/O** | For model weight streaming from SSD | `_measure_disk_io()` via psutil I/O counters |
| **Network bandwidth** | Determines which nodes can share adjacent shards | Measured during latency probing (1000 Mbps default) |
| **Network latency** | Hierarchical clustering — sub-1ms → same cluster; >10ms → separate pipeline stages | N×N probing matrix |
| **Power efficiency** | Low-power nodes get lighter shards to avoid thermal throttling | `_measure_power()` via pynvml GPU power / psutil battery |

### Pipeline Parallelism

```
Time ─────────────────────────────────────────────▶

Node 0: [F0,0][F1,0][F2,0][B3,0][B2,0][B1,0][F3,0][F4,0]...
Node 1:       [F0,1][F1,1][F2,1][B3,1][B2,1][B1,1][F3,1]...
Node 2:              [F0,2][F1,2][F2,2][B3,2][B2,2][B1,2]...

Warmup (3 steps) → Steady state (all nodes busy) → Cooldown (3 steps)
```

- Forward passes (Fx,y = micro-batch x, node y) and backward passes (Bx,y) overlap
- Gradient aggregation via all-reduce runs asynchronously
- No node idles after warmup for the entire training run

### Async Tensor Forwarding Protocol

```
Node produces tensor → serialize binary (numpy tobytes + SYNP compact header)
                     → non-blocking socket send (TCP_NODELAY enabled)
                     → immediately start next micro-batch
                     │
Node receives ───────▶ deserialize → queue for inference engine
                     → process when current batch finishes
                     → callback fires for result routing
```

- Tensors are encoded as **binary blobs** (not JSON) using the compact **SYNP** header format (`transport/codec.py`)
- Same-machine nodes use **zero-copy shared memory**: tensor payloads travel through `multiprocessing.shared_memory` segments (no socket, no OS copy of the payload), and the receiving node hands its handler a zero-copy numpy view over the shared buffer — implemented in `shm_transport.py` and covered by behavioral tests that assert the payload bypasses TCP
- Cross-machine tensor traffic uses the chunked binary **TCP transport** (`tcp_transport.py`, Nagle disabled, 1 MiB frames) on the port each peer advertises in its handshake. If that path fails, traffic falls back to the P2P event bridge (JSON + hex framing, 256 MiB cap). Hidden states travel as per-row INT8 to peers that negotiated `q8row1`, and only the recomputed rows are sent (`row_offset`)
- **RDMA / InfiniBand transport is planned, not yet implemented** — zero-copy shared memory covers same-machine traffic today; extending zero-copy to the network fabric via RDMA is future work

### Decode: one pipeline pass per step

Each decode step sends `[prompt + verified history + n-gram draft]` as ONE pass (`DistributedPipelineController._run_step`). The last stage samples a token at every draft position. Position j is conditioned on the history plus `draft[:j]`, with the same repetition penalty as sequential decoding. The orchestrator keeps the longest verified prefix plus one token (`verify_speculative_tokens`) and streams only those tokens. The last stage's raw samples never reach the stream. The design this replaced fired one full-context micro-batch per guess prefix, which cost k+1 passes for k guesses and had no way to cancel the stale ones. Verified token-identical to greedy decoding through 2–3 real engine stages with INT8 transport and adversarial drafts (`test_verify_protocol_engine.py`), and against an oracle last stage (`test_decode_pipeline_overlap.py`).

### Result Routing

Every tensor carries a `(request_id, orchestrator_id, micro_batch_id, source_stage_id)` tuple. Results are streamed back to the requesting node with in-order reassembly — the caller sees a clean token stream regardless of which nodes processed which micro-batches. Result collection verifies **batch identity**: an out-of-order or foreign result (wrong `micro_batch_id` or `source_stage_id`) is ignored rather than misattributed to the wrong micro-batch. Streaming appends every real token — legitimate adjacent repeated tokens survive (only a sampler's own echo of the same emission is skipped), verified by `test_decode_pipeline_overlap.py`.

### Multi-Resource Master Election

The master scheduler node is elected using the **LACP composite score** rather than simple VRAM+FLOPs. `ElectionManager.__init__` accepts an optional `resource_scorer: Callable` — when provided, `set_base_compute_weight()` uses the multi-resource scorer (weighing VRAM, TFLOPS, CPU, RAM, disk I/O, bandwidth, latency, and power) to produce a stable weight. If the scorer is unavailable, it falls back to `VRAM_GB + FP32_TFLOPS / 100`. The election implements **master stickiness**: a master only loses status when truly offline, preventing flapping during transient AI task load spikes.

---

## Multi-Node Scaling Characteristics (simulated timing model, not a measurement)

The single authoritative scaling table — generated by `backend/tests/benchmark_cluster_speedup.py`. The two never contradict: `test_b6b7_cleanup.py` asserts the table below is exactly the benchmark's output (regenerate it from `python backend/tests/benchmark_cluster_speedup.py`):

| Nodes | Stage (ms) | Network latency (ms) | Serialize (ms) | Micro-batches | Speedup | Efficiency |
|-------|-----------|----------------------|----------------|---------------|---------|------------|
| 1 | 2.0 | 0.5 | 0.1 | 4 | 1.00x | 100% |
| 2 | 2.0 | 0.5 | 0.1 | 4 | 1.21x | 61% |
| 4 | 2.0 | 0.5 | 0.1 | 4 | 1.36x | 34% |
| 8 | 2.0 | 0.5 | 0.1 | 4 | 1.44x | 18% |
| 10 | 2.0 | 0.5 | 0.1 | 4 | 1.46x | 15% |
| 4 | 3.0 | 1.0 | 0.2 | 8 | 1.95x | 49% |
| 8 | 3.0 | 1.0 | 0.2 | 8 | 2.32x | 29% |
| 10 | 3.0 | 1.0 | 0.2 | 8 | 2.41x | 24% |
| 100 | 2.0 | 2.0 | 0.5 | 16 | 3.44x | 3% |

Speedup is measured against the same workload on 1 node (pipeline-parallel inference on the simulated timing model: per-stage compute vs. network latency + serialization). With more micro-batches and faster interconnects the higher rows apply. Adding nodes primarily increases **capacity** — a 100-node cluster can run models no single machine can even load (~100 × per-node VRAM) — and the same pipeline parallelism accelerates training. Wall-clock token-speed gains are bounded by the slowest stage plus network overhead; actual results depend on model architecture, batch size, and network topology. No single-node baseline speedup is assumed beyond the timing model above.

---

## Known Limitations

- **TLS between nodes is opt-in.** `SYNAPSE_TLS=selfsigned|require` with `SYNAPSE_TLS_CERT`/`SYNAPSE_TLS_KEY` (optional `SYNAPSE_TLS_CA`) wraps both the event bridge and the chunked TCP tensor path (verified on loopback by `test_tls_interop.py` and `test_tcp_tls.py`; a plaintext client cannot inject into a TLS node); missing cert files fail loudly instead of downgrading. Default is plaintext, so run default clusters on trusted networks (LAN/Tailscale). TLS on real multi-host clusters has not been exercised yet.
- **RDMA/InfiniBand transport is not implemented** — zero-copy shared memory currently covers same-machine nodes only (see Async Tensor Forwarding above).
- **Checkpoint recovery is wired into distributed training and rebalancing.** Every distributed training step persists a JSON metadata checkpoint (model id, shard layer range, step) plus a sidecar manifest (`{node}_{model}.manifest.json`, sha256, last 10); every `SYNAPSE_CKPT_WEIGHTS_EVERY` steps (default 10, 0 = off) the node also writes its shard's layer weights (`*.weights.pt`, only its own layers) and the JSON records `weights_path`/`weights_step`. On failover the survivor restores exactly the dead node's layer range from that file into its loaded model (verified by `test_shard_weights_restore.py`: layer hashes match, other layers untouched, partial coverage refused). Weights can be up to N-1 steps older than the metadata step. Checkpoints (JSON metadata and weights) stay on the machine that wrote them, in `~/.synapse/checkpoints` or `SYNAPSE_CHECKPOINT_DIR`. Once that node is dead they are unreachable, so cross-machine restore needs `SYNAPSE_CHECKPOINT_DIR` on shared storage. Proactive replication to a peer is not implemented; the earlier "pull from the dead node" path could never work and was removed. Inference does not depend on this: any node reloads layers from the model checkpoint (see failover). Without either, the restore returns False and the rebalance is flagged degraded (`_last_recovery_degraded`) instead of silently resuming from scratch.
- **Wire transport is tiered.** The chunked binary TCP path (1 MiB frames, any blob size, 4 GiB admission, 30 s reassembly timeout) and the TCP/SHM auto-router are on by default and fall back to the JSON event bridge (256 MiB cap, explicit `frame_too_large`) on any failure; `SYNAPSE_CHUNKED_TCP=0` restores bridge-only (verified by `test_tcp_chunked_transfer.py`, incl. a 130 MiB blob). Bridge sockets set TCP_NODELAY both directions. Inter-stage hidden states travel as INT8 by default: one scale per row plus the 8 largest-magnitude values per row carried exactly, sent only to peers whose handshake advertises `q8row1` (older peers get exact dtype); `SYNAPSE_TRANSPORT_INT8=0` restores exact transport. Quality gate on the default model (SmolLM2-135M-Instruct, 2- and 3-stage splits, 6 prompts, teacher-forced): top-1 next-token agreement 100%, KL ≈ 5e-5 (`test_int8_quality.py`); payload ≈ 3.6× smaller than fp32 at hidden size 576.
- **Handshakes are versioned.** Bridges exchange `synapse_version`/`bridge_version`/`caps_hash`/`features`; major mismatch fails closed via `is_compatible()` (verified by `test_handshake_compat.py` both directions). Mixed-version pipeline refusal at the node layer and the `--version` CLI are wired; capability/demand gossip publishes advisory-only docs every ~5th heartbeat (tri-state fit, unknown rates excluded, never actuating).
- **CLI-first terminal surface exists.** `python main.py console|serve|join|client|status|topology|models|benchmark [--json] [--port N]` (verified by `test_cli.py`, `test_console.py`); `console` is the interactive TUI shell (see `UI.md`). `--log-format json` emits JSONL backend logs and `--headless` turns `GET /` into a 204 liveness stub. Launchers pass `SYNAPSE_PARENT_PID`, so the backend exits if its launcher is killed (no orphan holding ports).
- **Decode reuses a per-request KV cache on every stage.**
  - Each stage keeps a KV cache per request and shard range.
  - Full-context calls reuse the longest common prefix with the previous input. The cache is cropped after a rejected draft, and the last `verify_n` rows are always recomputed.
  - Non-last stages forward only the recomputed rows plus `row_offset` (`delta_rows`). A stage missing the prefix answers `kv_miss`, and the step is resent with full rows.
  - Callers that send only the newest token pass back the returned `inference_state` (it carries `kv_len`).
  - `SYNAPSE_SEQ_KV=0` disables full-context reuse.
  - The engine is pinned to the HF reference: full-shard logits vs `model(...)`, and 1/2/3-stage greedy decode vs `model.generate`, for tiny GPT-2 and SmolLM2 (`test_engine_reference_parity.py`). The distributed protocol is pinned to greedy decoding (`test_verify_protocol_engine.py`).
  - Sliding-window attention is not modelled: the engine applies a full causal mask.
- **The P2P event bridge no longer silently drops legitimate large frames.** The per-frame cap is raised from a hardcoded 10 MiB to a configurable 256 MiB default (`max_frame_size`, settable on `P2PSocketBridge`/`Libp2pNode`); a frame that still exceeds the configured cap gets an explicit `error`/`frame_too_large` packet back to the sender (recorded in `bridge._last_errors[peer_id]`) instead of a silent disconnect — verified by `test_p2p_bridge_binary.py` (a 12 MB tensor ≈ 24 MB framed is delivered intact end to end).
- **With the multi-resource LACP path enabled (`--no-tflops-proportional`), partitioning re-runs resource measurement and bandwidth probing on a cadence**: disk-I/O measurements are cached for 5 s (`_DISK_IO_CACHE_TTL`), so partition recomputes within that window reuse the last measurement instead of re-incurring the 0.5 s sampling sleep per node (verified by `test_resource_measurement.py`), and the bandwidth estimate is a transfer-time-based 64 KiB probe (5 samples averaged over request+echo round-trips) — a single-connection lower-bound proxy, not full link capacity (verified by `test_bandwidth_flow.py`).
- **Multi-process runs are verified on one machine, not yet across machines.**
  - `test_live_multiprocess_split.py`: 2 real backend processes answer exactly like one process, each running one contiguous layer range, with hidden states over chunked TCP.
  - `test_live_failover.py`: 3 processes. Once profiles are gossiped, measured placement runs. One stage process is killed while an answer streams; the orchestrator detects it from heartbeats, patches the plan, resends the step, and the streamed answer equals the healthy run.
  - A hardware multi-machine run (C1) is still pending, so real-network latency, bandwidth and failover times are unmeasured.
  - On Windows, check `netsh int ipv4 show excludedportrange protocol=tcp`: the default node port 50051 can fall inside a reserved range.
- **Training loads the whole model on each node.** Layer-range loading applies to inference; `train`/`evaluate` switch the engine to the full HF model.
- **A single request is not faster than one machine that can hold the model.** See "What it does not do" above.
- **No web frontend (CLI-first).** The terminal CLI/console is the user interface; `GET /` is a small built-in status page served by the API process (a 204 stub under `--headless`).

---

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `SYNAPSE_PARTIAL_LOAD` | `1` | Load only the served layer range (`0` = whole model on every node) |
| `SYNAPSE_WEIGHT_QUANT` | `auto` | `none` / `int8` / `int4` / `auto`. `auto` quantizes only when fp weights would not fit |
| `SYNAPSE_BATCH` | `1` | Continuous batching of concurrent decode steps |
| `SYNAPSE_BATCH_WINDOW_MS` | `3` | How long to wait for batch company, only while >1 request is active |
| `SYNAPSE_PARTITIONER` | `measured` | `legacy` = TFLOPS / LACP weighting only |
| `SYNAPSE_PARTITION_OBJECTIVE` | `throughput` | or `latency` |
| `SYNAPSE_REPLICAS` | `0` | `1` = also hold the next stage's layers (instant failover) |
| `SYNAPSE_TRANSPORT_INT8` | `1` | Per-row INT8 hidden states on the wire |
| `SYNAPSE_SEQ_KV` | `1` | Full-context KV reuse |
| `SYNAPSE_CHUNKED_TCP` | `1` | Chunked binary TCP tensor path |
| `SYNAPSE_TLS` | off | `selfsigned` / `require` |

## Installation

```bash
git clone https://github.com/letho1608/synapse_ai.git
cd synapse_ai
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r backend/requirements.txt
```

---

## Quick Start (Single Node)

```bash
python main.py console        # interactive TUI: boots a node + slash commands
# or headless:
python main.py serve          # then: python main.py status / client / benchmark
```

The API listens on `http://127.0.0.1:52415` (`/v1/*`); `GET /` is only a built-in status page.

---

## Running a Distributed Cluster

### Prerequisites

1. All nodes must be on the same network (LAN) or connected via Tailscale
2. Each node needs `python main.py` running with the same discovery config
3. A shared `cluster_config.json` file or Tailscale API access

### Cluster Config Example

```json
{
  "peers": {
    "node-gpu-01": {"address": "192.168.1.10", "port": 5678, "device_capabilities": {...}},
    "node-gpu-02": {"address": "192.168.1.11", "port": 5678, "device_capabilities": {...}},
    "node-cpu-03": {"address": "192.168.1.12", "port": 5678, "device_capabilities": {...}}
  }
}
```

### Running

```bash
# Each machine:
python main.py --discovery-module manual --node-id node-gpu-01 --node-port 5678
```

The cluster self-organizes:
1. **Discovery** → nodes find each other (manual config, Tailscale API, or UDP broadcast)
2. **Probing** → latency + bandwidth matrix built (N×N)
3. **Multi-resource election** → master node selected by LACP composite score (VRAM + TFLOPS + CPU + RAM + disk I/O + bandwidth + power); uses `resource_scorer` callable when available, falls back to VRAM+FLOPs
4. **LACP** → master computes optimal shard assignment via Largest Remainder Method (`method="lrm"`, fast) or ILP (`method="ilp"`, optimal via PuLP binary variables `x_{i,l}`)
5. **Pipeline** → 1F1B micro-batching begins; heartbeat monitoring (2s interval, 5s timeout = `HEARTBEAT_TIMEOUT`) checks every node; each node also publishes its fault-tolerance heartbeat over the event mesh independently of discovery/membership, and a node flagged by the monitor loop (`_pending_rebalance`, first missed heartbeat) is rebalanced by `check_fault_tolerance()` immediately instead of waiting for the 3-miss blacklist threshold; the master broadcasts the authoritative partition before serving distributed requests so every node agrees on shard assignment, and dead nodes trigger `rebalance_after_failure()` repartitioning layers among survivors with JSON checkpoint recovery (the distributed pipeline is initialized unconditionally at `node.start()`, so rebalancing works before any distributed request)
6. **API** → `/v1/chat/completions` serves requests with distributed acceleration

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat completion (OpenAI-compatible, streaming supported) |
| GET | `/v1/models` | Available models |
| POST | `/v1/models/pull` | Download model to all nodes |
| GET | `/v1/cluster/resources` | Per-node resource usage, throughput, shard assignment |
| GET | `/v1/topology` | Cluster topology (nodes, links, latency matrix) |
| GET | `/v1/distributed/status` | Distributed pipeline status |
| POST | `/v1/training/start` | Start distributed training job |

---

## Testing

```bash
# Unit tests
pytest backend/tests/ -v

# Multi-node simulation (single machine, real TCP)
python backend/tests/run_ralph_distributed_verify.py

# Real multi-machine test
python backend/tests/test_multimachine_distributed.py --config-based --port 6088 --node-id MACHINE-A --peer-host <OTHER_IP> --peer-port 6089
```

---

## Project Structure

```
synapse_ai/
├── main.py                          # Entry point
├── backend/
│   └── synapse/
│       ├── main.py                  # Backend entry, wiring
│       ├── cli.py                   # CLI subcommands (serve/status/...)
│       ├── console.py / tui.py      # Interactive console + full-screen TUI
│       ├── inference/               # Model execution
│       │   ├── distributed_engine.py  # Orchestrator: one-pass draft verify, failover
│       │   ├── partial_loader.py      # Load only a shard's layers from safetensors
│       │   ├── weight_quant.py        # Weight-only int8 / int4
│       │   ├── attn_backend.py        # Attention kernel probe (fa2 / sdpa / eager)
│       │   ├── inference_engine.py    # Base inference engine
│       │   ├── pytorch_hf_engine.py   # Shard engine: KV reuse, delta rows, batching
│       │   └── shard.py               # Model shard abstraction
│       ├── pipeline/                # Pipeline parallelism
│       │   ├── scheduler.py           # PipelineScheduler (1F1B schedule generation)
│       │   ├── stage.py               # PipelineStage (forward + backward support)
│       │   ├── gradient_sync.py       # DistributedTrainer (train_1f1b, ring all-reduce)
│       │   ├── fault_tolerance.py     # FaultToleranceManager (heartbeat 5s, JSON checkpoint, rebalance)
│       │   └── node_extension.py      # Node runtime extensions
│       ├── transport/               # Data transport layer
│       │   ├── codec.py               # Binary tensor encoding (SYNP header)
│       │   ├── router.py              # TCP/SHM/bridge route choice + quantization policy
│       │   ├── tls_wrap.py            # TLS contexts, self-signed certs, pinning
│       │   ├── tcp_transport.py       # Chunked TCP server/client (TCP_NODELAY, optional TLS)
│       │   ├── shm_transport.py       # Zero-copy shared-memory transport (same-machine)
│       │   └── result_router.py       # In-order result reassembly
│       ├── topology/                 # LACP partitioning engine
│       │   ├── resource_weight.py     # Multi-resource weighting (disk_io, power, bandwidth)
│       │   ├── latency_matrix.py      # N×N probing + hierarchical clustering
│       │   ├── partitioner.py         # compute_partition(method="lrm"|"ilp") (legacy)
│       │   ├── dp_partition.py        # Measured placement (exact DP)
│       │   ├── layer_profile.py       # ms/layer + memory cap per node
│       │   ├── device_capabilities.py # Hardware capability detection
│       │   └── topology.py            # Cluster topology management
│       ├── orchestration/           # Node runtime
│       │   ├── node.py               # OrchestratorNode (wiring fault tolerance rebalance)
│       │   ├── election.py           # ElectionManager (multi-resource scoring)
│       │   └── tracing.py            # Distributed tracing
│       ├── networking/              # Peer discovery
│       │   ├── discovery.py          # Base discovery interface
│       │   ├── manual/               # Manual discovery (JSON config)
│       │   ├── tailscale/            # Tailscale API discovery
│       │   └── udp/                  # UDP broadcast discovery
│       ├── api/                      # REST API
│       ├── routing/                  # Event router
│       └── training/                 # Finetuning (LoRA)
├── backend/
│   ├── synapse/                      # (see above)
│   └── tests/
│       ├── test_fault_tolerance_live.py    # 10 tests: heartbeat, timeout, rebalance, JSON safety, checkpoint writes
│       ├── test_cross_node_recovery.py     # 6 tests: on-disk checkpoint recovery across nodes
│       ├── test_tcp_nodelay.py             # 3 tests: server, accepted, client TCP_NODELAY
│       ├── test_shm_transport.py           # 6 tests: zero-copy shared-memory transport
│       ├── test_resource_measurement.py    # 8 tests: disk I/O (cached), power, bandwidth
│       ├── test_a7_rebalance_before_request.py # 1 test: rebalance before first distributed request
│       ├── test_partitioner_ilp.py         # 7 tests: ILP binary vars, coverage, contiguity
│       ├── test_ilp_production_flow.py     # 5 tests: ILP selectable from production path
│       ├── test_election_multiresource.py  # 7 tests: multi-resource scoring, fallback, stickiness
│       ├── test_election_bandwidth_wiring.py # 8 tests: bandwidth → resource scores
│       ├── test_decode_pipeline_overlap.py # 14 tests: full-context decode overlap + token verification + streaming fidelity
│       ├── test_1f1b_production.py         # 2 tests: 1F1B constructor contracts
│       ├── test_1f1b_training_production.py # 4 tests: 1F1B training through real pipeline stages
│       ├── test_multinode_lacp_flow.py     # 5 tests: multi-node LACP partition + SHM wiring
│       ├── test_benchmark_exists.py        # 5 tests: benchmark speedup projections
│       ├── test_api_endpoints.py           # API routes + built-in `GET /` page + distributed status
│       ├── test_p2p_bridge_binary.py       # 6 tests: binary tensor roundtrip + latency probe echo + large-frame handling (256 MiB cap, explicit error on oversize)
│       ├── test_b3b4b5_wiring.py           # 6 tests: latency-cache refresh, authoritative-partition broadcast, mesh heartbeats, pending-rebalance consumption
│       ├── test_b6b7_cleanup.py            # 4 tests: MicroBatch.step propagation, SHM active-segment bound, README↔benchmark scaling-table invariant
│       ├── test_lacp_multiresource.py      # LACP partitioning tests
│       ├── test_pipeline_1f1b.py           # 1F1B schedule + micro-batch tests
│       └── benchmark_cluster_speedup.py    # Speedup projection calculator (authoritative table)
└── README.md
```
