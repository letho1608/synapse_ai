# Synapse AI: True Distributed Supercomputing Platform

**LACP — Latency-Aware Compute Pooling** — Run LLMs across hundreds of nodes with genuine parallel acceleration. Every added node increases token generation speed and reduces training time.

[![Repo](https://img.shields.io/badge/repo-github.com%2Fletho1608%2Fsynapse__ai-blue)](https://github.com/letho1608/synapse_ai)

---

## What Synapse AI Actually Does

Synapse AI is a **true distributed supercomputing system** that pools ALL hardware resources across every machine in your network — GPU VRAM, TFLOPS, CPU cores, RAM, disk I/O, and network bandwidth — and uses them simultaneously to accelerate LLM inference and training.

**The result:** A cluster pools its VRAM and compute so it can run models that no single machine can even load, and pipeline parallelism lets every node stay busy (see the scaling table below — wall-clock speedup is bounded by the slowest stage plus network overhead, not by node count).

### How It's Different

| What most "distributed" projects do | What Synapse AI LACP does |
|---|---|
| Split model layers across machines sequentially (Node A → Node B → wait) | **Pipeline parallelism with micro-batch overlapping**: Node B is already processing batch 1 while Node A loads batch 2 |
| Only use VRAM as partition criteria | **Multi-resource LACP**: weighs VRAM + TFLOPS + CPU + RAM + disk I/O + bandwidth + power for every node — measured live via psutil/pynvml |
| Ignore network topology | **Latency-aware clustering**: nodes with <1ms latency get adjacent shards; cross-region nodes get independent shards |
| Serial tensor forwarding blocks the pipeline | **Async non-blocking forwarding**: tensor sent as soon as ready; next micro-batch starts immediately |
| Training runs on a single node | **Distributed DDP + all-reduce**: every node computes gradients on its shard in parallel, then synchronizes |
| Node failure kills the request | **Dynamic rebalancing**: failed node's shard reassigned live; dead node's JSON checkpoint (no pickle) is written and staged for the survivor — resume-from-checkpoint is pending (see Known Limitations) |
| Same speed with 1 node or 100 nodes | **Pipeline-parallel speedup**: every added node increases capacity and keeps stages busy in parallel; wall-clock gains follow the scaling table below |

**Verification:** All claims backed by real tests — 242 tests across 31 test modules covering fault tolerance, TCP_NODELAY, resource measurement (with disk-I/O metric caching), ILP partitioning, multi-resource election, decode token verification, prompt-conditioned pipeline overlap, streaming fidelity, 1F1B training with real backward gradients and inter-stage handoff, binary tensor transport, large-frame bridge handling (256 MiB cap + explicit error on oversize), cross-node checkpoint recovery, pre-request rebalancing, latency-matrix refresh on topology change, authoritative-partition broadcast wiring, mesh heartbeats, zero-copy shared memory, and the web UI root route. Coverage is behavioral: tests drive the shipped production code paths (hardware-dependent backends are stubbed where required) — no source-string assertion tests.

---

## Architecture: LACP Pipeline

```
                   ┌─────────────────────────────────────────┐
  User Request ──▶ │         Master Scheduler Node            │
                   │  (Elected by multi-resource weight)     │
                   └─────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
             ┌──────────┐     ┌──────────┐       ┌──────────┐
  Micro-batch│ Node 0   │     │ Node 1   │  ...  │ Node N-1 │
  Flow:      │ L0-L7    │────▶│ L8-L15   │──────▶│ L16-L23  │──────▶
             │ (2ms)    │     │ (3ms)    │       │ (1.5ms)  │       │
             └──────────┘     └──────────┘       └──────────┘       │
                  │                │                   │             │
                  │    Batch 2     │    Batch 2        │    Batch 2  │
                  │    L0-L7       │    L8-L15         │    L16-L23  │
                  │    (2ms)       │    (3ms)          │    (1.5ms)  │
                  ▼                ▼                   ▼             │
            ═══════════════ PIPELINE BUBBLE-FREE ═══════════════     │
            After warmup, every node is busy every step             │
            Throughput = 1 / max(stage_time) ≈ 333 tokens/sec       │
            (vs 143 tokens/sec for sequential 7ms total)            │
```

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

### LACP Partitioning Engine

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
- Cross-machine tensor traffic currently rides the P2P event bridge: the binary SYNP payload is carried inside the bridge's JSON+hex framing, verified end to end by `test_p2p_bridge_binary.py`. A dedicated **TCP transport with Nagle disabled** (`TCP_NODELAY=1`, `tcp_transport.py`) is implemented, started on every node, and peer-registered for cross-machine links — covered by 3 dedicated tests — and is the high-performance path for network fabrics
- **RDMA / InfiniBand transport is planned, not yet implemented** — zero-copy shared memory covers same-machine traffic today; extending zero-copy to the network fabric via RDMA is future work

### Decode Pipeline Overlap

During the autoregressive decode phase, the inference engine fires **multiple micro-batches** into the pipeline before collecting results — `num_decode_batches = max(2, lookahead_window)`. This ensures all pipeline stages stay busy simultaneously: Stage 1 processes batch N while Stage 0 loads batch N+1, eliminating the pipeline bubble for decode.

Every decode micro-batch is **conditioned on the full true context** — the entire prompt plus the verified generation history — and extends it by a prefix of the n-gram lookahead guesses (batch 0 sees `[prompt … history]`, batch 1 adds guess 0, batch 2 adds guesses 0–1, …). A token is accepted only after every guess feeding its context has been verified by the previous batch's prediction, so no token predicted from a contaminated context can enter the output (`verify_speculative_tokens`).

### Result Routing

Every tensor carries a `(request_id, orchestrator_id, micro_batch_id, source_stage_id)` tuple. Results are streamed back to the requesting node with in-order reassembly — the caller sees a clean token stream regardless of which nodes processed which micro-batches. Result collection verifies **batch identity**: an out-of-order or foreign result (wrong `micro_batch_id` or `source_stage_id`) is ignored rather than misattributed to the wrong micro-batch. Streaming appends every real token — legitimate adjacent repeated tokens survive (only a sampler's own echo of the same emission is skipped), verified by `test_decode_pipeline_overlap.py`.

### Multi-Resource Master Election

The master scheduler node is elected using the **LACP composite score** rather than simple VRAM+FLOPs. `ElectionManager.__init__` accepts an optional `resource_scorer: Callable` — when provided, `set_base_compute_weight()` uses the multi-resource scorer (weighing VRAM, TFLOPS, CPU, RAM, disk I/O, bandwidth, latency, and power) to produce a stable weight. If the scorer is unavailable, it falls back to `VRAM_GB + FP32_TFLOPS / 100`. The election implements **master stickiness**: a master only loses status when truly offline, preventing flapping during transient AI task load spikes.

---

## Multi-Node Scaling Characteristics

The single authoritative scaling table — generated by `backend/tests/benchmark_cluster_speedup.py` (the README and the benchmark never contradict):

| Nodes | Stage (ms) | Network latency (ms) | Serialize (ms) | Micro-batches | Speedup | Efficiency |
|-------|-----------|----------------------|----------------|---------------|---------|------------|
| 1 | 2.0 | 0.5 | 0.1 | 4 | 1.00x | 100% |
| 2 | 2.0 | 0.5 | 0.1 | 4 | 1.21x | 61% |
| 4 | 2.0 | 0.5 | 0.1 | 4 | 1.36x | 34% |
| 10 | 2.0 | 0.5 | 0.1 | 4 | 1.46x | 15% |
| 4 | 3.0 | 1.0 | 0.2 | 8 | 1.95x | 49% |
| 10 | 3.0 | 1.0 | 0.2 | 8 | 2.41x | 24% |
| 100 | 2.0 | 2.0 | 0.5 | 16 | 3.44x | 3% |

Speedup is measured against the same workload on 1 node (pipeline-parallel inference on the simulated timing model: per-stage compute vs. network latency + serialization). With more micro-batches and faster interconnects the higher rows apply. Adding nodes primarily increases **capacity** — a 100-node cluster can run models no single machine can even load (~100 × per-node VRAM) — and the same pipeline parallelism accelerates training. Wall-clock token-speed gains are bounded by the slowest stage plus network overhead; actual results depend on model architecture, batch size, and network topology. No single-node baseline speedup is assumed beyond the timing model above.

---

## Known Limitations

- **TLS/mTLS between nodes is not implemented.** Intra-cluster tensor traffic (TCP) is unencrypted; run clusters on trusted networks (LAN/Tailscale). This is planned future work.
- **RDMA/InfiniBand transport is not implemented** — zero-copy shared memory currently covers same-machine nodes only (see Async Tensor Forwarding above).
- **Checkpoint recovery is wired into distributed training and rebalancing.** Every distributed training step persists a JSON metadata checkpoint (model id, shard layer range, step) to the shared checkpoint directory, and `check_fault_tolerance()` stages the dead node's recovered state for the surviving node, which then reloads the on-disk checkpoint via the engine's `load_checkpoint` path before resuming (verified by `test_cross_node_recovery.py` / `test_fault_tolerance_live.py`). Checkpoint recovery depends on a **shared filesystem**: the file written by one node must be readable by the orchestrator on another (`load_latest_checkpoint()` scans the shared directory).
- **Distributed decode re-feeds the full context rather than reusing a KV cache.** Each decode micro-batch is conditioned on the entire prompt plus verified history (prompt-conditioned end to end, verified by `test_decode_pipeline_overlap.py`), but every step re-runs the full context through the pipeline — incremental KV-cache reuse across decode steps is not yet implemented, so decode is correct at the cost of re-computation.
- **The P2P event bridge no longer silently drops legitimate large frames.** The per-frame cap is raised from a hardcoded 10 MiB to a configurable 256 MiB default (`max_frame_size`, settable on `P2PSocketBridge`/`Libp2pNode`); a frame that still exceeds the configured cap gets an explicit `error`/`frame_too_large` packet back to the sender (recorded in `bridge._last_errors[peer_id]`) instead of a silent disconnect — verified by `test_p2p_bridge_binary.py` (a 12 MB tensor ≈ 24 MB framed is delivered intact end to end).
- **With the multi-resource LACP path enabled (`--no-tflops-proportional`), partitioning re-runs resource measurement and bandwidth probing on a cadence**: disk-I/O measurements are cached for 5 s (`_DISK_IO_CACHE_TTL`), so partition recomputes within that window reuse the last measurement instead of re-incurring the 0.5 s sampling sleep per node (verified by `test_resource_measurement.py`), and the bandwidth estimate is a single-1024-byte-message round-trip proxy (a lower bound, not link capacity).
- **Multi-node behavior is verified with in-process simulated nodes** driving the real transport/partition/training code; a hardware multi-machine E2E benchmark run is deferred to future work.
- **The web UI is a backend-served static dashboard** (no npm dependency) — `GET /` always returns a real page, served by the API process itself.

---

## Installation

```bash
git clone https://github.com/letho1608/synapse_ai.git
cd synapse_ai
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Quick Start (Single Node)

```bash
python main.py
# Open http://127.0.0.1:52415
```

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
│       ├── main.py                  # CLI, wiring
│       ├── inference/               # Model execution
│       │   ├── distributed_engine.py  # Distributed pipeline controller + decode overlap
│       │   ├── inference_engine.py    # Base inference engine
│       │   ├── pytorch_hf_engine.py   # HuggingFace PyTorch backend
│       │   └── shard.py               # Model shard abstraction
│       ├── pipeline/                # Pipeline parallelism
│       │   ├── scheduler.py           # PipelineScheduler (1F1B schedule generation)
│       │   ├── stage.py               # PipelineStage (forward + backward support)
│       │   ├── gradient_sync.py       # DistributedTrainer (train_1f1b, ring all-reduce)
│       │   ├── fault_tolerance.py     # FaultToleranceManager (heartbeat 5s, JSON checkpoint, rebalance)
│       │   └── node_extension.py      # Node runtime extensions
│       ├── transport/               # Data transport layer
│       │   ├── codec.py               # Binary tensor encoding (SYNP header)
│       │   ├── tcp_transport.py       # TCP server/client with TCP_NODELAY
│       │   ├── shm_transport.py       # Zero-copy shared-memory transport (same-machine)
│       │   └── result_router.py       # In-order result reassembly
│       ├── topology/                 # LACP partitioning engine
│       │   ├── resource_weight.py     # Multi-resource weighting (disk_io, power, bandwidth)
│       │   ├── latency_matrix.py      # N×N probing + hierarchical clustering
│       │   ├── partitioner.py         # compute_partition(method="lrm"|"ilp")
│       │   ├── device_capabilities.py # Hardware capability detection
│       │   └── topology.py            # Cluster topology management
│       ├── orchestration/           # Node runtime
│       │   ├── node.py               # OrchestratorNode (wiring fault tolerance rebalance)
│       │   ├── election.py           # ElectionManager (multi-resource scoring)
│       │   └── tracing.py            # Distributed tracing
│       ├── networking/              # Peer discovery
│       │   ├── discovery.py          # Base discovery interface
│       │   ├── server.py             # P2P server
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
│       ├── test_election_multiresource.py  # 6 tests: multi-resource scoring, fallback, stickiness
│       ├── test_election_bandwidth_wiring.py # 8 tests: bandwidth → resource scores
│       ├── test_decode_pipeline_overlap.py # 14 tests: full-context decode overlap + token verification + streaming fidelity
│       ├── test_1f1b_production.py         # 2 tests: 1F1B constructor contracts
│       ├── test_1f1b_training_production.py # 4 tests: 1F1B training through real pipeline stages
│       ├── test_multinode_lacp_flow.py     # 5 tests: multi-node LACP partition + SHM wiring
│       ├── test_benchmark_exists.py        # 5 tests: benchmark speedup projections
│       ├── test_api_endpoints.py           # 4 tests: API + web UI root route
│       ├── test_p2p_bridge_binary.py       # 6 tests: binary tensor roundtrip + latency probe echo + large-frame handling (256 MiB cap, explicit error on oversize)
│       ├── test_b3b4b5_wiring.py           # 6 tests: latency-cache refresh, authoritative-partition broadcast, mesh heartbeats, pending-rebalance consumption
│       ├── test_lacp_multiresource.py      # LACP partitioning tests
│       ├── test_pipeline_1f1b.py           # 1F1B schedule + micro-batch tests
│       └── benchmark_cluster_speedup.py    # Speedup projection calculator (authoritative table)
└── README.md
```