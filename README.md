# Synapse AI: LLM inference across your own machines

## What Synapse AI Does

Synapse splits one language model across the machines on your network. Each node loads and runs only a contiguous range of layers, and hidden states flow node to node over TCP. You get two things from this:

- **Capacity:** a model that fits in no single machine runs on several. A node holds only the layers it serves, plus the embedding on the first stage and the norm and head on the last. Weights are read straight from the safetensors checkpoint for that range, and for a sharded checkpoint a node downloads only the files that hold its tensors (a new range after a re-plan fetches just what is missing). A single-file checkpoint that is not cached is read with HTTP range requests: the header, then only the bytes of the node's tensors. Weight-only int8 / int4 quantization shrinks every layer when memory is short (int4 is mixed precision: the tensors it hurts most stay int8).
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
| Weight-only int4 (group 32, MSE-fitted scale/zero; v_proj, down_proj and the 2 end layers each side in int8) | top-1 74%, KL 0.066, decoder layers 0.19× fp32 bytes (before: top-1 51%, KL 0.63); used only when int8 cannot fit | `test_int4_mixed_precision_keeps_answers` |
| Pipeline-parallel training, 3 nodes | one step equals the full-model step parameter by parameter (< 1e-5), each node holds only its layers, tied embedding copies stay identical | `test_pipeline_training.py` |
| Continuous batching, 4 / 8 concurrent requests | 12.8 → 23.3 tok/s / 12.3 → 35.0 tok/s (1.8× / 2.8×); a lone request unchanged (~15 tok/s) | `test_continuous_batching.py` (correctness), `scratchpad` benchmark |
| Distributed decode protocol vs greedy | token-identical, 2–3 stages, INT8 wire, adversarial drafts | `test_verify_protocol_engine.py` |
| 2 real processes | answer equals the single-process one | `test_live_multiprocess_split.py` |
| 3 real processes, one killed while streaming | failover patches the plan and the streamed answer equals the healthy run | `test_live_failover.py` |
| INT8 hidden states on the wire | top-1 100%, KL ≈ 5e-5, 3.6× smaller than fp32 | `test_int8_quality.py` |

These runs use separate processes on one machine. Real-network latency and bandwidth numbers need the pending multi-machine run (C1).

### Also in the box

- Pipeline-parallel training: each node trains only the layers it serves; activations go forward and gradients go back between nodes. LoRA works the same way (adapters on each node's own layers, also over int8/int4 weights).
- Pairing: one code per cluster (Settings page of the web UI); machines that share it talk mutual TLS, machines without it cannot connect.
- Autoscale: under load the master re-places layers for throughput (spreading them over idle nodes), and back when idle.
- Versioned handshakes.
- Shard-weight checkpoints, replicated to a peer.
- A web UI served by every node at `http://<node>:52415`: dashboard, nodes and layer split, chat with measured TTFT / tok/s, models, training (full or LoRA), datasets, logs, settings and pairing. Any OpenAI-compatible chat app also works against the API (see "Chat UI").

---

## Architecture details

### Pipeline-parallel training (each node trains its own layers)

`pipeline/pp_training.py` (protocol) + `inference/stage_training.py` (per-stage work). A node trains exactly the layer range it serves for inference — the same weights, updated in place — so training needs no more memory per node than serving (plus activations, gradients and AdamW state for that range).

One step for a plan `[(node, shard), …]`:
- The orchestrator splits the batch into micro-batches and sends token ids (+ labels) to stage 0, back to back, so stage 0 works on micro-batch m+1 while later stages work on m.
- Each stage runs its layers with autograd, keeps the graph, and sends the hidden states (fp32, no wire quantization) to the next stage.
- The last stage computes the loss, backprops, and sends d loss / d input upstream; every stage backprops its own graph and passes its input gradient on.
- Gradient norm is computed globally (every stage reports its share) and clipped; every stage then applies AdamW to its own parameters. Parameters are disjoint, so there is no gradient all-reduce, except for tied input/output embeddings split across the first and last stage: the two copies exchange gradients before the step and stay identical.
- Every stage checkpoints its layers after the step (`SYNAPSE_CKPT_WEIGHTS_EVERY`).

Messages go over the chunked TCP tensor path, falling back to the event bridge. `train_step_distributed(schedule="1f1b")` (used by the training API for multi-node jobs) runs this; `schedule="ddp"` is data parallelism and keeps a full replica on every node by definition. Verified against the full HF model: one 3-stage step equals one full-model step (same micro-batch averaging, clipping and AdamW) to < 1e-5 per parameter on SmolLM2-135M and tiny-GPT2 (`test_pipeline_training.py`, which also drives two real `Node`s through their transport handlers).

**LoRA** (`training/stage_lora.py`, `use_lora` in the training API) uses the same pipeline: every stage wraps the target linears of its own layers (`q_proj,k_proj,v_proj,o_proj` by default, GPT-2's `c_attn`) and trains only the adapters; base weights and embeddings stay frozen. The settings travel with each step, so remote stages set up the same adapters. A 3-stage LoRA step equals a 1-node step on the same adapters, and LoRA over int8 weights trains (QLoRA-style: the gradient flows through the frozen quantized layers on the torch path) (`test_stage_lora.py`). Adapters are part of the layers' checkpoint, so a survivor restores them; `stage_lora_merge` folds them into the weights. Full fine-tuning of quantized weights is refused (train LoRA instead or set `SYNAPSE_WEIGHT_QUANT=none`).

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

Each node runs a contiguous layer range. Inference: a step's hidden states go node to node along the partition and the last stage returns the result. Training: micro-batches go forward back to back, gradients come back the same way, and each node updates only its own layers (see "Pipeline-parallel training" above).

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

## Known Limitations

- **TLS between nodes: pairing code, or manual certificates.**
  - Settings → "Ghép máy" creates a code (160 random bits) on one machine; paste it on every other machine (`POST /v1/cluster/pairing`). Each machine derives the same cluster CA key from the code, issues its own node certificate, and runs mutual TLS on the event bridge and the tensor path. A node with another code, or none, cannot handshake in either direction (`test_tls_pairing.py`). With a saved code TLS is on by default after a restart (`SYNAPSE_TLS=off` overrides); "Bỏ ghép" removes it.
  - Manual mode: `SYNAPSE_TLS=selfsigned|require` wraps both the event bridge and the chunked TCP tensor path.
  - Peers are authenticated by a CA (`SYNAPSE_TLS_CA`) or by sha256 certificate pins (`SYNAPSE_TLS_PINS`). With pins, every client connection checks the server certificate against them, on the bridge and on the tensor path (`test_tls_interop.py`).
  - Without a CA, the old code encrypted to whoever answered: the advertised `tls_fp` was never checked.
  - `require` refuses to start without a CA or pins.
  - `selfsigned` generates a per-node cert in `~/.synapse/tls` when `SYNAPSE_TLS_CERT`/`KEY` are not given. Each node logs its pin for the peers' `SYNAPSE_TLS_PINS`.
  - Missing files fail loudly; nothing falls back to plaintext.
  - With a CA, TLS is mutual: servers demand a CA-signed client certificate and clients present theirs (`test_ca_mode_is_mutual_tls`). Pins-only clusters authenticate the server side only: the stdlib cannot demand a client certificate while skipping chain verification.
  - Without a pairing code the default is plaintext, so run unpaired clusters on trusted networks (LAN/Tailscale).
  - TLS has not been exercised on real multi-host clusters yet.
- **API access.** Chat and read-only status routes are open to the network; admin routes (`/quit`, settings, model pull/delete/merge, datasets, training, logs) answer only to the local machine. With `SYNAPSE_API_KEY` set, every route except `GET /` and `/healthcheck` needs `Authorization: Bearer <key>` (`test_api_access.py`).
- **RDMA/InfiniBand transport is not implemented** — zero-copy shared memory currently covers same-machine nodes only (see Async Tensor Forwarding above).
- **Checkpoint recovery is wired into distributed training and rebalancing, and survives the loss of the writing machine.**
  - Every distributed training step writes a JSON metadata checkpoint plus a sha256 manifest.
  - Every `SYNAPSE_CKPT_WEIGHTS_EVERY` steps (default 10) it also writes the shard's layer weights.
  - Each new checkpoint (JSON + weights) is replicated to the node's pipeline successor, over chunked TCP when the peer advertises it, else over the bridge (`SYNAPSE_CKPT_REPLICATE=0` turns this off).
  - The receiver verifies both hashes, confines the writes to its checkpoint dir, and stores the replica under the writer's name. After the writer dies, `load_latest_checkpoint(dead)` on the survivor finds it and the survivor restores that layer range (`test_wiring_compat_replica.py`, `test_shard_weights_restore.py`).
  - Checkpoints live in `~/.synapse/checkpoints` or `SYNAPSE_CHECKPOINT_DIR`.
  - Weights can be up to N-1 steps older than the metadata step.
- **Wire transport is tiered.** The chunked binary TCP path (1 MiB frames, any blob size, 4 GiB admission, 30 s reassembly timeout) and the TCP/SHM auto-router are on by default and fall back to the JSON event bridge (256 MiB cap, explicit `frame_too_large`) on any failure; `SYNAPSE_CHUNKED_TCP=0` restores bridge-only (verified by `test_tcp_chunked_transfer.py`, incl. a 130 MiB blob). Bridge sockets set TCP_NODELAY both directions. Inter-stage hidden states travel as INT8 by default: one scale per row plus the 8 largest-magnitude values per row carried exactly, sent only to peers whose handshake advertises `q8row1` (older peers get exact dtype); `SYNAPSE_TRANSPORT_INT8=0` restores exact transport. Quality gate on the default model (SmolLM2-135M-Instruct, 2- and 3-stage splits, 6 prompts, teacher-forced): top-1 next-token agreement 100%, KL ≈ 5e-5 (`test_int8_quality.py`); payload ≈ 3.6× smaller than fp32 at hidden size 576.
- **Handshakes are versioned and enforced.** Bridges exchange `synapse_version`, `bridge_version`, `caps_hash` and `features`.
  - A peer whose bridge major version differs from ours is excluded from placement, from failover plans and from checkpoint replication (`_peer_compatible` → `bridge.is_compatible`; `test_wiring_compat_replica.py`, `test_handshake_compat.py`).
  - Capability/demand gossip also carries layer profiles and loaded layers.
- **The web UI is the interface** (`frontend/`, Vite + TypeScript, built into `frontend/dist` and served by the node at `GET /`). It shows only real data from the API: a failed call shows the error, never placeholder numbers. Admin actions follow the API access policy (from the node's own machine, or with the API key entered on the Settings page). The terminal console/CLI was removed; `python main.py` takes the backend flags (`--log-format json` for JSONL logs, `--headless` turns `GET /` into a 204 stub). The launcher passes `SYNAPSE_PARENT_PID`, so the backend exits if its launcher is killed (no orphan holding ports).
- **Decode reuses a per-request KV cache on every stage.**
  - Each stage keeps a KV cache per request and shard range.
  - Full-context calls reuse the longest common prefix with the previous input. The cache is cropped after a rejected draft, and the last `verify_n` rows are always recomputed.
  - Non-last stages forward only the recomputed rows plus `row_offset` (`delta_rows`). A stage missing the prefix answers `kv_miss`, and the step is resent with full rows.
  - Callers that send only the newest token pass back the returned `inference_state` (it carries `kv_len`).
  - `SYNAPSE_SEQ_KV=0` disables full-context reuse.
  - The engine is pinned to the HF reference: full-shard logits vs `model(...)`, and 1/2/3-stage greedy decode vs `model.generate`, for tiny GPT-2 and SmolLM2 (`test_engine_reference_parity.py`). The distributed protocol is pinned to greedy decoding (`test_verify_protocol_engine.py`).
  - Sliding-window attention is not modelled: the engine applies a full causal mask.
- **The P2P event bridge no longer silently drops legitimate large frames.** The per-frame cap is raised from a hardcoded 10 MiB to a configurable 256 MiB default (`max_frame_size`, settable on `P2PSocketBridge`/`Libp2pNode`); a frame that still exceeds the configured cap gets an explicit `error`/`frame_too_large` packet back to the sender (recorded in `bridge._last_errors[peer_id]`) instead of a silent disconnect — verified by `test_p2p_bridge_binary.py` (a 12 MB tensor ≈ 24 MB framed is delivered intact end to end).
- **With the multi-resource LACP path enabled (`--no-tflops-proportional`), partitioning re-runs resource measurement and bandwidth probing on a cadence**: disk I/O is the counter delta since the previous sample (never a sampling sleep: it runs on the event loop, where a 0.5 s sleep stalled every in-flight request each time it fired), cached for 5 s (`_DISK_IO_CACHE_TTL`) (verified by `test_resource_measurement.py`), and the bandwidth estimate is a transfer-time-based 64 KiB probe (5 samples averaged over request+echo round-trips) — a single-connection lower-bound proxy, not full link capacity (verified by `test_bandwidth_flow.py`).
- **Multi-process runs are verified on one machine, not yet across machines.**
  - `test_live_multiprocess_split.py`: 2 real backend processes answer exactly like one process, each running one contiguous layer range, with hidden states over chunked TCP.
  - `test_live_failover.py`: 3 processes. Once profiles are gossiped, measured placement runs. One stage process is killed while an answer streams; the orchestrator detects it from heartbeats, patches the plan, resends the step, and the streamed answer equals the healthy run.
  - A hardware multi-machine run (C1) is still pending, so real-network latency, bandwidth and failover times are unmeasured.
  - The tensor transport listens on node port + 1000 (47051 by default, `SYNAPSE_TENSOR_PORT` overrides); open it with the node port on firewalls. It used to be a random port, which a firewall cannot be opened for; a send to a blocked port waits out a 5 s connect timeout before the bridge fallback.
  - The default node port is 46051, below the dynamic range (49152+) where Windows reserves port blocks (Hyper-V/WSL; this test machine reserves 50000–50059). If a port still fails to bind, check `netsh int ipv4 show excludedportrange protocol=tcp` and pass `--node-port`.
- **Training uses the served layers.** A single node trains its full-range shard in place; several nodes train as a pipeline (each its own range, see above), full fine-tuning or LoRA.
- **A single request is not faster than one machine that can hold the model.** See "What it does not do" above.
- **The web UI needs a build step.** Without Node.js the installers skip it and `GET /` serves a small built-in status page that says how to build the UI.
- **The master dies.** Survivors ignore a broadcast partition that names a dead node and place layers over the live ones until a new master is elected (`test_master_death.py`); a request whose stage dies mid-generation is patched by the failover loop. A request whose *orchestrator* (the node the client talks to) dies ends with that connection: the client retries on another node.

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
| `SYNAPSE_FAST_PATH` | `1` | Fused Llama-family decoder with paged KV (`0` = run the HF modules) |
| `SYNAPSE_QKERNELS` | `1` | numba int8/int4 CPU kernels (`0` = torch dequantize fallback) |
| `SYNAPSE_INT4_KEEP` | `v_proj,down_proj,mlp.c_proj` | Linears kept int8 in int4 mode (`none` = pure int4) |
| `OMP_WAIT_POLICY` | `PASSIVE` | Set by the package before torch loads (spinning OpenMP threads starved the numba kernels); an explicit value wins |
| `SYNAPSE_CHUNKED_TCP` | `1` | Chunked binary TCP tensor path |
| `SYNAPSE_TLS` | `cluster` if paired, else off | `off` / `cluster` / `selfsigned` / `require` |
| `SYNAPSE_CLUSTER_CODE` | — | Pairing code (else `~/.synapse/cluster_code`, written by the Settings page) |
| `SYNAPSE_TLS_PINS` | — | Comma-separated sha256 cert pins that authenticate peers (manual mode) |
| `SYNAPSE_API_KEY` | — | Bearer key required on every API route; unset = chat open, admin routes local-only |
| `SYNAPSE_AUTOSCALE` | `on` | `on` = act on hints (master), `advise` = log only, `off` |
| `SYNAPSE_RANGE_FETCH` | `1` | Read uncached single-file checkpoints by HTTP range (`0` = download whole files) |
| `SYNAPSE_CKPT_REPLICATE` | `1` | Replicate training checkpoints to the pipeline successor |
| `SYNAPSE_CHECKPOINT_DIR` | `~/.synapse/checkpoints` | Where checkpoints and replicas are stored |

## Installation

```bash
git clone https://github.com/letho1608/synapse_ai.git
cd synapse_ai
install.bat        # Windows  (creates .venv, installs backend/requirements.txt)
sh install.sh      # Linux / macOS
```

---

## Quick Start (Single Node)

```bash
python main.py                # or run.bat on Windows (opens the browser)
```

Open `http://127.0.0.1:52415`: the web UI (dashboard, chat, models, training, settings). The API is under `/v1/*` on the same port.

Working on the UI itself: `npm --prefix frontend run dev` serves it on `:3000` with hot reload and proxies `/v1` to the node (`SYNAPSE_API` to point elsewhere).

---

## Running a Distributed Cluster

### Prerequisites

1. All nodes must be on the same network (LAN) or connected via Tailscale
2. Each node needs `python main.py` running with the same discovery config
3. A shared `cluster_config.json` file or Tailscale API access (UDP discovery on a LAN needs neither)
4. Optional but recommended: pair the machines once, so they talk mutual TLS. In the web UI of the first machine, Settings → "Tạo mã cho cụm mới"; on every other machine paste the code in "Tham gia cụm". Restart the nodes.

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
5. **Pipeline** → serving begins; heartbeat monitoring (2s interval, 5s timeout = `HEARTBEAT_TIMEOUT`) checks every node; each node also publishes its fault-tolerance heartbeat over the event mesh independently of discovery/membership, and a node flagged by the monitor loop (`_pending_rebalance`, first missed heartbeat) is rebalanced by `check_fault_tolerance()` immediately instead of waiting for the 3-miss blacklist threshold; the master broadcasts the authoritative partition before serving distributed requests so every node agrees on shard assignment, and dead nodes trigger `rebalance_after_failure()` repartitioning layers among survivors with JSON checkpoint recovery (the distributed pipeline is initialized unconditionally at `node.start()`, so rebalancing works before any distributed request)
6. **API** → `/v1/chat/completions` serves requests with distributed acceleration

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat completion (OpenAI-compatible, streaming supported) |
| GET | `/v1/models` | Available models |
| POST | `/v1/models/pull` | Download this node's part of a model (its layers' checkpoint files; `{"full": true}` = whole repo) |
| GET | `/v1/cluster/resources` | Per-node resource usage, throughput, shard assignment |
| GET | `/v1/topology` | Cluster topology (nodes, links, latency matrix) |
| GET | `/v1/distributed/status` | Distributed pipeline status |
| POST | `/v1/training/start` | Start distributed training job |

### Chat UI

The API speaks the OpenAI chat format, so existing chat apps work as the browser front end. Open WebUI, for example:

```bash
docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://<node-ip>:52415/v1 \
  -e OPENAI_API_KEY=${SYNAPSE_API_KEY:-none} ghcr.io/open-webui/open-webui:main
```

Then open `http://localhost:3000` and pick the model. LibreChat, Chatbox and other clients that take an OpenAI base URL work the same way: base URL `http://<node-ip>:52415/v1`, any key (or the value of `SYNAPSE_API_KEY` when set).

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
├── main.py                          # Entry point (starts a node; web UI at :52415)
├── frontend/                        # Web UI (Vite + TypeScript): src/main.ts pages, src/api.ts client
├── backend/
│   └── synapse/
│       ├── main.py                  # Backend entry, wiring
│       ├── inference/               # Model execution
│       │   ├── distributed_engine.py  # Orchestrator: one-pass draft verify, failover
│       │   ├── partial_loader.py      # Load only a shard's layers from safetensors
│       │   ├── weight_quant.py        # Weight-only int8 / int4 (mixed-precision int4)
│       │   ├── qkernels.py            # numba int8 / int4 CPU kernels
│       │   ├── fast_decoder.py        # Fused Llama-family decoder + paged KV cache
│       │   ├── stage_training.py      # Per-stage training on the served layers
│       │   ├── attn_backend.py        # Attention kernel probe (fa2 / sdpa / eager)
│       │   ├── inference_engine.py    # Base inference engine
│       │   ├── pytorch_hf_engine.py   # Shard engine: KV reuse, delta rows, batching
│       │   └── shard.py               # Model shard abstraction
│       ├── pipeline/                # Pipeline parallelism
│       │   ├── pp_training.py         # Pipeline-parallel training protocol across nodes
│       │   ├── gradient_sync.py       # DistributedTrainer (DDP, ring all-reduce)
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
│       └── training/                 # Datasets, LoRA on served layers (stage_lora.py)
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
│       ├── test_1f1b_training_production.py # 3 tests: training schedule routing (ddp / pipeline)
│       ├── test_pipeline_training.py       # pipeline step == full-model step, real Node wiring
│       ├── test_fast_decoder.py            # fast decoder == HF modules, paged KV
│       ├── test_multinode_lacp_flow.py     # 5 tests: multi-node LACP partition + SHM wiring
│       ├── test_api_endpoints.py           # API routes + built-in `GET /` page + distributed status
│       ├── test_p2p_bridge_binary.py       # 6 tests: binary tensor roundtrip + latency probe echo + large-frame handling (256 MiB cap, explicit error on oversize)
│       ├── test_b3b4b5_wiring.py           # 6 tests: latency-cache refresh, authoritative-partition broadcast, mesh heartbeats, pending-rebalance consumption
│       ├── test_b6b7_cleanup.py            # SHM active-segment bound
│       └── test_lacp_multiresource.py      # LACP partitioning tests
└── README.md
```
