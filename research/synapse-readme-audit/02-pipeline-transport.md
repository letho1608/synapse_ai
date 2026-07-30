# Lane 2 Audit: Pipeline & Transport Layer

**Auditor**: Grok Build — Codebase Auditor  
**Date**: 2026-07-31  
**Scope**: pipeline parallelism, tensor transport, networking infrastructure  
**Files examined**:
- `backend/synapse/pipeline/scheduler.py`
- `backend/synapse/pipeline/stage.py`
- `backend/synapse/pipeline/gradient_sync.py`
- `backend/synapse/transport/codec.py`
- `backend/synapse/transport/tcp_transport.py`
- `backend/synapse/transport/result_router.py`
- `backend/synapse/inference/distributed_engine.py`
- `backend/synapse/networking/peer_handle.py`
- `backend/synapse/networking/grpc/grpc_server.py`
- `backend/synapse/networking/grpc/grpc_peer_handle.py`
- `backend/tests/test_pipeline_1f1b.py`
- `backend/tests/test_async_forwarding.py`

---

## Claim-by-Claim Verification

### Claim 1: "Pipeline parallelism with micro-batch overlapping" — specific 1F1B schedule described with diagrams

**Verdict: PARTIALLY**  

**Severity: High**

**What exists:**  
`PipelineScheduler.generate_1f1b_schedule()` in `pipeline/scheduler.py:86-146` generates mathematically correct 1F1B schedules with warmup, steady state, and cooldown. Tests in `test_pipeline_1f1b.py` validate the slot assignments (forward ascending, backward descending), stage staggering, and no-duplicate-steps constraints.

```python
# scheduler.py:93-98
def generate_1f1b_schedule(self, num_stages, num_micro_batches, node_ids=None):
    """
    Generate a 1F1B schedule for GPipe-style pipeline parallelism.
    - Warmup: (num_stages - 1) forward passes to fill the pipeline
    - Steady: num_micro_batches * 2 - (num_stages - 1) ops alternating F/B
    - Cooldown: (num_stages - 1) backward passes to drain
    """
```

**What's missing:**
- `generate_1f1b_schedule()` is called **only from tests** (test_pipeline_1f1b.py, test_lacp.py) — never from production code.
- `PipelineExecutor.run_inference()` (`stage.py:158-221`) uses `generate_inference_schedule()` (forward-only), **not** the 1F1B schedule.
- `DistributedPipelineController` (`distributed_engine.py`) also uses `generate_inference_schedule()` exclusively.
- The training path (`gradient_sync.py`) uses DDP (ring all-reduce on full model copies), not pipeline parallelism at all — no backward pass is ever routed through the pipeline stages.
- `PipelineStage.process_loop()` (`stage.py:111-126`) only executes `run_forward()` — there is no backward execution path in the stage runner.

**Impact:** The 1F1B schedule is a correct *algorithmic reference model* but is **not wired into any runtime execution path**. The README's diagram (lines 80-90) showing F/B interleaving for training is aspirational.

---

### Claim 2: "Async non-blocking forwarding: tensor sent as soon as ready; next micro-batch starts immediately"

**Verdict: PARTIALLY**

**Severity: Medium**

**What exists at the transport layer:**  
`AsyncTensorForwarder.send()` in `codec.py:280-321` uses `asyncio.create_task(self._do_send(...))` for fire-and-forget sends, with a semaphore-based backpressure mechanism.

```python
# codec.py:315-318
# Fire-and-forget write (don't await - let it run in background)
asyncio.create_task(self._do_send(writer, encoded))
return True
```

**What doesn't exist / contradicts:**
- The README's diagram (lines 98-102) shows a dedicated `async_forwarder.py` module. **This file does not exist.** The `AsyncTensorForwarder` class lives inside `codec.py` instead.
- The distributed execution path in `DistributedPipelineController._send_input_to_stage()` (`distributed_engine.py:299-323`) uses `await self.event_router.publish(topic, ...)` — a **blocking await**, not fire-and-forget.
- `PipelineExecutor.run_inference()` feeds micro-batches sequentially via `enqueue_input` (line 203-208) and collects one-by-one via `dequeue_output` (line 209-214). While the async queue enables some overlap, the first stage blocks on sequential enqueues, and the caller blocks on sequential dequeues. True non-blocking pipelining would have all micro-batches injected immediately without awaiting each.
- The README states "next micro-batch starts immediately" — but `_send_input_to_stage` awaits the publish, which serializes the sends.

---

### Claim 3: "Tensors are encoded as binary blobs (not JSON) using a compact header format"

**Verdict: VERIFIED**

**Severity: Note (pass)**

**Evidence:**  
`codec.py` implements a binary codec with a 16-byte header:
```
MAGIC(4) + dtype_code(2) + ndim(2) + uncompressed_size(4) + compressed_size+flags(4) = 16 bytes
```

```python
# codec.py:32-35
HEADER_SIZE = 16  # 4 magic + 2 dtype + 2 ndim + 8 reserved

# codec.py:131-138
header = struct.pack(
    "!4sHHII",
    MAGIC,       # b"SYNP" — 4 bytes
    dtype_code,  # 2 bytes
    ndim,        # 2 bytes
    uncompressed_size,  # 4 bytes
    (compressed_size << 2) | is_compressed_flag,  # 4 bytes
)
```

- Tensor payload is `numpy.tobytes()` raw bytes — no JSON.
- Metadata uses JSON but is a small auxiliary field, not the tensor data.
- Optional LZ4/zlib compression + INT8 quantization supported.
- Tests validate roundtrip for float32, float16, int64, int8, bool, scalars, empty tensors (test_async_forwarding.py:32-105).

**Note:** The README claims "msgpack header" (line 98). The actual implementation uses a custom struct-based header, not msgpack. This is a minor documentation inaccuracy.

---

### Claim 4: "Same-machine nodes use zero-copy shared memory (numpy memmap)"

**Verdict: NOT FOUND**

**Severity: Blocker**

**Evidence:**  
Zero implementation exists anywhere in the codebase:
- No `numpy.memmap` usage
- No `multiprocessing.shared_memory.SharedMemory` import
- No `mmap` import
- No `shm_*` references
- No shared-memory transport module or transport variant

The only references are in documentation:
- `README.md:108`: "Same-machine nodes use **zero-copy shared memory** (numpy memmap)"
- `codec.py:9`: docstring comment "Supports zero-copy shared memory for same-machine transfers"

This is a **feature described as existing** that has no code backing whatsoever. It affects the entire same-machine fast-path claim.

---

### Claim 5: "Cross-machine nodes use TCP with Nagle disabled for minimal latency"

**Verdict: PARTIALLY**

**Severity: High**

**Evidence:**  
- `tcp_transport.py` implements `TCPServer`, `TCPClientPool`, and `TensorTCPTransport` using `asyncio.start_server` / `asyncio.open_connection`. **Neither sets `TCP_NODELAY`.** No socket option manipulation exists in this file.
- `TCP_NODELAY` IS enabled in the **gRPC transport** (`grpc_server.py:35`: `("grpc.tcp_nodelay", 1)`, `grpc_peer_handle.py:37`: same). But this is the gRPC stack, not the dedicated TCP transport described in the README.
- The README (line 109) specifically describes "TCP with Nagle disabled" as the cross-machine transport. The `tcp_transport.py` module is presented as this transport but does not configure TCP_NODELAY.
- The gRPC transport has Nagle disabled but uses protobuf serialization (not the binary codec from `codec.py`), so it's a different protocol entirely.

**Summary:** Nagle is disabled on the gRPC transport, but not on the dedicated TCP transport. The README describes the TCP transport as the cross-machine path, which contradicts the implementation.

---

### Claim 6: "Optional RDMA support for InfiniBand interconnects"

**Verdict: NOT FOUND**

**Severity: Medium**

**Evidence:**  
Zero implementation exists:
- No `ibverbs`, `rdma`, or InfiniBand imports anywhere
- No RDMA transport module
- No `--enable-rdma` configuration option
- The only mention is `README.md:110`: "Optional RDMA support for InfiniBand interconnects"

This is documented as a supported feature with no code, scaffolding, or configuration hook.

---

### Claim 7: "In-order reassembly — caller sees clean token stream"

**Verdict: VERIFIED**

**Severity: Note (pass)**

**Evidence — Two-layer reassembly:**

1. **Transport layer** (`AsyncTensorForwarder.receive()` in `codec.py:331-369`):
```python
# codec.py:361-369
self._reassembly[request_id][msg.micro_batch_id] = msg
next_id = self._next_batch_id[request_id]
if next_id in self._reassembly[request_id]:
    result = self._reassembly[request_id].pop(next_id)
    self._next_batch_id[request_id] = next_id + 1
    return result
return None  # buffered for later
```

2. **Result layer** (`ResultBuffer.insert()` in `result_router.py:41-52`):
```python
# result_router.py:42-52
self._pending[result.micro_batch_id] = result
delivered: List[RoutedResult] = []
while self._next_expected in self._pending:
    delivered.append(self._pending.pop(self._next_expected))
    self._next_expected += 1
return delivered
```

Both handle out-of-order delivery by buffering until predecessors arrive. Tests validate this (test_async_forwarding.py:140-168 — batches received in order 2, 0, 1 produce correct sequential output).

**Note:** The reassembly operates at the micro-batch level (complete tensors), not at the token-stream level. The README's "caller sees clean token stream" is slightly oversold — token-by-token streaming with micro-batch granularity means the caller gets bursts of tokens per micro-batch, not a fine-grained stream.

---

### Claim 8: "After warmup, every node is busy every step" / bubble-free pipeline

**Verdict: PARTIALLY**

**Severity: High**

**Evidence — schedule is correct:**
- `generate_1f1b_schedule()` produces mathematically bubble-free schedules with proper warmup/steady/cooldown phases.
- `generate_inference_schedule()` also staggers correctly: stage N starts at step N and processes all micro-batches with no gaps.

```python
# scheduler.py:150-178 — inference schedule generation
for stage in range(num_stages):
    for mb in range(num_micro_batches):
        step = stage + mb
        schedule.slots.append(PipelineSlot(...))
```

**Evidence — runtime does NOT match:**
- `PipelineExecutor.run_inference()` (`stage.py:158-221`) feeds micro-batches to stage 0 sequentially **before** collecting from the last stage. The `process_loop` tasks do run concurrently, which provides some overlap, but:
  - The first stage is fed one-at-a-time via `enqueue_input` — this is a sequential loop, not the burst injection that true bubble-free operation requires.
  - `run_inference()` waits for each micro-batch from the last stage one-by-one (`dequeue_output`), creating idle time at the caller.
  - The intermediate stages' `process_loop` relies on `asyncio.wait_for` with 0.5s timeout for polling (`stage.py:113`). During bubbles the stage spins with timeouts.
- For training: no 1F1B runtime exists at all, so the bubble-free claim for training is entirely unsupported.

**The theoretical schedule says "no bubbles." The runtime implementation has real bubbles from sequential feeding, timeout-polling receive loops, and only forward-only execution.**

---

## Module Structure Discrepancy

The README's project structure (lines 222-247) lists:
- `synapse/transport/async_forwarder.py` — **does not exist** (class lives in codec.py)
- `synapse/pipeline/scheduler.py` ✅
- `synapse/pipeline/stage.py` ✅
- `synapse/pipeline/gradient_sync.py` ✅
- `synapse/transport/codec.py` ✅
- `synapse/transport/result_router.py` ✅

The missing `async_forwarder.py` is a minor structural mismatch but indicates the documentation was written from a design spec rather than the actual code.

---

## Summary

**Confidence: 6/10**

The pipeline and transport layer shows a clear pattern: the **algorithmic/theoretical components are well-implemented** (binary codec, 1F1B schedule generator, ordered reassembly, async forwarder pattern), but the **runtime integration is incomplete or contradictory**:

- **What works and is verified:** Binary tensor codec (compact header, compression, quantization), in-order reassembly at both transport and result layers, the 1F1B schedule generation algorithm, async fire-and-forget send pattern in the forwarder class, Nagle-disabled gRPC.
- **What's documented but doesn't exist:** Zero-copy shared memory transport (claimed as existing, zero code), RDMA/InfiniBand support (claimed as optional, zero code), the `async_forwarder.py` module (listed in tree, doesn't exist).
- **What's documented but incorrect:** TCP_NODELAY is set on gRPC but **not** on the dedicated TCP transport; the 1F1B runtime doesn't exist (scheduler generates schedules, no executor runs them); 1F1B is only used in tests.
- **Theoretical vs practical gap:** The bubble-free claim is mathematically supported by the scheduler but not achieved by the runtime executor, which uses sequential feed and collection loops.

The most critical findings are the **missing shared memory transport** (a blocker for the same-machine fast-path claim) and the **lack of a 1F1B runtime** (the training pipeline claim is pure documentation). The TCP transport exists but lacks Nagle configuration, meaning cross-machine latency is likely higher than claimed.

### Ground-truth tally
- External-checked (code) facts: 6/8 claims have at least partial code existence
- Model judgment elements: severity ratings, runtime behavior analysis, integration completeness assessment
