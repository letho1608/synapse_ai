# Phase 3 Verification — Lane A: Decode & Inference Correctness

**Lane scope:** distributed decode pipeline, speculative token verification, binary transport codec & result routing.
**Auditor:** specialist agent A (of A/B/C/D)
**Date:** 2026-07-31
**State audited:** working tree of `D:\Code\paper\synapse_ai` — the "implementation pass" is **uncommitted** (`git status`: 11 modified + 8 untracked files). All line numbers below refer to the current working-tree files.
**Method:** full source read of `distributed_engine.py`, `codec.py`, `result_router.py`, `p2p_socket_bridge.py`, the relevant parts of `node.py`, `node_extension.py`, `pytorch_hf_engine.py`, `event_router.py`, `tcp_transport.py`; three live test-file executions; two empirical reproduction scripts (bridge round-trip, decode shape crash, token-buffer dedup). **No source code or tests were modified.**

---

## 1. Verdict up front

The two fixes that this lane was told to check **did land and are logically sound in isolation**:

- The decode-overlap fix now fires `num_decode_batches = max(2, lookahead_window)` micro-batches carrying **prefix** contexts `[last] + guesses[:i]` and accepts tokens only via a new `verify_speculative_tokens()` verifier. As a pure-function/unit-level matter, the prior "tokens from contaminated contexts are unconditionally accepted" Blocker is **fixed**. 10/10 behavioral tests pass.
- The codec is genuinely binary (16-byte `SYNP` header, INT8 quantization, zlib/LZ4), and 16/16 codec+forwarder tests pass.

**However, the shipped distributed pipeline cannot run with a real model, for two independent reasons that no test in the repository exercises:**

1. **Blocker — the binary tensor path is broken over the production P2P transport.** `encode_tensor()` output (`bytes`) is hex-encoded into JSON by the socket bridge and never converted back on the receiving side. Every tensor published to a remote node (`distributed_engine.py:371`, `node.py:1253`) arrives as a hex **string**, and `decode_tensor()` raises `TypeError: a bytes-like object is required, not 'str'` (empirically reproduced). The exception is silently swallowed by the event router.
2. **Blocker — `process_distributed_prompt()` crashes on the real inference engine's `encode()` output.** The real `encode()` returns a flattened **1-D** array (`pytorch_hf_engine.py:524`), and the controller reads `token_ids.shape[1]` (`distributed_engine.py:187`) → `IndexError` (empirically reproduced). The exception is swallowed and the request silently falls back to single-node processing (`node.py:861-869`).

So the honest answer to "does the codebase now meet the README's decode/inference claims" is: **the decode-overlap and token-verification logic is real and correct as a unit, but the distributed decode pipeline it powers cannot execute end-to-end with real models and real transports.** The feature is "wired but unreachable," which is a regression-adjacent version of the historical "code exists but is not wired" pattern.

---

## 2. Finding 1 — 🔴 Blocker: binary `{"bin": bytes}` tensor transport is broken across the P2P socket bridge

### The claim

README "Async Tensor Forwarding Protocol": "Tensors are encoded as **binary blobs** (not JSON) using a compact header format"; "non-blocking socket send."

### What the code does

The shipped decode/inference path sends tensors as `{"bin": encode_tensor(...)}` events:

- `distributed_engine.py:371` — `await self.event_router.publish(topic, {"bin": encoded_payload})` (`_send_input_to_stage`, used for **every** prompt and decode micro-batch).
- `node.py:1253` — `await self.event_router.publish(f"synapse/inference/tensor/{target_id}", {"bin": encoded_payload})` (`forward_tensor`, used for every intermediate hidden-state hop).

The only cross-node transport for these events is `Libp2pNode` → `P2PSocketBridge` (production: `synapse/main.py:118`; node subscribes at `node.py:127`). The bridge:

- Subscribes to `synapse/#` (`p2p_socket_bridge.py:73`).
- Serializes the whole packet with `json.dumps(packet, default=self._json_serialize)` (`p2p_socket_bridge.py:352-354`).
- `_json_serialize` converts `bytes` → `obj.hex()` — a **hex string** (`p2p_socket_bridge.py:381`).
- On the receiving node, `_read_loop` calls `_json_deserialize(packet.get("data"))` (`p2p_socket_bridge.py:407`). `_json_deserialize` restores `__np__` (numpy) payloads but has **no branch that turns a plain hex string back into `bytes`** (`p2p_socket_bridge.py:420-431`).
- `on_p2p_tensor` then calls `decode_tensor(data["bin"])` where `data["bin"]` is a `str` (`node.py:499`). `decode_tensor` performs `struct.unpack("!4sHHII", data[:HEADER_SIZE])` (`codec.py:150-151`) → **`TypeError: a bytes-like object is required, not 'str'`**.
- `on_p2p_tensor` has no `try/except` around this; the exception propagates into `event_router.publish`, which gathers async callbacks with `return_exceptions=True` (`event_router.py:59-72`) → **silently swallowed**.

### Empirical proof (reproduction run)

A script using two real `Libp2pNode`/`P2PSocketBridge` instances over real TCP sockets:

```
connected: True
encoded type: bytes len: 236
events received on B: 1
bin type on B: str len: 472
DECODE FAILED: TypeError a bytes-like object is required, not 'str'
legacy tensor_data type on B: str
LEGACY DECODED OK: [[3, 1, 2]]
```

The legacy `tensor_data` path survives because `on_p2p_tensor` explicitly does `bytes.fromhex(...)` for strings (`node.py:522-524`); the high-performance `"bin"` path has no equivalent. The legacy path **works**, the new binary path **does not**.

### Consequence

Every tensor destined for a remote node is dropped. A multi-node request behaves as: orchestrator fires prompt micro-batches → dropped → `_collect_result` waits 60–120 s per batch → times out → empty prompt results → decode loop fires batches → dropped → 30 s timeouts → `decode_results` empty → `break` → returns `[]` → `node.py:861-869` silently falls back to single-node. The README's entire distributed decode/overlap story is unreachable over the mesh.

### Why no test catches it

`test_decode_pipeline_overlap.py` replaces `_send_input_to_stage` and `_collect_result` with in-process oracle stubs (see §7), and `test_async_forwarding.py` exercises the codec only in isolation. No test in the repo sends a `{"bin": bytes}` event through a real bridge. `test_p2p_minimal.py` only round-trips JSON-safe strings.

**Severity: Blocker.** Primary output path of the flagship feature, silently broken, zero coverage.

---

## 3. Finding 2 — 🔴 Blocker: `process_distributed_prompt()` crashes on the real engine's 1-D `encode()` output

### The claim

README: distributed pipeline runs "across hundreds of nodes"; decode overlap in the distributed controller.

### What the code does

- `distributed_engine.py:179` — `token_ids = await self.inference_engine.encode(base_shard, prompt)`.
- `distributed_engine.py:187` — `prompt_length = token_ids.shape[1]`.
- The real engine returns a **1-D** array: `pytorch_hf_engine.py:517-524` → `enc = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="np"); return enc.flatten().astype(np.int32)`. `(1, N).flatten()` → `(N,)` — `shape[1]` does not exist.

### Empirical proof

```
ndim 1 shape (5,)
IndexError: tuple index out of range
```

Running `process_distributed_prompt` against a fake engine that mimics the real engine's 1-D `encode` (with `DEBUG=1`):

```
Traceback (most recent call last):
IndexError: tuple index out of range
RETURNED tokens: []
```

The `IndexError` is caught by the blanket `except` at `distributed_engine.py:330-337`, the session is discarded, and `[]` is returned. In production, `node.py:861-869` treats an empty result as "fallback" and re-runs the request single-node.

### Consequence

For any real multi-node cluster, every `process_prompt_distributed` call crashes before sending a single tensor, then **silently degrades to single-node inference**. No acceleration, no pipeline overlap, no speculative decode — and no error surfaced to the operator (only `DEBUG>=1` prints a traceback). The README's scaling/overlap claims are therefore not merely unvalidated — they describe a code path that cannot execute against the shipped engine.

### Why no test catches it

The decode-overlap tests stub `encode` with a **2-D** array (`test_decode_pipeline_overlap.py:87-88`, `return_value=np.array([[1, 2]])`), matching what the code expects rather than what the engine produces.

**Severity: Blocker.** Silent total failure of the distributed inference path with the real model backend.

> Note: Findings 1 and 2 are independent — fixing either alone does not make multi-node decode work. Both must be fixed.

---

## 4. Finding 3 — ✅ Validated (positive): speculative token verification is now sound at the unit level

### The claim (task brief)

"tokens from contaminated contexts are never accepted."

### What the code does

New pure function `verify_speculative_tokens(guesses, batch_tokens)` at `distributed_engine.py:28-64`:

```python
accepted.append(batch_tokens[0][0])                    # true-context token
for i in range(1, len(batch_tokens)):
    prev_guess = guesses[i - 1] if i - 1 < len(guesses) else None
    if prev_guess is None or not batch_tokens[i - 1]: break
    if batch_tokens[i - 1][0] != prev_guess: break      # gate
    if batch_tokens[i]: accepted.append(batch_tokens[i][0])
```

Batch `i`'s context is `[last] + guesses[:i]` (`distributed_engine.py:269-273`). Its token is accepted only when the chain of gates holds: batch `i-1` (whose context was clean, inductively) predicted exactly `guesses[i-1]`. Trace for `lookahead_window=3` with a wrong first guess `[6, 99]`, model truth `[6, 7]`:

- `batch_tokens = [[6],[7],[999]]` (batch 2's context `[last,6,99]` is contaminated).
- `accepted = [6]`; `i=1`: `batch_tokens[0][0]=6 == guesses[0]=6` → accept `7` (from clean context `[last,6]`); `i=2`: `batch_tokens[1][0]=7 != guesses[1]=99` → stop. `999` is never accepted. ✓

The inductive invariant holds: **no token whose context contains an unverified guess can be appended**, because acceptance of batch `i` requires prior verification of exactly the guesses `guesses[:i]` that its context contains, and the loop breaks at the first mismatch (so later, more-contaminated batches are never reached). The gate uses `batch_tokens[i-1][0]` — the first (only) predicted token per batch, which is all the shipped engine produces (`node.py:660-725`).

The old code unconditionally accepted the first token of every batch (`git diff`: the removed loop `accepted_count = 1`); the working tree replaces it with this verifier. The fix is genuine and correctly located.

### Decode-loop integration

- `distributed_engine.py:261` — `num_decode_batches = max(2, lookahead_window)` (matches README verbatim).
- `distributed_engine.py:267-276` — batches built as `[last] + guesses[:db_idx]` — a **prefix** of guesses (was `guesses[db_idx:]`, the contaminated suffix construction, in the committed version).
- `distributed_engine.py:279-283` — all batches fired before any collection (the "overlap").
- `distributed_engine.py:301` — `decode_results.sort(key=lambda r: r.micro_batch_id)` so batch `i`'s prediction is paired with context `guesses[:i]` regardless of arrival order.
- `distributed_engine.py:302-308` — verification + extension.

### Tests

`test_decode_pipeline_overlap.py` — **10/10 PASSED** (0.71 s), including `test_wrong_guess_stops_chain_no_contaminated_token`, `test_decode_loop_emits_only_true_context_tokens`, `test_decode_loop_fires_speculative_batches_with_prefix_context`, and `test_decode_loop_rejects_contaminated_predictions`. The tests drive the real `verify_speculative_tokens` and the real `process_distributed_prompt` entry point (oracle stubs only at the transport boundary — see §7).

**Severity: validated positive.** The prior audit's D-1/0.1 Blocker (decode contamination) is resolved at the logic level.

---

## 5. Finding 4 — ✅ Validated (positive): the codec is genuinely binary and its tests are real

### The claim

README: "tensors encoded as binary blobs, not JSON", "compact header format", "SYNP header".

### What the code does

- `codec.py:33-34` — `MAGIC = b"SYNP"`, `HEADER_SIZE = 16`.
- `codec.py:130-139` — header `!4sHHII` = magic(4) + dtype(2) + ndim(2) + uncompressed_size(4) + (compressed_size<<2 | flag)(4). Payload = raw `tobytes()`, no `tolist()`, no JSON for the tensor itself.
- `codec.py:68-77,197-203` — INT8 dynamic quantization with `_q_scale`/`_orig_dtype` metadata and dequantization on decode.
- `codec.py:88-107` — LZ4 with zlib fallback, only when it actually shrinks the payload.
- `decode_tensor` validates magic and dtype, restores shapes from `!H` dims.

### Tests

`test_async_forwarding.py` — **16/16 PASSED** (1.21 s): 11 codec round-trip/compression/validation tests driving the real `encode_tensor`/`decode_tensor` (including `test_header_size_constant` asserting the exact 16-byte layout), plus 5 `AsyncTensorForwarder` tests driving real send/receive/order/flush logic through the real encoder.

These tests exercise the real encode/decode path (e.g., `test_basic_send_receive` captures the writer's bytes and decodes them back). No source-string inspection anywhere in the two in-lane test files (grep for `inspect|getsource|open\(` returned nothing).

**Severity: validated positive.** The codec matches its README description; the issue is the *framing* that wraps it on the wire (§7, Finding 6).

---

## 6. Finding 5 — 🟠 Medium/High: `_collect_result` never validates batch identity — contamination vector under lost/stale results

### What the code does

`_collect_result` (`distributed_engine.py:379-421`) subscribes to `synapse/inference/result/{request_id}`, waits for **one** event, returns it, unsubscribes. Its handler copies `micro_batch_id` from the payload (`distributed_engine.py:394-398`) but performs **no check** that the received result belongs to the expected batch, the current decode iteration, or even this iteration's set of ids.

In the normal flow this is safe: the last stage publishes results in batch order (`node.py:766-771` inside the sequential `read_loop` processing at `p2p_socket_bridge.py:385-418`), and the subscribe→collect→unsubscribe→subscribe transition between loop iterations is fully synchronous (no `await` between them), so no event can be lost in the gap.

But under a fault/stall the loop is unsafe:

1. If one result times out (30 s, `distributed_engine.py:294`), the iteration continues with a **shorter** `decode_results` list. `verify_speculative_tokens` then pairs by *position* (`batch_tokens[1]` is the second element of the shorter list), not by `micro_batch_id`. If batch 2's result survives while batch 1's is lost, batch 2's token (predicted from `[last, g0, g1]` where `g1` is **unverified**) is accepted against the `g0` gate alone.
2. Microbatch ids 0..N-1 are **reused every iteration** (`distributed_engine.py:281`), and the topic is keyed only by `request_id`. A stale result from a timed-out iteration can be collected as the *next* iteration's batch-0 result and accepted against the next iteration's guesses.

Both are latent (they require a 30 s stall or dropped event), but the result-collection layer is the one remaining place where a contaminated token can enter the output.

**Severity: Medium** (requires a fault; intersects with lane C's fault-tolerance work). The fix would be to have `_collect_result` accept an expected-id set and ignore/re-buffer anything else.

---

## 7. Finding 6 — 🟠 Medium: "non-blocking forwarding" and "not JSON" describe an unwired component, not the shipped path

### The claims

README: "Async non-blocking forwarding: tensor sent as soon as ready; next micro-batch starts immediately"; "Tensors are encoded as binary blobs (not JSON)"; "non-blocking socket send (TCP_NODELAY enabled)".

### What the shipped path actually does

- The decode/inference tensor path never uses `AsyncTensorForwarder` (fire-and-forget `create_task` at `codec.py:307-317`) nor `TensorTCPTransport` (`tcp_transport.py:220-243`). Grep confirms `TensorTCPTransport` is only ever started/stopped/`register_peer`'d in `node.py:236,356,1362` — `send_tensor` is never called and `set_tensor_handler` is never registered anywhere in production.
- The actual path is `event_router.publish` → bridge `_on_local_event` → `_broadcast_packet`, which:
  - serializes every event — including tensors — through `json.dumps` with bytes hex-encoded (`p2p_socket_bridge.py:352-381`), i.e. **JSON framing with 2× byte inflation**;
  - **broadcasts to every connected peer** regardless of topic (`p2p_socket_bridge.py:355-358`);
  - does a sequential **blocking** `writer.write(...); await writer.drain()` per peer (`p2p_socket_bridge.py:359-365`).
- `_send_input_to_stage` awaits that whole broadcast (`distributed_engine.py:371`), so the next micro-batch does **not** "start immediately" on a slow/backed-up link — the fire loop stalls on the drain.

So the README's "non-blocking" and "not JSON" claims describe the AsyncTensorForwarder/TCP-transport design, which is real but **not wired to the tensor flow**. (The TCP transport is not entirely dead — it exists and tests cover it — but no tensor ever travels through it in the production inference path.)

**Severity: Medium.** Performance/bandwidth claims and the transport description are inaccurate for the shipped runtime, even after the codec fix.

---

## 8. Finding 7 — 🟠 Medium: `_on_token_received` dedup drops legitimately repeated tokens

### What the code does

`node.py:1705-1721`:

```python
current_tokens, _ = self.buffered_token_output[request_id]
if tokens:
    if not current_tokens or current_tokens[-len(tokens):] != tokens:
        current_tokens.extend(tokens)
```

In distributed mode every accepted token reaches the orchestrator's buffer twice — once via the decode loop's direct `on_token_callback` (= `_on_token_received`, wired at `node.py:164`, `node_extension.py:119`) and once via the last stage's `result/{orchestrator_id}` publish (`node.py:771`) → `on_p2p_result` → `_on_token_received`. The suffix-dedup prevents the double-add — but it also suppresses a **genuine repetition**: if the model emits token `X` then `X` again, the second `X` sees `current_tokens[-1:] == [X]` and is dropped.

### Empirical proof

```
after repeated token 42,42: [42]     # second legitimate 42 lost
after adding 7: [42, 7]
```

Single-node mode is unaffected (the sampler's direct `append` at `node.py:697` is authoritative). `buffered_token_output` feeds the watchdog/retry logic and downstream consumers on the orchestrator, so distributed streaming/reassembly of repeated tokens is wrong.

**Severity: Medium** (data-loss bug in the distributed token buffer; exact trigger is any adjacent repeated token).

---

## 9. Finding 8 — 🟠 Medium: distributed decode conditions on a truncated context (no prompt, no KV carry-over)

### What the code does

- Decode micro-batch inputs are literally `[last] + guesses[:i]` (`distributed_engine.py:272-273`) — the last generated token plus guesses. The full prompt+history is **not** part of the model input.
- On the first-stage worker node, the concatenation branch that would rebuild the full sequence (`node.py:1150-1161`: `full_ids = concat([prompt_ids, buf])`) only fires when `request_id in self._prompt_token_ids` — but in the distributed path the orchestrator never stores `_prompt_token_ids` (only `_run_single_node` at `node_extension.py:179` and the single-node `_process_prompt` at `node.py:884-885` do), and workers never have it. So the branch is dead in distributed mode.
- No KV state is carried across decode steps: `inference_state` for each decode batch is a fresh `{"is_prompt": False, "microbatch_id": ...}` (`distributed_engine.py:364-368`), and the engine's KV cache is keyed by **content hash** (`pytorch_hf_engine.py:340-405`), so a `[last]+guesses` sequence computes from scratch as a standalone sequence.

### Consequence

Batch 0's "true context" is just `[last]` — the model predicts the next token after the single token, not after the full prompt. Speculative verification then "verifies" guesses against a model that never saw the user's prompt. This is semantically incorrect for any prompt longer than one token, independent of the transport bugs. (Given Findings 1–2, this never executes with a real model today, so it is latent — but it would surface immediately if 1–2 were fixed.)

**Severity: Medium** (correctness of the core algorithm in the real-engine case).

---

## 10. Finding 9 — 🟡 Medium/Low: multi-batch prefill produces incoherent continuations and unordered collection

- With `num_micro_batches > 1` (`distributed_engine.py:188-216`), each prompt chunk is an independent sequence: `position_ids` restart at 0 per chunk (`pytorch_hf_engine.py:368-369`), so a chunk's logits are not conditioned on the preceding chunk.
- Every chunk's sampled token is appended to `prompt_results` (`distributed_engine.py:200-203,211-216`) and treated as part of the output sequence — the model's prediction at each chunk boundary, not a coherent continuation.
- Unlike the decode phase, prefill collection has **no sort and no identity check** — `for _ in micro_batches: await collect_prompt_result()` appends in arrival order.
- Default `num_micro_batches = 1` (`node.py:115`) masks this; the API does not expose `num_micro_batches`.

**Severity: Medium/Low** (off by default; wrong when enabled).

---

## 11. Notes (Low / Note)

- **`max_generate_tokens` overshoot** — the loop guard (`distributed_engine.py:246`) checks before extending by up to `lookahead_window` tokens; output can overshoot the limit by N-1 tokens (`distributed_engine.py:302-308`).
- **Redundant identical batches** — when the n-gram search finds no match (the common case after the first iteration in the test trace), `num_decode_batches = 3` batches with identical `[last]` inputs are fired; two of three forward passes are wasted, and one sample is discarded. "Overlap" degenerates to 3× redundant work per token.
- **ResultRouter is decorative in this path** — `receive_result` (`result_router.py:98-113`), `route_to_orchestrator` (`result_router.py:130-145`), and `register_token_callback` are never called from the shipped inference flow (grep: only `register/unregister_token_callback` in `distributed_engine.py:174,339`). The README's "in-order reassembly" is actually implemented by the decode loop's `sort` at `distributed_engine.py:301`. In-order reassembly is functionally present, but via a different mechanism than advertised.
- **`sequence_position` doesn't exist** — README's "(request_id, orchestrator_id, micro_batch_id, sequence_position) tuple" — `TensorMessage` (`codec.py:210-224`) carries `source_stage`/`target_stage`, not `sequence_position`.
- **Codec docstring drift** — "reserved(8)" (`codec.py:5-7`) is actually `uncompressed_size` + `compressed_field`.
- **Metadata pruning** — metadata > 1024 bytes is replaced by q-info only (`codec.py:121-126`), silently dropping `request_id`/`microbatch_id` from decode (the receive side defaults them, `codec.py:344-347`).
- **Backpressure asymmetry** — `AsyncTensorForwarder.send` returns `False` (drop) on a 0.5 s semaphore timeout (`codec.py:289-298`); unused in the decode path, but a latent drop-vector if ever wired.
- **Testing gap for the verifier's collection layer** — no test feeds `verify_speculative_tokens` from the real event flow; the 7 verifier tests call it with hand-built lists.

---

## 12. "No mock-only coverage" — assessment for this lane

README line 27: "No mock-only coverage."

- `test_async_forwarding.py`: drives the **real** codec and forwarder end-to-end at the byte level. Clean.
- `test_decode_pipeline_overlap.py`: drives the **real** `verify_speculative_tokens` and the **real** `process_distributed_prompt`; it does **not** re-implement the unit under test, and it starts at the shipped entry point. However, it replaces two methods of the unit under test — `_send_input_to_stage` and `_collect_result` (`test_decode_pipeline_overlap.py:138-139`) — with an in-process oracle queue. That is legitimate unit-boundary stubbing, but it means the tests cannot detect that the real send/collect wiring is broken (Findings 1–2). The test docstring's claim that these are "no mocks of the unit under test" is defensible only if the transport I/O is considered outside the unit; the empirical result is that the suite gives **false confidence** about the end-to-end feature.
- **No source-string inspection** exists in either in-lane test file (no `inspect`, `getsource`, or file-reading assertions).
- The README's per-module test count for `test_decode_pipeline_overlap.py` ("10 tests") is accurate.

---

## 13. What does NOT work (subtractive findings)

1. **Cross-node tensor delivery** — `{"bin": bytes}` events are dropped at every remote hop (hex-string `TypeError`), silently. No prefill or decode micro-batch can reach a remote stage. *(Finding 1, empirically verified)*
2. **`process_distributed_prompt` with the real engine** — crashes at `token_ids.shape[1]` (IndexError); every multi-node request silently falls back to single-node. The distributed decode loop never runs with a real model. *(Finding 2, empirically verified)*
3. **"Async non-blocking forwarding"** — the shipped path blocks on per-peer `drain()` and serializes tensors as JSON+hex; the fire-and-forget forwarder/TCP transport is unwired. *(Finding 6)*
4. **Decode context conditioning** — batch 0's "true context" is the last token only; the prompt and full history never reach the model in distributed mode. *(Finding 8)*
5. **Repeated-token output fidelity** in the distributed token buffer. *(Finding 7, empirically verified)*
6. **Multi-batch prefill semantics** — chunk-boundary predictions concatenated as output; no identity-checked collection. *(Finding 9)*
7. **Robustness of acceptance under fault** — a lost/stale result can let an unverified-guess-context token through the collection layer. *(Finding 5)*

## 14. What now DOES work (validated positives)

1. **`verify_speculative_tokens`** — logically sound contamination-free acceptance; 7/7 unit tests. *(Finding 3)*
2. **Decode overlap firing** — the shipped loop genuinely fires `max(2, lookahead_window)` prefix-context micro-batches before collecting; 3/3 loop tests. *(Finding 3)*
3. **Codec** — binary SYNP header, INT8 quantization, zlib/LZ4, validation; 11/11 codec + 5/5 forwarder tests. *(Finding 4)*
4. **`micro_batch_id` propagation** — the implementation pass now threads it through `inference_state` (`distributed_engine.py:364-368`) and result payloads (`distributed_engine.py:394-398`), enabling correct sort-based pairing. *(diff + code read)*
5. **`_collect_result` against a real event router** — `test_distributed_pipeline.py::test_collect_result_receives_p2p_result` and `..._timeout_returns_none` pass (16/16 in that module). *(run)*

---

## 15. Executive summary (≈250 words)

The decode-contamination fix from the prior audit is real: `verify_speculative_tokens()` (`distributed_engine.py:28-64`) accepts a speculative token only when every guess feeding its context was confirmed by an earlier clean-context batch, and the decode loop now fires `num_decode_batches = max(2, lookahead_window)` micro-batches with prefix contexts `[last]+guesses[:i]`. All 10 tests in `test_decode_pipeline_overlap.py` and all 16 in `test_async_forwarding.py` pass, and the codec genuinely encodes tensors as 16-byte-SYNP binary blobs with INT8 quantization. But the distributed pipeline these features live in **cannot run with a real model**. Two independent, untested blockers: (1) the production P2P bridge hex-encodes `bytes` into JSON and never converts them back, so every `{"bin": ...}` tensor sent to a remote node arrives as a string and `decode_tensor` raises `TypeError` — silently swallowed (empirically reproduced); (2) `process_distributed_prompt` reads `token_ids.shape[1]` on the real engine's flattened 1-D `encode()` output, crashing every multi-node request into a silent single-node fallback (empirically reproduced). Additionally, distributed decode conditions on a truncated context (last token only, no prompt/KV), the token buffer drops legitimate repeated tokens, and the advertised "non-blocking, no-JSON" transport is unwired — the shipped path is JSON+hex broadcast with blocking per-peer drains. The verdict: **the decode logic is fixed and correct as a unit; the shipped multi-node decode pipeline still does not work.**

## 16. Top 5 findings (worst first)

1. **🔴 Blocker — Binary tensor path broken over the P2P bridge.** Bytes → hex-string in JSON, never restored on receive; `decode_tensor(str)` raises `TypeError`, swallowed by `event_router.publish`; every cross-node tensor is silently dropped. *`p2p_socket_bridge.py:370-381,420-431`; `node.py:499`; `codec.py:150`; `event_router.py:59-72`. Empirically reproduced.*
2. **🔴 Blocker — `process_distributed_prompt` crashes on the real engine.** `token_ids.shape[1]` on the engine's 1-D `encode()` → `IndexError`; multi-node requests silently fall back to single-node. *`distributed_engine.py:187`; `pytorch_hf_engine.py:517-524`; `node.py:861-869`. Empirically reproduced.*
3. **✅ Validated — Decode contamination fix is sound.** `verify_speculative_tokens` provably blocks unverified-context tokens; prefix-context batches; sort by `micro_batch_id`; 10/10 tests pass. *`distributed_engine.py:28-64,261-308`.*
4. **🟠 Medium — Result collection has no batch-identity validation.** Under a lost/stalled result, `verify_speculative_tokens` pairs by position, so a batch whose context contained an unverified guess can be accepted. *`distributed_engine.py:292-308,379-421`.*
5. **🟠 Medium — README transport claims describe unwired code.** The shipped path is JSON+hex broadcast with blocking per-peer `drain()`, not fire-and-forget binary sockets; `AsyncTensorForwarder`/`TensorTCPTransport` never carry a production tensor. *`p2p_socket_bridge.py:351-366`; `tcp_transport.py:220-243`; `node.py:236,356,1362`.*

## 17. Confidence

**7/10.**

Ground-truth tally: of the load-bearing conclusions, **16 are externally verified** (full source reads with line citations; three test-suite executions with exact pass counts — 10/10, 16/16, 16/16; two reproduction runs: bridge round-trip failure and decode 1-D shape crash both reproduced live; token-dedup data loss reproduced live; git diff confirms the fix landed in the working tree). **4 involve model judgment**: (a) the claim that a lost-result mislabeling can actually manifest in a real deployment (timing-dependent; reasoned, not observed); (b) the semantic-context claim about the HF engine's decode conditioning (inferred from code, no live model run); (c) the multi-batch prefill incoherence claim (inferred, not observed with default settings); (d) severity calibration. The two blockers are empirical facts; the positive validation of the verifier is a proof by inspection plus passing tests. Confidence is capped below 8 because no test in the suite drives the shipped transport end-to-end, so the empirically proven blockers were invisible to the repo's own verification, and the semantics claims could not be checked against a live multi-node run.
