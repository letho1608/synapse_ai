# Phase 2 Verification: Multi-Resource Election + Pipeline/Decode Overlap + 1F1B Training

**Agent Lane:** Specialist #3 — Multi-Resource Election, Pipeline/Decode Overlap, 1F1B Training  
**Phase 2 Commit:** `91ec971`  
**Analysis Date:** 2026-07-31  
**Files Examined:** 10 source files across `backend/synapse/inference/`, `backend/synapse/pipeline/`, `backend/synapse/orchestration/`, `backend/synapse/topology/`

---

## 1. Decode Pipeline Overlap (was B1 — Blocker)

**Phase 1 finding:** The README claimed pipeline micro-batch overlapping during decode, but the code only overlapped during prompt prefill. Decode processed tokens one batch at a time serially through the pipeline.

### Claimed fix: `num_decode_batches = max(2, lookahead_window)`

**File:** `backend/synapse/inference/distributed_engine.py`, lines 207–272

#### What the code does

The decode loop (starting line 207) implements a speculative-decode-inspired pipeline overlap scheme:

```python
# Line 209
lookahead_window = getattr(self, "lookahead_window", 3)

# Line 226
num_decode_batches = max(2, lookahead_window)  # At least 2 for overlap
```

1. **N-gram lookahead** (lines 214–224): Builds speculative token guesses using an n-gram (size 3) suffix match against the generated sequence history.
2. **Multiple decode batches** (lines 229–235): Creates `num_decode_batches` micro-batches, each with the same last real token plus a subset of speculative guesses.
3. **Fire all micro-batches first** (lines 238–244): Sends every micro-batch to the first pipeline stage without awaiting any results:
   ```python
   for db_idx, input_tensor in decode_batches:
       await self._send_input_to_stage(...)
   ```
4. **Then collect all results** (lines 248–253):
   ```python
   decode_results = []
   for _ in decode_batches:
       next_result = await self._collect_result(...)
       if next_result is not None:
           decode_results.append(next_result)
   ```

#### What this achieves

Because the pipeline has multiple stages (N ≥ 2), firing `N` micro-batches before collecting results enables stage-level concurrency: Stage 1 processes batch 0 while Stage 0 loads batch 1. The minimum of 2 micro-batches guarantees at least two pipeline stages can work in parallel during the decode phase.

**Verdict: PARTIALLY RESOLVED** — the mechanism for pipeline overlap during decode exists and is structurally correct. However, several significant concerns remain.

#### Finding D-1 (Medium): Speculative-context correctness issue

Each decode micro-batch receives a **different input context** because the n-gram lookahead appends varying numbers of guessed tokens. This means:

- **Batch 0** input: `[last_real_token]` → model predicts token_{T+1} using the true context.
- **Batch 1** input: `[last_real_token, guess_0]` → model predicts using a context contaminated with guess_0.
- **Batch N** input: `[last_real_token, guess_0, ..., guess_N]` → context contaminated with N guesses.

The token acceptance logic (lines 262–272) does:

```python
for db_idx, result in enumerate(decode_results):
    batch_guesses = guesses[db_idx:] if db_idx < len(guesses) else []
    accepted_count = 1  # Always accept the first actual prediction
    for g_idx in range(len(batch_guesses)):
        if g_idx < len(result.tokens) and result.tokens[g_idx] == batch_guesses[g_idx]:
            accepted_count += 1
        else:
            break
    verified_tokens = result.tokens[:accepted_count]
    all_accepted.extend(verified_tokens)
```

**Issue:** `all_accepted` accumulates verified tokens across **all** decode batches. If batch 0 produces `[token_A]` and batch 1 produces `[token_B]` (because the guess didn't match), the result is `all_accepted = [token_A, token_B]`. Both tokens get appended to `all_generated_tokens` without passing token_B back through the autoregressive loop. This means **two tokens may be generated per loop iteration without the second token being validated against the true autoregressive context** — effectively a broken speculative decoding implementation.

The `accepted_count = 1` comment says "Always accept the first actual prediction" — but this refers to the FIRST predicted token in each batch's output, which comes from a different (contaminated) context. The approach conflates the model's forward-pass computation (next-token prediction) with sequence-generation tokens.

**Impact:** When n-gram guesses are wrong (the common case for natural language), the system may silently accept incorrect tokens from batches with divergent contexts. When guesses are correct, the speculative tokens are validated and properly accepted. The reliability depends heavily on the n-gram predictor's accuracy.

#### Finding D-2 (Note): `lookahead_window` is not a class attribute

`DistributedPipelineController.__init__` (lines 76–87) does not define `self.lookahead_window`. Line 209 uses `getattr(self, "lookahead_window", 3)`, which returns the default 3 in isolation. The value is set externally by `DistributedPipelineExtension.process_prompt_distributed()` at `node_extension.py:84`:

```python
self._controller.lookahead_window = getattr(self._node, "lookahead_window", 3)
```

This works but is fragile — any code path that calls `process_distributed_prompt()` without going through `DistributedPipelineExtension` will get the default value (3), which coincidentally enables the overlap behavior. Not a bug per se, but a maintenance smell.

#### Finding D-3 (Low): Pipeline overhead from incorrect speculation

If the n-gram lookahead guesses are wrong (the typical case for natural language), the extra micro-batches produce no useful tokens (all speculative tokens fail verification), yet the system still pays:
- The cost of encoding and transmitting `num_decode_batches` inputs to Stage 0
- The pipeline stage compute for each micro-batch
- The result collection and verification overhead

The `accepted_count = 1` from each batch still means every batch contributes one "new" token, which as noted in D-1, may be from a divergent context.

---

## 2. 1F1B Training Wired to Production (was H2 — High)

**Phase 1 finding:** `generate_1f1b_schedule()` was only called from test files. No backward-pass execution path existed in `PipelineStage.process_loop()`. Training via pipeline parallelism was aspirational.

### Claimed fix: `DistributedTrainer.train_1f1b()` implemented

**File:** `backend/synapse/pipeline/gradient_sync.py`, lines 163–231

#### What the code does

`train_1f1b()` does the following:

1. **Creates a PipelineScheduler** and calls `generate_1f1b_schedule()` (line 189):
   ```python
   schedule = scheduler.generate_1f1b_schedule(num_stages, num_micro_batches, node_ids=node_ids)
   ```
2. **Creates async tasks** running `process_loop()` on each `PipelineStage` (lines 199–201):
   ```python
   loop_tasks = [
       asyncio.create_task(s.process_loop(stop_events[s.stage_id]))
       for s in stages
   ]
   ```
3. **Feeds micro-batches** to stage 0 based on the schedule, with `op="backward"` when `MicroBatchOp.BACKWARD` (line 210):
   ```python
   op_name = "backward" if slot.op == MicroBatchOp.BACKWARD else "forward"
   ```
4. **Collects losses** from the last stage (lines 219–223).

**Verdict: STRUCTURALLY CORRECT BUT NOT INTEGRATED** — the `train_1f1b()` method is properly implemented and **would** work if called. However, it is **dead code** from the perspective of the running system.

#### Finding H2-A (High — NOT RESOLVED): `train_1f1b()` is never called in production

The actual production training path is `DistributedPipelineExtension.train_step_distributed()` in `backend/synapse/pipeline/node_extension.py`, lines 262–393. This method:

- Calls `self._trainer.step()` (ring all-reduce DDP-style gradient synchronization), NOT `train_1f1b()`
- Uses `DistributedTrainer.step()` which averages gradients via ring all-reduce and applies weight updates locally
- Calls `self._node.inference_engine.train()` for local forward+backward on the node's shard

The only production call site is `chatgpt_api.py:2598`:
```python
loss = await ext.train_step_distributed(...)
```

This calls `train_step_distributed()` which does **not** invoke `train_1f1b()`. The 1F1B pipeline schedule exists in source code but has no caller in the production codebase. The Phase 1 finding "Pipeline training doesn't exist at runtime" remains true for the 1F1B variant specifically.

#### Finding H2-B (Low): Test coverage for `train_1f1b` is superficial

The test file `backend/tests/test_1f1b_production.py` only verifies:

- That `generate_1f1b_schedule` appears as a string literal in `gradient_sync.py`
- That `PipelineScheduler` and `num_micro_batches` appear in the source of `train_1f1b()`
- That `PipelineStage` constructor accepts `backward_fn`

No test **calls** `train_1f1b()` with actual `PipelineStage` objects. The test `test_train_1f1b_constructs_schedule` (line 53) checks string patterns with `inspect.getsource()` rather than exercising the runtime behavior.

---

## 3. PipelineStage Backward Support (part of H2)

**Claimed fix:** `PipelineStage` accepts `backward_fn`, `process_loop()` handles `"backward"` ops, `run_backward()` exists.

**File:** `backend/synapse/pipeline/stage.py`

#### Verification

**`__init__` accepts `backward_fn: Optional[Callable]`** (line 68):
```python
def __init__(
    self,
    stage_id: int,
    node_id: str,
    num_stages: int,
    forward_fn: Callable,
    send_tensor_fn: Callable,
    recv_tensor_fn: Optional[Callable] = None,
    backward_fn: Optional[Callable] = None,  # <-- LINE 69
):
```
✓ **CONFIRMED**

**`process_loop()` handles `op == "backward"`** (lines 119–131):
```python
op = mb.metadata.get("op", "forward") if mb.metadata else "forward"
if op == "backward" and self._backward_fn is not None:
    result = await self.run_backward(mb)
else:
    result = await self.run_forward(mb)
```
✓ **CONFIRMED**

**`run_backward()` method exists** (lines 140–161):
```python
async def run_backward(self, micro_batch: MicroBatch) -> StageResult:
    """Execute a backward pass on this stage's layer range."""
    if self._backward_fn is None:
        raise RuntimeError("No backward function configured for this stage")
    start = time.perf_counter()
    try:
        result_data = await self._backward_fn(micro_batch)
    except Exception:
        self.stats.failures += 1
        raise
    elapsed = (time.perf_counter() - start) * 1000
    self.stats.total_backwards += 1
    self.stats.total_latency_ms += elapsed
    # ... returns StageResult
```
✓ **CONFIRMED** — properly incrementing `total_backwards`, catching exceptions, tracking latency.

#### Finding BKD-1 (Note): `StageStats` tracks backwards but `total_forwards` is never reset vs `total_backwards`

In `StageResult` creation (stage.py:108): `step=micro_batch.step if hasattr(micro_batch, "step") else 0` — since `MicroBatch` doesn't have a `step` attribute by default (it has `batch_id`), this will always fall to 0. This is cosmetic and applies to both forward and backward results.

**Verdict:** FULLY RESOLVED. `PipelineStage` has correct backward support in its interface, processing loop, and statistics tracking.

---

## 4. Multi-Resource Election (was M4 — Medium)

**Phase 1 finding:** Master election ignored `ResourceScorer.compute_scores()` and used the simple `vram_gb + fp32_flops/100` formula instead.

### Claimed fix: `ElectionManager` accepts and uses `resource_scorer` callable

**File:** `backend/synapse/orchestration/election.py`, lines 12–56

#### What the code does

**`__init__` accepts `resource_scorer: Optional[Callable]`** (line 22):
```python
def __init__(self, node_id: str, event_router: EventRouter, compute_weight: float = 1.0,
             dynamic_weight: bool = False, resource_scorer: Optional[Callable] = None):
```
✓ **CONFIRMED**

**`set_base_compute_weight()` checks for scorer** (lines 41–56):
```python
def set_base_compute_weight(self, vram_gb: float, fp32_flops: float) -> None:
    simple_weight = vram_gb + (fp32_flops / 100.0)
    if self._resource_scorer is not None:
        try:
            multi_weight = self._resource_scorer(self.node_id)
            self.base_compute_weight = max(simple_weight, multi_weight * 1000.0)
            self.compute_weight = self.base_compute_weight
            return
        except Exception:
            pass
    self.base_compute_weight = simple_weight
    self.compute_weight = self.base_compute_weight
```
✓ **CONFIRMED:** Falls back to `vram_gb + fp32_flops/100` when `_resource_scorer is None`. Uses the scorer when available, with `max()` to ensure multi-resource score never drops below the simple baseline.

**Verdict: CORRECT IN INTERFACE, NOT INTEGRATED AT RUNTIME**

#### Finding M4-A (High — NOT FULLY RESOLVED): `set_base_compute_weight` is never called in production

The `Node.__init__()` constructor at `node.py:64` creates the `ElectionManager`:

```python
self.election_manager = ElectionManager(self.id, self.event_router)
```

This passes **no** `resource_scorer`. The `set_base_compute_weight()` method is **never called** anywhere in the production codebase (`grep` confirms zero production call sites). The only calls are in `tests/test_election_multiresource.py`.

**Impact:** The initial `compute_weight` defaults to `1.0` (the constructor default). Since `set_base_compute_weight()` is never called, the election weight stays at `1.0` for every node, making every candidate equally weighted. The entire multi-resource election fix — including the `resource_scorer` parameter, the scorer call path, and the fallback formula — is dead code.

The `_election_loop()` broadcasts `self.compute_weight` (line 103), which is always `1.0`. All nodes have equal weight, so the tiebreaker is the node ID string sort (line 117: `sorted_candidates = sorted(all_candidates.keys(), key=lambda x: (-all_candidates[x]["weight"], x))`). The first node alphabetically wins.

#### Finding M4-B (Medium): Magic scaling factor `1000.0`

In `set_base_compute_weight()` line 51:
```python
self.base_compute_weight = max(simple_weight, multi_weight * 1000.0)
```

The `1000.0` multiplier is undocumented. `ResourceScorer.compute_scores()` returns composite scores that are weighted sums of normalized values (each 0–1 range, possibly with multiplicative GPU utilization penalties). The actual composite score could be in any range. The `1000.0` multiplier seems designed to ensure `multi_weight` dominates `simple_weight`, but this is fragile:
- If the scorer returns values in 0–1 range, `multi_weight * 1000.0` = 0–1000, making it always ≥ `simple_weight` (which is ~16–160 for typical GPUs).
- If the scorer returns values already in a higher range (e.g., 0–100), the multiplier becomes 0–100,000, completely swamping the simple formula.

Without tests exercising the actual `ResourceScorer.compute_scores()` output range, this multiplier is a guess.

---

## 5. Cross-File Consistency

### `train_1f1b` importability

`backend/synapse/pipeline/__init__.py` exports `DistributedTrainer`, which contains `train_1f1b` as a method. The import chain works:
- `synapse.pipeline.DistributedTrainer` → `synapse.pipeline.gradient_sync.DistributedTrainer` → `.train_1f1b()`
✓ **CONFIRMED importable**

However, since the method is never called, importability alone is insufficient.

### `compute_partition(method="ilp")` in production

`compute_partition(method="ilp")` is **not** referenced anywhere in the distributed engine or node production code:

- `backend/synapse/inference/distributed_engine.py` — does not call `compute_partition` at all.
- `backend/synapse/orchestration/node.py` — `_compute_lacp_partition()` (line 1412) calls `compute_partition()` without passing `method="ilp"`, or uses `TFLOPSProportionalPartitioner` when `self.tflops_proportional` is `True` (the default).
- `backend/synapse/pipeline/node_extension.py` — imports `compute_partition` but only uses it via `self._node._compute_lacp_partition()`.

The ILP path is only exercised in `tests/test_partitioner_ilp.py`. This is consistent with the default `method="lrm"` (largest remainder method), but the `method="ilp"` option exists and is tested.

**Finding CI-1 (Note):** The ILP partitioner is available and tested but is never selected in production code. It is a "choose your partitioner" option that defaults to LRM.

---

## Executive Summary

Phase 2 commit `91ec971` made structural improvements to all three findings but left critical integration gaps.

**Decode pipeline overlap (was B1):** The code now fires multiple micro-batches before collecting results during decode, enabling true stage-level pipeline parallelism. The minimum of 2 micro-batches ensures ≥2 stages can work concurrently. However, the speculative n-gram lookahead introduces a correctness risk: micro-batches with different input contexts produce different model outputs, and the token-acceptance logic may silently accept incorrect tokens from divergent contexts. The overlap mechanism is correct in structure but the speculative decoding integration is flawed.

**1F1B training (was H2):** `DistributedTrainer.train_1f1b()` is properly implemented — it calls `generate_1f1b_schedule()`, creates `PipelineStage` tasks with `process_loop()`, and handles backward operations. `PipelineStage` has full backward support (`backward_fn`, `process_loop("backward")`, `run_backward()`). However, `train_1f1b()` is **never called from any production code path**. The actual training pipeline uses DDP-style ring all-reduce via `DistributedTrainer.step()` and `train_step_distributed()`. The 1F1B schedule remains aspirational.

**Multi-resource election (was M4):** `ElectionManager` correctly accepts a `resource_scorer` callable, uses it in `set_base_compute_weight()` with a proper fallback to the simple VRAM+FLOPs formula. However, `set_base_compute_weight()` is **never called from production code**. The `Node` constructor omits the `resource_scorer` parameter and never calls `set_base_compute_weight()`, so every node's election weight stays at the default `1.0`, making election purely alphabetical.

**Bottom line:** The raw building blocks are in place — the data structures, schedule generators, and class interfaces all support the claimed features. But the integration into the runtime `Node` class is incomplete for both the 1F1B training path and the multi-resource election. The decode overlap works in practice but has correctness concerns from its speculative decoding integration.

---

## Top 5 Most Concerning Findings

1. **H2-A (High): `train_1f1b()` is dead code** — The function is never called from any production path. Phase 1 finding "Pipeline training doesn't exist at runtime" remains true for 1F1B. The only active training path is DDP-style ring all-reduce.

2. **M4-A (High): `set_base_compute_weight()` is never called** — Despite the `ElectionManager` API correctly supporting `resource_scorer`, the production `Node` constructor never calls `set_base_compute_weight()` and never passes a `resource_scorer`. All nodes have election weight `1.0`, making elections alphabetical.

3. **D-1 (Medium): Speculative decode may accept incorrect tokens** — Micro-batches with different n-gram lookahead contexts produce different model predictions. The acceptance logic accumulates "first token from each batch" into the generated sequence, which can include tokens from divergent/contaminated contexts. This is a latent correctness bug.

4. **M4-B (Medium): Undocumented magic scaling factor `1000.0`** — The `multi_weight * 1000.0` multiplier in `set_base_compute_weight()` has no justification. If `ResourceScorer.compute_scores()` output characteristics change, this silently breaks election weight calculations.

5. **H2-B (Low): No runtime test for `train_1f1b`** — The test coverage only checks for string patterns in the source code. No test actually calls `train_1f1b()` with live `PipelineStage` objects to verify schedule execution, tensor flow, or loss computation.

---

## Confidence Rating

**Confidence: 6/10**

**Reasoning:** The code changes are structurally correct at the data-structure and API level — the scheduler generates correct 1F1B schedules, the PipelineStage handles backward operations, the ElectionManager accepts resource scorers. However, two out of three fixes (1F1B training and multi-resource election) are disconnected from the runtime: the production `Node` class does not invoke the new code paths. The third fix (decode overlap) has a substantive correctness concern. Without seeing these code paths exercised end-to-end, my confidence is tempered.

**Ground-truth tally:**
- Code assertions verified directly: 8 of 8 structural claims confirmed (backward_fn acceptance, process_loop backward handling, run_backward existence, generate_1f1b_schedule call in method, resource_scorer parameter, fallback logic, decode batch firing, multi-batch collection)
- Runtime integrations verified: 1 of 3 (decode overlap fires; 1F1B training and multi-resource election do not)
- Test coverage gap: `train_1f1b()` and `set_base_compute_weight()` have no integration tests
- Correctness concerns found: 1 (speculative decode token acceptance)
