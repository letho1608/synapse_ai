"""
LACP — Distributed Inference Engine

Extends PyTorchHFEngine with distributed multi-node pipeline execution.
Coordinates inference across nodes assigned to contiguous layer ranges.

Key additions:
- _process_shard_range: execute forward pass for a subset of layers
- distributed_infer: coordinate full distributed inference pipeline
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable

from synapse.inference.shard import Shard
from synapse.inference.inference_engine import InferenceEngine
from synapse.transport.codec import encode_tensor
from synapse.transport.result_router import RoutedResult
from synapse.helpers import DEBUG


def verify_speculative_tokens(
    guesses: List[int],
    batch_tokens: List[List[int]],
) -> List[int]:
    """
    Verify speculative decode micro-batches against the true context.

    batch_tokens[i] holds the tokens predicted by the micro-batch whose
    context was [last] + guesses[:i] (batch 0 sees only the true context
    [last]). A token is accepted only when every guess that feeds its
    context has been verified by the previous batch's prediction, so no
    token predicted from a contaminated context is ever appended to the
    output.

    Returns the accepted token sequence (in order). The result is empty
    when nothing can be verified (e.g. batch 0 produced no tokens).
    """
    accepted: List[int] = []
    if not batch_tokens or not batch_tokens[0]:
        return accepted
    # Batch 0 predicts from the true context [last]: always valid.
    accepted.append(batch_tokens[0][0])
    for i in range(1, len(batch_tokens)):
        prev_guess = guesses[i - 1] if i - 1 < len(guesses) else None
        if prev_guess is None or not batch_tokens[i - 1]:
            break
        # Batch i-1's prediction must confirm guess i-1 before we trust
        # batch i (whose context assumed guesses[:i] were all correct).
        if batch_tokens[i - 1][0] != prev_guess:
            break
        if batch_tokens[i]:
            accepted.append(batch_tokens[i][0])
    return accepted


class _ResultWaiter:
    """One step's result, subscribed before the step is sent.

    Accepts the last stage's sampled tokens for this micro-batch id, or a
    kv_miss reply from any stage for it."""

    def __init__(self, event_router, request_id: str, node_id: str,
                 micro_batch_id: int, last_stage_start: int):
        self._router = event_router
        self._topic = f"synapse/inference/result/{request_id}"
        self._request_id = request_id
        self._node_id = node_id
        self._mb = micro_batch_id
        self._last = last_stage_start
        self._event = asyncio.Event()
        self._result: Optional[RoutedResult] = None
        self._router.subscribe(self._topic, self._on)

    def _on(self, event: Any) -> None:
        data = event.data if hasattr(event, "data") else event
        if not isinstance(data, dict) or self._event.is_set():
            return
        if data.get("micro_batch_id", 0) != self._mb:
            return
        miss = bool(data.get("kv_miss"))
        src = data.get("source_stage_id")
        if not miss and src is not None and src != self._last:
            return
        self._result = RoutedResult(
            request_id=self._request_id, orchestrator_id=self._node_id,
            micro_batch_id=self._mb, source_stage_id=self._last,
            tokens=list(data.get("tokens") or []),
            is_finished=bool(data.get("is_finished", False)),
            metadata={"kv_miss": miss, "node_id": data.get("node_id")},
        )
        self._event.set()

    async def wait(self, timeout: float) -> Optional[RoutedResult]:
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self._result

    def close(self) -> None:
        try:
            self._router.unsubscribe(self._topic, self._on)
        except Exception:
            pass


class DistributedInferenceSession:
    """
    Per-request state for distributed inference across multiple nodes.

    Tracks:
    - Active shard assignments per node
    - KV cache indices per node (which layers cache is held where)
    - Micro-batch flow state
    - Result routing info
    """

    def __init__(
        self,
        request_id: str,
        orchestrator_id: str,
        model_id: str,
        num_stages: int,
        num_micro_batches: int = 1,
    ):
        self.request_id = request_id
        self.orchestrator_id = orchestrator_id
        self.model_id = model_id
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.start_time = time.time()

        self.total_tokens_generated = 0
        self.is_finished = False
        self.error: Optional[Exception] = None


class DistributedPipelineController:
    """
    Coordinates distributed inference across multiple nodes using pipeline parallelism.

    The controller:
    1. Takes the partition (layer range per node) from the node
    2. Sends each step's input to the first stage; stages forward hidden
       states to the next node
    3. Awaits each step's result from the last stage (subscribe before
       send) and streams verified tokens
    """

    def __init__(
        self,
        node_id: str,
        inference_engine: InferenceEngine,
        event_router: Any,
    ):
        self.node_id = node_id
        self.inference_engine = inference_engine
        self.event_router = event_router
        self._sessions: Dict[str, DistributedInferenceSession] = {}
        # request_id -> [[node_id, start, end, n_layers], ...] in stage order.
        self._plans: Dict[str, List[List[Any]]] = {}
        self._sampling: Dict[str, Dict[str, Any]] = {}
        # request_id -> {"steps", "drafted", "accepted_draft", "kv_miss"}
        self.decode_stats: Dict[str, Dict[str, int]] = {}
        self.step_timeout = 60.0
        self.liveness_poll_s = 1.0
        self.max_failovers = 2
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def process_distributed_prompt(
        self,
        base_shard: Shard,
        prompt: str,
        request_id: str,
        partition_assignments: List[Tuple[str, Shard]],
        max_generate_tokens: int = 1024,
        on_token_callback: Optional[Callable] = None,
        num_micro_batches: int = 1,
        sampling_state: Optional[Dict[str, Any]] = None,
        replan_fn: Optional[Callable] = None,
        alive_fn: Optional[Callable[[str], bool]] = None,
    ) -> List[int]:
        """
        Run distributed inference across multiple nodes.

        Every step is one pipeline pass of [prompt + history + draft]
        (see _run_step); the prompt is step 0.

        Args:
            base_shard: Shard with full model info
            prompt: The input prompt
            request_id: Unique request ID
            partition_assignments: List of (node_id, shard) tuples in stage order
            max_generate_tokens: Max tokens to generate
            on_token_callback: Called with (request_id, tokens, is_finished)
            num_micro_batches: kept for API compatibility (unused)
            replan_fn: (current_assignments, excluded_node_ids) -> new
                assignments or None. Used when a stage node dies mid-request:
                the step is resent to the new plan with full hidden states
                and generation continues (Petals-style failover).
            alive_fn: node_id -> bool (heartbeat view), polled while waiting
                so a dead stage is detected in seconds, not at the timeout.

        Returns:
            List of generated token IDs
        """
        num_stages = len(partition_assignments)
        if num_stages == 0:
            return []

        session = DistributedInferenceSession(
            request_id=request_id,
            orchestrator_id=self.node_id,
            model_id=base_shard.model_id,
            num_stages=num_stages,
            num_micro_batches=num_micro_batches,
        )
        self._sessions[request_id] = session
        # Sampling knobs (e.g. temperature) ride every stage's inference_state:
        # the LAST stage samples, on another node, with only what we send it.
        self._sampling[request_id] = dict(sampling_state or {})
        self._plans[request_id] = [
            [nid, int(sh.start_layer), int(sh.end_layer), int(sh.n_layers)]
            for nid, sh in partition_assignments
        ]

        all_generated_tokens: List[int] = []

        async def emit(tokens: List[int], finished: bool) -> None:
            if not on_token_callback:
                return
            try:
                if asyncio.iscoroutinefunction(on_token_callback):
                    await on_token_callback(request_id, tokens, finished)
                else:
                    on_token_callback(request_id, tokens, finished)
            except Exception as e:
                if DEBUG >= 1: print(f"[{request_id}] on_token_callback error: {e}")

        try:
            token_ids = await self.inference_engine.encode(base_shard, prompt)
            if token_ids is None or np.asarray(token_ids).size == 0:
                return []
            prompt_ids = [int(t) for t in np.asarray(token_ids).reshape(-1)]
            stop_ids = self._stop_ids()
            lookahead = max(0, int(getattr(self, "lookahead_window", 3)))
            stats = self.decode_stats.setdefault(request_id, {
                "steps": 0, "drafted": 0, "accepted_draft": 0, "kv_miss": 0})

            # One pipeline pass per step: [prompt + history + draft] goes in,
            # the last stage samples every draft position, and we keep the
            # longest verified prefix (+1 token). A pass with an empty draft
            # is plain decoding; the prompt is simply step 0.
            step = 0
            plan = list(partition_assignments)
            excluded: set = set()
            force_full = False
            stats.setdefault("failovers", 0)
            while not session.is_finished and len(all_generated_tokens) < max_generate_tokens:
                draft = self._ngram_draft(prompt_ids + all_generated_tokens, lookahead)
                room = max_generate_tokens - len(all_generated_tokens)
                draft = draft[: max(0, room - 1)]
                context = prompt_ids + all_generated_tokens + draft
                sampled, dead = await self._run_step(
                    request_id, plan, context, draft,
                    all_generated_tokens, step, alive_fn=alive_fn, full=force_full)
                step += 1
                if sampled is None:
                    # A stage died (heartbeats) or the step timed out: re-plan
                    # around the dead nodes and resend this same step with
                    # full hidden states; the new stages rebuild their KV.
                    new_plan = None
                    if replan_fn is not None and stats["failovers"] < self.max_failovers:
                        excluded |= set(dead)
                        try:
                            new_plan = replan_fn(list(plan), set(excluded)) if dead else list(plan)
                        except Exception as e:
                            if DEBUG >= 1: print(f"[{request_id}] replan failed: {e}")
                            new_plan = None
                    if new_plan:
                        stats["failovers"] += 1
                        if DEBUG >= 1:
                            print(f"[{request_id}] [FAILOVER] dead={sorted(dead)} new plan="
                                  f"{[(n, s.start_layer, s.end_layer) for n, s in new_plan]}")
                        plan = list(new_plan)
                        self._plans[request_id] = [
                            [nid, int(sh.start_layer), int(sh.end_layer), int(sh.n_layers)]
                            for nid, sh in plan]
                        force_full = True
                        continue
                    session.error = session.error or TimeoutError("pipeline step timed out")
                    break
                force_full = False
                accepted = verify_speculative_tokens(draft, [[t] for t in sampled])
                if not accepted:
                    break
                stats["steps"] += 1
                stats["drafted"] += len(draft)
                stats["accepted_draft"] += len(accepted) - 1
                finished = False
                for i, t in enumerate(accepted):
                    if t in stop_ids:
                        accepted = accepted[:i]
                        finished = True
                        break
                accepted = accepted[:room]
                all_generated_tokens.extend(accepted)
                if accepted:
                    await emit(accepted, False)
                tail = all_generated_tokens[-10:]
                if len(tail) == 10 and len(set(tail)) == 1:
                    finished = True  # same token 10x: a repetition loop
                if finished:
                    break

            session.is_finished = True
            await emit([], True)

        except Exception as e:
            session.error = e
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
        finally:
            self._sessions.pop(request_id, None)
            self._plans.pop(request_id, None)
            self._sampling.pop(request_id, None)
            # Clear KV cache on all nodes
            await self.event_router.publish(
                "synapse/cluster/cache/clear",
                __import__('json').dumps({"request_id": request_id})
            )

        return all_generated_tokens

    def _stop_ids(self) -> set:
        from synapse.inference.inference_engine import stop_token_ids
        return stop_token_ids(getattr(self.inference_engine, "tokenizer", None))

    @staticmethod
    def _ngram_draft(history: List[int], k: int, n: int = 3) -> List[int]:
        """Prompt-lookup draft: find the latest earlier occurrence of the last
        n tokens and propose the k tokens that followed it."""
        if k <= 0 or len(history) < n + 1:
            return []
        suffix = history[-n:]
        for i in range(len(history) - n - 1, -1, -1):
            if history[i:i + n] == suffix:
                follow = history[i + n: i + n + k]
                if follow:
                    return list(follow)
        return []

    async def _run_step(self, request_id: str, partition_assignments, context: List[int],
                        draft: List[int], generated: List[int], step: int,
                        alive_fn: Optional[Callable[[str], bool]] = None, full: bool = False):
        """Send one context through the pipeline.

        Returns (tokens, dead): the sampled tokens for the last len(draft)+1
        positions, or (None, dead_node_ids) when the step did not complete
        (dead is empty on a plain timeout).

        Stages forward only recomputed hidden rows (delta_rows). A stage that
        lacks the cached prefix answers kv_miss (e.g. it just replaced a dead
        node), and the step is resent once with full hidden states."""
        first_node, first_shard = partition_assignments[0]
        last_node, last_shard = partition_assignments[-1]
        stats = self.decode_stats.setdefault(request_id, {})
        modes = (False,) if full else (True, False)
        for attempt, delta_rows in enumerate(modes):
            mb_id = step * 2 + attempt
            waiter = self._result_waiter(request_id, mb_id, last_shard.start_layer)
            try:
                await self._send_input_to_stage(
                    first_node, first_shard, np.array([context], dtype=np.int64),
                    request_id, is_prompt=(step == 0), microbatch_id=mb_id,
                    extra_state={"verify_n": len(draft) + 1, "draft": list(draft),
                                 "gen_ids": list(generated), "delta_rows": delta_rows},
                )
                res, dead = await self._await_result(waiter, partition_assignments, alive_fn)
            finally:
                waiter.close()
            if res is None:
                return None, dead
            if res.metadata.get("kv_miss"):
                stats["kv_miss"] = stats.get("kv_miss", 0) + 1
                continue
            return list(res.tokens), set()
        return None, set()

    async def _await_result(self, waiter, plan, alive_fn):
        """Wait for the step's result, polling stage liveness every second
        so a dead node is noticed after a few missed heartbeats instead of
        after the full step timeout. Returns (result, dead_node_ids)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.step_timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None, set()
            res = await waiter.wait(min(self.liveness_poll_s, remaining))
            if res is not None:
                return res, set()
            if alive_fn is not None:
                dead = {nid for nid, _ in plan if nid != self.node_id and not alive_fn(nid)}
                if dead:
                    return None, dead

    async def _send_input_to_stage(
        self,
        target_node_id: str,
        target_shard: Shard,
        input_data: np.ndarray,
        request_id: str,
        is_prompt: bool = False,
        microbatch_id: int = 0,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        topic = f"synapse/inference/tensor/{target_node_id}"

        # Per-row INT8 by default, only toward peers that negotiated it.
        from synapse.transport.router import should_quantize
        quantize = should_quantize(str(input_data.dtype), target_id=target_node_id,
                                   self_id=self.node_id)
        
        encoded_payload = encode_tensor(
            input_data,
            metadata={
                "request_id": request_id,
                "shard": target_shard.to_dict(),
                "inference_state": {
                    "is_prompt": is_prompt, "microbatch_id": microbatch_id,
                    # The request carries its own stage plan, so every hop
                    # runs exactly the layers the orchestrator assigned and
                    # forwards to the right owner, whatever its local view.
                    "partition_plan": self._plans.get(request_id),
                    **self._sampling.get(request_id, {}),
                    **(extra_state or {}),
                },
                "_orchestrator_id": self.node_id,
                "microbatch_id": microbatch_id,
                "_sent_at": time.time(),
            },
            quantize_int8=quantize
        )
        await self.event_router.publish(topic, {"bin": encoded_payload})

    def _result_waiter(self, request_id: str, micro_batch_id: int, last_stage_start: int):
        """Subscribe to the request's result topic now; wait() later."""
        return _ResultWaiter(self.event_router, request_id, self.node_id,
                             micro_batch_id, last_stage_start)

