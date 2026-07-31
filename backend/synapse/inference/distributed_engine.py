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
from synapse.pipeline.scheduler import MicroBatch, PipelineSchedule, PipelineScheduler
from synapse.pipeline.stage import PipelineStage, PipelineExecutor, StageResult
from synapse.transport.codec import encode_tensor, decode_tensor, TensorMessage, AsyncTensorForwarder
from synapse.transport.result_router import ResultRouter, RoutedResult
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

        self.results: Dict[int, StageResult] = {}
        self.total_tokens_generated = 0
        self.is_finished = False
        self.error: Optional[Exception] = None


class DistributedPipelineController:
    """
    Coordinates distributed inference across multiple nodes using pipeline parallelism.

    The controller:
    1. Computes LACP partition to assign layer ranges to nodes
    2. Uses PipelineScheduler to create a staggered inference schedule
    3. Creates PipelineStage per node, chains via AsyncTensorForwarder
    4. Executes pipeline via PipelineExecutor for efficient multi-batch
    5. Collects results from the last-stage node
    6. Routes results to the orchestrator via ResultRouter
    """

    def __init__(
        self,
        node_id: str,
        inference_engine: InferenceEngine,
        event_router: Any,
        forwarder: Optional[AsyncTensorForwarder] = None,
    ):
        self.node_id = node_id
        self.inference_engine = inference_engine
        self.event_router = event_router
        self.forwarder = forwarder or AsyncTensorForwarder(node_id)
        self.result_router = ResultRouter(node_id)
        self._sessions: Dict[str, DistributedInferenceSession] = {}
        self._running = False

    async def start(self) -> None:
        self._running = True
        await self.result_router.start()

    async def stop(self) -> None:
        self._running = False
        await self.result_router.stop()

    async def process_distributed_prompt(
        self,
        base_shard: Shard,
        prompt: str,
        request_id: str,
        partition_assignments: List[Tuple[str, Shard]],
        max_generate_tokens: int = 1024,
        on_token_callback: Optional[Callable] = None,
        num_micro_batches: int = 1,
    ) -> List[int]:
        """
        Run distributed inference across multiple nodes.

        For prompt (prefill) phase, splits input into micro-batches for
        pipeline-parallel overlapping. For decode (autoregressive) phase,
        processes one token at a time since each depends on previous.

        Args:
            base_shard: Shard with full model info
            prompt: The input prompt
            request_id: Unique request ID
            partition_assignments: List of (node_id, shard) tuples from LACP partitioner
            max_generate_tokens: Max tokens to generate
            on_token_callback: Called with (request_id, tokens, is_finished)
            num_micro_batches: Number of micro-batches to split prompt into (>=1)

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

        if on_token_callback:
            self.result_router.register_token_callback(request_id, on_token_callback)

        all_generated_tokens: List[int] = []

        try:
            token_ids = await self.inference_engine.encode(base_shard, prompt)
            if token_ids is None or token_ids.size == 0:
                return []

            # Real encoders (e.g. PyTorchHFEngine.encode) return a flattened
            # 1-D token array; the pipeline expects (1, seq_len). Normalize so
            # the prompt phase below (token_ids.shape[1], slicing) works and
            # multi-node inference does not crash and silently fall back to a
            # single node.
            if token_ids.ndim == 1:
                token_ids = token_ids.reshape(1, -1)

            first_stage_node_id, first_stage_shard = partition_assignments[0]
            last_stage_node_id, last_stage_shard = partition_assignments[-1]

            # Prompt phase: use micro-batching for pipeline parallelism
            prompt_length = token_ids.shape[1]
            if num_micro_batches > 1 and prompt_length >= num_micro_batches:
                mb_size = max(1, prompt_length // num_micro_batches)
                micro_batches = []
                for mb_idx in range(num_micro_batches):
                    start_idx = mb_idx * mb_size
                    end_idx = (mb_idx + 1) * mb_size if mb_idx < num_micro_batches - 1 else prompt_length
                    mb_tokens = token_ids[:, start_idx:end_idx]
                    micro_batches.append((mb_idx, mb_tokens))

                # Pipeline fill: send micro-batches to first stage sequentially
                prompt_results: List[int] = []
                async def collect_prompt_result() -> None:
                    res = await self._collect_result(
                        request_id, last_stage_node_id, last_stage_shard, timeout=120.0,
                    )
                    if res is not None and res.tokens:
                        prompt_results.extend(res.tokens)

                for mb_idx, mb_tokens in micro_batches:
                    await self._send_input_to_stage(
                        first_stage_node_id, first_stage_shard,
                        mb_tokens, request_id, is_prompt=True,
                        microbatch_id=mb_idx,
                    )

                for _ in micro_batches:
                    await collect_prompt_result()

                if prompt_results:
                    all_generated_tokens.extend(prompt_results)
                    if on_token_callback:
                        try:
                            if asyncio.iscoroutinefunction(on_token_callback):
                                await on_token_callback(request_id, prompt_results, False)
                            else:
                                on_token_callback(request_id, prompt_results, False)
                        except Exception as e:
                            if DEBUG >= 1: print(f"[{request_id}] on_token_callback error: {e}")
            else:
                # Single batch: send all tokens at once
                await self._send_input_to_stage(
                    first_stage_node_id, first_stage_shard,
                    token_ids, request_id, is_prompt=True,
                    microbatch_id=0,
                )

                result = await self._collect_result(
                    request_id, last_stage_node_id, last_stage_shard, timeout=60.0,
                )
                if result is not None and result.tokens:
                    all_generated_tokens.extend(result.tokens)

            # Decode phase: true pipeline-parallel micro-batch overlapping
            # Instead of sending one batch and waiting, fire multiple micro-batches
            # into the pipeline before collecting results. Each stage processes
            # concurrently: Stage 1 works on batch N while Stage 0 loads batch N+1.
            lookahead_window = getattr(self, "lookahead_window", 3)
            ngram_size = 3
            
            while not session.is_finished and len(all_generated_tokens) < max_generate_tokens:
                # Build N-gram lookahead guesses
                guesses = []
                if lookahead_window > 0 and len(all_generated_tokens) >= ngram_size + 1:
                    history = all_generated_tokens
                    suffix = history[-ngram_size:]
                    for i in range(len(history) - ngram_size - 1, 0, -1):
                        if history[i:i+ngram_size] == suffix:
                            guesses = history[i+ngram_size : i+ngram_size+lookahead_window]
                            if guesses:
                                break
                
                # Pipeline-parallel decode: fire multiple micro-batches concurrently
                # so stages overlap (Stage 1 processes batch N while Stage 0 processes batch N+1)
                num_decode_batches = max(2, lookahead_window)  # At least 2 for overlap
                decode_batches = []
                
                for db_idx in range(num_decode_batches):
                    # Each micro-batch extends the TRUE context by a PREFIX of
                    # guesses: batch 0 sees [last], batch 1 sees [last, g0],
                    # batch 2 sees [last, g0, g1], ... A token is only ever
                    # accepted after every guess feeding its context has been
                    # verified by the previous batch's prediction, so no token
                    # from a contaminated context can leak into the output.
                    batch_guesses = guesses[:db_idx]
                    candidate_tokens = [all_generated_tokens[-1]] + batch_guesses
                    input_tensor = np.array([candidate_tokens], dtype=np.int64)
                    decode_batches.append((db_idx, input_tensor))
                
                # Fire ALL micro-batches into the pipeline immediately (no waiting)
                # Stage 0 sends batch 0, then immediately sends batch 1, etc.
                # While Stage 0 loads batch 1, Stage 1 is already processing batch 0.
                for db_idx, input_tensor in decode_batches:
                    await self._send_input_to_stage(
                        first_stage_node_id, first_stage_shard,
                        input_tensor, request_id, is_prompt=False,
                        microbatch_id=db_idx,
                    )
                
                # Collect results from the last stage as they complete
                decode_results = []
                for _ in decode_batches:
                    next_result = await self._collect_result(
                        request_id, last_stage_node_id, last_stage_shard, timeout=30.0,
                    )
                    if next_result is not None:
                        decode_results.append(next_result)
                
                if not decode_results:
                    break

                # Verify guesses across all micro-batch results. Order by
                # micro-batch id so batch i's prediction is matched against
                # the context [last] + guesses[:i] regardless of arrival order.
                decode_results.sort(key=lambda r: r.micro_batch_id)
                all_accepted = verify_speculative_tokens(
                    guesses, [r.tokens for r in decode_results]
                )
                if not all_accepted:
                    break
                
                all_generated_tokens.extend(all_accepted)
                
                if on_token_callback:
                    try:
                        if asyncio.iscoroutinefunction(on_token_callback):
                            await on_token_callback(request_id, all_accepted, False)
                        else:
                            on_token_callback(request_id, all_accepted, False)
                    except Exception:
                        pass

                if any(r.is_finished for r in decode_results):
                    break

            session.is_finished = True
            if on_token_callback:
                try:
                    if asyncio.iscoroutinefunction(on_token_callback):
                        await on_token_callback(request_id, [], True)
                    else:
                        on_token_callback(request_id, [], True)
                except Exception:
                    pass

        except Exception as e:
            session.error = e
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
        finally:
            self._sessions.pop(request_id, None)
            self.result_router.unregister_token_callback(request_id)
            # Clear KV cache on all nodes
            await self.event_router.publish(
                "synapse/cluster/cache/clear",
                __import__('json').dumps({"request_id": request_id})
            )

        return all_generated_tokens

    async def _send_input_to_stage(
        self,
        target_node_id: str,
        target_shard: Shard,
        input_data: np.ndarray,
        request_id: str,
        is_prompt: bool = False,
        microbatch_id: int = 0,
    ) -> None:
        topic = f"synapse/inference/tensor/{target_node_id}"
        
        # Use INT8 quantization for hidden states to slash bandwidth on slow links
        is_hidden_states = np.issubdtype(input_data.dtype, np.floating)
        quantize = is_hidden_states 
        
        encoded_payload = encode_tensor(
            input_data,
            metadata={
                "request_id": request_id,
                "shard": target_shard.to_dict(),
                "inference_state": {"is_prompt": is_prompt, "microbatch_id": microbatch_id},
                "_orchestrator_id": self.node_id,
                "microbatch_id": microbatch_id,
                "_sent_at": time.time(),
            },
            quantize_int8=quantize
        )
        await self.event_router.publish(topic, {"bin": encoded_payload})

    async def _collect_result(
        self,
        request_id: str,
        last_stage_node_id: str,
        last_stage_shard: Shard,
        timeout: float = 30.0,
    ) -> Optional[RoutedResult]:
        result_event = asyncio.Event()
        collected_result: List[RoutedResult] = []

        # Listen on P2P result topic. The last node in the chain publishes
        # to "synapse/inference/result/{request_id}" via process_inference_result,
        # which the orchestrator's on_p2p_result handler receives.
        result_topic = f"synapse/inference/result/{request_id}"

        def on_p2p_result(event: Any):
            data = event.data if hasattr(event, 'data') else event
            tokens = data.get("tokens", [])
            is_finished = data.get("is_finished", False)
            micro_batch_id = data.get("micro_batch_id", 0)
            if DEBUG >= 1:
                print(f"[{request_id}] [DISTRIBUTED] P2P result received: tokens={tokens}, finished={is_finished}")
            collected_result.append(
                RoutedResult(
                    request_id=request_id,
                    orchestrator_id=self.node_id,
                    micro_batch_id=micro_batch_id,
                    source_stage_id=last_stage_shard.start_layer,
                    tokens=list(tokens),
                    is_finished=is_finished,
                )
            )
            result_event.set()

        self.event_router.subscribe(result_topic, on_p2p_result)

        try:
            await asyncio.wait_for(result_event.wait(), timeout=timeout)
            return collected_result[0] if collected_result else None
        except asyncio.TimeoutError:
            if DEBUG >= 1:
                print(f"[{request_id}] [DISTRIBUTED] Timeout waiting for result from stage on {last_stage_node_id}")
            return None
        finally:
            self.event_router.unsubscribe(result_topic, on_p2p_result)

    def get_session(self, request_id: str) -> Optional[DistributedInferenceSession]:
        return self._sessions.get(request_id)

    async def run_inference_pipeline(
        self,
        base_shard: Shard,
        input_data: np.ndarray,
        request_id: str,
        partition_assignments: List[Tuple[str, Shard]],
        num_micro_batches: int = 1,
        is_prompt: bool = False,
    ) -> List[np.ndarray]:
        """
        Execute a full pipeline inference schedule across all stages.

        Uses PipelineExecutor to run a staggered forward-only pipeline.
        All stages process micro-batches concurrently. First stage gets
        the input, intermediate stages chain via send/recv, last stage
        produces final results.

        Returns:
            List of result tensors, one per micro-batch
        """
        num_stages = len(partition_assignments)
        if num_stages == 0:
            return [input_data]

        scheduler = PipelineScheduler(num_micro_batches=num_micro_batches)
        node_ids = [pid for pid, _ in partition_assignments]
        schedule = scheduler.generate_inference_schedule(
            num_stages, num_micro_batches, node_ids=node_ids
        )

        stages: List[PipelineStage] = []
        for stage_idx in range(num_stages):
            _, shard = partition_assignments[stage_idx]
            is_last = stage_idx == num_stages - 1

            async def make_forward_fn(s: Shard, rid: str):
                async def forward(mb: MicroBatch) -> np.ndarray:
                    result, _ = await self.inference_engine.infer_tensor(
                        rid, s, mb.data,
                        {"is_prompt": mb.metadata.get("is_prompt", is_prompt)}
                    )
                    return result
                return forward

            forward_fn = await make_forward_fn(shard, request_id)

            if is_last:
                final_results: List[np.ndarray] = []
                async def last_stage_send(result: StageResult) -> None:
                    final_results.append(result.data)
                send_fn = last_stage_send
            else:
                async def relay_send(result: StageResult) -> None:
                    next_stage_idx = stage_idx + 1
                    if next_stage_idx < len(partition_assignments):
                        next_node_id, next_shard = partition_assignments[next_stage_idx]
                        topic = f"synapse/inference/tensor/{next_node_id}"
                        
                        # Use INT8 for hidden states
                        is_hidden_states = np.issubdtype(result.data.dtype, np.floating)
                        quantize = is_hidden_states
                        
                        encoded_payload = encode_tensor(
                            result.data,
                            metadata={
                                "request_id": request_id,
                                "shard": next_shard.to_dict(),
                                "inference_state": {"is_prompt": True},
                                "_orchestrator_id": self.node_id,
                                "_sent_at": time.time(),
                            },
                            quantize_int8=quantize
                        )
                        await self.event_router.publish(topic, {"bin": encoded_payload})
                send_fn = relay_send

            stage = PipelineStage(
                stage_id=stage_idx,
                node_id=node_ids[stage_idx],
                num_stages=num_stages,
                forward_fn=forward_fn,
                send_tensor_fn=send_fn,
            )
            stages.append(stage)

        executor = PipelineExecutor(
            stages=stages,
            scheduler=scheduler,
            timeout_per_step=30.0,
        )

        results = await executor.run_inference(
            input_data=input_data,
            num_micro_batches=num_micro_batches,
        )

        return results