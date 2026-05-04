"""
LACP 3.0 — Distributed Inference Engine

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
    ) -> List[int]:
        """
        Run distributed inference across multiple nodes.

        Args:
            base_shard: Shard with full model info
            prompt: The input prompt
            request_id: Unique request ID
            partition_assignments: List of (node_id, shard) tuples from LACP partitioner
            max_generate_tokens: Max tokens to generate
            on_token_callback: Called with (request_id, tokens, is_finished)

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
            num_micro_batches=1,
        )
        self._sessions[request_id] = session

        if on_token_callback:
            self.result_router.register_token_callback(request_id, on_token_callback)

        all_generated_tokens: List[int] = []

        try:
            token_ids = await self.inference_engine.encode(base_shard, prompt)
            if token_ids is None or token_ids.size == 0:
                return []

            first_stage_node_id, first_stage_shard = partition_assignments[0]
            last_stage_node_id, last_stage_shard = partition_assignments[-1]

            await self._send_input_to_stage(
                first_stage_node_id,
                first_stage_shard,
                token_ids,
                request_id,
                is_prompt=True,
            )

            result = await self._collect_result(
                request_id,
                last_stage_node_id,
                last_stage_shard,
                timeout=60.0,
            )

            if result is not None and result.tokens:
                all_generated_tokens.extend(result.tokens)

            for token_idx in range(1, max_generate_tokens):
                session.is_finished or result.is_finished

                if len(all_generated_tokens) >= max_generate_tokens:
                    break

                last_token = np.array([[all_generated_tokens[-1]]])
                await self._send_input_to_stage(
                    first_stage_node_id,
                    first_stage_shard,
                    last_token,
                    request_id,
                    is_prompt=False,
                )

                next_result = await self._collect_result(
                    request_id,
                    last_stage_node_id,
                    last_stage_shard,
                    timeout=30.0,
                )

                if next_result is None:
                    break

                if next_result.tokens:
                    all_generated_tokens.extend(next_result.tokens)

                if next_result.is_finished:
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

        return all_generated_tokens

    async def _send_input_to_stage(
        self,
        target_node_id: str,
        target_shard: Shard,
        input_data: np.ndarray,
        request_id: str,
        is_prompt: bool = False,
    ) -> None:
        topic = f"synapse/inference/tensor/{target_node_id}"
        payload = {
            "request_id": request_id,
            "shard": target_shard.to_dict(),
            "tensor_data": input_data.tobytes(),
            "shape": list(input_data.shape),
            "dtype": str(input_data.dtype),
            "inference_state": {"is_prompt": is_prompt},
            "_orchestrator_id": self.node_id,
        }
        await self.event_router.publish(topic, payload)

    async def _collect_result(
        self,
        request_id: str,
        last_stage_node_id: str,
        last_stage_shard: Shard,
        timeout: float = 30.0,
    ) -> Optional[RoutedResult]:
        result_event = asyncio.Event()
        collected_result: List[RoutedResult] = []

        def on_result(req_id: str, tokens: List[int], is_finished: bool) -> None:
            if req_id == request_id:
                collected_result.append(
                    RoutedResult(
                        request_id=req_id,
                        orchestrator_id=self.node_id,
                        micro_batch_id=0,
                        source_stage_id=last_stage_shard.start_layer,
                        tokens=list(tokens),
                        is_finished=is_finished,
                    )
                )
                result_event.set()

        self.result_router.register_token_callback(request_id, on_result)

        try:
            await asyncio.wait_for(result_event.wait(), timeout=timeout)
            return collected_result[0] if collected_result else None
        except asyncio.TimeoutError:
            if DEBUG >= 1:
                print(f"[{request_id}] Timeout waiting for result from stage on {last_stage_node_id}")
            return None
        finally:
            self.result_router.unregister_token_callback(request_id)

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
                    pass
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