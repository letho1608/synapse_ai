"""
LACP 3.0 — Distributed Pipeline Node Extension

Extends the Node class with:
1. Distributed inference pipeline (multi-node)
2. Distributed training pipeline (DDP + all-reduce)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from synapse.inference.shard import Shard
from synapse.inference.distributed_engine import (
    DistributedPipelineController,
    DistributedInferenceSession,
)
from synapse.transport.result_router import ResultRouter, RoutedResult
from synapse.topology.partitioner import compute_partition
from synapse.topology.resource_weight import NodeResources
from synapse.topology.partitioning_strategy import Partition, map_partitions_to_shards
from synapse.pipeline.gradient_sync import DistributedTrainer, GradientShard
from synapse.helpers import DEBUG

def _maybe_span(tracer, name: str, ctx, attrs: dict):
    """Start a tracing span if tracer is available, else no-op context manager."""
    if tracer is None:
        import contextlib
        return contextlib.nullcontext()
    return tracer.start_span(name, ctx, attrs)


class DistributedPipelineExtension:
    def __init__(self, node: Any):
        self._node = node
        self._controller = DistributedPipelineController(
            node_id=node.id,
            inference_engine=node.inference_engine,
            event_router=node.event_router,
        )
        self._trainer: Optional[DistributedTrainer] = None
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self._controller.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._controller.stop()
            self._started = False

    def handle_result_event(self, payload: Dict[str, Any], deliver_tokens: bool = True) -> bool:
        return self._controller.receive_result_payload(payload, deliver_tokens=deliver_tokens)

    def _get_partition_assignments(self, base_shard: Shard) -> List[Tuple[str, Shard]]:
        num_layers = base_shard.n_layers if base_shard.n_layers > 0 else 32

        partitions = self._node._compute_lacp_partition(base_shard)
        shards = map_partitions_to_shards(partitions, num_layers, base_shard.model_id)

        result: List[Tuple[str, Shard]] = []
        for i, partition in enumerate(partitions):
            if i < len(shards):
                result.append((partition.node_id, shards[i]))
        return result

    async def process_prompt_distributed(
        self,
        base_shard: Shard,
        prompt: str,
        request_id: Optional[str] = None,
        max_generate_tokens: Optional[int] = None,
    ) -> List[int]:
        if request_id is None:
            request_id = str(uuid.uuid4())

        await self.start()

        token_limit = (
            max_generate_tokens
            if isinstance(max_generate_tokens, int) and max_generate_tokens > 0
            else self._node.max_generate_tokens
        )

        partitions = self._get_partition_assignments(base_shard)
        num_stages = len(partitions)

        if num_stages <= 1:
            return await self._run_single_node(base_shard, prompt, request_id, token_limit)

        self._node._schedule_task(
            self._node.broadcast_opaque_status(
                request_id,
                json.dumps({
                    "type": "node_status",
                    "node_id": self._node.id,
                    "status": "start_process_prompt",
                    "base_shard": base_shard.to_dict(),
                    "prompt": prompt,
                    "request_id": request_id,
                    "pipeline_stages": num_stages,
                }),
            ),
            "broadcast_start",
        )

        async def on_token_cb(rid: str, tokens: List[int], is_finished: bool) -> None:
            self._node._on_token_received(rid, tokens, is_finished)
            self._node.trigger_on_token_callbacks(rid, tokens, is_finished)

        tracer = getattr(self._node, "tracer", None)
        if tracer is not None:
            from synapse.orchestration.tracing import TraceContext
            tc = TraceContext(request_id=request_id, sequence_number=0)
            tc.request_span = None
            attrs = {"node_id": self._node.id, "num_stages": num_stages, "model_id": base_shard.model_id}
            with tracer.start_span("node.process_prompt_distributed", tc, attrs) as span:
                all_tokens = await self._controller.process_distributed_prompt(
                    base_shard=base_shard,
                    prompt=prompt,
                    request_id=request_id,
                    partition_assignments=partitions,
                    max_generate_tokens=token_limit,
                    on_token_callback=on_token_cb,
                )
                if span and all_tokens:
                    span.set_attribute("num_tokens", len(all_tokens))
                return all_tokens
        else:
            all_tokens = await self._controller.process_distributed_prompt(
                base_shard=base_shard,
                prompt=prompt,
                request_id=request_id,
                partition_assignments=partitions,
                max_generate_tokens=token_limit,
                on_token_callback=on_token_cb,
            )

        self._node._schedule_task(
            self._node.broadcast_opaque_status(
                request_id,
                json.dumps({
                    "type": "node_status",
                    "node_id": self._node.id,
                    "status": "end_process_prompt",
                    "request_id": request_id,
                    "pipeline_stages": num_stages,
                }),
            ),
            "broadcast_end",
        )

        return all_tokens

    async def _run_single_node(
        self,
        base_shard: Shard,
        prompt: str,
        request_id: str,
        token_limit: int,
    ) -> List[int]:
        self._node.outstanding_requests[request_id] = "processing"
        all_tokens: List[int] = []

        try:
            prompt_ids = await self._node.inference_engine.encode(base_shard, prompt)
            self._node._prompt_token_ids[request_id] = prompt_ids

            result, inference_state = await self._node.inference_engine.infer_tensor(
                request_id, base_shard, prompt_ids
            )
            await self._node.process_inference_result(
                base_shard, result, request_id, inference_state
            )

            buffered_tokens, finished = self._node.buffered_token_output.get(
                request_id, ([], False)
            )
            all_tokens = list(buffered_tokens)
            prev_len = len(all_tokens)

            while not finished and len(all_tokens) < token_limit:
                await asyncio.sleep(0.1)

                buffered_tokens, finished = self._node.buffered_token_output.get(
                    request_id, ([], False)
                )
                current_len = len(buffered_tokens)

                if current_len == prev_len:
                    continue

                new_tokens = buffered_tokens[prev_len:]
                all_tokens.extend(new_tokens)
                prev_len = current_len

                if not finished and len(all_tokens) < token_limit:
                    last_token = np.array([[all_tokens[-1]]])
                    result, inference_state = await self._node.inference_engine.infer_tensor(
                        request_id, base_shard, last_token, inference_state
                    )
                    await self._node.process_inference_result(
                        base_shard, result, request_id, inference_state
                    )

        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
        finally:
            self._node._schedule_task(
                self._node.event_router.publish(
                    "synapse/cluster/cache/clear",
                    json.dumps({"request_id": request_id}),
                )
            )
            self._node.outstanding_requests.pop(request_id, None)
            self._node._prompt_token_ids.pop(request_id, None)

        return all_tokens

    async def process_shard_on_worker(
        self,
        shard: Shard,
        tensor: np.ndarray,
        request_id: str,
        inference_state: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        try:
            self._node.outstanding_requests[request_id] = "processing"

            result, new_inference_state = await self._node.inference_engine.infer_tensor(
                request_id, shard, tensor, inference_state
            )

            orchestrator_id = self._node._orchestrator_id_for_request(request_id)

            result_routed = await self._node.process_inference_result(
                shard,
                result,
                request_id,
                new_inference_state,
                orchestrator_id=orchestrator_id,
            )

            return result_routed

        except Exception as e:
            self._node.outstanding_requests.pop(request_id, None)
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            raise

    async def train_step_distributed(
        self,
        base_shard: Shard,
        input_ids: np.ndarray,
        labels: np.ndarray,
        request_id: Optional[str] = None,
        learning_rate: float = 1e-4,
    ) -> float:
        if request_id is None:
            request_id = str(uuid.uuid4())

        await self.start()

        partitions = self._get_partition_assignments(base_shard)
        num_stages = len(partitions)

        ring_order = [pid for pid, _ in partitions]

        if not self._trainer or self._trainer.world_size != num_stages:
            self._trainer = DistributedTrainer(
                node_id=self._node.id,
                world_size=num_stages,
                ring_order=ring_order,
                learning_rate=learning_rate,
            )

        shard = self._node.get_current_shard(base_shard)
        local_loss = None

        tracer = getattr(self._node, "tracer", None)
        try:
            self._node.outstanding_requests[request_id] = "training"

            _train_span = None
            if tracer is not None:
                from synapse.orchestration.tracing import TraceContext
                tc = TraceContext(request_id=request_id, sequence_number=0)
                tc.request_span = None
                attrs = {"node_id": self._node.id, "world_size": num_stages, "model_id": base_shard.model_id}
                _train_span = tracer.start_span("node.train_step_distributed", tc, attrs)

            loss, grad = await self._node.inference_engine.train(
                request_id, shard, input_ids, labels,
                np.ones(input_ids.shape[0] if input_ids.ndim >= 1 else 1),
                sync_gradients=True,
            )

            if grad is None:
                grad = {}

            async def send_grads(target_id: str, gradients):
                data = {k: v.tobytes() for k, v in gradients.items()}
                shapes = {k: list(v.shape) for k, v in gradients.items()}
                dtypes = {k: str(v.dtype) for k, v in gradients.items()}
                payload = json.dumps({
                    "from_node": self._node.id,
                    "step": self._trainer._train_step,
                    "data": {k: v.hex() for k, v in data.items()},
                    "shapes": shapes,
                    "dtypes": dtypes,
                })
                await self._node.event_router.publish(
                    f"synapse/training/gradient/{target_id}",
                    payload,
                )

            async def recv_grads(source_id: str):
                recv_future = asyncio.get_event_loop().create_future()
                topic = f"synapse/training/gradient/{self._node.id}"

                def on_response(event):
                    if recv_future.done():
                        return
                    try:
                        msg = json.loads(event.data) if isinstance(event.data, str) else event.data
                        if msg.get("from_node") == source_id:
                            data = msg.get("data", {})
                            shapes = msg.get("shapes", {})
                            dtypes = msg.get("dtypes", {})
                            gradients = {}
                            for k, v in data.items():
                                raw = bytes.fromhex(v)
                                gradients[k] = np.frombuffer(raw, dtype=np.dtype(dtypes.get(k, "float32"))).reshape(shapes.get(k, []))
                            recv_future.set_result(gradients)
                    except Exception:
                        if not recv_future.done():
                            recv_future.set_result({})

                self._node.event_router.subscribe(topic, on_response)
                try:
                    result = await asyncio.wait_for(recv_future, timeout=60.0)
                    return result
                finally:
                    self._node.event_router.unsubscribe(topic, on_response)

            updates = await self._trainer.step(
                local_gradients=grad,
                local_loss=float(loss) if loss else 0.0,
                send_fn=send_grads,
                recv_fn=recv_grads,
            )
            if updates:
                await self._node.inference_engine.apply_gradient_updates(shard, updates)
            local_loss = self._trainer.average_loss if self._trainer._loss_history else (float(loss) if loss else 0.0)
            if _train_span:
                _train_span.set_attribute("loss", local_loss)
                _train_span.set_attribute("num_grads", len(grad) if grad else 0)
                _train_span.set_status(type(_train_span)._status_type.OK)
                _train_span.end()
                _train_span = None

        except Exception as e:
            if _train_span:
                try:
                    from opentelemetry.trace import Status, StatusCode
                    _train_span.set_status(Status(StatusCode.ERROR, str(e)))
                    _train_span.end()
                except Exception:
                    pass
                _train_span = None
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            local_loss = None
        finally:
            if _train_span:
                try:
                    _train_span.end()
                except Exception:
                    pass
            self._node.outstanding_requests.pop(request_id, None)

        return local_loss if local_loss is not None else 0.0
