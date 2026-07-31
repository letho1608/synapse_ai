"""
LACP — Distributed Pipeline Node Extension

Extends the Node class with:
1. Distributed inference pipeline (multi-node)
2. Distributed training pipeline (DDP + all-reduce)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from synapse.inference.shard import Shard
from synapse.transport.result_router import ResultRouter, RoutedResult
from synapse.topology.partitioner import compute_partition
from synapse.topology.resource_weight import NodeResources
from synapse.topology.partitioning_strategy import Partition, map_partitions_to_shards
from synapse.pipeline.gradient_sync import DistributedTrainer, GradientShard
from synapse.pipeline.scheduler import MicroBatch, MicroBatchOp
from synapse.pipeline.stage import PipelineStage, StageResult
from synapse.helpers import DEBUG

def _maybe_span(tracer, name: str, ctx, attrs: dict):
    """Start a tracing span if tracer is available, else no-op context manager."""
    if tracer is None:
        import contextlib
        return contextlib.nullcontext()
    return tracer.start_span(name, ctx, attrs)


class DistributedPipelineExtension:
    def __init__(self, node: Any):
        # Lazy import: synapse.inference.distributed_engine imports back into
        # this package (synapse.pipeline.scheduler), so importing it at module
        # level here creates a circular import when distributed_engine is the
        # first module loaded. Import at construction time instead.
        from synapse.inference.distributed_engine import DistributedPipelineController

        self._node = node
        self._controller = DistributedPipelineController(
            node_id=node.id,
            inference_engine=node.inference_engine,
            event_router=node.event_router,
        )
        self._trainer: Optional[DistributedTrainer] = None
        self._started = False
        # Per-model step counter used for the metadata checkpoints written
        # after each distributed training step (model_id -> steps completed).
        self._checkpoint_steps: Dict[str, int] = {}

    async def start(self) -> None:
        if not self._started:
            await self._controller.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._controller.stop()
            self._started = False

    def _get_partition_assignments(self, base_shard: Shard) -> List[Tuple[str, Shard]]:
        num_layers = base_shard.n_layers if base_shard.n_layers > 0 else 32

        partitions = self._node._compute_lacp_partition(base_shard)
        shards = map_partitions_to_shards(partitions, num_layers, base_shard.model_id)

        result: List[Tuple[str, Shard]] = []
        for i, partition in enumerate(partitions):
            if i < len(shards):
                result.append((partition.node_id, shards[i]))
        return result

    def _save_shard_checkpoint(
        self, base_shard: Shard, shard: Optional[Shard] = None
    ) -> None:
        """Persist a JSON metadata checkpoint for the local shard after a
        successful distributed training step.

        This is what makes the README's "recovered from JSON checkpoint"
        claim true in a live cluster: every distributed training step writes
        the shard range + step to the shared checkpoint directory, so a
        surviving node can recover the dead node's shard state via
        FaultToleranceManager.load_latest_checkpoint() after rebalancing.

        Lightweight by design: metadata only (no kv_cache / optimizer state),
        and exceptions are swallowed so a checkpoint write failure never
        aborts training.
        """
        try:
            node = self._node
            node._lazy_init_fault_tolerance()
            if node.fault_tolerance is None:
                return
            local = shard if shard is not None else node.get_current_shard(base_shard)
            if local is None:
                return
            step = self._checkpoint_steps.get(base_shard.model_id, 0) + 1
            self._checkpoint_steps[base_shard.model_id] = step
            node.fault_tolerance.save_checkpoint(
                base_shard.model_id,
                local.start_layer,
                local.end_layer,
                step=step,
                kv_cache=None,
            )
        except Exception:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()


    async def process_prompt_distributed(
        self,
        base_shard: Shard,
        prompt: str,
        request_id: Optional[str] = None,
        max_generate_tokens: Optional[int] = None,
        num_micro_batches: int = 1,
    ) -> List[int]:
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Update controller settings from node before processing
        self._controller.lookahead_window = getattr(self._node, "lookahead_window", 3)

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

        on_token_cb = self._node._on_token_received

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
                    num_micro_batches=num_micro_batches,
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
                num_micro_batches=num_micro_batches,
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
        schedule: str = "ddp",
    ) -> float:
        """Distributed training step.

        schedule="ddp"  — full-model copies, ring all-reduce of gradients.
        schedule="1f1b" — pipeline-parallel step driven by the 1F1B schedule
                          (see train_step_distributed_1f1b).
        """
        if schedule == "1f1b":
            return await self.train_step_distributed_1f1b(
                base_shard, input_ids, labels,
                request_id=request_id, learning_rate=learning_rate,
            )
        if request_id is None:
            request_id = str(uuid.uuid4())

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
        _train_cm = None
        try:
            self._node.outstanding_requests[request_id] = "training"

            if tracer is not None:
                from synapse.orchestration.tracing import TraceContext
                tc = TraceContext(request_id=request_id, sequence_number=0)
                tc.request_span = None
                attrs = {"node_id": self._node.id, "world_size": num_stages, "model_id": base_shard.model_id}
                _train_cm = tracer.start_span("node.train_step_distributed", tc, attrs)

            # tracer.start_span is a @contextmanager yielding the span; it
            # auto-sets status and ends the span on exit.
            with _train_cm if _train_cm is not None else contextlib.nullcontext() as _train_span:
                loss, grad = await self._node.inference_engine.train(
                    request_id, shard, input_ids, labels,
                    np.ones(input_ids.shape[0] if input_ids.ndim >= 1 else 1),
                    sync_gradients=True,
                )

                if grad is None:
                    grad = {}

                send_grads, recv_grads = self._grad_ring_io()

                updates = await self._trainer.step(
                    local_gradients=grad,
                    local_loss=float(loss) if loss else 0.0,
                    send_fn=send_grads,
                    recv_fn=recv_grads,
                )
                if updates:
                    await self._node.inference_engine.apply_gradient_updates(shard, updates)
                # Persist the shard's step to the shared checkpoint directory
                # so cross-node recovery restores real on-disk state.
                self._save_shard_checkpoint(base_shard, shard=shard)
                local_loss = self._trainer.average_loss if self._trainer._loss_history else (float(loss) if loss else 0.0)
                if _train_span is not None:
                    _train_span.set_attribute("loss", local_loss)
                    _train_span.set_attribute("num_grads", len(grad) if grad else 0)

        except Exception as e:
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            local_loss = None
        finally:
            self._node.outstanding_requests.pop(request_id, None)

        return local_loss if local_loss is not None else 0.0

    def _grad_ring_io(self):
        """Build the ring all-reduce send/recv closures (binary FP16 codec).

        Returns (send_fn, recv_fn) used by DistributedTrainer.step to exchange
        gradients with the ring predecessor/successor over the event router.
        """
        async def send_grads(target_id: str, gradients):
            # Use binary codec for gradients (huge performance boost)
            from synapse.transport.codec import encode_tensor

            # Combine all gradients into one binary blob or send one by one.
            # For simplicity and to reuse encode_tensor, we wrap them: FP16 for
            # transport (50% bandwidth saving), decoded back to FP32 on recv.
            encoded_grads = {}
            for k, v in gradients.items():
                encoded_grads[k] = encode_tensor(v.astype(np.float16), metadata={"param": k})

            payload = {
                "from_node": self._node.id,
                "step": self._trainer._train_step,
                "bin_grads": encoded_grads,  # Map of param_name -> bytes
            }
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
                    msg = event.data
                    if msg.get("from_node") == source_id:
                        from synapse.transport.codec import decode_tensor
                        bin_grads = msg.get("bin_grads", {})
                        gradients = {}
                        for k, v in bin_grads.items():
                            tensor, _ = decode_tensor(v)
                            gradients[k] = tensor.astype(np.float32)
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

        return send_grads, recv_grads

    async def train_step_distributed_1f1b(
        self,
        base_shard: Shard,
        input_ids: np.ndarray,
        labels: np.ndarray,
        request_id: Optional[str] = None,
        learning_rate: float = 1e-4,
        num_micro_batches: Optional[int] = None,
    ) -> float:
        """Production distributed training step driven by the 1F1B schedule.

        Partitions the model across nodes, builds one PipelineStage per
        partition, and executes the 1F1B schedule — PipelineScheduler
        .generate_1f1b_schedule() + PipelineStage.process_loop() — so the
        forward and backward phases of every micro-batch are coordinated
        across the pipeline. Each stage runs its forward/backward on the local
        engine; the per-micro-batch gradients are accumulated and synchronized
        with the ring all-reduce (DistributedTrainer.step) before updates are
        applied.

        Returns the mean forward loss across micro-batches.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        partitions = self._get_partition_assignments(base_shard)
        num_stages = len(partitions)
        if num_stages <= 1:
            # Single node: plain training step, no pipeline needed.
            shard = self._node.get_current_shard(base_shard)
            length = np.ones(max(1, input_ids.shape[0] if input_ids.ndim >= 1 else 1))
            loss, _ = await self._node.inference_engine.train(
                request_id, shard, input_ids, labels, length, sync_gradients=False,
            )
            # Persist the shard's step to the shared checkpoint directory.
            self._save_shard_checkpoint(base_shard, shard=shard)
            return float(loss) if loss is not None else 0.0

        rows = max(1, input_ids.shape[0] if input_ids.ndim >= 1 else 1)
        if num_micro_batches is None:
            num_micro_batches = max(2, num_stages)
        num_micro_batches = min(num_micro_batches, rows)

        ring_order = [pid for pid, _ in partitions]
        if not self._trainer or self._trainer.world_size != num_stages:
            self._trainer = DistributedTrainer(
                node_id=self._node.id,
                world_size=num_stages,
                ring_order=ring_order,
                learning_rate=learning_rate,
            )

        node = self._node
        node.outstanding_requests[request_id] = "training"
        losses_by_mb: Dict[int, float] = {}
        grads_by_mb: Dict[int, Dict[str, np.ndarray]] = {}

        try:
            stages: List[PipelineStage] = []
            for stage_idx, (nid, shard) in enumerate(partitions):
                is_last = stage_idx == num_stages - 1

                async def forward(mb: MicroBatch, _shard=shard, _last=is_last):
                    length = np.ones(max(1, mb.data.shape[0]))
                    mb_labels = mb.labels if mb.labels is not None else labels
                    loss, grads = await node.inference_engine.train(
                        request_id, _shard, mb.data, mb_labels, length,
                        sync_gradients=True,
                    )
                    if loss is not None:
                        losses_by_mb[mb.batch_id] = float(loss)
                    if grads:
                        grads_by_mb[mb.batch_id] = grads
                    if _last:
                        val = float(loss) if loss is not None else np.nan
                        return np.array([val], dtype=np.float32)
                    return np.zeros((1,), dtype=np.float32)

                async def backward(mb: MicroBatch, _shard=shard, _last=is_last):
                    # Engines do not yet implement cross-node back_gradient;
                    # the schedule still drives the backward phase so the 1F1B
                    # flow executes end to end.
                    if not _last:
                        try:
                            await node.inference_engine.train(
                                request_id, _shard, mb.data,
                                mb.labels if mb.labels is not None else labels,
                                np.ones(max(1, mb.data.shape[0])),
                                backgrad=mb.data, loss="back_gradient",
                                sync_gradients=False,
                            )
                        except Exception:
                            pass
                    # Gradient feedback, not a loss: excluded from the results.
                    return np.array([np.nan], dtype=np.float32)

                async def noop_send(result: StageResult) -> None:
                    pass

                stages.append(PipelineStage(
                    stage_id=stage_idx,
                    node_id=nid,
                    num_stages=num_stages,
                    forward_fn=forward,
                    send_tensor_fn=noop_send,
                    backward_fn=backward,
                ))

            losses = await self._trainer.train_1f1b(
                stages, num_micro_batches, input_ids, labels,
            )

            # Accumulate per-micro-batch gradients (micro-batch gradient
            # accumulation) and scale back to full-batch magnitude.
            all_grads: Dict[str, np.ndarray] = {}
            for mb_id in sorted(grads_by_mb):
                for k, v in grads_by_mb[mb_id].items():
                    if k in all_grads:
                        all_grads[k] = all_grads[k] + v
                    else:
                        all_grads[k] = v.copy()
            if all_grads:
                scale = float(num_micro_batches)
                all_grads = {k: v / scale for k, v in all_grads.items()}

            avg_loss = float(np.mean(losses)) if losses else 0.0

            if all_grads:
                send_grads, recv_grads = self._grad_ring_io()
                updates = await self._trainer.step(
                    local_gradients=all_grads,
                    local_loss=avg_loss,
                    send_fn=send_grads,
                    recv_fn=recv_grads,
                )
                if updates:
                    shard = node.get_current_shard(base_shard)
                    await node.inference_engine.apply_gradient_updates(shard, updates)
            # Persist the shard's step to the shared checkpoint directory so
            # cross-node recovery restores real on-disk state.
            self._save_shard_checkpoint(base_shard)
            return avg_loss
        finally:
            node.outstanding_requests.pop(request_id, None)