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
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from synapse.inference.inference_engine import SHARD_WEIGHTS_SUFFIX
from synapse.inference.shard import Shard
from synapse.topology.partitioning_strategy import map_partitions_to_shards
from synapse.pipeline.gradient_sync import DistributedTrainer
from synapse.pipeline.scheduler import MicroBatch
from synapse.pipeline.stage import PipelineStage, StageResult
from synapse.helpers import DEBUG

_log = logging.getLogger(__name__)


class _GradientHandoff:
    """In-process pipeline backward channel for the 1F1B schedule.

    The backward phase runs in reverse stage order: stage k's backward
    consumes the input-gradient computed by stage k+1 (the downstream stage)
    as its ``backgrad``. Keys are ``(stage_id, micro_batch_id)``; the value
    is the real gradient tensor handed back through the stage's send path.
    """

    def __init__(self) -> None:
        self._queues: Dict[Tuple[int, int], asyncio.Queue] = {}

    def ensure(self, key: Tuple[int, int]) -> None:
        """Pre-create the queue for a key so a get() before any put() waits
        instead of returning immediately (the schedule runs the upstream
        backward before the downstream one delivers)."""
        self._queues.setdefault(key, asyncio.Queue(maxsize=1))

    def put(self, key: Tuple[int, int], value: np.ndarray) -> None:
        q = self._queues.setdefault(key, asyncio.Queue(maxsize=1))
        if q.empty():
            q.put_nowait(value)

    async def get(
        self, key: Tuple[int, int], timeout: float = 30.0
    ) -> Optional[np.ndarray]:
        q = self._queues.get(key)
        if q is None:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


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
        # (model_id, start, end) -> (weights file, step it holds).
        self._weights_saved: Dict[Tuple[str, int, int], Tuple[str, int]] = {}

    async def start(self) -> None:
        if not self._started:
            self.sync_inference_engine()
            await self._controller.start()
            self._started = True

    def sync_inference_engine(self) -> None:
        """Refresh the controller's engine reference from the node.

        The extension may be constructed at node.start() (A7: initialized
        unconditionally so rebalancing works before any distributed request),
        while node.select_best_inference_engine() can swap the node's engine
        later; the controller must always forward with the current engine.
        """
        self._controller.inference_engine = self._node.inference_engine

    async def stop(self) -> None:
        if self._started:
            await self._controller.stop()
            self._started = False

    def _get_partition_assignments(self, base_shard: Shard) -> List[Tuple[str, Shard]]:
        num_layers = base_shard.n_layers if base_shard.n_layers > 0 else 32

        # B4: prefer the master's broadcast partition (already computed and
        # consistent cluster-wide) over recomputing LACP locally per forward.
        partitions = self._node._get_authoritative_partitions(base_shard)
        if partitions is None:
            partitions = self._node._compute_lacp_partition(base_shard)
        shards = map_partitions_to_shards(partitions, num_layers, base_shard.model_id)

        result: List[Tuple[str, Shard]] = []
        for i, partition in enumerate(partitions):
            if i < len(shards):
                result.append((partition.node_id, shards[i]))
        return result

    def _maybe_save_shard_weights(
        self, model_id: str, local: Shard, step: int
    ) -> Tuple[Optional[str], Optional[int]]:
        """Write the local shard's layer weights every N steps
        (SYNAPSE_CKPT_WEIGHTS_EVERY, default 10; 0 disables) to one
        overwritten file per node+model+range. Returns the newest weights
        file and the step it holds, so metadata-only steps in between still
        point recovery at real weights (possibly a few steps older)."""
        import os
        import re
        try:
            every = int(os.getenv("SYNAPSE_CKPT_WEIGHTS_EVERY", "10"))
        except ValueError:
            every = 10
        key = (model_id, local.start_layer, local.end_layer)
        last = self._weights_saved.get(key)
        if every > 0 and (step == 1 or step % every == 0):
            node = self._node
            safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
            path = os.path.join(
                node.fault_tolerance.checkpoint_dir,
                f"{node.id}_{safe_model}_L{local.start_layer}-{local.end_layer}"
                f"{SHARD_WEIGHTS_SUFFIX}")
            engine = node.inference_engine
            if engine is not None and engine.save_shard_weights(local, path):
                last = (path, step)
                self._weights_saved[key] = last
        return last if last else (None, None)

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

        Metadata every step; the shard's layer weights every N steps (see
        _maybe_save_shard_weights) so recovery can restore real weights. No
        kv_cache / optimizer state.
        and exceptions are caught so a checkpoint write failure never aborts
        training — but they are logged at WARNING, not silently swallowed, so
        a failing checkpoint directory is observable in production.
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
            weights_path, weights_step = self._maybe_save_shard_weights(
                base_shard.model_id, local, step)
            ckpt_path = node.fault_tolerance.save_checkpoint(
                base_shard.model_id,
                local.start_layer,
                local.end_layer,
                step=step,
                kv_cache=None,
                weights_path=weights_path,
                weights_step=weights_step,
            )
            replicate = getattr(node, "_replicate_checkpoint", None)
            if callable(replicate) and ckpt_path:
                # Survives the loss of THIS machine: a copy on the successor.
                node._schedule_task(replicate(ckpt_path), "ckpt_replicate")
        except Exception as e:
            _log.warning(
                "checkpoint write failed for %s (step %s): %s",
                base_shard.model_id,
                self._checkpoint_steps.get(base_shard.model_id, 0),
                e,
            )
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
        sampling_state: Optional[Dict[str, Any]] = None,
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

        if num_stages == 1 and partitions[0][0] == self._node.id:
            # One stage and it is us: run the WHOLE model locally. (base_shard
            # is (0, 0, n): running it ran layer 0 only.)
            return await self._run_single_node(partitions[0][1], prompt, request_id, token_limit)
        if num_stages == 0:
            from synapse.models import build_full_shard
            full = build_full_shard(base_shard.model_id, "PyTorchHFInferenceEngine") or \
                Shard(base_shard.model_id, 0, base_shard.n_layers - 1, base_shard.n_layers)
            return await self._run_single_node(full, prompt, request_id, token_limit)
        # num_stages >= 1 with a remote stage: the controller drives it (a
        # single remote stage is both first and last).

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

        node = self._node

        def on_token_cb(rid, tokens, finished):
            # Verified tokens only: feed the request buffer AND the stream
            # callbacks (API / console) — the last stage's raw samples are
            # never streamed.
            node._on_token_received(rid, tokens, finished)
            node.trigger_on_token_callbacks(rid, tokens, finished)

        def replan(current, exclude):
            fn = getattr(node, "_replan_excluding", None)
            return fn(base_shard, current, exclude) if callable(fn) else None

        alive = getattr(node, "_node_alive", None)
        if not callable(alive):
            alive = None

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
                    sampling_state=sampling_state,
                    replan_fn=replan,
                    alive_fn=alive,
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
                sampling_state=sampling_state,
                replan_fn=replan,
                alive_fn=alive,
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

        except Exception:
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
                # so cross-node recovery restores real on-disk state. Only
                # multi-node steps write checkpoints: a single-node run has no
                # peer to recover state for, so writing would be pure chatter.
                if num_stages > 1:
                    self._save_shard_checkpoint(base_shard, shard=shard)
                local_loss = self._trainer.average_loss if self._trainer._loss_history else (float(loss) if loss else 0.0)
                if _train_span is not None:
                    _train_span.set_attribute("loss", local_loss)
                    _train_span.set_attribute("num_grads", len(grad) if grad else 0)

        except Exception:
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
            # Single node: plain training step, no pipeline needed. No
            # checkpoint is written — with no peer, there is no cross-node
            # recovery value and the write would be per-step disk chatter.
            shard = self._node.get_current_shard(base_shard)
            length = np.ones(max(1, input_ids.shape[0] if input_ids.ndim >= 1 else 1))
            loss, _ = await self._node.inference_engine.train(
                request_id, shard, input_ids, labels, length, sync_gradients=False,
            )
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
        # Real 1F1B backward handoff: each stage's backward computes the
        # gradient of the loss w.r.t. its input and delivers it upstream via
        # the pipeline send path; the upstream stage consumes it as backgrad
        # (reverse-order gradient flow across stage boundaries).
        handoff = _GradientHandoff()
        self._last_handoff_sends: List[Tuple[int, int]] = []
        # Pre-create every (stage, micro-batch) queue: the upstream stage's
        # backward runs before the downstream stage's delivers, so gets must
        # block until the value arrives rather than miss it.
        for s in range(num_stages):
            for m in range(num_micro_batches):
                handoff.ensure((s, m))

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

                async def backward(mb: MicroBatch, _shard=shard, _last=is_last, _idx=stage_idx):
                    # Receive the downstream stage's input-gradient: gradient
                    # of the loss w.r.t. the next stage's input, which is the
                    # gradient of the loss w.r.t. this stage's output. The
                    # last stage owns the loss, so it has no downstream grad.
                    backgrad = None
                    if not _last:
                        backgrad = await handoff.get((_idx + 1, mb.batch_id))
                    grads = None
                    try:
                        _, grads = await node.inference_engine.train(
                            request_id, _shard, mb.data,
                            mb.labels if mb.labels is not None else labels,
                            np.ones(max(1, mb.data.shape[0])),
                            backgrad=backgrad, loss="back_gradient",
                            sync_gradients=False,
                        )
                    except Exception:
                        grads = None
                    if grads and "__input_grad__" in grads:
                        input_grad = np.asarray(grads["__input_grad__"], dtype=np.float32)
                    elif backgrad is not None:
                        input_grad = backgrad
                    else:
                        input_grad = np.zeros_like(mb.data, dtype=np.float32)
                    # The last stage has no downstream send path, so it
                    # delivers its input-gradient straight to the channel.
                    if _last:
                        handoff.put((_idx, mb.batch_id), input_grad)
                    # Gradient feedback, not a loss: excluded from results.
                    return input_grad

                async def handoff_send(result: StageResult, _idx=stage_idx) -> None:
                    # Real inter-stage handoff: the backward phase's
                    # input-gradient flows to the upstream stage via the
                    # pipeline send path (previously a noop_send).
                    if result.op == "backward":
                        handoff.put(
                            (_idx, result.micro_batch_id),
                            np.asarray(result.data, dtype=np.float32),
                        )
                        self._last_handoff_sends.append((_idx, result.micro_batch_id))

                stages.append(PipelineStage(
                    stage_id=stage_idx,
                    node_id=nid,
                    num_stages=num_stages,
                    forward_fn=forward,
                    send_tensor_fn=handoff_send,
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