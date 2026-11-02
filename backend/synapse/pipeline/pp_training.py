"""
Pipeline-parallel training across nodes (each node trains only its layers).

One training step for a plan [(node, shard), ...] and a batch:

  orchestrator ── fwd(ids, labels) ──▶ stage 0 ── fwd(hidden) ──▶ ... ▶ last
  last: loss + backward ── bwd(dL/dx) ──▶ ... ──▶ stage 0 ── mbdone ──▶ orch
                └──────── loss ─────────────────────────────────────▶ orch
  (micro-batches are sent back to back, so stage 0 runs micro-batch m+1
   while later stages still work on m)
  orch ── norm ──▶ every stage: tied-embedding gradients are exchanged
                   between first and last stage, then each reports its
                   squared gradient norm
  orch ── step(lr, scale) ──▶ every stage: grads * (1/M x global clip
                   coefficient), AdamW step on its own parameters ── stepped

Every message is (tensor, metadata); `send(target_node, tensor, meta)` is
injected (the node routes it over chunked TCP or the event bridge, or
straight back into on_message for its own stages), so the protocol runs the
same in-process (tests) and across machines. Hidden states and gradients
travel as fp32: training gradients get no wire quantization.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from synapse.helpers import DEBUG
from synapse.inference.shard import Shard

KIND = "pp_train"
SendFn = Callable[[str, np.ndarray, Dict[str, Any]], Awaitable[None]]
_EMPTY = np.zeros((1,), dtype=np.float32)


def _plan_to_wire(plan: List[Tuple[str, Shard]]) -> List[List[Any]]:
    return [[nid, int(s.start_layer), int(s.end_layer), int(s.n_layers)] for nid, s in plan]


class PipelineTrainer:
    """Both roles: the orchestrator of a step (train_step) and the stage
    worker for whichever stages of a plan this node holds (on_message)."""

    def __init__(self, node_id: str, engine: Any, send: SendFn,
                 timeout: float = 300.0,
                 on_step: Optional[Callable[[Shard], Any]] = None):
        self.node_id = node_id
        self.engine = engine
        self.send = send
        self.timeout = timeout
        self.on_step = on_step  # called after a stage applied its step (checkpointing)
        self._jobs: Dict[str, Dict[str, Any]] = {}            # orchestrator side
        self._tied_inbox: Dict[Tuple[str, int], asyncio.Future] = {}
        self._prepared: Dict[Tuple[str, int, int], bool] = {}  # (model, start, end) ready
        self.stats = {"steps": 0, "fwd": 0, "bwd": 0, "errors": 0}

    # ------------------------------------------------------------------ util
    @staticmethod
    def _shard(meta: Dict[str, Any], stage: int) -> Tuple[str, Shard]:
        nid, s, e, n = meta["plan"][stage]
        return nid, Shard(meta["model"], int(s), int(e), int(n))

    async def _to(self, target: str, meta: Dict[str, Any], op: str,
                  tensor: Optional[np.ndarray] = None, **extra) -> None:
        m = {"kind": KIND, "op": op, "job": meta["job"], "plan": meta["plan"],
             "model": meta["model"], "orch": meta["orch"], **extra}
        if meta.get("lora"):
            m["lora"] = meta["lora"]
        await self.send(target, _EMPTY if tensor is None else tensor, m)

    async def _prepare(self, shard: Shard, lora: Optional[Dict[str, Any]] = None) -> None:
        key = (shard.model_id, shard.start_layer, shard.end_layer, repr(lora))
        if not self._prepared.get(key):
            await self.engine.stage_train_prepare(shard, lora=lora)
            self._prepared[key] = True

    # ---------------------------------------------------------- stage worker
    async def on_message(self, tensor: np.ndarray, meta: Dict[str, Any]) -> None:
        """Entry point for every pipeline-training message for this node."""
        op = meta.get("op")
        try:
            if op in ("loss", "mbdone", "normrep", "stepped", "err"):
                self._orch_event(op, tensor, meta)
            elif op == "fwd":
                await self._on_fwd(tensor, meta)
            elif op == "bwd":
                await self._on_bwd(tensor, meta)
            elif op == "norm":
                await self._on_norm(meta)
            elif op == "tied":
                fut = self._tied_future(meta["job"], int(meta["stage"]))
                if not fut.done():
                    fut.set_result(np.asarray(tensor, dtype=np.float32))
            elif op == "step":
                await self._on_step(meta)
            elif op == "drop":
                self.engine.stage_drop(meta["job"])
        except Exception as e:
            self.stats["errors"] += 1
            if DEBUG >= 1:
                import traceback
                traceback.print_exc()
            if op not in ("loss", "mbdone", "normrep", "stepped", "err"):
                await self._to(meta["orch"], meta, "err", error=f"{self.node_id} {op}: {e!r}")

    async def _on_fwd(self, tensor, meta) -> None:
        stage, mb = int(meta["stage"]), int(meta["mb"])
        _, shard = self._shard(meta, stage)
        await self._prepare(shard, meta.get("lora"))
        labels = None
        if shard.is_last_layer():
            labels = np.asarray(meta["labels"], dtype=np.int64).reshape(meta["labels_shape"])
        res = await self.engine.stage_forward((meta["job"], mb, stage), shard, tensor, labels)
        self.stats["fwd"] += 1
        if "loss" in res:
            await self._to(meta["orch"], meta, "loss", mb=mb, loss=float(res["loss"]))
            if stage == 0:
                await self._to(meta["orch"], meta, "mbdone", mb=mb)
            else:
                prev, _ = self._shard(meta, stage - 1)
                await self._to(prev, meta, "bwd", res["input_grad"], mb=mb, stage=stage - 1)
        else:
            nxt, _ = self._shard(meta, stage + 1)
            await self._to(nxt, meta, "fwd", res["hidden"], mb=mb, stage=stage + 1,
                           labels=meta.get("labels"), labels_shape=meta.get("labels_shape"))

    async def _on_bwd(self, tensor, meta) -> None:
        stage, mb = int(meta["stage"]), int(meta["mb"])
        grad_in = await self.engine.stage_backward((meta["job"], mb, stage), tensor)
        self.stats["bwd"] += 1
        if stage == 0:
            await self._to(meta["orch"], meta, "mbdone", mb=mb)
        else:
            prev, _ = self._shard(meta, stage - 1)
            await self._to(prev, meta, "bwd", grad_in, mb=mb, stage=stage - 1)

    def _tied_future(self, job: str, stage: int) -> asyncio.Future:
        key = (job, stage)
        fut = self._tied_inbox.get(key)
        if fut is None:
            fut = self._tied_inbox[key] = asyncio.get_running_loop().create_future()
        return fut

    async def _on_norm(self, meta) -> None:
        stage = int(meta["stage"])
        _, shard = self._shard(meta, stage)
        last = len(meta["plan"]) - 1
        if stage in (0, last) and last > 0 and self.engine._tied_param(shard) is not None:
            # Tied embedding split over first/last stage: swap gradients so
            # both copies take the same (summed) step.
            peer_stage = last if stage == 0 else 0
            peer, _ = self._shard(meta, peer_stage)
            mine = self.engine.stage_tied_grad(shard)
            await self._to(peer, meta, "tied", mine if mine is not None else _EMPTY, stage=peer_stage)
            other = await asyncio.wait_for(self._tied_future(meta["job"], stage), self.timeout)
            self._tied_inbox.pop((meta["job"], stage), None)
            if other.size > 1:  # a peer without a gradient sends the 1-element placeholder
                self.engine.stage_add_tied_grad(shard, other)
        sq = self.engine.stage_grad_sq(shard, float(meta["scale"]))
        await self._to(meta["orch"], meta, "normrep", stage=stage, sq=sq)

    async def _on_step(self, meta) -> None:
        stage = int(meta["stage"])
        _, shard = self._shard(meta, stage)
        await self.engine.stage_step(shard, float(meta["lr"]), float(meta["scale"]))
        self.stats["steps"] += 1
        if self.on_step is not None:
            try:
                r = self.on_step(shard)
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[PP] on_step hook failed: {e}")
        await self._to(meta["orch"], meta, "stepped", stage=stage)

    # ---------------------------------------------------------- orchestrator
    def _orch_event(self, op, tensor, meta) -> None:
        job = self._jobs.get(meta["job"])
        if job is None:
            return
        if op == "err":
            if not job["error"].done():
                job["error"].set_result(meta.get("error", "stage error"))
            return
        bucket, key = {"loss": ("loss", "mb"), "mbdone": ("done", "mb"),
                       "normrep": ("norm", "stage"), "stepped": ("stepped", "stage")}[op]
        fut = job[bucket].get(int(meta[key]))
        if fut is not None and not fut.done():
            fut.set_result(meta.get("loss") if op == "loss" else meta.get("sq"))

    async def _gather(self, job, futs: List[asyncio.Future], what: str) -> List[Any]:
        """Wait for all futures, failing fast on a stage error or timeout."""
        waiter = asyncio.ensure_future(asyncio.gather(*futs))
        done, _ = await asyncio.wait({waiter, job["error"]}, timeout=self.timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if waiter in done:
            return waiter.result()
        waiter.cancel()
        if job["error"] in done:
            raise RuntimeError(f"pipeline training failed during {what}: {job['error'].result()}")
        raise TimeoutError(f"pipeline training timed out during {what} after {self.timeout:.0f}s")

    async def train_step(self, plan: List[Tuple[str, Shard]], input_ids: np.ndarray,
                         labels: np.ndarray, lr: float = 1e-5,
                         num_micro_batches: Optional[int] = None,
                         max_grad_norm: float = 1.0,
                         lora: Optional[Dict[str, Any]] = None) -> float:
        """Run one optimizer step of the pipeline; returns the mean loss.
        `lora` (stage_lora settings) trains adapters instead of the weights."""
        if not plan:
            raise ValueError("empty plan")
        ids = np.asarray(input_ids)
        lab = np.asarray(labels)
        if ids.ndim == 1:
            ids, lab = ids[None], lab.reshape(1, -1)
        rows = ids.shape[0]
        n_stages = len(plan)
        m = min(rows, num_micro_batches or max(2, n_stages))
        chunks = np.array_split(np.arange(rows), m)
        loop = asyncio.get_running_loop()
        job_id = uuid.uuid4().hex
        job = {"loss": {i: loop.create_future() for i in range(m)},
               "done": {i: loop.create_future() for i in range(m)},
               "norm": {s: loop.create_future() for s in range(n_stages)},
               "stepped": {s: loop.create_future() for s in range(n_stages)},
               "error": loop.create_future()}
        self._jobs[job_id] = job
        meta = {"job": job_id, "plan": _plan_to_wire(plan), "model": plan[0][1].model_id,
                "orch": self.node_id}
        if lora:
            meta["lora"] = lora
        first = plan[0][0]
        try:
            for i, idx in enumerate(chunks):
                await self._to(first, meta, "fwd", np.ascontiguousarray(ids[idx]), mb=i, stage=0,
                               labels=lab[idx].reshape(-1).tolist(), labels_shape=list(lab[idx].shape))
            losses = await self._gather(job, list(job["loss"].values()), "forward/backward")
            await self._gather(job, list(job["done"].values()), "backward")

            scale = 1.0 / m
            for s, (nid, _) in enumerate(plan):
                await self._to(nid, meta, "norm", stage=s, scale=scale)
            sq = await self._gather(job, list(job["norm"].values()), "gradient norm")
            norm = float(np.sqrt(max(0.0, sum(sq))))
            clip = min(1.0, max_grad_norm / (norm + 1e-6)) if max_grad_norm else 1.0
            for s, (nid, _) in enumerate(plan):
                await self._to(nid, meta, "step", stage=s, lr=float(lr), scale=scale * clip)
            await self._gather(job, list(job["stepped"].values()), "optimizer step")
            self.last_grad_norm = norm
            return float(np.mean(losses))
        except BaseException:
            for nid in {n for n, _ in plan}:
                try:
                    await self._to(nid, meta, "drop")
                except Exception:
                    pass
            raise
        finally:
            self._jobs.pop(job_id, None)
