"""
Training on the layers a node serves (pipeline parallelism).

Before, every training step loaded the WHOLE model on every node
(_ensure_full) and each "stage" re-ran the full forward locally; nothing
crossed node boundaries but a loss. Now a node trains exactly the layer
range it holds for inference (the same PartialModel weights, updated in
place), and a step is a real pipeline:

  stage 0 : token ids -> embed + its layers -> hidden  ──▶ next stage
  stage i : hidden (leaf, requires_grad) -> its layers -> hidden ──▶
  last    : hidden -> its layers -> norm + head -> loss; backward
            ◀── d loss / d input ── each stage backprops its own graph
  then    : every stage applies its own optimizer step (its params only)

Parameters are disjoint across stages, so no gradient all-reduce is needed,
except for tied input/output embeddings split across the first and last
stage: those two copies exchange gradients before the step and stay equal
(see pp_training.PipelineTrainer).

The engine exposes the per-stage pieces; pp_training.py moves the tensors.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np



class StageTrainingMixin:
    """Mixed into PyTorchHFInferenceEngine (needs _pm, _forward_lock,
    _get_transformer_components, _build_position_embeddings, _causal_mask,
    _apply_blocks, _device_dtype, ensure_shard, _seq_kv)."""

    WEIGHT_DECAY = 0.01

    # -- parameters ---------------------------------------------------------
    def _stage_params(self, shard) -> List[Tuple[str, Any]]:
        """(name, parameter) for everything this node trains for `shard`.
        Names are stable across nodes ("layers.7.mlp.up_proj.weight",
        "embed.weight", "head.weight", "norm.weight"); a tensor shared by
        two modules (head tied to embed on the same node) is listed once.
        With LoRA on, only the adapters of the served layers are trained."""
        pm = self._pm
        if self._lora_cfg is not None:
            from synapse.training.stage_lora import lora_parameters
            return [(f"layers.{i}.{n}", p)
                    for i in range(shard.start_layer, shard.end_layer + 1)
                    for n, p in lora_parameters(pm.layers[i])]
        out: List[Tuple[str, Any]] = []
        seen = set()

        def add(prefix, module):
            if module is None:
                return
            for n, p in module.named_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append((f"{prefix}.{n}", p))

        if shard.is_first_layer():
            add("embed", pm.embed)
            add("pos_embed", pm.pos_embed)
        for i in range(shard.start_layer, shard.end_layer + 1):
            add(f"layers.{i}", pm.layers[i])
        if shard.is_last_layer():
            add("norm", pm.final_norm)
            add("head", pm.lm_head)
        return out

    def _tied_param(self, shard):
        """The local copy of a tied embedding that another stage also holds:
        the input embedding on a first-only stage, the head on a last-only
        stage. None when not tied or when one stage holds both ends."""
        pm = self._pm
        if self._lora_cfg is not None:
            return None  # embeddings are frozen under LoRA
        if not getattr(pm, "head_tied", False) or (shard.is_first_layer() and shard.is_last_layer()):
            return None
        if shard.is_first_layer():
            return pm.embed.weight
        if shard.is_last_layer():
            return pm.lm_head.weight
        return None

    # LoRA settings of the current training job (stage_lora.normalize_config),
    # None for full fine-tuning.
    _lora_cfg: Optional[Dict[str, Any]] = None

    async def stage_train_prepare(self, shard, lora: Optional[Dict[str, Any]] = None) -> None:
        """Load the served range (no whole-model load) and make it trainable:
        every weight (full fine-tuning) or LoRA adapters over frozen weights."""
        from synapse.training import stage_lora
        await self.ensure_shard(shard)
        if self._pm is None:
            raise RuntimeError("pipeline training needs layer-range loading (SYNAPSE_PARTIAL_LOAD=1)")
        lora = stage_lora.normalize_config(lora)
        layers = [self._pm.layers[i] for i in range(shard.start_layer, shard.end_layer + 1)]
        if lora is None:
            if getattr(self._pm, "quant_mode", "none") != "none":
                raise RuntimeError(
                    f"cannot fully fine-tune quantized weights (quant={self._pm.quant_mode}); "
                    "train LoRA adapters instead (use_lora) or set SYNAPSE_WEIGHT_QUANT=none")
            if any(stage_lora.has_lora(l) for l in layers):
                raise RuntimeError("layers carry LoRA adapters: merge them (stage_lora_merge) "
                                   "before full fine-tuning")
            self._lora_cfg = None
        else:
            if self._lora_cfg is not None and self._lora_cfg != lora:
                raise RuntimeError(f"LoRA already set up with {self._lora_cfg}; merge first")
            with self._forward_lock:
                wrapped = sum(stage_lora.inject(l, lora) for l in layers)
                if wrapped and getattr(self, "_fast_dec", None) is not None:
                    self._fast_dec.forget()  # projections replaced: rebuild plans
            if not any(stage_lora.has_lora(l) for l in layers):
                raise RuntimeError(f"no LoRA target {lora['target_modules']} in this model's layers")
            self._lora_cfg = lora
        for _, p in self._stage_params(shard):
            p.requires_grad_(True)

    def stage_lora_merge(self, shard) -> int:
        """Fold this stage's adapters into its weights (full-speed inference)
        and return to full fine-tuning mode. Returns adapters merged."""
        from synapse.training import stage_lora
        with self._forward_lock:
            n = sum(stage_lora.merge(self._pm.layers[i])
                    for i in range(shard.start_layer, shard.end_layer + 1) if i in self._pm.layers)
            if getattr(self, "_fast_dec", None) is not None:
                self._fast_dec.forget()
            self._seq_kv.clear()
            self._stage_opt = None
            self._lora_cfg = None
        return n

    def stage_lora_state(self, shard) -> Dict[str, Any]:
        """This stage's adapters (CPU tensors) + settings, for saving."""
        if self._lora_cfg is None:
            return {"config": None, "tensors": {}}
        return {"config": self._lora_cfg,
                "tensors": {n: p.detach().cpu().clone() for n, p in self._stage_params(shard)}}

    # -- forward / backward -------------------------------------------------
    def _stage_modules(self, shard):
        pm = self._pm
        mods = [pm.layers[i] for i in range(shard.start_layer, shard.end_layer + 1)]
        mods += [m for m in (pm.embed if shard.is_first_layer() else None,
                             pm.final_norm if shard.is_last_layer() else None) if m is not None]
        return mods

    def _stage_run(self, shard, data: np.ndarray, labels: Optional[np.ndarray], grad: bool):
        """Forward through the stage. Returns (input_leaf, output, loss)."""
        import torch
        import torch.nn.functional as F

        comps = self._get_transformer_components()
        device, dtype = self._device_dtype()
        inp = None
        if shard.is_first_layer():
            ids = torch.from_numpy(np.asarray(data).astype(np.int64)).to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            h = comps["embed"](ids)
            if comps.get("pos_embed") is not None:
                pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
                h = h + comps["pos_embed"](pos)
        else:
            x = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).to(device=device, dtype=dtype)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            inp = x.requires_grad_(grad)
            h = inp
        T = h.shape[1]
        pos = torch.arange(T, device=device).unsqueeze(0)
        pe = self._build_position_embeddings(h, pos)
        mask = self._causal_mask(T, 0, h.dtype, device)
        h = self._apply_blocks(comps["layers"], shard, h, mask, pos, pe, None,
                               torch.arange(T, device=device), grad=grad)
        loss = None
        if shard.is_last_layer():
            if comps["final_norm"] is not None:
                h = comps["final_norm"](h)
            logits = comps["lm_head"](h)
            lab = torch.from_numpy(np.asarray(labels).astype(np.int64)).to(device)
            if lab.dim() == 1:
                lab = lab.unsqueeze(0)
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.shape[-1]).float(),
                                   lab[..., 1:].reshape(-1), ignore_index=-100)
        return inp, h, loss

    def _stage_forward_sync(self, key, shard, data, labels) -> Dict[str, Any]:
        import torch
        mods = self._stage_modules(shard)
        for m in mods:
            m.train()
        try:
            with torch.enable_grad():
                inp, out, loss = self._stage_run(shard, data, labels, grad=True)
                if loss is not None:
                    loss.backward()  # the last stage owns the loss: backprop now
                    return {"loss": float(loss.detach()),
                            "input_grad": None if inp is None else inp.grad.float().cpu().numpy()}
            self._pp_graphs[key] = (inp, out)
            return {"hidden": out.detach().float().cpu().numpy()}
        finally:
            for m in mods:
                m.eval()

    def _stage_backward_sync(self, key, grad: np.ndarray) -> Optional[np.ndarray]:
        import torch
        inp, out = self._pp_graphs.pop(key)
        g = torch.from_numpy(np.ascontiguousarray(grad, dtype=np.float32)).to(out.device, out.dtype)
        out.backward(g.view_as(out))
        return None if inp is None else inp.grad.float().cpu().numpy()

    @property
    def _pp_graphs(self) -> Dict[Any, Any]:
        g = getattr(self, "_pp_graph_store", None)
        if g is None:
            g = self._pp_graph_store = {}
        return g

    async def _locked(self, fn, *args):
        def run():
            with self._forward_lock:
                return fn(*args)
        return await asyncio.to_thread(run)

    async def stage_forward(self, key, shard, data: np.ndarray,
                            labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run this stage's forward for one micro-batch, keeping the graph.

        Non-last: {"hidden": output rows} (graph kept under `key` until
        stage_backward). Last: computes the loss and backprops at once:
        {"loss": float, "input_grad": d loss / d input or None}."""
        return await self._locked(self._stage_forward_sync, key, shard, data, labels)

    async def stage_backward(self, key, grad: np.ndarray) -> Optional[np.ndarray]:
        """Backprop d loss / d output into this stage's params; returns
        d loss / d input (None on the first stage)."""
        return await self._locked(self._stage_backward_sync, key, grad)

    def stage_drop(self, job) -> None:
        """Forget every kept graph of `job` (failed or abandoned step)."""
        for k in [k for k in self._pp_graphs if k[0] == job]:
            del self._pp_graphs[k]

    # -- optimizer step -----------------------------------------------------
    def stage_tied_grad(self, shard) -> Optional[np.ndarray]:
        p = self._tied_param(shard)
        if p is None or p.grad is None:
            return None
        return p.grad.detach().float().cpu().numpy()

    def stage_add_tied_grad(self, shard, other: Optional[np.ndarray]) -> None:
        """Sum the peer's gradient into our copy of the tied embedding: both
        copies then take the same optimizer step and stay identical."""
        import torch
        p = self._tied_param(shard)
        if p is None or other is None:
            return
        g = torch.from_numpy(np.asarray(other, dtype=np.float32)).to(p.device, p.dtype)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.add_(g)

    def stage_grad_sq(self, shard, scale: float = 1.0) -> float:
        """Sum of squared (scaled) gradients of this stage, for global-norm
        clipping. A tied copy on the last stage is left out (the first
        stage counts the shared tensor once)."""
        tied = self._tied_param(shard) if shard.is_last_layer() and not shard.is_first_layer() else None
        total = 0.0
        for _, p in self._stage_params(shard):
            if p.grad is not None and p is not tied:
                total += float(p.grad.detach().float().pow(2).sum()) * scale * scale
        return total

    def _stage_optimizer(self, shard, lr: float):
        import torch
        params = [p for _, p in self._stage_params(shard)]
        key = tuple(id(p) for p in params)
        opt = getattr(self, "_stage_opt", None)
        if opt is None or getattr(self, "_stage_opt_key", None) != key:
            opt = self._stage_opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.WEIGHT_DECAY)
            self._stage_opt_key = key
        for g in opt.param_groups:
            g["lr"] = lr
        return opt

    def _stage_step_sync(self, shard, lr: float, scale: float) -> None:
        opt = self._stage_optimizer(shard, lr)
        for _, p in self._stage_params(shard):
            if p.grad is not None and scale != 1.0:
                p.grad.mul_(scale)
        opt.step()
        opt.zero_grad(set_to_none=True)
        # Weights changed under every cached prefix: inference KV is stale.
        self._seq_kv.clear()

    async def stage_step(self, shard, lr: float, scale: float = 1.0) -> None:
        """Apply this stage's optimizer step with grads multiplied by
        `scale` (1/micro-batches x the global clip coefficient)."""
        await self._locked(self._stage_step_sync, shard, lr, scale)

    # -- single-node conveniences (whole model on one node) -----------------
    async def stage_loss(self, shard, ids: np.ndarray, labels: np.ndarray) -> float:
        """Evaluation loss of a full-range shard, no graph kept."""
        import torch

        def run():
            with torch.no_grad():
                return float(self._stage_run(shard, ids, labels, grad=False)[2])
        return await self._locked(run)

    def stage_named_grads(self, shard) -> Dict[str, np.ndarray]:
        return {n: p.grad.detach().float().cpu().numpy()
                for n, p in self._stage_params(shard) if p.grad is not None}
