"""
Weight-only quantization for the layers a node serves (int8 / int4).

Fewer bytes per layer means fewer nodes are needed to hold a model, and
every node removed from the chain removes one network hop per token (the
Petals paper halves BLOOM-176B's node count this way). Pure PyTorch, so it
works on CPU, CUDA and DirectML alike; bitsandbytes is CUDA-only.

- int8: symmetric, one scale per output row.
- int4: asymmetric (scale + zero point) per group of GROUP input columns,
  two values packed per byte. Scale and zero are fitted per group (range
  shrink chosen by squared error, then a least-squares refit), and the
  tensors int4 hurts most stay int8 (mixed precision, like llama.cpp's
  Q4_K_M): v_proj / down_proj and the first and last INT4_END_LAYERS layers.
  On SmolLM2-135M this took int4 from KL 0.63 / top-1 51% to KL ~0.07.

On CPU, forward runs numba kernels that read the packed weights directly
(qkernels.py: int8 2-3x faster than fp32 on decode, int4 about as fast);
elsewhere the weights are dequantized inside forward. Embeddings and the LM
head stay unquantized. Quality is gated by tests/test_partial_load_and_quant.py.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

GROUP = 32
MODES = ("none", "int8", "int4")
# int4 mode keeps these linears (dotted-path suffixes) and the first/last
# INT4_END_LAYERS decoder layers in int8. SYNAPSE_INT4_KEEP overrides the
# list ("none" = pure int4).
INT4_KEEP_INT8 = ("v_proj", "down_proj", "mlp.c_proj")
INT4_END_LAYERS = 2
_SHRINK_GRID = (1.0, 0.95, 0.9, 0.85, 0.8)


def _fit_int4(g: torch.Tensor):
    """Per-group asymmetric 4-bit fit of g [rows, groups, GROUP].

    For each range-shrink factor, quantize, refit scale/zero by least squares
    for those codes (a few rounds), and keep the factor with the smallest
    squared error per group. Returns (codes uint8, scale fp16, zero fp16),
    with the codes recomputed from the fp16 scale/zero actually stored."""
    lo0 = g.amin(dim=2, keepdim=True)
    hi0 = g.amax(dim=2, keepdim=True)
    mid, half = (lo0 + hi0) / 2, (hi0 - lo0) / 2
    gm = g.mean(dim=2, keepdim=True)
    best_s = best_z = best_err = None
    for f in _SHRINK_GRID:
        s = (half * f * 2 / 15.0).clamp_min(1e-8)
        z = mid - half * f
        for _ in range(3):
            q = torch.round((g - z) / s).clamp_(0, 15)
            qm = q.mean(dim=2, keepdim=True)
            cov = ((q - qm) * (g - gm)).sum(dim=2, keepdim=True)
            var = ((q - qm) ** 2).sum(dim=2, keepdim=True)
            s = torch.where(var > 0, cov / var.clamp_min(1e-12), s).clamp_min(1e-8)
            z = gm - s * qm
        q = torch.round((g - z) / s).clamp_(0, 15)
        err = ((q * s + z - g) ** 2).sum(dim=2, keepdim=True)
        if best_err is None:
            best_s, best_z, best_err = s, z, err
        else:
            better = err < best_err
            best_s = torch.where(better, s, best_s)
            best_z = torch.where(better, z, best_z)
            best_err = torch.where(better, err, best_err)
    s16 = best_s.clamp_min(6e-8).to(torch.float16)
    z16 = best_z.to(torch.float16)
    q = torch.round((g - z16.float()) / s16.float()).clamp_(0, 15).to(torch.uint8)
    return q, s16.squeeze(2), z16.squeeze(2)


class QuantLinear(nn.Module):
    """y = x @ W^T + b with W stored as int8 (per-row) or packed int4."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], mode: str):
        super().__init__()
        if mode not in ("int8", "int4"):
            raise ValueError(f"unknown quant mode {mode!r}")
        self.mode = mode
        self.out_features, self.in_features = weight.shape
        self.compute_dtype = weight.dtype
        w = weight.detach().float()
        if mode == "int8":
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
            self.register_buffer("qweight", q)
            self.register_buffer("scales", scale.to(torch.float16))
        else:
            pad = (-self.in_features) % GROUP
            if pad:
                w = F.pad(w, (0, pad))
            q, scale, zero = _fit_int4(w.view(self.out_features, -1, GROUP))
            q = q.view(self.out_features, -1)
            packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
            self.register_buffer("qweight", packed)
            self.register_buffer("scales", scale)
            self.register_buffer("zeros", zero)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)

    def dequantize(self) -> torch.Tensor:
        if self.mode == "int8":
            w = self.qweight.float() * self.scales.float()
        else:
            p = self.qweight
            q = torch.empty(p.shape[0], p.shape[1] * 2, dtype=torch.int16, device=p.device)
            q[:, 0::2] = (p & 0x0F).to(torch.int16)
            q[:, 1::2] = (p >> 4).to(torch.int16)
            g = q.float().view(self.out_features, -1, GROUP)
            w = (g * self.scales.float().unsqueeze(2) + self.zeros.float().unsqueeze(2))
            w = w.view(self.out_features, -1)
            w = w[:, : self.in_features]
        return w.to(self.compute_dtype)

    def _fp32_params(self):
        """fp32 copies of the (small) scale/zero tensors for the kernels;
        rebuilt if the buffers are replaced (e.g. load_state_dict)."""
        key = (self.scales.data_ptr(), None if self.mode == "int8" else self.zeros.data_ptr())
        if getattr(self, "_f32_key", None) != key:
            self._f32 = (self.scales.float().reshape(self.out_features, -1).contiguous(),
                         None if self.mode == "int8" else self.zeros.float().contiguous())
            self._f32_key = key
        return self._f32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from synapse.inference import qkernels
        y = None
        if x.device.type == "cpu":
            s32, z32 = self._fp32_params()
            if self.mode == "int8":
                y = qkernels.int8_linear(x, self.qweight, s32.view(-1))
            else:
                y = qkernels.int4_linear(x, self.qweight, s32, z32, self.in_features, GROUP)
        if y is None:
            if self.mode == "int8":
                # Scale the outputs, not the matrix: one pass fewer over W.
                y = F.linear(x, self.qweight.to(x.dtype)) * self.scales.view(-1).to(x.dtype)
            else:
                y = F.linear(x, self.dequantize().to(x.dtype))
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, {self.mode}"


def _conv1d_type():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except Exception:
        return None


def int4_keep() -> tuple:
    env = os.getenv("SYNAPSE_INT4_KEEP")
    if env is None:
        return INT4_KEEP_INT8
    env = env.strip()
    if env.lower() in ("", "none"):
        return ()
    return tuple(x.strip() for x in env.split(",") if x.strip())


def layer_mode(mode: str, layer_index: int, n_layers: int) -> str:
    """Mode for one decoder layer: int4 keeps the first/last layers in int8."""
    if mode == "int4" and int4_keep() and (
            layer_index < INT4_END_LAYERS or layer_index >= n_layers - INT4_END_LAYERS):
        return "int8"
    return mode


def quantize_module(module: nn.Module, mode: str, _prefix: str = "") -> int:
    """Replace every nn.Linear (and GPT-2 Conv1D) inside `module` in place.

    In int4 mode the linears named in int4_keep() become int8 instead.
    Returns the number of replaced layers. mode "none" is a no-op."""
    if mode in (None, "", "none"):
        return 0
    conv1d = _conv1d_type()
    keep = int4_keep() if mode == "int4" else ()
    count = 0
    for name, child in list(module.named_children()):
        path = f"{_prefix}.{name}" if _prefix else name
        m = "int8" if any(path == k or path.endswith("." + k) for k in keep) else mode
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child.weight, child.bias, m))
            count += 1
        elif conv1d is not None and isinstance(child, conv1d):
            # Conv1D computes x @ W + b with W [in, out]: same as Linear(W^T).
            setattr(module, name, QuantLinear(child.weight.t().contiguous(), child.bias, m))
            count += 1
        else:
            count += quantize_module(child, mode, path)
    return count


def module_bytes(module: nn.Module, seen: Optional[set] = None) -> int:
    """Bytes actually held by a module's parameters and buffers. Pass the
    same `seen` set across modules to count shared (tied) tensors once."""
    seen = set() if seen is None else seen
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def requested_mode() -> str:
    """SYNAPSE_WEIGHT_QUANT = none | int8 | int4 | auto (default auto)."""
    mode = os.getenv("SYNAPSE_WEIGHT_QUANT", "auto").strip().lower()
    return mode if mode in MODES + ("auto",) else "auto"


def choose_mode(needed_bytes: int, available_bytes: int, requested: str) -> str:
    """Resolve "auto": quantize only when the fp weights would not fit.

    Keeps a 20% margin for activations and the KV cache."""
    if requested != "auto":
        return requested
    if available_bytes <= 0 or needed_bytes <= 0:
        return "none"
    budget = available_bytes * 0.8
    if needed_bytes <= budget:
        return "none"
    # int8 ~ 1 byte/param vs 4 (fp32) or 2 (fp16/bf16): judge by ratio.
    if needed_bytes / 2 <= budget:
        return "int8"
    return "int4"
