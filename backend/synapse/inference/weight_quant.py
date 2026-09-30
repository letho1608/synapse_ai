"""
Weight-only quantization for the layers a node serves (int8 / int4).

Fewer bytes per layer means fewer nodes are needed to hold a model, and
every node removed from the chain removes one network hop per token (the
Petals paper halves BLOOM-176B's node count this way). Pure PyTorch, so it
works on CPU, CUDA and DirectML alike; bitsandbytes is CUDA-only.

- int8: symmetric, one scale per output row.
- int4: asymmetric (scale + zero point) per group of GROUP input columns,
  two values packed per byte.

Weights are dequantized to the compute dtype inside forward (memory drops,
compute stays in fp32/fp16). Embeddings and the LM head stay unquantized.
Quality is gated by tests/test_weight_quant.py on the default model.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

GROUP = 32
MODES = ("none", "int8", "int4")


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
            g = w.view(self.out_features, -1, GROUP)
            # Asymmetric (min/max) per group: all 16 levels span the group's
            # actual range. Symmetric int4 lost half the top-1 agreement on
            # SmolLM2-135M.
            lo = g.amin(dim=2, keepdim=True)
            hi = g.amax(dim=2, keepdim=True)
            scale = ((hi - lo) / 15.0).clamp_min(1e-8)
            q = torch.round((g - lo) / scale).clamp_(0, 15).to(torch.uint8)
            q = q.view(self.out_features, -1)
            packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
            self.register_buffer("qweight", packed)
            self.register_buffer("scales", scale.squeeze(2).to(torch.float16))
            self.register_buffer("zeros", lo.squeeze(2).to(torch.float16))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize().to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, {self.mode}"


def _conv1d_type():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except Exception:
        return None


def quantize_module(module: nn.Module, mode: str) -> int:
    """Replace every nn.Linear (and GPT-2 Conv1D) inside `module` in place.

    Returns the number of replaced layers. mode "none" is a no-op."""
    if mode in (None, "", "none"):
        return 0
    conv1d = _conv1d_type()
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child.weight, child.bias, mode))
            count += 1
        elif conv1d is not None and isinstance(child, conv1d):
            # Conv1D computes x @ W + b with W [in, out]: same as Linear(W^T).
            setattr(module, name, QuantLinear(child.weight.t().contiguous(), child.bias, mode))
            count += 1
        else:
            count += quantize_module(child, mode)
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
