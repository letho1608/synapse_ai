"""
LoRA on the decoder layers a node serves.

PEFT wraps a whole PreTrainedModel, so LoRA used to load the full model on
one machine. Here adapters go straight into the PartialModel layers of this
node's range: base weights stay frozen (plain or int8/int4 QuantLinear, so
QLoRA-style training works), only A/B are trained, and each pipeline stage
trains its own adapters exactly like full fine-tuning trains its own layers.

    y = base(x) + (dropout(x) @ A^T @ B^T) * alpha / r

B starts at zero, so a fresh adapter leaves the model's output unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn

# attention projections: Llama/Mistral/Qwen names, and GPT-2's fused c_attn
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn")


def normalize_config(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Canonical, JSON-safe LoRA settings (sent to every stage), or None."""
    if not cfg:
        return None
    targets = cfg.get("target_modules") or cfg.get("lora_target_modules") or DEFAULT_TARGETS
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    return {
        "r": int(cfg.get("r", cfg.get("lora_r", 8))),
        "alpha": float(cfg.get("alpha", cfg.get("lora_alpha", 16))),
        "dropout": float(cfg.get("dropout", cfg.get("lora_dropout", 0.0))),
        "target_modules": sorted(set(targets)),
    }


class LoRALinear(nn.Module):
    """A frozen linear (nn.Linear, QuantLinear or GPT-2 Conv1D) plus a
    trainable low-rank update."""

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        if _is_conv1d(base):  # GPT-2: weight is [in, out]
            in_f, out_f = (int(d) for d in base.weight.shape)
        else:
            out_f, in_f = int(base.out_features), int(base.in_features)
        ref = next((t for t in list(base.parameters()) + list(base.buffers())
                    if t.is_floating_point()), None)
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = getattr(base, "compute_dtype", None) or (ref.dtype if ref is not None else torch.float32)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.in_features, self.out_features = in_f, out_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        upd = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()
        return y + (upd * self.scaling).to(y.dtype)

    def merged_weight(self) -> torch.Tensor:
        """Base weight with the update folded in (unquantized base only)."""
        delta = (self.lora_B.data @ self.lora_A.data) * self.scaling
        if _is_conv1d(self.base):
            delta = delta.t()
        return self.base.weight.data + delta.to(self.base.weight.dtype)


def _is_conv1d(m: nn.Module) -> bool:
    return type(m).__name__ == "Conv1D" and hasattr(m, "nf")


def _is_linear(m: nn.Module) -> bool:
    from synapse.inference.weight_quant import QuantLinear
    return isinstance(m, (nn.Linear, QuantLinear)) or _is_conv1d(m)


def inject(layer: nn.Module, cfg: Dict[str, Any]) -> int:
    """Wrap every target linear of `layer` in LoRALinear (idempotent);
    returns how many were wrapped now."""
    targets = set(cfg["target_modules"])
    n = 0
    for parent in list(layer.modules()):
        if isinstance(parent, LoRALinear):
            continue
        for name, child in list(parent.named_children()):
            if name in targets and _is_linear(child):
                setattr(parent, name, LoRALinear(child, cfg["r"], cfg["alpha"], cfg["dropout"]))
                n += 1
    return n


def has_lora(layer: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in layer.modules())


def lora_parameters(layer: nn.Module) -> List[tuple]:
    return [(n, p) for n, p in layer.named_parameters() if "lora_" in n]


def merge(layer: nn.Module) -> int:
    """Fold adapters over plain nn.Linear into the base weights and unwrap
    them (inference runs at full speed again). Adapters over quantized
    weights stay wrapped: folding would need a re-quantization. Returns the
    number merged."""
    n = 0
    for parent in list(layer.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear) and (type(child.base) is nn.Linear or _is_conv1d(child.base)):
                with torch.no_grad():
                    # in place: the fast decoder's fused QKV/MLP tensor is a
                    # view of these weights and must see the update
                    child.base.weight.data.copy_(child.merged_weight())
                setattr(parent, name, child.base)
                n += 1
    return n
