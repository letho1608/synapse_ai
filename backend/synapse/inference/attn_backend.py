"""
Attention-backend capability probe (T3.4 ladder rung-4 prerequisite / C6).

Reports the best attention implementation THIS machine can actually run —
never a wish list. Order: flash-attn-2 (needs CUDA + package) -> triton
paged kernels (needs package + GPU) -> sdpa (torch built-in) -> eager.

Checkers are injectable so selection order is unit-testable without GPUs;
the default checkers perform real imports + torch.cuda gating. Anything
requiring SM90 / torchao FP8 / custom csrc stays out (GPU-gated, see the
T3.4 note at the bottom): this module only answers "what exists here".
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

BACKENDS = ("fa2", "triton", "sdpa", "eager")


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _check_fa2() -> Tuple[bool, str]:
    try:
        import flash_attn  # noqa: F401
    except Exception as e:
        return False, "flash_attn import failed: %s" % (e,)
    if not _has_cuda():
        return False, "flash_attn present but no CUDA device"
    return True, "ok"


def _check_triton() -> Tuple[bool, str]:
    try:
        import triton  # noqa: F401
    except Exception as e:
        return False, "triton import failed: %s" % (e,)
    if not _has_cuda():
        return False, "triton present but no CUDA device"
    return True, "ok"


def _check_sdpa() -> Tuple[bool, str]:
    try:
        import torch
        from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
        return True, "torch %s" % getattr(torch, "__version__", "?")
    except Exception as e:
        return False, "sdpa unavailable: %s" % (e,)


def _check_eager() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
        return True, "always available with torch"
    except Exception as e:
        return False, "no torch: %s" % (e,)


DEFAULT_CHECKERS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "fa2": _check_fa2,
    "triton": _check_triton,
    "sdpa": _check_sdpa,
    "eager": _check_eager,
}


def probe_attn_backend(
    prefer: Tuple[str, ...] = BACKENDS,
    checkers: Optional[Dict[str, Callable[[], Tuple[bool, str]]]] = None,
) -> Dict[str, object]:
    """Return {name, available: {backend: bool}, reasons: {backend: str}}.

    name is the first available backend in prefer order; falls back to the
    last entry (eager) only when its checker passes, else None (no torch).
    """
    checks = checkers or DEFAULT_CHECKERS
    available: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}
    for name in BACKENDS:
        fn = checks.get(name)
        if fn is None:
            available[name], reasons[name] = False, "no checker"
            continue
        try:
            ok, why = fn()
        except Exception as e:
            ok, why = False, "checker raised: %s" % (e,)
        available[name], reasons[name] = bool(ok), str(why)
    chosen = next((n for n in prefer if available.get(n)), None)
    out: Dict[str, object] = {"name": chosen, "available": available,
                              "reasons": reasons}
    return out


def available_backends(**kw) -> List[str]:
    """Backends usable here, best first."""
    probed = probe_attn_backend(**kw)
    avail = probed["available"]
    if not isinstance(avail, dict):
        return []
    return [n for n in BACKENDS if avail.get(n)]


# ── GPU-gated T3.4 remainder (NOT done — needs SM90/GPU + model + C1) ──
# - torchao FP8 weights (rung 3): torchao not probed here on purpose; adding
#   it as a selection without a perplexity/token-parity gate and SM90
#   hardware to validate on would be an unmeasured claim (red-team M2).
# - Fused attention + torch.compile fixed-shape decode (rung 4): requires the
#   T3.1 paged-KV integration first (stable decode shapes), then A/B numbers
#   on GPU. The probe above is the capability input that rung consumes.
