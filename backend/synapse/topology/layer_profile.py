"""
What one decoder layer of a model costs on THIS node, in milliseconds, and
how many layers fit in its memory. Published in capability gossip and fed
to the DP placement (synapse.topology.dp_partition).

ms_per_layer comes from real forwards when the engine has run decode steps
of that model ("forward"); otherwise from a matrix-vector bandwidth probe
scaled by the layer's weight bytes ("probe"): decode reads every weight once
per token, so on CPUs and GPUs alike it is bound by memory bandwidth.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

_PARAMS_CACHE: Dict[str, int] = {}
_PROBE_CACHE: Dict[str, float] = {}


def layer_params(model_id: str) -> Optional[int]:
    """Parameters in one decoder layer, from the config (meta skeleton,
    no weights read). None when the config is unavailable."""
    if model_id in _PARAMS_CACHE:
        return _PARAMS_CACHE[model_id]
    try:
        import copy
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from synapse.model_list import resolve_hf_id
        cfg = copy.deepcopy(AutoConfig.from_pretrained(resolve_hf_id(model_id), trust_remote_code=True))
        cfg.num_hidden_layers = 1
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        from synapse.inference.partial_loader import PartialModel
        layers = m
        for part in PartialModel._find_layers_path(m).split("."):
            layers = getattr(layers, part)
        n = sum(p.numel() for p in layers[0].parameters())
        _PARAMS_CACHE[model_id] = n
        return n
    except Exception:
        return None


def probe_ms_per_mb(device: str = "cpu", mb: int = 32, reps: int = 5) -> float:
    """Milliseconds to stream 1 MB of fp32 weights through a matvec."""
    key = f"{device}:{mb}"
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    import torch
    cols = 4096
    rows = max(1, (mb * 1024 * 1024) // (4 * cols))
    w = torch.randn(rows, cols, device=device)
    x = torch.randn(cols, 1, device=device)
    for _ in range(2):
        (w @ x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        (w @ x)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / reps
    val = ms / (rows * cols * 4 / (1024 * 1024))
    _PROBE_CACHE[key] = val
    return val


def bytes_per_param(engine: Any) -> float:
    pm = getattr(engine, "_pm", None)
    mode = getattr(pm, "quant_mode", "none") if pm is not None else "none"
    if mode == "int8":
        return 1.0 + 1e-3
    if mode == "int4":
        return 0.5 + 0.13
    dtype = getattr(pm, "dtype", None)
    try:
        import torch
        return float(torch.tensor([], dtype=dtype).element_size()) if dtype is not None else 4.0
    except Exception:
        return 4.0


def available_bytes(engine: Any) -> int:
    fn = getattr(engine, "_available_bytes", None)
    pm = getattr(engine, "_pm", None)
    device = getattr(pm, "device", None) or ("cuda" if getattr(engine, "_device", "cpu") == "cuda" else "cpu")
    try:
        free = int(fn(device)) if callable(fn) else 0
    except Exception:
        free = 0
    # Layers this node already holds are part of what it can offer.
    try:
        held = int(pm.loaded_bytes()) if pm is not None else 0
    except Exception:
        held = 0
    return free + held


def local_profile(engine: Any, model_id: str) -> Optional[Dict[str, Any]]:
    """{"ms_per_layer", "max_layers", "source"} for model_id, or None."""
    n = layer_params(model_id)
    if not n:
        return None
    layer_bytes = n * bytes_per_param(engine)
    measured = (getattr(engine, "layer_ms", {}) or {}).get(model_id)
    if measured:
        ms, source = float(measured), "forward"
    else:
        pm = getattr(engine, "_pm", None)
        device = getattr(pm, "device", "cpu") if pm is not None else "cpu"
        ms, source = probe_ms_per_mb(device) * layer_bytes / (1024 * 1024), "probe"
    avail = available_bytes(engine)
    max_layers = int((avail * 0.8) // layer_bytes) if avail > 0 else 0
    return {"ms_per_layer": round(ms, 4), "max_layers": max_layers,
            "layer_bytes": int(layer_bytes), "source": source, "ts": time.time()}
