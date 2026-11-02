"""
CPU kernels for weight-only int8 / int4 linears (numba, optional).

Decode is memory-bound: every token reads every weight once. Reading int8
(1 byte) or packed int4 (0.5 byte) instead of fp32 (4 bytes) only pays off if
the kernel never materializes the fp32 matrix. PyTorch's CPU paths here do
(int8 -> fp32 conversion runs at ~1 G elements/s on the test machine, and
aten._weight_int8pack_mm / _weight_int4pack_mm_for_cpu measured 4-12x slower
than fp32), and its dynamic-quantized linears quantize the activations too,
which SmolLM2's activation outliers do not survive (top-1 49%).

These kernels read the packed weights directly and accumulate in fp32, so
they compute exactly x @ dequant(W)^T. Measured (4 threads, M=1):
4096x4096 fp32 4.4 ms, int8 2.0 ms, int4 4.2 ms.

Without numba (or for unsupported inputs) callers fall back to torch.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

try:  # optional dependency
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False


def enabled() -> bool:
    """SYNAPSE_QKERNELS=0 forces the torch fallback (debugging, A/B)."""
    return HAVE_NUMBA and os.getenv("SYNAPSE_QKERNELS", "1") != "0"


if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _i8_gemm(q, s, x, y):
        # y[m, n] = s[n] * sum_k q[n, k] * x[m, k]; one weight row stays in
        # cache while every input row uses it.
        N, K = q.shape
        M = x.shape[0]
        for n in prange(N):
            for m in range(M):
                acc = np.float32(0.0)
                for k in range(K):
                    acc += np.float32(q[n, k]) * x[m, k]
                y[m, n] = acc * s[n]

    @njit(parallel=True, fastmath=True, cache=True)
    def _i4_gemm(p, sc, zp, xe, xo, xg, y):
        # p[n, j] packs columns 2j (low nibble) and 2j+1 (high nibble);
        # w = q * scale + zero per group, so
        # y[m, n] = sum_g scale[n,g] * (q . x)_g + zero[n,g] * sum(x_g).
        # xe / xo are the even / odd input columns, xg the per-group sums.
        N, KH = p.shape
        G = sc.shape[1]
        H = KH // G
        M = xe.shape[0]
        for n in prange(N):
            for m in range(M):
                tot = np.float32(0.0)
                for g in range(G):
                    acc = np.float32(0.0)
                    o = g * H
                    for j in range(H):
                        b = np.int32(p[n, o + j])
                        acc += np.float32(b & 15) * xe[m, o + j] + np.float32(b >> 4) * xo[m, o + j]
                    tot += acc * sc[n, g] + zp[n, g] * xg[m, g]
                y[m, n] = tot


_threads_synced = False


def _sync_threads() -> None:
    """Use as many numba threads as torch: both pools on all logical cores
    oversubscribe the physical ones."""
    global _threads_synced
    if not _threads_synced:
        try:
            import numba
            numba.set_num_threads(max(1, min(numba.config.NUMBA_NUM_THREADS, torch.get_num_threads())))
        except Exception:
            pass
        _threads_synced = True


def _rows(x: torch.Tensor) -> Optional[np.ndarray]:
    if x.device.type != "cpu" or x.requires_grad and torch.is_grad_enabled():
        return None
    x2 = x.detach().reshape(-1, x.shape[-1])
    if x2.dtype != torch.float32:
        x2 = x2.float()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    return x2.numpy()


def int8_linear(x: torch.Tensor, q: torch.Tensor, s32: torch.Tensor) -> Optional[torch.Tensor]:
    """x @ (q * s)^T for q int8 [N, K], s fp32 [N]; None when not applicable."""
    if not enabled() or q.device.type != "cpu":
        return None
    xr = _rows(x)
    if xr is None:
        return None
    _sync_threads()
    y = torch.empty((xr.shape[0], q.shape[0]), dtype=torch.float32)
    _i8_gemm(q.numpy(), s32.numpy(), xr, y.numpy())
    y = y.view(*x.shape[:-1], q.shape[0])
    return y if x.dtype == torch.float32 else y.to(x.dtype)


def int4_linear(x: torch.Tensor, packed: torch.Tensor, sc32: torch.Tensor, zp32: torch.Tensor,
                in_features: int, group: int) -> Optional[torch.Tensor]:
    """x @ dequant(W)^T for group-wise asymmetric int4 (see weight_quant)."""
    if not enabled() or packed.device.type != "cpu" or group % 2:
        return None
    xr = _rows(x)
    if xr is None:
        return None
    _sync_threads()
    k_pad = packed.shape[1] * 2
    if k_pad != in_features:
        xr = np.ascontiguousarray(np.pad(xr, ((0, 0), (0, k_pad - in_features))))
    xe = np.ascontiguousarray(xr[:, 0::2])
    xo = np.ascontiguousarray(xr[:, 1::2])
    xg = np.ascontiguousarray(xr.reshape(xr.shape[0], -1, group).sum(axis=2, dtype=np.float32))
    y = torch.empty((xr.shape[0], packed.shape[0]), dtype=torch.float32)
    _i4_gemm(packed.numpy(), sc32.numpy(), zp32.numpy(), xe, xo, xg, y.numpy())
    y = y.view(*x.shape[:-1], packed.shape[0])
    return y if x.dtype == torch.float32 else y.to(x.dtype)
