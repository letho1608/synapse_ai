"""
Fast decoder path for Llama-family layers + a paged KV cache.

Why: on CPU a decode step of a small model spends about half its time in
Python inside the HF modules (per-layer kwargs plumbing, config attribute
lookups, eager attention with repeat_kv) and DynamicCache re-concatenates
every layer's whole K/V on each token. This path runs the same math on the
same weights with fused projections (q|k|v and gate|up are one matmul each;
the HF modules keep views into the fused tensors, so nothing is duplicated)
and PyTorch SDPA, reading K/V from a paged pool.

Paged KV (vLLM-style): every engine owns one KVPool of fixed-size blocks per
layer; a request's PagedSequence is a block table plus a length. Cropping
after a rejected draft just shortens the length and returns whole blocks to
the pool, finished requests give their blocks back, and memory is shared by
all requests instead of each holding a growing contiguous tensor.

Covered: model_type llama / mistral / qwen2 / qwen3 (q/k norm) with SiLU MLP,
full attention (no sliding window), rotary over the whole head. Anything
else returns None from FastDecoder.build and the HF modules run as before.
Logits match the HF path (same weights, same op order up to SDPA vs eager).
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3")
BLOCK = 16


def enabled() -> bool:
    """SYNAPSE_FAST_PATH=0 runs every forward through the HF modules."""
    return os.getenv("SYNAPSE_FAST_PATH", "1") != "0"


# ---------------------------------------------------------------------------
# Paged KV cache
# ---------------------------------------------------------------------------

class KVPool:
    """Fixed-size KV blocks per layer, shared by every request of an engine.

    Layer pools are [capacity, BLOCK, n_kv, head_dim] and are created lazily
    (a node only touches the layers it serves). The pool doubles when it runs
    out of blocks."""

    def __init__(self, n_kv: int, head_dim: int, dtype, device, block: int = BLOCK,
                 init_blocks: int = 64):
        self.n_kv, self.head_dim, self.dtype, self.device = n_kv, head_dim, dtype, device
        self.block = block
        self.capacity = init_blocks
        self.free: List[int] = list(range(init_blocks - 1, -1, -1))
        self.k: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}

    def _new(self, cap: int) -> torch.Tensor:
        return torch.zeros(cap, self.block, self.n_kv, self.head_dim,
                           dtype=self.dtype, device=self.device)

    def layer(self, li: int):
        if li not in self.k:
            self.k[li], self.v[li] = self._new(self.capacity), self._new(self.capacity)
        return self.k[li], self.v[li]

    def alloc(self, n: int) -> List[int]:
        while len(self.free) < n:
            self._grow()
        return [self.free.pop() for _ in range(n)]

    def release(self, blocks: Sequence[int]) -> None:
        self.free.extend(blocks)

    def _grow(self) -> None:
        old = self.capacity
        new = old * 2
        for store in (self.k, self.v):
            for li, t in store.items():
                bigger = self._new(new)
                bigger[:old] = t
                store[li] = bigger
        self.free.extend(range(new - 1, old - 1, -1))
        self.capacity = new

    @property
    def used_blocks(self) -> int:
        return self.capacity - len(self.free)


class PagedSequence:
    """One request's KV for one shard range: a block table and a length.

    Offers the subset of the HF Cache interface the engine relies on
    (get_seq_length, crop) so it can sit in the engine's per-request state
    exactly where a DynamicCache would."""

    def __init__(self, pool: KVPool):
        self.pool = pool
        self.blocks: List[int] = []
        self.length = 0
        self._index: Optional[torch.Tensor] = None  # flat slot of each position

    def __del__(self):
        try:
            self.pool.release(self.blocks)
            self.blocks = []
        except Exception:
            pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def crop(self, n: int) -> None:
        n = max(0, min(int(n), self.length))
        self.length = n
        keep = -(-n // self.pool.block)
        if keep < len(self.blocks):
            self.pool.release(self.blocks[keep:])
            del self.blocks[keep:]
        if self._index is not None and self._index.numel() > n:
            self._index = self._index[:n]

    def slots(self, start: int, count: int) -> torch.Tensor:
        """Flat pool slots for positions [start, start+count), allocating
        blocks as needed (the length is advanced separately, after the
        forward, by `advance`)."""
        need = -(-(start + count) // self.pool.block)
        if need > len(self.blocks):
            self.blocks.extend(self.pool.alloc(need - len(self.blocks)))
        total = start + count
        if self._index is None or self._index.numel() < total:
            pos = torch.arange(total, device=self.pool.device)
            table = torch.tensor(self.blocks, dtype=torch.long, device=self.pool.device)
            self._index = table[pos // self.pool.block] * self.pool.block + pos % self.pool.block
        return self._index[start:total]

    def advance(self, total: int) -> None:
        self.length = total

    def write(self, li: int, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
        """k, v: [T, n_kv, head_dim] for the positions in `slots`."""
        kp, vp = self.pool.layer(li)
        flat_k = kp.view(-1, self.pool.n_kv, self.pool.head_dim)
        flat_v = vp.view(-1, self.pool.n_kv, self.pool.head_dim)
        flat_k.index_copy_(0, slots, k)
        flat_v.index_copy_(0, slots, v)

    def read(self, li: int, total: int):
        """K, V for positions [0, total) as [1, n_kv, total, head_dim]."""
        kp, vp = self.pool.layer(li)
        idx = self._index[:total]
        k = kp.view(-1, self.pool.n_kv, self.pool.head_dim).index_select(0, idx)
        v = vp.view(-1, self.pool.n_kv, self.pool.head_dim).index_select(0, idx)
        return k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)


# ---------------------------------------------------------------------------
# Fast Llama-family decoder
# ---------------------------------------------------------------------------

def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Same op order as LlamaRMSNorm (fp32 statistics).
    dt = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dt)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


class _LayerPlan:
    """What one decoder layer needs on the fast path."""
    __slots__ = ("qkv_w", "qkv_b", "gu_w", "gu_b", "proj", "attn", "mlp",
                 "n1", "n2", "eps1", "eps2", "q_norm", "k_norm", "fused")


def _is_plain_linear(m) -> bool:
    return type(m) is nn.Linear


def _fuse_rows(mods: List[nn.Linear]):
    """Concatenate the weights (and biases) of linears sharing an input and
    re-point every module at a view of the fused tensor: one matmul instead
    of several, no extra memory, HF modules (training, fallback) unchanged."""
    w = torch.cat([m.weight.data for m in mods], dim=0).contiguous()
    has_b = [m.bias is not None for m in mods]
    b = None
    if any(has_b):
        b = torch.cat([m.bias.data if m.bias is not None else
                       torch.zeros(m.out_features, dtype=w.dtype, device=w.device)
                       for m in mods]).contiguous()
    off = 0
    for m in mods:
        n = m.out_features
        m.weight.data = w[off:off + n]
        if m.bias is not None:
            m.bias.data = b[off:off + n]
        off += n
    return w, b


class FastDecoder:
    """Runs decoder layers [start, end] with paged KV. Build with `build`."""

    def __init__(self, config, rotary: nn.Module, device, dtype):
        self.config = config
        self.rotary = rotary
        self.device, self.dtype = device, dtype
        self.n_heads = int(config.num_attention_heads)
        self.n_kv = int(getattr(config, "num_key_value_heads", None) or self.n_heads)
        self.head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // self.n_heads)
        self.scaling = self.head_dim ** -0.5
        self.pool = KVPool(self.n_kv, self.head_dim, dtype, device)
        self._plans: Dict[int, _LayerPlan] = {}

    # -- eligibility ------------------------------------------------------
    @staticmethod
    def build(config, rotary, device, dtype) -> Optional["FastDecoder"]:
        if not enabled() or rotary is None:
            return None
        if getattr(config, "model_type", None) not in SUPPORTED_MODEL_TYPES:
            return None
        if str(getattr(config, "hidden_act", "silu")) != "silu":
            return None
        if getattr(config, "model_type", None) == "mistral" and getattr(config, "sliding_window", None):
            return None
        if getattr(config, "use_sliding_window", False):
            return None
        types = getattr(config, "layer_types", None)
        if isinstance(types, list) and any(t != "full_attention" for t in types):
            return None
        return FastDecoder(config, rotary, device, dtype)

    def plan(self, li: int, layer: nn.Module) -> Optional[_LayerPlan]:
        p = self._plans.get(li)
        if p is not None and p.attn is getattr(layer, "self_attn", None) and self._still_valid(p):
            return p
        p = self._make_plan(layer)
        if p is not None:
            self._plans[li] = p
        return p

    @staticmethod
    def _still_valid(p: _LayerPlan) -> bool:
        """A plan goes stale when a projection is replaced (quantized) or its
        weight re-pointed away from the fused tensor."""
        q = p.attn.q_proj
        if p.fused != _is_plain_linear(q):
            return False
        return not p.fused or q.weight.data_ptr() == p.qkv_w.data_ptr()

    def forget(self, keep: Optional[Sequence[int]] = None) -> None:
        """Drop plans for layers no longer loaded (release_layers)."""
        keep = set(keep or ())
        for li in [i for i in self._plans if i not in keep]:
            del self._plans[li]

    def _make_plan(self, layer) -> Optional[_LayerPlan]:
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        n1 = getattr(layer, "input_layernorm", None)
        n2 = getattr(layer, "post_attention_layernorm", None)
        if attn is None or mlp is None or n1 is None or n2 is None:
            return None
        need = [getattr(attn, a, None) for a in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
               [getattr(mlp, a, None) for a in ("gate_proj", "up_proj", "down_proj")]
        if any(m is None for m in need):
            return None
        if getattr(attn, "sliding_window", None):
            return None
        if getattr(attn, "head_dim", self.head_dim) != self.head_dim:
            return None
        p = _LayerPlan()
        p.attn, p.mlp = attn, mlp
        p.n1, p.n2 = n1.weight, n2.weight
        p.eps1 = float(getattr(n1, "variance_epsilon", getattr(n1, "eps", 1e-6)))
        p.eps2 = float(getattr(n2, "variance_epsilon", getattr(n2, "eps", 1e-6)))
        p.q_norm = getattr(attn, "q_norm", None)
        p.k_norm = getattr(attn, "k_norm", None)
        p.fused = all(_is_plain_linear(m) for m in need)
        if p.fused:
            p.qkv_w, p.qkv_b = _fuse_rows([attn.q_proj, attn.k_proj, attn.v_proj])
            p.gu_w, p.gu_b = _fuse_rows([mlp.gate_proj, mlp.up_proj])
        else:
            # Quantized (QuantLinear) layers: call the modules themselves.
            p.qkv_w = p.qkv_b = p.gu_w = p.gu_b = None
        return p

    # -- forward ----------------------------------------------------------
    def _qkv(self, p: _LayerPlan, h: torch.Tensor):
        if p.fused:
            qkv = F.linear(h, p.qkv_w, p.qkv_b)
            nq, nk = self.n_heads * self.head_dim, self.n_kv * self.head_dim
            return qkv[..., :nq], qkv[..., nq:nq + nk], qkv[..., nq + nk:]
        a = p.attn
        return a.q_proj(h), a.k_proj(h), a.v_proj(h)

    def _mlp(self, p: _LayerPlan, h: torch.Tensor) -> torch.Tensor:
        if p.fused:
            g, u = F.linear(h, p.gu_w, p.gu_b).chunk(2, dim=-1)
            return p.mlp.down_proj(F.silu(g) * u)
        m = p.mlp
        return m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))

    def _attend(self, q, k_all, v_all, past: int, T: int) -> torch.Tensor:
        """q [1, H, T, D], k/v [1, KV, past+T, D] -> [1, T, H*D]."""
        if T == 1:
            mask, causal = None, False
        elif past == 0:
            mask, causal = None, True
        else:
            qpos = torch.arange(past, past + T, device=q.device).unsqueeze(1)
            kpos = torch.arange(past + T, device=q.device).unsqueeze(0)
            mask, causal = kpos <= qpos, False
        o = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, is_causal=causal,
                                           scale=self.scaling, enable_gqa=self.n_kv != self.n_heads)
        return o.transpose(1, 2).reshape(1, T, self.n_heads * self.head_dim)

    @torch.no_grad()
    def run(self, layers, start: int, end: int, hidden: torch.Tensor,
            seq: PagedSequence, past: int) -> torch.Tensor:
        """hidden [1, T, H] = rows for positions [past, past+T)."""
        return self.run_batched(layers, start, end, [hidden], [seq], [past])[0]

    @torch.no_grad()
    def run_batched(self, layers, start: int, end: int, hiddens: List[torch.Tensor],
                    seqs: List[PagedSequence], pasts: List[int]) -> List[torch.Tensor]:
        """Several requests' new rows in one pass: the rows are concatenated
        for every projection / MLP (weights read once per step for the whole
        batch) and attention runs per request against its own pages."""
        lens = [h.shape[1] for h in hiddens]
        x = torch.cat(hiddens, dim=1) if len(hiddens) > 1 else hiddens[0]
        pos = torch.cat([torch.arange(p, p + n, device=x.device) for p, n in zip(pasts, lens)]).unsqueeze(0)
        cos, sin = self.rotary(x, pos)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [1, 1, sumT, D]
        slots = [s.slots(p, n) for s, p, n in zip(seqs, pasts, lens)]
        bounds = [0]
        for n in lens:
            bounds.append(bounds[-1] + n)
        H, KV, D = self.n_heads, self.n_kv, self.head_dim
        S = x.shape[1]
        for li in range(start, end + 1):
            p = self.plan(li, layers[li])
            if p is None:
                raise RuntimeError(f"layer {li} is not eligible for the fast path")
            h = _rms(x, p.n1, p.eps1)
            q, k, v = self._qkv(p, h)
            q = q.view(1, S, H, D)
            k = k.view(1, S, KV, D)
            if p.q_norm is not None:
                q = p.q_norm(q)
                k = p.k_norm(k)
            q = _rope(q.transpose(1, 2), cos, sin)                      # [1, H, S, D]
            k = _rope(k.transpose(1, 2), cos, sin)                      # [1, KV, S, D]
            v = v.view(1, S, KV, D).transpose(1, 2)
            outs = []
            for i, (seq, past) in enumerate(zip(seqs, pasts)):
                a, b = bounds[i], bounds[i + 1]
                seq.write(li, k[0, :, a:b].transpose(0, 1), v[0, :, a:b].transpose(0, 1), slots[i])
                k_all, v_all = seq.read(li, past + (b - a))
                outs.append(self._attend(q[:, :, a:b], k_all, v_all, past, b - a))
            o = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            x = x + p.attn.o_proj(o)
            x = x + self._mlp(p, _rms(x, p.n2, p.eps2))
        for seq, past, n in zip(seqs, pasts, lens):
            seq.advance(past + n)
        return [x[:, a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    def stats(self) -> Dict[str, int]:
        return {"kv_blocks_used": self.pool.used_blocks, "kv_blocks_capacity": self.pool.capacity,
                "kv_block_size": self.pool.block}
