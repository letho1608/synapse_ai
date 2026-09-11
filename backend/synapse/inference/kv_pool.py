"""
Per-stage paged KV pool (T3.1 foundation / C3).

Pure index management — no torch, no attention wiring — so it is fully
testable on CPU. Each pipeline stage will own one pool for its layer range;
stages NEVER exchange KV bytes, only lengths (cached_len) plus delta tokens,
exactly per the Lane-3 design. The attention-prologue integration (reading /
writing real K/V pages inside _forward_transformer_range) is GPU-gated work
with an exact insertion spec at the bottom of this file; until it lands, this
module changes no hot-path behavior (nothing imports it yet except tests).

Design:
- num_pages x page_size token-slots, shared across sequences on this stage.
- append(seq, n) extends a sequence, allocating fresh pages as needed.
- truncate(seq, length) drops pages past a divergence point (speculation
  reject path). free(seq) releases everything.
- bytes(token) is caller-supplied (2*L*d*h for the stage's layer range) so
  the same pool does budget math without knowing model internals.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple


class PoolExhausted(Exception):
    """No free pages left (caller must evict or refuse, never OOM silently)."""


class PagedKVPool:
    """Fixed pool of token-slot pages for ONE pipeline stage."""

    def __init__(self, num_pages: int, page_size: int = 16,
                 bytes_per_token: int = 0):
        if num_pages <= 0 or page_size <= 0:
            raise ValueError("num_pages and page_size must be positive")
        self.num_pages = num_pages
        self.page_size = page_size
        self.bytes_per_token = bytes_per_token
        self._free: List[int] = list(range(num_pages))
        # seq_id -> {"pages": [page_ids in order], "length": tokens}
        self._seqs: Dict[str, Dict] = {}
        self.evictions = 0

    @property
    def used_pages(self) -> int:
        return self.num_pages - len(self._free)

    @property
    def sequences(self) -> List[str]:
        return sorted(self._seqs)

    def seq_length(self, seq_id: str) -> int:
        return self._seqs.get(seq_id, {}).get("length", 0)

    def append(self, seq_id: str, n_tokens: int) -> List[Tuple[int, int, int]]:
        """Extend seq by n tokens. Returns [(page, offset, len)] extents.

        Raises PoolExhausted instead of over-committing.
        """
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        st = self._seqs.get(seq_id)
        if st is None:
            st = {"pages": [], "length": 0}
            self._seqs[seq_id] = st
        need_total = st["length"] + n_tokens
        need_pages = (need_total + self.page_size - 1) // self.page_size
        pages_before = len(st["pages"])
        while len(st["pages"]) < need_pages:
            if not self._free:
                # Atomicity: return this call's pages to the free list.
                self._free.extend(st["pages"][pages_before:])
                del st["pages"][pages_before:]
                raise PoolExhausted(
                    "need %d pages, %d free" % (need_pages, len(self._free)))
            st["pages"].append(self._free.pop())
        # Emit extents for the newly added range only.
        extents: List[Tuple[int, int, int]] = []
        start = st["length"]
        remaining = n_tokens
        while remaining > 0:
            page_idx = start // self.page_size
            off = start % self.page_size
            take = min(remaining, self.page_size - off)
            extents.append((st["pages"][page_idx], off, take))
            start += take
            remaining -= take
        st["length"] = need_total
        return extents

    def truncate(self, seq_id: str, length: int) -> int:
        """Drop pages past `length` (speculation-rollback path).

        Returns the number of pages released. Truncating to the current
        length is a no-op; truncating an unknown seq is a no-op.
        """
        st = self._seqs.get(seq_id)
        if st is None or length >= st["length"]:
            return 0
        keep_pages = (length + self.page_size - 1) // self.page_size if length else 0
        released = st["pages"][keep_pages:]
        st["pages"] = st["pages"][:keep_pages]
        st["length"] = length
        self._free.extend(released)
        return len(released)

    def free(self, seq_id: str) -> int:
        """Release a whole sequence. Returns pages released."""
        st = self._seqs.pop(seq_id, None)
        if st is None:
            return 0
        self._free.extend(st["pages"])
        return len(st["pages"])

    def evict_oldest(self, exclude: Optional[str] = None) -> Optional[str]:
        """Evict one sequence (LRU by last touch is tracked via _ts)."""
        cands = [(s, v.get("_ts", 0.0)) for s, v in self._seqs.items()
                 if s != exclude]
        if not cands:
            return None
        victim = min(cands, key=lambda kv: kv[1])[0]
        self.free(victim)
        self.evictions += 1
        return victim

    def touch(self, seq_id: str, ts: Optional[float] = None) -> None:
        if seq_id in self._seqs:
            self._seqs[seq_id]["_ts"] = ts if ts is not None else time.time()

    def memory_bytes(self) -> int:
        """Committed bytes under the caller's bytes_per_token model."""
        total_tokens = sum(v["length"] for v in self._seqs.values())
        return total_tokens * self.bytes_per_token

    def snapshot(self) -> Dict:
        return {"num_pages": self.num_pages, "page_size": self.page_size,
                "used_pages": self.used_pages, "free_pages": len(self._free),
                "sequences": {s: v["length"] for s, v in self._seqs.items()},
                "evictions": self.evictions,
                "memory_bytes": self.memory_bytes()}


# ── GPU-gated attention integration (exact insertion spec, NOT done) ──
# File: backend/synapse/inference/pytorch_hf_engine.py::_forward_transformer_range
# 1. Engine owns one PagedKVPool per loaded shard (num_pages from the elastic
#    budget: (free_vram - weights - headroom) / bytes_per_token, where
#    bytes_per_token = 2 * layers * n_kv_heads * head_dim * dtype_size).
# 2. On infer_tensor with past_len>0: pool.append(request_id, delta) gives
#    page extents; the attention prologue attends over (cached prefix pages
#    + delta) instead of re-running the full prefix; past_key_values route
#    through pool pages instead of the _content_kv_cache dict.
# 3. On verify_speculative_tokens reject at position i: pool.truncate(
#    request_id, P + t + i) on every stage + broadcast truncate_to (refines
#    the synapse/cluster/cache/clear channel).
# 4. Gates (must all pass on GPU before enabling): >=4x tok/s @2k ctx,
#    >=8x @8k, 200-prompt greedy parity 100%, reject-injection test.
# Until then this module is foundation only: zero hot-path impact.
