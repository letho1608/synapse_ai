"""
Behavioral tests: paged KV pool + session KV-length tracking (T3.1 / C3).

The pool is pure index management (no torch, no attention): append extents,
truncate/rollback exactness, exhaustion atomicity, eviction, isolation,
budget math. Session tests cover commit monotonicity + truncate protocol.
The GPU-gated attention integration is specified in kv_pool.py, not here.
"""
import pytest

from synapse.inference.distributed_engine import DistributedInferenceSession
from synapse.inference.kv_pool import PagedKVPool, PoolExhausted


def test_append_extents_cover_range_exactly():
    pool = PagedKVPool(num_pages=8, page_size=16)
    ext = pool.append("s", 20)
    assert pool.seq_length("s") == 20
    assert pool.used_pages == 2
    # Extents chain contiguously in SEQUENCE space (page ids are slot ids,
    # not positions): lengths tile [0, 20) with no gaps/overlaps.
    assert sum(ln for _, _, ln in ext) == 20
    pos, seen = 0, set()
    for page, off, ln in ext:
        assert 0 <= off and off + ln <= 16
        for i in range(ln):
            assert (page, off + i) not in seen
            seen.add((page, off + i))
        pos += ln
    assert pos == 20
    ext2 = pool.append("s", 12)
    assert pool.seq_length("s") == 32
    assert pool.used_pages == 2
    assert sum(ln for _, _, ln in ext2) == 12


def test_truncate_rollback_releases_pages():
    pool = PagedKVPool(num_pages=8, page_size=16)
    pool.append("s", 40)  # 3 pages (16+16+8)
    assert pool.used_pages == 3
    assert pool.truncate("s", 16) == 2
    assert pool.seq_length("s") == 16
    assert pool.used_pages == 1
    assert pool.truncate("s", 16) == 0  # idempotent
    assert pool.truncate("missing", 0) == 0
    assert pool.truncate("s", 0) == 1
    assert pool.seq_length("s") == 0
    assert pool.used_pages == 0


def test_exhaustion_is_atomic_and_counted():
    pool = PagedKVPool(num_pages=2, page_size=4)
    pool.append("a", 8)  # fills the pool
    with pytest.raises(PoolExhausted):
        pool.append("b", 1)
    assert pool.seq_length("b") == 0
    assert pool.used_pages == 2  # failed call leaked nothing
    assert pool.free("a") == 2
    pool.append("b", 1)
    assert pool.seq_length("b") == 1


def test_sequences_isolated_and_eviction():
    pool = PagedKVPool(num_pages=4, page_size=4)
    pool.append("old", 8)
    pool.touch("old", ts=1.0)
    pool.append("new", 8)
    pool.touch("new", ts=2.0)
    assert pool.evict_oldest(exclude="new") == "old"
    assert pool.evictions == 1
    assert pool.seq_length("old") == 0
    assert pool.seq_length("new") == 8
    assert pool.free("ghost") == 0
    assert pool.evict_oldest() == "new"
    assert pool.evict_oldest() is None


def test_budget_math_and_snapshot():
    pool = PagedKVPool(num_pages=10, page_size=16, bytes_per_token=1024)
    pool.append("s", 32)
    assert pool.memory_bytes() == 32 * 1024
    snap = pool.snapshot()
    assert snap["used_pages"] == 2 and snap["free_pages"] == 8
    assert snap["sequences"] == {"s": 32}
    with pytest.raises(ValueError):
        PagedKVPool(num_pages=0, page_size=16)
    with pytest.raises(ValueError):
        pool.append("s", 0)


def test_session_commit_monotonic_and_truncate():
    sess = DistributedInferenceSession("r", "o", "m", num_stages=3)
    assert sess.kv_cached_len == {}
    sess.commit_stage_lengths({0: 100, 1: 100, 2: 100})
    sess.commit_stage_lengths({0: 90, 1: 104})  # 90 ignored (regress), 104 kept
    assert sess.kv_cached_len == {0: 100, 1: 104, 2: 100}
    dropped = sess.truncate_stages(100)
    assert dropped == {1: 4}
    assert sess.kv_cached_len == {0: 100, 1: 100, 2: 100}
    assert sess.truncate_stages(200) == {}
