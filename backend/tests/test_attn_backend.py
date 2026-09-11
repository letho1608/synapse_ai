"""
Behavioral tests: attention-backend capability probe (T3.4 / C6).

- Selection order with injected checkers (no GPU needed).
- Raising checkers fail safe (False + reason, never propagate).
- Live probe on THIS box: returns a valid name, never claims fa2/triton
  without CUDA (the anti-overclaim gate).
"""
from synapse.inference.attn_backend import (
    BACKENDS,
    available_backends,
    probe_attn_backend,
)


def _ok(_why="ok"):
    return lambda: (True, _why)


def _no(why="missing"):
    return lambda: (False, why)


def test_order_prefers_fa2_then_triton_then_sdpa():
    r = probe_attn_backend(checkers={"fa2": _ok(), "triton": _ok(),
                                     "sdpa": _ok(), "eager": _ok()})
    assert r["name"] == "fa2"
    r = probe_attn_backend(checkers={"fa2": _no(), "triton": _ok(),
                                     "sdpa": _ok(), "eager": _ok()})
    assert r["name"] == "triton"
    r = probe_attn_backend(checkers={"fa2": _no(), "triton": _no(),
                                     "sdpa": _ok(), "eager": _ok()})
    assert r["name"] == "sdpa"
    assert available_backends(checkers={"fa2": _no(), "triton": _no(),
                                        "sdpa": _ok(), "eager": _ok()}) == ["sdpa", "eager"]


def test_all_missing_falls_to_none_not_eager():
    r = probe_attn_backend(checkers={"fa2": _no(), "triton": _no(),
                                     "sdpa": _no(), "eager": _no()})
    assert r["name"] is None
    assert r["available"] == {b: False for b in BACKENDS}


def test_raising_checker_fails_safe():
    def _boom():
        raise RuntimeError("import machinery on fire")
    r = probe_attn_backend(checkers={"fa2": _boom, "triton": _no(),
                                     "sdpa": _ok(), "eager": _ok()})
    assert r["name"] == "sdpa"
    assert "raised" in r["reasons"]["fa2"]


def test_live_probe_never_overclaims():
    r = probe_attn_backend()
    assert r["name"] in list(BACKENDS) + [None]
    import torch
    if not torch.cuda.is_available():
        assert r["name"] not in ("fa2", "triton"), (
            "must never claim GPU backends without CUDA: %r" % (r,))
        assert r["available"].get("fa2") is False
        assert r["available"].get("triton") is False
