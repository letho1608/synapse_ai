"""
Transport auto-route decision (T1.2 / B3-B4).

Pure decision logic — no sockets, no node imports — so it is unit-testable
and the node layer stays a thin caller. Order (single function, first hit wins):

  1. target is self            -> in_process (defensive; forward_tensor
                                   short-circuits this before calling)
  2. same-machine + small      -> shm (zero-copy; SHM miss falls through,
                                   never silent loss)
  3. tcp peer + chunking on    -> tcp-chunked (any blob size; 1 MiB frames)
  4. otherwise                 -> bridge (legacy JSON event mesh, 256 MiB cap)

B4: wire quantization is exact-dtype by default. should_quantize() returns
True only for floating tensors with explicit opt-in (env
SYNAPSE_TRANSPORT_INT8=1 or a negotiated offer flag).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

ROUTE_IN_PROCESS = "in_process"
ROUTE_SHM = "shm"
ROUTE_TCP_CHUNKED = "tcp-chunked"
ROUTE_BRIDGE = "bridge"

# Must stay in sync with transport/shm_transport.py::MAX_SHM_BYTES.
MAX_SHM_BYTES = 512 * 1024 * 1024


def should_quantize(dtype_name: str, opt_in: Optional[bool] = None) -> bool:
    """Exact-dtype default (B4): True only for float tensors with opt-in."""
    if opt_in is None:
        opt_in = os.getenv("SYNAPSE_TRANSPORT_INT8", "0") == "1"
    if not opt_in:
        return False
    return dtype_name in ("float32", "float64", "float16")


def chunking_enabled(env: Optional[dict] = None) -> bool:
    """Chunked-TCP fast path. Default ON: single-node traffic never qualifies
    (no TCP peers registered), and any fast-path failure falls back to the
    bridge automatically. SYNAPSE_CHUNKED_TCP=0 restores bridge-only."""
    get = (env or os.environ).get
    return get("SYNAPSE_CHUNKED_TCP", "1") == "1"


def choose_transport(
    self_id: str,
    target_id: str,
    target_addr: Optional[str] = None,
    blob_bytes: int = 0,
    shm_registered: bool = False,
    tcp_registered: bool = False,
    chunked: Optional[bool] = None,
) -> Tuple[str, str]:
    """Return (route, reason). Never raises; unknown inputs -> bridge."""
    try:
        if target_id == self_id:
            return ROUTE_IN_PROCESS, "target-is-self"
        if (shm_registered and target_addr
                and 0 < blob_bytes <= MAX_SHM_BYTES):
            return ROUTE_SHM, "same-machine-zero-copy"
        use_chunked = chunked if chunked is not None else chunking_enabled()
        if tcp_registered and target_addr and use_chunked:
            return ROUTE_TCP_CHUNKED, "tcp-chunked-1mib-frames"
        if not target_addr:
            return ROUTE_BRIDGE, "no-target-addr"
        if not tcp_registered:
            return ROUTE_BRIDGE, "tcp-peer-unregistered"
        return ROUTE_BRIDGE, "chunking-disabled"
    except Exception:
        return ROUTE_BRIDGE, "router-error"


class TransportStats:
    """Observability counters: route -> count (B3 transport_selected)."""

    def __init__(self):
        self.counts: Dict[str, int] = {}

    def record(self, route: str) -> None:
        self.counts[route] = self.counts.get(route, 0) + 1

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counts)
