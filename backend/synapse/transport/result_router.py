"""
LACP — Result Router

Handles in-order reassembly of results from distributed pipeline nodes.
Every tensor carries a (request_id, orchestrator_id, micro_batch_id,
source_stage, target_stage) tuple. Results are streamed back to the
orchestrator with in-order delivery — the caller sees a clean token
stream regardless of which nodes processed which micro-batches.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable


@dataclass
class RoutedResult:
    request_id: str
    orchestrator_id: str
    micro_batch_id: int
    source_stage_id: int
    tokens: List[int]
    is_finished: bool
    tensor: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ResultBuffer:
    def __init__(self):
        self._pending: Dict[int, RoutedResult] = {}
        self._next_expected: int = 0
        self._finished: bool = False
        self._creation_time: float = time.time()

    def insert(self, result: RoutedResult) -> List[RoutedResult]:
        self._pending[result.micro_batch_id] = result

        if result.is_finished:
            self._finished = True

        delivered: List[RoutedResult] = []
        while self._next_expected in self._pending:
            delivered.append(self._pending.pop(self._next_expected))
            self._next_expected += 1
        return delivered

    def is_stale(self, ttl: float = 300.0) -> bool:
        return time.time() - self._creation_time > ttl

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class ResultRouter:
    """
    In-order result reassembly for distributed pipeline inference.

    Each request_id gets its own ResultBuffer. Out-of-order results are
    held until predecessors arrive, then delivered sequentially via callbacks.
    """

    TTL = 300.0
    GC_INTERVAL = 30.0

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._buffers: Dict[str, ResultBuffer] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._on_token_callbacks: Dict[str, Callable] = {}
        self._gc_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._gc_task = asyncio.create_task(self._gc_loop())

    async def stop(self) -> None:
        self._running = False
        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass

    def register_token_callback(self, request_id: str, callback: Callable) -> None:
        self._on_token_callbacks[request_id] = callback

    def unregister_token_callback(self, request_id: str) -> None:
        self._on_token_callbacks.pop(request_id, None)

    def receive_result(self, result: RoutedResult) -> List[RoutedResult]:
        if result.request_id not in self._buffers:
            self._buffers[result.request_id] = ResultBuffer()

        buffer = self._buffers[result.request_id]
        delivered = buffer.insert(result)

        for dr in delivered:
            self._deliver_tokens(dr)

        if buffer._finished and buffer.pending_count == 0:
            self._finish_request(result.request_id)

        return delivered

    def _deliver_tokens(self, result: RoutedResult) -> None:
        if not result.tokens:
            return

        cb = self._on_token_callbacks.get(result.request_id)
        if cb:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(result.request_id, result.tokens, result.is_finished))
                else:
                    cb(result.request_id, result.tokens, result.is_finished)
            except Exception:
                pass

    def _finish_request(self, request_id: str) -> None:
        self._buffers.pop(request_id, None)


    async def _gc_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.GC_INTERVAL)
            stale = [
                rid
                for rid, buf in self._buffers.items()
                if buf.is_stale(self.TTL)
            ]
            for rid in stale:
                self._finish_request(rid)