"""
LACP — Result Router

Handles in-order reassembly of results from distributed pipeline nodes.
Every tensor carries a (request_id, orchestrator_id, micro_batch_id,
source_stage, target_stage) tuple. Results are streamed back to the
orchestrator with in-order delivery — the caller sees a clean token
stream regardless of which nodes processed which micro-batches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any


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
