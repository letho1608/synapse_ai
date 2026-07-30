import asyncio
import time
from typing import Dict, Optional, List, Callable
from loguru import logger
from synapse.routing import EventRouter, Event


class ElectionManager:
    """
    ElectionManager handles leading node selection for the Synapse cluster.
    Inspired by Exo's Master Election but simplified for Synapse's P2P mesh.

    Uses capacity-based (multi-resource LACP score) weighting instead of
    simple VRAM+FLOPs to prevent Master flapping during AI tasks.
    Implements Master Stickiness: a Master only loses status when truly offline.
    """
    TICK_INTERVAL = 2.0
    TIMEOUT = 6.0
    MASTER_STICKINESS_GRACE_PERIOD = 30.0

    def __init__(self, node_id: str, event_router: EventRouter, compute_weight: float = 1.0,
                 dynamic_weight: bool = False, resource_scorer: Optional[Callable] = None):
        self.node_id = node_id
        self.event_router = event_router
        self.base_compute_weight = compute_weight
        self.compute_weight = compute_weight
        self.dynamic_weight = dynamic_weight
        self._resource_scorer = resource_scorer  # Callable: fn(node_id) -> composite_score
        self._monitor = None

        self.active_nodes: Dict[str, Dict] = {}
        self.current_master_id: Optional[str] = None
        self._running = False
        self._election_task: Optional[asyncio.Task] = None
        self._master_became_offline_at: Optional[float] = None

    def set_monitor(self, monitor) -> None:
        """Wire a RealtimeMonitor (not used for weight anymore, kept for compatibility)."""
        self._monitor = monitor

    def set_base_compute_weight(self, vram_gb: float, fp32_flops: float) -> None:
        """
        Set base compute weight from hardware capacity.
        Falls back to VRAM + FLOPs if no multi-resource scorer is available.
        When a resource_scorer is available, it takes precedence.
        """
        simple_weight = vram_gb + (fp32_flops / 100.0)
        if self._resource_scorer is not None:
            try:
                multi_weight = self._resource_scorer(self.node_id)
                self.base_compute_weight = max(simple_weight, multi_weight * 1000.0)
                self.compute_weight = self.base_compute_weight
                return
            except Exception:
                pass
        self.base_compute_weight = simple_weight
        self.compute_weight = self.base_compute_weight

    async def start(self):
        """Start the election background task."""
        self._running = True
        self.event_router.subscribe("synapse/cluster/election", self._handle_election_event)
        self._election_task = asyncio.create_task(self._election_loop())

    async def stop(self):
        """Stop the election background task."""
        self._running = False
        if self._election_task:
            self._election_task.cancel()
        self.event_router.unsubscribe("synapse/cluster/election", self._handle_election_event)
        logger.info("Election Manager stopped.")

    async def _handle_election_event(self, event: Event):
        """Processes incoming election heartbeat from other nodes."""
        data = event.data
        if not isinstance(data, dict):
            return

        node_id = data.get("node_id")
        if not node_id:
            return

        self.active_nodes[node_id] = {
            "weight": data.get("weight", 0.0),
            "last_seen": event.timestamp,
        }

    def _update_compute_weight(self):
        """Compute weight is now static based on hardware capacity (VRAM + FLOPs).
        This prevents Master flapping when AI tasks cause temporary load spikes.
        """
        if self.dynamic_weight and self._monitor is not None:
            pass
        self.compute_weight = self.base_compute_weight

    async def _election_loop(self):
        """Periodic loop to broadcast heartbeat and evaluate leadership."""
        while self._running:
            try:
                self._update_compute_weight()

                await self.event_router.publish("synapse/cluster/election", {
                    "node_id": self.node_id,
                    "weight": self.compute_weight
                })

                now = asyncio.get_event_loop().time()
                self.active_nodes = {
                    nid: info for nid, info in self.active_nodes.items()
                    if now - info["last_seen"] < self.TIMEOUT
                }

                all_candidates = {**self.active_nodes, self.node_id: {"weight": self.compute_weight, "last_seen": now}}

                sorted_candidates = sorted(
                    all_candidates.keys(),
                    key=lambda x: (-all_candidates[x]["weight"], x)
                )

                new_master_id = sorted_candidates[0] if sorted_candidates else self.node_id

                if self.current_master_id is None:
                    self.current_master_id = new_master_id
                    if not self.is_master():
                        logger.info(f"Node {self.node_id} recognizes {self.current_master_id} as Master")
                elif self.current_master_id == self.node_id:
                    if new_master_id != self.node_id:
                        logger.info(f"Node {self.node_id} is still Master (stickiness active)")
                    self.current_master_id = self.node_id
                elif self.current_master_id in self.active_nodes:
                    self._master_became_offline_at = None
                    if new_master_id != self.current_master_id:
                        logger.info(f"Node {self.node_id} recognizes {self.current_master_id} as Master (sticky)")
                    self.current_master_id = new_master_id
                else:
                    if self._master_became_offline_at is None:
                        self._master_became_offline_at = now
                    elif now - self._master_became_offline_at > self.MASTER_STICKINESS_GRACE_PERIOD:
                        logger.warning(f"Master {self.current_master_id} offline for {now - self._master_became_offline_at:.0f}s, electing new Master")
                        self.current_master_id = new_master_id
                        self._master_became_offline_at = None
                    else:
                        if new_master_id != self.current_master_id:
                            logger.info(f"Previous Master {self.current_master_id} went offline, waiting {self.MASTER_STICKINESS_GRACE_PERIOD - (now - self._master_became_offline_at):.0f}s before accepting new Master")

                await asyncio.sleep(self.TICK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in election loop: {e}")
                await asyncio.sleep(self.TICK_INTERVAL)

    def is_master(self) -> bool:
        """Return True if this node is currently the elected Master."""
        return self.current_master_id == self.node_id

    def get_cluster_nodes(self) -> List[str]:
        """Returns a list of all currently active node IDs in the cluster."""
        return list(self.active_nodes.keys()) + [self.node_id]
