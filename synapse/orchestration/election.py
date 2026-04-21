import asyncio
import time
from typing import Dict, Optional, List
from loguru import logger
from synapse.routing import EventRouter, Event

class ElectionManager:
    """
    ElectionManager handles leading node selection for the Synapse cluster.
    Inspired by Exo's Master Election but simplified for Synapse's P2P mesh.
    """
    TICK_INTERVAL = 2.0
    TIMEOUT = 6.0

    def __init__(self, node_id: str, event_router: EventRouter, compute_weight: float = 1.0):
        self.node_id = node_id
        self.event_router = event_router
        self.compute_weight = compute_weight
        
        self.active_nodes: Dict[str, Dict] = {}
        self.current_master_id: Optional[str] = None
        self._running = False
        self._election_task: Optional[asyncio.Task] = None

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
        if not isinstance(data, dict): return
        
        node_id = data.get("node_id")
        if not node_id: return
        
        # Update node status in our local view
        self.active_nodes[node_id] = {
            "weight": data.get("weight", 0.0),
            "last_seen": event.timestamp,
        }

    async def _election_loop(self):
        """Periodic loop to broadcast heartbeat and evaluate leadership."""
        while self._running:
            try:
                # 1. Broadcast our existence
                await self.event_router.publish("synapse/cluster/election", {
                    "node_id": self.node_id,
                    "weight": self.compute_weight
                })

                # 2. Cleanup timed out nodes
                now = asyncio.get_event_loop().time()
                self.active_nodes = {
                    nid: info for nid, info in self.active_nodes.items()
                    if now - info["last_seen"] < self.TIMEOUT
                }

                # 3. Evaluate who should be the Master
                # Add ourselves to the pool for evaluation
                all_candidates = {**self.active_nodes, self.node_id: {"weight": self.compute_weight}}
                
                # Selection Logic: 
                # Highest weight first. Tie-break with lexicographical Node ID (smallest ID wins).
                sorted_candidates = sorted(
                    all_candidates.keys(),
                    key=lambda x: (-all_candidates[x]["weight"], x)
                )
                
                new_master_id = sorted_candidates[0] if sorted_candidates else self.node_id
                
                if new_master_id != self.current_master_id:
                    self.current_master_id = new_master_id
                    if not self.is_master():
                        logger.info(f"Node {self.node_id} recognizes {self.current_master_id} as Master")

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
