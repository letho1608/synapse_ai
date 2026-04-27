from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

@dataclass
class NodeState:
    node_id: str
    is_active: bool = True
    last_seen: float = field(default_factory=time.time)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    assigned_shards: List[str] = field(default_factory=list)

@dataclass
class ClusterState:
    """
    ClusterState holds the source of truth for the entire distributed system.
    This state is synchronized across all nodes via the P2P Mesh.
    """
    master_node_id: Optional[str] = None
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    model_registries: Dict[str, Any] = field(default_factory=dict)
    active_inferences: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    updated_at: float = field(default_factory=time.time)

    def update_node(self, node_id: str, capabilities: Dict[str, Any] = None):
        """Register or update a node in the cluster state."""
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeState(node_id=node_id)
        
        node = self.nodes[node_id]
        node.last_seen = time.time()
        if capabilities:
            node.capabilities = capabilities
        
        self.version += 1
        self.updated_at = time.time()

    def remove_stale_nodes(self, timeout: float = 30.0):
        """Cleanup nodes that haven't been seen for a while."""
        now = time.time()
        stale_ids = [nid for nid, node in self.nodes.items() if now - node.last_seen > timeout]
        for nid in stale_ids:
            del self.nodes[nid]
        
        if stale_ids:
            self.version += 1
            self.updated_at = time.time()

    def to_dict(self) -> Dict:
        """Serialize state for network transmission."""
        return {
            "master_node_id": self.master_node_id,
            "version": self.version,
            "updated_at": self.updated_at,
            "nodes": {nid: vars(ns) for nid, ns in self.nodes.items()}
        }
