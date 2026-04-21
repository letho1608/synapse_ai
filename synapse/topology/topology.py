from .device_capabilities import DeviceCapabilities
from typing import Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class PeerConnection:
  from_id: str
  to_id: str
  description: Optional[str] = None

  def __hash__(self):
    # Use both from_id and to_id for uniqueness in sets
    return hash((self.from_id, self.to_id))

  def __eq__(self, other):
    if not isinstance(other, PeerConnection):
      return False
    # Compare both from_id and to_id for equality
    return self.from_id == other.from_id and self.to_id == other.to_id

class Topology:
  def __init__(self):
    self.nodes: Dict[str, DeviceCapabilities] = {}
    self.peer_graph: Dict[str, Set[PeerConnection]] = {}
    self.active_node_id: Optional[str] = None

  def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
    self.nodes[node_id] = device_capabilities

  def get_node(self, node_id: str) -> DeviceCapabilities:
    return self.nodes.get(node_id)

  def all_nodes(self):
    return self.nodes.items()

  def add_edge(self, from_id: str, to_id: str, description: Optional[str] = None):
    if from_id not in self.peer_graph:
      self.peer_graph[from_id] = set()
    conn = PeerConnection(from_id, to_id, description)
    self.peer_graph[from_id].add(conn)

  def merge(self, peer_node_id: str, other: "Topology"):
      """Merge all nodes from other topology (not just peer_node_id)."""
      # Get current specs of 'self' (the node running this code)
      # This is tricky because self.id is not stored in Topology itself, 
      # but we can look for the node marked as self (placeholder logic or passed from elsewhere).
      # For now, we perform local hardware deduplication in the Orchestration Node layer instead
      # to keep the basic Topology class generic.
      
      # Basic merge:
      for node_id, capabilities in other.nodes.items():
          self.update_node(node_id, capabilities)
      
      # Merge edges - only add edges that originate from peer_node_id
      # to avoid adding edges from nodes not belonging to this peer
      for node_id, connections in other.peer_graph.items():
          for conn in connections:
              if conn.from_id == peer_node_id:
                  self.add_edge(conn.from_id, conn.to_id, conn.description)

  def __str__(self):
    nodes_str = ", ".join(f"{node_id}: {cap}" for node_id, cap in self.nodes.items())
    edges_str = ", ".join(f"{node}: {[f'{c.to_id}({c.description})' for c in conns]}"
                         for node, conns in self.peer_graph.items())
    return f"Topology(Nodes: {{{nodes_str}}}, Edges: {{{edges_str}}})"

  def to_json(self):
    return {
      "nodes": {
        node_id: capabilities.to_dict()
        for node_id, capabilities in self.nodes.items()
      },
      "peer_graph": {
        node_id: [
          {
            "from_id": conn.from_id,
            "to_id": conn.to_id,
            "description": conn.description
          }
          for conn in connections
        ]
        for node_id, connections in self.peer_graph.items()
      },
      "active_node_id": self.active_node_id
    }
