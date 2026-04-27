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
      # Merge all nodes from other topology (not just peer_node_id)
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

  def to_dict(self):
    return {
      "nodes": {
        node_id: {
          **capabilities.to_dict(),
          "status": "master" if node_id == self.active_node_id else "worker"
        }
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

  def to_json(self):
    return self.to_dict()

  @classmethod
  def from_dict(cls, data: dict) -> "Topology":
    topology = cls()
    if not data: return topology
    
    nodes_data = data.get("nodes", {})
    for node_id, cap_data in nodes_data.items():
        topology.update_node(node_id, DeviceCapabilities.from_dict(cap_data))
    
    peer_graph_data = data.get("peer_graph", {})
    for from_id, conns in peer_graph_data.items():
        for conn_data in conns:
            topology.add_edge(
                conn_data["from_id"], 
                conn_data["to_id"], 
                conn_data.get("description")
            )
            
    topology.active_node_id = data.get("active_node_id")
    return topology
