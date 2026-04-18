"""
Advanced AllReduce Strategies for Distributed Training
Includes Tree-AllReduce and Mesh-AllReduce implementations for fault tolerance
"""

import asyncio
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class AllReduceMetrics:
    """Metrics for AllReduce operation"""
    strategy: str  # "ring", "tree", "mesh"
    start_time: float
    end_time: Optional[float] = None
    num_nodes: int = 0
    total_size_mb: float = 0.0
    steps: int = 0
    failed_nodes: List[int] = None
    success: bool = False
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class TreeAllReduceStrategy:
    """
    Tree-based all-reduce algorithm: O(log N) complexity
    
    Topology: Binary tree where:
    - Leaf nodes send to parent
    - Parents aggregate and pass up
    - Root broadcasts back down
    
    Advantages:
    - O(log N) steps (vs O(2N-2) for ring)
    - Better for latency-critical scenarios
    - Parallelizes better across network
    
    Disadvantages:
    - More complex topology
    - Root becomes bottleneck if bandwidth-limited
    
    Example with 8 nodes:
              0 (root)
            /   \
           1     2
          / \   / \
         3   4 5   6
        /
       7
    """
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.height = math.ceil(math.log2(num_nodes)) if num_nodes > 1 else 1
    
    def get_parent(self, rank: int) -> Optional[int]:
        """Get parent node in binary tree"""
        if rank == 0:
            return None
        return (rank - 1) // 2
    
    def get_children(self, rank: int) -> List[int]:
        """Get child nodes in binary tree"""
        children = []
        left = 2 * rank + 1
        right = 2 * rank + 2
        
        if left < self.num_nodes:
            children.append(left)
        if right < self.num_nodes:
            children.append(right)
        
        return children
    
    def get_ancestors(self, rank: int) -> List[int]:
        """Get all ancestors from rank to root"""
        ancestors = []
        current = rank
        while current != 0:
            current = self.get_parent(current)
            if current is not None:
                ancestors.append(current)
        return ancestors
    
    def get_descendants(self, rank: int) -> List[int]:
        """Get all descendants of rank"""
        descendants = []
        queue = [rank]
        
        while queue:
            current = queue.pop(0)
            children = self.get_children(current)
            for child in children:
                descendants.append(child)
                queue.append(child)
        
        return descendants


class MeshAllReduceStrategy:
    """
    Fully connected mesh all-reduce: Fault-tolerant but slower
    
    Topology: Every node communicates with every other node
    
    Advantages:
    - Can tolerate multiple node failures
    - More robust to network issues
    - Each node has full gradient information
    
    Disadvantages:
    - O(N) communication steps
    - Higher bandwidth requirements
    - Slower than tree or ring
    
    Algorithm:
    1. Each node sends gradient to all others (N-1 sends)
    2. Each node receives from all others (N-1 receives)
    3. Average received gradients
    """
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
    
    def get_peer_ranks(self, rank: int) -> List[int]:
        """Get all other node ranks to communicate with"""
        return [r for r in range(self.num_nodes) if r != rank]
    
    def get_broadcast_order(self, rank: int, seed: int = 0) -> List[int]:
        """Get optimal broadcast order to balance network"""
        peers = self.get_peer_ranks(rank)
        # Shuffle to distribute traffic
        np.random.seed(seed + rank)
        np.random.shuffle(peers)
        return peers


class AdaptiveAllReduceManager:
    """
    Manages all-reduce with automatic strategy selection and fallback
    
    Selection criteria:
    - Latency profile: N and average RTT
    - If all RTTs < 10ms: Use Tree (N=2,4,8) or Ring (N=16+)
    - If RTTs vary widely: Use Mesh (robust)
    - If network unreliable: Use Mesh with retries
    """
    
    def __init__(self, num_nodes: int, rank: int):
        self.num_nodes = num_nodes
        self.rank = rank
        
        self.tree_strategy = TreeAllReduceStrategy(num_nodes)
        self.mesh_strategy = MeshAllReduceStrategy(num_nodes)
        
        # Metrics
        self.metrics_history: List[AllReduceMetrics] = []
        self.strategy_success_rates: Dict[str, Tuple[int, int]] = {
            "tree": (0, 0),  # (successes, attempts)
            "ring": (0, 0),
            "mesh": (0, 0),
        }
    
    def select_strategy(
        self,
        num_nodes: int,
        estimated_latency_ms: float = 50.0,
        network_reliability: float = 0.95
    ) -> str:
        """
        Select best all-reduce strategy based on conditions.
        
        Args:
            num_nodes: Number of nodes in cluster
            estimated_latency_ms: Average round-trip latency
            network_reliability: Probability of message delivery (0-1)
        
        Returns:
            Strategy name: "tree", "ring", or "mesh"
        """
        
        # Calculate theoretical times
        tree_time = estimated_latency_ms * math.ceil(math.log2(num_nodes))
        ring_time = estimated_latency_ms * 2 * (num_nodes - 1)
        mesh_time = estimated_latency_ms * (num_nodes - 1)
        
        # Tree-AllReduce: Best for small clusters with low latency
        if num_nodes <= 8 and estimated_latency_ms < 20:
            return "tree"
        
        # Ring-AllReduce: Good for medium clusters with good reliability
        if network_reliability > 0.99 and num_nodes > 8:
            return "ring"
        
        # Mesh-AllReduce: When reliability is questionable or very small cluster
        if network_reliability < 0.95 or num_nodes <= 4:
            return "mesh"
        
        # Default: Most efficient for cluster size
        min_time = min(tree_time, ring_time, mesh_time)
        if min_time == tree_time:
            return "tree"
        elif min_time == ring_time:
            return "ring"
        else:
            return "mesh"
    
    async def execute_adaptive_allreduce(
        self,
        gradient_data: np.ndarray,
        peer_send_callback,
        peer_recv_callback,
        timeout: float = 60.0
    ) -> Tuple[bool, np.ndarray, str]:
        """
        Execute all-reduce with automatic strategy selection and fallback.
        
        Args:
            gradient_data: 1D array of gradients to synchronize
            peer_send_callback: async func(peer_rank, data) to send data
            peer_recv_callback: async func(peer_rank) to receive data
            timeout: Maximum time to wait for completion
        
        Returns:
            (success, synchronized_data, strategy_used)
        """
        
        # Try Tree first (fastest if network is good)
        try:
            result = await asyncio.wait_for(
                self.execute_tree_allreduce(
                    gradient_data,
                    peer_send_callback,
                    peer_recv_callback
                ),
                timeout=timeout * 0.3  # Give tree 30% of total time
            )
            if result is not None:
                self._record_metric("tree", True, gradient_data.nbytes / 1e6)
                return (True, result, "tree")
        except (asyncio.TimeoutError, Exception) as e:
            if isinstance(e, asyncio.TimeoutError):
                print("⏱ Tree-AllReduce timeout, trying Ring...")
            else:
                print(f"⚠ Tree-AllReduce failed: {e}, trying Ring...")
        
        # Try Ring (more reliable)
        try:
            result = await asyncio.wait_for(
                self.execute_ring_allreduce(
                    gradient_data,
                    peer_send_callback,
                    peer_recv_callback
                ),
                timeout=timeout * 0.4  # Give ring 40% of total time
            )
            if result is not None:
                self._record_metric("ring", True, gradient_data.nbytes / 1e6)
                return (True, result, "ring")
        except (asyncio.TimeoutError, Exception) as e:
            if isinstance(e, asyncio.TimeoutError):
                print("⏱ Ring-AllReduce timeout, falling back to Mesh...")
            else:
                print(f"⚠ Ring-AllReduce failed: {e}, falling back to Mesh...")
        
        # Finally try Mesh (most robust)
        try:
            result = await asyncio.wait_for(
                self.execute_mesh_allreduce(
                    gradient_data,
                    peer_send_callback,
                    peer_recv_callback
                ),
                timeout=timeout * 0.3  # Give mesh remaining time
            )
            if result is not None:
                self._record_metric("mesh", True, gradient_data.nbytes / 1e6)
                return (True, result, "mesh")
        except (asyncio.TimeoutError, Exception) as e:
            print(f"❌ All AllReduce strategies failed: {e}")
            self._record_metric("mesh", False, gradient_data.nbytes / 1e6)
        
        return (False, gradient_data, "none")
    
    async def execute_tree_allreduce(
        self,
        gradient_data: np.ndarray,
        peer_send_callback,
        peer_recv_callback
    ) -> Optional[np.ndarray]:
        """Execute tree-based all-reduce"""
        
        accumulated_data = gradient_data.copy()
        children = self.tree_strategy.get_children(self.rank)
        parent = self.tree_strategy.get_parent(self.rank)
        
        # Phase 1: Reduce (bottom-up)
        for child_rank in children:
            try:
                child_data = await peer_recv_callback(child_rank)
                if child_data is not None:
                    accumulated_data += child_data
            except Exception as e:
                print(f"⚠ Failed to receive from child {child_rank}: {e}")
                raise
        
        # Phase 2: Broadcast (top-down)
        if parent is not None:
            # Send to parent
            try:
                await peer_send_callback(parent, accumulated_data)
            except Exception as e:
                print(f"⚠ Failed to send to parent {parent}: {e}")
                raise
            
            # Receive from parent
            try:
                accumulated_data = await peer_recv_callback(parent)
            except Exception as e:
                print(f"⚠ Failed to receive from parent {parent}: {e}")
                raise
        
        # Broadcast down to children
        for child_rank in children:
            try:
                await peer_send_callback(child_rank, accumulated_data)
            except Exception as e:
                print(f"⚠ Failed to send to child {child_rank}: {e}")
        
        return accumulated_data
    
    async def execute_ring_allreduce(
        self,
        gradient_data: np.ndarray,
        peer_send_callback,
        peer_recv_callback
    ) -> Optional[np.ndarray]:
        """Execute ring-based all-reduce (standard algorithm)"""
        
        # Successor rank in ring
        successor_rank = (self.rank + 1) % self.num_nodes
        predecessor_rank = (self.rank - 1) % self.num_nodes
        
        chunks = np.array_split(gradient_data, self.num_nodes)
        
        # Scatter-Reduce phase
        for step in range(self.num_nodes - 1):
            send_idx = (self.rank - step) % self.num_nodes
            recv_idx = (self.rank - step - 1) % self.num_nodes
            
            try:
                await peer_send_callback(successor_rank, chunks[send_idx])
                chunks[recv_idx] = await peer_recv_callback(predecessor_rank)
                chunks[recv_idx] += chunks[recv_idx]  # Reduce
            except Exception as e:
                print(f"⚠ Ring scatter-reduce step {step} failed: {e}")
                raise
        
        # All-Gather phase
        for step in range(self.num_nodes - 1):
            send_idx = (self.rank - step + 1) % self.num_nodes
            recv_idx = (self.rank - step) % self.num_nodes
            
            try:
                await peer_send_callback(successor_rank, chunks[send_idx])
                chunks[recv_idx] = await peer_recv_callback(predecessor_rank)
            except Exception as e:
                print(f"⚠ Ring all-gather step {step} failed: {e}")
                raise
        
        return np.concatenate(chunks)
    
    async def execute_mesh_allreduce(
        self,
        gradient_data: np.ndarray,
        peer_send_callback,
        peer_recv_callback
    ) -> Optional[np.ndarray]:
        """Execute mesh-based all-reduce (all-to-all)"""
        
        peer_ranks = self.mesh_strategy.get_peer_ranks(self.rank)
        accumulated_data = gradient_data.copy()
        
        # Send to all peers and collect responses
        send_tasks = [
            asyncio.create_task(peer_send_callback(rank, gradient_data))
            for rank in peer_ranks
        ]
        
        recv_tasks = [
            asyncio.create_task(peer_recv_callback(rank))
            for rank in peer_ranks
        ]
        
        try:
            # Send all (fire and forget)
            await asyncio.gather(*send_tasks)
            
            # Receive all
            results = await asyncio.gather(*recv_tasks)
            
            # Average all received data
            all_data = [gradient_data] + [r for r in results if r is not None]
            averaged_data = np.mean(all_data, axis=0)
            
            return averaged_data
        
        except Exception as e:
            print(f"⚠ Mesh all-reduce failed: {e}")
            raise
    
    def _record_metric(self, strategy: str, success: bool, size_mb: float):
        """Record metrics for this all-reduce operation"""
        metric = AllReduceMetrics(
            strategy=strategy,
            start_time=time.time(),
            num_nodes=self.num_nodes,
            total_size_mb=size_mb,
            success=success
        )
        metric.end_time = metric.start_time  # Simplified
        
        self.metrics_history.append(metric)
        
        # Update success rate
        successes, attempts = self.strategy_success_rates[strategy]
        self.strategy_success_rates[strategy] = (
            successes + (1 if success else 0),
            attempts + 1
        )
    
    def get_success_rate(self, strategy: str) -> float:
        """Get success rate for a strategy"""
        successes, attempts = self.strategy_success_rates.get(strategy, (0, 0))
        return (successes / attempts) if attempts > 0 else 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all-reduce metrics"""
        return {
            "strategies": {
                strategy: {
                    "success_rate": self.get_success_rate(strategy),
                    "successes": self.strategy_success_rates[strategy][0],
                    "attempts": self.strategy_success_rates[strategy][1],
                }
                for strategy in ["tree", "ring", "mesh"]
            },
            "total_operations": len(self.metrics_history),
            "avg_tree_time_ms": float(np.mean([
                m.duration_ms for m in self.metrics_history if m.strategy == "tree"
            ]) or 0),
            "avg_ring_time_ms": float(np.mean([
                m.duration_ms for m in self.metrics_history if m.strategy == "ring"
            ]) or 0),
            "avg_mesh_time_ms": float(np.mean([
                m.duration_ms for m in self.metrics_history if m.strategy == "mesh"
            ]) or 0),
        }
