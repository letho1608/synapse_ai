import numpy as np
import json
import asyncio
import uuid
import time
import os
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set, Any
from loguru import logger
from synapse.inference.inference_engine import InferenceEngine, Shard, get_inference_engine
from synapse.topology.topology import Topology
from synapse.topology.device_capabilities import device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.topology.partitioning_strategy import Partition, map_partitions_to_shards, TFLOPSProportionalPartitioner
from synapse.topology.partitioner import compute_partition, LACPPartition, ShardAssignment
from synapse.topology.resource_weight import NodeResources
from synapse import DEBUG
from synapse.helpers import AsyncCallbackSystem
from synapse.loading import RepoProgressEvent
from synapse.inference.pytorch_hf_engine import HFShardDownloader as ShardDownloader
from synapse.routing import EventRouter, Event
from synapse.routing.libp2p_node import Libp2pNode
from synapse.orchestration.election import ElectionManager
from synapse.networking.peer_handle import PeerHandle
from synapse.shared.state import ClusterState, NodeState
from synapse.monitoring.realtime import RealtimeMonitor, NodeMetrics

try:
    from synapse.orchestration.tracing import tracer
except ImportError:
    tracer = None

class Node:
  def __init__(
    self,
    _id: str,
    event_router: EventRouter,
    libp2p_node: Libp2pNode,
    inference_engine: InferenceEngine,
    shard_downloader: ShardDownloader,
    max_generate_tokens: int = 1024,
    default_sample_temperature: float = 0.0,
    max_concurrent_requests: int = 32,
    topology_viz: Optional[Any] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.event_router = event_router
    self.libp2p_node = libp2p_node
    self.shard_downloader = shard_downloader
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    
    # Initialize Election Manager
    self.election_manager = ElectionManager(self.id, self.event_router)

    # LACP 3.0 Distributed Pipeline Extension
    self._distributed_pipeline = None

    # LACP 3.0 TCP Transport for binary tensor forwarding
    self.tcp_transport: Optional[Any] = None

    # LACP 3.0 Fault Tolerance Manager
    self.fault_tolerance = None

    # Tracks whether distributed inference is currently active across multiple nodes.
    # Set to True when distributed pipeline starts, False on fallback to single-node.
    # Used by web UI to display distributed status.
    self.distributed_active = False
    # Number of nodes in the current distributed inference cluster (0 = single-node)
    self.distributed_node_count = 0
    # Human-readable status for the web UI
    self.distributed_status = "single-node"  # "distributed" or "single-node" or "fallback"

    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self._prompt_token_ids: Dict[str, np.ndarray] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}
    
    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self.default_sample_temperature = default_sample_temperature
    self.num_micro_batches = 1
    self.tflops_proportional = True  # Use TFLOPS-proportional sharding by default (Cake concept)
    
    self.peers = []
    
    self.event_router.subscribe("synapse/cluster/cache/clear", self.on_p2p_clear_cache)
    self.event_router.subscribe("synapse/cluster/status", self.on_p2p_status)
    self.event_router.subscribe(f"synapse/inference/tensor/{self.id}", self.on_p2p_tensor)
    self.event_router.subscribe(f"synapse/inference/prompt/{self.id}", self.on_p2p_prompt)
    # CRITICAL FIX: Subscribe to BOTH request_id and node_id result topics.
    # The last-layer node publishes to result/{self.id}, but also
    # publishes to result/{orchestrator_id}. We need to listen on our own ID
    # to receive results when WE are the orchestrator.
    self.event_router.subscribe(f"synapse/inference/result/{self.id}", self.on_p2p_result)
    # CRITICAL FIX: Subscribe to authoritative topology broadcasts from the master node.
    # This ensures all nodes use the SAME topology for partition calculations.
    self.event_router.subscribe("synapse/cluster/topology", self.on_p2p_topology_broadcast)
    # CRITICAL FIX: Subscribe to authoritative partition broadcasts from master.
    self.event_router.subscribe("synapse/cluster/partition", self.on_p2p_partition_broadcast)
    self.event_router.subscribe("synapse/cluster/state", self.on_p2p_cluster_state)
    self._setup_p2p_rpc_handlers()

    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.outstanding_requests = {}
    self._background_tasks: Set[asyncio.Task] = set()
    # CRITICAL FIX: Map request_id -> orchestrator node_id for result routing
    self._request_orchestrator_map: Dict[str, str] = {}
    # CRITICAL FIX: Authoritative partition from master node for consistent shard assignment
    self._authoritative_partitions: Optional[List[Partition]] = None
    self._authoritative_base_shard: Optional[Shard] = None

    self._on_token = AsyncCallbackSystem()
    self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)
    self.max_concurrent_requests = max_concurrent_requests
    self._on_opaque_status = AsyncCallbackSystem()

    self.monitor = RealtimeMonitor()
    self.cluster_state = ClusterState(master_node_id=None)
    self.tracer = tracer

    self._current_metrics: NodeMetrics = NodeMetrics()

    # Register internal callback for token synchronization in distributed mode
    self.on_token.register("internal_token_sync").on_next(self._on_token_received)

  @property
  def distributed_pipeline(self):
    """LACP 3.0 Distributed Pipeline Extension — lazy-initialized."""
    if self._distributed_pipeline is None:
      from synapse.pipeline.node_extension import DistributedPipelineExtension
      self._distributed_pipeline = DistributedPipelineExtension(self)
    return self._distributed_pipeline

  def _lazy_init_fault_tolerance(self):
    """LACP 3.0 Fault Tolerance Manager — lazy-initialized."""
    if self.fault_tolerance is None:
      from synapse.pipeline.fault_tolerance import FaultToleranceManager
      self.fault_tolerance = FaultToleranceManager(self.id)
      # Pre-register known nodes from topology
      for nid in self.topology.nodes:
        self.fault_tolerance.register_node(nid)

  def _log_task_exception(self, task: asyncio.Task, name: str) -> None:
    self._background_tasks.discard(task)
    try:
      task.result()
    except asyncio.CancelledError:
      if DEBUG >= 2:
        print(f"Background task {name} was cancelled")
    except Exception as e:
      print(f"Background task {name} raised an exception: {e}")
      traceback.print_exc()

  def _schedule_task(self, coro, name: str = "background_task") -> asyncio.Task:
    task = asyncio.create_task(coro)
    self._background_tasks.add(task)
    task.add_done_callback(lambda t, task_name=name: self._log_task_exception(t, task_name))
    return task

  async def start(self, wait_for_peers: int = 0) -> None:
    if DEBUG >= 1: print("Starting node: Getting device capabilities...")
    self.device_capabilities = await device_capabilities()
    
    if DEBUG >= 1: print("Starting node: Starting Event Router and P2P Mesh...")
    await self.event_router.start()
    await self.libp2p_node.start()
    await self.election_manager.start()
    self.election_manager.set_monitor(self.monitor)
    self._setup_p2p_rpc_handlers()

    await self.monitor.start()
    self._schedule_task(self._metrics_update_loop(), "realtime_metrics")
    
    if hasattr(self, "discovery") and self.discovery:
        if DEBUG >= 1: print(f"Starting discovery module: {type(self.discovery).__name__}")
        await self.discovery.start()
    
    if DEBUG >= 1: print(f"P2P Node started. Waiting for {wait_for_peers} peers (simulated)...")
    if wait_for_peers > 0:
        await asyncio.sleep(2.0) # Grace period for P2P connection
    if DEBUG >= 1: print(f"Peers updated: {len(self.peers)} peers")
    
    if DEBUG >= 1: print("Starting node: Collecting topology...")
    await self.collect_topology(set())
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    if DEBUG >= 1: print("Node started successfully!")
    self._schedule_task(self.periodic_topology_collection(2.0), "periodic_topology_collection")

    # Start TCP transport for binary tensor forwarding
    try:
        from synapse.transport.tcp_transport import TensorTCPTransport
        self.tcp_transport = TensorTCPTransport(node_id=self.id)
        tcp_port = await self.tcp_transport.start()
        if DEBUG >= 1:
            print(f"[TCP] Binary tensor transport listening on port {tcp_port}")
    except Exception as e:
        if DEBUG >= 1:
            print(f"[TCP] Failed to start binary transport: {e}")
        self.tcp_transport = None

    # Start distributed pipeline extension
    if self._distributed_pipeline:
        await self._distributed_pipeline.start()

    # Start fault tolerance manager
    self._lazy_init_fault_tolerance()
    if self.fault_tolerance:
        await self.fault_tolerance.start()

    self.cluster_state.update_node(self.id, {
        "device": str(self.device_capabilities),
        "max_concurrent_requests": self.max_concurrent_requests,
        "has_inference_engine": self.inference_engine is not None,
    })
    await self._broadcast_cluster_state()
    self._schedule_task(self._cluster_state_heartbeat(), "cluster_state_heartbeat")

  async def _cluster_state_heartbeat(self) -> None:
    """Periodically update own node state and broadcast to cluster."""
    while True:
        await asyncio.sleep(5.0)
        self.cluster_state.update_node(self.id, {
            "device": str(self.device_capabilities),
            "max_concurrent_requests": self.max_concurrent_requests,
            "has_inference_engine": self.inference_engine is not None,
        })
        self.cluster_state.remove_stale_nodes(timeout=30.0)
        try:
            await self._broadcast_cluster_state()
        except Exception as e:
            if DEBUG >= 1:
                print(f"Cluster state broadcast failed: {e}")

  async def _broadcast_cluster_state(self) -> None:
    """Broadcast cluster state to all peers via P2P event router."""
    import msgpack
    state_dict = self.cluster_state.to_dict()
    payload = msgpack.packb(state_dict)
    await self.event_router.publish("synapse/cluster/state", {"data": payload})

  async def on_p2p_cluster_state(self, event: Event) -> None:
    """Receive and merge cluster state from a peer node."""
    import msgpack
    try:
        data = event.data if isinstance(event.data, bytes) else event.data.get("data", b"")
        peer_state = msgpack.unpackb(data)
        peer_version = peer_state.get("version", 0)
        if peer_version > self.cluster_state.version:
            for nid, nstate in peer_state.get("nodes", {}).items():
                self.cluster_state.update_node(nid, nstate.get("capabilities", {}))
            self.cluster_state.version = peer_version
            self.cluster_state.updated_at = peer_state.get("updated_at", time.time())
    except Exception as e:
        if DEBUG >= 1:
            print(f"Failed to merge peer cluster state: {e}")

  async def stop(self) -> None:
    if DEBUG >= 1: print("Stopping node: Cleaning up components...")

    # Stop fault tolerance manager
    if self.fault_tolerance:
        await self.fault_tolerance.stop()

    # Stop distributed pipeline extension
    if self._distributed_pipeline:
        await self._distributed_pipeline.stop()

    # Cancel background tasks
    for task in self._background_tasks:
        task.cancel()
    if self._background_tasks:
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    # Stop election manager
    if hasattr(self, "election_manager") and self.election_manager:
        await self.election_manager.stop()

    await self.monitor.stop()
        
    # Stop discovery
    if hasattr(self, "discovery") and self.discovery:
        await self.discovery.stop()
        
    # Stop libp2p and event router
    if hasattr(self, "libp2p_node") and self.libp2p_node:
        await self.libp2p_node.stop()
        
    if hasattr(self, "event_router") and self.event_router:
        await self.event_router.stop()

  async def on_p2p_status(self, event: Event):
    """Handle opaque status updates from other nodes via P2P."""
    self.on_node_status("", event.data)

  async def on_p2p_clear_cache(self, event: Event):
    """Xử lý lệnh xóa KV cache từ các node khác trong cluster."""
    try:
        data = event.data
        if isinstance(data, str):
            import json
            data = json.loads(data)
        request_id = data.get("request_id")
        if request_id:
            if hasattr(self.inference_engine, "clear_cache"):
                self.inference_engine.clear_cache(request_id)
    except Exception as e:
        if DEBUG >= 1:
            print(f"[on_p2p_clear_cache] Error clearing cache for {request_id}: {e}")

  def on_node_status(self, request_id: str, data: str) -> None:
    """EXO-INSPIRED: Handle node status updates for cluster health monitoring and topology visualization."""
    try:
      if isinstance(data, str):
        status_data = json.loads(data)
      else:
        status_data = data
      
      status_type = status_data.get("type", "")
      
      # Health and Fault Tolerance logic
      if status_type == "node_status":
        node_id = status_data.get("node_id")
        status_val = status_data.get("status", "")
        if DEBUG >= 2: print(f"[FAULT_TOLERANCE] Received status update from node: {node_id} - {status_val}")
        
        # UI/Topology visualization updates
        if status_val.startswith("start_"):
          self.topology.active_node_id = status_data.get("node_id")
        elif status_val.startswith("end_"):
          if status_data.get("node_id") == self.topology.active_node_id:
            self.topology.active_node_id = None
            
      elif status_type == "supported_inference_engines":
        engines = status_data.get("engines", [])
        self.topology_inference_engines_pool.append(engines)
        
      elif status_type == "download_progress":
        if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
        progress = RepoProgressEvent.from_dict(status_data.get('progress'))
        self.node_download_progress[status_data.get('node_id')] = progress

      if self.topology_viz:
        self.topology_viz.update_visualization(self.topology, self._compute_lacp_partition(), self.id, self.node_download_progress)
        
    except Exception as e:
      if DEBUG >= 1: print(f"Error in on_node_status: {e}")
      if DEBUG >= 2: traceback.print_exc()

  async def check_fault_tolerance(self) -> None:
    """LACP 3.0: Dynamic Fault Tolerance with rebalancing on node failure."""
    current_time = time.time()

    # Phase 1: Check stuck requests (original watchdog)
    stuck_requests = []
    for request_id, req_info in list(self.outstanding_requests.items()):
      if isinstance(req_info, dict):
          start_time = req_info.get("start_time", 0)
      else:
          start_time = 0
      if start_time and (current_time - start_time) > 10.0:
        stuck_requests.append(request_id)

    if stuck_requests:
      if DEBUG >= 1: print(f"!!! [FAULT_TOLERANCE] Found {len(stuck_requests)} stuck requests. Triggering topology refresh.")
      await self.collect_topology(set())

    # Phase 2: LACP 3.0 node failure detection + rebalancing
    if self.fault_tolerance and self._distributed_pipeline:
      dead_nodes = self.fault_tolerance.get_dead_nodes()
      for dead_id in dead_nodes:
        if self.fault_tolerance._health.get(dead_id, None) and self.fault_tolerance._health[dead_id].is_blacklisted:
          if DEBUG >= 1: print(f"!!! [LACP_FAULT] Node {dead_id} is dead, triggering rebalance")

          current_assignments = {}
          partitions = self._compute_lacp_partition()
          for p in partitions:
            num_layers = 32
            start = int(p.start * num_layers)
            end = min(int(p.end * num_layers) - 1, num_layers - 1)
            current_assignments[p.node_id] = (start, end)

          async def repartition(alive_nodes, total_layers):
            nodes_resources = {}
            for nid in alive_nodes:
              if nid in self.topology.nodes:
                nodes_resources[nid] = NodeResources.from_device_capabilities(self.topology.nodes[nid])
            result = compute_partition(
              nodes=nodes_resources,
              total_layers=total_layers,
              model_id="",
            )
            new_assignments = {}
            for a in result.assignments:
              new_assignments[a.node_id] = (a.start_layer, a.end_layer)
            return new_assignments

          await self.fault_tolerance.rebalance_after_failure(
            dead_node_id=dead_id,
            current_assignments=current_assignments,
            total_layers=32,
            lacp_partition_fn=repartition,
          )

  async def on_p2p_tensor(self, event: Event):
    """Handle incoming tensor for inference via P2P."""
    data = event.data
    request_id = data.get("request_id")

    # CRITICAL FIX: Support both binary (tobytes) and legacy list-based format
    tensor_data_raw = data.get("tensor_data")
    shape = data.get("shape")
    dtype = data.get("dtype")

    if tensor_data_raw is not None and shape is not None:
        # Binary format: tainted from bytes
        import numpy as np
        if isinstance(tensor_data_raw, bytes):
            tensor_data = np.frombuffer(tensor_data_raw, dtype=np.dtype(dtype))
        elif isinstance(tensor_data_raw, str):
            tensor_data = np.frombuffer(bytes.fromhex(tensor_data_raw), dtype=np.dtype(dtype))
        else:
            tensor_data = np.frombuffer(tensor_data_raw, dtype=np.dtype(dtype))
        tensor_data = tensor_data.reshape(shape)
    else:
        # Legacy format: list-based
        tensor_data = np.array(data.get("tensor"))
        if dtype:
            tensor_data = tensor_data.astype(np.dtype(dtype))

    shard_dict = data.get("shard")
    shard = Shard.from_dict(shard_dict)
    inference_state = data.get("inference_state")

    # CRITICAL FIX: Pass orchestrator_id so results are routed back correctly
    orchestrator_id = data.get("_orchestrator_id")
    await self.process_tensor(shard, tensor_data, request_id, inference_state, orchestrator_id=orchestrator_id)

  async def on_p2p_prompt(self, event: Event):
    """Handle incoming prompt for inference via P2P."""
    data = event.data
    request_id = data.get("request_id")
    prompt = data.get("prompt")
    shard_dict = data.get("shard")
    shard = Shard.from_dict(shard_dict)
    inference_state = data.get("inference_state")

    # CRITICAL FIX: Store orchestrator_id so subsequent tensor forwarding
    # knows where to route results back
    orchestrator_id = data.get("_orchestrator_id")
    if orchestrator_id:
        self._request_orchestrator_map[request_id] = orchestrator_id
    
    await self.process_prompt(shard, prompt, request_id, inference_state)

  async def on_p2p_result(self, event: Event):
    """Handle incoming inference results/tokens via P2P."""
    data = event.data
    request_id = data.get("request_id")
    tokens = data.get("tokens", [])
    is_finished = data.get("is_finished", False)

    self._on_token_received(request_id, tokens, is_finished)
    self.trigger_on_token_callbacks(request_id, tokens, is_finished)

  async def get_tailscale_nodes(self) -> List[Dict]:
    """Lấy danh sách thiết bị từ mạng Tailscale."""
    from synapse.networking.tailscale.tailscale_discovery import TailscaleDiscovery
    from synapse.networking.tailscale.tailscale_helpers import get_tailscale_devices
    
    api_key = None
    tailnet = None
    
    # Ưu tiên lấy từ module discovery đang chạy
    if hasattr(self, "discovery") and isinstance(self.discovery, TailscaleDiscovery):
      api_key = self.discovery.tailscale_api_key
      tailnet = self.discovery.tailnet
    
    # Fallback nếu không có discovery hoặc discovery không phải Tailscale
    if not api_key:
      api_key = os.getenv("TS_API_KEY") or os.getenv("TAILSCALE_API_KEY")
    if not tailnet:
      tailnet = os.getenv("TS_TAILNET") or os.getenv("TAILSCALE_TAILNET")
      
    # Tailscale credentials must be set via environment variables
    if not api_key or not tailnet:
      print("Tailscale API key and tailnet are required. Set TS_API_KEY and TS_TAILNET environment variables.")

    try:
      devices_dict = await get_tailscale_devices(api_key, tailnet)
      results = []
      for name, dev in devices_dict.items():
        results.append({
          "device_id": dev.device_id,
          "name": dev.name,
          "addresses": dev.addresses,
          "last_seen": dev.last_seen.isoformat() if dev.last_seen else None,
          "is_self": False # Sẽ được đánh dấu ở API handler nếu cần
        })
      return results
    except Exception as e:
      if DEBUG >= 1: print(f"Error fetching tailscale nodes: {e}")
      return []


  def get_supported_inference_engines(self):
    return ["pytorch"]

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.event_router.publish("synapse/cluster/status", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool
  
  token_count = 0
  first_token_time = 0
  async def process_inference_result(
      self,
      shard,
      result: np.ndarray,
      request_id: Optional[str] = None,
      inference_state: Optional[dict] = None,
      orchestrator_id: Optional[str] = None,
  ):
    # CRITICAL DEBUG: Print immediately when entering this function
    if DEBUG >= 2:
      print(f"[NODE_DEBUG] process_inference_result called: shard={shard.model_id}, is_last={shard.is_last_layer()}, result_shape={result.shape if hasattr(result, 'shape') else 'N/A'}, request_id={request_id}")
    
    if request_id not in self.buffered_token_output:
      self.buffered_token_output[request_id] = ([], False)
    
    is_finished = self.buffered_token_output[request_id][1] or (len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens)

    # Tokens generation logic: only needed for LLMs
    if shard.is_last_layer() and not is_finished:
      # Check for empty results before sampling
      if result.size == 0 or (len(result.shape) >= 2 and result.shape[1] == 0):
        if DEBUG >= 2:
          print(f"[{request_id}] Warning: Empty result from inference, cannot sample. Marking as finished.")
        is_finished = True
        self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
        forward = np.array([[]], dtype=np.int32).reshape(1, 0)
        intermediate_result = []
      else:
        # Sample token from logits
        buf = self.buffered_token_output[request_id][0]
        
        # DEBUG: Print result shape and logits info
        if DEBUG >= 2:
          print(f"[{request_id}] [DEBUG] result shape: {result.shape}, dtype: {result.dtype}")
          if result.size > 0 and len(result.shape) >= 3:
            # Get logits at last position
            last_logits = result[0, -1, :] if result.shape[1] > 0 else result[0, 0, :]
            top_k = 5
            top_indices = np.argsort(last_logits)[-top_k:][::-1]
            print(f"[{request_id}] [DEBUG] Top {top_k} logits: {[(int(idx), float(last_logits[idx])) for idx in top_indices]}")
        
        token = await self.inference_engine.sample(result, temp=self.default_sample_temperature, generated_ids=buf if buf else None)
        await self.inference_engine.ensure_shard(shard)
        token_value = token.item()
        
        if DEBUG >= 1:
          print(f"[{request_id}] [DEBUG] Sampled token_value: {token_value}")

        # Get eos_token_id safely
        eos_token_id = None
        if hasattr(self.inference_engine, 'tokenizer') and self.inference_engine.tokenizer is not None:
          eos_token_id = getattr(self.inference_engine.tokenizer, 'eos_token_id', None)
          if eos_token_id is None and hasattr(self.inference_engine.tokenizer, '_tokenizer'):
            eos_token_id = getattr(self.inference_engine.tokenizer._tokenizer, 'eos_token_id', None)

        # Qwen-specific stop tokens
        qwen_stop_tokens = [151643, 151644, 151645]

        stop_ids = set(qwen_stop_tokens)
        if eos_token_id is not None:
          stop_ids.add(eos_token_id)
        
        if len(buf) == 0 and token_value in stop_ids and result.size > 0:
          try:
            retry_logits = np.array(result, copy=True)
            for sid in stop_ids:
              if 0 <= sid < retry_logits.shape[-1]:
                retry_logits[0, -1, sid] = -1e9
            retry_token = await self.inference_engine.sample(
              retry_logits,
              temp=self.default_sample_temperature,
              generated_ids=buf if buf else None,
            )
            token = retry_token
            token_value = token.item()
          except Exception as e:
            if DEBUG >= 1: print(f"[sample] retry sampling error: {e}")

        # Add to buffer
        self.buffered_token_output[request_id][0].append(token_value)
        token_emitted = True
        
        # Repetition detection
        buffered_tokens = self.buffered_token_output[request_id][0]
        if len(buffered_tokens) >= 10:
          last_10_tokens = buffered_tokens[-10:]
          if len(set(last_10_tokens)) == 1:
            if DEBUG >= 1: 
              print(f"[{request_id}] Detected repetition loop: token {token_value} repeated 10 times. Stopping generation.")
            is_finished = True
        
        # Stop token detection
        if (eos_token_id is not None and token_value == eos_token_id) or (token_value in qwen_stop_tokens):
          if DEBUG >= 1:
            print(f"[{request_id}] Detected EOS/stop token (ID: {token_value}). Stopping generation.")
          is_finished = True
          if self.buffered_token_output[request_id][0]:
            self.buffered_token_output[request_id][0].pop()
          token_emitted = False 
        
        is_finished = is_finished or len(buffered_tokens) >= self.max_generate_tokens
        if DEBUG >= 2: 
          print(f"[{request_id}] Token: {token_value}, Emitted: {token_emitted}, Finished: {is_finished}")
        
        forward = token.reshape(1, -1)
        if self.buffered_token_output[request_id][0] and token_emitted:
          intermediate_result = [self.buffered_token_output[request_id][0][-1]]
        else:
          intermediate_result = []
          if is_finished and not self.buffered_token_output[request_id][0]:
            if DEBUG >= 2:
              print(f"[{request_id}] Generation finished with no tokens, sending empty result with is_finished=True")
    else:
      forward = result
      intermediate_result = []

    # Forward tensor to next partition
    if not shard.is_last_layer():
      self.outstanding_requests[request_id] = "waiting"
      forwarded_tokens = await self.forward_tensor(
        shard,
        forward,
        request_id,
        self.get_partition_index(offset=1, base_shard=shard),
        inference_state,
      )
      if forwarded_tokens is not None and forwarded_tokens.size > 0:
        self._on_token_received(request_id, forwarded_tokens.flatten().tolist(), False)
      return forwarded_tokens
    else:
      if DEBUG >= 1:
        print(f"[{request_id}] [LAST_LAYER] Triggering callback: tokens={intermediate_result}, is_finished={is_finished}")
      self.trigger_on_token_callbacks(request_id, intermediate_result, is_finished)
      
      if intermediate_result or is_finished:
        # CRITICAL FIX: Publish result to BOTH request_id topic and orchestrator_id topic.
        # The orchestrator node subscribes to result/{orchestrator_id} events,
        # while peers subscribe to result/{request_id} events.
        # We publish on both to ensure delivery regardless of which side the
        # orchestrator listens on.
        result_payload = {
            "request_id": request_id,
            "tokens": intermediate_result,
            "is_finished": is_finished
        }

        # 1. Publish to request_id-based topic (general broadcast)
        await self.event_router.publish(f"synapse/inference/result/{request_id}", result_payload)

        # 2. Publish directly to orchestrator if this is not the orchestrator
        if orchestrator_id and orchestrator_id != self.id:
            await self.event_router.publish(f"synapse/inference/result/{orchestrator_id}", result_payload)
    
      if is_finished:
        if request_id in self.buffered_token_output:
          self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
        self.outstanding_requests.pop(request_id, None)
        self._prompt_token_ids.pop(request_id, None)
    
        return np.array(self.buffered_token_output.get(request_id, ([], False))[0])


  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = {},
    max_generate_tokens: Optional[int] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    self._schedule_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
        }),
      )
    )
    if self.tracer is not None:
      from synapse.orchestration.tracing import TraceContext
      trace_ctx = TraceContext(request_id=request_id or "", sequence_number=0)
      trace_ctx.request_span = None
      with self.tracer.start_span("node.process_prompt", trace_ctx, {"node_id": self.id, "prompt_len": len(prompt)}) as span:
        _start = time.perf_counter_ns()
        resp = await self._process_prompt(base_shard, prompt, request_id, inference_state, max_generate_tokens)
        _elapsed = time.perf_counter_ns() - _start
        if span:
          span.set_attribute("request_id", request_id or "")
          span.set_attribute("elapsed_ms", _elapsed / 1e6)
        return resp
    else:
      _start = time.perf_counter_ns()
      resp = await self._process_prompt(base_shard, prompt, request_id, inference_state, max_generate_tokens)
      elapsed_time_ns = time.perf_counter_ns() - _start
    self._schedule_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=} {elapsed_time_ns=}")
    return resp

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None, inference_state: Optional[dict] = None, max_generate_tokens: Optional[int] = None) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=}")

    # LACP 3.0: Use distributed pipeline when multiple nodes are available
    node_count = len(self.topology.nodes) if self.topology.nodes else 0
    if node_count > 1 and self.is_orchestrator_for_request(request_id):
      if DEBUG >= 1: print(f"[{request_id}] LACP 3.0: Using distributed pipeline across {node_count} nodes")
      self.distributed_active = True
      self.distributed_node_count = node_count
      self.distributed_status = "distributed"
      try:
        tokens = await self.distributed_pipeline.process_prompt_distributed(
          base_shard=base_shard,
          prompt=prompt,
          request_id=request_id,
          max_generate_tokens=max_generate_tokens,
          num_micro_batches=self.num_micro_batches,
        )
        if tokens:
          return np.array(tokens) if tokens else None
        # Empty tokens from distributed pipeline — may be a timeout
        self.distributed_status = "fallback"
        self.distributed_active = False
      except Exception as e:
        if DEBUG >= 1:
          print(f"[{request_id}] Distributed pipeline failed: {e}, falling back to single-node")
          traceback.print_exc()
        self.distributed_status = "fallback"
        self.distributed_active = False

    if not shard.is_first_layer():
      if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
      self.outstanding_requests[request_id] = "waiting"
      resp = await self.forward_prompt(shard, prompt, request_id, 0, inference_state)
      return None
    else:
      self.outstanding_requests[request_id] = "processing"
      try:
        token_limit = max_generate_tokens if isinstance(max_generate_tokens, int) and max_generate_tokens > 0 else self.max_generate_tokens
        latest_result = None

        # First decoding step from full prompt.
        prompt_ids = await self.inference_engine.encode(shard, prompt)
        self._prompt_token_ids[request_id] = prompt_ids
        result, inference_state = await self.inference_engine.infer_prompt(request_id, shard, prompt, inference_state)
        latest_result = await self.process_inference_result(shard, result, request_id, inference_state)

        # Next decoding steps are autoregressive from last generated token.
        prev_len = -1
        wait_start = time.perf_counter()
        while True:
          # Wait for tokens arriving via callbacks or local processing
          buffered_tokens, finished = self.buffered_token_output.get(request_id, ([], False))
          current_len = len(buffered_tokens)
          
          if finished or current_len >= token_limit:
            break
            
          if current_len == 0 or current_len == prev_len:
            # yield to allow callback processing
            await asyncio.sleep(0.1) # Throttled wait
            
            # Dynamic Fault Tolerance: Nếu 10s không nhận được trả lời (Node sập/Đứt mạng)
            if time.perf_counter() - wait_start > 10.0:
              if DEBUG >= 1: 
                print(f"[{request_id}] [Watchdog] Generation stalled for 10s. Triggers Dynamic Fault Tolerance (Reroute)!")
              
              # 1. Heartbeat của ElectionManager (đang chạy ngầm) đã/sẽ loại bỏ node sập khỏi mesh
              # 2. Ta chỉ cần GỬI LẠI last_token, P2P Router sẽ tự đẩy task cho một Node khỏe (mới được bầu lên).
              last_token_for_retry = np.array([[buffered_tokens[-1]]]) if len(buffered_tokens) > 0 else prompt_ids
              if len(buffered_tokens) > 0:
                  await self.process_tensor(base_shard, last_token_for_retry, request_id, inference_state)
              else: # Retry from process_prompt
                  # Lỗi ngay từ token đầu, encode lại prompt
                  await self.process_tensor(base_shard, prompt_ids, request_id, inference_state)
                  
              wait_start = time.perf_counter() # Reset watchdog

              
            continue
            
          # Progress logging
          tokens = self.buffered_token_output[request_id][0]
          if DEBUG >= 1:
            print(f"[{request_id}] PROGRESS: Generated {len(tokens)}/{self.max_generate_tokens} tokens.")
          
          # Check if we are finished
          if self.buffered_token_output[request_id][1] or len(tokens) >= self.max_generate_tokens:
            break
            
          prev_len = current_len
          last_token = np.array([[tokens[-1]]])
          
          # Continue autoregressive loop
          latest_result = await self.process_tensor(base_shard, last_token, request_id, inference_state)

        buffered_tokens, finished = self.buffered_token_output.get(request_id, ([], False))
        if not finished:
          self.buffered_token_output[request_id] = (buffered_tokens, True)
          self.trigger_on_token_callbacks(request_id, [], True)
        return latest_result if latest_result is not None else result
      except Exception as e:
        # CRITICAL FIX: If exception occurs, ensure API is notified
        if DEBUG >= 1:
          print(f"[{request_id}] Error in _process_prompt: {e}")
          import traceback
          traceback.print_exc()
        # Trigger callback with error state so API doesn't hang
        if shard.is_last_layer():
          if DEBUG >= 1:
            print(f"[{request_id}] Triggering error callback to notify API")
          self.trigger_on_token_callbacks(request_id, [], True)
        raise
      finally:
        # Delete local tensor references explicitly to free PyTorch graphs
        if 'latest_result' in locals(): del latest_result
        if 'last_token' in locals(): del last_token
        if 'buffered_tokens' in locals(): del buffered_tokens
        if 'result' in locals(): del result
        self.outstanding_requests.pop(request_id, None)
        self._prompt_token_ids.pop(request_id, None)
        if hasattr(self.inference_engine, "clear_cache"):
            self.inference_engine.clear_cache(request_id)
        # Broadcast clear cache to cluster
        self._schedule_task(self.event_router.publish("synapse/cluster/cache/clear", json.dumps({"request_id": request_id})))
        # Ensure CUDA cache is freed
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

  async def enqueue_example(
      self,
      base_shard: Shard,
      example: np.ndarray,
      target: np.ndarray,
      length: np.ndarray,
      request_id: Optional[str] = None,
      train: bool = False,
  ):
      shard = self.get_current_shard(base_shard)
      if shard.is_first_layer():
          loss = await self.process_example(shard, example, target, length, train, request_id)
          return loss
      else:
          if request_id is None:
              request_id = str(uuid.uuid4())
          self.outstanding_requests[request_id] = "waiting"
          # FIX: Use get_partition_index(offset=1) instead of hardcoded 0
          # This ensures example is forwarded to the next partition in the ring
          next_partition_index = self.get_partition_index(offset=1, base_shard=base_shard)
          loss = await self.forward_example(shard, example, target, length, train, request_id, next_partition_index)
          return loss

  async def coordinate_save(
    self,
    base_shard: Shard,
    iteration: int,
    destination: str,
  ):
    shard = self.get_current_shard(base_shard)
    model = shard.model_id
    sid = shard.__hash__()
    path = f"{destination}/{model}/{sid}-{iteration}.safetensors"
    self.outstanding_requests[f"{sid}::{iteration}"] = "Checking"
    if model not in self.checkpoints:
      self.checkpoints[model] = {}
    if sid not in self.checkpoints[model]:
      self.checkpoints[model][sid] = []
    if len(self.checkpoints[model][sid]) < 1 or self.checkpoints[model][sid][-1] < iteration:
      print(f"Saving checkpoint to {path}")
      self.outstanding_requests[f"{sid}::{iteration}"] = "Saving"
      import os
      os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
      await self.inference_engine.save_checkpoint(shard, path)
      self.checkpoints[model][sid] = sorted(self.checkpoints[model][sid] + [iteration])
    self.outstanding_requests.pop(f"{sid}::{iteration}")

  async def process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ):
    shard = self.get_current_shard(base_shard)
    self._schedule_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"start_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "example_size": example.size,
          "example_shape": example.shape,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_example(shard, example, target, length, train, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"end_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    return resp

  async def _process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 1: print(f"[{request_id}] process_example: {example.shape=}")
    try:
      target = target.astype(int)
      if train:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "training"
          loss, grad = await self.inference_engine.train(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss, backgrad = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
          self.outstanding_requests[request_id] = "training"
          partial_loss, grad = await self.inference_engine.train(request_id, shard, example, backgrad, length, loss="back_gradient")
        self.outstanding_requests.pop(request_id)
        if shard.is_first_layer():
          return loss
        else:
          return loss, grad
      else:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "evaluating"
          loss = await self.inference_engine.evaluate(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
        self.outstanding_requests.pop(request_id)
        return loss
    except Exception as e:
      self.outstanding_requests.pop(request_id)
      print(f"Error processing example for shard {shard}: {e}")
      traceback.print_exc()
      raise
        
  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
    orchestrator_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id, inference_state, orchestrator_id=orchestrator_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    if DEBUG >= 2: print(f"[{request_id}] process_tensor: {base_shard=} {shard=} {tensor.size=} {tensor.shape=} {elapsed_time_ns=}")
    return resp

  async def _process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
    orchestrator_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)

    # CRITICAL FIX: Store orchestrator_id for downstream forwarding chain
    if orchestrator_id:
        self._request_orchestrator_map[request_id] = orchestrator_id

    try:
      self.outstanding_requests[request_id] = "processing"
      input_data = tensor
      if shard.is_first_layer() and request_id in self._prompt_token_ids and request_id in self.buffered_token_output:
        buf = self.buffered_token_output[request_id][0]
        if buf:
          full_ids = np.concatenate([
            self._prompt_token_ids[request_id],
            np.array(buf, dtype=np.int64),
          ])
          input_data = full_ids
          if DEBUG >= 2:
            print(f"[{request_id}] First layer: full sequence length {len(full_ids)} (prompt + {len(buf)} generated)")
      result, inference_state = await self.inference_engine.infer_tensor(request_id, shard, input_data, inference_state)
      ret = await self.process_inference_result(shard, result, request_id, inference_state, orchestrator_id=orchestrator_id)
      return ret
    except Exception as e:
      self.outstanding_requests.pop(request_id, None)
      print(f"Error processing tensor for shard {shard}: {e}")
      traceback.print_exc()
      raise
  
  async def forward_example(
    self,
    base_shard: Shard,
    step: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self._compute_lacp_partition(base_shard)[target_index].node_id
    target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")
    target_peer = next((p for p in self.peers if p.id() == target_id), None)
    if not target_peer:
      raise ValueError(f"peer for {target_index} not found")
    if DEBUG >= 1: print(f"sending example to {target_peer.id()}: {step} => {target} ({length})")
    resp = await target_peer.send_example(target_shard, step, target, length, request_id=request_id, train=train)
    return resp

  async def forward_prompt(
      self,
      base_shard: Shard,
      prompt: str,
      request_id: str,
      target_index: int,
      inference_state: Optional[dict] = None,
  ) -> None:
      if DEBUG >= 1: print(f"target partition index: {target_index}")
      target_id = self._compute_lacp_partition(base_shard)[target_index].node_id
      next_shard = self.get_current_shard(base_shard, target_index)
      if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
      if target_id == self.id:
          await self._process_prompt(next_shard, prompt, request_id, inference_state)
      else:
          logger.info(f"[P2P_MESH] Publishing prompt to Node: {target_id}")
          await self.event_router.publish(f"synapse/inference/prompt/{target_id}", {
              "request_id": request_id,
              "shard": next_shard.to_dict(),
              "prompt": prompt,
              "inference_state": inference_state,
              # CRITICAL FIX: Include orchestrator_id so downstream nodes know
              # where to route results and tokens back to the original requester
              "_orchestrator_id": self.id,
          })
  
  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self._compute_lacp_partition(base_shard)[target_index].node_id
    target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")

    orchestrator_id = self._orchestrator_id_for_request(request_id)

    if target_id == self.id:
      return await self.process_tensor(target_shard, tensor, request_id, inference_state, orchestrator_id=orchestrator_id)
    else:
      logger.info(f"[P2P_MESH] Publishing work to Node: {target_id}")
      # CRITICAL FIX: Use binary tensor encoding (tobytes + shape + dtype)
      # instead of tensor.tolist() which creates massive nested JSON arrays.
      await self.event_router.publish(f"synapse/inference/tensor/{target_id}", {
        "request_id": request_id,
        "shard": target_shard.to_dict(),
        "tensor_data": tensor.tobytes(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "inference_state": inference_state,
        "_orchestrator_id": orchestrator_id,
      })
      return None

  def get_partition_index(self, offset: int = 0, base_shard: Optional[Shard] = None):
    if not self.topology.nodes:
      if DEBUG >= 1: print("No nodes in topology. Skipping forward.")
      return None
    # CRITICAL FIX: Use authoritative partitions from master if available,
    # otherwise compute locally (fallback for single-node or before first broadcast)
    if self._authoritative_partitions is not None:
        partitions = self._authoritative_partitions
    else:
        partitions = self._compute_lacp_partition(base_shard)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      if DEBUG >= 1: print(f"Node {self.id} not found in partitions — may have departed or topology changed")
      return 0
    return (current_partition_index + offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    if index is None:
      index = self.get_partition_index(base_shard=base_shard)
    # CRITICAL FIX: Use authoritative partitions if available
    if self._authoritative_partitions is not None:
        partitions = self._authoritative_partitions
    else:
        partitions = self._compute_lacp_partition(base_shard)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    next_peers = await self.discovery.discover_peers(wait_for_peers) if hasattr(self, "discovery") else []
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_to_disconnect = [peer for peer in peers_removed if await peer.is_connected()]
    peers_to_connect = [peer for peer in peers_added + peers_updated + peers_unchanged if not await peer.is_connected()]

    if DEBUG >= 1:
      print(f"update_peers (Node {self.id}): discovery={type(self.discovery).__name__ if hasattr(self, 'discovery') and self.discovery else 'None'}")
      print(f"  next_peers IDs: {next_peer_ids}")

    if DEBUG >= 3:
      print(f"update_peers details: added={[p.id() for p in peers_added]} removed={[p.id() for p in peers_removed]}")

    async def disconnect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.connect(), timeout)
        return True
      except Exception as e:
        print(f"Error connecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    disconnect_results = await asyncio.gather(*(disconnect_with_timeout(peer) for peer in peers_to_disconnect), return_exceptions=True)
    connect_results = await asyncio.gather(*(connect_with_timeout(peer) for peer in peers_to_connect), return_exceptions=True)

    successful_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is True]
    failed_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is False]
    successful_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is True]
    failed_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is False]
    if DEBUG >= 3:
      if successful_disconnects: print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
      if failed_disconnects: print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
      if successful_connects: print(f"Successfully connected peers: {_pretty(successful_connects)}")
      if failed_connects: print(f"Failed to connect peers: {_pretty(failed_connects)}")

    self.peers = next_peers
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def select_best_inference_engine(self):
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)

  async def periodic_topology_collection(self, interval: int):
    while True:
      await asyncio.sleep(interval)
      try:
        did_peers_change = await self.update_peers()
        if DEBUG >= 3: print(f"{did_peers_change=}")
        
        # LACP 3.0: Record heartbeat for all connected peers
        self._lazy_init_fault_tolerance()
        if self.fault_tolerance:
          for peer_id in [p.id() for p in self.peers]:
            self.fault_tolerance.record_heartbeat(peer_id)
          # Register any new peers
          for peer_id in [p.id() for p in self.peers]:
            if peer_id not in self.fault_tolerance._health:
              self.fault_tolerance.register_node(peer_id)

        # EXO-INSPIRED: Dynamic Fault Tolerance check
        await self.check_fault_tolerance()
        
        await self.collect_topology(set())

        # CRITICAL FIX: Master broadcasts authoritative topology after each collection
        # so that all nodes have a consistent view of the cluster
        self._schedule_task(self.broadcast_authoritative_topology(), "broadcast_topology")

        if did_peers_change:
          await self.select_best_inference_engine()
      except Exception as e:
        print(f"Error in periodic_topology_collection: {e}")
        if DEBUG >= 2: traceback.print_exc()

  async def _metrics_update_loop(self):
    """Update cached NodeMetrics from realtime monitor every cycle."""
    while True:
      await asyncio.sleep(RealtimeMonitor.SAMPLE_INTERVAL)
      self._current_metrics = self.monitor.latest

  async def collect_topology(self, visited: set[str], max_depth: int = 4) -> Topology:
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    if DEBUG >= 3: print(f"Collecting topology {max_depth=} {visited=}")

    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    for peer in self.peers:
      next_topology.update_node(peer.id(), peer.device_capabilities())
      next_topology.add_edge(self.id, peer.id(), peer.description())

      if peer.id() in prev_visited:
        continue

      if max_depth <= 0:
        if DEBUG >= 2: print("Max depth reached. Skipping...")
        continue

      try:
        other_topology = await asyncio.wait_for(peer.collect_topology(visited, max_depth=max_depth - 1), timeout=10.0)
        if DEBUG >= 2: print(f"Collected topology from: {peer.id()}: {other_topology}")
        next_topology.merge(peer.id(), other_topology)
      except asyncio.TimeoutError:
        # Timeout là lỗi thông thường, không cần in traceback
        if DEBUG >= 2:
          print(f"Timeout collecting topology from {peer.id()} (timeout=10.0s)")
      except asyncio.CancelledError:
        # CancelledError có thể xảy ra khi timeout, xử lý gracefully
        if DEBUG >= 2:
          print(f"Cancelled collecting topology from {peer.id()}")
      except Exception as e:
        # Chỉ in traceback cho các lỗi không mong đợi
        if DEBUG >= 2:
          print(f"Error collecting topology from {peer.id()}: {e}")
          traceback.print_exc()
        else:
          print(f"Error collecting topology from {peer.id()}: {type(e).__name__}: {e}")

    next_topology.active_node_id = self.election_manager.current_master_id
    self.topology = next_topology
    if self.topology_viz:
      self.topology_viz.update_visualization(self.topology, self._compute_lacp_partition(), self.id)

    # CRITICAL FIX: Master broadcasts authoritative topology to ensure all nodes agree
    if self.election_manager.is_master():
        self._schedule_task(self.broadcast_authoritative_topology(), "broadcast_topology")
    return self.topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    if DEBUG >= 1: 
      print(f"[{request_id}] [CALLBACK] Triggering all on_token callbacks: tokens={tokens}, is_finished={is_finished}, num_callbacks={len(self.on_token.callbacks)}")
    self.on_token.trigger_all(request_id, tokens, is_finished)
    if DEBUG >= 1:
      print(f"[{request_id}] [CALLBACK] Callbacks triggered successfully")
  
  async def broadcast_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Broadcasting result: {request_id=} {result=} {is_finished=}")
    async def send_result_to_peer(peer):
      try:
        if DEBUG >= 1: print(f"  [BROADCAST] Sending to {peer.id()}...")
        await asyncio.wait_for(peer.send_result(request_id, result, is_finished), timeout=30.0)
        if DEBUG >= 1: print(f"  [BROADCAST] Sent to {peer.id()} successfully.")
      except asyncio.TimeoutError:
        print(f"!!! [BROADCAST_TIMEOUT] Timeout broadcasting result to {peer.id()} (30s)")
      except Exception as e:
        print(f"!!! [BROADCAST_ERROR] Error broadcasting result to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_result_to_peer(peer) for peer in self.peers], return_exceptions=True)

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")

    async def send_status_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_opaque_status(request_id, status), timeout=15.0)
      except asyncio.TimeoutError:
        if DEBUG >= 2:
          print(f"Timeout sending opaque status to {peer.id()}")
      except Exception as e:
        # Lỗi đã được xử lý trong send_opaque_status với retry logic
        # Chỉ log ở mức DEBUG để tránh spam log
        if DEBUG >= 3:
          print(f"Error sending opaque status to {peer.id()}: {e}")
          if DEBUG >= 5:
            traceback.print_exc()

    await asyncio.gather(*[send_status_to_peer(peer) for peer in self.peers], return_exceptions=True)
    # in the case of opaque status, we also want to receive our own opaque statuses
    self.on_opaque_status.trigger_all(request_id, status)

  def _compute_lacp_partition(self, base_shard: Optional[Shard] = None) -> List[Partition]:
    """Compute partition using TFLOPS-proportional (default) or LACP multi-resource scoring."""
    num_layers = base_shard.n_layers if base_shard and base_shard.n_layers > 0 else 32
    model_id = base_shard.model_id if base_shard else ""

    if self.tflops_proportional:
        partitioner = TFLOPSProportionalPartitioner()
        return partitioner.partition(self.topology, base_shard)

    nodes_resources = {}
    for node_id, dc in self.topology.nodes.items():
        nr = NodeResources.from_device_capabilities(dc)
        if node_id == self.id:
            m = self._current_metrics
            nr.gpu_util_pct = m.gpu_util_pct
            nr.system_ram_used_mb = m.ram_used_mb
            nr.current_power_w = m.gpu_power_w
        nodes_resources[node_id] = nr

    latency_matrix = self._build_latency_matrix()

    lacp_result = compute_partition(
        nodes=nodes_resources,
        latency_matrix=latency_matrix,
        total_layers=num_layers,
        model_id=model_id,
    )

    partitions = []
    for a in lacp_result.assignments:
        partitions.append(Partition(
            node_id=a.node_id,
            start=a.start_layer / num_layers,
            end=(a.end_layer + 1) / num_layers,
        ))
    return partitions

  def _build_latency_matrix(self) -> Optional[Dict[str, Dict[str, float]]]:
    """Build latency matrix.
    Returns cached/default matrix synchronously.
    Triggers background TCP RTT probing to update cache for next call."""
    node_ids = list(self.topology.nodes.keys())
    if len(node_ids) < 2:
      return None

    if not hasattr(self, '_latency_cache'):
      self._latency_cache: Dict[str, Dict[str, float]] = {}
      for nid in node_ids:
        self._latency_cache[nid] = {other: 0.0 if other == nid else 50.0 for other in node_ids}

    async def probe_peers_background():
      for peer in self.peers:
        try:
          if hasattr(peer, 'ping') and callable(peer.ping):
            latency_ms = await asyncio.wait_for(peer.ping(timeout=2.0), timeout=3.0)
            if latency_ms > 0:
              for nid in node_ids:
                if nid != self.id:
                  self._latency_cache[self.id][nid] = latency_ms
                  self._latency_cache[nid][self.id] = latency_ms
              if DEBUG >= 1:
                print(f"[latency] probed {peer.id()}: {latency_ms:.1f}ms")
        except (asyncio.TimeoutError, Exception) as e:
          if DEBUG >= 2:
            print(f"[latency] probe failed for {peer.id()}: {e}")

    if self.peers:
      try:
        loop = asyncio.get_running_loop()
        loop.create_task(probe_peers_background())
      except RuntimeError:
        pass

    return self._latency_cache

  def _setup_p2p_rpc_handlers(self):
    """Setup listeners for P2P RPC requests (Health, Topology)."""
    self.event_router.subscribe(f"synapse/rpc/health/request/{self.id}", self.on_p2p_health_request)
    self.event_router.subscribe(f"synapse/rpc/topology/request/{self.id}", self.on_p2p_topology_request)

  async def on_p2p_health_request(self, event: Event):
    try:
        response_topic = event.data.get("_response_topic")
        if response_topic:
            await self.event_router.publish(response_topic, {"status": "ok"}, origin=self.id)
    except Exception as e:
        logger.error(f"Error in on_p2p_health_request: {e}")

  async def on_p2p_topology_request(self, event: Event):
    try:
        response_topic = event.data.get("_response_topic")
        if response_topic:
            await self.event_router.publish(response_topic, {"topology": self.topology.to_dict()}, origin=self.id)
    except Exception as e:
        logger.error(f"Error in on_p2p_topology_request: {e}")
        traceback.print_exc()

  def _on_token_received(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    """Internal callback to synchronize buffered tokens on orchestrator node."""
    if request_id not in self.buffered_token_output:
      self.buffered_token_output[request_id] = ([], False)
    
    current_tokens, _ = self.buffered_token_output[request_id]
    
    # We only add tokens that are not already at the END of our buffer.
    # This prevents double-adding on the node that actually did the sampling.
    if tokens:
      # If we receive multiple tokens, we might need a more robust check, 
      # but usually it's one token at a time in streaming.
      if not current_tokens or current_tokens[-len(tokens):] != tokens:
        current_tokens.extend(tokens)
    
    self.buffered_token_output[request_id] = (current_tokens, is_finished)
    if DEBUG >= 2:
      print(f"[{request_id}] Internal buffer updated: {len(current_tokens)} tokens, finished={is_finished}")

  @property
  def current_topology(self) -> Topology:
    return self.topology

  def _orchestrator_id_for_request(self, request_id: str) -> str:
    """Return the orchestrator node_id that should receive results for this request.
    If this node initiated the request, returns self.id.
    If the request came from another node, returns the stored orchestrator_id.
    """
    if request_id in self._request_orchestrator_map:
        return self._request_orchestrator_map[request_id]
    # We are the orchestrator
    self._request_orchestrator_map[request_id] = self.id
    return self.id

  def is_orchestrator_for_request(self, request_id: str) -> bool:
    """Check if this node is the orchestrator for the given request."""
    if request_id in self._request_orchestrator_map:
      return self._request_orchestrator_map[request_id] == self.id
    return True

  async def broadcast_authoritative_topology(self):
    """If this node is master, broadcast the authoritative topology to all peers."""
    if not self.election_manager.is_master():
        return

    topology_dict = self.topology.to_dict()
    await self.event_router.publish("synapse/cluster/topology", {
        "topology": topology_dict,
        "master_id": self.id,
    })

  async def on_p2p_topology_broadcast(self, event: Event):
    """Receive authoritative topology from the master node."""
    data = event.data
    if isinstance(data, dict):
        topology_data = data.get("topology")
        if topology_data:
            received_topo = Topology.from_dict(topology_data)
            master_id = data.get("master_id")
            if master_id == self.election_manager.current_master_id or not self.election_manager.current_master_id:
                self.topology = received_topo
                if self.topology_viz:
                    self.topology_viz.update_visualization(
                        self.topology,
                        self._compute_lacp_partition(),
                        self.id
                    )

  async def broadcast_authoritative_partition(self, base_shard: Shard):
    """If this node is master, compute and broadcast the authoritative partition."""
    if not self.election_manager.is_master():
        return

    partitions = self._compute_lacp_partition(base_shard)
    self._authoritative_partitions = partitions
    self._authoritative_base_shard = base_shard

    partition_data = [{"node_id": p.node_id, "start": p.start, "end": p.end} for p in partitions]
    await self.event_router.publish("synapse/cluster/partition", {
        "partitions": partition_data,
        "base_shard": base_shard.to_dict(),
        "master_id": self.id,
    })

  async def on_p2p_partition_broadcast(self, event: Event):
    """Receive authoritative partition from the master node."""
    data = event.data
    if isinstance(data, dict):
        master_id = data.get("master_id")
        if master_id == self.election_manager.current_master_id or not self.election_manager.current_master_id:
            partition_data = data.get("partitions", [])
            base_shard_dict = data.get("base_shard")
            self._authoritative_partitions = [Partition(**p) for p in partition_data]
            if base_shard_dict:
                self._authoritative_base_shard = Shard.from_dict(base_shard_dict)
