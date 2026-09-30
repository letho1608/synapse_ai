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
from synapse.topology.resource_weight import NodeResources, ResourceScorer, apply_measured_bandwidth
from synapse.topology.latency_matrix import LatencyMatrix, LatencyEntry, LatencyProber
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
from synapse.transport.codec import encode_tensor, decode_tensor

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
    partition_method: str = "lrm",
    tflops_proportional: bool = True,
    ilp_time_limit_seconds: Optional[float] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.event_router = event_router
    self.libp2p_node = libp2p_node
    self.shard_downloader = shard_downloader
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    
    # Adaptive LACP: Real-time latency tracking
    self._last_forward_latencies: Dict[str, float] = {}
    self._adaptive_rebalance_threshold = 0.5 # 500ms jitter triggers rebalance
    
    # Distributed Engine Optimization Settings
    self.load_quantization = "auto"
    self.lookahead_window = 3
    self.transport_quantization = "int8"
    
    # Initialize Election Manager. The LACP composite resource scorer is wired
    # in so the elected master's weight derives from the multi-resource score
    # (VRAM, TFLOPS, CPU, RAM, disk, bandwidth, latency, power) instead of the
    # default 1.0. The weight is (re)computed once hardware is known and again
    # whenever the cluster topology grows.
    self._resource_scorer = ResourceScorer()
    self.election_manager = ElectionManager(
        self.id,
        self.event_router,
        resource_scorer=self._election_resource_scorer,
    )

    # Real bandwidth probing for same-machine/peer links (TCP echo probes).
    self._latency_prober = LatencyProber()
    self._measured_bandwidth: Dict[str, float] = {}
    self._measured_rtt: Dict[str, float] = {}  # peer -> measured ping RTT (ms)
    # Measured-placement objective set from API settings (None = env default).
    self.partition_objective: Optional[str] = None

    # LACP Distributed Pipeline Extension
    self._distributed_pipeline = None

    # LACP TCP Transport for binary tensor forwarding
    self.tcp_transport: Optional[Any] = None

    # LACP zero-copy shared-memory transport for same-machine peers
    self.shm_transport: Optional[Any] = None

    # LACP Fault Tolerance Manager
    self.fault_tolerance = None

    # Most recent LACP rebalance result (kept so the recovered cross-node
    # checkpoint staged by rebalance_after_failure is not discarded).
    self._last_rebalance_result = None

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
    # Request ids for which THIS node is the sampling stage (it appended the
    # tokens directly in process_inference_result). Any later delivery of the
    # same tokens via the on_token stream / P2P loopback is an echo and must
    # not be double-appended — while genuine adjacent repeats from OTHER
    # nodes must always be appended.
    self._locally_sampled: set = set()
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}
    
    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self.default_sample_temperature = default_sample_temperature
    self.num_micro_batches = 1
    self.tflops_proportional = tflops_proportional  # TFLOPS-proportional sharding (multi-resource LACP off by default)
    # Partition method for the LACP multi-resource path: "lrm" (Largest
    # Remainder, fast) or "ilp" (PuLP integer programming, optimal). The
    # production partitioner path reads this knob and passes it to
    # compute_partition(method=...). Selectable via the --partition-method
    # CLI flag or the Node constructor (the production partitioner can use
    # the PuLP ILP solver through configuration).
    self.partition_method: str = partition_method
    # Wall-clock budget (seconds) for the ILP partition solve; None = no cap.
    # When the budget is exceeded the shipped path degrades to the fast LRM
    # method, observable via LACPPartition.method_used/solver_status.
    self.ilp_time_limit_seconds: Optional[float] = ilp_time_limit_seconds
    
    self.peers = []
    
    self.event_router.subscribe("synapse/cluster/cache/clear", self.on_p2p_clear_cache)
    self.event_router.subscribe("synapse/cluster/status", self.on_p2p_status)
    self.event_router.subscribe(f"synapse/inference/tensor/{self.id}", self.on_p2p_tensor)
    self.event_router.subscribe(f"synapse/inference/prompt/{self.id}", self.on_p2p_prompt)
    # C-W4: peer-sent training examples + opaque status had publishers but no
    # subscribers (black holes). Route them into enqueue_example/on_node_status.
    self.event_router.subscribe(f"synapse/inference/example/{self.id}", self.on_p2p_example)
    self.event_router.subscribe(f"synapse/inference/status/{self.id}", self.on_p2p_status_direct)
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

  @property
  def distributed_pipeline(self):
    """LACP Distributed Pipeline Extension — lazy-initialized."""
    if self._distributed_pipeline is None:
      from synapse.pipeline.node_extension import DistributedPipelineExtension
      self._distributed_pipeline = DistributedPipelineExtension(self)
    return self._distributed_pipeline

  def _lazy_init_fault_tolerance(self):
    """LACP Fault Tolerance Manager — lazy-initialized."""
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
    # Election weight must derive from the LACP composite resource score, not
    # the default 1.0 — compute it now that real hardware is known.
    self._refresh_election_weight()
    
    # Start TCP transport for binary tensor forwarding BEFORE the bridge, so
    # the bridge handshake can advertise its (ephemeral) port to peers.
    # Peers used to register our bridge port as the tensor address, so the
    # chunked-TCP path could never connect between real processes.
    try:
        from synapse.transport.tcp_transport import TensorTCPTransport
        # Reuse the bridge's TLS contexts so SYNAPSE_TLS covers tensors too.
        _bridge = getattr(self.libp2p_node, "_bridge", None)
        self.tcp_transport = TensorTCPTransport(
          node_id=self.id,
          ssl_server_ctx=getattr(_bridge, "_ssl_server_ctx", None),
          ssl_client_ctx=getattr(_bridge, "_ssl_client_ctx", None),
        )
        tcp_port = await self.tcp_transport.start()
        if _bridge is not None:
          _bridge.tcp_port = tcp_port
        if DEBUG >= 1:
            print(f"[TCP] Binary tensor transport listening on port {tcp_port}")
    except Exception as e:
        if DEBUG >= 1:
            print(f"[TCP] Failed to start binary transport: {e}")
        self.tcp_transport = None

    if DEBUG >= 1: print("Starting node: Starting Event Router and P2P Mesh...")
    await self.event_router.start()
    await self.libp2p_node.start()
    # INT8 wire payloads go only to peers whose handshake negotiated them.
    _bridge = getattr(self.libp2p_node, "_bridge", None)
    if _bridge is not None and hasattr(_bridge, "negotiated_features"):
      from synapse.transport.router import set_peer_features_lookup
      set_peer_features_lookup(self.id, _bridge.negotiated_features)
    await self.election_manager.start()
    self.election_manager.set_monitor(self.monitor)
    self._setup_p2p_rpc_handlers()

    await self.monitor.start()
    self._schedule_task(self._metrics_update_loop(), "realtime_metrics")
    
    if hasattr(self, "discovery") and self.discovery:
        if DEBUG >= 1: print(f"Starting discovery module: {type(self.discovery).__name__}")
        await self.discovery.start()
    
    if DEBUG >= 1: print(f"P2P Node started. Waiting for {wait_for_peers} peers...")
    if hasattr(self, "discovery") and self.discovery:
        if wait_for_peers > 0:
            await asyncio.sleep(2.0)  # Grace period for peer processes to bind sockets.
        await self.update_peers(wait_for_peers)
    if DEBUG >= 1: print(f"Peers updated: {len(self.peers)} peers")
    
    if DEBUG >= 1: print("Starting node: Collecting topology...")
    await self.collect_topology(set())
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    if DEBUG >= 1: print("Node started successfully!")
    self._schedule_task(self.periodic_topology_collection(2.0), "periodic_topology_collection")

    # Start zero-copy shared-memory transport for same-machine peers
    try:
        from synapse.transport.shm_transport import SharedMemoryTransport
        self.shm_transport = SharedMemoryTransport(node_id=self.id)
        await self.shm_transport.start()
        if DEBUG >= 1:
            print(f"[SHM] Zero-copy shared-memory transport ready (node {self.id})")
    except Exception as e:
        if DEBUG >= 1:
            print(f"[SHM] Failed to start shared-memory transport: {e}")
        self.shm_transport = None

    # T1.2: wire binary transports into process_tensor (receiver side).
    # Handlers are sync per the transport contract; fan out via _schedule_task.
    # SHM delivery is same-process-only (see transport/shm_transport); TCP
    # chunks reassemble in TensorTCPTransport before reaching the handler.
    try:
        if self.tcp_transport is not None:
            self.tcp_transport.set_tensor_handler(
                lambda msg: self._schedule_task(
                    self._on_transport_tensor(msg, "tcp"), "transport_tensor"))
        if self.shm_transport is not None:
            self.shm_transport.set_tensor_handler(
                lambda msg: self._schedule_task(
                    self._on_transport_tensor(msg, "shm"), "transport_tensor"))
    except Exception as e:
        if DEBUG >= 1:
            print(f"[TRANSPORT] handler wiring failed: {e}")

    # Start distributed pipeline extension unconditionally: rebalancing and
    # partition checks in check_fault_tolerance() gate on
    # self._distributed_pipeline, so a node that dies before the first
    # distributed request must still have the pipeline initialized (and its
    # ResultRouter started) to be rebalanced.
    if self._distributed_pipeline is None:
        from synapse.pipeline.node_extension import DistributedPipelineExtension
        self._distributed_pipeline = DistributedPipelineExtension(self)
    await self._distributed_pipeline.start()

    # Start fault tolerance manager
    self._lazy_init_fault_tolerance()
    if self.fault_tolerance:
        await self.fault_tolerance.start()
        # B5: exchange fault-tolerance heartbeats over the event mesh,
        # independently of discovery/membership, so liveness is tracked even
        # before peers join the topology (and a dead peer keeps being
        # rebalanced even if discovery never drops it).
        self.event_router.subscribe("synapse/cluster/heartbeat", self._on_ft_heartbeat)
        self._schedule_task(self._ft_heartbeat_loop(), "ft_heartbeat")
        # T3.3 (advisory-only): capability + demand gossip on the same mesh.
        # Old peers ignore these topics; handlers store with hygiene caps.
        try:
          from synapse.topology.capability import CAPABILITIES_TOPIC, DEMAND_TOPIC
          self._peer_capabilities = {}
          self._peer_demand = {}
          self.event_router.subscribe(CAPABILITIES_TOPIC, self._on_capability)
          self.event_router.subscribe(DEMAND_TOPIC, self._on_demand)
        except Exception:
          pass

    self.cluster_state.update_node(self.id, {
        "device": str(self.device_capabilities),
        "max_concurrent_requests": self.max_concurrent_requests,
        "has_inference_engine": self.inference_engine is not None,
    })
    await self._broadcast_cluster_state()
    self._schedule_task(self._cluster_state_heartbeat(), "cluster_state_heartbeat")

  async def _cluster_state_heartbeat(self) -> None:
    """Periodically update own node state and broadcast to cluster."""
    beats = 0
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
        # T3.3 advisory (default ON, observe-only): aggregate gossip demand
        # every ~6th beat so operators see the loop working live. Set
        # SYNAPSE_AUTOSCALE=off to silence. Advisory HINTS (not actions) are
        # evaluated only under SYNAPSE_AUTOSCALE=advise; actuating stays
        # gated on the C1 live gate — "on" currently means "advise".
        beats += 1
        if beats % 6 == 0:
            try:
                await self._log_demand_summary()
            except Exception:
                pass

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

    # Stop binary tensor transports (releases shared-memory segments)
    if self.shm_transport is not None:
        try:
            await self.shm_transport.stop()
        except Exception as e:
            if DEBUG >= 1:
                print(f"[SHM] Error stopping shared-memory transport: {e}")
        self.shm_transport = None
    if self.tcp_transport is not None:
        try:
            await self.tcp_transport.stop()
        except Exception as e:
            if DEBUG >= 1:
                print(f"[TCP] Error stopping binary transport: {e}")
        self.tcp_transport = None

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

  async def _ft_heartbeat_loop(self):
    """Periodically publish this node's fault-tolerance heartbeat over the
    event mesh (B5): peers record it regardless of topology membership."""
    beats = 0
    while True:
      try:
        interval = 2.0
        if self.fault_tolerance is not None:
          interval = self.fault_tolerance.HEARTBEAT_INTERVAL
        await asyncio.sleep(interval)
        if self.fault_tolerance is not None:
          await self.event_router.publish(
            "synapse/cluster/heartbeat",
            {"type": "heartbeat", "node_id": self.id},
            origin="local",
          )
        # T3.3: capability + demand every ~5th beat (~10 s, advisory-only).
        beats += 1
        if beats % 5 == 0:
          try:
            await self._publish_capability_demand()
          except Exception:
            pass
      except asyncio.CancelledError:
        break
      except Exception as e:
        if DEBUG >= 1:
          print(f"[FT_HEARTBEAT] publish failed: {e}")

  async def _publish_capability_demand(self) -> None:
    """Publish our capability + demand docs (T3.3, best-effort)."""
    from synapse.topology.capability import (
      CAPABILITIES_TOPIC, DEMAND_TOPIC, build_capability_doc, build_demand_doc)
    from synapse.helpers import VERSION as _VER
    eng = getattr(self, "inference_engine", None)
    model_ids = [getattr(eng, "_model_id", None)] if eng is not None else []
    model_ids = [m for m in model_ids if m]
    profiles = {}
    for mid in set(model_ids) | set(getattr(self, "_profile_models", set())):
      prof = await asyncio.get_running_loop().run_in_executor(
        None, self._local_layer_profile, mid)
      if prof:
        profiles[mid] = prof
    doc = build_capability_doc(
      self.id, getattr(self, "device_capabilities", None),
      model_ids=model_ids, version=_VER, layer_profiles=profiles)
    # Which layers we hold: failover prefers a node that already has them.
    loaded = getattr(eng, "loaded_layers", None)
    if model_ids and callable(loaded):
      doc["loaded_layers"] = {model_ids[0]: list(loaded())}
    await self.event_router.publish(CAPABILITIES_TOPIC, doc, origin="local")
    outstanding = getattr(self, "outstanding_requests", {})
    in_flight = len(outstanding) if hasattr(outstanding, "__len__") else 0
    maxc = getattr(self, "max_concurrent_requests", 32) or 32
    await self.event_router.publish(DEMAND_TOPIC, build_demand_doc(
      self.id, model_id=model_ids[0] if model_ids else "",
      req_per_min=None, queue_depth=max(0, in_flight - maxc),
      in_flight=in_flight), origin="local")

  async def _on_capability(self, event: Event):
    """Store a peer's capability doc (T3.3, bounded)."""
    try:
      from synapse.topology.capability import store_bounded
      data = event.data
      if isinstance(data, dict):
        nid = data.get("node_id")
        if nid and nid != self.id:
          store = getattr(self, "_peer_capabilities", None)
          if store is None:
            store = {}
            self._peer_capabilities = store
          store_bounded(store, nid, data)
    except Exception:
      pass

  async def _on_demand(self, event: Event):
    """Store a peer's demand doc (T3.3, bounded)."""
    try:
      from synapse.topology.capability import store_bounded
      data = event.data
      if isinstance(data, dict):
        nid = data.get("node_id")
        if nid and nid != self.id:
          store = getattr(self, "_peer_demand", None)
          if store is None:
            store = {}
            self._peer_demand = store
          store_bounded(store, nid, data)
    except Exception:
      pass

  async def _log_demand_summary(self) -> None:
    """Aggregate gossip demand and log it (T3.3, observe-only).

    SYNAPSE_AUTOSCALE=off silences this. =advise additionally evaluates
    advisory HINTS (never actions) using the pipeline truth that every
    partitioned node serves the one sharded model; standby = topology
    nodes outside the partition. Actuating ("on" meaning actions) stays
    gated on the C1 live gate — today "on" behaves as "advise".
    """
    import os as _os
    if _os.getenv("SYNAPSE_AUTOSCALE", "on") == "off":
      return
    try:
      from synapse.topology.capability import aggregate_demand
      from synapse.orchestration.autoscale import evaluate
      docs = list(getattr(self, "_peer_demand", {}).values())
      agg = aggregate_demand(docs)
      if DEBUG >= 1:
        print(f"[AUTOSCALE] demand summary: {agg}")
      if _os.getenv("SYNAPSE_AUTOSCALE", "on") not in ("advise", "on"):
        return
      try:
        parts = self._compute_lacp_partition()
        serving_nodes = [p.node_id for p in parts]
      except Exception:
        return
      topo_ids = list(getattr(self.topology, "nodes", {}).keys())
      standby = [nid for nid in topo_ids if nid not in serving_nodes]
      # Standby registry (T3.3): topology members holding zero layers.
      # Published in the summary so operators (and future actuating mode)
      # see promotion candidates; computation stays advisory-only.
      self._last_standby = standby
      models = {getattr(getattr(self.inference_engine, "shard", None),
                        "model_id", "") or ""}
      servers = {m: serving_nodes for m in models}
      hints = evaluate(agg, servers, standby)
      if hints and DEBUG >= 1:
        print(f"[AUTOSCALE] advisory hints (no action taken): {hints}")
      self._last_autoscale_hints = hints
      if DEBUG >= 1 and (standby or hints):
        print(f"[AUTOSCALE] serving={serving_nodes} standby={standby}")
    except Exception as e:
      if DEBUG >= 1:
        print(f"[AUTOSCALE] summary failed: {e}")

  async def _on_ft_heartbeat(self, event: Event):
    """Record a peer's fault-tolerance heartbeat (B5)."""
    try:
      data = event.data
      if isinstance(data, dict):
        nid = data.get("node_id")
        if nid and nid != self.id and self.fault_tolerance is not None:
          self.fault_tolerance.record_heartbeat(nid)
    except Exception as e:
      if DEBUG >= 1:
        print(f"[FT_HEARTBEAT] record failed: {e}")

  def _infer_total_layers(self, current_assignments) -> int:
    """Total transformer layers without hardcoding 32 (H4).

    Preference: loaded engine shard -> widest assigned range -> 32.
    """
    try:
      eng = getattr(self, "inference_engine", None)
      n = getattr(getattr(eng, "shard", None), "n_layers", None)
      if isinstance(n, int) and n > 0:
        return n
    except Exception:
      pass
    try:
      if current_assignments:
        widest = max(e for _, e in current_assignments.values())
        if isinstance(widest, int) and widest >= 0:
          return widest + 1
    except Exception:
      pass
    return 32

  async def check_fault_tolerance(self) -> None:
    """LACP: Dynamic Fault Tolerance with rebalancing on node failure."""
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

    # Phase 2: LACP node failure detection + rebalancing
    if self.fault_tolerance and self._distributed_pipeline:
      ft = self.fault_tolerance
      dead_nodes = ft.get_dead_nodes()
      # The monitor loop flags a node at its FIRST missed heartbeat
      # (_pending_rebalance) while blacklisting waits for 3 consecutive
      # misses; rebalance it now instead of waiting for the threshold (B5).
      if ft._pending_rebalance and ft._pending_rebalance not in dead_nodes:
        dead_nodes.append(ft._pending_rebalance)

      for dead_id in dead_nodes:
        is_pending = (dead_id == ft._pending_rebalance)
        if is_pending:
          ft._pending_rebalance = None  # consumed by this rebalance
        health = ft._health.get(dead_id)
        if health and (health.is_blacklisted or is_pending):
          if DEBUG >= 1: print(f"!!! [LACP_FAULT] Node {dead_id} is dead, triggering rebalance")

          current_assignments = {}
          partitions = self._compute_lacp_partition()
          # H4: never hardcode 32 — prefer the loaded shard's layer count,
          # else the widest assigned range, else 32 as last resort.
          total_layers = self._infer_total_layers(None)
          for p in partitions:
            num_layers = total_layers
            start = int(p.start * num_layers)
            end = min(int(p.end * num_layers) - 1, num_layers - 1)
            current_assignments[p.node_id] = (start, end)

          async def repartition(alive_nodes, total_layers):
            nodes_resources = {}
            for nid in alive_nodes:
              if nid in self.topology.nodes:
                nodes_resources[nid] = NodeResources.from_device_capabilities(self.topology.nodes[nid])
            latency_matrix = self._build_latency_matrix()
            apply_measured_bandwidth(nodes_resources, latency_matrix, self.id)
            result = compute_partition(
              nodes=nodes_resources,
              latency_matrix=latency_matrix,
              total_layers=total_layers,
              model_id="",
              method=self.partition_method,
              time_limit_seconds=self.ilp_time_limit_seconds,
            )
            new_assignments = {}
            for a in result.assignments:
              new_assignments[a.node_id] = (a.start_layer, a.end_layer)
            return new_assignments

          rebalance_result = await self.fault_tolerance.rebalance_after_failure(
            dead_node_id=dead_id,
            current_assignments=current_assignments,
            total_layers=total_layers,
            lacp_partition_fn=repartition,
          )
          # Consume the result instead of discarding it: the recovered
          # cross-node checkpoint is staged for the surviving node that takes
          # over the dead node's shard range, retrievable via
          # fault_tolerance.get_recovery_checkpoint(<target>).
          self._last_rebalance_result = rebalance_result
          if rebalance_result.recovered_from_checkpoint and rebalance_result.recovery_checkpoint:
            ckpt = rebalance_result.recovery_checkpoint
            target = rebalance_result.recovery_target_node
            if DEBUG >= 1:
              print(
                f"!!! [LACP_FAULT] Recovered checkpoint for {dead_id} "
                f"(step {ckpt.step}, layers {ckpt.start_layer}-{ckpt.end_layer}) "
                f"staged for node {target}"
              )
            # Hand the recovered on-disk checkpoint to the engine that takes
            # over the dead node's shard range, so training/inference resumes
            # from the last saved state instead of cold-starting.
            if ckpt.checkpoint_path and target:
              engine = getattr(self, "inference_engine", None)
              if engine is not None:
                shard = Shard(
                  model_id=ckpt.model_id,
                  start_layer=ckpt.start_layer,
                  end_layer=ckpt.end_layer,
                  n_layers=self._infer_total_layers(current_assignments),
                )
                # B1: the engine reports whether weights were REALLY restored.
                # A False here means metadata-only resume (degraded, not silent).
                # Prefer the shard layer-weights file; the JSON path alone is
                # metadata and can only ever yield a degraded resume.
                restored = await engine.load_checkpoint(
                  shard, getattr(ckpt, "weights_path", None) or ckpt.checkpoint_path)
                if not restored:
                  self._last_recovery_degraded = {
                    "dead_node_id": dead_id, "target": target,
                    "step": ckpt.step, "reason": "weights_not_restored",
                  }
                  if DEBUG >= 1:
                    print(
                      f"!!! [LACP_FAULT] Recovery DEGRADED for {dead_id}: "
                      f"checkpoint step {ckpt.step} staged for {target} but "
                      f"weights were NOT restored (metadata-only resume)"
                    )
                else:
                  self._last_recovery_degraded = None

  async def on_p2p_tensor(self, event: Event):
    """Handle incoming tensor for inference via P2P."""
    data = event.data
    
    # Check for new binary format (LACP High-Perf)
    if isinstance(data, dict) and "bin" in data:
        tensor_data, metadata = decode_tensor(data["bin"])
        request_id = metadata.get("request_id")
        shard = Shard.from_dict(metadata.get("shard"))
        inference_state = metadata.get("inference_state")
        orchestrator_id = metadata.get("_orchestrator_id")
        
        # Adaptive LACP: Calculate Jitter and inform cluster if rebalance needed
        sent_at = metadata.get("_sent_at", 0)
        if sent_at > 0:
            latency = time.time() - sent_at
            if latency > self._adaptive_rebalance_threshold:
                if DEBUG >= 1: print(f"[ADAPTIVE] Latency spike: {latency:.2f}s! Peer may be lagging.")
                # We could broadcast a "rebalance hint" here
    else:
        # Legacy/JSON format fallback
        request_id = data.get("request_id")
        tensor_data_raw = data.get("tensor_data")
        shape = data.get("shape")
        dtype = data.get("dtype")

        if tensor_data_raw is not None and shape is not None:
            import numpy as np
            if isinstance(tensor_data_raw, bytes):
                tensor_data = np.frombuffer(tensor_data_raw, dtype=np.dtype(dtype))
            elif isinstance(tensor_data_raw, str):
                tensor_data = np.frombuffer(bytes.fromhex(tensor_data_raw), dtype=np.dtype(dtype))
            else:
                tensor_data = np.frombuffer(tensor_data_raw, dtype=np.dtype(dtype))
            tensor_data = tensor_data.reshape(shape)
        else:
            tensor_data = np.array(data.get("tensor"))
            if dtype:
                tensor_data = tensor_data.astype(np.dtype(dtype))

        shard_dict = data.get("shard")
        shard = Shard.from_dict(shard_dict)
        inference_state = data.get("inference_state")
        orchestrator_id = data.get("_orchestrator_id")

    # Run EXACTLY the shard the sender assigned: recomputing it from our
    # local partition view let two nodes with a different view both run
    # layers 0..k and bounce hidden states back and forth.
    await self.process_tensor(shard, tensor_data, request_id, inference_state,
                              orchestrator_id=orchestrator_id, exact_shard=True)
    await self._ack_peer(data if isinstance(data, dict) else metadata)

  async def _ack_peer(self, data) -> None:
    """C-W1: answer P2PPeerHandle ack waits; absent key = fire-and-forget."""
    try:
      topic = data.get("_ack_topic") if isinstance(data, dict) else None
      if topic:
        await self.event_router.publish(topic, True, origin=self.id)
    except Exception:
      pass

  def _decode_p2p_array(self, raw, shape, dtype):
    """Decode a peer-sent array (bytes post-bridge, hex str, or list)."""
    import numpy as np
    if isinstance(raw, (bytes, bytearray)):
      buf = bytes(raw)
    elif isinstance(raw, str):
      buf = bytes.fromhex(raw)
    else:
      return np.array(raw, dtype=np.dtype(dtype)).reshape(shape)
    return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape)

  async def on_p2p_example(self, event: Event):
    """C-W4: route a peer-sent training example into enqueue_example."""
    data = event.data if isinstance(event.data, dict) else {}
    try:
      shard = Shard.from_dict(data.get("shard"))
      example = self._decode_p2p_array(data.get("example"), data.get("example_shape"), data.get("example_dtype"))
      target = self._decode_p2p_array(data.get("target"), data.get("target_shape"), data.get("target_dtype"))
      length = self._decode_p2p_array(data.get("length"), data.get("length_shape"), data.get("length_dtype"))
      await self.enqueue_example(shard, example, target, length,
                                 bool(data.get("train", False)), data.get("request_id"))
    except Exception as e:
      if DEBUG >= 1:
        print(f"[P2P] on_p2p_example failed: {e}")

  async def on_p2p_status_direct(self, event: Event):
    """C-W4: route a peer-sent opaque status into on_node_status."""
    try:
      data = event.data if isinstance(event.data, dict) else {}
      request_id = data.get("request_id", "")
      status = data.get("status", "")
      node_id = getattr(event, "origin", "") or ""
      if not isinstance(status, str):
        status = json.dumps(status)
      self.on_node_status(request_id, {"type": "node_status", "node_id": node_id, "status": status})
    except Exception as e:
      if DEBUG >= 1:
        print(f"[P2P] on_p2p_status_direct failed: {e}")

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
    await self._ack_peer(data)

  async def on_p2p_result(self, event: Event):
    """Handle incoming inference results/tokens via P2P."""
    data = event.data
    request_id = data.get("request_id")
    tokens = data.get("tokens", [])
    is_finished = data.get("is_finished", False)
    if data.get("verify") or data.get("kv_miss"):
      # Raw per-position samples (incl. unverified draft positions) for the
      # orchestrator's decode loop only; it streams the verified tokens.
      return

    # The token buffer is fed by the verified decode-loop callback
    # (node_extension) or by the sampling node's direct append — the P2P
    # result is a redundant echo of those. Only the UI/stream callbacks fire
    # here, so a legitimate adjacent repeated token is never dropped and a
    # raw unverified guess never double-appends.
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
    
    # Fallback nếu không có discovery hoặc discovery không phải Tailscale.
    # C-W10: canonical env names via helpers (TAILSCALE_API_KEY/TS_API_KEY,
    # TAILSCALE_TAILNET/TS_TAILNET/TAILNET_NAME legacy) — same as backend main.
    if not api_key:
      from synapse.helpers import get_tailscale_env
      _env_key, _env_tailnet = get_tailscale_env()
      api_key = api_key or _env_key
    if not tailnet:
      from synapse.helpers import get_tailscale_env
      _env_key, _env_tailnet = get_tailscale_env()
      tailnet = tailnet or _env_tailnet
      
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

    # Orchestrator-driven decode (distributed path): sample one token per
    # requested position and hand them back; the orchestrator verifies the
    # draft, applies stop rules and streams. No per-request buffer here.
    if shard.is_last_layer() and isinstance(inference_state, dict)         and inference_state.get("verify_n") is not None:
      await self._publish_verify_result(shard, result, request_id, inference_state, orchestrator_id)
      return None

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
        
        # Per-request temperature travels in inference_state (set by the API
        # from the request body); the node default applies otherwise.
        temp = self.default_sample_temperature
        if inference_state and inference_state.get("temperature") is not None:
          temp = float(inference_state["temperature"])
        token = await self.inference_engine.sample(result, temp=temp, generated_ids=buf if buf else None)
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
              temp=temp,
              generated_ids=buf if buf else None,
            )
            token = retry_token
            token_value = token.item()
          except Exception as e:
            if DEBUG >= 1: print(f"[sample] retry sampling error: {e}")

        # Add to buffer
        self.buffered_token_output[request_id][0].append(token_value)
        # This node is the sampling stage: subsequent deliveries of these
        # tokens (own stream trigger / P2P loopback) are echoes.
        self._locally_sampled.add(request_id)
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
        target=self._next_stage_from_plan(shard, inference_state),
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
            "is_finished": is_finished,
            "micro_batch_id": inference_state.get("microbatch_id", 0) if inference_state else 0,
            "source_stage_id": shard.start_layer,
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
        self._locally_sampled.discard(request_id)
    
        return np.array(self.buffered_token_output.get(request_id, ([], False))[0])


  def _stop_token_ids(self) -> set:
    ids = {151643, 151644, 151645}  # Qwen chat stop tokens
    tok = getattr(self.inference_engine, "tokenizer", None)
    eos = getattr(tok, "eos_token_id", None) if tok is not None else None
    if eos is None and tok is not None and hasattr(tok, "_tokenizer"):
      eos = getattr(tok._tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
      ids.add(eos)
    return ids

  async def _publish_verify_result(self, shard, result, request_id, state, orchestrator_id) -> None:
    """Sample the last `verify_n` positions (position j conditioned on the
    history + draft[:j], same repetition penalty as sequential decoding) and
    publish all of them. With a deterministic draft, "sample, then accept
    while the sample equals the guess" is exact speculative sampling."""
    n = int(state.get("verify_n") or 1)
    draft = [int(t) for t in (state.get("draft") or [])]
    gen = [int(t) for t in (state.get("gen_ids") or [])]
    temp = self.default_sample_temperature
    if state.get("temperature") is not None:
      temp = float(state["temperature"])
    arr = np.asarray(result)
    rows = arr.reshape(-1, arr.shape[-1]) if arr.size else arr.reshape(0, 1)
    if rows.shape[0] < n:
      # Fewer output rows than draft positions would shift every sampled
      # token by the missing count: never answer, ask for a full resend.
      await self._publish_kv_miss(shard, request_id, state, orchestrator_id)
      return
    stop_ids = self._stop_token_ids()
    tokens: List[int] = []
    for j in range(n):
      row = rows[rows.shape[0] - n + j][None, None, :]
      hist = gen + draft[:j]
      tok = int(np.asarray(await self.inference_engine.sample(
        row, temp=temp, generated_ids=hist or None)).reshape(-1)[0])
      if not hist and tok in stop_ids:
        # Same rule as the sequential path: never end on the first token.
        masked = np.array(row, copy=True)
        for sid in stop_ids:
          if 0 <= sid < masked.shape[-1]:
            masked[..., sid] = -1e9
        tok = int(np.asarray(await self.inference_engine.sample(
          masked, temp=temp, generated_ids=None)).reshape(-1)[0])
      tokens.append(tok)
    payload = {
      "request_id": request_id,
      "tokens": tokens,
      "is_finished": False,
      "verify": True,
      "micro_batch_id": state.get("microbatch_id", 0),
      "source_stage_id": shard.start_layer,
    }
    await self.event_router.publish(f"synapse/inference/result/{request_id}", payload)
    if orchestrator_id and orchestrator_id != self.id:
      await self.event_router.publish(f"synapse/inference/result/{orchestrator_id}", payload)

  async def _publish_kv_miss(self, shard, request_id, state, orchestrator_id) -> None:
    """A stage lacks the cached prefix it was asked to extend: tell the
    orchestrator to resend this step with full hidden states."""
    payload = {
      "request_id": request_id, "tokens": [], "is_finished": False,
      "kv_miss": True, "node_id": self.id,
      "micro_batch_id": (state or {}).get("microbatch_id", 0),
      "source_stage_id": None,
    }
    await self.event_router.publish(f"synapse/inference/result/{request_id}", payload)
    if orchestrator_id and orchestrator_id != self.id:
      await self.event_router.publish(f"synapse/inference/result/{orchestrator_id}", payload)

  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
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

    # LACP: Use distributed pipeline when multiple nodes are available
    node_count = len(self.topology.nodes) if self.topology.nodes else 0
    if node_count > 1 and self.is_orchestrator_for_request(request_id):
      if DEBUG >= 1: print(f"[{request_id}] LACP: Using distributed pipeline across {node_count} nodes")
      self.distributed_active = True
      self.distributed_node_count = node_count
      self.distributed_status = "distributed"
      try:
        # B4: the master computes and broadcasts the authoritative partition
        # before serving, so every node agrees on shard assignment instead of
        # each recomputing the LACP partition locally on every forward.
        if self.election_manager.is_master():
          await self.broadcast_authoritative_partition(base_shard)
        tokens = await self.distributed_pipeline.process_prompt_distributed(
          base_shard=base_shard,
          prompt=prompt,
          request_id=request_id,
          max_generate_tokens=max_generate_tokens,
          num_micro_batches=self.num_micro_batches,
          sampling_state={k: v for k, v in (inference_state or {}).items()
                          if k in ("temperature",)},
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
          self._on_token_received(request_id, [], True)
          self.trigger_on_token_callbacks(request_id, [], True)
        raise
      finally:
        # Delete local tensor references explicitly to free PyTorch graphs
        if 'latest_result' in locals(): del latest_result
        if 'last_token' in locals(): del last_token
        if 'buffered_tokens' in locals(): del buffered_tokens
        if 'result' in locals(): del result
        self.outstanding_requests.pop(request_id, None)
        self._locally_sampled.discard(request_id)
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
    exact_shard: bool = False,
  ) -> Optional[np.ndarray]:
    shard = base_shard if exact_shard else self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id, inference_state,
                                      orchestrator_id=orchestrator_id, exact_shard=True)
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
    exact_shard: bool = False,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = base_shard if exact_shard else self.get_current_shard(base_shard)

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
      try:
        result, inference_state = await self.inference_engine.infer_tensor(request_id, shard, input_data, inference_state)
      except Exception as e:
        if type(e).__name__ == "KVMissError":
          self.outstanding_requests.pop(request_id, None)
          if DEBUG >= 1: print(f"[{request_id}] kv miss on {shard}: {e}")
          await self._publish_kv_miss(shard, request_id, inference_state,
                                      orchestrator_id or self._request_orchestrator_map.get(request_id))
          return None
        raise
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
  
  def _next_stage_from_plan(self, shard: Shard, inference_state: Optional[dict]):
    """(node_id, Shard) owning layer shard.end_layer + 1 in the request's
    partition_plan, or None when the request carries no plan."""
    plan = (inference_state or {}).get("partition_plan") if isinstance(inference_state, dict) else None
    if not plan:
      return None
    nxt = shard.end_layer + 1
    for entry in plan:
      try:
        nid, start, end, n = entry[0], int(entry[1]), int(entry[2]), int(entry[3])
      except (TypeError, ValueError, IndexError):
        return None
      if start == nxt:
        return nid, Shard(shard.model_id, start, end, n)
    return None

  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
    target: Optional[Tuple[str, Shard]] = None,
  ) -> None:
    if target is not None:
      target_id, target_shard = target
    else:
      if DEBUG >= 1: print(f"target partition index: {target_index}")
      target_id = self._compute_lacp_partition(base_shard)[target_index].node_id
      target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Forward target: {target_id} shard {target_shard}")

    orchestrator_id = self._orchestrator_id_for_request(request_id)

    if target_id == self.id:
      return await self.process_tensor(target_shard, tensor, request_id, inference_state,
                                       orchestrator_id=orchestrator_id, exact_shard=True)
    else:
      start_send = time.perf_counter()
      # Per-row INT8 by default, only toward peers that negotiated it
      # (SYNAPSE_TRANSPORT_INT8=0 = exact dtype).
      try:
        from synapse.transport.router import should_quantize as _should_q
        quantize = _should_q(str(tensor.dtype), target_id=target_id, self_id=self.id)
      except Exception:
        quantize = False
      if DEBUG >= 2:
        print(f"[P2P_MESH] Publishing work to Node: {target_id} (quantize_int8={quantize})")

      encoded_payload = encode_tensor(
        tensor,
        metadata={
          "request_id": request_id,
          "shard": target_shard.to_dict(),
          "inference_state": inference_state,
          "_orchestrator_id": orchestrator_id,
          "_sent_at": time.time(),
        },
        quantize_int8=quantize
      )

      # T1.2: auto-route (opt-in SYNAPSE_CHUNKED_TCP=1 for tcp-chunked/shm;
      # bridge legacy stays the default). Every decision is counted (B3);
      # a failed fast path falls through to the bridge, never silent loss.
      routed = False
      try:
        from synapse.transport.router import choose_transport as _choose
        _addr = None
        try:
          for _p in (self.peers or []):
            if _p.id() == target_id:
              _addr = _p.addr()
              break
        except Exception:
          _addr = None
        _tcp = getattr(self, "tcp_transport", None)
        _shm = getattr(self, "shm_transport", None)
        _tcp_addr = self._peer_tcp_addr(target_id, _addr)
        if _tcp is not None and _tcp_addr and _tcp_addr not in _tcp._peer_map:
          _tcp.register_peer(_tcp_addr, target_id)
        _route, _reason = _choose(
          self.id, target_id, _addr, len(encoded_payload),
          # SHM delivers only within one process (in-process notifier); a
          # same-host peer in another process always failed -> bridge.
          shm_registered=bool(_shm is not None and _addr and _shm.can_deliver(_addr)),
          tcp_registered=bool(_tcp is not None and _tcp_addr and _tcp_addr in _tcp._peer_map),
        )
        if _route == "tcp-chunked" and _tcp_addr is not None and _tcp is not None:
          routed = await _tcp.send_blob_chunked(_tcp_addr, encoded_payload)
          self._record_transport("tcp-chunked" if routed else "tcp-chunked:fallback")
        elif _route == "shm" and _addr is not None and _shm is not None:
          routed = await _shm.send_tensor(
            request_id, int(target_shard.start_layer), tensor, _addr,
            metadata={
              "shard": target_shard.to_dict(),
              "inference_state": inference_state,
              "_orchestrator_id": orchestrator_id,
              "micro_batch_id": 0,
              "source_stage": int(target_shard.start_layer),
            },
          )
          self._record_transport("shm" if routed else "shm:fallback")
        else:
          self._record_transport("bridge")
      except Exception as e:
        self._record_transport("bridge:router-error")
        if DEBUG >= 1:
          print(f"[TRANSPORT] Router error, using bridge: {e}")

      if not routed:
        await self.event_router.publish(f"synapse/inference/tensor/{target_id}", {"bin": encoded_payload})
      
      elapsed = time.perf_counter() - start_send
      if target_id not in self._last_forward_latencies:
          self._last_forward_latencies[target_id] = elapsed
      else:
          self._last_forward_latencies[target_id] = 0.7 * self._last_forward_latencies[target_id] + 0.3 * elapsed
          
      return None

  def _transport_counter(self):
    """Lazy transport_selected counters (B3 observability, no __init__ churn)."""
    counts = getattr(self, "_transport_counts", None)
    if counts is None:
      counts = {}
      self._transport_counts = counts
    return counts

  def _record_transport(self, route: str) -> None:
    try:
      counts = self._transport_counter()
      counts[route] = counts.get(route, 0) + 1
    except Exception:
      pass

  async def _on_transport_tensor(self, msg, via: str) -> None:
    """Route a TensorMessage from binary transports into process_tensor.

    The SYNP metadata carries the sender's shard dict (see forward_tensor);
    SHM senders must include the same "shard" key in metadata. Messages
    without a shard are dropped explicitly and counted (never silent).
    """
    try:
      meta = msg.metadata or {}
      shard_d = meta.get("shard")
      if not isinstance(shard_d, dict):
        self._record_transport(via + ":drop-no-shard")
        if DEBUG >= 1:
          print(f"[TRANSPORT] Dropping {via} tensor without shard metadata "
                f"(request {getattr(msg, 'request_id', '?')})")
        return
      base_shard = Shard.from_dict(shard_d)
      self._record_transport(via)
      await self.process_tensor(
        base_shard,
        msg.tensor,
        meta.get("request_id", getattr(msg, "request_id", None)),
        meta.get("inference_state"),
        orchestrator_id=meta.get("_orchestrator_id", getattr(msg, "orchestrator_id", None)),
        exact_shard=True,  # same rule as on_p2p_tensor: run the assigned shard
      )
    except Exception as e:
      self._record_transport(via + ":error")
      if DEBUG >= 1:
        print(f"[TRANSPORT] Error routing {via} tensor: {e}")

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
      def _pretty(ps):
        return [f"{p.id()}@{p.addr()}" for p in ps]
      if successful_disconnects: print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
      if failed_disconnects: print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
      if successful_connects: print(f"Successfully connected peers: {_pretty(successful_connects)}")
      if failed_connects: print(f"Failed to connect peers: {_pretty(failed_connects)}")

    self.peers = next_peers
    self._register_transport_peers()
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  def _peer_tcp_addr(self, peer_id: str, bridge_addr: Optional[str]) -> Optional[str]:
    """host:tcp_port of a peer's tensor transport (from its handshake)."""
    bridge = getattr(self.libp2p_node, "_bridge", None)
    info = getattr(bridge, "_peer_handshakes", {}).get(peer_id) if bridge else None
    port = (info or {}).get("tcp_port")
    if not port or not bridge_addr or ":" not in bridge_addr:
      return None
    return f"{bridge_addr.rsplit(':', 1)[0]}:{int(port)}"

  def _register_transport_peers(self):
    """Register every peer on the binary transports.

    The TCP transport registers all peers; same-machine peers are additionally
    registered on the zero-copy shared-memory transport so node-to-node tensor
    traffic between co-located nodes can bypass TCP entirely.
    """
    import socket as _socket
    from synapse.transport.shm_transport import is_same_machine

    own_addr = f"{_socket.gethostname()}:0"
    for peer in self.peers or []:
      try:
        addr = peer.addr()
        peer_id = peer.id()
      except Exception:
        continue
      if self.tcp_transport is not None:
        # The peer's tensor transport listens on its own (ephemeral) port,
        # advertised in the bridge handshake; its bridge port is not it.
        tcp_addr = self._peer_tcp_addr(peer_id, addr)
        if tcp_addr:
          try:
            self.tcp_transport.register_peer(tcp_addr, peer_id)
          except Exception:
            pass
      if self.shm_transport is not None and is_same_machine(addr, own_addr):
        try:
          self.shm_transport.register_peer(addr, peer_id)
        except Exception:
          pass

  async def select_best_inference_engine(self):
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)
      # The distributed pipeline extension is initialized unconditionally at
      # start(); keep its controller forwarding with the current engine.
      if self._distributed_pipeline is not None:
        self._distributed_pipeline.sync_inference_engine()

  async def periodic_topology_collection(self, interval: int):
    while True:
      await asyncio.sleep(interval)
      try:
        did_peers_change = await self.update_peers()
        if DEBUG >= 3: print(f"{did_peers_change=}")
        
        # LACP: Record heartbeat for all connected peers
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
    # Cluster composition changed: refresh the election weight so a stronger
    # node's master weight reflects the LACP composite score over the cluster.
    self._refresh_election_weight()
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

  def _election_resource_scorer(self, node_id: str) -> float:
    """
    LACP composite resource score for master-election weighting.

    Computes cluster-relative normalized scores over the nodes this node
    currently knows about (same topology view every node converges on), so a
    stronger node gets a higher election weight. Falls back to the node's own
    device capabilities when the node is not yet in the topology.
    """
    try:
      nodes_resources = {}
      for nid, dc in self.topology.nodes.items():
        nr = NodeResources.from_device_capabilities(dc)
        if nid == self.id:
          m = self._current_metrics
          nr.gpu_util_pct = m.gpu_util_pct
          nr.system_ram_used_mb = m.ram_used_mb
          nr.current_power_w = m.gpu_power_w
        nodes_resources[nid] = nr
      if node_id not in nodes_resources:
        nodes_resources[node_id] = NodeResources.from_device_capabilities(
          self.device_capabilities
        )
      latency_matrix = self._build_latency_matrix()
      apply_measured_bandwidth(nodes_resources, latency_matrix, self.id)
      scores = self._resource_scorer.compute_scores(nodes_resources)
      return scores.get(node_id, (0.0, None))[0]
    except Exception:
      return 0.0

  def _refresh_election_weight(self) -> None:
    """
    Recompute the election base weight from current hardware + LACP composite.
    Called when hardware is detected and when the cluster topology changes.
    """
    try:
      vram_gb = float(getattr(self.device_capabilities, "memory", 0) or 0) / 1024.0
      flops = getattr(self.device_capabilities, "flops", None)
      fp32_flops = float(getattr(flops, "fp32", 0) or 0)
      self.election_manager.set_base_compute_weight(vram_gb, fp32_flops)
    except Exception:
      pass

  def _node_alive(self, node_id: str) -> bool:
    """Heartbeat view of a node (used to detect a dead stage mid-request).

    Unknown to the fault-tolerance monitor = alive (no false failovers)."""
    if node_id == self.id:
      return True
    ft = getattr(self, "fault_tolerance", None)
    health = getattr(ft, "_health", {}).get(node_id) if ft is not None else None
    if health is None:
      return True
    if getattr(health, "is_blacklisted", False):
      return False
    last = getattr(health, "last_heartbeat", None)
    interval = getattr(ft, "HEARTBEAT_INTERVAL", 2.0)
    return not last or (time.time() - last) <= 3 * interval + 1.0

  def _layers_held(self, node_id: str, model_id: str) -> set:
    """Decoder layers a node has in memory (own engine, or peer gossip)."""
    if node_id == self.id:
      eng = self.inference_engine
      if getattr(eng, "_model_id", None) != model_id:
        return set()
      fn = getattr(eng, "loaded_layers", None)
      return set(fn()) if callable(fn) else set()
    doc = (getattr(self, "_peer_capabilities", {}) or {}).get(node_id) or {}
    return set((doc.get("loaded_layers") or {}).get(model_id) or [])

  def _replan_excluding(self, base_shard: Shard, current, exclude: set):
    """New stage plan for a running request without the `exclude` nodes.

    Patch first (Petals-style): each dead stage's range goes to a live node,
    preferring one that already HOLDS those layers (a replica), then a
    neighbouring stage (the ranges merge), within its memory cap. Other
    stages keep their KV caches. If no patch fits, a fresh placement over
    the surviving nodes."""
    model_id, n = base_shard.model_id, base_shard.n_layers
    alive = [nid for nid in self.topology.nodes.keys()
             if nid not in exclude and self._node_alive(nid)]
    if not alive:
      return None
    stages = [(nid, sh.start_layer, sh.end_layer) for nid, sh in current]
    caps = {}
    for nid in alive:
      prof = (self._local_layer_profile(model_id) if nid == self.id else
              ((getattr(self, "_peer_capabilities", {}).get(nid) or {}).get("layer_profiles") or {}).get(model_id))
      caps[nid] = int(prof["max_layers"]) if prof and prof.get("max_layers") else n
    patched = []
    ok = True
    for i, (nid, a, b) in enumerate(stages):
      if nid not in exclude:
        patched.append([nid, a, b])
        continue
      need = set(range(a, b + 1))
      used = lambda x: sum(bb - aa + 1 for m, aa, bb in patched if m == x) + \
        sum(bb - aa + 1 for m, aa, bb in stages[i + 1:] if m == x)
      neighbours = {patched[-1][0] if patched else None,
                    stages[i + 1][0] if i + 1 < len(stages) else None}
      cands = [c for c in alive if used(c) + (b - a + 1) <= caps.get(c, n)]
      if not cands:
        ok = False
        break
      cands.sort(key=lambda c: (not need <= self._layers_held(c, model_id),
                                c not in neighbours, used(c)))
      patched.append([cands[0], a, b])
    if ok:
      merged = []
      for nid, a, b in patched:
        if merged and merged[-1][0] == nid and merged[-1][2] + 1 == a:
          merged[-1][2] = b
        else:
          merged.append([nid, a, b])
      # a node may appear once in a chain (one KV cache per range is fine,
      # but a loop back to an earlier node would need two forwards per hop)
      if len({m[0] for m in merged}) == len(merged):
        return [(nid, Shard(model_id, a, b, n)) for nid, a, b in merged]
    saved = self.topology
    try:
      from synapse.topology.topology import Topology
      sub = Topology()
      for nid in alive:
        sub.update_node(nid, saved.nodes[nid])
      self.topology = sub
      self._last_placement = None
      parts = self._compute_lacp_partition(base_shard)
    finally:
      self.topology = saved
    from synapse.topology.partitioning_strategy import map_partitions_to_shards
    shards = map_partitions_to_shards(parts, n, model_id)
    return [(p.node_id, shards[i]) for i, p in enumerate(parts) if i < len(shards)]

  async def _warm_replica(self, base_shard: Shard, partitions) -> None:
    """SYNAPSE_REPLICAS=1: also hold the NEXT stage's layers (if they fit),
    so when that node dies the patch plan finds a replica already in memory
    instead of loading weights mid-request."""
    import os as _os
    if _os.getenv("SYNAPSE_REPLICAS", "0") != "1" or not partitions:
      return
    try:
      from synapse.topology.partitioning_strategy import map_partitions_to_shards
      n = base_shard.n_layers
      shards = map_partitions_to_shards(partitions, n, base_shard.model_id)
      ids = [p.node_id for p in partitions]
      if self.id not in ids or len(ids) < 2:
        return
      i = ids.index(self.id)
      succ = shards[(i + 1) % len(shards)]
      mine = shards[i]
      prof = self._local_layer_profile(base_shard.model_id) or {}
      if (mine.end_layer - mine.start_layer + 1) + (succ.end_layer - succ.start_layer + 1) \
          > int(prof.get("max_layers") or 0):
        return
      await self.inference_engine.ensure_shard(succ)
      if DEBUG >= 1:
        print(f"[REPLICA] {self.id} holds layers {succ.start_layer}-{succ.end_layer} of {ids[(i + 1) % len(ids)]}")
    except Exception as e:
      if DEBUG >= 1: print(f"[REPLICA] warm-up failed: {e}")

  def _note_profile_model(self, model_id: Optional[str]) -> None:
    if model_id:
      if not hasattr(self, "_profile_models"):
        self._profile_models = set()
      self._profile_models.add(model_id)

  def _local_layer_profile(self, model_id: str):
    """This node's measured layer cost for model_id (cached ~30 s)."""
    cache = getattr(self, "_profile_cache", None)
    if cache is None:
      cache = self._profile_cache = {}
    hit = cache.get(model_id)
    if hit and time.time() - hit.get("ts", 0) < 30.0:
      return hit
    try:
      from synapse.topology.layer_profile import local_profile
      prof = local_profile(self.inference_engine, model_id)
    except Exception as e:
      if DEBUG >= 1: print(f"[PROFILE] {model_id}: {e}")
      prof = None
    if prof:
      cache[model_id] = prof
    return prof

  def _measured_placement(self, base_shard: Optional[Shard]) -> Optional[List[Partition]]:
    """DP placement from measured ms/layer, memory caps and hop times.

    Needs a profile from EVERY topology node for this model (own + gossip);
    otherwise None and the caller uses the legacy weighting. Keeps the
    previous placement unless the new one is >= 10% cheaper under the
    current measurements (no churn from measurement noise).
    SYNAPSE_PARTITIONER=legacy disables it; SYNAPSE_PARTITION_OBJECTIVE =
    throughput (default) | latency."""
    import os as _os
    if base_shard is None or _os.getenv("SYNAPSE_PARTITIONER", "measured") == "legacy":
      return None
    from synapse.topology.dp_partition import NodeProfile, plan, _dp_for_order
    model_id, total = base_shard.model_id, base_shard.n_layers
    if not model_id or total <= 0:
      return None
    self._note_profile_model(model_id)
    profiles = []
    peers = getattr(self, "_peer_capabilities", {}) or {}
    for nid in self.topology.nodes.keys():
      if nid == self.id:
        prof = self._local_layer_profile(model_id)
      else:
        prof = ((peers.get(nid) or {}).get("layer_profiles") or {}).get(model_id)
      if not prof or not prof.get("ms_per_layer") or not prof.get("max_layers"):
        return None
      profiles.append(NodeProfile(nid, float(prof["ms_per_layer"]), int(prof["max_layers"])))
    rtt = {}
    for peer, ms in getattr(self, "_measured_rtt", {}).items():
      rtt[(self.id, peer)] = ms / 2.0  # one-way hop
    objective = getattr(self, "partition_objective", None) or       _os.getenv("SYNAPSE_PARTITION_OBJECTIVE", "throughput")
    placement = plan(profiles, total, rtt, objective)
    if placement is None:
      return None
    prev = getattr(self, "_last_placement", None)
    if prev is not None and prev.get("model_id") == model_id and prev.get("total") == total \
        and all(nid in self.topology.nodes for nid, _, _ in prev["stages"]):
      by_id = {p.node_id: p for p in profiles}
      order = [by_id[nid] for nid, _, _ in prev["stages"]]
      stages = prev["stages"]
      # current cost of the previous placement
      cost = 0.0
      for i, (nid, a, b) in enumerate(stages):
        stage = (b - a + 1) * by_id[nid].ms_per_layer
        if i:
          stage += rtt.get((stages[i - 1][0], nid), rtt.get((nid, stages[i - 1][0]), 2.0))
        cost = cost + stage if objective == "latency" else max(cost, stage)
      fits = all(b - a + 1 <= by_id[nid].max_layers for nid, a, b in stages)
      if fits and placement.cost_ms >= 0.9 * cost:
        return [Partition(node_id=nid, start=a / total, end=(b + 1) / total) for nid, a, b in stages]
    self._last_placement = {"model_id": model_id, "total": total, "stages": placement.stages,
                            "cost_ms": placement.cost_ms, "objective": objective,
                            "profiles": {p.node_id: {"ms_per_layer": p.ms_per_layer,
                                                     "max_layers": p.max_layers} for p in profiles}}
    if DEBUG >= 1:
      print(f"[PLACEMENT] {objective}: {placement.stages} cost={placement.cost_ms:.2f}ms")
    return [Partition(node_id=nid, start=a / total, end=(b + 1) / total)
            for nid, a, b in placement.stages]

  def _compute_lacp_partition(self, base_shard: Optional[Shard] = None) -> List[Partition]:
    """Measured DP placement when every node has reported its layer cost;
    else TFLOPS-proportional (default) or LACP multi-resource scoring."""
    num_layers = base_shard.n_layers if base_shard and base_shard.n_layers > 0 else 32
    model_id = base_shard.model_id if base_shard else ""
    try:
      measured = self._measured_placement(base_shard)
    except Exception as e:
      measured = None
      if DEBUG >= 1: print(f"[PLACEMENT] measured placement failed: {e}")
    if measured:
      return measured

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

    # Propagate measured network bandwidth into resource scores so the
    # partitioner weights real links, not the 1000 Mbps placeholder.
    apply_measured_bandwidth(nodes_resources, latency_matrix, self.id)

    lacp_result = compute_partition(
        nodes=nodes_resources,
        latency_matrix=latency_matrix,
        total_layers=num_layers,
        model_id=model_id,
        method=self.partition_method,
        time_limit_seconds=self.ilp_time_limit_seconds,
    )

    partitions = []
    for a in lacp_result.assignments:
        partitions.append(Partition(
            node_id=a.node_id,
            start=a.start_layer / num_layers,
            end=(a.end_layer + 1) / num_layers,
        ))
    return partitions

  def _get_authoritative_partitions(self, base_shard: Optional[Shard] = None) -> Optional[List[Partition]]:
    """Return the master's broadcast partition when it still matches the
    requested base shard, else None (caller recomputes locally)."""
    parts = self._authoritative_partitions
    if parts is None:
      return None
    cached = self._authoritative_base_shard
    if base_shard is None or cached is None:
      return parts
    if (cached.model_id == base_shard.model_id
        and cached.n_layers == base_shard.n_layers
        and cached.start_layer == base_shard.start_layer
        and cached.end_layer == base_shard.end_layer):
      return parts
    return None

  def _build_latency_matrix(self) -> Optional[LatencyMatrix]:
    """
    Build latency matrix for LACP partitioning.
    Uses real-time EWMA forward latencies (Adaptive LACP) to react to network jitter.
    Returns a LatencyMatrix (the type compute_partition expects), or None when
    fewer than two nodes are present.
    """
    node_ids = list(self.topology.nodes.keys())
    if len(node_ids) < 2:
      return None

    # Rebuild the cache when the node set changes: a cache initialized once
    # (hasattr guard) goes stale as nodes join/leave, distorting the matrix
    # with missing nodes or ghosts (B3).
    if (not hasattr(self, '_latency_cache')
        or set(self._latency_cache.keys()) != set(node_ids)):
      self._latency_cache: Dict[str, Dict[str, float]] = {}
      for nid in node_ids:
        self._latency_cache[nid] = {other: 0.0 if other == nid else 50.0 for other in node_ids}

    # Inject real-time forward latencies into cache
    for peer_id, lat_sec in self._last_forward_latencies.items():
        if self.id in self._latency_cache and peer_id in self._latency_cache[self.id]:
            self._latency_cache[self.id][peer_id] = lat_sec * 1000.0 # Convert to ms
            self._latency_cache[peer_id][self.id] = lat_sec * 1000.0

    async def probe_peers_background():
      for peer in self.peers:
        try:
          if hasattr(peer, 'ping') and callable(peer.ping):
            latency_ms = await asyncio.wait_for(peer.ping(timeout=2.0), timeout=3.0)
            # P2 (was: one peer's ping fanned out to EVERY self->other cell,
            # masking slow peers behind one fast one): record per-peer only.
            if latency_ms > 0:
              self._measured_rtt[peer.id()] = float(latency_ms)
              try:
                if peer.id() in self._latency_cache.get(self.id, {}):
                  self._latency_cache[self.id][peer.id()] = latency_ms
                if self.id in self._latency_cache.get(peer.id(), {}):
                  self._latency_cache[peer.id()][self.id] = latency_ms
              except Exception:
                pass
              if DEBUG >= 1:
                print(f"[latency] probed {peer.id()}: {latency_ms:.1f}ms")
          # Bandwidth probe (best-effort): real TCP echo round-trips over the
          # peer's address. The measured value lands in _measured_bandwidth and
          # is injected into every matrix entry built from the cache, so
          # partitioner/election resource scores reflect real link capacity.
          try:
            entry = await asyncio.wait_for(
              self._latency_prober.probe(peer.id(), peer.addr(), timeout=2.0),
              timeout=3.0,
            )
            if entry and entry.bandwidth_mbps > 0:
              self._measured_bandwidth[peer.id()] = entry.bandwidth_mbps
              if DEBUG >= 1:
                print(f"[latency] bandwidth to {peer.id()}: {entry.bandwidth_mbps:.1f} Mbps")
          except Exception:
            pass
        except (asyncio.TimeoutError, Exception) as e:
          if DEBUG >= 2:
            print(f"[latency] probe failed for {peer.id()}: {e}")

    # Bandwidth/latency probing is gated to at most once per 10 s: the
    # election scorer and the LACP partition path can call
    # _build_latency_matrix on a short cadence, and every probe opens a TCP
    # connection the peer's bridge echoes then closes. Re-probing too
    # eagerly would turn measurement into standing connection churn.
    if self.peers and (time.time() - getattr(self, "_last_probe_at", 0.0)) >= 10.0:
      self._last_probe_at = time.time()
      try:
        loop = asyncio.get_running_loop()
        loop.create_task(probe_peers_background())
      except RuntimeError:
        pass

    # Convert the raw {from_node: {to_node: ms}} cache into a LatencyMatrix
    # so compute_partition / HierarchicalClusterer receive the type they
    # expect (LatencyMatrix with .nodes and .entries). Measured bandwidth is
    # attached symmetrically on self<->peer edges (was: master->peer only,
    # so election weights were view-dependent under asymmetric links).
    matrix = LatencyMatrix()
    for nid in node_ids:
      for other, latency_ms in self._latency_cache.get(nid, {}).items():
        if nid == self.id:
          bw = self._measured_bandwidth.get(other, 0.0)
        elif other == self.id:
          bw = self._measured_bandwidth.get(nid, 0.0)
        else:
          bw = 0.0
        matrix.set(LatencyEntry(
          from_node=nid,
          to_node=other,
          latency_ms=float(latency_ms),
          bandwidth_mbps=bw,
        ))

    return matrix

  def _setup_p2p_rpc_handlers(self):
    """Setup listeners for P2P RPC requests (Health, Topology, Example)."""
    self.event_router.subscribe(f"synapse/rpc/health/request/{self.id}", self.on_p2p_health_request)
    self.event_router.subscribe(f"synapse/rpc/topology/request/{self.id}", self.on_p2p_topology_request)
    # C-W5: train=True example RPC had a caller but no handler (always
    # TimeoutError -> fake 0.0 loss). Reply with the real local loss.
    self.event_router.subscribe(f"synapse/rpc/example/request/{self.id}", self.on_p2p_example_rpc)

  async def on_p2p_example_rpc(self, event: Event):
    """Answer a peer's train=True example RPC with the real local loss."""
    try:
      data = event.data if isinstance(event.data, dict) else {}
      response_topic = data.get("_response_topic")
      shard = Shard.from_dict(data.get("shard"))
      loss = await self.enqueue_example(
        shard,
        self._decode_p2p_array(data.get("example"), data.get("example_shape"), data.get("example_dtype")),
        self._decode_p2p_array(data.get("target"), data.get("target_shape"), data.get("target_dtype")),
        self._decode_p2p_array(data.get("length"), data.get("length_shape"), data.get("length_dtype")),
        True,
        data.get("request_id"),
      )
      if response_topic:
        await self.event_router.publish(response_topic, {"loss": loss}, origin=self.id)
    except Exception as e:
      if DEBUG >= 1:
        print(f"[P2P] on_p2p_example_rpc failed: {e}")
      try:
        response_topic = (event.data or {}).get("_response_topic")
        if response_topic:
          await self.event_router.publish(response_topic, {"loss": 0.0, "error": str(e)[:200]}, origin=self.id)
      except Exception:
        pass

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
    """Append a token delivery to the request's streaming buffer.

    Real tokens are always appended — including legitimate adjacent repeats,
    which a naive suffix-dedup would silently drop. Deliveries of tokens this
    node itself sampled are skipped: the sampler appends directly in
    process_inference_result, so its own stream trigger / P2P loopback are
    echoes of that single append.
    """
    if request_id not in self.buffered_token_output:
      self.buffered_token_output[request_id] = ([], False)

    current_tokens, _ = self.buffered_token_output[request_id]
    if tokens and request_id not in self._locally_sampled:
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
    self._schedule_task(self._warm_replica(base_shard, partitions), "warm_replica")

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
                self._note_profile_model(self._authoritative_base_shard.model_id)
                self._schedule_task(self._warm_replica(
                  self._authoritative_base_shard, self._authoritative_partitions), "warm_replica")
