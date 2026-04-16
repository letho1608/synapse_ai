import numpy as np
import json
import asyncio
import uuid
import time
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set
from synapse.networking import Discovery, PeerHandle, Server
from synapse.inference.inference_engine import InferenceEngine, Shard, get_inference_engine
from synapse.topology.topology import Topology
from synapse.topology.device_capabilities import device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from synapse.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards
from synapse import DEBUG
from synapse.helpers import AsyncCallbackSystem
from synapse.viz.topology_viz import TopologyViz
from synapse.loading import RepoProgressEvent, ShardDownloader

class Node:
  def __init__(
    self,
    _id: str,
    server: Server,
    inference_engine: InferenceEngine,
    discovery: Discovery,
    shard_downloader: ShardDownloader,
    partitioning_strategy: PartitioningStrategy = None,
    max_generate_tokens: int = 1024,
    default_sample_temperature: float = 0.0,
    topology_viz: Optional[TopologyViz] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.server = server
    self.discovery = discovery
    self.shard_downloader = shard_downloader
    self.partitioning_strategy = partitioning_strategy
    self.peers: List[PeerHandle] = []
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self._prompt_token_ids: Dict[str, np.ndarray] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}
    
    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self.default_sample_temperature = default_sample_temperature
    self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.outstanding_requests = {}
    self._background_tasks: Set[asyncio.Task] = set()

    # === Ring-AllReduce State ===
    self.ring_rank = -1
    self.ring_world_size = 0
    self.ring_successor: Optional[PeerHandle] = None
    # chunk_index -> Event (triggered when chunk is received)
    self.received_chunk_events: Dict[int, asyncio.Event] = {}
    # chunk_index -> np.ndarray (the received data)
    self.received_chunk_data: Dict[int, np.ndarray] = {}
    self._ring_lock = asyncio.Lock()

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
    
    if DEBUG >= 1: print("Starting node: Starting gRPC server...")
    await self.server.start()
    if DEBUG >= 1: print("gRPC server started")
    
    if DEBUG >= 1: print("Starting node: Starting discovery...")
    await self.discovery.start()
    if DEBUG >= 1: print("Discovery started")
    
    if DEBUG >= 1: print(f"Starting node: Updating peers (wait_for_peers={wait_for_peers})...")
    await self.update_peers(wait_for_peers)
    if DEBUG >= 1: print(f"Peers updated: {len(self.peers)} peers")
    
    if DEBUG >= 1: print("Starting node: Collecting topology...")
    await self.collect_topology(set())
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    if DEBUG >= 1: print("Node started successfully!")
    self._schedule_task(self.periodic_topology_collection(2.0), "periodic_topology_collection")

  async def stop(self) -> None:
    await self.discovery.stop()
    await self.server.stop()

  # === Ring-AllReduce Handlers ===
  def setup_ring(self, rank: int, world_size: int, successor_url: str) -> None:
    self.ring_rank = rank
    self.ring_world_size = world_size
    # Find or create peer handle for successor
    self.ring_successor = next((p for p in self.peers if p.url == successor_url), None)
    if not self.ring_successor and successor_url:
      # If not found in current peers, we might need a dynamic PeerHandle
      # In this system, GRPCPeerHandle is the implementation
      from synapse.networking.grpc.grpc_peer_handle import GRPCPeerHandle
      self.ring_successor = GRPCPeerHandle(successor_url)
    
    if DEBUG >= 1:
      print(f"Node {self.id} Ring Setup: Rank {rank}/{world_size}, Successor: {successor_url}")

  async def handle_incoming_chunk(self, chunk_index: int, data: np.ndarray, step_type: str) -> None:
    """Called by GRPCServer when a chunk arrives from the predecessor."""
    async with self._ring_lock:
      if chunk_index not in self.received_chunk_events:
        self.received_chunk_events[chunk_index] = asyncio.Event()
      
      self.received_chunk_data[chunk_index] = data
      # Note: In a real implementation we might want to store step_type too 
      # but the order is guaranteed by the algorithm steps.
      self.received_chunk_events[chunk_index].set()

  async def execute_ring_allreduce(self, model_id: str) -> None:
    """
    Performs the 2-phase Ring-AllReduce: Scatter-Reduce followed by All-Gather.
    Operates on the gradients currently stored in the inference engine.
    """
    if self.ring_world_size <= 1:
      return # Single node, nothing to reduce

    if not self.ring_successor:
      if DEBUG >= 1: print("Ring-AllReduce Error: Successor not configured!")
      return

    # 1. Flatten all gradients into a single 1D buffer
    grads_dict = self.inference_engine.get_gradients(model_id)
    if not grads_dict:
      return

    param_names = sorted(grads_dict.keys())
    flat_grads = np.concatenate([grads_dict[name].flatten() for name in param_names])
    total_size = flat_grads.size
    
    # 2. Split into N chunks
    # Note: Using standard equal splitting. If not divisible, the last chunk is padded or smaller.
    # For research stability, we assume divisible or slightly uneven is handled by array_split.
    chunks = np.array_split(flat_grads, self.ring_world_size)
    N = self.ring_world_size
    
    if DEBUG >= 2: print(f"Ring-AllReduce Starting: {N} nodes, Total params: {total_size}")

    # Clear previous events
    self.received_chunk_events.clear()
    self.received_chunk_data.clear()

    # --- Phase 1: Scatter-Reduce ($N-1$ steps) ---
    # In each step, send chunk[rank - step] to successor and receive chunk from predecessor
    for step in range(N - 1):
      # Index of chunk we currently hold that we want to SEND
      send_idx = (self.ring_rank - step) % N
      # Index of chunk we expect to RECEIVE from predecessor
      recv_idx = (self.ring_rank - step - 1) % N
      
      # Ensure event for receiver exists
      if recv_idx not in self.received_chunk_events:
        self.received_chunk_events[recv_idx] = asyncio.Event()
      
      # Step A: Send our current version of chunk[send_idx] to successor
      if DEBUG >= 3: print(f"Scatter-Reduce Step {step}: Node {self.ring_rank} sending chunk {send_idx} to successor")
      await self.ring_successor.transfer_chunk(send_idx, chunks[send_idx], "reduce")
      
      # Step B: Wait for chunk from predecessor
      await self.received_chunk_events[recv_idx].wait()
      
      # Step C: Reduce (Sum) received data into our local chunk
      async with self._ring_lock:
        received_data = self.received_chunk_data[recv_idx]
        chunks[recv_idx] += received_data # Operation: REDUCE
        self.received_chunk_events[recv_idx].clear() # Reset for next phase if needed
        del self.received_chunk_data[recv_idx]

    if DEBUG >= 2: print(f"Scatter-Reduce Finished. Node {self.ring_rank} now has complete sum for chunk {(self.ring_rank+1)%N}")

    # --- Phase 2: All-Gather ($N-1$ steps) ---
    # At this point, Node i has the full sum for chunk (i+1)%N
    for step in range(N - 1):
      # Index of the fully reduced chunk we want to BROADCAST
      send_idx = (self.ring_rank - step + 1) % N
      recv_idx = (self.ring_rank - step) % N
      
      if recv_idx not in self.received_chunk_events:
        self.received_chunk_events[recv_idx] = asyncio.Event()

      # Step A: Send fully summed chunk to successor
      await self.ring_successor.transfer_chunk(send_idx, chunks[send_idx], "replace")
      
      # Step B: Wait for finished chunk from predecessor
      await self.received_chunk_events[recv_idx].wait()
      
      # Step C: Replace local chunk with the fully reduced one received
      async with self._ring_lock:
        chunks[recv_idx] = self.received_chunk_data[recv_idx] # Operation: REPLACE
        self.received_chunk_events[recv_idx].clear()
        del self.received_chunk_data[recv_idx]

    if DEBUG >= 2: print("All-Gather Finished. All nodes now have synced gradients.")

    # 3. Unflatten chunks back into the inference engine
    synced_flat_grads = np.concatenate(chunks)
    offset = 0
    for name in param_names:
      shape = grads_dict[name].shape
      size = grads_dict[name].size
      new_grad = synced_flat_grads[offset:offset+size].reshape(shape)
      grads_dict[name] = new_grad
      offset += size
    
    self.inference_engine.set_gradients(model_id, grads_dict)

  async def get_tailscale_nodes(self) -> List[Dict]:
    """Danh sách node Tailscale cho Web UI giám sát (chỉ có khi discovery là TailscaleDiscovery)."""
    if hasattr(self.discovery, "get_devices_for_ui"):
      return await self.discovery.get_devices_for_ui()
    return []

  def on_node_status(self, request_id, opaque_status):
    try:
      status_data = json.loads(opaque_status)
      status_type = status_data.get("type", "")
      if status_type == "supported_inference_engines":
        node_id = status_data.get("node_id")
        engines = status_data.get("engines", [])
        self.topology_inference_engines_pool.append(engines)
      elif status_type == "node_status":
        if status_data.get("status", "").startswith("start_"):
          self.current_topology.active_node_id = status_data.get("node_id")
        elif status_data.get("status", "").startswith("end_"):
          if status_data.get("node_id") == self.current_topology.active_node_id:
            self.current_topology.active_node_id = None

      download_progress = None
      if status_type == "download_progress":
        if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
        download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
        self.node_download_progress[status_data.get('node_id')] = download_progress

      if self.topology_viz:
        self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id, self.node_download_progress)
    except Exception as e:
      if DEBUG >= 1: print(f"Error on_node_status: {e}")
      if DEBUG >= 1: traceback.print_exc()

  def get_supported_inference_engines(self):
    return ["pytorch"]

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool

  async def process_sync_weights(self, model_id: str, weights: np.ndarray, step: int) -> np.ndarray:
    """Xử lý đồng bộ trọng số (giai đoạn sau: Ring-AllReduce). Hiện tại trả về trực tiếp."""
    if DEBUG >= 1: print(f"Node: Received SyncWeights for {model_id} at step {step}")
    # TODO: Implement Ring-AllReduce logic here
    return weights

  async def process_hardware_profile(self, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray, n_iters: int, skip_iters: int) -> Tuple[float, float]:
    """Đo đạc throughput thực tế trên phần cứng cục bộ sử dụng chiến lược Drop-First (Real Data Cache)."""
    if DEBUG >= 1:
      print(f"Node profiling started: model={shard.model_id}, iters={n_iters}, skip={skip_iters}, batch_size={example.shape[0]}")
    
    times = []
    # Đảm bảo Shard (model layers) đã được load vào GPU/RAM
    await self.inference_engine.ensure_shard(shard)
    
    for i in range(n_iters):
      start_t = time.perf_counter()
      # Chạy 1 batch training (forward + backward) cục bộ
      await self.inference_engine.process_example(shard, example, target, length, train=True)
      end_t = time.perf_counter()
      
      duration = end_t - start_t
      if i >= skip_iters:
        times.append(duration)
        if DEBUG >= 2:
          print(f"  Iteration {i+1}/{n_iters}: {duration:.4f}s")
      else:
        if DEBUG >= 2:
          print(f"  Warmup Iteration {i+1}/{n_iters}: {duration:.4f}s (dropped)")
    
    if not times:
      return 0.0, 0.0
    
    avg_time = sum(times) / len(times)
    batch_size = example.shape[0]
    samples_per_sec = batch_size / avg_time
    avg_latency_ms = avg_time * 1000
    
    if DEBUG >= 1:
      print(f"Node profiling completed: {samples_per_sec=:.2f} samples/s, avg_latency={avg_latency_ms:.2f}ms")
    
    return samples_per_sec, avg_latency_ms
  
  token_count = 0
  first_token_time = 0
  async def process_inference_result(
    self,
    shard,
    result: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ):
    if shard.model_id != 'stable-diffusion-2-1-base':
      if request_id not in self.buffered_token_output:
        self.buffered_token_output[request_id] = ([], False)
      is_finished = len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
      
      # Kiểm tra result rỗng trước khi sample - đây là nguyên nhân gốc rễ
      if shard.is_last_layer() and not is_finished:
        # Kiểm tra xem result có rỗng không (sequence length = 0)
        if result.size == 0 or (len(result.shape) >= 2 and result.shape[1] == 0):
          if DEBUG >= 2:
            print(f"[{request_id}] Warning: Empty result from inference, cannot sample. Marking as finished.")
          is_finished = True
          self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
          forward = np.array([[]], dtype=np.int32).reshape(1, 0)  # Empty tensor với shape đúng
          intermediate_result = []
        else:
          # Sample token from logits
          buf = self.buffered_token_output[request_id][0]
          token = await self.inference_engine.sample(result, temp=self.default_sample_temperature, generated_ids=buf if buf else None)
          await self.inference_engine.ensure_shard(shard)
          token_value = token.item()
          
          # Add to buffer
          self.buffered_token_output[request_id][0].append(token_value)
          token_emitted = True
          
          # Get eos_token_id safely
          eos_token_id = None
          if hasattr(self.inference_engine, 'tokenizer') and self.inference_engine.tokenizer is not None:
            eos_token_id = getattr(self.inference_engine.tokenizer, 'eos_token_id', None)
            if eos_token_id is None and hasattr(self.inference_engine.tokenizer, '_tokenizer'):
              eos_token_id = getattr(self.inference_engine.tokenizer._tokenizer, 'eos_token_id', None)
          
          # Qwen-specific stop tokens (in addition to eos_token_id)
          qwen_stop_tokens = [151643, 151644, 151645]  # <|im_start|>, <|im_end|>, etc
          
          # Check for repetition: if token repeats >= 10 times, stop generation
          buffered_tokens = self.buffered_token_output[request_id][0]
          if len(buffered_tokens) >= 10:
            last_10_tokens = buffered_tokens[-10:]
            if len(set(last_10_tokens)) == 1:  # All 10 last tokens are identical
              if DEBUG >= 1: 
                print(f"[{request_id}] Detected repetition loop: token {token_value} repeated 10 times. Stopping generation.")
              is_finished = True
          
          # Check if this is a stop token (eos_token_id or Qwen-specific)
          if (eos_token_id is not None and token_value == eos_token_id) or (token_value in qwen_stop_tokens):
              if DEBUG >= 1:
                  print(f"[{request_id}] Detected EOS/stop token (ID: {token_value}). Stopping generation.")
              is_finished = True
              # Remove the eos token from buffer so it doesn't show in output
              if self.buffered_token_output[request_id][0]:
                  self.buffered_token_output[request_id][0].pop()
              token_emitted = False 
          
          is_finished = is_finished or len(buffered_tokens) >= self.max_generate_tokens
          if DEBUG >= 2: 
            print(f"[{request_id}] Token: {token_value}, Emitted: {token_emitted}, EOS_ID: {eos_token_id}, Finished: {is_finished}")
          
          forward = token.reshape(1, -1)
          # CRITICAL FIX: Send only new token for streaming (delta), not all buffered tokens
          # This allows API to accumulate tokens correctly
          if self.buffered_token_output[request_id][0] and token_emitted:
              # Send only the new token (last one in buffer)
              intermediate_result = [self.buffered_token_output[request_id][0][-1]]
              if DEBUG >= 2:
                print(f"[{request_id}] Sending new token: {intermediate_result}, total buffered: {len(self.buffered_token_output[request_id][0])}")
          else:
              # CRITICAL FIX: Even if no token emitted, we need to send signal to API
              # If generation is finished, we must trigger callback with is_finished=True
              # If not finished, send empty list but ensure callback is still triggered
              intermediate_result = []
              # If finished but no tokens, we still need to notify API
              if is_finished and not self.buffered_token_output[request_id][0]:
                # Send empty tokens but with is_finished=True so API knows to stop waiting
                if DEBUG >= 2:
                  print(f"[{request_id}] Generation finished with no tokens, sending empty result with is_finished=True")
      else:
        forward = result
        intermediate_result = []
    else:
      await self.inference_engine.ensure_shard(shard)
      is_finished = inference_state.get("is_finished", False)
      intermediate_result, inference_state = self.handle_stable_diffusion(inference_state, result)
      forward = result
    if shard.is_last_layer():
      # CRITICAL FIX: Always trigger callback, even with empty tokens
      # This ensures API receives the signal that generation is finished
      if DEBUG >= 1:
        print(f"[{request_id}] [LAST_LAYER] Triggering callback: tokens={intermediate_result}, is_finished={is_finished}, buffered_tokens={self.buffered_token_output.get(request_id, 'N/A')}")
      self.trigger_on_token_callbacks(request_id, intermediate_result, is_finished)
      self._schedule_task(self.broadcast_result(request_id, intermediate_result, is_finished), "broadcast_result")

    if is_finished:
      if shard.model_id != 'stable-diffusion-2-1-base':
        self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
      self.outstanding_requests.pop(request_id, None)
      self._prompt_token_ids.pop(request_id, None)
    else:
      self.outstanding_requests[request_id] = "waiting"
      self._schedule_task(
        self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset = 1), inference_state),
        "forward_tensor",
      )

    return  np.array(self.buffered_token_output[request_id][0]) if shard.model_id != 'stable-diffusion-2-1-base' else intermediate_result


  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = {},
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
    start_time = time.perf_counter_ns()
    resp = await self._process_prompt(base_shard, prompt, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
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

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None, inference_state: Optional[dict] = None) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=}")

    if not shard.is_first_layer():
      if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
      self.outstanding_requests[request_id] = "waiting"
      resp = await self.forward_prompt(shard, prompt, request_id, 0, inference_state)
      return None
    else:
      self.outstanding_requests[request_id] = "processing"
      try:
        prompt_ids = await self.inference_engine.encode(shard, prompt)
        self._prompt_token_ids[request_id] = prompt_ids
        result, inference_state = await self.inference_engine.infer_prompt(request_id, shard, prompt, inference_state)
        ret = await self.process_inference_result(shard, result, request_id, inference_state)
        return result
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
        self.outstanding_requests.pop(request_id, None)
        self._prompt_token_ids.pop(request_id, None)
        raise

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
      loss = await self.forward_example(shard, example, target, length, train, request_id, 0) 
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

  async def process_hardware_profile(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    n_iters: int = 5,
    skip_iters: int = 2
  ) -> Tuple[float, float]:
    """Measures local training performance (samples/sec and latency)."""
    # Tối ưu cho CPU: nếu không có CUDA, giảm số lần chạy thử xuống tối thiểu
    try:
      import torch
      if not torch.cuda.is_available():
        n_iters = min(n_iters, 2)
        skip_iters = min(skip_iters, 1)
        if DEBUG >= 1: print("[HardwareProfile] CPU detected. Optimization: Reducing iterations to prevent hang.")
    except Exception:
      pass

    if DEBUG >= 1: print(f"[HardwareProfile] Profiling shard {base_shard.model_id} for {n_iters} iterations...")
    
    shard = self.get_current_shard(base_shard)
    latencies = []
    
    # Warmup
    for _ in range(skip_iters):
        try:
            # Sử dụng request_id giả lập để không trùng với job thật
            await self.inference_engine.train(f"warmup-{uuid.uuid4()}", shard, example, target, length)
        except Exception:
            pass
            
    # Measure
    for i in range(n_iters):
        start = time.perf_counter()
        request_id = f"profile-{i}-{uuid.uuid4()}"
        try:
            await self.inference_engine.train(request_id, shard, example, target, length)
            latencies.append(time.perf_counter() - start)
        except Exception as e:
            if DEBUG >= 1: print(f"[HardwareProfile] Step {i} failed: {e}")
            
    if not latencies:
        return 0.1, 1000.0 # Fallback
        
    avg_latency = sum(latencies) / len(latencies)
    throughput = 1.0 / avg_latency if avg_latency > 0 else 0.1 # samples/sec
    
    return throughput, avg_latency * 1000.0 # throughput, latency_ms
        
  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    if DEBUG >= 2: print(f"[{request_id}] process_tensor: {base_shard=} {shard=} {tensor.size=} {tensor.shape=} {elapsed_time_ns=}")

  async def _process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)

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
      ret = await self.process_inference_result(shard, result, request_id, inference_state)
      return ret
    except Exception as e:
      self.outstanding_requests.pop(request_id)
      print(f"Error processing tensor for shard {shard}: {e}")
      traceback.print_exc()
  
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
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
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
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
    if target_id == self.id:
      await self.process_prompt(next_shard, prompt, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending prompt to {target_peer.id()}: {prompt}")
      await target_peer.send_prompt(next_shard, prompt, request_id=request_id, inference_state=inference_state)
  
  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {next_shard}")
    if target_id == self.id:
      await self.process_tensor(next_shard, tensor, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: {tensor}")
      await target_peer.send_tensor(next_shard, tensor, request_id=request_id, inference_state=inference_state)

  def get_partition_index(self, offset: int = 0):
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return None
    partitions = self.partitioning_strategy.partition(self.topology)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    return (current_partition_index + offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    if index is None:
      index = self.get_partition_index()
    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_to_disconnect = [peer for peer in peers_removed if await peer.is_connected()]
    peers_to_connect = [peer for peer in peers_added + peers_updated + peers_unchanged if not await peer.is_connected()]

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{peer.id()}@{peer.addr()}" for peer in peers]

    if DEBUG >= 3:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_disconnect={peers_to_disconnect} to_connect={peers_to_connect}")

    async def disconnect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(peer, timeout=15):
      try:
        await asyncio.wait_for(peer.connect(), timeout)
        
        # --- Network Profiling on Join ---
        if peer.id() != self.id:
          if DEBUG >= 1: print(f"Profiling network for new peer: {peer.id()}...")
          # Sửa dụng 2MB để test nhanh khi join
          test_payload = b"0" * (2 * 1024 * 1024)
          start_time = time.perf_counter()
          latency_ms = await peer.test_network(test_payload)
          end_time = time.perf_counter()
          
          # Thực tế RTT (Round Trip Time)
          rtt_ms = (end_time - start_time) * 1000
          # Băng thông: (2MB * 2 / RTT_s) để tính cả upload/download mô phỏng, 
          # hoặc đơn giản (2MB / latency_ms_reported)
          bandwidth_mbps = (2.0 * 8) / (rtt_ms / 1000.0)
          
          caps = peer.device_capabilities()
          caps.latency_ms = round(rtt_ms / 2, 2) #Ước lượng chiều đi
          caps.bandwidth_mbps = round(bandwidth_mbps, 2)
          if DEBUG >= 1: print(f"Peer {peer.id()} profiled: {caps.bandwidth_mbps} Mbps, {caps.latency_ms} ms")
        
        return True
      except Exception as e:
        print(f"Error connecting/profiling peer {peer.id()}@{peer.addr()}: {e}")
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
        await self.collect_topology(set())
        if did_peers_change:
          await self.select_best_inference_engine()
      except Exception as e:
        print(f"Error collecting topology: {e}")
        traceback.print_exc()

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

    next_topology.active_node_id = self.topology.active_node_id
    self.topology = next_topology
    if self.topology_viz:
      self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id)
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
        await asyncio.wait_for(peer.send_result(request_id, result, is_finished), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout broadcasting result to {peer.id()}")
      except Exception as e:
        print(f"Error broadcasting result to {peer.id()}: {e}")
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

  @property
  def current_topology(self) -> Topology:
    return self.topology

  def handle_stable_diffusion(self, inference_state, result):
    if inference_state['is_step_finished']:
      inference_state['step']+=1
    progress = [inference_state['step'],inference_state['total_steps']]
    intermediate_result = result
    if progress[0] == progress[1]:
      intermediate_result = result
    return intermediate_result, inference_state
