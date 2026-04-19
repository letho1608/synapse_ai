import numpy as np
import json
import asyncio
import uuid
import time
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set, Any
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
    self.peers: List[PeerHandle] = {}
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

    # Register internal callback for token synchronization in distributed mode
    self.on_token.register("internal_token_sync").on_next(self._on_token_received)

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
  
  token_count = 0
  first_token_time = 0
  async def process_inference_result(
      self,
      shard,
      result: np.ndarray,
      request_id: Optional[str] = None,
      inference_state: Optional[dict] = None,
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
          except Exception:
            pass

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
        self._schedule_task(self.broadcast_result(request_id, intermediate_result, is_finished), name=f"broadcast_{request_id}")
    
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
    start_time = time.perf_counter_ns()
    resp = await self._process_prompt(base_shard, prompt, request_id, inference_state, max_generate_tokens)
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
    return resp

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None, inference_state: Optional[dict] = None, max_generate_tokens: Optional[int] = None) -> Optional[np.ndarray]:
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
            
            # Watchdog: if we've been waiting too long without any progress
            if time.perf_counter() - wait_start > 60.0:
              if DEBUG >= 1: 
                print(f"[{request_id}] Generation watchdog triggered: no progress in 60s. State: {self.outstanding_requests.get(request_id)}")
              
              # If we are waiting for a remote node, maybe try to nudge it?
              # For now, just continue waiting but log the stall
              wait_start = time.perf_counter() # Reset watchdog to log again in 60s
              
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
        self.outstanding_requests.pop(request_id, None)
        self._prompt_token_ids.pop(request_id, None)
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
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id, inference_state)
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
    target_id = self.partitioning_strategy.partition(self.topology, base_shard)[target_index].node_id
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
      target_id = self.partitioning_strategy.partition(self.topology, base_shard)[target_index].node_id
      next_shard = self.get_current_shard(base_shard, target_index)
      if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
      if target_id == self.id:
          # FIX: When target is self, call _process_prompt directly to avoid infinite loop
          # process_prompt -> _process_prompt -> forward_prompt -> process_prompt loop
          await self._process_prompt(next_shard, prompt, request_id, inference_state)
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
    target_id = self.partitioning_strategy.partition(self.topology, base_shard)[target_index].node_id
    target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")
    if target_id == self.id:
      return await self.process_tensor(target_shard, tensor, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: {tensor}")
      # HIỆN THÔNG BÁO PHÂN TÁN RÕ RÀNG
      print(f"[DISTRIBUTED] Forwarding workload to Node: {target_id}")
      # Add a safety watchdog to prevent infinite hangs in forwarding
      result = await asyncio.wait_for(
        target_peer.send_tensor(
          target_shard,
          tensor,
          request_id=request_id,
          inference_state=inference_state,
        ),
        timeout=300.0 # Extreme safety margin
      )
      return result

  def get_partition_index(self, offset: int = 0, base_shard: Optional[Shard] = None):
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return None
    partitions = self.partitioning_strategy.partition(self.topology, base_shard)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    return (current_partition_index + offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    if index is None:
      index = self.get_partition_index(base_shard=base_shard)
    partitions = self.partitioning_strategy.partition(self.topology, base_shard)
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
