import numpy as np
import json
import asyncio
import grpc
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
    # FIX: For distributed inference, we need to forward tensor to next partition
    # regardless of is_finished status. The is_finished only affects token generation
    # at the LAST_LAYER, but tensor should still be forwarded through the pipeline.
    if not shard.is_last_layer():
        # Not last layer - always forward tensor to next partition
        self.outstanding_requests[request_id] = "waiting"
        self._schedule_task(
            self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset=1, base_shard=shard), inference_state),
            "forward_tensor",
        )
    else:
        # Last layer - trigger callbacks for token output
        if DEBUG >= 1:
            print(f"[{request_id}] [LAST_LAYER] Triggering callback: tokens={intermediate_result}, is_finished={is_finished}, buffered_tokens={self.buffered_token_output.get(request_id, 'N/A')}")
        self.trigger_on_token_callbacks(request_id, intermediate_result, is_finished)
        self._schedule_task(self.broadcast_result(request_id, intermediate_result, is_finished), "broadcast_result")
        
        if is_finished:
            if shard.model_id != 'stable-diffusion-2-1-base':
                self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
            self.outstanding_requests.pop(request_id, None)
            self._prompt_token_ids.pop(request_id, None)

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
          # ─── DISTRIBUTED: Node này là LAST LAYER, tự train ─────────
          is_from_remote = shard.start_layer > 0  # Không phải first layer → nhận từ node khác
          if is_from_remote:
            print(
              f"\n[DISTRIBUTED-TRAIN] 🎯  NODE RECEIVER [{self.id[:8]}]"
              f" | Nhận intermediate tensor từ node khác"
              f" | shard={shard.start_layer}-{shard.end_layer}/{shard.n_layers-1}"
              f" | example_shape={example.shape}"
              f" | request={request_id[:8]}"
              f" → Đang tính loss & backward..."
            )
          # ────────────────────────────────────────────────────────────
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
    target_id = self.partitioning_strategy.partition(self.topology, base_shard)[target_index].node_id
    target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")
    target_peer = next((p for p in self.peers if p.id() == target_id), None)
    if not target_peer:
      raise ValueError(f"peer for {target_index} not found")
    if DEBUG >= 1: print(f"sending example to {target_peer.id()}: {step} => {target} ({length})")

    # ─── DISTRIBUTED TRAINING DETECTION ─────────────────────────────
    action = "TRAIN" if train else "EVAL"
    print(
      f"\n[DISTRIBUTED-{action}] ✈  NODE SENDER  [{self.id[:8]}]"
      f" → NODE RECEIVER [{target_peer.id()[:8]}]"
      f" | shard={target_shard.start_layer}-{target_shard.end_layer}/{target_shard.n_layers-1}"
      f" | tensor_shape={step.shape}"
      f" | request={request_id[:8]}"
    )
    # ─────────────────────────────────────────────────────────────────

    try:
      resp = await asyncio.wait_for(
        target_peer.send_example(target_shard, step, target, length, request_id=request_id, train=train),
        timeout=30.0,
      )
      print(
        f"[DISTRIBUTED-{action}] ✅  Gửi thành công [{self.id[:8]}] → [{target_peer.id()[:8]}]"
        f" | request={request_id[:8]}"
      )
      return resp

    except asyncio.TimeoutError:
      # wait_for(30s) hết hạn — bao gồm cả 3 lần retry bên trong _rpc_with_retry
      print(
        f"\n[DISTRIBUTED-ERROR] ⏰  TIMEOUT (30s)"
        f"\n  Sender  : [{self.id[:8]}]"
        f"\n  Receiver: [{target_peer.id()[:8]}] @ {target_peer.addr()}"
        f"\n  Node B không phản hồi — có thể treo hoặc quá tải"
        f"\n  request={request_id[:8]} | tensor_shape={step.shape}"
      )
      raise RuntimeError(
        f"[Distributed] Timeout 30s: Node B [{target_peer.id()[:8]}] không phản hồi."
      )

    except grpc.aio.AioRpcError as e:
      # Lỗi gRPC thực — xảy ra sau khi _rpc_with_retry đã thử 3 lần
      code = e.code()
      if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
        # Node B đã tắt hoặc mất mạng (sau 3 lần retry)
        print(
          f"\n[DISTRIBUTED-ERROR] 🔌  NODE B KHÔNG KHẢ DỤNG [{code.name}]"
          f"\n  Sender  : [{self.id[:8]}]"
          f"\n  Receiver: [{target_peer.id()[:8]}] @ {target_peer.addr()}"
          f"\n  Đã retry {target_peer.max_rpc_retries} lần nhưng vẫn thất bại"
          f"\n  request={request_id[:8]}"
        )
        raise RuntimeError(
          f"[Distributed] Node B [{target_peer.id()[:8]}] không kết nối được ({code.name}). "
          f"Node B có thể đã tắt hoặc mất mạng."
        ) from e
      else:
        # Lỗi gRPC khác (PERMISSION_DENIED, UNIMPLEMENTED, DATA_LOSS...)
        print(
          f"\n[DISTRIBUTED-ERROR] ❌  gRPC ERROR [{code.name}]"
          f"\n  Sender  : [{self.id[:8]}]"
          f"\n  Receiver: [{target_peer.id()[:8]}] @ {target_peer.addr()}"
          f"\n  Chi tiết: {e.details()}"
          f"\n  request={request_id[:8]}"
        )
        raise

    except ConnectionError as e:
      # _rpc_with_retry hết 3 lần retry và tự raise ConnectionError
      print(
        f"\n[DISTRIBUTED-ERROR] 🔌  KẾT NỐI THẤT BẠI (sau {target_peer.max_rpc_retries} lần retry)"
        f"\n  Sender  : [{self.id[:8]}]"
        f"\n  Receiver: [{target_peer.id()[:8]}] @ {target_peer.addr()}"
        f"\n  Chi tiết: {e}"
        f"\n  request={request_id[:8]}"
      )
      raise RuntimeError(
        f"[Distributed] Mất kết nối tới Node B [{target_peer.id()[:8]}] @ {target_peer.addr()}."
      ) from e

    except Exception as e:
      # Mọi lỗi không mong đợi (serialization, OOM bên Node B...)
      print(
        f"\n[DISTRIBUTED-ERROR] 💥  LỖI KHÔNG XÁC ĐỊNH [{type(e).__name__}]"
        f"\n  Sender  : [{self.id[:8]}]"
        f"\n  Receiver: [{target_peer.id()[:8]}] @ {target_peer.addr()}"
        f"\n  shard={target_shard.start_layer}-{target_shard.end_layer}/{target_shard.n_layers-1}"
        f"\n  Chi tiết: {e}"
        f"\n  request={request_id[:8]}"
      )
      traceback.print_exc()
      raise

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
