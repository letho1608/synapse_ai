import grpc
import numpy as np
import asyncio
from typing import Optional, Tuple, List

from . import node_service_pb2
from . import node_service_pb2_grpc

from ..peer_handle import PeerHandle
from synapse.inference.shard import Shard
from synapse.topology.topology import Topology
from synapse.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from synapse.helpers import DEBUG
import json
import time


class GRPCPeerHandle(PeerHandle):
  def __init__(self, _id: str, address: str, desc: str, device_capabilities: DeviceCapabilities):
    self._id = _id
    self.address = address
    self.desc = desc
    self._device_capabilities = device_capabilities
    self.channel = None
    self.stub = None
    self.channel_options = [
      ("grpc.max_metadata_size", 32 * 1024 * 1024),
      ("grpc.max_receive_message_length", 256 * 1024 * 1024),
      ("grpc.max_send_message_length", 256 * 1024 * 1024),
      ("grpc.max_concurrent_streams", 100),
      ("grpc.http2.min_time_between_pings_ms", 10000),
      ("grpc.keepalive_time_ms", 10000),
      ("grpc.keepalive_timeout_ms", 5000),
      ("grpc.keepalive_permit_without_calls", 1),
      ("grpc.http2.max_pings_without_data", 0),
      ("grpc.http2.min_ping_interval_without_data_ms", 5000),
      ("grpc.tcp_nodelay", 1),
      ("grpc.optimization_target", "throughput"),
    ]
    self.max_rpc_retries = 3
    self.retryable_status_codes = {
      grpc.StatusCode.UNAVAILABLE,
      grpc.StatusCode.UNKNOWN,
      grpc.StatusCode.INTERNAL,
      grpc.StatusCode.DEADLINE_EXCEEDED,
    }
    self.rpc_retry_base_delay = 0.5

  def id(self) -> str:
    return self._id

  def addr(self) -> str:
    return self.address

  def description(self) -> str:
    return self.desc

  def device_capabilities(self) -> DeviceCapabilities:
    return self._device_capabilities

  async def connect(self):
    # Cleanup channel cũ nếu có
    if self.channel is not None:
      try:
        await self.disconnect()
      except Exception:
        pass  # Ignore errors khi cleanup
    
    try:
      self.channel = grpc.aio.insecure_channel(
        self.address,
        options=self.channel_options,
        compression=grpc.Compression.Gzip
      )
      self.stub = node_service_pb2_grpc.NodeServiceStub(self.channel)
      await asyncio.wait_for(self.channel.channel_ready(), timeout=10.0)
    except Exception as e:
      # Nếu channel bị đóng hoặc lỗi khác, cleanup và raise
      # Kiểm tra xem có phải lỗi "Channel is closed" không
      error_msg = str(e)
      if "Channel is closed" in error_msg or "UsageError" in error_msg:
        if DEBUG >= 2:
          print(f"Channel closed error during connect for {self._id}@{self.address}: {e}")
      await self.disconnect()
      raise

  async def is_connected(self) -> bool:
    if self.channel is None:
      return False
    state = self.channel.get_state()
    # Chỉ coi là connected nếu channel ở trạng thái READY
    # Các trạng thái khác (CONNECTING, TRANSIENT_FAILURE, SHUTDOWN) đều không phải connected
    return state == grpc.ChannelConnectivity.READY

  async def disconnect(self):
    if self.channel:
      await self.channel.close()
    self.channel = None
    self.stub = None
  
  async def _rpc_with_retry(self, rpc_name: str, call_factory, timeout: Optional[float] = None):
    last_exc: Optional[Exception] = None
    for attempt in range(1, self.max_rpc_retries + 1):
      try:
        await self._ensure_connected()
        call = call_factory()
        if timeout is not None:
          return await asyncio.wait_for(call, timeout=timeout)
        return await call
      except asyncio.TimeoutError as e:
        last_exc = e
        if DEBUG >= 2:
          print(f"Timeout calling {rpc_name} on {self._id}@{self.address} (attempt {attempt}/{self.max_rpc_retries})")
        await self.disconnect()
      except grpc.aio.AioRpcError as e:
        if e.code() in self.retryable_status_codes:
          last_exc = e
          if DEBUG >= 2:
            print(f"gRPC error {e.code()} in {rpc_name} for {self._id}@{self.address} (attempt {attempt}/{self.max_rpc_retries})")
          await self.disconnect()
        else:
          raise
      except Exception:
        raise

      if attempt < self.max_rpc_retries:
        delay = self.rpc_retry_base_delay * attempt
        await asyncio.sleep(delay)

    if last_exc:
      raise last_exc
    raise ConnectionError(f"{rpc_name} failed for {self._id}@{self.address} after retries")

  async def _ensure_connected(self):
    if not (await self.is_connected()):
      try:
        await asyncio.wait_for(self.connect(), timeout=10.0)
        # Đảm bảo stub đã được tạo sau khi connect thành công
        if self.stub is None:
          raise ConnectionError(f"Stub is None after successful connection for {self._id}@{self.address}")
      except asyncio.TimeoutError:
        if DEBUG >= 2: print(f"Connection timeout for {self._id}@{self.address}")
        await self.disconnect()
        raise
      except Exception as e:
        if DEBUG >= 2: print(f"Connection error for {self._id}@{self.address}: {e}")
        await self.disconnect()
        raise
    # Kiểm tra stub ngay cả khi đã connected (có thể bị None do lỗi trước đó)
    if self.stub is None:
      if DEBUG >= 2: print(f"Stub is None but channel is connected for {self._id}@{self.address}, reconnecting...")
      try:
        await asyncio.wait_for(self.connect(), timeout=10.0)
      except Exception as e:
        await self.disconnect()
        raise ConnectionError(f"Failed to recreate stub for {self._id}@{self.address}: {e}")

  async def health_check(self) -> bool:
    try:
      await self._ensure_connected()
      # Đảm bảo stub đã được tạo
      if self.stub is None:
        return False
      request = node_service_pb2.HealthCheckRequest()
      response = await asyncio.wait_for(self.stub.HealthCheck(request), timeout=5)
      return response.is_healthy
    except asyncio.TimeoutError:
      return False
    except Exception:
      if DEBUG >= 4:
        print(f"Health check failed for {self._id}@{self.address}.")
        import traceback
        traceback.print_exc()
      return False

  async def probe(self) -> Tuple[str, DeviceCapabilities]:
    """Tự nhận diện Node ID và cấu hình của Peer bằng cách gọi CollectTopology."""
    try:
      await self._ensure_connected()
      topo = await self.collect_topology(visited=set(), max_depth=0)
      # Tìm node ID của chính nó trong topology (là node duy nhất nếu max_depth=0)
      for node_id, caps in topo.all_nodes():
        return node_id, caps
      raise RuntimeError("Peer returned empty topology during probe")
    except Exception as e:
      if DEBUG >= 1: print(f"Probe failed for {self.address}: {e}")
      raise

  async def send_prompt(self, shard: Shard, prompt: str, inference_state: Optional[dict] = None, request_id: Optional[str] = None) -> Optional[np.array]:
    request = node_service_pb2.PromptRequest(
      prompt=prompt,
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      request_id=request_id,
      inference_state=None if inference_state is None else self.serialize_inference_state(inference_state)
    )
    await self._rpc_with_retry(
      "SendPrompt",
      lambda: self.stub.SendPrompt(request),
    )

  async def send_tensor(self, shard: Shard, tensor: np.ndarray, inference_state: Optional[dict] = None, request_id: Optional[str] = None) -> Optional[np.array]:
    request = node_service_pb2.TensorRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      tensor=self._serialize_tensor(tensor),
      request_id=request_id,
      inference_state=None if inference_state is None else self.serialize_inference_state(inference_state)
    )
    response = await self._rpc_with_retry(
      "SendTensor",
      lambda: self.stub.SendTensor(request),
    )

    if not response.tensor_data or not response.shape or not response.dtype:
      return None

    return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

  async def send_example(self, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray, train: bool, request_id: Optional[str] = None) -> Optional[np.array]:
    request = node_service_pb2.ExampleRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      example=self._serialize_tensor(example),
      target=self._serialize_tensor(target),
      length=self._serialize_tensor(length),
      train=train,
      request_id=request_id,
    )
    response = await self._rpc_with_retry(
      "SendExample",
      lambda: self.stub.SendExample(request),
    )
    loss = response.loss
    if train and not shard.is_first_layer():
      grads = None
      if response.HasField("grads") and response.grads.tensor_data and response.grads.dtype:
        grads = np.frombuffer(response.grads.tensor_data, dtype=np.dtype(response.grads.dtype)).reshape(response.grads.shape)
      return loss, grads
    else:
      return loss

  async def send_loss(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> Optional[np.array]:
    request = node_service_pb2.TensorRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      tensor=node_service_pb2.Tensor(tensor_data=tensor.tobytes(), shape=tensor.shape, dtype=str(tensor.dtype)),
      request_id=request_id,
    )
    response = await self._rpc_with_retry(
      "SendLoss",
      lambda: self.stub.SendLoss(request),
    )

    if not response.tensor_data or not response.shape or not response.dtype:
      return None

    return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

  async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
    try:
      request = node_service_pb2.CollectTopologyRequest(visited=visited, max_depth=max_depth)
      response = await self._rpc_with_retry(
        "CollectTopology",
        lambda: self.stub.CollectTopology(request),
      )
      topology = Topology()
      for node_id, capabilities in response.nodes.items():
        device_capabilities = DeviceCapabilities(
          model=capabilities.model,
          chip=capabilities.chip,
          memory=capabilities.memory,
          flops=DeviceFlops(fp16=capabilities.flops.fp16, fp32=capabilities.flops.fp32, int8=capabilities.flops.int8),
          gpu_backend=capabilities.gpu_backend,
          cpu_cores=capabilities.cpu_cores,
          system_ram_mb=capabilities.system_ram_mb,
          gpu_count=capabilities.gpu_count,
          disk_gb=capabilities.disk_gb,
          warmup_throughput=capabilities.warmup_throughput,
          cpu_usage_pct=capabilities.cpu_usage_pct,
          gpu_usage_pct=capabilities.gpu_usage_pct,
          ram_used_mb=capabilities.ram_used_mb,
          gpu_memory_used_mb=capabilities.gpu_memory_used_mb,
        )
        topology.update_node(node_id, device_capabilities)
      for node_id, peer_connections in response.peer_graph.items():
        for conn in peer_connections.connections:
          topology.add_edge(node_id, conn.to_id, conn.description)
      return topology
    except asyncio.CancelledError:
      # Xử lý CancelledError một cách graceful - có thể do timeout hoặc connection bị đóng
      raise  # Re-raise để asyncio.wait_for có thể xử lý
    except Exception as e:
      # Log lỗi nhưng không crash
      if DEBUG >= 2:
        print(f"Error in collect_topology for {self._id}@{self.address}: {e}")
      raise  # Re-raise để caller có thể xử lý

  async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    tensor = None
    if isinstance(result, np.ndarray):
      tensor = node_service_pb2.Tensor(tensor_data=result.tobytes(), shape=result.shape, dtype=str(result.dtype))
      result = []
    request = node_service_pb2.SendResultRequest(request_id=request_id, result=result, tensor=tensor, is_finished=is_finished)
    await self._rpc_with_retry(
      "SendResult",
      lambda: self.stub.SendResult(request),
    )

  async def setup_ring(self, rank: int, world_size: int, successor_url: str) -> None:
    request = node_service_pb2.SetupRingRequest(
      rank=rank,
      world_size=world_size,
      successor_url=successor_url
    )
    await self._rpc_with_retry(
      "SetupRing",
      lambda: self.stub.SetupRing(request),
    )

  async def transfer_chunk(self, chunk_index: int, tensor: np.ndarray, step_type: str) -> None:
    s_type = node_service_pb2.RingStepType.REDUCE if step_type.lower() == "reduce" else node_service_pb2.RingStepType.REPLACE
    request = node_service_pb2.TransferChunkRequest(
      chunk_index=chunk_index,
      tensor=self._serialize_tensor(tensor),
      step_type=s_type
    )
    await self._rpc_with_retry(
      "TransferChunk",
      lambda: self.stub.TransferChunk(request),
    )

  async def trigger_ring_allreduce(self, model_id: str) -> None:
    request = node_service_pb2.TriggerRingAllReduceRequest(model_id=model_id)
    await self._rpc_with_retry(
      "TriggerRingAllReduce",
      lambda: self.stub.TriggerRingAllReduce(request),
    )

  async def send_opaque_status(self, request_id: str, status: str) -> None:
    max_retries = 2
    for attempt in range(max_retries):
      try:
        await self._ensure_connected()
        request = node_service_pb2.SendOpaqueStatusRequest(request_id=request_id, status=status)
        await asyncio.wait_for(self.stub.SendOpaqueStatus(request), timeout=10.0)
        return  # Thành công, thoát khỏi hàm
      except grpc.aio.AioRpcError as e:
        # Kiểm tra nếu là lỗi kết nối cần reconnect
        if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN):
          if DEBUG >= 2:
            print(f"gRPC error {e.code()} for {self._id}@{self.address}, attempt {attempt + 1}/{max_retries}")
          # Đóng channel để force reconnect lần sau
          await self.disconnect()
          if attempt < max_retries - 1:
            # Chờ một chút trước khi retry
            await asyncio.sleep(0.1)
            continue
        # Nếu không phải lỗi kết nối hoặc đã hết retry, raise lỗi
        raise
      except asyncio.TimeoutError:
        # Timeout cũng có thể do kết nối bị hỏng
        if DEBUG >= 2:
          print(f"Timeout sending opaque status to {self._id}@{self.address}, attempt {attempt + 1}/{max_retries}")
        await self.disconnect()
        if attempt < max_retries - 1:
          await asyncio.sleep(0.1)
          continue
        raise

  async def sync_weights(self, model_id: str, weights: np.ndarray, step: int) -> np.ndarray:
    request = node_service_pb2.SyncWeightsRequest(
      model_id=model_id,
      weights=node_service_pb2.Tensor(tensor_data=weights.tobytes(), shape=weights.shape, dtype=str(weights.dtype)),
      step=step
    )
    response = await self._rpc_with_retry(
      "SyncWeights",
      lambda: self.stub.SyncWeights(request),
    )
    return np.frombuffer(response.averaged_weights.tensor_data, dtype=np.dtype(response.averaged_weights.dtype)).reshape(response.averaged_weights.shape)

  async def test_network(self, payload: bytes) -> float:
    request = node_service_pb2.TestNetworkRequest(payload=payload)
    import time
    start = time.perf_counter()
    await self._rpc_with_retry(
      "TestNetwork",
      lambda: self.stub.TestNetwork(request),
    )
    duration_ms = (time.perf_counter() - start) * 1000
    return duration_ms

  async def profile_hardware(self, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray, n_iters: int, skip_iters: int) -> Tuple[float, float]:
    request = node_service_pb2.ProfileHardwareRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      example=self._serialize_tensor(example),
      target=self._serialize_tensor(target),
      length=self._serialize_tensor(length),
      n_iters=n_iters,
      skip_iters=skip_iters,
    )
    # Profiling might take a significant amount of time
    response = await self._rpc_with_retry(
      "ProfileHardware",
      lambda: self.stub.ProfileHardware(request),
      timeout=120.0
    )
    return response.samples_per_sec, response.avg_latency_ms

  def _serialize_tensor(self, tensor: np.ndarray) -> node_service_pb2.Tensor:
    return node_service_pb2.Tensor(
      tensor_data=tensor.tobytes(),
      shape=tensor.shape,
      dtype=str(tensor.dtype)
    )

  def serialize_inference_state(self, inference_state: dict) -> node_service_pb2.InferenceState:
    proto_inference_state = node_service_pb2.InferenceState()
    other_data = {}
    for k, v in inference_state.items():
      if isinstance(v, np.ndarray):
        tensor_data = node_service_pb2.Tensor(tensor_data=v.tobytes(), shape=list(v.shape), dtype=str(v.dtype))
        proto_inference_state.tensor_data[k].CopyFrom(tensor_data)
      elif isinstance(v, list) and all(isinstance(item, np.ndarray) for item in v):
        tensor_list = node_service_pb2.TensorList()
        for tensor in v:
          tensor_data = node_service_pb2.Tensor(tensor_data=tensor.tobytes(), shape=list(tensor.shape), dtype=str(tensor.dtype))
          tensor_list.tensors.append(tensor_data)
        proto_inference_state.tensor_list_data[k].CopyFrom(tensor_list)
      else:
        # For non-tensor data, we'll still use JSON
        other_data[k] = v
    if other_data:
      proto_inference_state.other_data_json = json.dumps(other_data)
    return proto_inference_state
