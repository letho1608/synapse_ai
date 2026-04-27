import asyncio
import uuid
from typing import Optional, Tuple, List, Dict, Any, Set
import numpy as np
from synapse.networking.peer_handle import PeerHandle
from synapse.inference.shard import Shard
from synapse.topology.device_capabilities import DeviceCapabilities
from synapse.topology.topology import Topology
from synapse.routing.event_router import EventRouter, Event
from synapse.helpers import DEBUG
from loguru import logger

class P2PPeerHandle(PeerHandle):
    """
    P2PPeerHandle implements the PeerHandle interface using the EventRouter.
    It performs orchestration calls by publishing events and waiting for responses.
    """
    def __init__(
        self,
        _id: str,
        address: str,
        desc: str,
        device_capabilities: DeviceCapabilities,
        event_router: EventRouter,
        libp2p_node: Any
    ):
        self._id = _id
        self.address = address
        self.desc = desc
        self._device_capabilities = device_capabilities
        self.event_router = event_router
        self.libp2p_node = libp2p_node
        self._pending_requests: Dict[str, asyncio.Future] = {}

    def id(self) -> str:
        return self._id

    def addr(self) -> str:
        return self.address

    def description(self) -> str:
        return self.desc

    def device_capabilities(self) -> DeviceCapabilities:
        return self._device_capabilities

    async def connect(self):
        # Ensure connection is established
        await self.libp2p_node.connect_to_peer(self._id, self.address)

    async def is_connected(self) -> bool:
        return self.libp2p_node.is_peer_connected(self._id)

    async def disconnect(self):
        # TCP connection handled by bridge, nothing to do here
        pass

    async def health_check(self) -> bool:
        try:
            print(f"[DEBUG-HC] {self._id}: connecting to {self.address}...", flush=True)
            connected = await self.libp2p_node.connect_to_peer(self._id, self.address)
            print(f"[DEBUG-HC] {self._id}: connect_to_peer = {connected}", flush=True)
            if not connected:
                if DEBUG >= 2: print(f"Health check failed to connect to {self._id}")
                return False

            print(f"[DEBUG-HC] {self._id}: doing RPC health call...", flush=True)
            res = await self._rpc_call("synapse/rpc/health", {}, timeout=15.0)
            print(f"[DEBUG-HC] {self._id}: RPC result = {res}", flush=True)
            return res.get("status") == "ok"
        except asyncio.TimeoutError:
            print(f"[DEBUG-HC] {self._id}: RPC TIMEOUT!", flush=True)
            if DEBUG >= 2: print(f"Health check RPC TIMEOUT for {self._id} - check P2P event routing")
            return False
        except Exception as e:
            print(f"[DEBUG-HC] {self._id}: ERROR {type(e).__name__}: {e}", flush=True)
            if DEBUG >= 2: print(f"Health check error for {self._id}: {type(e).__name__}: {e}")
            return False

    async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> bool:
        ack_topic = f"synapse/inference/ack/{self._id}/{uuid.uuid4()}"
        payload = {
            "shard": shard.to_dict(),
            "prompt": prompt,
            "request_id": request_id,
            "_ack_topic": ack_topic
        }
        future = asyncio.get_event_loop().create_future()
        def on_ack(event: Event):
            if not future.done():
                future.set_result(True)
        self.event_router.subscribe(ack_topic, on_ack)
        try:
            await self.event_router.publish(f"synapse/inference/prompt/{self._id}", payload)
            await asyncio.wait_for(future, timeout=5.0)
            return True
        except asyncio.TimeoutError:
            if DEBUG >= 1: print(f"send_prompt to {self._id}: ack timeout")
            return False
        finally:
            self.event_router.unsubscribe(ack_topic, on_ack)

    async def send_tensor(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> bool:
        ack_topic = f"synapse/inference/ack/{self._id}/{uuid.uuid4()}"
        payload = {
            "shard": shard.to_dict(),
            "tensor_data": tensor.tobytes(),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "request_id": request_id,
            "_ack_topic": ack_topic
        }
        future = asyncio.get_event_loop().create_future()
        def on_ack(event: Event):
            if not future.done():
                future.set_result(True)
        self.event_router.subscribe(ack_topic, on_ack)
        try:
            await self.event_router.publish(f"synapse/inference/tensor/{self._id}", payload)
            await asyncio.wait_for(future, timeout=5.0)
            return True
        except asyncio.TimeoutError:
            if DEBUG >= 1: print(f"send_tensor to {self._id}: ack timeout")
            return False
        finally:
            self.event_router.unsubscribe(ack_topic, on_ack)

    async def send_example(
        self,
        shard: Shard,
        example: np.ndarray,
        target: np.ndarray,
        length: np.ndarray,
        train: bool,
        request_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        payload = {
            "shard": shard.to_dict(),
            "example": example.tobytes(),
            "example_shape": list(example.shape),
            "example_dtype": str(example.dtype),
            "target": target.tobytes(),
            "target_shape": list(target.shape),
            "target_dtype": str(target.dtype),
            "length": length.tobytes(),
            "length_shape": list(length.shape),
            "length_dtype": str(length.dtype),
            "train": train,
            "request_id": request_id,
        }
        await self.event_router.publish(f"synapse/inference/example/{self._id}", payload)

        # For training, wait for loss response
        if train:
            try:
                res = await self._rpc_call("synapse/rpc/example", payload, timeout=30.0)
                loss = res.get("loss", 0.0)
                grads_data = res.get("grads_data")
                if grads_data is not None:
                    from synapse.inference.shard import Shard
                    grads_shape = res.get("grads_shape", [])
                    grads_dtype = res.get("grads_dtype", "float32")
                    grads = np.frombuffer(grads_data, dtype=np.dtype(grads_dtype)).reshape(grads_shape)
                    return loss, grads
                return loss
            except Exception as e:
                logger.error(f"Failed to get example response from {self._id}: {e}")
                return 0.0
        return None

    async def send_opaque_status(self, request_id: str, status: str) -> None:
        payload = {
            "request_id": request_id,
            "status": status,
        }
        await self.event_router.publish(f"synapse/inference/status/{self._id}", payload)

    async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
        payload = {
            "request_id": request_id,
            "result": result,
            "is_finished": is_finished
        }
        await self.event_router.publish(f"synapse/inference/result/{self._id}", payload)

    async def collect_topology(self, visited: Set[str], max_depth: int) -> Topology:
        payload = {
            "visited": list(visited),
            "max_depth": max_depth
        }
        try:
            res = await self._rpc_call("synapse/rpc/topology", payload)
            return Topology.from_dict(res.get("topology"))
        except Exception as e:
            logger.error(f"Failed to collect topology from {self._id}: {e}")
            return Topology()

    async def _rpc_call(self, topic_base: str, data: Any, timeout: float = 10.0) -> Any:
        """Helper to perform Request/Response over EventRouter."""
        call_id = str(uuid.uuid4())
        request_topic = f"{topic_base}/request/{self._id}"
        response_topic = f"{topic_base}/response/{call_id}"

        print(f"[DEBUG-RPC] call_id={call_id} target={self._id} request_topic={request_topic} response_topic={response_topic}", flush=True)

        future = asyncio.get_event_loop().create_future()
        self._pending_requests[call_id] = future

        # Subscribe to the unique response topic
        def on_response(event: Event):
            print(f"[DEBUG-RPC] on_response fired! call_id={call_id} data={event.data}", flush=True)
            if not future.done():
                if DEBUG >= 3: print(f"[RPC] {self._id} received response on {response_topic}")
                future.set_result(event.data)

        self.event_router.subscribe(response_topic, on_response)

        try:
            # Publish request with the return topic information
            data["_response_topic"] = response_topic
            if DEBUG >= 3: print(f"[RPC] {self._id} publishing to {request_topic}")
            print(f"[DEBUG-RPC] publishing to event_router topic={request_topic}", flush=True)
            await self.event_router.publish(request_topic, data)
            print(f"[DEBUG-RPC] publish returned, waiting for future...", flush=True)
            if DEBUG >= 3: print(f"[RPC] {self._id} waiting for {response_topic}, timeout={timeout}s")
            # Wait for response
            result = await asyncio.wait_for(future, timeout=timeout)
            print(f"[DEBUG-RPC] got result: {result}", flush=True)
            return result
        except asyncio.TimeoutError:
            print(f"[DEBUG-RPC] TIMEOUT after {timeout}s for {topic_base} to {self._id}", flush=True)
            if DEBUG >= 1: print(f"[RPC] TIMEOUT for {topic_base} to {self._id} on {response_topic}")
            raise
        finally:
            self.event_router.unsubscribe(response_topic, on_response)
            self._pending_requests.pop(call_id, None)