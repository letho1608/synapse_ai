import asyncio
import uuid
from typing import Optional, Tuple, List, Dict, Any, Set
import numpy as np
from synapse.networking.peer_handle import PeerHandle
from synapse.inference.shard import Shard
from synapse.topology.device_capabilities import DeviceCapabilities
from synapse.topology.topology import Topology
from synapse.routing.event_router import EventRouter, Event
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
        # In P2P system, connection is handled by Libp2pNode/SocketBridge
        pass

    async def is_connected(self) -> bool:
        # Success check logic could be more complex, but for now we trust the mesh
        return True

    async def disconnect(self):
        pass

    async def health_check(self) -> bool:
        try:
            # Proactively ensure bridge connection is initiated
            await self.libp2p_node.connect_to_peer(self.address)
            # Short wait for handshake to complete if it's the first time
            await asyncio.sleep(0.5) 
            
            res = await self._rpc_call("synapse/rpc/health", {}, timeout=5.0)
            if res and res.get("status") == "ok":
                logger.info(f"Verified peer {self._id} at {self.address} is a Synapse node.")
                return True
            logger.warning(f"Peer {self._id} at {self.address} responded but status is not OK.")
            return False
        except asyncio.TimeoutError:
            logger.error(f"Connection failure: Peer {self._id} at {self.address} timed out during health check.")
            return False
        except Exception as e:
            if "Connection refused" in str(e) or "1225" in str(e):
                logger.error(f"Connection failure: {self.address} refused the connection.")
            else:
                logger.error(f"Connection failure: Could not verify peer {self.address}: {e}")
            return False

    async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.ndarray]:
        # For prompt/tensor, we often use the publish-only pattern in Synapse
        # But if the handle requires a return value (like gRPC did), we do RPC.
        # Note: Synapse's event-driven architecture usually broadcasts these.
        payload = {
            "shard": shard.to_dict(),
            "prompt": prompt,
            "request_id": request_id
        }
        await self.event_router.publish(f"synapse/inference/prompt/{self._id}", payload)
        return None # Result comes back via synapse/inference/result/{request_id}

    async def send_tensor(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> Optional[np.ndarray]:
        payload = {
            "shard": shard.to_dict(),
            "tensor_data": tensor.tobytes(),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "request_id": request_id
        }
        await self.event_router.publish(f"synapse/inference/tensor/{self._id}", payload)
        return None

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
        
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[call_id] = future
        
        # Subscribe to the unique response topic
        def on_response(event: Event):
            if not future.done():
                future.set_result(event.data)
        
        self.event_router.subscribe(response_topic, on_response)
        
        try:
            # Publish request with the return topic information
            data["_response_topic"] = response_topic
            await self.event_router.publish(request_topic, data)
            
            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.event_router.unsubscribe(response_topic, on_response)
            self._pending_requests.pop(call_id, None)
