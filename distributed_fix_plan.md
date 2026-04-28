# Distributed LLM Data Flow Fix Plan

## Problem Summary
When running LLM inference in distributed mode, data is not being properly distributed across nodes. The root cause is a **mismatch between synchronous gRPC expectations and asynchronous token generation**.

## Root Cause
The gRPC RPCs (`SendPrompt`, `SendTensor`) expect to **synchronously return a tensor** to the caller, but the actual processing flow is:
1. `process_prompt()` / `process_tensor()` trigger **asynchronous processing**
2. Tokens are streamed back via `on_token` callbacks and `broadcast_result()`
3. The caller never receives the result via gRPC response

### Specific Issues

| Location | Issue |
|----------|-------|
| `grpc_server.py:SendPrompt` (line 68) | Calls `process_prompt()` but expects immediate tensor return |
| `grpc_server.py:SendTensor` (line 85) | Calls `process_tensor()` but expects immediate tensor return |
| `grpc_peer_handle.py:send_prompt` (line 193) | Waits for gRPC response that never comes properly |
| `grpc_peer_handle.py:send_tensor` (line 211) | Same issue as send_prompt |
| `node.py:forward_prompt` (line 761) | Doesn't properly handle async result collection |
| `node.py:forward_tensor` (line 784) | Same issue as forward_prompt |

## Proposed Solution

### Approach: Use Callback-Based Result Collection
Instead of expecting synchronous gRPC responses, use the existing `on_token` callback system to collect results from remote nodes.

### Changes Required

#### 1. Fix `send_prompt` in `grpc_peer_handle.py`
**File**: `synapse/networking/grpc/grpc_peer_handle.py`  
**Lines**: ~193-209

**Current behavior**: Calls `SendPrompt` RPC and waits for tensor response (which doesn't work properly).

**New behavior**:
- Call `SendPrompt` RPC without expecting a tensor return value
- Register a callback to collect tokens via `on_token` system
- Wait for `is_finished=True` signal

```python
async def send_prompt(self, shard: Shard, prompt: str, inference_state: Optional[dict] = None, request_id: Optional[str] = None) -> None:
    # Send prompt to remote node (fire and forget)
    request = node_service_pb2.PromptRequest(...)
    await self._rpc_with_retry("SendPrompt", lambda: self.stub.SendPrompt(request), timeout=120.0)
    # Result will come via broadcast_result -> on_token callbacks
```

#### 2. Fix `SendPrompt` in `grpc_server.py`
**File**: `synapse/networking/grpc/grpc_server.py`  
**Lines**: ~56-71

**Current behavior**: Tries to return `result.tobytes()` which is None or wrong.

**New behavior**: Return empty Tensor immediately; actual results are sent via `broadcast_result`.

```python
async def SendPrompt(self, request, context):
    shard = Shard(...)
    prompt = request.prompt
    request_id = request.request_id
    inference_state = ...
    
    # Kick off async processing (don't wait for result here)
    asyncio.create_task(self.node.process_prompt(shard, prompt, request_id, inference_state))
    
    # Return immediately - results come via broadcast
    return node_service_pb2.Tensor()
```

#### 3. Fix `send_tensor` in `grpc_peer_handle.py`
**File**: `synapse/networking/grpc/grpc_peer_handle.py`  
**Lines**: ~211-232

**Same pattern as send_prompt**: Use callbacks instead of expecting return value.

#### 4. Fix `SendTensor` in `grpc_server.py`
**File**: `synapse/networking/grpc/grpc_server.py`  
**Lines**: ~73-90

**Same pattern as SendPrompt**: Return immediately, let results flow via callbacks.

#### 5. Verify/Fix Result Broadcasting
**File**: `synapse/orchestration/node.py`  
**Method**: `broadcast_result` (line 988)

Ensure that when a node finishes processing (especially last layer), it properly broadcasts results to all peers including the originator.

#### 6. Fix `forward_prompt` and `forward_tensor` in `node.py`
**File**: `synapse/orchestration/node.py`  
**Lines**: ~761-815

These methods need to:
- Send prompt/tensor to remote node
- Wait for result via callback registration
- Return the collected result

Consider using a `Future` or similar mechanism:

```python
async def forward_prompt(self, base_shard, prompt, request_id, target_index, inference_state):
    target_id = self.partitioning_strategy.partition(self.topology, base_shard)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    
    if target_id == self.id:
        return await self._process_prompt(next_shard, prompt, request_id, inference_state)
    
    target_peer = next((p for p in self.peers if p.id() == target_id), None)
    
    # Create a Future to collect the result
    loop = asyncio.get_running_loop()
    result_future = loop.create_future()
    
    # Register callback to capture result
    def on_result(req_id, tokens, is_finished):
        if req_id == request_id and is_finished:
            result_future.set_result(tokens)
    
    self.on_token.register(f"forward_callback_{request_id}").on_next(on_result)
    
    # Send to remote node
    await target_peer.send_prompt(next_shard, prompt, inference_state=inference_state, request_id=request_id)
    
    # Wait for result
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)
        return result
    finally:
        self.on_token.deregister(f"forward_callback_{request_id}")
```

## Implementation Order

1. **Phase 1: Fix gRPC Server Response Handling**
   - [ ] Modify `SendPrompt` in `grpc_server.py` to return immediately
   - [ ] Modify `SendTensor` in `grpc_server.py` to return immediately
   - [ ] Modify `SendResult` to properly trigger callbacks

2. **Phase 2: Fix gRPC Client (Peer Handle)**
   - [ ] Modify `send_prompt` in `grpc_peer_handle.py` to not expect tensor return
   - [ ] Modify `send_tensor` in `grpc_peer_handle.py` to not expect tensor return

3. **Phase 3: Fix Forwarding Logic**
   - [ ] Update `forward_prompt` in `node.py` to use callback-based result collection
   - [ ] Update `forward_tensor` in `node.py` to use callback-based result collection

4. **Phase 4: Verify Broadcasting**
   - [ ] Ensure `broadcast_result` properly sends to all peers
   - [ ] Verify `on_token` callbacks work across nodes

5. **Phase 5: Testing**
   - [ ] Test with 2 nodes
   - [ ] Test with 3+ nodes
   - [ ] Verify token flow in distributed mode
   - [ ] Check logs for proper forwarding messages

## Files to Modify

| File | Changes |
|------|---------|
| `synapse/networking/grpc/grpc_server.py` | Fix SendPrompt, SendTensor return handling |
| `synapse/networking/grpc/grpc_peer_handle.py` | Fix send_prompt, send_tensor to use callbacks |
| `synapse/orchestration/node.py` | Fix forward_prompt, forward_tensor, verify broadcast_result |

## Verification Steps

1. Start 2+ nodes with Tailscale discovery
2. Send a prompt via API: `curl -X POST http://localhost:52415/v1/chat/completions -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Hello"}]}'`
3. Check logs for:
   - `[DISTRIBUTED] Forwarding workload to Node: XXX`
   - `[DISTRIBUTED] Receiving PROMPT workload from remote peer`
   - Token generation messages
4. Verify response returns properly to API caller

## Additional Notes

- The current code has some debug prints with emojis and Vietnamese text - keep these for troubleshooting
- The `broadcast_result` method already exists and should work - verify it's being called at the right time
- Consider adding more debug logging to trace the full flow from API → Node A → Node B → back
