# Distributed LLM Testing Plan

## Overview
Test plan for verifying distributed LLM inference after fixing the data distribution bugs.

## Pre-Test Setup

### Environment Requirements
- 2+ machines with Tailscale installed and running
- Python 3.10+ on all machines
- NVIDIA GPU optional (CPU mode supported)
- Network connectivity between machines verified

### Configuration
1. **Machine A (Coordinator)**:
   - Run with `--discovery-module tailscale`
   - API port: 52415
   - Model: qwen2.5:1.5b (small, fast for testing)

2. **Machine B (Worker)**:
   - Run with `--discovery-module tailscale`
   - Same Tailscale network
   - No API server needed (or different port)

### Test Data
- Simple prompt: "Xin chào, bạn là ai?"
- Complex prompt: "Hãy giải thích về machine learning trong 3 câu."
- Long prompt to test streaming

## Test Cases

### Phase 1: Basic Connectivity Tests

#### Test 1.1: Node Discovery
**Objective**: Verify nodes discover each other via Tailscale

**Steps**:
1. Start Machine A with `python main.py`
2. Start Machine B with `python main.py`
3. Check logs for peer discovery

**Expected Results**:
- Machine A logs: `Peers updated: 1 peers`
- Machine B logs: `Peers updated: 1 peers`
- Both nodes appear in each other's peer list

**Pass Criteria**: Both nodes discover each other within 30 seconds

---

#### Test 1.2: Health Check
**Objective**: Verify gRPC connectivity between nodes

**Steps**:
1. After discovery, check health status
2. Use the health check endpoint or manual check

**Expected Results**:
- ✅ `[HEALTH CHECK OK] Node XXX tại IP:PORT đã phản hồi thành công!`

**Pass Criteria**: Health check passes on both nodes

---

### Phase 2: Topology and Partitioning Tests

#### Test 2.1: Topology Collection
**Objective**: Verify topology is correctly collected across nodes

**Steps**:
1. Start 2+ nodes
2. Check `CollectTopology` logs
3. Verify topology JSON structure

**Expected Results**:
- Topology contains all nodes with correct device capabilities
- Peer graph shows connections between nodes
- No timeout errors

**Pass Criteria**: `self.topology` contains all nodes and edges

---

#### Test 2.2: LACP Partitioning
**Objective**: Verify model layers are properly partitioned

**Steps**:
1. Start with model qwen2.5:1.5b (28 layers)
2. Check partitioning output
3. Verify partitions cover all layers

**Expected Results**:
- Partitions created: 2 (one per node)
- Layer ranges: Node A (0-13), Node B (14-27) or similar
- No overlapping or missing layers

**Pass Criteria**: All layers accounted for in partitions

---

### Phase 3: Inference Tests (Critical)

#### Test 3.1: Single Node Inference (Baseline)
**Objective**: Verify single-node mode works before testing distributed

**Steps**:
1. Start only Machine A (1 node)
2. Send API request:
   ```bash
   curl -X POST http://localhost:52415/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Xin chào"}],"stream":false}'
   ```
3. Verify response

**Expected Results**:
- Response contains generated text in Vietnamese
- No errors in logs
- Tokens generated: 50-200

**Pass Criteria**: Valid response with Vietnamese text

---

#### Test 3.2: Distributed Inference - First Layer Node
**Objective**: Verify prompt sent to first layer node works

**Steps**:
1. Start Machine A (first layers) and Machine B (last layers)
2. Send API request to Machine A
3. Check logs for forwarding

**Expected Results**:
- Machine A logs: `[DISTRIBUTED] Forwarding workload to Node: XXX`
- Machine B logs: `[DISTRIBUTED] Receiving PROMPT workload from remote peer`
- Both nodes show activity

**Pass Criteria**: Prompt successfully forwarded to remote node

---

#### Test 3.3: Distributed Inference - Token Flow
**Objective**: Verify tokens flow back to requesting node

**Steps**:
1. Run distributed inference
2. Monitor `on_token` callbacks
3. Check API response

**Expected Results**:
- Machine B (last layer) generates tokens
- Tokens broadcasted back to Machine A
- Machine A receives tokens via `SendResult`
- API returns complete response

**Pass Criteria**: Full response received at API caller

---

#### Test 3.4: Streaming Response
**Objective**: Verify streaming works in distributed mode

**Steps**:
1. Send streaming request:
   ```bash
   curl -X POST http://localhost:52415/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Xin chào"}],"stream":true}'
   ```
2. Check SSE chunks arrive

**Expected Results**:
- SSE chunks: `data: {...}`
- `done: true` at end
- Tokens appear incrementally

**Pass Criteria**: Streaming works end-to-end

---

### Phase 4: Error Handling Tests

#### Test 4.1: Node Failure Recovery
**Objective**: Verify system handles node failure gracefully

**Steps**:
1. Start 2 nodes, begin inference
2. Kill Machine B (Ctrl+C)
3. Check Machine A logs

**Expected Results**:
- Machine A detects node failure
- Logs: `[DISTRIBUTED-ERROR] 🔌 NODE B KHÔNG KHẢ DỤNG`
- No hang or crash, returns error to API

**Pass Criteria**: Graceful error handling, no infinite hang

---

#### Test 4.2: Timeout Handling
**Objective**: Verify timeouts work correctly

**Steps**:
1. Start 2 nodes
2. Block network between nodes (firewall)
3. Send request

**Expected Results**:
- Timeout after 300s (configurable)
- Error returned to API caller
- Log shows timeout reason

**Pass Criteria**: Timeout fires and is handled

---

### Phase 5: Performance Tests

#### Test 5.1: Latency Measurement
**Objective**: Measure distributed vs single-node latency

**Steps**:
1. Run same prompt in single-node mode
2. Run same prompt in distributed mode
3. Compare response times

**Expected Results**:
- Single-node: baseline
- Distributed: slightly higher (network overhead)
- Difference < 2x for small models

**Pass Criteria**: Distributed mode works without excessive overhead

---

#### Test 5.2: Multiple Concurrent Requests
**Objective**: Verify system handles concurrent requests

**Steps**:
1. Send 3 concurrent requests
2. Monitor all complete successfully

**Expected Results**:
- All requests complete
- No mixed responses
- Tokens correctly attributed to each request_id

**Pass Criteria**: All concurrent requests succeed

---

## Test Execution Checklist

### Pre-Test
- [ ] Tailscale running on all machines
- [ ] API keys configured (if using Tailscale discovery)
- [ ] Ports 50051, 52415 open in firewall
- [ ] `requirements.txt` installed on all machines
- [ ] Model qwen2.5:1.5b available or will auto-download

### Phase 1: Connectivity
- [ ] Test 1.1: Node Discovery - PASS/FAIL
- [ ] Test 1.2: Health Check - PASS/FAIL

### Phase 2: Topology
- [ ] Test 2.1: Topology Collection - PASS/FAIL
- [ ] Test 2.2: LACP Partitioning - PASS/FAIL

### Phase 3: Inference
- [ ] Test 3.1: Single Node Baseline - PASS/FAIL
- [ ] Test 3.2: Distributed Forwarding - PASS/FAIL
- [ ] Test 3.3: Token Flow - PASS/FAIL
- [ ] Test 3.4: Streaming - PASS/FAIL

### Phase 4: Error Handling
- [ ] Test 4.1: Node Failure - PASS/FAIL
- [ ] Test 4.2: Timeout - PASS/FAIL

### Phase 5: Performance
- [ ] Test 5.1: Latency - PASS/FAIL
- [ ] Test 5.2: Concurrency - PASS/FAIL

## Log Analysis Guide

### Success Indicators
```
✅ [HEALTH CHECK OK] Node XXX tại IP:PORT đã phản hồi thành công!
[DISTRIBUTED] Forwarding workload to Node: XXX
[DISTRIBUTED] Receiving PROMPT workload from remote peer for Request: ABC123
[RequestID] PROGRESS: Generated 50/10000 tokens.
```

### Error Indicators
```
❌ [HEALTH CHECK FAILED] Node XXX không kết nối được
[DISTRIBUTED-ERROR] ⏰ TIMEOUT (300s)
[DISTRIBUTED-ERROR] 🔌 NODE B KHÔNG KHẢ DỤNG
Error processing prompt: ...
```

## Quick Test Script

Save as `test_distributed.sh` (Linux/Mac) or `test_distributed.bat` (Windows):

```bash
#!/bin/bash
# Quick test for distributed inference

API_URL="http://localhost:52415/v1/chat/completions"

echo "Test 1: Simple prompt"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:1.5b",
    "messages": [{"role": "user", "content": "Xin chào, bạn là ai?"}],
    "stream": false
  }' | jq '.choices[0].message.content'

echo -e "\nTest 2: Streaming"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:1.5b",
    "messages": [{"role": "user", "content": "Chào bạn"}],
    "stream": true
  }' | grep -o '"delta".*"content":"[^"]*"'

echo -e "\nTest 3: Check nodes"
curl -s http://localhost:52415/v1/distributed/status | jq '.'
```

## Post-Test Actions

If all tests pass:
1. Mark fix as complete
2. Run existing test suite: `pytest tests/ -v`
3. Commit changes with message: `fix: distributed inference data flow`

If tests fail:
1. Check logs for specific error
2. Refer to `distributed_fix_plan.md` for which component failed
3. Add additional debugging
4. Re-test
