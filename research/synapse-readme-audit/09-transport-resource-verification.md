# Lane 4 — TCP Transport & Resource Measurement Verification

**Agent:** Specialist (TCP Transport & Resource Measurement)  
**Phase:** 2 — Verify fixes for Blocker/High findings from Phase 1 audit  
**Scope:** `tcp_transport.py`, `resource_weight.py`, `partitioner.py`, `latency_matrix.py`, `node.py`  
**Commit audited:** `91ec971` — "Fix all Blocker and High findings from deep-dive audit"  
**Test results:** All 9 dedicated tests pass (3 TCP_NODELAY + 6 resource measurement)

---

## Table of Contents

1. [Summary of Phase 1 Findings](#1-summary-of-phase-1-findings)
2. [Finding 1: TCP_NODELAY on Server Socket](#2-finding-1-tcp_nodelay-on-server-socket)
3. [Finding 2: TCP_NODELAY on Accepted Connections](#3-finding-2-tcp_nodelay-on-accepted-connections)
4. [Finding 3: TCP_NODELAY on Outbound Client Connections](#4-finding-3-tcp_nodelay-on-outbound-client-connections)
5. [Finding 4: Disk I/O Measurement](#5-finding-4-disk-io-measurement)
6. [Finding 5: Power Measurement](#6-finding-5-power-measurement)
7. [Finding 6: Bandwidth Measurement](#7-finding-6-bandwidth-measurement)
8. [Finding 7: All 8 Resource Dimensions in Dataclasses](#8-finding-7-all-8-resource-dimensions-in-dataclasses)
9. [Finding 8: from_device_capabilities Wiring](#9-finding-8-from_device_capabilities-wiring)
10. [Additional Findings Discovered During Verification](#10-additional-findings-discovered-during-verification)
11. [Test Coverage Assessment](#11-test-coverage-assessment)
12. [Executive Summary](#12-executive-summary)
13. [Top 5 Most Concerning Findings](#13-top-5-most-concerning-findings)
14. [Confidence Rating & Ground-Truth Tally](#14-confidence-rating--ground-truth-tally)

---

## 1. Summary of Phase 1 Findings

The Phase 1 deep-dive audit identified two key findings in this lane:

| ID | Finding | Severity | Phase 1 Reference |
|---|---|---|---|
| **H4** | TCP_NODELAY not set on TCP transport — README claims "TCP with Nagle disabled" but `socket.TCP_NODELAY` was never called in `tcp_transport.py` | **High** | `02-pipeline-transport.md:144-146` |
| **H3** | 4 of 8 LACP resource dimensions never measured — Disk I/O, Bandwidth, Power, and remote Latency were always 0 in production | **High** | `05-synthesis.md:123` |

Commit `91ec971` claims to resolve both. This report verifies each fix by reading the actual production source code (not just test files).

---

## 2. Finding 1: TCP_NODELAY on Server Socket

**Claim:** TCP_NODELAY is set on the server's listening socket.  
**File:** `backend/synapse/transport/tcp_transport.py`  
**Severity if broken:** High  

### Code Verification

At line 17, `import socket` was added (not present in the pre-fix version). At lines 50-59, `TCPServer.start()` now reads:

```python
async def start(self) -> int:
    self._server = await asyncio.start_server(
        self._handle_connection, self.host, self.port,
    )
    for sock in self._server.sockets:
        self.port = sock.getsockname()[1]
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # LINE 56
        except Exception:
            pass
        break
    ...
    return self.port
```

**Verdict: FIXED** — `socket.TCP_NODELAY` is set to 1 on the server socket.

### Concerns (Medium)

1. **Only first socket gets TCP_NODELAY (line 58 `break`):** The `for` loop breaks after the first iteration. If `asyncio.start_server` creates multiple sockets (e.g., IPv4 + IPv6 dual-stack binding), only the first receives `TCP_NODELAY`. For a standard `"0.0.0.0"` bind this means one IPv4-only socket, so the concern is minor in practice.

2. **Setting TCP_NODELAY on a listening socket is mostly cosmetic:** `TCP_NODELAY` is a per-connection socket option. On most platforms (Linux, Windows), setting it on a listening socket has no effect on accepted connections — each accepted connection inherits the system default and must be set individually. The real fix is in `_handle_connection` (see Finding 2). The listening-socket setting does no harm but is functionally redundant.

3. **Silent `except Exception: pass`:** Any error during `setsockopt` is swallowed without logging. If `socket.TCP_NODELAY` is somehow not defined (hypothetical exotic platform), the failure is invisible.

**Severity: Medium** (concern about multi-socket scenario and silent failure, but does not affect correctness of primary fix)

---

## 3. Finding 2: TCP_NODELAY on Accepted Connections

**Claim:** TCP_NODELAY is set on each accepted client socket.  
**File:** `backend/synapse/transport/tcp_transport.py`  
**Severity if broken:** High  

### Code Verification

At lines 72-78, `_handle_connection()` now sets TCP_NODELAY before reading frames:

```python
async def _handle_connection(
    self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
):
    peer = writer.get_extra_info("peername")
    try:
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # LINE 77
    except Exception:
        pass
    ...
```

**Verdict: FIXED** ✓ — Every accepted connection gets TCP_NODELAY immediately after the connection handler is invoked.

### Notes

- The `writer.get_extra_info("socket")` may return `None` on some platforms (e.g., when using SSL wrapping or certain asyncio event loop implementations). The `if sock is not None` guard handles this correctly.
- TCP_NODELAY is set before the read loop begins, so all data frames benefit from Nagle-free delivery.

**Severity: No concerns — correctly implemented.**

---

## 4. Finding 3: TCP_NODELAY on Outbound Client Connections

**Claim:** TCP_NODELAY is set on outbound sockets in the client connection pool.  
**File:** `backend/synapse/transport/tcp_transport.py`  
**Severity if broken:** High  

### Code Verification

At lines 141-148, `TCPClientPool._get_connection()` now sets TCP_NODELAY after establishing a new connection:

```python
try:
    writer = await asyncio.wait_for(
        asyncio.open_connection(host, port),
        timeout=self._connect_timeout,
    )
    writer = writer[1]
    try:
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # LINE 146
    except Exception:
        pass
    self._conns[key] = (writer, time.time())
    return writer
except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
    return None
```

**Verdict: FIXED** ✓ — Every new outbound connection gets TCP_NODELAY before it is cached in the pool.

### Notes

- The `writer = writer[1]` extracts the `StreamWriter` from the `(reader, writer)` tuple returned by `asyncio.open_connection`. This is correct.
- The `socket` is retrieved and set before the connection is added to `self._conns`, so all pooled connections have Nagle disabled.
- Reused connections from the pool (lines 124-128) retain their TCP_NODELAY setting from initial setup.

**Severity: No concerns — correctly implemented.**

---

## 5. Finding 4: Disk I/O Measurement

**Claim:** Disk I/O throughput is measured using `psutil.disk_io_counters()`.  
**File:** `backend/synapse/topology/resource_weight.py`  
**Severity if broken:** High  

### Code Verification

The `_measure_disk_io()` function was added at lines 124-151:

```python
def _measure_disk_io() -> Dict[str, float]:
    """Measure disk I/O throughput using psutil."""
    result = {"read_mbps": 0.0, "write_mbps": 0.0}
    if not HAS_PSUTIL:
        return result
    try:
        before = psutil.disk_io_counters()          # LINE 134
        if before is None:
            return result
        time.sleep(0.5)                              # LINE 136
        after = psutil.disk_io_counters()            # LINE 137
        if after is None:
            return result

        elapsed = 0.5  # sleep duration
        read_bytes = after.read_bytes - before.read_bytes
        write_bytes = after.write_bytes - before.write_bytes
        result["read_mbps"] = (read_bytes / (1024 * 1024)) / elapsed
        result["write_mbps"] = (write_bytes / (1024 * 1024)) / elapsed
    except Exception:
        pass
    return result
```

Called from `from_device_capabilities()` at line 99:
```python
disk_metrics = _measure_disk_io()
```

And populated at lines 110-111:
```python
disk_read_mbps=disk_metrics.get("read_mbps", 0.0),
disk_write_mbps=disk_metrics.get("write_mbps", 0.0),
```

**Verdict: FIXED** ✓ — Disk I/O is now measured via `psutil.disk_io_counters()` with before/after sampling, and the results are populated into `NodeResources.disk_read_mbps` and `NodeResources.disk_write_mbps`.

### Concerns (Low)

1. **Hardcoded elapsed time (line 143):** The elapsed time is hardcoded to `0.5` rather than computed from actual clock time. If `psutil.disk_io_counters()` takes measurable time, or if the system clock jumps, this could produce a small error. A more robust approach would be `elapsed = time.perf_counter() - start_time`.

2. **Blocking sleep (line 136):** `time.sleep(0.5)` blocks the calling thread. `from_device_capabilities()` is a synchronous classmethod, so this is technically safe, but if called from an async context — and `_compute_lacp_partition()` in `node.py:1423` is synchronous, so it's fine — it would block the event loop for 0.5 seconds. However, this is called during partitioning, not on a hot path, so the performance impact is acceptable.

3. **Granularity:** 0.5 seconds is a short sampling window for disk I/O counters on low-throughput systems. A longer window (1-2 seconds) would produce more stable readings. The README itself doesn't specify measurement methodology, so this is not a correctness issue.

**Severity: Low** (pragmatic implementation with minor quality concerns)

---

## 6. Finding 5: Power Measurement

**Claim:** Power is measured via `pynvml` for GPU power, with `psutil.sensors_battery()` fallback.  
**File:** `backend/synapse/topology/resource_weight.py`  
**Severity if broken:** High  

### Code Verification

The `_measure_power()` function was added at lines 153-192:

```python
def _measure_power() -> Dict[str, float]:
    result = {"power_limit_w": 0.0, "current_power_w": 0.0}
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()                                     # LINE 163
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)      # LINE 166
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    result["power_limit_w"] = power_limit / 1000.0
                except Exception:
                    pass
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
                    result["current_power_w"] = power_usage / 1000.0
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()                              # LINE 180
            except Exception:
                pass
    elif HAS_PSUTIL:
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                result["current_power_w"] = battery.power_plugged * 50.0
                result["power_limit_w"] = result["current_power_w"] * 1.2
        except Exception:
            pass
    return result
```

Called from `from_device_capabilities()` at line 100:
```python
power_metrics = _measure_power()
```

And populated at lines 113-114:
```python
power_limit_w=power_metrics.get("power_limit_w", 0.0),
current_power_w=power_metrics.get("current_power_w", 0.0),
```

**Verdict: FIXED** ✓ — Power is now measured using pynvml for GPU power limit and current power draw, with a fallback to psutil battery information.

### Concerns (Medium)

1. **Only reads GPU 0 (line 166):** `nvmlDeviceGetHandleByIndex(0)` only reads the first GPU. Multi-GPU nodes will have incomplete power data. The code checks `device_count > 0` but never iterates over multiple GPUs. For a system with 4 GPUs, the total system power draw could be 4× what's reported.

2. **pynvml init/shutdown on every call (lines 163, 180):** `nvmlInit()` and `nvmlShutdown()` are called on every invocation. This is wasteful — `nvmlInit()` should be called once at module load or application start. The repeated init/shutdown pattern risks race conditions and adds latency (~10-50ms each call on some systems).

3. **Power management limit vs. TDP (line 168):** `nvmlDeviceGetPowerManagementLimit()` returns the current power management limit, which may differ from the hardware's true TDP. This is acceptable as a runtime measurement.

4. **psutil battery fallback is a boolean heuristic (lines 186-187):** `psutil.sensors_battery().power_plugged` returns a boolean (True/False). Multiplying by 50.0 gives either 0W or 50W, which is a very rough estimate. The comment acknowledges this ("rough estimate"), but the output appears indistinguishable from real measurement to callers. This could give a false sense of accuracy.

5. **Overwriting in `_compute_lacp_partition()` (node.py:1427):** The `from_device_capabilities()` measurement of `current_power_w` is later overwritten by `m.gpu_power_w` from live metrics in `_compute_lacp_partition()`. This means the pynvml measurement from `_measure_power()` is actually _never used_ for the calling node in the LACP scoring path — it's replaced by the (presumably more current) live metric. For remote nodes, the pynvml measurement is used. This is intentional but undocumented.

**Severity: Medium** (multi-GPU blind spot, wasteful init/shutdown pattern, boolean-based fallback creates false precision)

---

## 7. Finding 6: Bandwidth Measurement

**Claim:** Bandwidth has a default value and is later overwritten by latency probing.  
**File:** `backend/synapse/topology/resource_weight.py`  
**Severity if broken:** High  

### Code Verification

The `_measure_dummy_bandwidth()` function was added at lines 194-202:

```python
def _measure_dummy_bandwidth() -> Dict[str, float]:
    """Return a placeholder bandwidth value.

    Real bandwidth is measured during latency probing (latency_matrix.py)
    and populated into NodeResources.bandwidth_mbps afterward.
    This value is a reasonable default for single-node operation.
    """
    return {"bandwidth_mbps": 1000.0}  # 1 Gbps default, overwritten by probing
```

Called from `from_device_capabilities()` at line 101:
```python
bandwidth_metrics = _measure_dummy_bandwidth()
```

And populated at line 112:
```python
bandwidth_mbps=bandwidth_metrics.get("bandwidth_mbps", 0.0),
```

**Verdict: PARTIALLY FIXED** ⚠️ — A default value (1000 Mbps) is provided for bandwidth, which fixes the "always 0" problem. However, the claim that "real bandwidth is measured during latency probing and populated into NodeResources.bandwidth_mbps afterward" is **not implemented**.

### Evidence of the Gap

1. **`LatencyProber.probe()` in `latency_matrix.py:167-174`** does estimate bandwidth:
   ```python
   transfer_time_s = rtt / 1000.0
   if transfer_time_s > 0:
       bw = (self.PROBE_SIZE * 8) / (transfer_time_s * 1_000_000)
       bandwidths.append(bw)
   ```
   This produces a `LatencyEntry.bandwidth_mbps` value.

2. **However, the production code path in `node.py:1422-1428`** builds a _separate_, simplified latency dict via `_build_latency_matrix()` (line 1428), which returns `Dict[str, Dict[str, float]]` — a plain dict of latencies only, with no bandwidth information. This is the matrix passed to `compute_partition()`.

3. **The `LatencyProber` class is never instantiated or called** from `node.py`'s `_compute_lacp_partition()` method. It exists in `latency_matrix.py` but is wired to no production path.

4. **`_build_latency_matrix()` (node.py:1449-1492)** populates its cache from `_last_forward_latencies` (EWMA forward latencies from tensor forwarding) and optional peer pings. Neither source provides bandwidth data.

5. **`compute_partition()` (partitioner.py:334-485)** and `compute_partition_ilp()` (partitioner.py:98-332) accept `latency_matrix: Optional[LatencyMatrix]` but `node.py` passes a plain `dict`, creating a **type mismatch** (see Additional Finding B).

**Conclusion:** The `NodeResources.bandwidth_mbps` field will always be 1000.0 (the dummy default) in production unless something else overwrites it. The `LatencyProber`'s bandwidth estimation capability exists but is disconnected.

**Severity: High** — The documentation/comment claims real bandwidth measurement, but it's not wired through. The hardcoded 1000 Mbps default is reasonable for a fallback but does not constitute "bandwidth measurement" as claimed.

---

## 8. Finding 7: All 8 Resource Dimensions in Dataclasses

**Claim:** All 8 LACP resource dimensions are represented with non-zero weights.  
**File:** `backend/synapse/topology/resource_weight.py`  
**Severity if broken:** High  

### ResourceWeights Dataclass (lines 219-246)

```python
@dataclass
class ResourceWeights:
    vram: float = 0.25
    tflops: float = 0.25
    cpu: float = 0.10
    ram: float = 0.10
    disk: float = 0.05
    bandwidth: float = 0.10
    latency: float = 0.10
    power_headroom: float = 0.05
```

All 8 dimensions present with non-zero weights. Sum = 1.0 (with `__post_init__` auto-normalization if weights are adjusted).

### NormalizedScores Dataclass (lines 249-259)

```python
@dataclass
class NormalizedScores:
    vram: float = 0.0
    tflops: float = 0.0
    cpu: float = 0.0
    ram: float = 0.0
    disk: float = 0.0
    bandwidth: float = 0.0
    latency: float = 0.0
    power_headroom: float = 0.0
```

All 8 score fields present.

### NodeResources Dataclass (lines 46-78)

All 12 resource fields present:
- GPU: `gpu_name`, `gpu_vram_mb`, `gpu_tflops_fp16`, `gpu_tflops_fp32`, `gpu_count`, `gpu_util_pct`
- CPU: `cpu_cores`, `cpu_model`, `cpu_util_pct`
- Memory: `system_ram_mb`, `system_ram_used_mb`
- Disk: `disk_read_mbps`, `disk_write_mbps`
- Network: `bandwidth_mbps`, `latency_ms`, `jitter_ms`
- Power: `power_limit_w`, `current_power_w`

**Verdict: FIXED** ✓ — All 8 LACP resource dimensions are fully represented across all three dataclasses.

---

## 9. Finding 8: from_device_capabilities Wiring

**Claim:** `from_device_capabilities()` calls all three measurement functions and populates all fields.  
**File:** `backend/synapse/topology/resource_weight.py`, lines 80-115  
**Severity if broken:** High  

### Code Verification

```python
@classmethod
def from_device_capabilities(cls, dc) -> "NodeResources":
    """Convert existing DeviceCapabilities to NodeResources."""
    flops = dc.flops if hasattr(dc, "flops") else {}
    # ... flops extraction ...

    # Measure disk I/O, power, and other resources from live system
    disk_metrics = _measure_disk_io()                          # LINE 99
    power_metrics = _measure_power()                           # LINE 100
    bandwidth_metrics = _measure_dummy_bandwidth()             # LINE 101

    return cls(
        gpu_name=getattr(dc, "chip", "") or getattr(dc, "model", ""),
        gpu_vram_mb=float(getattr(dc, "memory", 0)),
        gpu_tflops_fp16=float(fp16_val),
        gpu_tflops_fp32=float(fp32_val),
        cpu_cores=int(getattr(dc, "cpu_cores", 0) or 0),
        system_ram_mb=float(getattr(dc, "system_ram_mb", 0) or 0),
        disk_read_mbps=disk_metrics.get("read_mbps", 0.0),    # LINE 110
        disk_write_mbps=disk_metrics.get("write_mbps", 0.0),  # LINE 111
        bandwidth_mbps=bandwidth_metrics.get("bandwidth_mbps", 0.0),  # LINE 112
        power_limit_w=power_metrics.get("power_limit_w", 0.0),        # LINE 113
        current_power_w=power_metrics.get("current_power_w", 0.0),    # LINE 114
    )
```

**Verdict: FIXED** ✓ — The method correctly calls all three measurement functions and populates all previously-zero fields.

### Concerns (Low)

1. **`hasattr(dc, "flops")` pattern (line 83):** The method accesses `dc.flops` and then checks if the result has `fp16`/`fp32` attributes or is a dict. This ad-hoc type checking is fragile — it works for the `DeviceCapabilities` pydantic model but would silently default to 0 for unexpected input types.

2. **No `latency_ms` population (line 72):** The `latency_ms` field is initialized to 0.0 and is not populated by `from_device_capabilities()`. It must be set externally (e.g., from latency probing). This is the intended design (latency requires pairwise probing), but there's no validation that `latency_ms > 0` before `compute_scores()` is called.

**Severity: Low** (type-checking fragility and missing latency validation are pre-existing patterns, not regressions)

---

## 10. Additional Findings Discovered During Verification

### Finding A: Latency Matrix Type Mismatch (Blocker)

**File:** `backend/synapse/orchestration/node.py:1428` → `backend/synapse/topology/partitioner.py:336`  
**Severity: Blocker**

`_build_latency_matrix()` (node.py:1449) returns `Optional[Dict[str, Dict[str, float]]]` — a plain nested dict:

```python
def _build_latency_matrix(self) -> Optional[Dict[str, Dict[str, float]]]:
    ...
    return self._latency_cache
```

But `compute_partition()` (partitioner.py:336) expects `Optional[LatencyMatrix]` — a custom dataclass:

```python
def compute_partition(
    nodes: Dict[str, NodeResources],
    latency_matrix: Optional[LatencyMatrix],  # Expects LatencyMatrix, not dict
    ...
```

At node.py line 1428:
```python
latency_matrix = self._build_latency_matrix()
lacp_result = compute_partition(
    nodes=nodes_resources,
    latency_matrix=latency_matrix,  # TYPE MISMATCH: dict passed where LatencyMatrix expected
    ...
)
```

If the returned dict is truthy (≥2 nodes), `compute_partition()` will attempt `len(latency_matrix.nodes)` (partitioner.py:394), which will raise `AttributeError` because a `dict` has no `nodes` attribute.

**Impact:** Any multi-node deployment calling `_compute_lacp_partition()` will crash with `AttributeError` when trying to evaluate the clustering condition. Single-node or test deployments may not trigger this code path.

**Root cause:** Either `_build_latency_matrix()` should return a `LatencyMatrix` object, or `compute_partition()` should accept the plain dict format. The two interfaces are incompatible.

---

### Finding B: Bandwidth Measurement Not Wired Through (High)

**Files:** `backend/synapse/topology/latency_matrix.py:167-174`, `backend/synapse/orchestration/node.py:1422-1436`  
**Severity: High**

The `LatencyProber.probe()` method in `latency_matrix.py` (lines 167-174) estimates bandwidth from probe packet RTT. However, this measurement is never propagated to `NodeResources.bandwidth_mbps` in the production code path:

1. The `LatencyProber` class is not instantiated or used by `node.py`'s `_compute_lacp_partition()`.
2. `_build_latency_matrix()` builds a simplified latency dict from `_last_forward_latencies` (EWMA) and peer pings — both latency-only sources.
3. The `from_device_capabilities()` hardcodes bandwidth to 1000 Mbps (Finding 6).
4. Nothing in the chain ever sets `NodeResources.bandwidth_mbps` to the prober's measured value.

The comment in `_measure_dummy_bandwidth()` ("Real bandwidth is measured during latency probing... and populated into NodeResources.bandwidth_mbps afterward") is misleading — no such propagation exists.

**Impact:** Bandwidth-based resource weighting uses a hardcoded 1000 Mbps default for all nodes, making the `bandwidth` weight dimension effectively meaningless in resource scoring.

---

### Finding C: pynvml Init/Shutdown on Every Power Measurement (Medium)

**File:** `backend/synapse/topology/resource_weight.py:163,180`  
**Severity: Medium**

`_measure_power()` calls `pynvml.nvmlInit()` on every invocation and `nvmlShutdown()` in the `finally` block. This is called once per node during partitioning. For a cluster of 20 nodes, this means 20 init/shutdown cycles.

Standard practice is to call `nvmlInit()` once at module load or application start, and `nvmlShutdown()` at shutdown. The current pattern:
- Adds ~10-50ms latency per call
- Risks race conditions if `_measure_power()` is called concurrently (though currently it's not)
- Is generally contrary to NVML documentation recommendations

---

### Finding D: Power Measurement Only Reads GPU 0 (Medium)

**File:** `backend/synapse/topology/resource_weight.py:166`  
**Severity: Medium**

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Only GPU 0
```

For multi-GPU nodes, only the first GPU's power is measured. The code correctly gets `device_count` but never iterates. Total system power for a 4-GPU node would be underreported by ~75%.

---

### Finding E: Blocking 0.5s Sleep in Synchronous Context (Low)

**File:** `backend/synapse/topology/resource_weight.py:136`  
**Severity: Low**

`time.sleep(0.5)` in `_measure_disk_io()` blocks the calling thread. While `from_device_capabilities()` is synchronous, the 0.5s delay occurs during partitioning — a non-hot-path operation. However, if partitioning is triggered during a request (e.g., during rebalance), the caller is blocked for 0.5s.

---

### Finding F: Server Socket Loop-Then-Break (Low)

**File:** `backend/synapse/transport/tcp_transport.py:53-59`  
**Severity: Low**

```python
for sock in self._server.sockets:
    self.port = sock.getsockname()[1]
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    break  # Only processes first socket
```

Only the first socket gets TCP_NODELAY set. For IPv4-only binds this is fine, but for dual-stack servers this leaves the second socket without the setting. Since TCP_NODELAY on a listening socket is cosmetic (Finding 1, Concern 2), this has no real operational impact, but it's inconsistent with the README claim.

---

## 11. Test Coverage Assessment

### TCP_NODELAY Tests (`backend/tests/test_tcp_nodelay.py`)

| Test | What it verifies | Status |
|---|---|---|
| `test_tcp_server_sets_nodelay` | Server socket has TCP_NODELAY=1 | PASS |
| `test_tcp_connection_handler_sets_nodelay` | Accepted connection has TCP_NODELAY=1 | PASS |
| `test_tcp_client_pool_sets_nodelay` | Client pool outbound socket has TCP_NODELAY=1 | PASS |

The tests use `getsockopt` to verify the flag is actually set on the OS socket, not just that the code calls `setsockopt`. This is proper behavioral testing.

**Coverage gap:** No test verifies behavior when `writer.get_extra_info("socket")` returns `None` (edge case on exotic platforms). No test for multiple server sockets.

### Resource Measurement Tests (`backend/tests/test_resource_measurement.py`)

| Test | What it verifies | Status |
|---|---|---|
| `test_psutil_available` | psutil is importable | PASS |
| `test_disk_io_measurement_function_exists` | `_measure_disk_io()` returns dict with non-negative values | PASS |
| `test_power_measurement_function_exists` | `_measure_power()` returns dict with expected keys | PASS |
| `test_bandwidth_measurement_has_default` | `_measure_dummy_bandwidth()` returns positive default | PASS |
| `test_from_device_capabilities_uses_bare_dc` | All 8 dimension fields present after conversion | PASS |
| `test_resource_scorer_includes_all_dimensions` | `ResourceScorer.compute_scores()` uses all 8 dimensions | PASS |

**Coverage gap:** No test verifies that `_measure_power()` actually calls pynvml functions (mock test). No test for multi-GPU power measurement. No test for the bandwidth propagation path (which would fail, revealing Finding B).

---

## 12. Executive Summary

Commit `91ec971` successfully addresses both primary Phase 1 findings (TCP_NODELAY and resource measurement) in the intended locations. The three TCP_NODELAY calls in `tcp_transport.py` are correct and properly placed — on the server listening socket (line 56), each accepted connection (line 77), and each outbound client connection (line 146). All 3 dedicated tests pass, confirming the flag propagates to the OS socket level.

The resource measurement fix adds three functions to `resource_weight.py`: `_measure_disk_io()` using `psutil.disk_io_counters()`, `_measure_power()` using `pynvml` with `psutil.sensors_battery()` fallback, and `_measure_dummy_bandwidth()` returning a 1000 Mbps default. All are wired into `from_device_capabilities()` and populate the previously-zero fields. All 6 dedicated tests pass. The `ResourceWeights` and `NormalizedScores` dataclasses contain all 8 LACP resource dimensions with non-zero weights.

However, the verification uncovered two significant issues beyond the original scope. **Finding A (Blocker):** A type mismatch between `_build_latency_matrix()` (returns `Dict[str, Dict[str, float]]`) and `compute_partition()` (expects `LatencyMatrix`) will crash any multi-node deployment with `AttributeError`. **Finding B (High):** The bandwidth measurement capability in `LatencyProber` is never wired into `NodeResources.bandwidth_mbps`, so bandwidth scoring always uses the hardcoded 1000 Mbps default. Additional medium-severity concerns exist around pynvml initialization patterns and single-GPU power measurement.

The core fixes for the Phase 1 findings are real and correct. The newly discovered issues are downstream integration bugs in the broader codebase — they affect the _effectiveness_ of the resource scoring rather than the presence of the measurements.

**Overall assessment: Primary fixes verified. Two significant downstream integration defects found (1 Blocker, 1 High).**

---

## 13. Top 5 Most Concerning Findings

1. **[Blocker] Latency Matrix Type Mismatch (`node.py:1428` → `partitioner.py:336`):** `_build_latency_matrix()` returns a plain `dict` but `compute_partition()` expects a `LatencyMatrix` dataclass. Any multi-node deployment will crash with `AttributeError: 'dict' object has no attribute 'nodes'` on line 394 of `partitioner.py`. This is a critical integration defect that makes the production partitioning code path unusable for clusters with ≥2 nodes.

2. **[High] Bandwidth Measurement Disconnected (`latency_matrix.py:167-174` → `resource_weight.py:198-199`):** The `LatencyProber` measures bandwidth but nothing propagates it to `NodeResources.bandwidth_mbps`. The comment in `_measure_dummy_bandwidth()` claims this happens, but it does not. Bandwidth scoring always uses the hardcoded 1000 Mbps default for all nodes, making the bandwidth resource dimension non-functional in practice.

3. **[Medium] pynvml Init/Shutdown on Every Call (`resource_weight.py:163,180`):** `nvmlInit()` and `nvmlShutdown()` are called on every power measurement, contrary to NVML best practices. This adds latency and risks race conditions. For a 20-node cluster at partitioning time, this means 20 full NVML init/shutdown cycles.

4. **[Medium] Power Measurement Only Reads GPU 0 (`resource_weight.py:166`):** Multi-GPU nodes have power underreported. The code reads `device_count` but only samples GPU index 0. For a 4-GPU DGX-style node, the total power draw is ~4× what's reported.

5. **[Note] Server Socket TCP_NODELAY Only on First Socket (`tcp_transport.py:58`):** The `break` after the first `sock` in the loop means only one listening socket gets TCP_NODELAY. This is cosmetic (TCP_NODELAY on a listening socket has no effect on accepted connections), but inconsistent with the README's blanket claim.

---

## 14. Confidence Rating & Ground-Truth Tally

### Confidence: 7/10

**Rationale:** The primary fixes (TCP_NODELAY and resource measurement functions) are clearly present and verified through both code reading and test execution. All 9 dedicated tests pass, and the code changes in the diff directly correspond to the claimed fixes. However, the discovery of two significant integration defects (type mismatch blocker and bandwidth disconnection) lowers confidence because:

- Both defects were missed by the commit's test suite (no test exercises the production `node.py` → `partitioner.py` path end-to-end)
- The bandwidth propagation claim in the docstring is misleading — code exists but isn't wired
- The pynvml init/shutdown pattern and single-GPU blind spot indicate the power measurement was implemented without considering multi-GPU systems or NVML best practices
- The test files verify function _existence_ but not function _integration_ in the production pipeline

Confidence is not higher because of these integration gaps. The fixes work in isolation but may not work correctly in the full production pipeline.

### Ground-Truth Tally

| Category | Count | Details |
|---|---|---|
| **External-checked facts** | 9 | 3 TCP_NODELAY set calls verified at specific lines; 3 measurement functions verified at specific lines; 3 dataclass structures verified |
| **Model judgment claims** | 5 | Type mismatch would crash (deduced from code analysis, not observed); bandwidth not propagated (deduced from trace); pynvml pattern is suboptimal (coding convention); multi-GPU gap (logical inference); server socket cosmetic (network protocol knowledge) |
| **Test results confirmed** | 9/9 | All 3 TCP_NODELAY tests pass; all 6 resource measurement tests pass |

### Map of Verified Lines

| Finding | File | Line(s) | Verified |
|---|---|---|---|
| TCP_NODELAY — server socket | `tcp_transport.py` | 56 | ✓ |
| TCP_NODELAY — accepted conn | `tcp_transport.py` | 77 | ✓ |
| TCP_NODELAY — client outbound | `tcp_transport.py` | 146 | ✓ |
| Disk I/O measurement | `resource_weight.py` | 124-151 | ✓ |
| Disk I/O call in `from_device_capabilities` | `resource_weight.py` | 99, 110-111 | ✓ |
| Power measurement | `resource_weight.py` | 153-192 | ✓ |
| Power call in `from_device_capabilities` | `resource_weight.py` | 100, 113-114 | ✓ |
| Bandwidth default | `resource_weight.py` | 194-202 | ✓ |
| Bandwidth call in `from_device_capabilities` | `resource_weight.py` | 101, 112 | ✓ |
| ResourceWeights — all 8 dims | `resource_weight.py` | 221-229 | ✓ |
| NormalizedScores — all 8 dims | `resource_weight.py` | 251-258 | ✓ |
| Type mismatch (Blocker) | `node.py:1428` → `partitioner.py:336,394` | ✗ FAIL |
| Bandwidth not propagated (High) | `latency_matrix.py:167-174` → `resource_weight.py:198` | ✗ FAIL |
| pynvml init/shutdown per call (Med) | `resource_weight.py` | 163, 180 | NOTED |
| Power only reads GPU 0 (Med) | `resource_weight.py` | 166 | NOTED |
| Server socket loop-break (Low) | `tcp_transport.py` | 58 | NOTED |

---

*Report generated by Lane 4 specialist agent. Research only — no source code was modified.*
