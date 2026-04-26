# Kế hoạch sửa lỗi: Health Check không giữ trạng thái kết nối Node

## Hiện tượng quan sát

Từ log runtime, tôi thấy pattern lặp đi lặp lại:

```
✅ [HEALTH CHECK OK] Node 533701d2... tại 100.115.83.109:50051 đã phản hồi thành công!
✅ [HEALTH CHECK OK] Node af3c1af0... tại 100.74.96.93:50051 đã phản hồi thành công!
✅ [HEALTH CHECK OK] Node 533701d2... tại 100.115.83.109:50051 đã phản hồi thành công!  ← lặp lại mãi
✅ [HEALTH CHECK OK] Node af3c1af0... tại 100.74.96.93:50051 đã phản hồi thành công!  ← lặp lại mãi
```

Mặc dù health check thành công, **node vẫn liên tục bị ping lại** như thể nó chưa từng được "nhớ" là đang connected. Điều này cho thấy trạng thái kết nối **không được duy trì** giữa các lần kiểm tra.

---

## Phân tích nguyên nhân gốc rễ

### Vấn đề 1: `health_check()` không gọi `connect()` — nhưng `_ensure_connected()` lại tạo channel mới mỗi lần

Ở [grpc_peer_handle.py:170-191](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/grpc/grpc_peer_handle.py#L170-L191):

```python
async def health_check(self) -> bool:
    await self._ensure_connected()  # ← gọi connect() nếu channel không READY
    request = node_service_pb2.HealthCheckRequest()
    response = await asyncio.wait_for(self.stub.HealthCheck(request), timeout=5)
    return response.is_healthy
```

`_ensure_connected()` kiểm tra `is_connected()`, mà `is_connected()` chỉ trả `True` khi `channel.get_state() == READY`.

**Vấn đề**: gRPC channel state tự chuyển từ `READY` → `IDLE` sau khoảng 10-30 giây không activity (do `keepalive_time_ms` và HTTP2 idle). Khi state = `IDLE`:
- `is_connected()` trả `False`
- `_ensure_connected()` gọi `connect()` → **đóng channel cũ, tạo channel mới**
- Health check thành công nhưng channel mới lại sẽ chuyển IDLE → vòng lặp lặp lại mãi

### Vấn đề 2: `check_peer()` xóa node dựa trên `is_connected() = False` — dù health check OK

Ở [tailscale_discovery.py:322-334](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/tailscale/tailscale_discovery.py#L322-L334):

```python
async def check_peer(self, peer_id, current_time):
    is_connected = await peer_handle.is_connected()
    health_ok = await peer_handle.health_check()

    should_remove = (
        (not is_connected and current_time - connected_at > self.discovery_timeout)
        or (current_time - last_seen > self.discovery_timeout)
        or (not health_ok)
    )
```

**Thứ tự gọi gây lỗi logic:**
1. `is_connected()` kiểm tra **trước** → trả `False` (channel đang IDLE)
2. `health_check()` kiểm tra **sau** → gọi `_ensure_connected()` → tạo channel mới → ping OK
3. Nhưng `is_connected` đã là `False` từ bước 1
4. Nếu `current_time - connected_at > discovery_timeout` (30s) → **node bị xóa** dù health check vừa thành công!

### Vấn đề 3: Duplicate node entry — cùng IP nhưng khác ID (hostname vs UUID)

Từ log:
```
✅ Node dodat.tailf0c6fd.ts.net tại 100.115.83.109:50051 đã phản hồi
✅ Node 533701d2-bb6a-4aa7-b93f-f8e9e96b42e2 tại 100.115.83.109:50051 đã phản hồi
```

Cùng IP `100.115.83.109:50051` nhưng **2 entry**: hostname `dodat.tailf0c6fd.ts.net` và UUID `533701d2...`. Cleanup cuối chu kỳ có logic dedup nhưng nó chạy **sau** khi cả hai đã được add vào → gây health check chồng chéo.

### Vấn đề 4: Health check gọi quá nhiều lần — hao tổn tài nguyên

Mỗi chu kỳ discovery (5s), **mỗi node** bị health check:
1. Lần 1: Trong `task_discover_peers()` khi add/update peer (line 168, 174, 220)
2. Lần 2: Trong `task_cleanup_peers()` → `check_peer()` (line 328)
3. Lần 3 (nếu DEBUG >= 2): Trong log status (line 307)

→ Mỗi node bị health check **2-3 lần / 5 giây**, gây spam log và lãng phí kết nối.

---

## Kế hoạch sửa lỗi

### Bước 1: Sửa `is_connected()` — chấp nhận IDLE là "vẫn connected"

**File**: [grpc_peer_handle.py](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/grpc/grpc_peer_handle.py#L95-L101)

```diff
  async def is_connected(self) -> bool:
    if self.channel is None:
      return False
    state = self.channel.get_state()
-   return state == grpc.ChannelConnectivity.READY
+   # IDLE và READY đều là trạng thái hợp lệ — channel có thể tự chuyển READY→IDLE
+   # khi không có activity, nhưng vẫn có thể dùng lại ngay
+   return state in (grpc.ChannelConnectivity.READY, grpc.ChannelConnectivity.IDLE)
```

**Kiểm chứng**: Sau fix, `is_connected()` trả `True` cho channel IDLE → `_ensure_connected()` không tạo channel mới liên tục.

---

### Bước 2: Sửa `check_peer()` — health_check thành công phải cập nhật `last_seen`

**File**: [tailscale_discovery.py](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/tailscale/tailscale_discovery.py#L322-L334)

```diff
  async def check_peer(self, peer_id, current_time):
    peer_handle, connected_at, last_seen = self.known_peers.get(peer_id, (None, None, None))
    if peer_handle is None: return False

    try:
-     is_connected = await peer_handle.is_connected()
      health_ok = await peer_handle.health_check()
    except Exception as e:
      if DEBUG_DISCOVERY >= 2: print(f"Error checking peer {peer_id}: {e}")
      return True

-   should_remove = ((not is_connected and current_time - connected_at > self.discovery_timeout) or (current_time - last_seen > self.discovery_timeout) or (not health_ok))
+   if health_ok:
+     # Health check thành công → cập nhật last_seen, giữ node
+     self.known_peers[peer_id] = (peer_handle, connected_at, current_time)
+     return False
+   
+   # Health check thất bại → xóa node nếu quá timeout
+   should_remove = (current_time - last_seen > self.discovery_timeout)
    return should_remove
```

Cùng fix cho [udp_discovery.py](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/udp/udp_discovery.py#L251-L263) tương tự.

**Kiểm chứng**: Khi health check thành công, `last_seen` được cập nhật → node không bị timeout xóa.

---

### Bước 3: Sửa `_ensure_connected()` — không destroy channel đang IDLE

**File**: [grpc_peer_handle.py](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/grpc/grpc_peer_handle.py#L146-L168)

```diff
  async def _ensure_connected(self):
-   if not (await self.is_connected()):
+   if self.channel is None or self.stub is None:
      try:
        await asyncio.wait_for(self.connect(), timeout=10.0)
```

Kết hợp với Bước 1, `is_connected()` đã chấp nhận IDLE. Nhưng `_ensure_connected()` nên kiểm tra đơn giản hơn: chỉ reconnect khi **channel = None** hoặc **stub = None**.

**Kiểm chứng**: Channel IDLE không bị close+reconnect, giảm log spam health check.

---

### Bước 4: Giảm health check thừa trong `task_discover_peers()`

**File**: [tailscale_discovery.py](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/tailscale/tailscale_discovery.py#L218-L223)

Khi peer đã có trong `known_peers` và cùng địa chỉ (case "TRÙNG ID"), **không cần health check lại** — `task_cleanup_peers` đã làm việc đó rồi.

```diff
          else:
-           # [CASE TRÙNG ID] - Chỉ cần cập nhật timestamp và health check
-           handle = self.known_peers[peer_id][0]
-           if not await handle.health_check():
-             if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.")
-             if peer_id in self.known_peers: del self.known_peers[peer_id]
-             continue
+           # [CASE TRÙNG ID] - Chỉ cập nhật last_seen timestamp
+           # Health check sẽ do task_cleanup_peers xử lý riêng
+           handle = self.known_peers[peer_id][0]
```

**Kiểm chứng**: Giảm ~50% số lần health check mỗi chu kỳ, log sạch hơn.

---

### Bước 5: Xử lý duplicate hostname/UUID sớm hơn

**File**: [tailscale_discovery.py](file:///d:/CODE/test/nckh/synapse_ai/synapse/networking/tailscale/tailscale_discovery.py#L158-L164)

Trước khi tạo peer handle mới, kiểm tra xem IP đã có trong `known_peers` chưa. Nếu đã có UUID cho IP đó, skip hostname fallback hoàn toàn:

```diff
          # Nếu IP này đã tồn tại với UUID, không cần thêm bằng hostname
+         existing_uuid_for_ip = next(
+           (pid for pid, (h, _, _) in self.known_peers.items()
+            if h.addr() == f"{peer_host}:{peer_port}" and _is_uuid(pid)),
+           None
+         )
+         if existing_uuid_for_ip and not _is_uuid(peer_id):
+           # Đã có UUID entry cho IP này, skip hostname entry
+           continue
```

**Kiểm chứng**: Không còn thấy 2 entry cho cùng IP trong log.

---

## Tóm tắt thứ tự thực hiện

| Bước | File | Mục tiêu | Kiểm chứng |
|------|------|----------|-------------|
| 1 | `grpc_peer_handle.py` | `is_connected()` chấp nhận IDLE | Chạy → không thấy reconnect liên tục |
| 2 | `tailscale_discovery.py`, `udp_discovery.py` | `check_peer()` cập nhật `last_seen` khi health OK | Node không bị remove sau 30s |
| 3 | `grpc_peer_handle.py` | `_ensure_connected()` chỉ reconnect khi channel=None | Giảm log spam |
| 4 | `tailscale_discovery.py` | Bỏ health check thừa trong discover loop | Giảm ~50% health check calls |
| 5 | `tailscale_discovery.py` | Skip hostname khi đã có UUID entry | Không còn duplicate entry |

> [!IMPORTANT]
> **Bước 1 + 2 là quan trọng nhất** — chúng giải quyết trực tiếp bug "health check OK nhưng không giữ trạng thái connected". Bước 3-5 là tối ưu phụ.
