# Kế Hoạch Sửa Lỗi Tận Gốc Hệ Thống P2P Synapse AI

Bản kế hoạch này tập trung vào việc giải quyết các lỗi kiến trúc cốt lõi gây ra tình trạng mất kết nối, trùng ID và loạn Master election. Mục tiêu là đảm bảo hệ thống chạy ổn định 100% trên cả môi trường đơn máy (nhiều node) và đa máy (mạng LAN/Tailscale).

## 1. Danh Sách Lỗi & Nguyên Nhân Gốc (Root Causes)

| Lỗi | Nguyên Nhân Gốc | Hậu Quả |
| :--- | :--- | :--- |
| **"Tôi gửi tôi" (Self-ID)** | File ID tạm dùng chung trong thư mục Temp của OS. | Các node trên cùng máy từ chối kết nối nhau. |
| **Loạn Master (Flapping)** | Trọng số bầu chọn phụ thuộc vào tải tức thời (CPU/GPU load). | Master thay đổi liên tục mỗi khi có tác vụ AI chạy. |
| **Timeout 15s / 121** | TCP Head-of-Line Blocking: Control packet kẹt sau Data packet. | Node bị coi là sập khi đang truyền dữ liệu nặng. |
| **Phân xử lỗi (Tiebreaker)** | Logic đóng kết nối khi trùng ID không dọn dẹp sạch tài nguyên. | Treo socket, rò rỉ bộ nhớ và deadlock kết nối. |

---

## 2. Giải Pháp Sửa Lỗi Tận Gốc (Hard Fixes)

### A. Định danh Node Định hướng (Deterministic Identity)
*   **Giải pháp:** Xóa bỏ file `.synapse_node_id` cũ. Tạo ID mới bằng cách băm: `Hash(Địa chỉ MAC + Cổng lắng nghe)`.
*   **Tệp ảnh hưởng:** `synapse/helpers.py`, `synapse/main.py`.
*   **Lợi ích:** Đảm bảo mỗi Node trên cùng một máy luôn có ID duy nhất (vì Port là duy nhất) và cố định (không đổi khi restart).

### B. Ổn định Master Election (Capacity-based Election)
*   **Giải pháp:** 
    *   Thay đổi trọng số bầu chọn: Sử dụng **Tổng VRAM + Tổng FLOPs** (Năng lực phần cứng) thay vì độ rảnh rỗi.
    *   Thêm cơ chế **Master Stickiness**: Một Node chỉ mất quyền Master nếu nó thực sự Offline. Node mới vào dù mạnh hơn cũng không được "cướp ngôi" ngay lập tức trừ khi Master cũ bị lỗi.
*   **Tệp ảnh hưởng:** `synapse/orchestration/election.py`.

### C. Tách biệt Kênh Điều khiển (Priority Control Plane)
*   **Giải pháp:** 
    *   Sử dụng cơ chế đọc gói tin không chặn (Non-blocking packet reading).
    *   Đảm bảo các gói tin nhỏ (Ping, Pong, Election) được xử lý ngay lập tức mà không cần chờ gói tin Tensor lớn được đọc xong hoàn toàn.
*   **Tệp ảnh hưởng:** `synapse/routing/p2p_socket_bridge.py`.

### D. Giao thức Truyền tải Nhị phân (Binary Transport)
*   **Giải pháp:** Sử dụng định dạng nhị phân thay vì Hex-string trong JSON. 
*   **Cấu trúc gói mới:** `[4 bytes length][1 byte type][Payload]`. Với Payload của Tensor sẽ là dữ liệu binary thô (`raw bytes`).
*   **Tệp ảnh hưởng:** `synapse/routing/p2p_socket_bridge.py`.

---

## 3. Các Bước Thực Hiện

### Bước 1: Dọn dẹp và Tái định danh
1.  Viết script tự động tìm và xóa tất cả các file `.synapse_node_id` cũ trong thư mục Temp.
2.  Cập nhật hàm `get_or_create_node_id` trong `helpers.py` để lấy mã MAC.
3.  Cập nhật `main.py` để truyền `node_port` vào trình tạo ID.

### Bước 2: Cải tiến P2P Socket Bridge
1.  Refactor lại hàm `_read_packet` để có thể nhận diện nhanh loại gói tin.
2.  Sửa lỗi tiebreaker: Đảm bảo khi đóng kết nối trùng, mọi task liên quan đều được `cancel()` và socket được đóng sạch.

### Bước 3: Cập nhật Election Manager
1.  Viết lại hàm `_update_compute_weight` để tính toán dựa trên cấu hình phần cứng tĩnh.
2.  Triển khai logic Master Stickiness.

---

## 4. Kế Hoạch Kiểm Thử (Verification)

1.  **Chạy đa node nội bộ:** Mở 3 terminal trên cùng 1 máy, chạy 3 node với các port khác nhau (50051, 50052, 50053). Kiểm tra xem chúng có nhận diện nhau là các peer riêng biệt không.
2.  **Kiểm tra Master:** Chạy một tác vụ nặng trên Master, quan sát xem Master có bị thay đổi (flapping) không.
3.  **Kiểm tra truyền tải:** Gửi một Tensor lớn qua Tailscale và kiểm tra xem các gói tin Ping/Pong có bị timeout không.

> [!IMPORTANT]
> Giải pháp này tập trung vào việc sửa lại **logic cốt lõi** của hệ thống, không dựa vào việc tăng timeout để "che đậy" lỗi mạng.
