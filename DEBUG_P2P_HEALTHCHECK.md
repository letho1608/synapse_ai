# Nhật Ký Sửa Lỗi & Tối Ưu Hóa Synapse AI

## Ngày thực hiện

2026-04-26 (chẩn đoán) → 2026-04-27 (hoàn tất, xác minh đa node)

---

## Tổng quan

Dự án Synapse AI — hệ thống phân tán cho phép nhiều máy tính cùng chạy chung một mô hình AI. Khi 2 máy khởi động cùng lúc, chúng không thể giao tiếp được với nhau mặc dù đã kết nối mạng thành công. Nguyên nhân gốc rễ là lỗi "bắt tay" giữa các máy khi cả hai cùng kết nối đến nhau một cách đồng thời.

Sau khi sửa lỗi nền tảng đó, đã tiến hành kiểm tra toàn diện và phát hiện thêm 9 điểm yếu trong thuật toán. Tất cả đã được vá.

---

## Vòng 1: Lỗi Nền Tảng — 2 Máy Không Nói Chuyện Được Với Nhau

### Vấn đề

Tưởng tượng 2 người cùng gọi điện cho nhau một lúc. Mỗi người đều cầm máy gọi đi, nhưng không ai chịu nhấc máy của người kia. Cả hai đều nói vào ống nghe của mình nhưng không ai nghe thấy ai. Hệ thống rơi vào bế tắc hoàn toàn.

Khi 2 máy gọi `connect_to_peer()` cùng lúc, mỗi bên tạo ra một kết nối riêng. Code cũ khiến mỗi máy giữ kết nối "gọi đi" của mình, nhưng đây là 2 đường dây khác nhau. Dữ liệu máy A gửi đi đến một đường dây mà máy B không đọc — và ngược lại. Hậu quả: yêu cầu kiểm tra sức khỏe (health check) luôn hết thời gian chờ.

### Cách sửa (3 thay đổi)

1. **Luật trọng tài (tiebreaker):** Máy nào có ID nhỏ hơn thì được quyền giữ kết nối "gọi đi". Máy có ID lớn hơn phải dùng kết nối "gọi đến". Như vậy cả hai dùng chung một đường dây duy nhất, giao tiếp hai chiều hoạt động bình thường.

2. **Không ghi đè kết nối đã có:** Máy có ID nhỏ hơn chỉ lưu tạm kết nối đến nếu chưa kịp gọi đi. Tuyệt đối không xóa kết nối đã thiết lập xong.

3. **Dọn dẹp an toàn:** Khi một vòng đọc dữ liệu kết thúc, chỉ xóa kết nối nếu nó vẫn là kết nối hiện tại — tránh xóa nhầm kết nối mới đã được thay thế.

---

## Vòng 2: 9 Điểm Yếu Thuật Toán

### 1. Không có heartbeat — không phát hiện máy chết

Trước đây không có cơ chế kiểm tra máy còn sống hay không. Nếu một máy âm thầm mất kết nối, các máy khác không hề hay biết.

**Sửa:** Cứ mỗi 5 giây gửi một tín hiệu "ping", máy kia phải trả lời "pong". Nếu sau 15 giây không thấy trả lời → đánh dấu máy đó đã chết và bắt đầu thử kết nối lại.

### 2. Không tự kết nối lại khi mất mạng

Khi đường dây đứt, hệ thống không cố gắng nối lại.

**Sửa:** Khi phát hiện mất kết nối (do heartbeat timeout hoặc lỗi ghi dữ liệu), lập tức lên lịch kết nối lại. Dùng chiến lược "lùi dần": thử lại sau 1 giây, rồi 2, 4, 8... tối đa 60 giây. Tránh gọi dồn dập gây quá tải mạng.

### 3. Gửi tin nhắn không có xác nhận

`send_prompt()` và `send_tensor()` luôn trả về `None` — người gửi không biết tin đã đến nơi hay chưa.

**Sửa:** Người nhận lập tức gửi lại tín hiệu "đã nhận". Người gửi đợi tối đa 5 giây. Nếu có xác nhận → trả về `True`, nếu hết thời gian → trả về `False`.

### 4. Không giới hạn số yêu cầu đồng thời

Không có cơ chế kiểm soát lượng việc xử lý cùng lúc, dễ gây quá tải khi nhiều máy cùng gửi yêu cầu.

**Sửa:** Thêm giới hạn tối đa 32 yêu cầu xử lý cùng lúc. Yêu cầu thứ 33 phải xếp hàng đợi đến lượt. Có thể tùy chỉnh giới hạn này.

### 5. Nhận diện GPU còn hạn chế

Chỉ nhận diện được khoảng 60 loại card màn hình, nhiều dòng mới không có trong danh sách. Ngoài ra, thuật toán tìm kiếm bị lỗi khớp một phần (ví dụ "A10" khớp nhầm với "A100").

**Sửa:** Mở rộng lên ~108 loại, thêm vào tất cả các dòng mới: H100, H200, B100, B200, L40S, L4, RTX Ada Generation, Intel Arc, AMD Instinct MI300X, AMD Radeon RX 7000, Apple Silicon M1-M4. Sửa lỗi khớp một phần.

### 6. Trọng lượng bầu cử cố định

Trước đây mỗi máy có một điểm số cố định để bầu chọn máy trưởng. Máy có GPU mạnh thì điểm cao, nhưng điểm này không thay đổi dù máy đang bận 100%.

**Sửa:** Điểm số giờ thay đổi theo mức độ bận rộn thực tế. Cứ mỗi 2 giây, kiểm tra mức sử dụng GPU và CPU. Máy rảnh → giữ nguyên điểm. Máy bận 100% → điểm giảm còn 10%. Máy trưởng luôn là máy khỏe nhất và rảnh nhất.

### 7. Khóa bảo mật bị lộ trong code

Tailscale API key và email bị viết cứng trong mã nguồn — lỗ hổng bảo mật nghiêm trọng.

**Sửa:** Xóa sạch. Hệ thống giờ dùng biến môi trường.

### 8. Crash khi một máy rời mạng

Hàm `get_partition_index()` ném lỗi `ValueError` khi một node không còn trong danh sách phân vùng → crash toàn bộ.

**Sửa:** Trả về 0 thay vì crash, kèm dòng log cảnh báo. Hệ thống tiếp tục chạy bình thường.

### 9. Kết nối một chiều bị bỏ qua

Khi chỉ có máy A gọi đến máy B (máy B không gọi lại, như khi một máy tham gia muộn), máy có ID lớn hơn đóng kết nối của mình và chờ kết nối từ đối phương — nhưng kết nối đó không bao giờ tới.

**Sửa:** Nếu không có kết nối đến từ đối phương, giữ lại kết nối đi của mình thay vì chờ đợi vô ích.

---

## Vòng 3: Lỗi Runtime — Bầu Cử Không Hoạt Động

### Vấn đề

Khi thêm tính năng "trọng lượng động" (Vòng 2, mục 6), code đã dùng 2 biến `dynamic_weight` và `base_compute_weight` nhưng quên khai báo chúng trong hàm khởi tạo. Kết quả: mỗi chu kỳ bầu cử đều gây ra lỗi `AttributeError`.

Lỗi này bị khối `try/except` nuốt mất một cách âm thầm. Vòng lặp bầu cử không bao giờ chạy đến bước phát tín hiệu heartbeat. Toàn bộ cơ chế bầu máy trưởng bị tê liệt — không ai biết vì không có log nào báo lỗi nghiêm trọng.

### Cách sửa

Khai báo đầy đủ 2 biến trong hàm khởi tạo, đặt giá trị mặc định hợp lý.

---

## Bảng Tổng Kết 15 Lần Sửa

| # | Vấn đề | File | Phân loại |
|---|--------|------|-----------|
| 1 | Thiếu thư viện numpy | requirements.txt | Hạ tầng |
| 2 | Xung đột cổng UDP | tests | Hạ tầng |
| 3 | Sai tên model ID | synapse/models.py | Hạ tầng |
| 4 | Thư viện giả py-libp2p | requirements.txt | Hạ tầng |
| 5 | Bế tắc bắt tay TCP | synapse/routing/p2p_socket_bridge.py | Nền tảng |
| 6 | Xóa nhầm kết nối mới | synapse/routing/p2p_socket_bridge.py | Nền tảng |
| 7 | Thiếu heartbeat & tự kết nối lại | synapse/routing/p2p_socket_bridge.py | Độ tin cậy |
| 8 | Gửi tin không có xác nhận | synapse/networking/p2p_peer_handle.py | Độ tin cậy |
| 9 | Không giới hạn tải | synapse/orchestration/node.py | Độ tin cậy |
| 10 | Nhận diện GPU hạn chế | synapse/topology/device_capabilities.py | Năng lực |
| 11 | Trọng lượng bầu cử cố định | synapse/orchestration/election.py | Tối ưu |
| 12 | Lộ API key trong code | synapse/orchestration/node.py | Bảo mật |
| 13 | Crash khi node rời mạng | synapse/orchestration/node.py | Ổn định |
| 14 | Kết nối một chiều bị bỏ qua | synapse/routing/p2p_socket_bridge.py | Nền tảng |
| 15 | Biến bầu cử bị thiếu | synapse/orchestration/election.py | Runtime |

### Phân loại theo mức độ

| Mức | Số lượng | Mô tả |
|-----|----------|-------|
| Nền tảng | 4 | Hệ thống không hoạt động nếu thiếu |
| Độ tin cậy | 3 | Hoạt động nhưng không ổn định |
| Tối ưu | 1 | Hoạt động nhưng chưa hiệu quả |
| Bảo mật | 1 | Nguy cơ bị xâm nhập |
| Ổn định | 1 | Crash không mong muốn |
| Năng lực | 1 | Hạn chế phần cứng hỗ trợ |
| Runtime | 1 | Lỗi âm thầm khi chạy |
| Hạ tầng | 3 | Thiếu phụ thuộc, cấu hình sai |

---

## Các File Được Thay Đổi

| File | Vai trò | Số lần sửa |
|------|---------|------------|
| `synapse/routing/p2p_socket_bridge.py` | Cầu nối TCP giữa các máy | 4 |
| `synapse/orchestration/election.py` | Bầu máy trưởng | 2 |
| `synapse/orchestration/node.py` | Điều phối toàn bộ hệ thống | 3 |
| `synapse/networking/p2p_peer_handle.py` | Gửi/nhận dữ liệu giữa các máy | 1 |
| `synapse/topology/device_capabilities.py` | Đo năng lực phần cứng | 1 |
| `requirements.txt` | Danh sách thư viện cần cài | 2 |
| `tests/` | Bộ kiểm tra | 4 file mới |

---

## Bộ Kiểm Tra — 14/14 Đạt

### Kiểm tra cơ bản (Vòng 1)

| # | File | Số máy | Kịch bản |
|---|------|--------|----------|
| 1 | test_p2p_minimal.py | 2 | Kết nối TCP một chiều |
| 2 | test_inprocess.py | 2 | Kiểm tra sức khỏe tuần tự |
| 3 | test_simultaneous.py | 2 | Kết nối đồng thời + 3 vòng stress |
| 4 | test_cluster_inprocess.py | 2 | Cluster đầy đủ: khám phá, bầu cử, topology, 10 vòng stress |
| 5 | test_three_nodes.py | 3 | 6 kết nối đồng thời, kiểm tra toàn bộ, 3 vòng stress |

### Kiểm tra biên (Vòng 2)

| # | Kịch bản | Mục đích |
|---|----------|----------|
| 6 | Chu kỳ ping/pong | Xác minh heartbeat hoạt động |
| 7 | Tự kết nối lại | Ngắt kết nối → kết nối lại trong vòng 5 giây |
| 8 | Lưới 4 máy | 12 kết nối đồng thời + 2 vòng kiểm tra |
| 9 | Tham gia muộn | Máy C vào sau khi A-B đã tạo lưới |
| 10 | Gửi tin thất bại | Gửi prompt/tensor đến máy chết → trả về False |

### Kiểm tra bầu cử (Vòng 3)

| # | Kịch bản | Mục đích |
|---|----------|----------|
| 11 | Vòng bầu cử chạy | Xác minh bầu cử đơn node với trọng lượng động |
| 12 | Hòa điểm | 3 máy bằng điểm → thắng theo thứ tự tên |
| 13 | Dọn máy chết | Máy hết hạn (6 giây) bị loại khỏi danh sách |
| 14 | Tắt trọng lượng động | Xác minh điểm cố định khi dynamic_weight=False |

---

## Kết Quả Xác Minh

- Hình thành lưới (mesh): dưới 0.5 giây
- Bầu máy trưởng: máy có điểm cao nhất được chọn
- Đồng bộ topology: hai chiều qua RPC
- Kiểm tra đồng thời 10 yêu cầu: tất cả đạt
- Lưới 3 máy: tất cả các cặp đều kết nối, 3 vòng kiểm tra toàn bộ đều đạt
- Kết nối lại sau mất mạng: ~1.5 giây
- Phát hiện gửi tin thất bại: chính xác, trả về False sau 5 giây

---

## Cách Chạy

```bash
cd D:\Code\synapse_ai

# Kiểm tra đơn lẻ
python tests/test_p2p_minimal.py
python tests/test_inprocess.py
python tests/test_simultaneous.py
python tests/test_cluster_inprocess.py
python tests/test_three_nodes.py
python tests/test_election_edge.py

# Cluster đầy đủ (tiến trình riêng, cần model)
python tests/test_cluster.py
```

---

## Những Gì Chưa Làm

Có 3 thay đổi kiến trúc lớn chưa được thực hiện do đòi hỏi thiết kế lại toàn bộ hệ thống:

1. **Mã hóa kết nối (TLS):** Dữ liệu giữa các máy hiện truyền ở dạng thô, không mã hóa
2. **Lan truyền tin đồn (gossip protocol):** Hiện mỗi máy phát tới tất cả máy khác (O(n²)), chưa tối ưu
3. **Chống chia não (brain-split):** Chưa có cơ chế đại biểu (quorum) để xử lý khi mạng bị chia cắt làm đôi

---

## Branch

Tất cả thay đổi nằm trên branch **mainfix2**.