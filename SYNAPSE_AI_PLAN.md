# Synapse AI - Tài Liệu Kỹ Thuật Dành Cho Người Đọc

## Lời Nói Đầu

Dự án này là một hệ thống **trí tuệ nhân tạo phân tán** cho phép nhiều máy tính "góp sức" để chạy các mô hình AI lớn. Nếu bạn chỉ có một laptop yếu với 8GB VRAM, bạn không thể chạy model 7 tỷ tham số (7B). Nhưng nếu có 3 người bạn mỗi người cho mượt laptop, 3 máy kết nối lại thì có thể chạy được.


## Phần 1: Tổng Quan Hệ Thống

### 1.1 Bài Toán Gốc

Bạn có 3 laptop, mỗi cái có card đồ họa yếu:

- Laptop A: RTX 3060 8GB
- Laptop B: RTX 3060 8GB
- Laptop C: RTX 3060 8GB

Bạn muốn chạy model **Qwen2.5-7B** - một model ngôn ngữ lớn. Nhưng model này cần **14GB VRAM** để load, trong khi mỗi laptop chỉ có 8GB.

**Giải pháp của Synapse:** Chia model thành 3 phần, mỗi laptop chịu trách nhiệm một phần. Khi user hỏi "Xin chào", cả 3 máy phải "cộng tác" để sinh ra câu trả lời.

### 1.2 Mô Hình Tổng Quan

```
┌──────────────────────────────────────────────────────────────┐
│                        NGƯỜI DÙNG                              │
│                    ChatGPT API / Web UI                        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                  MASTER NODE (Laptop A)                       │
│                                                              │
│   • Nhận yêu cầu từ người dùng                              │
│   • Điều phối các máy khác                                   │
│   • Tính toán cách chia model                                │
│   • Chạy phần đầu của model (layers 1-19)                   │
└────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                   │
          ▼                                   ▼
┌─────────────────────┐               ┌─────────────────────┐
│   LAPTOP B          │               │   LAPTOP C          │
│   Layers 20-26     │               │   Layers 27-32      │
│   (phần giữa)       │               │   (phần cuối)       │
└─────────────────────┘               └─────────────────────┘
```

**Quy trình cơ bản:**
1. User gửi câu hỏi qua API
2. Laptop A (master) nhận câu hỏi, encode thành numbers
3. Laptop A xử lý layers 1-19, gửi kết quả sang B
4. Laptop B xử lý layers 20-26, gửi kết quả sang C
5. Laptop C xử lý layers 27-32, sinh ra câu trả lời cuối cùng
6. Câu trả lời stream về người dùng từng chữ một

---

## Danh Sách 18 Thuật Toán Cốt Lõi

Dưới đây là bảng tra cứu nhanh 18 thuật toán và cơ chế kỹ thuật được sử dụng trong hệ thống:

| STT | Tên Thuật Toán | Mục Đích |
|-----|----------------|----------|
| 1 | **UDP Broadcast Discovery** | Tự động tìm kiếm các máy trong mạng LAN. |
| 2 | **Tailscale VPN Bridge** | Kết nối các máy qua internet xuyên tường lửa. |
| 3 | **Manual Discovery** | Cấu hình IP tĩnh cho các cụm máy cố định. |
| 4 | **Heartbeat Monitoring** | Theo dõi trạng thái "sống/chết" của các node. |
| 5 | **Lexicographical Tiebreaker** | Giải quyết xung đột khi hai máy cùng kết nối tới nhau. |
| 6 | **Binary Packet Protocol** | Giao thức truyền tin nhị phân siêu nhẹ và nhanh. |
| 7 | **Hardware Scoring** | Chấm điểm phần cứng để phân chia công việc. |
| 8 | **LACP (Layer Partitioning)** | Chia các lớp model AI theo năng lực từng máy. |
| 9 | **Latency Clustering** | Nhóm các máy gần nhau để giảm độ trễ truyền tin. |
| 10 | **1F1B Pipeline Scheduler** | Lập lịch song song giúp GPU không bị rảnh. |
| 11 | **Ring All-Reduce** | Đồng bộ hóa kiến thức training không cần server. |
| 12 | **Capacity-based Election** | Bầu chọn trưởng nhóm (Master) dựa trên sức mạnh. |
| 13 | **Election Stickiness** | Giữ ổn định Master, tránh đổi nhóm trưởng liên tục. |
| 14 | **Exponential Backoff** | Tự động kết nối lại với thời gian chờ tăng dần. |
| 15 | **Dynamic Rebalancing** | Chia lại model ngay lập tức khi có máy sập. |
| 16 | **Incremental Checkpointing** | Lưu trạng thái tạm thời để phục hồi không mất dữ liệu. |
| 17 | **MoE-Aware Estimation** | Ước tính RAM thông minh cho kiến trúc Mixture of Experts. |
| 18 | **HF Scraper & Metadata** | Tự động thu thập thông tin model từ HuggingFace. |

---

## Phần 2: Mạng Lưới Ngang Hàng (P2P Mesh)

### 2.1 Các Máy Tìm Thấy Nhau Như Thế Nào?

Khi mới khởi động, các máy không biết nhau tồn tại. Chúng cần một cách để **tự động tìm thấy nhau** trong mạng nội bộ hoặc qua internet.

**Thuật toán 1: UDP Broadcast Discovery (Mạng LAN)**
> Giống như bạn hét trong phòng "Ai đó tên A đang ở đây?". Ai nghe thấy sẽ trả lời.

Khi laptop A khởi động:
- Nó gửi một gói tin UDP broadcast ra toàn mạng LAN trên cổng định trước.
- B và C đang "nghe" trên cổng đó sẽ nhận được thông tin nhận dạng của A.
- B và C gửi phản hồi trực tiếp kèm thông tin (IP, port, khả năng máy) về cho A.
- A tổng hợp danh sách và thiết lập kết nối TCP tin cậy tới B và C.

**Thuật toán 2: Tailscale VPN Bridge (Qua Internet)**
> Giải quyết vấn đề NAT và tường lửa khi các máy không cùng mạng LAN.

- Synapse sử dụng Tailscale API để liệt kê tất cả các máy trong cùng một "Tailnet".
- Thuật toán xác định các địa chỉ IP nội bộ của VPN (dạng `100.x.x.x`).
- Hệ thống tự động tạo các "cầu nối" (bridges) để dữ liệu AI có thể đi xuyên qua các lớp bảo mật internet mà không cần mở port trên router.

**Thuật toán 3: Cấu Hình Tay (Manual Discovery)**
- Đơn giản là đọc file cấu hình tĩnh. Hữu ích cho các cụm máy server có IP cố định.

### 2.2 Giữ Kết Nối Sống (Heartbeat Monitoring)

**Thuật toán 4: Heartbeat Monitoring**
Sau khi kết nối, các máy phải duy trì trạng thái "còn sống".
- Cơ chế: Mỗi node gửi một gói tin "Ping" siêu nhỏ mỗi 2 giây cho tất cả các node khác.
- Nếu một máy không nhận được "Pong" phản hồi trong một khoảng thời gian (Grace Period), nó sẽ đánh dấu node đó là "Mất kết nối".
- Điều này kích hoạt quy trình tự phục hồi và tính toán lại toàn cụm.

### 2.3 Giải Quyết Xung Đột Kết Nối (Tiebreaker)

**Thuật toán 5: Lexicographical Tiebreaker**
Đôi khi A và B cùng mở kết nối TCP tới nhau cùng lúc (Race Condition).
- Quy tắc: So sánh Node ID (là chuỗi SHA256).
- Node có ID "nhỏ hơn" (theo bảng chữ cái) sẽ là bên giữ kết nối. Node còn lại sẽ tự động đóng kết nối vừa mở để tránh lãng phí tài nguyên và trùng lặp dữ liệu.

### 2.4 Giao Thức Truyền Tin Hiệu Năng Cao (Binary Protocol)

**Thuật toán 6: Binary Packet Protocol**
Synapse không dùng JSON vì nó quá nặng. Thay vào đó là một giao thức nhị phân tự định nghĩa:
```
[4 bytes: Length][1 byte: Type][N bytes: Payload]
```
- **Length**: Xác định kích thước gói tin để bộ đệm mạng biết khi nào nhận đủ dữ liệu.
- **Type**: Phân loại gói tin (Ping, Layer Data, Gradient, Command). Giúp xử lý nhanh các gói điều khiển mà không cần đọc hết Payload lớn (như tensor của AI).
- **Payload**: Dữ liệu thực tế được nén hoặc giữ nguyên tùy theo loại.

---

---

## Phần 3: Phân Chia Model (LACP Partitioning)

### 3.1 Tại Sao Cần Chia?

Model 7 tỷ tham số gồm 32 "lớp" (layers). Mỗi lớp cần VRAM để chạy. Laptop yếu không chạy nổi cả 32 lớp.

**Câu hỏi:** Làm sao chia 32 lớp cho 3 máy sao cho công bằng và hiệu quả?

### 3.2 Chấm Điểm Tài Nguyên (Hardware Scoring)

**Thuật toán 7: Hardware Scoring Algorithm**
Trước khi chia, hệ thống phải biết mỗi máy "mạnh" cỡ nào. Nó đo:

| Yếu tố | Cách đo | Tầm quan trọng |
|--------|---------|----------------|
| VRAM | nvidia-smi (NVIDIA), rocm-smi (AMD), WMI (Windows) | **Cốt lõi** - Quyết định số lượng lớp có thể chứa. |
| TFLOPS | Tra cứu dựa trên tên Model GPU phát hiện được. | **Quan trọng** - GPU nhanh sẽ xử lý nhiều việc hơn. |
| RAM & CPU | psutil.virtual_memory(), psutil.cpu_count() | **Hỗ trợ** - Quyết định tốc độ tải model và xử lý bổ trợ. |

### 3.3 Chiến Lược Phân Chia LACP

**Thuật toán 8: LACP (Layer-Aware Capacity Partitioning)**
Đây là thuật toán "tham lam" (greedy) để chia model:
1. Tính tổng điểm năng lực của toàn bộ Cluster.
2. Với mỗi Node, tính tỷ lệ phần trăm đóng góp dựa trên điểm số.
3. Nhân tỷ lệ này với tổng số lớp (VD: 32 layers) để ra con số mục tiêu.
4. Điều chỉnh con số này sao cho không máy nào bị tràn VRAM (nếu tràn, phần dư sẽ được đẩy sang máy mạnh hơn).

### 3.4 Nhóm Theo Độ Trễ (Latency Clustering)

**Thuật toán 9: Hierarchical Latency Clustering**
Không phải cứ máy mạnh là gán nhiều việc nếu nó ở quá xa về mặt mạng.
1. Đo độ trễ (ping) giữa từng cặp máy tạo thành ma trận độ trễ.
2. Nhóm các máy có độ trễ cực thấp (VD < 5ms) vào cùng một cụm (Cluster).
3. Ưu tiên gán các lớp AI liên tiếp cho các máy trong cùng cụm. Điều này giúp dữ liệu truyền qua lại giữa các lớp diễn ra cực nhanh, chỉ truyền qua internet/mạng chậm khi thực sự cần thiết (giữa các cụm).

**Ví dụ:**
- A và C latency: 2ms (cùng nhà)
- B và A latency: 50ms (khác mạng)
- B và C latency: 48ms

**Kết quả clustering:**
- Cluster 1: A + C (2ms) → layers 1-26
- Cluster 2: B (48-50ms) → layers 27-32

→ Chỉ 1 lần gửi data cross-cluster (từ C sang B)

### 3.5 Gán Layers Cụ Thể

```
┌─────────────────────────────────────────────────────────────┐
│ CLUSTER 1 (A + C, latency 2ms - cùng nhà)                   │
│                                                             │
│  Laptop A: Layers 1-19  (embedding + attention đầu)        │
│  Laptop C: Layers 20-26 (phần giữa)                         │
│                                                             │
│  A và C trao đổi data nội bộ, nhanh                         │
├─────────────────────────────────────────────────────────────┤
│ CLUSTER 2 (B, latency cao hơn - khác mạng)                  │
│                                                             │
│  Laptop B: Layers 27-32  (output + kết quả cuối)          │
│                                                             │
│  Chỉ nhận data 1 lần từ C, tính toán cuối                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Phần 4: Xử Lý Song Song (Pipeline)

### 4.1 Vấn Đề: Máy Rảnh

Nếu chạy tuần tự (A xong → B xong → C xong), trong lúc B đang chạy thì A và C đang ngồi chơi → lãng phí GPU.

### 4.2 Giải Pháp: Pipeline Parallelism

**Thuật toán 10: 1F1B Pipeline Scheduler (One-Forward-One-Backward)**
Để GPU không ngồi chơi, Synapse sử dụng cơ chế dây chuyền:
- Chia một yêu cầu lớn thành nhiều phần nhỏ (micro-batches).
- Khi Trạm 1 xử lý xong phần 1 và gửi cho Trạm 2, nó sẽ bắt đầu xử lý phần 2 ngay lập tức thay vì đợi Trạm 3 hoàn thành.
- **1F1B**: Trong quá trình training, hệ thống xen kẽ một bước tiến (Forward) và một bước lùi (Backward) để tối ưu bộ nhớ VRAM, đảm bảo không có GPU nào bị "nghẽn cổ chai".

### 4.3 Minh Họa Pipeline
(Giống như dây chuyền lắp ráp, giúp tăng tốc độ sinh từ lên gấp 2-3 lần tùy số lượng máy).

---

## Phần 5: Đồng Bộ Khi Training

### 5.1 Khi Training, Phải Tính Gradient
### 5.2 Ring All-Reduce

**Thuật toán 11: Ring All-Reduce**
Đồng bộ hóa các "kiến thức" học được giữa các máy mà không cần server trung tâm:
1. Các máy xếp thành một vòng tròn ảo: A -> B -> C -> D -> A.
2. Mỗi máy chia dữ liệu học (Gradients) thành nhiều phần nhỏ.
3. Máy A gửi phần 1 cho B, B cộng dồn với phần của mình rồi gửi cho C...
4. Sau hai vòng truyền tin quanh vòng tròn, mọi máy đều sở hữu tổng hợp kiến thức của toàn bộ Cluster.
> Ưu điểm: Băng thông được tận dụng tối đa trên tất cả các node, không máy nào bị quá tải.

---

## Phần 6: Bầu Chọn Trưởng Nhóm (Master Election)

### 6.3 Thuật Toán Bầu Chọn

**Thuật toán 12: Capacity-based Master Election**
Ai sẽ là người điều phối?
- Mỗi máy tự tính toán dựa trên danh sách các máy online.
- Quy tắc: Máy có **Hardware Score** cao nhất sẽ tự nhận là Master.
- Vì thuật toán là nhất quán, tất cả các máy sẽ cùng đồng ý về một Master mà không cần gửi tin nhắn bầu chọn phức tạp.

### 6.4 Chống Nhảy Master (Stickiness)

**Thuật toán 13: Election Stickiness Algorithm**
Nếu một máy mạnh hơn vừa gia nhập, hệ thống sẽ **không** đổi Master ngay lập tức.
- Master hiện tại được giữ nguyên trừ khi nó offline quá 30 giây.
- Điều này ngăn chặn tình trạng "Master Flapping" - Master thay đổi liên tục gây gián đoạn việc inference.

---

## Phần 7: Phát Hiện Lỗi và Phục Hồi

### 7.1 Phát Hiện Lỗi

Cơ chế Heartbeat (Thuật toán 4) liên tục theo dõi trạng thái các kết nối.

### 7.2 Tự Kết Nối Lại (Backoff)

**Thuật toán 14: Exponential Backoff Reconnection**
Khi một node mất kết nối, hệ thống không thử lại liên tục một cách mù quáng.
- Lần 1: Đợi 1 giây.
- Lần 2: Đợi 2 giây.
- Lần 3: Đợi 4 giây... tối đa 60 giây.
- Điều này giúp giảm tải cho mạng và cho phép máy bị lỗi có thời gian khởi động lại hoàn toàn.

### 7.3 Tính Toán Lại Phân Chia (Dynamic Rebalancing)

**Thuật toán 15: Dynamic Shard Rebalancing**
Khi Master phát hiện một Node đã chết (sau 15-30 giây không phản hồi):
1. **Cô lập Node lỗi:** Master ngay lập tức loại bỏ ID của node đó khỏi danh sách hoạt động (Active Cluster).
2. **Tính toán lại ma trận:** Master chạy lại thuật toán LACP (Thuật toán 8) để chia lại 32-80 lớp AI cho số máy còn lại.
3. **Phát lệnh cấu hình mới:** Master gửi bảng phân chia mới cho tất cả các node.
4. **Tải lớp (Hot-swapping):** Các node còn sống sẽ so sánh bảng cũ và mới. Nếu chúng nhận thêm lớp, chúng sẽ tải từ RAM vào VRAM. Nếu mất lớp, chúng giải phóng VRAM ngay lập tức.
> Kết quả: Hệ thống tiếp tục chạy mà không cần khởi động lại, chỉ mất khoảng 5-10 giây để "ổn định" lại đội hình.

### 7.4 Khôi Phục Từ Điểm Lưu (Checkpointing)

**Thuật toán 16: Incremental Checkpointing**
Để tránh việc phải chạy lại từ đầu khi một node sập:
- **Lưu trữ định kỳ:** Mỗi 30-60 giây, các node tự lưu "trạng thái suy nghĩ" (KV Cache) của các token gần nhất vào đĩa cứng.
- **Trạng thái huấn luyện:** Trong lúc train, các trọng số tạm (gradients) được lưu vào bộ nhớ chia sẻ.
- **Khôi phục nhanh:** Khi một máy mới thay thế máy cũ, nó sẽ tìm bản Checkpoint gần nhất trên mạng lưới hoặc đĩa chung để "nối tiếp" luồng suy nghĩ của AI thay vì bắt đầu lại từ token đầu tiên.

---

## Phần 8: API và Giao Diện

### 8.1 ChatGPT API Tương Thích

Synapse cung cấp REST API giống hệt OpenAI. Ứng dụng nào dùng được OpenAI API thì có thể dùng Synapse thay thế.

```
POST http://localhost:52415/v1/chat/completions

Request:
{
  "model": "qwen2.5:1.5b",
  "messages": [{"role": "user", "content": "Xin chào"}]
}

Response:
{
  "id": "chatcmpl-xxx",
  "choices": [{
    "message": {"role": "assistant", "content": "Xin chào, tôi có thể giúp gì?"}
  }]
}
```

### 8.2 Web Dashboard

Giao diện web cho phép:
- Chat trực tiếp
- Quản lý model (download, xóa, xem thông tin)
- Upload dataset cho training
- Xem trạng thái cluster (máy nào online, ai là Master)
- Visualization topology (sơ đồ các node kết nối như thế nào)
---

---
## Phần 9: Quản Lý Model

### 9.1 HuggingFace Integration

Synapse tích hợp sẵn khả năng download model từ HuggingFace Hub - nơi lưu trữ hàng ngàn model open-source.

Khi user chọn model "qwen2.5:1.5b":
1. Hệ thống tra hf_models.json (chứa 500+ model)
2. Tìm repo HuggingFace tương ứng: `Qwen/Qwen2.5-1.5B-Instruct`
3. Tự động download và cache

### 9.2 Ước Tính Tài Nguyên MoE (Resource Estimation)

**Thuật toán 17: MoE-Aware RAM/VRAM Estimation**
Các model hiện đại như Mixtral hay DeepSeek dùng kiến trúc MoE (Mixture of Experts). Khi chạy, không phải toàn bộ trọng số đều được nạp vào VRAM cùng lúc, mà chỉ các "chuyên gia" (experts) được kích hoạt mới tham gia tính toán.

Thuật toán của Synapse thực hiện các bước sau để bóc tách và ước tính:

1. **Nhận dạng kiến trúc:** Trích xuất `num_experts` (tổng số chuyên gia) và `active_experts` (số chuyên gia chạy trên mỗi token, hay `top_k`).
2. **Phân tách tham số chung:** Quy ước khoảng 5% tổng số tham số (`total_params`) của model là các lớp dùng chung (Shared Layers như Attention, Embedding):
   `Shared_Params = Total_Params * 0.05`
3. **Tính toán dung lượng từng chuyên gia:**
   `Expert_Pool = Total_Params - Shared_Params`
   `Per_Expert_Params = Expert_Pool / Num_Experts`
4. **Tính toán tham số hoạt động thực tế:**
   `Active_Params = Shared_Params + (Active_Experts * Per_Expert_Params)`
5. **Quy đổi sang VRAM (GB):** Phụ thuộc vào định dạng nén (Quantization). Ví dụ nén Q4_K_M tương đương 0.5 Bits-Per-Parameter (BPP):
   `VRAM_Needed = (Active_Params * BPP) / 8 / (1024^3) * 1.2 (Overhead)`

**Kết quả:** Nhờ thuật toán này, Master phân bổ model khổng lồ cực kỳ chính xác. Ví dụ một model 100B tham số (8 chuyên gia, chạy 2), VRAM thực tế để chạy chỉ tương đương model 28B, cho phép cụm máy yếu vẫn kham nổi.

### 9.3 Tự Động Quét Model (HF Scraper)

**Thuật toán 18: HuggingFace Scraper & Metadata Extraction**
Hệ thống có script tự động truy cập API của HuggingFace để:
1. **Quét file cấu hình:** Tải `config.json` để đọc số layers, kiến trúc (Llama, Qwen, Mistral).
2. **Ước tính sai số:** Dựa trên kích thước file `.safetensors` để tính toán số tham số thực tế.
3. **Ánh xạ Quantization:** Tự động đề xuất mức nén (Q4_K_M, Q8_0) phù hợp với dung lượng VRAM hiện có của Cluster.
4. **Cập nhật Database:** Tự động ghi đè các thông tin mới vào file `hf_models.json` để Master luôn có dữ liệu mới nhất về các model AI trên thế giới.

---

## Phần 10: Quản Lý Dataset

### 10.1 Dataset Format

Dataset training định dạng JSONL (1 JSON object mỗi dòng):

```
{"text": "Câu hỏi: 2 + 2 = ?"}
{"text": "Đáp án: 4"}
{"messages": [{"role": "user", "content": "Xin chào"}, {"role": "assistant", "content": "Chào bạn"}]}
```

### 10.2 Tải Dữ Liệu Thông Minh (Streaming & Validation)

**Thuật toán 19 (Bonus): Dataset Streaming & Validation**
Đối với việc training với dữ liệu hàng Gigabyte:
1. **Lazy Loading (Streaming):** Không nạp cả file vào RAM. Thuật toán dùng `yield` trong Python để đọc từng dòng JSONL, xử lý xong dòng nào xóa dòng đó.
2. **Schema Validation:** Kiểm tra xem mỗi dòng có đủ các trường `instruction`, `input`, `output` hay không.
3. **Tokenization Preview:** Ước tính số lượng token của dataset để dự báo thời gian training Master cần điều phối.
4. **Auto-Cleaning:** Tự động loại bỏ các dòng bị lỗi format hoặc chứa ký tự rác trước khi đưa vào vòng lặp training chính.

---

## Phần 11: Cấu Trúc File

```
synapse/
│
├── main.py                     ← Điểm khởi đầu của chương trình
│
├── api/
│   └── chatgpt_api.py        ← REST API giống OpenAI (port 52415)
│
├── topology/                   ← Quản lý cluster và phân chia model
│   ├── partitioner.py          ← Thuật toán chia model LACP
│   ├── latency_matrix.py       ← Đo latency + clustering
│   ├── resource_weight.py       ← Chấm điểm phần cứng
│   ├── device_capabilities.py  ← Detect GPU, CPU, RAM
│   └── topology.py             ← Trạng thái các node
│
├── orchestration/
│   └── node.py                 ← Node chính, điều phối mọi thứ
│
├── pipeline/                   ← Pipeline parallelism
│   ├── stage.py               ← Một stage (1 phần model)
│   ├── scheduler.py           ← Lập lịch 1F1B
│   └── gradient_sync.py       ← Ring All-Reduce
│
├── networking/
│   ├── p2p_socket_bridge.py   ← TCP P2P transport
│   ├── udp/                   ← Discovery qua UDP
│   ├── tailscale/            ← Discovery qua Tailscale VPN
│   └── manual/               ← Discovery qua config file
│
├── training/
│   └── dataset.py            ← Load JSONL training data
│
├── inference/
│   └── pytorch_hf_engine.py  ← Inference engine PyTorch
│
├── config/
│   ├── settings.json          ← Cấu hình hệ thống
│   └── hf_models.json        ← Database 500+ model
│
├── data/                      ← Training data
│   └── *.jsonl
│
├── tinychat/                  ← Web UI (Alpine.js)
│   └── dashboard.html
│
└── tests/                    ← Unit tests
```

---

## Phần 12: Thuật Toán Định Danh Node (SHA256 Node ID)

### 12.1 Vấn Đề

Khi 3 máy cùng khởi động, mỗi máy cần một tên định danh (ID) **duy nhất và ổn định**:
- **Nếu dùng số ngẫu nhiên:** Hai máy có thể tình cờ chọn cùng một ID, hoặc sau khi khởi động lại ID thay đổi khiến các máy khác "không nhận ra".
- **Nếu dùng hostname:** Hai máy đặt tên giống nhau (ví dụ: cả hai tên `PC-Gaming`) sẽ xung đột.

### 12.2 Thuật Toán: SHA256 Deterministic Identity

Thay vì ngẫu nhiên, Synapse tạo ID từ **thông tin vật lý không thể thay đổi** của máy:

```
Node_ID = SHA256( MAC_Address + ":" + Listen_Port )
```

**Các bước thực hiện:**
1. Đọc địa chỉ MAC của card mạng đang hoạt động (ví dụ: `AA:BB:CC:DD:EE:FF`). MAC là con số được ghi vĩnh viễn vào phần cứng — không thể trùng giữa hai thiết bị khác nhau.
2. Ghép thêm cổng lắng nghe (ví dụ: `:52415`). Điều này cho phép cùng một máy chạy nhiều Synapse instance mà không xung đột ID.
3. Đưa chuỗi ghép qua hàm băm SHA256 — thuật toán mã hóa chuẩn công nghiệp biến bất kỳ đầu vào nào thành 64 ký tự hex cố định.
4. Lấy 16 ký tự hex đầu tiên làm Node ID để dễ đọc trong log.

**Ví dụ cụ thể:**
```
MAC của Laptop A: AA:BB:CC:DD:EE:FF
Cổng nghe:       52415

Chuỗi đầu vào: "AA:BB:CC:DD:EE:FF:52415"
SHA256:          a3f9b1c2d4e5f6789abc0123...
Node ID cuối:  "a3f9b1c2"
```

**Tại sao đảm bảo không trùng lặp?**
- Xác suất hai SHA256 khác nhau trùng kết quả ("collision") nhỏ hơn 1/2¹²⁸ — gần như bằng 0 trong thực tế.
- Cùng đầu vào → luôn cho cùng kết quả: Dù khởi động lại bao nhiêu lần, Node ID không bao giờ thay đổi.

**Ứng dụng trong hệ thống:**
- Thuật toán Tiebreaker (#5) dùng Node ID để so sánh ai "nhỏ hơn" khi xung đột kết nối.
- Heartbeat (#4) dùng Node ID để xác định gói tin đến từ máy nào.
- Checkpointing (#16) lưu file theo Node ID để khi phục hồi tìm đúng file.

---

## Phần 13: Performance

### 13.1 Latency Tham Khảo

| Loại kết nối | Độ trễ |
|--------------|--------|
| Cùng máy (loopback) | ~0.5ms |
| Mạng LAN 1Gbps | ~2ms |
| Tailscale cùng vùng | ~40ms |
| Tailscale khác vùng | ~80ms |

### 13.2 Tốc Độ Inference

| Cấu hình | Tốc độ |
|----------|--------|
| 1x RTX 3060 8GB (Qwen2.5-1.5B) | ~12 tokens/s |
| 3x RTX 3060 LAN (Qwen2.5-7B) | ~25 tokens/s |
| 1x RTX 4090 24GB (Qwen2.5-7B) | ~45 tokens/s |

### 13.3 Scalability

- **Tối đa nodes:** ~100 theo Tailscale (do full mesh O(n²))
- **Model lớn nhất:** ~70B (cần đủ VRAM tổng)
- **Batch size:** Giới hạn bởi tổng VRAM cluster


---

## Phần 14: Thuật Toán Phát Hiện GPU Đa Nền Tảng

### 14.1 Vấn Đề

Synapse chạy trên Windows, Linux và macOS với đủ loại GPU: NVIDIA (RTX series), AMD (RX series), Apple Silicon (M1/M2/M3), Intel Arc. Không một công cụ nào hỗ trợ tất cả.

### 14.2 Thuật Toán: Waterfall Detection với Fallback Chain

Hệ thống thử từng phương pháp theo **thứ tự ưu tiên giảm dần**, dừng lại khi tìm thấy GPU:

```
BƯỚC 1: Gọi lệnh nvidia-smi
├── Thành công → GPU NVIDIA tìm thấy, ghi nhận VRAM + TFLOPS → XONG
└── Thất bại (lệnh không tồn tại) → Sang bước 2

BƯỚC 2: Gọi lệnh rocm-smi
├── Thành công → GPU AMD (Linux) tìm thấy → XONG
└── Thất bại → Sang bước 3

BƯỚC 3: Đọc hệ thống file /sys/class/drm/ (Linux)
├── Thành công → GPU bất kỳ có driver Linux chuẩn → XONG
└── Thất bại (Windows hoặc GPU lạ) → Sang bước 4

BƯỚC 4: Chạy WMI Query (chỉ Windows)
├── Thành công → GPU Windows tìm thấy → XONG
└── Thất bại → Sang bước 5

BƯỚC 5: Fallback CPU-only
└── Luôn thành công: Chạy bằng CPU, điểm phần cứng thấp
```

**Chi tiết cơ chế từng bước:**

**Bước 1 — nvidia-smi (NVIDIA GPU):**
- Chạy lệnh terminal, bắt output XML.
- Parse field `<fb_memory_usage><total>` để lấy VRAM (MiB).
- Tra bảng TFLOPS trong database nội bộ theo tên model (ví dụ: "RTX 3060" → 12.7 TFLOPS).

**Bước 2 — rocm-smi (AMD GPU):**
- Chạy lệnh terminal với flag `--showmeminfo vram`.
- Parse kết quả dạng JSON để lấy `vram_total`.
- AMD thường dùng ký hiệu khác NVIDIA nên cần parse riêng.

**Bước 3 — sysfs Linux:**
- Đọc trực tiếp file `/sys/class/drm/card0/device/mem_info_vram_total`.
- Đây là cách đọc thẳng từ kernel — không cần phần mềm phụ trợ nào.
- Hiệu quả cho GPU Intel Arc hoặc AMD không cài rocm-smi.

**Bước 4 — WMI Query (Windows):**
- Dùng thư viện Python `wmi` để truy vấn cơ sở dữ liệu hệ thống Windows.
- Câu truy vấn: `SELECT Name, AdapterRAM FROM Win32_VideoController`.
- Trả về tên và dung lượng RAM cho mọi GPU được Windows nhận diện, kể cả GPU không phải NVIDIA/AMD.

**Bước 5 — CPU Fallback:**
- Dùng `psutil` để đọc số nhân CPU và dung lượng RAM.
- Gán VRAM = 0, TFLOPS = 0 để Hardware Scoring (#7) tính điểm thấp.
- Master sẽ gán số layers tối thiểu cho node này.

**Chuẩn hóa kết quả đầu ra:**

Dù phát hiện qua bất kỳ phương pháp nào, mọi thông tin đều được trả về cùng một cấu trúc để Hardware Scoring (#7) và LACP (#8) hoạt động mà không cần biết hệ điều hành:

```
{
  "vendor": "NVIDIA" | "AMD" | "Intel" | "CPU",
  "model":  "RTX 3060",
  "vram_gb": 8.0,
  "tflops_fp32": 12.7
}
```

---

## Lời Kết

Synapse AI là một dự án thể hiện sự **tự viết hệ thống phân tán** từ đầu mà không dùng bất kỳ middleware có sẵn nào. Toàn bộ 18 thuật toán trên — từ việc tìm kiếm nhau qua UDP, chia model AI, bầu chọn trưởng nhóm, đến tự phục hồi khi sập — đều được implement bằng Python thuần.

Điều này cho thấy:
1. Thuật toán Greedy đơn giản (LACP #8) có thể cho kết quả đủ tốt thay vì tối ưu hóa tuyến tính (ILP) phức tạp.
2. Binary protocol tự thiết kế (#6) nhanh hơn JSON mà không cần thư viện bên ngoài.
3. Ring All-Reduce (#11) có thể thực hiện hiệu quả mà không cần NCCL hay gRPC.
4. Fault tolerance đạt được chỉ với: Heartbeat (#4) + Checkpointing (#16) + LACP tái tính (#15).

---

*Tài liệu này giải thích Synapse AI v1.0*
