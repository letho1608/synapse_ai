# Synapse AI - Tài Liệu Kỹ Thuật Dành Cho Người Đọc

## Lời Nói Đầu

Dự án này là một hệ thống **trí tuệ nhân tạo phân tán** cho phép nhiều máy tính "góp sức" để chạy các mô hình AI lớn. Nếu bạn chỉ có một laptop yếu với 8GB VRAM, bạn không thể chạy model 7 tỷ tham số (7B). Nhưng nếu có 3 người bạn mỗi người cho mượt laptop, 3 máy kết nối lại thì có thể chạy được.

Tài liệu này giải thích **tất cả các thuật toán và thiết kế** của dự án theo cách ai cũng hiểu được.

---

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

## Phần 2: Mạng Lưới Ngang Hàng (P2P Mesh)

### 2.1 Các Máy Tìm Thấy Nhau Như Thế Nào?

Khi mới khởi động, các máy không biết nhau tồn tại. Chúng cần một cách để **tự động tìm thấy nhau** trong mạng nội bộ hoặc qua internet.

**Cách 1: UDP Broadcast (Mạng LAN)**
> Giống như bạn hét trong phòng "Ai đó tên A đang ở đây?". Ai nghe thấy sẽ trả lời.

Khi laptop A khởi động:
- Nó gửi một gói tin UDP broadcast ra toàn mạng LAN
- B và C nhận được, biết A đang online
- B và C gửi thông tin của mình (IP, port, khả năng máy) về cho A
- A giờ biết có 2 máy khác đang chạy, kết nối TCP tới B và C

**Cách 2: Tailscale (Qua Internet)**
> Tailscale tạo một "mạng riêng ảo" VPN. Dù các máy ở nhà, quán net, hay văn phòng khác thành phố, chúng vẫn thấy nhau như thể cùng mạng LAN.

Mỗi máy cài Tailscale sẽ có một IP đặc biệt dạng `100.x.x.x`. Dự án gọi Tailscale API để lấy danh sách tất cả máy đang chạy Synapse, rồi kết nối TCP tới chúng.

**Cách 3: Cấu Hình Tay (Manual)**
> Đơn giản là đọc file JSON liệt kê IP và port của các máy.

### 2.2 Giữ Kết Nối Sống (Heartbeat)

Sau khi kết nối thành công, các máy phải thường xuyên "thông báo" mình còn sống. Nếu không, một máy chết mà các máy khác không biết.

**Cơ chế heartbeat mỗi 5 giây:**
- A gửi "ping" cho B và C
- Nếu B không trả lời "pong" trong 15 giây → A coi B đã chết
- A thông báo cho C và các máy khác rằng B đã offline
- Hệ thống tự động **tính lại phân chia model** mà không cần B

### 2.3 Kết Nối Bị Trùng (Tiebreaker)

Đôi khi A và B cùng mở kết nối TCP tới nhau cùng lúc (gọi là race condition). Kết quả là có 2 kết nối A↔B, gây lãng phí và có thể nhận message trùng lặp.

**Quy tắc giải quyết:**
- So sánh tên máy theo bảng chữ cái
- Máy có tên "nhỏ hơn" thắng
- VD: A < B → A giữ kết nối của mình, B đóng kết nối của B

### 2.4 Mã Giao Tiếp (Binary Protocol)

Thông thường dữ liệu gửi qua mạng theo định dạng JSON. Nhưng JSON có nhược điểm:
- Viết dài, tốn băng thông
- Parse chậm

Synapse dùng **binary protocol** tự thiết kế:
```
[4 bytes độ dài][1 byte loại][N bytes dữ liệu]
```

- 4 bytes đầu cho biết gói dài bao nhiêu
- Byte thứ 5 cho biết đây là gói **ping** hay **pong** hay **dữ liệu model**
- Nhờ byte "loại" này, máy đọc 5 bytes đã biết đây là ping nhỏ → xử lý ngay, không phải đọc hết gói lớn

---

## Phần 3: Phân Chia Model (LACP Partitioning)

### 3.1 Tại Sao Cần Chia?

Model 7 tỷ tham số gồm 32 "lớp" (layers). Mỗi lớp cần VRAM để chạy. Laptop yếu không chạy nổi cả 32 lớp.

**Câu hỏi:** Làm sao chia 32 lớp cho 3 máy sao cho công bằng và hiệu quả?

### 3.2 Đo Lường Khả Năng Mỗi Máy

Trước khi chia, hệ thống phải biết mỗi máy "mạnh" cỡ nào. Nó đo:

| Yếu tố | Cách đo | Tầm quan trọng |
|--------|---------|----------------|
| VRAM | nvidia-smi (NVIDIA), rocm-smi (AMD) | **Rất quan trọng** - quyết định bao nhiêu layers |
| Tốc độ GPU | Tra bảng TFLOPS theo tên card | Quan trọng - GPU nhanh chạy được nhiều hơn |
| CPU | psutil.cpu_count() | Ít quan trọng - chỉ hỗ trợ inference |

**Công thức tính điểm:**
```
Điểm = VRAM_GB × 10 + TFLOPS_FP32/100
```

Ví dụ:
- Laptop A (RTX 4090 24GB): 24×10 + 82/100 = **240.82 điểm**
- Laptop B (RTX 3060 8GB): 8×10 + 10/100 = **80.1 điểm**
- Laptop C (RTX 4060 8GB): 8×10 + 15/100 = **80.15 điểm**

### 3.3 Chia Theo Tỷ Lệ

Tổng điểm = 240.82 + 80.1 + 80.15 = **401.07**

| Máy | Tỷ lệ | Số layers (32 total) |
|-----|-------|---------------------|
| A | 240.82/401 = 60% | 19 layers |
| B | 80.1/401 = 20% | 6 layers |
| C | 80.15/401 = 20% | 7 layers |

### 3.4 Nhóm Máy Gần Nhau (Hierarchical Clustering)

Không phải cứ điểm cao là được nhiều layers. Còn phải xem **latency** (độ trễ) giữa các máy.

**Nguyên tắc:** Các máy có latency thấp (gần nhau về mạng) nên được gán layers **liền kề nhau** trong model. Điều này giảm số lần phải gửi data qua mạng.

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

### 4.2 Giải Pháp: Pipeline

Ý tưởng: khi A đang xử lý batch 2, B xử lý batch 1, C xử lý batch 0. Tất cả GPU đều bận 100%.

**Hình dung như dây chuyền lắp ráp ô tô:**
- Trạm 1 (A): Lắp thân xe
- Trạm 2 (B): Lắp bánh xe
- Trạm 3 (C): Sơn và kiểm tra

Thay vì đợi 1 xe hoàn thiện rồi mới làm xe tiếp theo, các trạm làm **song song**: trạm 1 làm xe 2 trong khi trạm 2 đang làm xe 1.

### 4.3 Minh Họa Pipeline

```
4 batches, 3 stages

t1: A[b1]  B[ - ] C[ - ]    (A bắt đầu batch 1)
t2: A[b2]  B[b1]  C[ - ]    (A tiếp tục, B bắt đầu)
t3: A[b3]  B[b2]  C[b1]     (Pipeline "đầy" - tất cả đang làm)
t4: A[b4]  B[b3]  C[b2]
t5: A[ - ] B[b4]  C[b3]     (A xong, B tiếp tục)
t6: A[ - ] B[ - ]  C[b4]     (B xong, C tiếp tục)

→ Tổng thời gian: 6 "bước" thay vì 12 bước tuần tự
→ Nhanh hơn 50%
```

---

## Phần 5: Đồng Bộ Khi Training

### 5.1 Khi Training, Phải Tính Gradient

Khi train model, mỗi máy tính gradient (đạo hàm) trên phần batch của mình. Gradient này phải được **tổng hợp** để cập nhật weights.

### 5.2 Ring All-Reduce

Cách đơn giản nhất: gửi gradient về 1 máy tổng, rồi gửi lại. Nhưng nếu máy đó chết thì sao?

**Ring All-Reduce:** Các máy xếp thành vòng tròn, lần lượt gửi và cộng gradient với nhau.

```
4 máy: A → B → C → D → A

Vòng 1: A gửi gradient cho B, B cộng vào gradient của mình
Vòng 2: B gửi (gradient A + B) cho C, C cộng vào
Vòng 3: C gửi (A+B+C) cho D, D cộng vào
Vòng 4: D gửi (A+B+C+D) cho A, A cộng vào

→ Kết quả: Mọi máy có cùng gradient tổng
```

---

## Phần 6: Bầu Chọn Trưởng Nhóm (Master Election)

### 6.1 Tại Sao Cần Trưởng Nhóm?

Cần 1 máy đóng vai trò "điều phối" - nhận yêu cầu từ user, quyết định cách chia model, thu thập kết quả từ các máy khác. Máy này gọi là **Master**.

### 6.2 Không Có Server Trung Tâm

Không ai "bầu" ai. Mỗi máy tự quyết định ai là Master dựa trên thông tin mình có.

### 6.3 Thuật Toán

Mỗi 2 giây, mỗi máy gửi heartbeat cho biết "tôi đang online với điểm số X".

```
Giả sử điểm:
A: 240.82
B: 80.1
C: 80.15

Mỗi máy tính:
1. Liệt kê tất cả máy còn sống
2. Sort theo điểm giảm dần
3. Máy đầu tiên = Master

→ A có điểm cao nhất → A là Master
```

### 6.4 Chống Flapping

Nếu A đang là Master, B mới vào với điểm cao hơn, **A vẫn giữ vai trò Master**. Không ai "cướp" ngay được.

Chỉ khi A thực sự offline (không heartbeat > 30 giây), B mới trở thành Master mới.

**Tại sao 30 giây?** Vì khi A đang inference nặng, có thể CPU/GPU tạm spike không gửi được heartbeat. 30 giây đủ để A phục hồi mà không bị "kick".

---

## Phần 7: Phát Hiện Lỗi và Phục Hồi

### 7.1 Node Chết

Mỗi 5 giây, mỗi node kiểm tra các kết nối:
- Gửi ping
- Đợi pong 15 giây
- Quá 15 giây không pong → coi là đã chết

### 7.2 Tự Kết Nối Lại

Khi một node chết rồi quay lại:
- Thử ngay lập tức
- Thất bại → đợi 1 giây → thử lại
- Thất bại → đợi 2 giây → thử lại
- Thất bại → đợi 4 giây → thử lại
- ... cứ thế nhân đôi, tối đa 60 giây

### 7.3 Tính Lại Phân Chia

Khi 1 node chết:
- Các node còn sống nhận ra
- Master tính lại phân chia model (không có node chết)
- Các node load/unload layers theo phân chia mới
- Tiếp tục inference không cần restart toàn bộ

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

## Phần 9: Quản Lý Model

### 9.1 HuggingFace Integration

Synapse tích hợp sẵn khả năng download model từ HuggingFace Hub - nơi lưu trữ hàng ngàn model open-source.

Khi user chọn model "qwen2.5:1.5b":
1. Hệ thống tra hf_models.json (chứa 500+ model)
2. Tìm repo HuggingFace tương ứng: `Qwen/Qwen2.5-1.5B-Instruct`
3. Tự động download và cache

### 9.2 Model Metadata Database

File `hf_models.json` chứa thông tin về 500+ model:
- Tên model trên HuggingFace
- Số tham số (7B, 13B, 70B...)
- Lượng VRAM cần thiết
- Số layers
- Có hỗ trợ vision không
- Quantization nào tối ưu (Q4_K_M, Q8_0...)

### 9.3 Scrape Models Tự Động

Dự án có script tự động gọi HuggingFace API để thu thập thông tin model mới, cập nhật vào database.

---

## Phần 10: Quản Lý Dataset

### 10.1 Dataset Format

Dataset training định dạng JSONL (1 JSON object mỗi dòng):

```
{"text": "Câu hỏi: 2 + 2 = ?"}
{"text": "Đáp án: 4"}
{"messages": [{"role": "user", "content": "Xin chào"}, {"role": "assistant", "content": "Chào bạn"}]}
```

### 10.2 Upload và Validate

User upload file qua Web UI:
1. Kiểm tra định dạng .jsonl
2. Parse mỗi dòng, đảm bảo valid JSON
3. Đếm số samples
4. Lưu vào thư mục `synapse/data/`

### 10.3 Preview

Trước khi train, user có thể xem trước 10 dòng đầu tiên để đảm bảo format đúng.

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

## Phần 12: Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3 |
| AI | PyTorch, HuggingFace Transformers |
| Mạng | asyncio TCP sockets (tự viết) |
| API Server | aiohttp |
| Web UI | Alpine.js + Tailwind CSS |
| Hardware Detection | nvidia-smi, psutil |

**Điểm đặc biệt:** Không dùng gRPC, Kafka, Redis, hay bất kỳ middleware nào. Tất cả networking được tự viết thuần TCP Python.

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

- **Tối đa nodes:** ~16 (do full mesh O(n²))
- **Model lớn nhất:** ~70B (cần đủ VRAM tổng)
- **Batch size:** Giới hạn bởi tổng VRAM cluster

---

## Phần 14: Các Vấn Đề Đã Giải Quyết

| Vấn đề | Giải pháp |
|--------|-----------|
| Self-ID conflict | Node ID = SHA256(MAC + Port) |
| Master flapping | Capacity-based weight + 30s stickiness |
| Timeout 15s | Binary protocol với type byte |
| Tiebreaker leak | Đóng socket cũ đúng cách |
| GPU detection Windows | WMI queries |
| Cross-platform GPU | nvidia-smi/rocm-smi/sysfs |

---

## Lời Kết

Synapse AI là một dự án thể hiện sự **tự viết hệ thống phân tán** từ đầu - không reliance vào middleware có sẵn. Toàn bộ networking, partitioning, election, và fault tolerance được implement bằng Python thuần.

Điều này cho thấy:
1. Có thể xây dựng hệ thống P2P phức tạp mà không cần gRPC hay Kafka
2. Binary protocol đơn giản có thể hiệu quả hơn JSON
3. Greedy algorithm có thể đủ tốt thay vì ILP phức tạp
4. Không cần SciPy/Pulp - pure Python đủ dùng

---

*Tài liệu này giải thích Synapse AI v1.0*