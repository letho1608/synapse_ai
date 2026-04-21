# Kế hoạch Tu bổ Synapse AI (Branch: mainfix2)

Dựa trên phân tích từ `code-review-graph` và so sánh với kiến trúc của dự án **Exo**, tài liệu này đề xuất lộ trình đại tu hệ thống **Synapse AI** để giải quyết triệt để lỗi không phân tán được và lỗi treo khi sinh output.

## Mục tiêu chính
1.  **Ổn định hóa phân tán**: Loại bỏ hoàn toàn gRPC đồng bộ, thay thế hoàn toàn bằng mạng lưới **P2P Mesh (libp2p/Gossipsub)**.
2.  **Chính xác hóa phân đoạn**: Nâng cấp thuật toán LACP lên bản **2.0** (áp dụng Largest Remainder và Topology-aware của Exo).
3.  **Khả dụng trên Windows**: Tối ưu hóa các Engine thực thi (PyTorch + CUDA/DirectML) cho đa dạng phần cứng trên Windows.
4.  **Cơ chế đẩy (Push Model)**: Chuyển đổi luồng inference sang hướng sự kiện (Asynchronous Push) để tránh treo request.

---

## Các thay đổi đề xuất

### 1. Hạ tầng Mạng (Networking Layer - libp2p Integration)
Loại bỏ hoàn toàn hệ thống gRPC cũ và thay thế bằng Router tin nhắn bất đồng bộ.

*   **[DELETE] `synapse/networking/grpc/grpc_server.py`**: Xóa bỏ vì không còn sử dụng mô hình RPC đồng bộ.
*   **[NEW] `synapse/routing/event_router.py`**: Quản lý các Topic và Pub/Sub.
*   **[NEW] `synapse/routing/libp2p_node.py`**: Tích hợp libp2p để chạy Mesh trên Tailscale.
*   **[MODIFY] `synapse/networking/tailscale/tailscale_discovery.py`**: Chuyển từ việc chỉ lấy IP sang việc cung cấp Node ID và Keypair cho mạng Mesh.

### 2. Bộ não điều phối (Orchestration - Master-Worker Architecture)
Chuyển từ mô hình chuỗi (Chain) sang mô hình tập trung điều phối nhưng phân tán thực thi.

*   **[NEW] `synapse/orchestration/election.py`**: Triển khai thuật toán bầu chọn Master (Elected Master).
*   **[MODIFY] `synapse/orchestration/node.py`**: 
    *   Hạ cấp logic điều phối của node thường thành Worker.
    *   Thêm logic lắng nghe Topic để nhận Tensor đầu vào.
*   **[NEW] `synapse/shared/state.py`**: Quản lý trạng thái chung của Cluster (Global Topology).

### 3. Thuật toán LACP 2.0 (Partitioning Upgrade)
Nâng cấp thuật toán `RingMemoryWeightedPartitioningStrategy` theo tiêu chuẩn Exo.

*   **[MODIFY] `synapse/topology/partitioning_strategy.py`**: 
    *   Tích hợp thuật toán **Largest Remainder** để chia layer chính xác 100%.
    *   Bổ sung tham số **Network Latency** và **RAM Available** vào trọng số tính toán.
*   **[MODIFY] `synapse/topology/device_capabilities.py`**: Thu thập thêm thông tin về băng thông giữa các node để tối ưu hóa vị trí (Placement).

### 4. Luồng Inference & Engine (Windows Optimized)
Đảm bảo sinh output mượt mà trên đa phần cứng Windows.

*   **[MODIFY] `synapse/inference/pytorch_hf_engine.py`**: 
    *   Tích hợp **DirectML** để hỗ trợ card đồ họa AMD/Intel trên Windows.
    *   Chuyển đổi `process_prompt` và `process_tensor` thành các Task bất đồng bộ (Background Tasks).
*   **[MODIFY] `synapse/inference/shard.py`**: Cải tiến cơ chế quản lý bộ nhớ đệm (Cache) để giảm tải cho RAM hệ thống.

---

## Lộ trình thực hiện (Phases)

| Giai đoạn | Nội dung | Tiêu chí thành công |
| :--- | :--- | :--- |
| **Phase 1** | Cập nhật Dependency & P2P Router | `requirements.txt` có thêm `anyio`, `libp2p`; Router khởi chạy thành công. |
| **Phase 2** | Triển khai Master Election | Hệ thống tự động chọn ra 1 Master khi khởi chạy Cluster. |
| **Phase 3** | Cài đặt LACP 2.0 | Model được chia shard chính xác, không sót layer, ưu tiên node gần mạng. |
| **Phase 4** | Refactor Inference Flow | Test inference sinh được output token-by-token qua 3 node Windows (loại bỏ gRPC). |
| **Phase 5** | Stress Test & Polish | Chốt độ ổn định trên mạng Tailscale thực tế. |

---

## Các rủi ro và Lưu ý (WARNING)

> [!WARNING]
> **Thay thế hoàn toàn gRPC**: Điều này sẽ làm mất tính tương thích với các phiên bản Synapse cũ. Tất cả các node trong cụm phải được cập nhật đồng thời.

> [!IMPORTANT]
> **Windows/PyTorch**: Do không dùng được MLX của Apple, chúng ta sẽ tập trung tối ưu hóa `DirectML`. Bạn cần cài đặt `torch-directml` nếu muốn chạy trên card AMD/Intel.

---

## Kế hoạch kiểm chứng (Verification)

### Kiểm thử tự động
1.  Chạy script giả lập 3 node trên cùng 1 máy (Windows).
2.  Gửi request qua `ChatGPTAPI`.
3.  Kiểm tra log để đảm bảo Tensor được "đẩy" qua các Topic libp2p chính xác.

### Kiểm thử thực tế
1.  Kết nối 2 máy tính Windows qua Tailscale.
2.  Mở Dashboard Synapse (tạm thời xem log) để xác nhận Master đã bầu chọn.
3.  Inference model `Llama-3-8B` chia đôi qua 2 máy.

---

## Quyết định đã chốt (Decisions Made)
*   **Xóa bỏ gRPC**: Đồng ý loại bỏ hoàn toàn `grpc_server.py` và các phụ thuộc liên quan để chuyển sang 100% P2P.
*   **Bổ sung Thư viện**: Đồng ý thêm `anyio`, `py-libp2p`, và `torch-directml` vào dự án.
*   **Kiểm thử mới**: Viết lại hoàn toàn `tests/test_cluster.py` để tương thích với cơ chế P2P và thuật toán LACP 2.0.
