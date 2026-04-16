# Kế Hoạch Cập Nhật Web UI

## 1. Mục tiêu
Cập nhật giao diện web UI của `synapse` để tích hợp và hiển thị thông tin phần cứng và ước tính tài nguyên mô hình LLM một cách trực quan và chi tiết hơn, sử dụng logic từ `llmfit-main`.

## 2. Phân tích hiện trạng

### 2.1. Các tệp web UI hiện tại:
- [`synapse/tinychat/dashboard.html`](file:///c:/Users/Admin/Downloads/synapse_ai/synapse/tinychat/dashboard.html): Trang tổng quan chính, chứa phần "System Information" và "Model Hardware Check".
- [`synapse/api/chatgpt_api.py`](file:///c:/Users/Admin/Downloads/synapse_ai/synapse/api/chatgpt_api.py): API cung cấp dữ liệu cho web UI.

### 2.2. Logic đã tích hợp từ `llmfit-main`:
- [`synapse/topology/device_capabilities.py`](file:///c:/Users/Admin/Downloads/synapse_ai/synapse/topology/device_capabilities.py): Phát hiện phần cứng nâng cao.
- [`synapse/helpers.py`](file:///c:/Users/Admin/Downloads/synapse_ai/synapse/helpers.py): Ước tính tài nguyên nâng cao.

## 3. Yêu cầu cập nhật

### 3.1. Cập nhật trang "System Information" (dashboard.html)
- **Hiển thị chi tiết phần cứng**: CPU, RAM, GPU/VRAM với thông tin đầy đủ hơn (tên, dung lượng, loại).
- **Tự động phát hiện và hiển thị**: Các thành phần phần cứng được phát hiện tự động và hiển thị chi tiết.

### 3.2. Cập nhật phần "Model Hardware Check" (dashboard.html)
- **Hiển thị ước tính tài nguyên**: Bộ nhớ cần thiết, mức độ phù hợp, tốc độ dự kiến.
- **Tích hợp logic từ `llmfit-main`**: Sử dụng logic ước tính tài nguyên mới để cung cấp thông tin chính xác hơn.

## 4. Thiết kế chi tiết

### 4.1. Trang "System Information" mới

#### 4.1.1. Bố cục
```
System Information
├── CPU: Intel Core i9-12900K (16 cores, 24 threads)
├── RAM: 64GB total, 48GB available
├── GPU: NVIDIA RTX 4090 (24GB VRAM) [Unified Memory: No]
└── OS: Windows 11
```

#### 4.1.2. Dữ liệu
- **CPU**: Tên CPU, số lõi, số luồng.
- **RAM**: Tổng dung lượng, dung lượng khả dụng.
- **GPU**: Tên GPU, dung lượng VRAM, có phải là Unified Memory không.
- **OS**: Hệ điều hành.

### 4.2. Phần "Model Hardware Check" mới (dashboard.html)

#### 4.2.1. Bố cục
```
Model Hardware Check
├── Model: meta-llama/Llama-3.1-8B-Instruct
├── Parameters: 8.0B
├── Quantization: Q4_K_M
├── Context Length: 8192 tokens
├── Memory Needed: ~6.2GB
├── Fit: 🟢 Perfect
├── Estimated Speed: 45.2 tok/s
└── Notes: [Optional notes from the analysis]
```

#### 4.2.2. Dữ liệu (dashboard.html)
- **Model**: Tên mô hình.
- **Parameters**: Số lượng tham số của mô hình.
- **Quantization**: Mức lượng tử hóa được chọn.
- **Context Length**: Độ dài ngữ cảnh.
- **Memory Needed**: Bộ nhớ cần thiết để chạy mô hình.
- **Fit**: Mức độ phù hợp (Perfect, Good, Marginal, Too Tight).
- **Estimated Speed**: Tốc độ dự kiến (tokens/giây).
- **Notes**: Ghi chú tùy chọn từ phân tích.

## 5. Kế hoạch thực hiện

### 5.1. Cập nhật [`synapse/tinychat/dashboard.html`](file:///c:/Users/Admin/Downloads/synapse_ai/synapse/tinychat/dashboard.html)

#### 5.1.1. Trang "System Information" (dashboard.html)
- **Thay đổi HTML/CSS**:
  - Tạo một phần riêng biệt để hiển thị thông tin phần cứng chi tiết.
  - Sử dụng các thành phần HTML phù hợp để hiển thị CPU, RAM, GPU.
  - Thêm CSS để tạo kiểu cho các phần này, đảm bảo dễ đọc và trực quan.
- **Tích hợp JavaScript**:
  - Gọi API để lấy thông tin phần cứng mới.
  - Cập nhật giao diện với thông tin nhận được từ API.

#### 5.1.2. Phần "Model Hardware Check" (dashboard.html)
- **Thay đổi HTML/CSS**:
  - Tạo một phần riêng biệt để hiển thị kết quả kiểm tra phần cứng cho mô hình.
  - Sử dụng các thành phần HTML phù hợp để hiển thị các chỉ số.
  - Thêm CSS để tạo kiểu cho các phần này, đảm bảo dễ đọc và trực quan.
- **Tích hợp JavaScript**:
  - Gọi API để lấy kết quả kiểm tra phần cứng cho mô hình.
  - Cập nhật giao diện với kết quả nhận được từ API.

#### 5.1.3. Trang quản lý Model đã có sẵn (dashboard.html) - Bổ sung thông tin tương thích phần cứng
- **Mục tiêu**: Bổ sung thông tin tương thích phần cứng vào trang quản lý model đã có sẵn.
- **Thay đổi HTML/CSS**:
  - Thêm cột mới vào bảng model hiện có để hiển thị mức độ phù hợp (Fit).
  - Thêm cột Memory Needed và Estimated Speed.
  - Thêm màu sắc để phân biệt mức độ phù hợp (🟢 Perfect, 🟡 Good, 🟠 Marginal, 🔴 Too Tight).
- **Tích hợp JavaScript**:
  - Gọi API để lấy thông tin tương thích phần cứng cho từng model.
  - Cập nhật giao diện với thông tin nhận được từ API.

### 5.2. Cập nhật [`synapse/api/chatgpt_api.py`](file:///c:/Users/Admin/Downloads/synapse_ai/synapse/api/chatgpt_api.py)

#### 5.2.1. API "System Information"
- **Tạo endpoint mới hoặc cập nhật endpoint hiện có**:
  - Endpoint: `GET /api/v1/system/info`
  - Trả về: Thông tin phần cứng chi tiết (CPU, RAM, GPU).
- **Tích hợp logic từ `device_capabilities.py`**:
  - Sử dụng `SystemSpecs.detect()` để lấy thông tin phần cứng.
  - Trả về thông tin dưới dạng JSON.

#### 5.2.2. API "Model Hardware Check"
- **Tạo endpoint mới hoặc cập nhật endpoint hiện có**:
  - Endpoint: `POST /api/v1/model/check`
  - Tham số: Tên mô hình, dung lượng RAM/GPU khả dụng.
  - Trả về: Kết quả kiểm tra phần cứng cho mô hình (Memory Needed, Fit, Estimated Speed).
- **Tích hợp logic từ `helpers.py`**:
  - Sử dụng các hàm trong `helpers.py` để ước tính tài nguyên và kiểm tra mức độ phù hợp.
  - Trả về thông tin dưới dạng JSON.

#### 5.2.3. API "Models Compatibility List"
- **Tạo endpoint mới**:
  - Endpoint: `GET /api/v1/models/compatibility`
  - Tham số: `filter` (Perfect, Good, Marginal, Too Tight), `sort` (score, params, memory), `page`, `limit`.
  - Trả về: Danh sách model với thông tin tương thích phần cứng.
- **Tích hợp logic từ `helpers.py`**:
  - Sử dụng các hàm trong `helpers.py` để ước tính tài nguyên và kiểm tra mức độ phù hợp cho từng model.
  - Trả về thông tin dưới dạng JSON.

## 6. Lợi ích của việc cập nhật

- **Cung cấp thông tin chi tiết**: Người dùng có thể hiểu rõ hơn về phần cứng và khả năng chạy mô hình của hệ thống.
- **Trải nghiệm người dùng tốt hơn**: Giao diện trực quan và dễ hiểu hơn.
- **Tăng tính minh bạch**: Người dùng có thể tin tưởng hơn vào kết quả kiểm tra phần cứng.

## 7. Kết luận

Việc cập nhật web UI sẽ giúp người dùng có được cái nhìn tổng quan và chi tiết hơn về hệ thống của họ và khả năng chạy các mô hình LLM. Kế hoạch này bao gồm việc cập nhật cả giao diện người dùng và các API hỗ trợ. Việc thực hiện kế hoạch này sẽ cải thiện đáng kể trải nghiệm người dùng và tính minh bạch của `synapse`.