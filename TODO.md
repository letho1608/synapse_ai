# TODO — Synapse AI

Cập nhật: 2026-11-02 · nhánh `distributed-v2`. Chi tiết kỹ thuật và số đo: `PLAN.md`, `README.md`.

## Mục tiêu

**Một cluster AI tại nhà kiểu [exo](https://github.com/exo-explore/exo):** gom các máy bình thường
(laptop, PC, có hoặc không GPU) thành một cụm để chạy model quá lớn cho một máy.

Định nghĩa "xong" (thứ tự là thứ tự ưu tiên):

1. **Chạy được trên nhiều máy thật.** Bật `synapse` trên 2–3 máy cùng LAN hoặc Tailscale, không cấu hình tay,
   các máy tự thấy nhau, tự chia layer theo bộ nhớ, chat được, có số đo TTFT, tok/s.
2. **Chạy model mà một máy không chứa nổi.** Mỗi máy chỉ tải và giữ phần layer của mình.
3. **Dễ dùng.** Cài một lệnh, API tương thích ChatGPT, có giao diện chat.
4. **Chịu lỗi.** Một máy rời cụm thì cụm tự chia lại và vẫn trả lời.

Mọi việc không phục vụ 4 điểm trên là **phụ**: không làm thêm cho đến khi điểm 1 có số đo.

## Hiện trạng so với mục tiêu

| Tính năng kiểu exo | Synapse | Ghi chú |
|---|---|---|
| Tự phát hiện thiết bị | Có | UDP, Tailscale, file cấu hình tay |
| Tự chia model theo tài nguyên | Có | Chia theo dải layer, trọng số theo bộ nhớ / tốc độ đo được |
| Mỗi node chỉ giữ layer của mình trong RAM | Có | `partial_loader` |
| Mỗi node chỉ tải phần trọng số nó cần | Có | Checkpoint chia nhiều file: chỉ tải file chứa tensor của node, lúc cần (`partial_loader`). Model một file thì vẫn phải tải cả file |
| API tương thích ChatGPT | Có | `/v1/chat/completions`, cổng 52415 |
| Giao diện chat | Một phần | Có TUI console; chưa có web chat |
| Nhiều loại máy / GPU | Một phần | PyTorch CPU (CUDA chưa kiểm kỹ); chưa có MLX cho Mac |
| Nhiều họ model | Một phần | Chạy qua HF; đường nhanh chỉ llama / mistral / qwen2 / qwen3 |
| Không có master (các node ngang hàng) | Khác | Synapse bầu master; exo thì các node ngang hàng |
| **Chạy thật trên nhiều máy** | **Chưa** | Mọi test đều trên một máy (loopback) |

## Việc cần làm (theo thứ tự)

### 1. Chạy thật trên nhiều máy — chặn mọi thứ khác
- [ ] Chạy 2 máy cùng LAN với SmolLM2-135M: tự thấy nhau, chia layer, chat được.
- [ ] Rồi 3 máy, rồi qua Tailscale.
- [ ] Đo TTFT, tok/s; so với chạy một máy.
- [ ] Rút một máy giữa chừng: đo thời gian cụm tự chia lại.
- [ ] Ghi mọi lỗi gặp được (firewall, cổng, IP, thời gian máy lệch) vào README.
- [ ] Harness phải chặn loopback giả làm nhiều máy.

### 2. Model lớn hơn một máy
- [x] **Mỗi máy chỉ tải phần trọng số nó cần** (2026-11-02):
  - [x] `checkpoint_index` chỉ đọc `model.safetensors.index.json`; file shard tải lúc lần đầu cần tensor trong đó.
  - [x] Bỏ `HFShardDownloader.ensure_shard` (tải cả repo, không còn ai gọi).
  - [x] Chia lại layer → chỉ tải thêm file còn thiếu.
  - [x] Test `test_sharded_checkpoint_fetches_only_needed_files`.
  - [ ] `/v1/models/pull` vẫn tải cả repo (lệnh tải tay); cân nhắc chỉ tải phần của node.
- [ ] Chạy một model ~7–8B chia cho 2–3 máy mà một máy không chứa nổi ở fp16.
- [ ] Đo int8 / int4 trên model lớn (trên SmolLM2 int8 chưa nhanh hơn fp32 vì ma trận quá nhỏ).

### 3. Dễ dùng
- [ ] Cài một lệnh (`install.sh` / `install.bat`) chạy được trên máy sạch.
- [ ] Web chat tối giản dùng API sẵn có.
- [ ] Kiểm CUDA; cân nhắc MLX cho Mac.

### 4. Chịu lỗi trên mạng thật
- [ ] Failover khi một máy rời cụm (đã có, chỉ mới kiểm trên một máy).
- [ ] Checkpoint không phụ thuộc thư mục dùng chung (C2).

### Việc vặt đang mở
- [ ] Một lượt chạy toàn bộ test sạch (lượt cuối dừng vì hết RAM; lượt trước 547/549, 2 test hỏng đã sửa).

## Tính năng ngoài lõi

Giữ lại (không bắt buộc giống exo): huấn luyện (pipeline, LoRA, DDP),
autoscale (advisory), TLS hai chiều, bầu master, ba bộ chia layer (chuỗi dự phòng: đo thật → TFLOPS / LACP),
lượng tử hoá + kernel. Ưu tiên vẫn là mục 1, nhưng không gỡ các phần này.

Trùng lặp còn lại, để gộp khi làm giao diện: ba giao diện terminal (`tui.py`, `console.py`, `cli.py`)
+ `viz/topology_viz.py`.

## Đã làm

### Nền tảng (các commit trước)
- Chia dải layer thật; verify draft một lượt; batching; đặt layer theo tài nguyên đo được; failover.
- KV cache tái dùng ngữ cảnh; transport TCP chunked; TLS + pin; checkpoint replicate sang peer; TUI console.

### Đợt 2026-11-02 (commit `930d468`)
- Cổng node mặc định 50051 → 46051 (dải 50000–50059 bị Windows giữ); test TLS dùng cổng OS cấp.
- Sửa chỗ chậm: cache tokenizer, bỏ `sleep` trên event loop, stats chạy trong thread, `OMP_WAIT_POLICY=PASSIVE`,
  prefill chỉ tính lm_head hàng cuối. TTFT 2,3 s → 0,39 s; 6,7 → 8,8 tok/s.
- Fast decoder họ Llama (~+40%, token giống hệt) + paged KV.
- Kernel numba int8/int4; int4 top-1 51% → 74%, KL 0,63 → 0,066.
- Train pipeline: mỗi node train layer của mình, khớp model đầy đủ (< 1e-5).

### Dọn dead code / trùng lặp (2026-11-02)
- Gỡ pipeline mô phỏng cũ: `pipeline/scheduler.py`, `pipeline/stage.py`, `DistributedTrainer.train_1f1b`
  (thay bằng `pp_training`), bảng "scaling" mô phỏng trong README và các test chỉ kiểm phần này.
- API: gỡ route trùng/không ai gọi (`/download`, `/initial_models`, `DELETE /models/{name}`,
  `/v1/settings/save`, `/v1/models/list` đăng ký 2 lần); `/v1/models/pull` tải trong thread thay vì chặn event loop.
- Hàm không ai gọi: `shards_overlap`, `get_checkpoint_manifest`, `get_monitor`, `resolve_hf_model_id` (2 bản),
  `PagedSequence.get_seq_length`; import thừa; `if True:` / biểu thức thừa.
- Gỡ tính năng trợ lý ảo: 11 tool gọi mạng ngoài (thời tiết, giá vàng, crypto, tin tức…), bắt từ khoá tiếng Việt
  tự gọi tool và chèn thêm chữ vào prompt người dùng, function calling `TOOL_CALL`, `requirements-tools.txt`.
  Bỏ cả trường `tools` trong request (server bỏ qua nếu client gửi). Endpoint chat giờ stream ngay từ token đầu.
