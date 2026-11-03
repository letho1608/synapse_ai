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

Ưu tiên điểm 1 trước: đa số việc khác (kể cả các tính năng hỗ trợ bên dưới) chỉ kiểm được khi đã có cụm thật.

## Hiện trạng so với mục tiêu

| Tính năng kiểu exo | Synapse | Ghi chú |
|---|---|---|
| Tự phát hiện thiết bị | Có | UDP, Tailscale, file cấu hình tay |
| Tự chia model theo tài nguyên | Có | Chia theo dải layer, trọng số theo bộ nhớ / tốc độ đo được |
| Mỗi node chỉ giữ layer của mình trong RAM | Có | `partial_loader` |
| Mỗi node chỉ tải phần trọng số nó cần | Có | Checkpoint chia nhiều file: chỉ tải file chứa tensor của node, lúc cần (`partial_loader`). Model một file thì vẫn phải tải cả file |
| API tương thích ChatGPT | Có | `/v1/chat/completions`, cổng 52415 |
| Giao diện | Có | Web UI do node phục vụ tại :52415 (chat, node, model, huấn luyện, dataset, log, cài đặt) |
| Nhiều loại máy / GPU | Một phần | PyTorch CPU (CUDA chưa kiểm kỹ); chưa có MLX cho Mac |
| Nhiều họ model | Một phần | Chạy qua HF; đường nhanh chỉ llama / mistral / qwen2 / qwen3 |
| Không có master (các node ngang hàng) | Khác | Synapse bầu master; exo thì các node ngang hàng |
| **Chạy thật trên nhiều máy** | **Chưa** | Mọi test đều trên một máy (loopback) |

## Việc cần làm (theo thứ tự)

### 1. Chạy thật trên nhiều máy — chặn mọi thứ khác
Các bước từng bước: `MULTI_MACHINE.md`.
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
  - [x] `/v1/models/pull` chỉ tải config/tokenizer + file trọng số của dải layer node đang giữ (`{"full": true}` = cả repo).
  - [x] Model một file safetensors chưa có trong cache: đọc header + đúng các đoạn byte của tensor node cần bằng HTTP Range
        (`RangeFile`), không tải cả file (`SYNAPSE_RANGE_FETCH=0` để tắt).
- [ ] Chạy một model ~7–8B chia cho 2–3 máy mà một máy không chứa nổi ở fp16.
- [ ] Đo int8 / int4 trên model lớn (trên SmolLM2 int8 chưa nhanh hơn fp32 vì ma trận quá nhỏ).

### 3. Dễ dùng
- [x] Cài một lệnh: requirements thiếu `msgpack` (gossip trạng thái cụm hỏng im lặng trên máy sạch), `lz4`, `cryptography`
      → đã thêm; bỏ 7 gói không ai import. Đã cài thật vào venv sạch và chạy toàn bộ test (xem "Đã làm").
      Còn: thử trên một máy khác hẳn (Linux/macOS).
- [x] Giao diện web (`frontend/`, khôi phục từ commit trước `e6f4b0e`, bỏ toàn bộ dữ liệu giả, nối API thật):
      tổng quan, node & chia layer, chat (stream, đo TTFT/tok/s), model (tải phần của node), huấn luyện (full/LoRA),
      dataset, log, cài đặt (node, ghép máy, API key). Đã gỡ console/TUI/CLI.
- [ ] Web UI cần Node.js để build; máy không có Node chỉ có trang trạng thái cơ bản.
- [ ] Kiểm CUDA (cần máy NVIDIA); MLX cho Mac (cần máy Mac).

### 4. Chịu lỗi trên mạng thật
- [ ] Failover khi một máy rời cụm (đã có, chỉ mới kiểm trên một máy).
- [x] Checkpoint không phụ thuộc thư mục dùng chung (C2): bản sao gửi sang node kế tiếp. Còn phải thử trên máy thật.

### Việc vặt đang mở
- [ ] Một lượt chạy toàn bộ test sạch sau đợt web UI (lần trước dừng ở 83% vì hết RAM, xem "Đã làm").

## Tính năng hỗ trợ mục tiêu nhưng chưa hoàn thiện

Không phải phần phụ: mỗi tính năng phục vụ một tiêu chí ở trên, nhưng mới làm được một nửa.
Làm sau mục 1 vì đa số cần cụm thật mới kiểm được.

### Bầu master (phục vụ 1, 4)
- [ ] Chưa thử trên mạng thật: mất gói, hai nhóm máy mất liên lạc với nhau (có thể sinh hai master).
- [x] Master chết: node còn sống bỏ bảng chia layer có node chết và tự chia lại trên node sống ngay (trước đây mọi request
      dồn vào node chết, chờ timeout failover cho tới khi có master mới); request đang chạy được vá bằng vòng failover sẵn có.
      Node nhận request (orchestrator) chết thì kết nối đó kết thúc, client gửi lại sang node khác (`test_master_death.py`).

### Autoscale (phục vụ 2, 4)
- [x] Tự hành động (`SYNAPSE_AUTOSCALE=on`, mặc định): tải cao + có máy rảnh vừa bộ nhớ → master chia layer theo throughput
      (trải ra nhiều máy hơn); rảnh → về objective cấu hình. Có cooldown 60 s chống dao động. `advise` = chỉ log.
- [ ] Đo trên cụm thật (ngưỡng rpm, cooldown) khi mục 1 xong.

### TLS (phục vụ 3: dùng an toàn qua Tailscale / LAN lạ)
- [x] Ghép máy bằng mã: trang Cài đặt sinh mã 160 bit; dán mã ở trang Cài đặt của máy khác. Mọi máy dẫn xuất cùng
      CA cụm từ mã, tự cấp cert riêng, mutual TLS; máy khác mã / không mã không bắt tay được. Có mã thì TLS bật mặc định.
- [ ] Chưa chạy TLS giữa hai máy thật.

### Huấn luyện (phục vụ 2: train model lớn hơn một máy)
- [x] LoRA trên dải layer của từng node (`training/stage_lora.py`), không nạp cả model; 3 stage == 1 node; adapter nằm
      trong checkpoint nên node thay thế phục hồi được; `stage_lora_merge` gộp vào trọng số.
- [x] LoRA trên trọng số int8/int4 (kiểu QLoRA). Full fine-tune trọng số lượng tử vẫn bị từ chối (có hướng dẫn).
- [ ] Train pipeline mới khớp trên một máy, chưa chạy qua mạng thật.

### Checkpoint khi train (phục vụ 4)
- [x] Đã gửi bản sao checkpoint sang node kế tiếp (`_replicate_checkpoint`), không cần thư mục chung.
- [ ] Chưa thử rút máy thật rồi phục hồi từ bản sao; node giữ bản sao chết cùng lúc thì mất checkpoint.

### Lượng tử hoá (phục vụ 2)
- [ ] int8 chưa nhanh hơn fp32 trên model nhỏ; cần đo trên model 7–8B (trùng mục 2).

Ba bộ chia layer giữ nguyên: là chuỗi dự phòng (đo thật → TFLOPS / LACP), không trùng nhau.

Giao diện: chỉ còn web UI. Console/TUI/CLI (`console.py`, `tui.py`, `cli.py`, `cli_defaults.py`, `theme.py`) và
`viz/topology_viz.py` đã gỡ.

## Đã làm

### Giao diện web thay CLI (2026-11-03)
- Khôi phục `frontend/` (Vite + TS): giữ thiết kế, viết lại lớp dữ liệu theo API thật; lỗi API hiện thành thông báo,
  không còn số giả (bản cũ hiện "3 node RTX 4090", "341 tok/s" khi API lỗi; proxy trỏ nhầm cổng 8000).
- Backend phục vụ `frontend/dist` tại `/` (file tĩnh giới hạn trong `dist`); thêm `/v1/cluster/pairing` (thay `synapse pair`);
  `/v1/distributed/status` có master, TLS, autoscale; route đọc trạng thái mở cho LAN để mở UI từ máy khác.
- Sửa "model đã tải": trước chỉ tính khi cả repo trong cache, nên model tải theo phần của node không bao giờ hiện.
- Sửa huấn luyện: model không có chat template (vd tiny-GPT2) bị loại hết mẫu dạng `messages`; giờ rơi về định dạng role thường.
- Gỡ console/TUI/CLI và test của chúng; `python main.py` bật node; installer build UI khi có npm.
- Kiểm bằng Chrome headless trên server thật: 8 trang không lỗi JS, chat stream, ghép máy tạo/bỏ, xem dataset,
  LoRA tiny-GPT2 chạy xong từ UI. Lượt chạy toàn bộ test bị dừng ở 83% vì máy hết RAM; 2 lỗi thấy được đã sửa /
  qua khi chạy riêng (`test_parent_watchdog`), phần còn lại chưa chạy.

### Đợt hoàn thiện (2026-11-02, commit `aad4daf`, `e530312`, `302927e`)
- Bảo mật API: không có `SYNAPSE_API_KEY` thì chat + trạng thái mở cho LAN, route quản trị (quit, settings, pull/xoá model,
  dataset, training, log) chỉ nhận từ máy local; có key thì mọi route trừ `/` và `/healthcheck` cần Bearer (so sánh hằng thời gian).
  Trước đây ai cùng mạng cũng gọi được `/quit` và xoá model; `/quit` không được bảo vệ kể cả khi có key.
- `/v1/models/pull` chỉ tải phần của node; đọc model một file bằng HTTP Range.
- LoRA theo dải layer + QLoRA; ghép máy bằng mã (mutual TLS); autoscale tự hành động; master chết không làm request treo.
- Gỡ `viz/topology_viz.py` và cờ `serve --tui`.
- requirements: thêm `msgpack`, `lz4`, `cryptography`; bỏ 7 gói không dùng.
- Cài sạch thật (venv mới, torch 2.14.1 / transformers 5.20.2 / huggingface_hub 2.4.0) tìm ra 2 lỗi chặn:
  - `torch-directml` (mặc định trên Windows) ghim torch 2.4.1 < 2.6 mà transformers 5 cần → bỏ khỏi bộ cài mặc định
    (`requirements-windows.txt` thành tuỳ chọn có cảnh báo), thêm `torch>=2.6`.
  - transformers 5.20 không nhận `cache.crop(<độ dài dương>)` nữa → KV cache cắt sai, decode/verify hỏng.
    `_crop_kv` gửi số âm chỉ cho các layer stage đã ghi (chạy được cả bản cũ lẫn mới).
  - `install.bat` gợi ý khi gặp giới hạn 260 ký tự đường dẫn của Windows.
- `test_tcp_tls.py` dùng cổng OS cấp (cổng cố định 59421 bị tiến trình khác chiếm).

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
