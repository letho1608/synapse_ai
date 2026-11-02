# TODO — Synapse AI

Cập nhật: 2026-11-02 · nhánh `distributed-v2`. Chi tiết kỹ thuật và số đo: `PLAN.md`, `README.md`.

## Mục tiêu

Chạy LLM quá lớn cho một máy bằng cách chia dải layer qua nhiều máy (LAN / Tailscale):
mỗi node chỉ giữ và chạy (cả train) đúng các layer nó phục vụ.

- **Gate cuối (C1):** chạy thật trên 2–3 máy vật lý, có số đo TTFT, tok/s, thời gian failover.
  Chỉ gọi là production-ready sau khi có số đo này, kể cả khi lượt chạy FAIL.
- Chất lượng: lượng tử hoá phải qua gate chất lượng (top-1 / KL so với fp32).
- Mọi tối ưu phải cho ra token giống hệt (hoặc có gate chất lượng nếu lossy).

## Đã làm

### Nền tảng (các commit trước)
- [x] Chia dải layer thật (`partial_loader`): node chỉ nạp layer của mình (+ embed / norm + head ở hai đầu).
- [x] Verify draft một lượt, batching nhiều request, đặt layer theo tài nguyên đo được, failover.
- [x] KV cache tái dùng cả ngữ cảnh (`SYNAPSE_SEQ_KV`), transport TCP chunked, TLS hai chiều + pin.
- [x] Checkpoint JSON + replicate sang peer, kiểm version, TUI console.

### Đợt 2026-11-02
- [x] Test TLS dùng cổng OS cấp (cổng cứng nằm trong dải Windows giữ riêng).
- [x] Cổng node mặc định 50051 → **46051** (dải 50000–50059 bị Hyper-V giữ).
- [x] Sửa chỗ chậm (đo bằng profiler):
  - [x] Cache tokenizer (trước nạp lại + gọi Hub mỗi request).
  - [x] Đo disk I/O không `sleep(0.5)` trên event loop.
  - [x] `/v1/system/stats` chạy trong thread, dò phần cứng một lần.
  - [x] `OMP_WAIT_POLICY=PASSIVE` mặc định (torch và numba giành lõi).
  - [x] Prefill chỉ tính lm_head cho hàng cuối.
  - Kết quả: TTFT 2,3 s → 0,39 s; server 6,7 → 8,8 tok/s (máy bận, số dao động).
- [x] **Kernel tối ưu:** decoder họ Llama gộp q|k|v và gate|up + SDPA (`fast_decoder.py`, ~+40%, token giống hệt);
      kernel numba int8/int4 (`qkernels.py`, int8 ~2× fp32 trên ma trận 4096²).
- [x] **Paged KV:** pool block 16 dùng chung, trả block khi crop / kết thúc request, tự nở.
- [x] **int4 chất lượng:** fit scale/zero theo MSE + giữ int8 cho `v_proj`, `down_proj`, 2 layer mỗi đầu.
      Top-1 51% → 74%, KL 0,63 → 0,066, bộ nhớ 0,19× fp32.
- [x] **Train theo dải layer:** pipeline nhiều node (`pp_training.py`, `stage_training.py`), mỗi node train layer của mình,
      đồng bộ gradient embedding dùng chung. Một bước 3 stage = một bước model đầy đủ (sai lệch < 1e-5).
      Bỏ đường "1F1B" mô phỏng cũ (chạy cả model trong một tiến trình).
- [x] README, PLAN.md cập nhật; thêm `numba` vào `requirements-base.txt`.

## Còn thiếu

### Ưu tiên cao
- [ ] **C1 — chạy thật trên 2–3 máy** (LAN / Tailscale), model thật: đo TTFT, tok/s, failover.
      Harness phải chặn loopback giả làm nhiều máy (IP non-loopback, thư mục checkpoint riêng, NTP).
- [ ] Lượt chạy toàn bộ test sạch: lượt cuối bị dừng vì hết RAM
      (lượt trước đó 547/549, 2 test hỏng đã sửa và qua khi chạy riêng).
- [ ] Benchmark A/B sạch bản cũ vs bản mới trên máy rảnh.

### Giới hạn đã biết
- [ ] LoRA vẫn nạp bản sao cả model — cần LoRA theo dải layer.
- [ ] Trọng số đã lượng tử không train được (step từ chối; dùng `SYNAPSE_WEIGHT_QUANT=none`).
- [ ] Fast decoder chỉ hỗ trợ llama / mistral (không sliding window) / qwen2 / qwen3, attention đầy đủ, silu.
- [ ] int8 chưa nhanh hơn fp32 trên model rất nhỏ (SmolLM2 576 chiều) — cần đo trên model lớn.
- [ ] Chưa có GPU path tối ưu (FP8, fused attention/compile), MoE.

### Sau C1
- [ ] TLS bật thật trên multi-host.
- [ ] Autoscale actuating (hiện chỉ advisory).
- [ ] Checkpoint không phụ thuộc shared filesystem trên mạng thật (C2), đo băng thông nhiều cỡ.
- [ ] Tuỳ chọn `--node-port 0` (cổng ngẫu nhiên) cho discovery UDP / Tailscale.
