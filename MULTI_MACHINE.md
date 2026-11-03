# Chạy thật trên nhiều máy (C1)

Mục tiêu số 1 trong `TODO.md`. Mọi test hiện tại chạy trên một máy; tài liệu này là các bước để đo trên 2–3 máy thật.
Ghi lại mọi lỗi gặp được (kể cả đã tự sửa) vào cuối file.

## 0. Chuẩn bị (mỗi máy)

```bash
git clone https://github.com/letho1608/synapse_ai.git && cd synapse_ai
install.bat            # Windows  |  sh install.sh   # Linux/macOS
```

- Cùng một commit trên mọi máy (`git rev-parse HEAD`). Node khác phiên bản bridge chính sẽ bị loại khỏi cụm.
- Tải trước model trên mỗi máy để lần đo đầu không tính thời gian tải:
  `python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct')"`
- Đồng hồ các máy lệch < 100 ms (Windows: `w32tm /resync`, Linux: `timedatectl`).
- Mở cổng (firewall, cả TCP lẫn UDP nếu dùng UDP discovery):

| Cổng | Dùng cho |
|---|---|
| 46051 | node P2P (bridge) |
| 47051 | truyền tensor (cổng node + 1000, đổi bằng `SYNAPSE_TENSOR_PORT`) |
| 52415 | API (chỉ cần trên máy nhận request / máy đo) |
| 5678 (UDP) | UDP discovery trên LAN |

Windows: nếu cổng không bind được, xem `netsh int ipv4 show excludedportrange protocol=tcp` và dùng `--node-port`.

## 1. Bật node (mỗi máy)

```bash
# máy A và máy B, mỗi máy một lệnh:
python main.py --node-id A     # máy B: --node-id B (không cần --discovery-module: mặc định UDP trên LAN)
```

Mở giao diện web trên từng máy: `http://127.0.0.1:52415`.

Trên cùng LAN, các node tự tìm thấy nhau qua UDP, không cần làm gì thêm (truyền không mã hoá).

## 2. Ghép máy (tuỳ chọn, chỉ khi cần mã hoá)

Bỏ qua bước này trên LAN tin cậy. Cần mã hoá (Tailscale, mạng lạ) thì:
máy A: **Cài đặt → Ghép máy → Tạo mã cho cụm mới**, sao chép mã;
máy B: **Cài đặt → Ghép máy**, dán mã, **Tham gia cụm**. Khởi động lại cả hai node (Ctrl+C rồi chạy lại lệnh bước 1).
Đã ghép thì máy chưa ghép không vào được cụm.

## 3. Kiểm tra cụm

- Trang **Node & chia layer**: thấy cả A và B, mỗi node một dải layer, có độ trễ đo được giữa hai máy.
- Trang **Chat**: gửi "Say hello", có câu trả lời; dòng dưới câu trả lời ghi TTFT và tok/s.
- Thanh bên: master là ai; TLS `off` (không ghép) hoặc `cluster` (đã ghép).

## 4. Đo (chạy từ một máy thấy được API của mọi node)

```bash
python backend/tests/benchmark_live_cluster.py \
  --api-urls http://<IP-A>:52415,http://<IP-B>:52415 \
  --single-url http://<IP-A>:52415 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --max-tokens 64 --reps 5 --out c1-lan-2.jsonl
```

Harness từ chối (exit 3) nếu mọi URL là loopback: phải dùng IP LAN/Tailscale thật.
Đo: hình thành mesh (M1), TTFT (M2), tok/s (M3), văn bản temp-0 giống chạy một máy (M6).

## 5. Failover

Chạy lại lệnh đo với `--failover`; khi harness nhắc, tắt hẳn một máy đang giữ layer (rút mạng hoặc tắt tiến trình).
Ghi: thời gian cụm tự chia lại, câu trả lời đang stream có giữ nguyên không.

## 6. Mở rộng

- [ ] 3 máy LAN (bước 1–5 với 3 URL).
- [ ] Qua Tailscale: `--discovery-module tailscale` (cần `TAILSCALE_API_KEY`) hoặc `manual` với file cấu hình dùng IP 100.x.
- [ ] Rút máy master: các node còn lại phải chia lại ngay (không treo request mới).
- [ ] Train: trang **Huấn luyện**, bật LoRA, trên cụm 2 máy; rút một máy rồi xem checkpoint phục hồi.
- [ ] Model ~7–8B mà một máy không chứa nổi ở fp16 (đặt `SYNAPSE_WEIGHT_QUANT` = `none` / `int8` / `int4` để so).
- [ ] Máy có GPU NVIDIA: `python -c "import torch; print(torch.cuda.is_available())"`, chạy lại bước 4.

## Kết quả

Dán các file `*.jsonl` và ghi chú vào đây (máy, OS, mạng, commit).

## Lỗi gặp được

(trống)
