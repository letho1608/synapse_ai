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
| 46051 | node P2P (bridge + tensor) |
| 52415 | API (chỉ cần trên máy nhận request / máy đo) |
| 5678 (UDP) | UDP discovery trên LAN |

Windows: nếu cổng không bind được, xem `netsh int ipv4 show excludedportrange protocol=tcp` và dùng `--node-port`.

## 1. Ghép máy (một lần)

```bash
python main.py pair                  # máy 1: in ra mã
python main.py pair --code <mã>      # các máy còn lại
```

## 2. Hai máy cùng LAN

```bash
# máy A và máy B, mỗi máy một lệnh:
python main.py serve --discovery-module udp --node-id A     # (B: --node-id B)
```

Kiểm tra từ máy bất kỳ:

```bash
python main.py status --port 52415          # thấy cả A và B, mỗi node một dải layer
python main.py topology --port 52415
python main.py client --prompt "Say hello"  # có câu trả lời
```

## 3. Đo (chạy từ một máy thấy được API của mọi node)

```bash
python backend/tests/benchmark_live_cluster.py \
  --api-urls http://<IP-A>:52415,http://<IP-B>:52415 \
  --single-url http://<IP-A>:52415 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --max-tokens 64 --reps 5 --out c1-lan-2.jsonl
```

Harness từ chối (exit 3) nếu mọi URL là loopback: phải dùng IP LAN/Tailscale thật.
Đo: hình thành mesh (M1), TTFT (M2), tok/s (M3), văn bản temp-0 giống chạy một máy (M6).

## 4. Failover

Chạy lại lệnh đo với `--failover`; khi harness nhắc, tắt hẳn một máy đang giữ layer (rút mạng hoặc tắt tiến trình).
Ghi: thời gian cụm tự chia lại, câu trả lời đang stream có giữ nguyên không.

## 5. Mở rộng

- [ ] 3 máy LAN (bước 2–4 với 3 URL).
- [ ] Qua Tailscale: `--discovery-module tailscale` (cần `TAILSCALE_API_KEY`) hoặc `manual` với file cấu hình dùng IP 100.x.
- [ ] Rút máy master: các node còn lại phải chia lại ngay (không treo request mới).
- [ ] Train: `POST /v1/training/start` với `use_lora: true` trên cụm 2 máy; rút một máy rồi xem checkpoint phục hồi.
- [ ] Model ~7–8B mà một máy không chứa nổi ở fp16 (đặt `SYNAPSE_WEIGHT_QUANT` = `none` / `int8` / `int4` để so).
- [ ] Máy có GPU NVIDIA: `python -c "import torch; print(torch.cuda.is_available())"`, chạy lại bước 3.

## Kết quả

Dán các file `*.jsonl` và ghi chú vào đây (máy, OS, mạng, commit).

## Lỗi gặp được

(trống)
