# 10 — Phát hiện môi trường máy này (2026-09-11, kiểm chứng bằng chạy thật)

## E1. torch_directml vỡ với torch 2.13 CPU (đã kiểm chứng)
- `python -c "import torch_directml"` → `ImportError: DLL load failed ... The specified procedure could not be found` (sạch, bắt được).
- NHƯNG import nó SAU KHI torch đã load trong pytest → crash FATAL cả process (0xc0000139), không bắt được.
- Hệ quả: test parity model thật phải chích `sys.modules["torch_directml"] = None` (xem test_seq_kv_parity.py fixture) để ép đường CPU.
- Việc cho bạn anh: cài lại torch-directml đúng build khớp torch đang dùng NẾU cần đường Intel-GPU; nếu không, gỡ nó đi cho khỏi nổ process. Đừng bao giờ `import torch_directml` sau khi torch đã load khi chưa chắc DLL khớp.

## E2. Model thật chạy được trên CPU ở đây (đã kiểm chứng)
- Cache có sẵn: tiny-random-GPT2LMHeadModel, tiny-gpt2 (sshleifer), Qwen2.5-1.5B-Instruct.
- Parity incremental-KV == full-refeed trên tiny-random, 8 bước greedy, token-identical (test_seq_kv_parity.py).
- Suy ra: vòng decode distributed CÓ THỂ kiểm chứng đúng/sai trên CPU trước khi cần GPU; chỉ số tốc độ (tok/s) vẫn phải đo trên GPU + máy thật.

## E3. Số đo ILP frontier (từ test_partition_scale.py, máy này)
- Budget 2s: 2x8/4x8/8x8 Optimal; 4x32 Optimal; 2x32 và 8x32 fallback LRM (Not Solved).
- Kết luận: ILP không qua nổi L=32 ở 2 và 8 node với budget 2s — C5 "chưa kiểm chứng ≤32 node" vẫn đúng, giờ có số.
