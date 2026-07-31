# PLAN.md — Việc chưa làm với synapse_ai

**Nguồn:** deep-dive phase 4 (`research/synapse-readme-audit/phase4-verification/01-synthesis.md`, `02-red-team.md`, `03-executive-briefing.md`), ngày 2026-07-31.
**Trạng thái:** README đã trung thực với code (220 test xanh). File này liệt kê những gì **chưa** hoàn thiện — chia 2 nhóm: (A) việc lớn đã khai báo trong README, (B) 7 mục nhỏ Medium/Low backlog.

---

## Phần A — Việc chưa làm (đã khai báo trong README "Known Limitations")

| # | Việc | Hiện trạng | Hướng sửa đề xuất | Ưu tiên |
|---|------|-----------|-------------------|---------|
| A1 | **Resume-from-checkpoint**: survivor nạp lại weights/KV từ checkpoint của node chết trước khi tiếp tục training | Checkpoint JSON được ghi sau mỗi step + staged qua `get_recovery_checkpoint()`, nhưng **chưa có production consumer** nạp lại | Trong `node_extension.py` / `check_fault_tolerance()` (`node.py:476–489`): sau rebalance, survivor gọi `fault_tolerance.get_recovery_checkpoint(target)` → `inference_engine.load_checkpoint(...)` trước khi resume | Tier 1 |
| A2 | **1F1B backprop pipeline thật** (hidden-state handoff giữa stage; `back_gradient` trả gradient thật) | Schedule pacing chạy end-to-end; gradient được ring all-reduce. Nhưng `pytorch_hf_engine.py:610–625` trả `(None, None)` cho `back_gradient`; `noop_send` ở `node_extension.py` | Implement backward thật theo `PipelineStage.backward_fn` + `send_tensor_fn`; bỏ no-op | Tier 1 (lớn) |
| A3 | **Distributed decode nhìn thấy full prompt** (prompt-conditioned end-to-end) | Context mỗi micro-batch chỉ là `[last] + guesses[:i]` (`distributed_engine.py:272–282`); prompt/history không vào được model | Truyền prompt/KV cache qua các micro-batch trong vòng lặp decode | Tier 0 (đo) → Tier 1 (sửa) |
| A4 | **Bridge chặn frame JSON >10 MiB** (≈5 MB tensor sau hex) | `p2p_socket_bridge.py:483–486` rớt frame quá lớn (đã có WARNING log + README disclosure) | Nâng cap bằng size header thương lượng, hoặc tự động route tensor lớn qua TCP/SHM | Tier 1 |
| A5 | **Live hardware multi-machine E2E** (2–3 máy vật lý, model thật) | Chưa từng chạy; mọi thứ kiểm chứng trên máy ảo in-process | Chạy `python main.py` trên 2–3 máy LAN/Tailscale; đo decode context, failover latency, training throughput | Tier 0 (gate trước khi production) |
| A6 | **LACP path cost**: ~0.5 s × node disk-I/O đo mỗi lần partition recompute; bandwidth proxy là single-message (≈8 Mbps-class, thấp hơn capacity) | `resource_weight.py:99,137` `time.sleep(0.5)`; `latency_matrix.py:169–175` `PROBE_SIZE*8/RTT` | Cache disk metrics vài giây; đo bandwidth nhiều message / song song | Tier 2 |
| A7 | **Rebalance không phụ thuộc request**: gate `self._distributed_pipeline` (`node.py:443`) khiến node chết trước request distributed đầu tiên không được rebalance | Đã khai báo trong README Running §5 | Init `_distributed_pipeline` vô điều kiện trong `node.start()` | Tier 2 |

---

## Phần B — 7 mục nhỏ (Medium/Low backlog, README không tuyên bố gì về chúng)

| # | Mục | Mô tả | Vị trí | Hướng sửa |
|---|-----|-------|--------|-----------|
| B1 | Token lặp hợp lệ bị bỏ | Dedup suffix trong streaming làm mất token lặp cạnh nhau chính đáng | `node.py:1709–1726` | Chỉ dedup token do speculative sinh ra, không dedup token thật |
| B2 | `_collect_result` không kiểm tra batch-identity | Nhận event đầu tiên bất kỳ trên topic kết quả, không verify đúng micro-batch | `distributed_engine.py:379–423` | Lọc theo `micro_batch_id` / `source_stage_id` |
| B3 | `_latency_cache` stale khi topology đổi | Cache init một lần qua `hasattr` guard; thêm/bớt node làm méo ma trận | `node.py:1620–1623` | Reset/rebuild cache khi topology change |
| B4 | `broadcast_authoritative_partition` chết | Không có caller → `_compute_lacp_partition` recompute mỗi forward khi bật LACP | `node.py:1777` | Dùng `_authoritative_partitions`; broadcast sau partition |
| B5 | Heartbeat theo membership + `_pending_rebalance` mồ côi | FT layer không trao đổi heartbeat riêng; `_pending_rebalance` được ghi không ai đọc | `node.py:1389–1399`; `fault_tolerance.py:110` | Gửi heartbeat message riêng; đọc `_pending_rebalance` trong rebalance |
| B6 | ILP contiguity constraints vacuous + placeholder probing | Ràng buộc contiguity chết (`x`/`y` linking) — contiguity vẫn đảm bảo bằng coverage; `networking/latency_probing.py` là bản thứ 2 không dùng, bịa 50/200 ms | `partitioner.py:205–218`; `networking/latency_probing.py` | Bỏ constraints chết; xóa/đồng bộ placeholder |
| B7 | Vệ sinh nhỏ | `MicroBatch.step` không bao giờ set (`stage.py`); `shm_transport._active` tăng không giới hạn; tên test còn `lacp3`; npm launcher trong root `main.py`; invariant benchmark↔README chưa có test ép buộc | nhiều file | Set step; giới hạn `_active`; rename; ép benchmark table bằng test |

---

## Quy tắc làm việc (đề xuất)

- **Mỗi mục = 1 commit nhỏ** + 1 test behavioral (dẫn shipped function thật, không soi nguồn).
- **Sau mỗi mục:** chạy test ảnh hưởng; **sau Phần A:** chạy full suite (220 test) + launch 2 lần.
- **A5 là gate cuối:** chỉ coi là production-ready sau live run 2–3 máy.
- Cập nhật README ngay khi thay đổi hành vi (giữ nguyên nguyên tắc "README = sự thật").
