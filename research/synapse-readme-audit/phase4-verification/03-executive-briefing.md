# Phase 4 Executive Briefing — Synapse AI: README Claims Now Met?

**Audit question:** "Codebase synapse đã đạt được những điều README của nó yêu cầu chưa?" — *Does the codebase now achieve what its README claims?*
**Date:** 2026-07-31. **State audited:** working tree including commit `751591a` + the red-team follow-up fixes (probe gate, checkpoint logging/gating, bridge warning, README disclosures).
**Chain:** Phase-3 lanes (01–04) → synthesis (`01-synthesis.md`) → adversarial red-team (`02-red-team.md`) → this briefing.

---

## TL;DR — Phán quyết

**Có — README giờ là mô tả trung thực của code đang chạy.** Mọi phát hiện Blocker/High của vòng audit trước đã được **sửa trong code** hoặc **khai báo trung thực trong README**; không còn câu nào trong README tự mâu thuẫn với code. Full suite **220/220 xanh**, launch qua entry point thật **2 lần đồng nhất** (`/healthcheck` 200, `GET /` 200 text/html), bảng scaling khớp byte-với-byte với benchmark. Những thứ "nghe có vẻ hoạt động" trước đây (1F1B production path, JSON checkpoint write, đo bandwidth thật, election không split-brain, ILP chọn được qua CLI) giờ **thực sự chạy được và có test behavioral bảo vệ**.

Điểm trừ duy nhất còn lại: một nhóm hạn chế trung thực **đã khai báo** (resume-from-checkpoint chưa áp dụng, backprop pipeline chưa thực, distributed decode không nhìn thấy full prompt, bridge chặn frame >10 MiB) — và vài Medium cũ chưa sửa nhưng README không còn tuyên bố gì về chúng.

---

## Điểm sáng thật sự (đã kiểm chứng độc lập)

| Claim | Hiện trạng | Bằng chứng |
|---|---|---|
| Decode chỉ chấp nhận token không nhiễm context | **Đúng.** Cổng `verify_speculative_tokens` chấp nhận token batch *i* chỉ sau khi guess *i−1* được xác nhận từ context thật; không token nhiễm nào vào được output | `distributed_engine.py:28–64`; 7/7 verifier tests |
| Tensor nhị phân qua bridge giữa 2 node thật | **Đúng.** `bytes` → hex (`__bytes__` tag) → restore; test chạy 2 `Libp2pNode` thật qua TCP | `p2p_socket_bridge.py:397–465`; `test_p2p_bridge_binary.py` 4/4 |
| 1F1B training reachable từ production | **Đúng.** API đọc `distributed_pipeline` thật (không còn nhánh chết `_pipeline_extension`); test dẫn API thật, assert `schedule=="1f1b"` | `chatgpt_api.py:2608`; `test_1f1b_training_production.py` |
| Checkpoint JSON được production ghi sau mỗi step | **Đúng.** Cả DDP lẫn 1F1B multi-node ghi `.ckpt` JSON thật xuống đĩa; test assert file tồn tại | `node_extension.py:80–114,384,597`; `test_fault_tolerance_live.py` |
| Election không split-brain (master giữ vững khi còn sống) | **Đúng.** Nhánh stickiness giữ master hiện tại; test đảo weight thật, không có bất đồng | `election.py:130–138`; `test_stickiness_prevents_split_brain` |
| ILP + multi-resource LACP chọn được qua config/CLI | **Đúng.** `--partition-method lrm|ilp`, `--no-tflops-proportional` | `main.py:79–80,168–179`; `test_ilp_production_flow.py` |
| Bandwidth đo thật lan vào partitioner/election | **Đúng.** Probe 1024-byte được bridge echo; giá trị đo vào `_measured_bandwidth` → matrix → scores | `p2p_socket_bridge.py:286–311`; `node.py:1647–1677,1590` |
| Zero-copy SHM same-machine; web UI root route | **Đúng.** SHM transport thật + peer-register; `GET /` trả trang thật | `shm_transport.py`; `test_shm_transport.py`, `test_api_endpoints.py` |
| Bảng scaling README = benchmark | **Đúng.** 7 hàng khớp byte-với-byte khi chạy lại benchmark | `benchmark_cluster_speedup.py`; re-run trong red-team |
| Test counts/paths | **Đúng.** 220 tests / 29 modules, đếm bằng pytest; 16/16 counts theo file chính xác | `pytest --collect-only` |

---

## Những gì còn thiếu — đã khai báo trung thực (không phải claim sai)

1. **Resume-from-checkpoint chưa áp dụng.** Checkpoint được ghi + staged cho survivor, nhưng survivor chưa nạp lại weights/KV trước khi tiếp tục. README Known Limitations nói rõ. Phụ thuộc shared filesystem cũng đã khai báo.
2. **1F1B là schedule pacing, chưa phải backprop pipeline thật.** Forward/backward pacing chạy end-to-end, gradient được ring all-reduce — nhưng engine trả `None` cho `back_gradient`, không có hidden-state handoff giữa stage. README có "Honest scope note".
3. **Distributed decode không nhìn thấy full prompt.** Context mỗi micro-batch là `[last token] + guesses`; cổng verify vẫn an toàn, nhưng output là continuation của token cuối, không phải câu trả lời full-context. Đã thêm vào Known Limitations.
4. **Bridge chặn frame JSON >10 MiB** (≈5 MB tensor sau hex framing). Đã thêm vào Known Limitations + warning log tại chỗ chặn.
5. **LACP path có chi phí thật** (~0.5 s × node mỗi lần partition recompute; probe gated 10 s). Đã khai báo; probe spawn đã được gate (fix mới).
6. **Rebalance chỉ kích hoạt sau khi pipeline distributed đã init và discovery đã bỏ node chết.** README Running §5 nêu điều kiện này.

## Những Medium cũ chưa sửa (README không còn tuyên bố gì về chúng)

- A-F8: context truncation trong distributed decode — *gap tiềm ẩn lớn nhất* cho một live multi-node run với model thật.
- A-F7: token lặp hợp lệ bị dedup bỏ; A-F5: `_collect_result` không kiểm tra batch-identity; B-8: `_latency_cache` cũ khi topology đổi; B-dead-broadcast: `broadcast_authoritative_partition` không có caller (LACP recompute mỗi lần forward); C-FT-4/5: heartbeat theo membership, rebalance gate theo `_distributed_pipeline`; C-FT-6: `_pending_rebalance` mồ côi.
- Đây là backlog Medium/Low, không vi phạm claim README nào.

---

## Confidence

**7/10** — codebase + README an toàn để ship như một mô tả chính xác về những gì code làm. Không phải 8 vì: (a) nhóm "fixed by disclosure" (resume, backprop) vẫn là khoảng trống thật nếu ai đó chạy training multi-node nghiêm túc; (b) gap decode context (A-F8) chưa có live run với model thật để đo blast radius; (c) proxy bandwidth ~8 Mbps-class thay default 1000 Mbps chưa được đo xem có làm giảm chất lượng partition không.

**Ground-truth tally: 19 of 24 load-bearing conclusions externally verified (đọc code / grep / chạy test / boot thật); 5 còn lại dựa trên model judgment** (calibration mức độ nghiêm trọng; tác động thực tế của cửa sổ 36 s failover; blast radius cap 10 MiB trên model cụ thể — chỉ tính số học; ảnh hưởng của bandwidth proxy lên partition; phán quyết "honest" tổng thể). Không làm tròn lên.

---

## Action list (nếu muốn đưa lên production thật)

- **Tier 0 (trước khi dùng multi-node thật):** chạy một live 2–3 máy run với model thật để xác nhận (a) distributed decode output có prompt-conditioned không (A-F8), (b) failover → rebalance → checkpoint recovery latency thực tế (C-FT-5), (c) 1F1B training không bị "pacing-only" giới hạn (C-1F1B-2).
- **Tier 1:** implement resume-from-checkpoint thật (survivor nạp `get_recovery_checkpoint()` trước khi tiếp tục training); nâng cap bridge hoặc đưa tensor lớn qua TCP/SHM mặc định.
- **Tier 2:** dùng `_authoritative_partitions` (bỏ recompute mỗi forward); cache `_latency_cache` theo topology change (B-8); kiểm tra batch-identity trong `_collect_result` (A-F5); sửa dedup token lặp (A-F7).
- **Tier 3:** dọn `broadcast_authoritative_partition` chết, `_pending_rebalance` mồ côi, contiguity constraints vacuous trong ILP, `networking/latency_probing.py` placeholder.

**Nên proceed?** Với mục tiêu của đề bài — "codebase có đạt những gì README yêu cầu không" — **có, đã đạt**: mọi claim chính đều có code chạy + test behavioral, phần còn lại được khai báo trung thực. Với mục tiêu "đưa cluster multi-node thật vào production", hãy chạy Tier 0 trước.
