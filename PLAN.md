# PLAN.md — Việc còn lại với synapse_ai

**Nguồn:** deep-dive phase 4 (ngày 2026-07-31) + đối chiếu code ngày 2026-09-11.
**Trạng thái:** Phần A/B cũ đã xong hết (xem bảng đối chiếu). File này giữ lại lịch sử + liệt kê gap production còn lại (Phần C), hướng lấy từ FreeToken / mesh-llm (Phần D), và chốt hướng **CLI-first, không dashboard nặng (Phần E)**.

---

## Phần A — Đối chiếu (từng là "việc chưa làm", nay DONE)

| # | Việc | Hiện trạng 2026-09-11 | Commit sửa |
|---|------|----------------------|------------|
| A1 | Resume-from-checkpoint: survivor nạp lại checkpoint | **REOPENED (red-team B1, 2026-09-11)** — rebalance có gọi `engine.load_checkpoint()` (`orchestration/node.py:554-575`) nhưng `PyTorchHF.load_checkpoint()` là `pass` (`inference/pytorch_hf_engine.py:586-587`) trong khi save vẫn persist (`:589-596`); staging không check identity (`fault_tolerance.py:334-339`). Survivor khôi phục metadata, KHÔNG khôi phục weights | `120f856 A1` (wire) + REOPEN `08-red-team B1` — implement real load hoặc `raise NotImplementedError`; test assert weight-hash đổi, không chỉ flag |
| A2 | 1F1B backprop thật | DONE — `inference/pytorch_hf_engine.py:613-704` torch autograd thật + seed bằng downstream grad; chỉ trả `(None,None)` khi không có model | `5c46160 A2` |
| A3 | Distributed decode full prompt | DONE — `inference/distributed_engine.py:290` dùng `full_context = prompt + history`; `_collect_result` check batch-identity | `22201b1 A3+B1+B2` |
| A4 | Bridge chặn frame >10 MiB | DONE 1 nửa — cap nâng lên 256 MiB configurable + explicit `frame_too_large` (`routing/p2p_socket_bridge.py:22,505-514`); test 12 MB tensor pass | `5e6bbf0 A4` |
| A5 | Live hardware multi-machine E2E | CHƯA — vẫn là gap C1 bên dưới; `README:168` tự nhận chỉ verify in-process | — |
| A6 | LACP path cost (disk sleep 0.5s, bandwidth proxy) | DONE 1 nửa — disk-I/O cache 5s (`topology/resource_weight.py:141-144`); bandwidth probe 64 KiB transfer-time 5 samples (`topology/latency_matrix.py:138-179`) | `013ec53 A6+A7` |
| A7 | Rebalance không phụ thuộc request | DONE — init pipeline vô điều kiện trong `node.start()` (`orchestration/node.py:270-278`) | `013ec53 A6+A7` |

## Phần B — 7 mục nhỏ (nay DONE)

| # | Mục | Hiện trạng |
|---|-----|-----------|
| B1 | Token lặp bị dedup sai | DONE (`22201b1`) |
| B2 | `_collect_result` không check batch-identity | DONE — lọc `micro_batch_id` + `source_stage_id` (`distributed_engine.py:434-445`) |
| B3 | `_latency_cache` stale | DONE — refresh khi topology đổi (`e0a8068 B3+B4+B5`) |
| B4 | `broadcast_authoritative_partition` chết | DONE — đã wire broadcast (`e0a8068`) |
| B5 | Heartbeat + `_pending_rebalance` mồ côi | DONE — heartbeat riêng `synapse/cluster/heartbeat` + consume trong rebalance (`node.py:288,504-510`) |
| B6 | ILP contiguity vacuous + placeholder probing | DONE — bỏ constraints chết, xóa prober bịa số (`610458a`) |
| B7 | Vệ sinh nhỏ | DONE — bound SHM `_active` (LRU), invariant benchmark↔README có test ép (`610458a`, `46db903`) |

---

## Phần C — Gap production còn lại (2026-09-11)

| # | Gap | Vì sao quan trọng | Hướng sửa |
|---|-----|-------------------|-----------|
| C1 | **Chưa có live E2E 2-3 máy vật lý** (kế thừa A5) | Mọi test là in-process/loopback; checkpoint recovery, decode context, failover latency chưa ai đo cross-machine. **Red-team B3: gate hiện tại đậu được trên loopback** — harness bắt buộc: IP non-loopback phân biệt + MAC, shared-nothing checkpoint dirs (assert), NTP<100ms, pre-pulled ckpt (repo+rev), bảng ports 50051/5678/52415 mở từng port, trusted-network label + Tailscale status; loopback → exit 3 `loopback-masquerade`, không PASS; wire-or-delete `--node-host` + expose bind/advertised addrs | Chạy `python main.py` trên 2-3 máy LAN/Tailscale với model thật; đo time-to-first-token, token/s, failover s; gate trước khi gọi là production |
| C2 | **Checkpoint phụ thuộc shared filesystem** | `load_latest_checkpoint()` quét dir local — node khác máy không đọc được file của node chết (`README:164`). **Red-team H5: pull phải confine `Path.resolve().is_relative_to(checkpoints/)` (traversal), sha256-before-apply (không signature → MITM trên LAN), size cap + per-peer in-flight cap** | Dual-read memory→P2P pull→shared-dir (giữ tripwire: chỉ xóa shared-dir sau demo C1); format giữ JSON |
| C3 | **Decode đúng nhưng phí: re-feed full context, chưa tái dùng KV-cache** | `README:165` tự nhận — mỗi step chạy lại cả prompt qua pipeline | Paged KV + prefix/radix cache liên stage; chỉ gửi delta + KV handle (xem D1) |
| C4 | **Transport yếu: JSON bridge + cap 256 MiB, không chunking, không mã hóa** | Tensor lớn vẫn có trần; JSON+hex phình payload; không TLS; SHM chỉ cùng máy (`README:162-163`). **Red-team B4 (blocker): `node.py:1338-1350` ép INT8 mọi float tensor, không opt-out — sửa exact-dtype-default TRƯỚC mọi chunking/KV**; bridge Nagle đang bật (TCP transport có NODELAY, bridge không) | `quantize_requested` default False + fp32 bit-exact test; binary framing + chunk/stream + `transport_selected`/forward observability; TCP/SHM auto-route; TLS cho inter-node; lâu dài QUIC multiplex (xem D2) |
| C5 | **Bandwidth probe chỉ là lower-bound đơn kết nối** | 64 KiB probe × 5 samples đo RTT, không phản ánh capacity thực; partition ILP bound ≤32 node chưa kiểm chứng | Probe song song/đa message + EWMA; benchmark partition scale; công bố điều kiện áp dụng scaling table |
| C6 | **Engine đơn node yếu: không quantization, không fused kernel, không elastic VRAM** | Chạy PyTorch naive + numpy — thua xa FreeToken trên cùng GPU | Lấy quantized expert/GEMM, paged KV, chunked prefill, elastic budget (xem D1) |
| C7 | **Thiếu lớp vận hành CLI-first** | Không có CLI chuẩn (serve/join/status/topology/benchmark), log JSON cho script, metrics tối thiểu; dashboard nặng chưa cần. **Red-team M4: GPU summary + prints đang ra stdout — `--json` single-doc và JSONL fail ngày đầu nếu không chuyển prints→stderr, TUI→stderr**; `eval`/`train` wired-or-removed; secrets (Tailscale key) env/file-only + redacted | CLI-first theo Phần E; chỉ lấy status API + TUI text + packaging từ mesh-llm (xem D2) |

---

## Phần D — Lấy gì từ 2 repo còn lại

### D1. Từ FreeToken (`../FreeToken/python/freetoken/`) — tối ưu 1 node
- `kvcache/` (paged KV pools + radix prefix cache) + `scheduler/` (chunked prefill) → sửa C3: hết re-feed full context.
- `moe/` (expert offload LRU cache, quantized experts MXFP4/NVFP4/FP8) + `kernel/` (Triton/CUDA, `csrc/`) → sửa C6: cùng GPU chạy model lớn hơn, nhanh hơn.
- `engine/` (cache budget planning, elastic VRAM giữa expert cache và KV) → tránh OOM khi partition đổi.
- `checkpoint/` (FTW fast-load) + semantic anchor checkpoint → thay JSON checkpoint thô, hợp với C2.
- Lấy ý tưởng trước, không bê nguyên CUDA lock-in (FreeToken chỉ Linux+NVIDIA).

### D2. Từ mesh-llm (`../mesh-llm/crates/`, `docs/design/DESIGN.md`) — liên máy
- `network/` + `protocol/` (QUIC iroh, protobuf `GossipFrame`, stream 0x01-0x07, additive-only) → sửa C4: thay JSON bridge bằng binary multiplex, versioned, mixed-version tương thích.
- `models/capabilities.rs` + `network/router.rs` (capability-aware routing, prefix-affinity, session-sticky) → sửa C7: request chỉ tới peer đủ năng lực, giảm cross-node traffic kiểu MoE-sharding của mesh-llm.
- `runtime/` demand map qua gossip + standby promotion (`DESIGN.md: Demand-Aware Rebalancing`) → sửa C7: autoscale thay vì partition tĩnh.
- `inference/skippy/` (B2B direct transfer, worker-to-worker 1 hop) + latency-aware split (`rtt<80ms`, greedy VRAM×1.1) → sửa C4/C5: stage traffic đi thẳng, bớt qua host.
- `api/` (chỉ lấy `:3131`-style status/models/events tối thiểu cho CLI + script, không bê nguyên web console) + `mesh-llm-tui` (TUI text) + `metrics-server` tối thiểu + packaging (install.sh/ps1, OCI) + speculative decoding/MTP → sửa C7/C1: có công cụ đo live E2E thật.
- Không lấy: mô hình "mỗi node 1 model + router" nguyên xi (đi ngược pipeline-parallel của synapse); không lấy web dashboard nặng / mobile app / MoA / plugin system — synapse CLI-first theo Phần E.

### Thứ tự đề xuất (red-team B2: thứ tự là luật cơ học, không phải prose — T3 PR bắt buộc `features` negotiation + `bridge_version` bump; `status/topology --json` expose `features_negotiated`, fail-closed; thêm `.github/workflows/tiers.yml` T0→T1→T2→T3; skew test 2 chiều fail-closed là merge gate)
1. **T0 requirements-split + markers (unblock Linux CI)** — DONE 2026-09-11: `requirements-base/-windows/-tools/-dev.txt` + markers `pywin32`/`torch-directml` + `install.bat`/`install.sh` + harness `backend/tests/benchmark_live_cluster.py` (loopback-refuse exit 3 verified, `test_benchmark_exists` 5 passed). Chưa làm: handshake versioned (T0.3) — việc code tiếp theo.
2. **T0 handshake versioned (T0.3) trước mọi wire-type mới** — DONE 2026-09-11: envelope `synapse_version` 0.1.0/`bridge_version`/`caps_hash`/`features` + `is_compatible` fail-closed + `PACKET_MIN_VERSION` registry + 0x06-0x08 reserved; `test_handshake_compat` 4 passed (2 chiều, legacy, mismatch-999, unknown-type). Old peers ignore keys mới (.get). `caps_hash` plumbing từ node + node-layer pipeline refusal còn lại (ghi rõ trong 08-red-team).
3. **E CLI-first**: DONE 2026-09-11 khung lệnh: `serve|join|client|status|topology|models|benchmark [--json] [--port N]` + `--version` + exit 0/2/3/4 + secrets redacted; `test_cli` 9 passed; cả 2 thứ tự `--port` đều đúng. Chưa: `--log-format json` backend + `GET /` stub (cần chatgpt_api, file bạn đang sửa).
4. **C1 live gate (amended R3+G1)**: khai `tflops_proportional` effective (flag `lrm` là no-op dưới default); bảng ports từng port; shared-nothing assert; loopback → exit 3.
5. **D2-transport + C4 + C2**: DONE 2026-09-11: exact-dtype-default (B4, `should_quantize`, test) + chunked-TCP 1MiB/`!IIBI`/offer-less direct (receiver reassembles; size negotiation = admission cap) + `choose_transport` + receiver wiring vào `process_tensor` + bridge NODELAY 2 chiều; `test_tcp_chunked_transfer` 9 passed (130MiB blob, live 12MiB, refusal, purge, router, NODELAY). Chunked/SHM sau env `SYNAPSE_CHUNKED_TCP=1`, mặc định bridge cũ. Manifest + dual-read + P2P pull metadata (T1.1): `test_ckpt_p2p_pull` 6 passed (separate tmpdirs, stale-step refuse, too_large explicit, engine honesty).
6. **D1-KV + C3**: MECHANISM DONE 2026-09-11: pool + session + rollback (test 6+6 passed) + **parity chứng minh trên model thật** (`test_seq_kv_parity`, tiny-GPT2 CPU, 8 bước greedy token-identical, fast path engaged 7/8 forwards) sau `SYNAPSE_SEQ_KV=1` (mặc định tắt). Perf multiples vẫn cấm quote tới khi đo GPU.
7. **D2-router + C7**: DONE 2026-09-11 advisory: probe song song + EWMA + variance gate (17 tests bandwidth/resource xanh) + per-peer ping (bỏ fan-out) + symmetric bw attach + `_infer_total_layers` (bỏ hardcoded-32 failover) + scale matrix test (8 passed, số đo: ILP 2x32/8x32 budget-2s fallback, 4x32 Optimal) + capability/demand gossip publish/subscribe/store + autoscale evaluator (test 6 passed) + node hook advisory (`SYNAPSE_AUTOSCALE=off` tắt; `advise`/`on` = hints, actuating gated C1). Chưa: actuating (sau C1), multi-size sweep.
8. **D1-kernel + C6**: PARTIAL 2026-09-11: rung-2 observability (`_choose_quant` pure + test 4 passed, missing-bnb loud warn, quant_wanted/active stamps) + base `load_checkpoint -> bool` contract + attention probe (`attn_backend.py`, test 4 passed incl. anti-overclaim gate) + TLS HOÀN CHỈNH loopback (`tls_wrap.py` + gen cert tự ký + bridge ssl params + Libp2pNode forward + handshake `tls_fp` + `test_tls_interop` pass: handshake + event + SSLObject 2 chiều + pin) + web stub (`SYNAPSE_HEADLESS=1` → 204, test) + `logging_json` chuẩn + CLI dùng chung (test 5 passed) + standby registry trong demand summary. Chưa (cần GPU + model + C1 xanh): torchao FP8, fused attention/compile, MoE banks, TLS bật thật trên multi-host, autoscale actuating.
- **Full suite 2026-09-11: 329 passed** (bao gồm mọi test mới). Nếu C1 đỏ: perf SLIP, safety KHÔNG slip, product phụ CUT; giữ README:168 + timing-model labels; cấm chữ "production-ready".

---

## Phần E — Chốt hướng CLI-first (không dashboard nặng)

Quyết định: synapse xây theo kiểu **CLI terminal**, giữ web UI hiện tại ở mức tối thiểu.

- **Giữ nguyên:** `GET /` trang tĩnh do backend serve (`README:169`, không npm) — chỉ để check sống/chết, không đầu tư thêm.
- **CLI chuẩn (học mesh-llm `serve/client/status`, FreeToken `ft`):**
  - `synapse serve [--model ...] [--port ...] [--join <token|config>]` — chạy node foreground, TUI text tối thiểu.
  - `synapse console [serve-flags]` — VỎ TƯƠNG TÁC kiểu Claude Code (KHÔNG agent): boot node nền + stream log `[node]` + REPL `/slash` (`/status /topology /models /benchmark /client /help /quit`); chữ lạ không slash thì không làm gì; Ctrl+C dừng node exit 0; lịch sử phím lên/xuống. Mọi dòng in ra đều là kết quả lệnh thật hoặc log thật (`backend/synapse/console.py`, `test_console.py` 7 passed).
  - Zero-flag boot + đổi sau trong console: cờ nào cũng có mặc định (`~/.synapse/cli-defaults.json`, đổi qua `SYNAPSE_DEFAULTS_PATH`); `python main.py console` trần boot model mặc định; trong console gõ `/set <key> <value>` (validate + lưu), `/config` (xem giá trị + nguồn cli/file/builtin), `/unset <key>`; đổi chỉ áp dụng lần boot sau, node đang chạy giữ nguyên (`backend/synapse/cli_defaults.py`, test 7 passed).
  - `synapse join <token|config>` / `synapse client --auto` — vào mesh có sẵn, không serve model.
  - `synapse status [--json]` — node, peers, partition, shard range, checkpoint step.
  - `synapse topology [--json]` — latency matrix, resource scores, partition hiện tại.
  - `synapse models` / `synapse benchmark` — inventory + đo token/s, time-to-first-token, failover s cho C1.
  - `--log-format json` cho mọi lệnh để script/CI parse; `--headless` chỉ tắt web UI, không tắt log.
- **Không làm:** dashboard React nặng, mobile app, MoA gateway, plugin store, image/audio generation — vượt scope pipeline-parallel + training của synapse.
- **Khi nào mới làm dashboard:** sau khi C1 có số đo live 2-3 máy thật và có người dùng ngoài dev yêu cầu; lúc đó cũng chỉ bọc mỏng quanh `status`/`events` API, không logic riêng.

---

## Quy tắc làm việc (giữ nguyên + bổ sung)

- **Mỗi mục C/D/E = 1 commit nhỏ** + 1 test behavioral (dẫn shipped function thật, không soi nguồn; cấm source-string assertion tests).
- **Sau mỗi mục:** chạy test ảnh hưởng; **sau Phần C:** full suite + launch 2 lần + (với C1) live run 2-3 máy.
- **C1 là gate cuối:** chỉ coi là production-ready sau live run có số đo (kể cả run FAIL nhưng trung thực — số fail vẫn hơn không số).
- Cập nhật README ngay khi đổi hành vi (giữ nguyên tắc "README = sự thật"); pin commit hash trong test artifacts; re-pin line-refs mỗi lần sửa (refs drift nhanh).

---

## Audit 2026-09-11 (3 lanes + sửa, full suite 334 passed)

**Bật mặc định (sau suite xanh):** `SYNAPSE_SEQ_KV=1` (parity tiny-GPT2), `SYNAPSE_CHUNKED_TCP=1` (fallback bridge tự động), `--headless`/`--log-format` backend + stdout JSONL redirect, TLS wiring (env `SYNAPSE_TLS` + cert paths, fail LOUD), `--node-host` bind thật, `discovery_timeout` forward, Tailscale env chuẩn 1 đầu mối, eval/train nối vào hàm thật (xóa 4 flags chết), route `GET /v1/download/progress`, settings round-trip thêm system_prompt, `/v1/distributed/status` phơi degraded/standby/hints/transport/quant/versions, ACK + example/status subscribers (hết black-hole), `--no-frontend` → headless + deprecation.
**Giữ tắt (có lý do ghi rõ):** INT8 mặc định (lossy, cần gate chất lượng), autoscale actuating (cần C1), TLS on (cần certs + hosts).
**Xóa:** `models/` OOP (~655 dòng), `networking/grpc/` + `server.py`, `LocalModelLoader`, `parse_device_attributes`, `save/clear_session`, pyc stale, `grpcio` dep; `requirements-windows.txt` thành pointer; thêm `training/__init__.py`.
**Gộp:** QUANT_BPP 1 nguồn (helpers), `_get_all_local_ips` 1 nguồn.
**Lane sai đã hiệu đính:** `simple_dataset` targets==inputs là quy ước causal-LM (ghi chú, không sửa); `--no-disable-tui` tồn tại qua BooleanOptionalAction (`--tui` không hỏng); `get_devices_for_ui` giữ lại (UI bạn đang làm); test trùng/gộp test + merge partitioner/dataset + frontend scaffold HOÃN (đụng việc bạn + cần quyết sản phẩm).
**Env máy này (research/plan-followup/10):** torch_directml vỡ DLL (import sau torch = fatal process); model thật chạy CPU được; ILP 2x32/8x32 budget-2s fallback.

## Audit lanes A/B/C đợt 2 (2026-09-11, commit 445d4fb, suite 334 passed)

**Đã nối (có test `test_p2p_wiring` 5 passed):** ACK cho `send_prompt`/`send_tensor` (hết timeout oan 5s); subscribers `example`/`status` + handler RPC `example` (hết black-hole, loss thật thay 0.0 giả); eval/train chạy hàm thật; `--node-host` bind thật + **preflight cổng fail-nhanh** (kẹt cổng in đúng PID + cách gỡ netstat/taskkill, exit 2 — hết traceback 10048 mù; test `test_port_preflight`, đã chứng minh end-to-end với cổng chiếm thật).
**Đã sửa:** mock capabilities phải có `SYNAPSE_TEST_MODE=1` (+ test_cluster script đã cập nhật); autoscale crash thì log (không còn `[]` câm); train/evaluate base warn 1 lần; power heuristic gắn cờ `power_estimated`; multi_gpu không đăng ký key None; QUANT + IP 1 nguồn (đã liệt kê); settings thêm system_prompt; route download progress; status phơi observability.
**Đã xóa:** liệt kê ở trên (verify zero-importer trước khi xóa).
**Hoãn có lý do:** gộp test/partitioner/dataset/frontend (đụng việc bạn); autoscale actuating + INT8 default + TLS on (cần C1/GPU/certs); đo multi-size probe (cần mạng thật).
**Tailscale không CLI (log spam):** `start()` probe 1 lần, thiếu CLI thì bỏ hẳn vòng posture + in 1 dòng hướng sang udp/manual; `stop()` chịu None (`test_tailscale_nocli` 2 passed).
