# PLAN.md — Việc còn lại với synapse_ai

**Nguồn:** deep-dive phase 4 (ngày 2026-07-31) + đối chiếu code ngày 2026-09-11.
**Trạng thái:** Phần A/B cũ đã xong hết (xem bảng đối chiếu). File này giữ lại lịch sử + liệt kê gap production còn lại (Phần C), hướng lấy từ FreeToken / mesh-llm (Phần D), và chốt hướng **CLI-first, không dashboard nặng (Phần E)**.

---

## Phần A — Đối chiếu (từng là "việc chưa làm", nay DONE)

| # | Việc | Hiện trạng 2026-09-11 | Commit sửa |
|---|------|----------------------|------------|
| A1 | Resume-from-checkpoint: survivor nạp lại checkpoint | DONE 2026-09-29 — training ghi `*.weights.pt` (chỉ layer của shard) mỗi `SYNAPSE_CKPT_WEIGHTS_EVERY` bước (mặc định 10), JSON ghi `weights_path/weights_step`; rebalance gọi `load_checkpoint(weights_path)` → copy đúng dải layer của node chết vào model survivor (test hash layer khớp, layer ngoài dải không đổi, thiếu layer → từ chối). Còn thiếu: file weights nằm trên máy node chết → khác máy cần shared dir hoặc replicate chủ động sang peer (chưa làm; không có → degraded) | `120f856` + `08-red-team B1` + đợt 2026-09-29 |
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

## Đợt 2026-09-29 (suite 428 passed + test mới)

**Sửa lỗi thật (mỗi mục có test behavioral):**
- Checkpoint **không bao giờ ghi được với model mặc định**: model id `HuggingFaceTB/SmolLM2-135M-Instruct` có `/` → tên file trỏ vào thư mục con không tồn tại. `_safe_name` giờ thay `/` và `:` (NTFS ADS) bằng `_`, dùng chung cho save/load/pull; `load_latest_checkpoint` không còn nhận nhầm file của node `a_b` khi hỏi node `a` (`test_shard_weights_restore.py`).
- A1 thật: khôi phục weights dải layer của node chết (xem bảng A).
- Launcher bị kill cứng → backend mồ côi giữ cổng (nguyên nhân `test_system::test_api_full` flaky). `SYNAPSE_PARENT_PID` + watchdog khởi động trước import nặng, chống PID tái dùng (`test_parent_watchdog.py`, kill cứng end-to-end).
- `SYNAPSE_TLS` chỉ bọc bridge; đường tensor chunked-TCP (mặc định bật) vẫn plaintext. `TensorTCPTransport` giờ dùng chung ctx TLS với bridge; client plaintext không chèn được frame (`test_tcp_tls.py`).
- `synapse.model_discovery` không import được (`List/Dict/Any` thiếu) → `/v1/models/search` luôn 500, pull repo HF ngoài danh sách luôn lỗi; thêm timeout + auth header cho search (`test_api_endpoints.py`).
- `_pretty` không định nghĩa trong `update_peers` (NameError khi DEBUG>=3); xoá handler `handle_get_download_progress` trùng (bản chết).
- CLI: `main.py --help` chỉ hiện `--no-frontend`; `--port N console` lỗi "invalid choice". Giờ help hiện đủ lệnh (có `console`), cờ global map sang console (`test_cli.py`).
- README: sửa các khẳng định lỗi thời (TLS, checkpoint, chunked mặc định bật, KV mặc định bật, CLI/console, đường `pip install`, cây thư mục).

**Còn lại (cần phần cứng/mạng thật):** C1 live 2-3 máy; replicate weights chủ động sang peer (C2 — pull từ node đã chết là bất khả); TLS bật trên multi-host; autoscale actuating; perf GPU.

## Đợt 2026-09-29 (2): xóa frontend, bật INT8 mặc định

- **Xóa `frontend/`** (CLI-first): API không còn `mkdir frontend/` + `/_static/`; `GET /` = trang trạng thái tích hợp (204 khi `--headless`). README/launcher cập nhật.
- **Lỗi engine nghiêm trọng (tìm ra khi đo INT8):** CPU dùng `eager` + `attention_mask=None` → KHÔNG có causal mask (SmolLM2 trả lời "France is land"); transformers 5 đổi `past_key_value`→`past_key_values` (Cache) nên KV không bao giờ được ghi mà cache-hit vẫn cắt input; stage sau chỉ nhận hidden của token mới; GPT-2 thiếu `wpe`; CPU chạy bf16 ("auto"). Sửa: mask causal 4-D tường minh, `DynamicCache` theo (request, dải layer), stage không cuối trả hidden đủ độ dài, stage sau khớp prefix theo hàng byte-exact, bỏ cache theo hash nội dung (Cache mutate tại chỗ bị chia sẻ), CPU fp32. `test_engine_reference_parity.py`: token-identical với HF `model()`/`generate` ở 1/2/3 stage (tiny GPT-2 + SmolLM2). Test parity cũ so hai đường cùng hỏng nên không phát hiện.
- **INT8 codec:** per-tensor scale → per-row + 8 outlier/hàng giữ fp32, làm tròn thay vì cắt, tensor toàn 0 không còn trả về int8, metadata >1 KiB không còn bị cắt mất `request_id` (quá 64 KiB thì raise). Đo teacher-forced (SmolLM2, 6 prompt, 2 và 3 stage): per-tensor cũ top-1 1–6% / KL 8–11; per-row 95–96% / 7e-3; per-row+outlier 100% / 5e-5.
- **Bật INT8 mặc định** (`SYNAPSE_TRANSPORT_INT8=0` để tắt), chỉ tới peer đã negotiate feature `q8row1` (peer cũ nhận exact dtype; tra theo node gửi). Gate: `test_int8_quality.py` (top-1 ≥ 0.98, min ≥ 0.95, KL ≤ 1e-3).
- **Chat qua API trả lời rác (kiểm chứng end-to-end bằng server thật):**
  - Số layer đoán bừa 28 cho mọi model không có trong `HF_MODEL_LAYERS` (~480/500+, gồm model mặc định SmolLM2 = 30) → shard "cuối" dừng sớm, lm_head chạy trên hidden giữa model. Giờ đọc `num_hidden_layers` từ `config.json` (`models.layers_from_config`) và engine từ chối shard có `n_layers` sai.
  - Engine phục vụ đủ 2 kiểu gọi: full-context (khớp prefix) và delta (chỉ token mới + `kv_len` trong `inference_state`), phân loại không mơ hồ (nối dài chính xác = full-context; ids int32/int64 so theo giá trị).
  - Handler bỏ qua system prompt đã cấu hình và chèn cứng "Trả lời bằng tiếng Việt"; `temperature` của request bị bỏ qua (node luôn dùng 0.7). Giờ dùng đúng system prompt cấu hình, temperature đi trong `inference_state`; bỏ default dict dùng chung `inference_state={}`.
  - Kết quả end-to-end (`python main.py`, POST `/v1/chat/completions`, temperature 0): "The capital of France is Paris." — trước đó "land" / "Din Son T shallow…". Test: `test_engine_reference_parity.py` (logits + generate khớp HF cho full-context, delta, và đúng convention của node).

## Đợt 2026-09-29 (3): kiểm chứng "đã thật sự phân tán chưa"

**Cách kiểm:** chạy 2 rồi 3 process backend thật (manual discovery, 127.0.0.1, cổng riêng), gửi 1 request chat, đọc log `forward layers a-b` của từng node.

**Trước khi sửa: CHƯA phân tán đúng.** Lần 1: trả lời rỗng sau 172 s; lần sau: "The capital capital capital" (105 s). Nguyên nhân (đều đã sửa, test trong `test_distributed_real_wiring.py`):
1. Node tự coi **chính nó** chết (heartbeat của mình không bao giờ quay về) → rebalance bỏ chính nó mỗi vài giây.
2. Node nhận **bỏ qua shard trong message**, tự tính lại shard theo view phân vùng cục bộ; 2 node ngang nhau có thứ tự hòa khác nhau → **cả 2 cùng chạy layer 0–14**, hidden state bị đẩy qua lại ~170 lần. Sửa: request mang `partition_plan`; mỗi hop chạy đúng shard được gửi (`exact_shard`) và chuyển tiếp theo plan.
3. Broadcast topology luôn fail (pydantic `DeviceFlops` không JSON được); mọi lỗi handler bị `gather(return_exceptions=True)` nuốt im lặng → giờ log ra.
4. Stage cuối lấy mẫu với nhiệt độ mặc định 0.7 (temperature của request không tới node cuối) → giờ đi trong `inference_state`.
5. Controller bắn 3 micro-batch **y hệt** mỗi bước decode khi chưa có guess (3× compute) và engine chạy song song trên cùng KV cache → hỏng cache. Giờ 1 batch khi không guess + khóa forward trong engine.
6. Đường TCP chunked "mặc định bật" **chưa bao giờ chạy giữa 2 process**: peer đăng ký cổng *bridge* của nhau làm địa chỉ tensor; phía nhận gom mọi frame vào 1 bucket `"__inbound__"` chờ micro_batch 1,2,… mãi → mọi bước decode sau bước đầu bị treo. Giờ `tcp_port` quảng bá trong handshake, phía nhận giao ngay. SHM (chỉ trong 1 process) không còn bị chọn cho peer khác process.

**Sau khi sửa (INT8 bật, temperature 0):** 2 process — A chạy 0–14 ×8, B chạy 15–29 ×8, tensor qua `tcp-chunked`, "The capital of France is Paris." (giống 1 node), ~4–5 s. 3 process — 0–9 / 10–19 / 20–29, cùng đáp án, ~8.5 s. Test live: `test_live_multiprocess_split.py` (2 process thật, assert dải layer + đáp án + transport).

**Chưa kiểm được:** nhiều **máy** thật (C1) — mạng thật, băng thông, failover. Lưu ý Windows: dải cổng 50000–50059 có thể bị hệ thống giữ (Hyper-V) → cổng node mặc định cũ 50051 không bind được; nay đổi sang 46051 (dưới dải động 49152+).

## Đợt 2026-09-30: bổ sung theo tài liệu liên quan + rà soát tính năng

Làm theo `research/related-work.md`, các mục 1, 3, 4, 5, 6, 7.

**Mục 6 – nạp theo dải layer + lượng tử trọng số**
- Mỗi node trước đây nạp **toàn bộ** model.
- Giờ `partial_loader.py` dựng skeleton trên meta và chỉ nạp các layer của shard (+ embedding ở stage đầu, head ở stage cuối) từ safetensors.
- `weight_quant.py`:
  - int8: top-1 100%, KL 0,006.
  - int4 bất đối xứng: top-1 51% trên model 135M, nên chỉ dùng khi int8 không vừa.
- `encode()` chỉ nạp tokenizer.

**Mục 4 – kiểm tra draft trong một lượt**
- Mỗi bước decode = một lượt `[prompt + lịch sử + draft]`; stage cuối lấy mẫu ở mọi vị trí draft.
- KV được cắt về tiền tố chung.
- Giữa các stage chỉ gửi các dòng mới (`row_offset`); thiếu KV thì trả `kv_miss` và bước được gửi lại đầy đủ.
- Lỗi thật tìm thấy khi chạy live: mẫu chưa kiểm chứng lọt vào stream ("That in one Paris").

**Mục 3 – continuous batching**
- 4 request: 12,8 → 23,3 tok/s.
- 8 request: 12,3 → 35,0 tok/s.
- Token của từng request giống hệt khi chạy riêng.

**Mục 5 – phân chia theo số đo**
- `dp_partition.py` (quy hoạch động, đã đối chiếu vét cạn) + `layer_profile.py` (ms/layer thật hoặc probe, giới hạn bộ nhớ).
- Profile đi qua gossip; phân chia chỉ đổi khi rẻ hơn ≥10%.

**Mục 7 – failover giữa request**
- Theo dõi heartbeat trong lúc chờ.
- Vá kế hoạch (ưu tiên node đang giữ bản sao, rồi đến stage liền kề), gửi lại đúng bước đó.
- `SYNAPSE_REPLICAS=1` giữ sẵn layer của stage kế tiếp.
- Live: giết 1 trong 3 tiến trình khi đang stream, câu trả lời vẫn giống hệt.

**Mục 1 – README** viết lại phần mở đầu và kiến trúc: không còn "bubble-free / 333 tok/s"; có bảng số đo và biến môi trường.

**Lỗi sửa khi rà soát tính năng**
- Bầu master: hai node nhường nhau cùng lúc thì không còn master nào; nay tự hội tụ.
- Stream API mất khoảng trắng (giải mã từng token rồi strip): nay giải mã tăng dần.
- Stream giữ lại 20 token dù không có tool.
- Chunk cuối không có `usage`.
- Settings `load_quantization` / `lookahead_window` / `transport_quantization` / `lacp_priority` không có tác dụng; nay được áp dụng và lưu lại.
- `/v1/models` báo `ready: true` cho mọi model.
- `benchmark` dùng model "default" nên bị 400; `--prompt` bị bỏ qua; tok/s bị thổi phồng (~400) vì đo từ chunk đầu tiên.
- `settings.json` trong repo đè model mặc định (Qwen chưa có trong cache); `--default-model` nay thắng.
- Checkpoint FT ghi vào thư mục gốc cache HF; nay ở `~/.synapse/checkpoints`, test dùng thư mục tạm.

## Đợt 2026-11-02: tìm chỗ chậm + kernel, paged KV, int4, huấn luyện theo dải layer

**Chỗ chậm tìm được (py-spy trên server thật, SmolLM2-135M, CPU):**
- TTFT ~2 s mỗi request: API gọi `AutoTokenizer.from_pretrained` (kèm HTTPS lên Hub) ở **mỗi** chat request → cache tokenizer theo repo (`inference/tokenizers.py`).
- `_measure_disk_io` `time.sleep(0.5)` chạy trên event loop (election scoring định kỳ) → treo mọi request 0,5 s mỗi lần cache hết hạn → đo bằng hiệu số bộ đếm, không sleep.
- `/v1/system/stats`: `SystemSpecs.detect()` (PowerShell/CIM, 4–5 s) chạy trên event loop, 2 lần/request → chạy trong thread + cache dùng chung.
- Engine: ~nửa thời gian decode là Python trong module HF; DynamicCache nối lại toàn bộ K/V mỗi token.
- Int8 hiện có chậm hơn fp32 ~10× (giải lượng tử cả ma trận mỗi lần gọi); chuyển int8→fp32 của torch bản này chỉ ~1 G phần tử/s.
- Luồng OpenMP của torch quay chờ, giành lõi với kernel numba (cặp torch+numba chậm 3,5×) → `OMP_WAIT_POLICY=PASSIVE` mặc định (`synapse/__init__.py`).

**Đã làm:**
- `inference/fast_decoder.py`: decoder họ Llama (llama/mistral/qwen2/qwen3) gộp q|k|v và gate|up (module HF giữ view, không nhân đôi bộ nhớ), SDPA, **paged KV** (pool block 16 dùng chung, crop trả block, request xong trả block, pool tự nở); batch decode ghép hàng cho mọi matmul. Logits = đường HF; parity/batching/verify test giữ nguyên xanh.
- `inference/qkernels.py` (numba, tùy chọn): GEMV/GEMM int8 và int4 đọc thẳng trọng số nén; 4096×4096: fp32 4,4 ms, int8 2,0 ms, int4 4,2 ms (trước: int4 ~5× chậm hơn fp32). Không có numba → fallback torch (vẫn nhanh hơn bản cũ).
- int4: fit scale/zero theo MSE + bình phương tối thiểu; giữ v_proj/down_proj và 2 layer mỗi đầu ở int8 (kiểu Q4_K_M) → top-1 74%, KL 0,066, 0,19× fp32 (trước 51% / 0,63). Thử và loại: oneDNN dynamic int8 (lượng tử activation, top-1 49%), LLM.int8 outlier split (≤ 90%, op đã deprecated), `_weight_int8pack_mm` (chậm 4–12×).
- Huấn luyện: `inference/stage_training.py` + `pipeline/pp_training.py` — mỗi node train đúng dải layer nó phục vụ; activation đi xuôi, dL/dx đi ngược giữa các node (chunked TCP / bridge), clip theo norm toàn cục, AdamW từng stage, embedding tied giữa stage đầu/cuối trao đổi gradient. 1 bước 3 stage = 1 bước model đầy đủ (< 1e-5/tham số). Thay đường "1F1B" giả lập (mọi stage chạy trong 1 tiến trình với cả model).
- Cổng node mặc định 46051 (50051 nằm trong dải Windows giữ); test TLS dùng cổng do OS cấp.

**Còn lại / giới hạn:** C1 (chạy nhiều máy thật) vẫn chưa làm; LoRA vẫn nạp cả model; trọng số lượng tử không train được; int8 không nhanh hơn fp32 trên model rất nhỏ (ma trận 576 chiều, chi phí gọi kernel chiếm phần lớn) — lợi ích thật ở model lớn; GPT-2 train có dropout nên không bit-exact với eval.
