# Executive Briefing — synapse_ai PLAN follow-up deep-dive (2026-09-11)

## TL;DR
Exhaustive 6-lane deep-dive + synthesis + red-team xong. Kết luận trung thực:
- **PLAN cũ đã lỗi thời theo hướng tốt**: A2/A3/A4/A6/A7/B1-B7 đúng là DONE trong code (đối chiếu file:line + commit).
- **Nhưng red-team lật lại 1 mục**: **A1 phải REOPEN** — `load_checkpoint()` là `pass` (`pytorch_hf_engine.py:586-587`), survivor chỉ khôi phục metadata chứ không khôi phục weights. Mọi demo failover hiện tại đều "thành công giả".
- **3 sự thật load-bearing khác**: default production chạy TFLOPS-proportional (flag `--partition-method lrm` trong runbook là no-op); mọi float tensor bị ép INT8 không opt-out (`node.py:1338-1351`); mọi bằng chứng phân tán đều in-process/loopback (chưa có số đo 2 máy thật).
- **Mức an toàn để thực thi: 5/10**. Lên ~8/10 sau: honest failed 2-machine M1–M6 bundle + T0 skew test + T1 pull test + T2 CLI-single-doc suite.
- Evidence package: `research/plan-followup/01-06` lanes, `07-synthesis`, `08-red-team`, file này `09-executive-briefing`.

## Synapse giỏi ở đâu (ghi nhận thật)
- Election + stickiness + demotion ~7-8s có test bounded — piece production-ready nhất.
- Scoring LACP 8 chiều + ILP budget + fallback observable (`method_used/solver_status`) + behavioral tests — khung tốt.
- Disk-I/O cache 5s, latency-refresh, authoritative broadcast, heartbeat riêng — các fix B3-B5 thật.
- Cross-platform detect (CUDA/DirectML/CPU, WMI/sysfs/rocm) + Windows-first venv — differentiator vs FreeToken (Linux+NVIDIA-only), cần bảo vệ.

## Vấn đề nghiêm trọng (theo severity, file:line)
**Blocker:**
1. **A1 hollow** — `pytorch_hf_engine.py:586-587` `pass` vs `node.py:554-575` gọi như thật; staging không check identity (`fault_tolerance.py:334-339`). → REOPEN A1; implement hoặc `raise NotImplementedError`; test assert weight-hash đổi, không chỉ flag.
2. **Tier order không thực thi được** — không CI (`.github` absent, re-verified), không flag framework. → gates cơ học: T3 PR bắt buộc `features` + `bridge_version`; `status/topology --json` expose `features_negotiated`, fail-closed; `.github/workflows/tiers.yml` T0→T1→T2→T3; skew test 2 chiều.
3. **Gate C1 đậu trên loopback** — `--node-host` chết (0 consumers), config default 127.0.0.1, `forward_tensor` không qua TCP/SHM router, README:168 in-process-only. → harness: IP non-loopback phân biệt + MAC, shared-nothing assert, NTP<100ms, artifacts/host; loopback → exit 3 `loopback-masquerade`; wire-or-delete `--node-host`; log `transport_selected`/forward (T1.2 cần `tcp-chunked>0` cross-machine).
4. **Ép INT8 toàn bộ float** (`node.py:1337-1351`) mâu thuẫn rule "exact-dtype default". → `quantize_requested` default False; fp32 bit-exact test; INT8 chỉ negotiated + error bound; reject bf16/f64 rõ ràng; hash token-ids only.
**High:** `get_event_loop` ~15 sites (đầu `event_router.py:44` — nhiễm mọi EWMA latency); Tailscale key qua argv (env/file-only + `test_secrets_redacted`); restart-on-42 mồ côi P2P/step (terminate+reap + monotonic test; wire-or-delete `CHECKPOINT_INTERVAL`); divergence election-composite vs TFLOPS-slice + hardcoded-32 failover (thread `n_layers`; expose effective path); security pre-C1 (traversal confine, sha-before-apply, per-peer cap, trusted-LAN label + Tailscale status).
**Medium:** ILP/scale single-sourced (log construction vs solve ms; stamp phân biệt; bảng n×L); multiples analytic (`measured:false`); fit-check đơn-node không cover slice (C1 ghi free-VRAM + slice); stdout indiscipline (prints→stderr; single-doc test); `GET /` allow-list chưa đo (byte test); "zero-copy" = in-process branch (counters tách 3 loại).
**Low:** requirements split chi tiết (markers pywin32/directml, dedup rich, extras; verify zero-pygame-imports bằng tracked grep trước khi xóa); stale texts (run.bat Vite, port comment, disable-tui, `""` sentinel, triple selector — fix cùng commit install; quyết `frontend/`); `0.0.1` bump nguyên tử với handshake+`--version`; giữ timing-model label + lưu ý prompt Việt.

## Nên làm gì (Tier 0/1/2/3 — thứ tự bắt buộc)
- **T0 (giờ–2 ngày)**: split requirements + markers; run.bat + install.bat/sh + semver 0.1.0 + `--version`; handshake versioned + additive-only + skew test 2 chiều (merge gate).
- **T1**: ckpt P2P pull + manifest + dual-read memory→P2P→shared-dir, tripwire giữ shared-dir (làm TRƯỚC framing); chunked TCP 1MiB/`!IIBI`/offer-accept/4GiB/30s + NODELAY + `choose_transport` + evict; exact-dtype default.
- **T2**: CLI skeleton (subcommands, `--json` single-doc, exit 0/2/3/4, `--headless` hẹp, schemas verbatim + requested/effective, eval/train wired-or-removed, secrets redacted); benchmark contract + JSONL events + endpoint deltas; live C1 amended (R3+G1) với M1–M6 + artifacts.
- **T3 (chỉ sau C1 xanh, sau flags)**: paged KV + delta + rollback → chunked prefill (gates ≥4×@2k, ≥8×@8k, parity 200 prompts); probe EWMA + bảng scale + fix hardcoded-32 + budget 5s; gossip capability/demand advisory-first + autoscale; quant ladder cuối (FP8-KV → bnb harden → torchao → fused/compile; budget đo thay `p×2.0`; cấm train-on-quant).
- Quy tắc nếu C1 đỏ: perf SLIP, safety KHÔNG slip, product phụ CUT; giữ README:168 + timing labels; cấm chữ "production-ready".

## Có nên tiếp tục? (trả lời thẳng)
**Có — nhưng chỉ Tier 0+T1+T2 trước, và việc đầu tiên là REOPEN A1 + honest 2-machine run (kể cả fail).** Synapse có khung LACP/election/FT tốt hơn vẻ ngoài PLAN cũ mô tả, và các borrow FreeToken (KV/prefill/quant) + mesh-llm (handshake/gossip/status) đều khớp đúng chỗ đau. Nhưng hôm nay chưa có gì chứng minh cross-machine: mọi số đều loopback/formula. Đừng xây T3 trước khi có số thật — đó chính là cách tiền mất mà cluster vẫn cold-start sau failover.

## Confidence: 5/10 safe-to-execute (6.5/10 analysis)
**Tally: ~30 claims red-team tự re-verify + ~150 lane claims file-anchored; 0 multi-host measurements; 6 single-sourced (ILP rows, multiples, 12×32 max, SHM callers, INTERVAL dead, absence claims); giả định sai đã bắt: `lrm` flag no-op, exact-dtype chưa tồn tại, handshake/CLI/manifest là đề xuất, 2nd host không có.** Tăng: bundle M1–M6 thật (fail cũng được) + 3 suite T0/T1/T2 → ~8. Giảm: divergence default-path thứ hai, wire-type trước handshake, quote số simulated như measured (<6).
