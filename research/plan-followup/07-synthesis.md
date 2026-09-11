# Synthesis — 6-Lane Unification (synapse_ai deep-dive)

(Full synthesis output from synthesis agent — see conversation record task ses_f71326becffeBVC0A6zZRYbSCL for complete text.)

## Key resolutions (re-verified in source)
- **R1 Install scope**: Lane 1 OUT = sibling ownership, not a ban. `install.bat/sh` + requirements split + run.bat fix + semver = Tier 0 in. OCI/Fly/Homebrew matrix = out.
- **R2 INT8 ownership**: two independent domains — wire `_q_scale` (transport, `codec.py:69-76,114-116`, `node.py:1335-1351`) vs weight bnb (engine, `pytorch_hf_engine.py:190-207,453-464`). Unified rule: offer carries both `quantize_requested` + `weight_quant`; hash token-ids only; exact-dtype default for chunked TCP.
- **R3 Default-path divergence (LOAD-BEARING)**: production default `tflops_proportional=True` (`node.py:46-47,125,1672-1674`) takes TFLOPS-proportional path — Lane 5 runbook `--partition-method lrm` is a NO-OP under defaults. Runbook amended; `status/topology --json` must expose `*_requested` vs `*_effective` + `tflops_proportional_effective`.
- **R4 Handshake**: Lane 6 envelope canonical (`synapse_version` 0.1.0, `bridge_version:1`, `caps_hash`, `features`); Lane 2 fp folds in as `tls_fp`; major mismatch → single-node fallback, never silent mis-partition.
- **R5 GET / truth**: single allow-list policy (200 HTML <10KB, name+version+liveness+3 links; `--headless` → stub; API/logs/healthcheck untouched).
- **R6 Checkpoint dual-read**: memory → P2P → shared-dir; tripwire = delete shared-dir ONLY after live C1 P2P demo. Path-prefix correction: HF no-symlink + Win loop live in `backend/synapse/main.py:38-47`, not root main.py.

## Missed gaps (all six missed)
- **G1 [High, C1 blocker]**: triple-port matrix (50051/5678/52415) undocumented per-port reachability + `--node-host` dead flag → silent non-formation. Gate: runbook port table + `status --json` bind/advertised addrs.
- **G2 [Med-High]**: Windows loop policy unpinned + deprecated `get_event_loop` (`chatgpt_api.py:454,996`) + restart-on-42 (`main.py:81-85`) orphan risk. Gate: pin policy, replace lookups, shutdown test.
- **G3 [High, security]**: Tailscale key via CLI argv → `ps`/history/`Popen` leak; no redaction proof. Gate: env/file-only secrets + grep-clean test.
- **G4 [Medium]**: partitioned slice never re-fit-checked (single-node `check_model_hardware_fit` only; KV blind spot). Gate: P3 gossip soft-gate ×1.1 before autoscale; C1 records free-VRAM + slice.
- **G5 [Medium]**: restart-on-42 vs step monotonicity (`_checkpoint_steps`, staged checkpoints) — regress risk for manifest ordering. Gate: orchestrator-issued step authority + restart test.

## Unified Tier 0-3 (order enforced: requirements → handshake → ckpt-pull → framing → CLI → live → KV/router/quant behind flags)
- **T0.1** requirements split + markers (pywin32/directml/bnb lazy, dedup rich, relax urllib3, tools/dev extras, py≥3.10).
- **T0.2** run.bat fix + install.bat/sh + semver 0.1.0 + `--version`; `GET /` <10KB gate.
- **T0.3** versioned handshake + additive-only policy + skew test both directions (merge gate).
- **T1.1** ckpt P2P pull + manifest + dual-read (BEFORE framing); no pickle; step monotonicity; tripwire kept.
- **T1.2** chunked TCP (`!IIBI`+1MiB, offer/accept, reassembler, 4GiB admission) + NODELAY one-liner + `choose_transport` auto-route + peer eviction; exact-dtype default; explicit bf16/f64 reject.
- **T2.1** CLI skeleton (subcommands, `--json` single-doc, exits 0/2/3/4, `--headless` narrow, schemas verbatim + effective/requested pairs, eval/train wired-or-removed, secrets redacted).
- **T2.2** benchmark contract + JSONL events + endpoint deltas (latency N×N, master/partitions[], ckpt step); no Prometheus/React/mobile/MoA.
- **T2.3** live C1 run (amended R3+G1): shared-nothing, NTP, pre-pulled ckpt; convergence = identical shards 3×; M1-M6 with thresholds (formation ≤60/120s, TTFT ≤2×, tok/s parity, failover ≤30s both kill shapes, LoRA no-hang, temp-0 5/5); artifacts per host; README:168 + timing-model labels stay until green.
- **T3.1** paged KV + delta + rollback → chunked prefill (gates: ≥4×@2k, ≥8×@8k, ≥10× bytes, 80% prefix, 200-prompt parity).
- **T3.2** probe hardening (parallel multi-size + EWMA + symmetric) + n×L scale table + fix hardcoded-32 + default budget 5s.
- **T3.3** capability/demand gossip advisory-first + master autoscale (mesh-llm trigger values; actuating only post-C1).
- **T3.4** quant ladder last (FP8-KV → bnb harden → torchao → fused/compile; measured budget replaces `p×2.0`; refuse train-on-quant).

## Follow-up verification (verify before building on)
1. ILP 68k-row math (construction vs solve split; budget only bounds CBC).
2. 1000-2500× analytic recompute (instrument before quoting savings).
3. 12×32 max + "no 32 bound" (tracked-grep + scale-table proof).
4. SHM zero-caller vs loopback `shm_sends==1` (prod path vs test fake).
5. `CHECKPOINT_INTERVAL` dead vs A1-working (two checkpoint families distinct — don't delete blindly).
6. Absence claims (no .github, no pygame/cv2) via tracked `git grep`.

## Confidence: 6.5/10
~150 file:line claims re-sampled, ~25 load-bearing lines re-verified; 0 multi-host measurements; 6 single-sourced items above. Raise: T0 skew + T1 pull + T2 CLI suite + any 2-machine M1-M6 bundle (even failed → ~8). Lower: second default-path divergence, wire-type before handshake, quoting simulated/analytic multiples as measured (<6).
