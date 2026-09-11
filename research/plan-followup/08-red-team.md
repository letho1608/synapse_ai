# Red-Team Review — Synthesis `07-synthesis.md` (synapse_ai)

(Full red-team output — see conversation record task ses_f71304777ffeseWW9aO3bpM226 for complete text.)

## Verdict: directionally correct, NOT safe-to-execute as written — 5/10

## BLOCKERs
- **B1. PLAN.md `A1 DONE` is FALSE** — `pytorch_hf_engine.py:586-587 load_checkpoint()` is literally `pass` while `save_checkpoint` persists (`:589-596`); `node.py:554-575` calls it pretending recovery works; staging (`fault_tolerance.py:334-339`) has no identity check. Any C1 failover demo recovers metadata, not weights, then reports success. FIX: `A1 → REOPENED`; implement real load or `raise NotImplementedError`; recovery test must assert weight-hash change, not just `recovered_from_checkpoint==True`.
- **B2. Tier ordering unenforceable** — no `.github` CI (re-verified absent), no flags framework. "Order enforced" is prose. T3-before-T0 corrupts (KV over forced-INT8 cements lossy error; autoscale on lower-bound probe flaps; 0x06-0x08 before handshake breaks old peers). FIX: mechanical gates — T3 PRs carry `features` negotiation + `bridge_version` bump; `status/topology --json` exposes `features_negotiated`, fail closed; add `.github/workflows/tiers.yml` T0→T1→T2→T3 dependency; skew test both directions fails closed.
- **B3. C1/M1–M6 pass on loopback claiming multi-machine** — `--node-host` dead (zero consumers; binds always 0.0.0.0), `make_config` defaults 127.0.0.1, `forward_tensor` never calls TCP/SHM router (publishes `{"bin"}` to bridge), README:168 in-process-only, README:121 overclaims TCP path. Trivial passes: identical shards 3× single-host; `shm_sends==1` in-process; contiguity single-node; failover killing non-participant; temp-0 on random CPU weights. FIX: harness requires distinct non-loopback IPs/MACs, shared-nothing checkpoint dirs (asserted), NTP<100ms, per-host artifacts; loopback → exit 3 `loopback-masquerade`; wire-or-delete `--node-host` + expose bind/advertised addrs; per-forward `transport_selected` observability (T1.2 accept needs `tcp-chunked>0` cross-machine).
- **B4. Forced INT8 contradicts "exact-dtype default"** — `node.py:1337-1351` quantizes EVERY float tensor, no opt-out; `codec.py:69-76,114-116` ephemeral scale. Rule doesn't exist in code; KV over it cements loss. FIX: `quantize_requested` default False threaded from offer/CLI; fp32 bit-exact default test; INT8 only negotiated with error bound; explicit bf16/f64 reject; hash token-ids only.

## HIGHs
- **H1. `get_event_loop` tail (~15 sites, not 2)**: `chatgpt_api.py:454,851,996,1415`, `event_router.py:44` (timestamp source — contaminates latency EWMA), `election.py:116`, p2p/udp/tokenizer/training files; `main.py:42-47` unpinned policy. FIX: pin policy per platform, `get_running_loop()` in coroutines, `test_loop_policy.py` grep gate. First: `event_router.py:44`.
- **H2. Tailscale key via argv leaks** (`main.py:91-92`, `root main.py:76-78 Popen`): ps/history/env-dump/config-copy artifacts. FIX: env/file-only (`--tailscale-key-file`, 0600 check), deprecate flag, `test_secrets_redacted.py` grep-clean (cmdline, status --json, JSONL, bundle).
- **H3. Restart-on-42 orphans state** (`main.py:81-85`, no handshake/flush/reap); `CHECKPOINT_INTERVAL` confirmed dead (1 grep hit) vs per-step JSON family — two families, no seq/fencing. FIX: terminate+reap before restart; `test_restart_step_monotonic.py`; wire-or-delete the constant with grep-gate.
- **H4. Election-vs-partition wider than R3**: composite election vs TFLOPS slices + never-refresh-on-hotplug + hardcoded-32 failover (`node.py:518,546`, `chatgpt_api.py:2864,3024`, `node_extension.py:119`, `partitioning_strategy.py:53`) → non-32L failover silently wrong. FIX: thread `n_layers` through `rebalance_after_failure`; expose `tflops_proportional_effective/master_weight_basis`; heterogeneous 3-node test.
- **H5. Security pre-C1**: ckpt path traversal (confine `Path.resolve().is_relative_to`), sha-without-signature (MITM), `--tls off` downgrade-by-default no sunset, HOL-DoS (single peer stalls/kills streams), 4GiB per-transfer no global cap. FIX: traversal+oversize tests, per-peer in-flight cap, trusted-LAN label + Tailscale status enforced by harness.

## MEDIUMs
- **M1.** ILP 68k/CBC/`timeLimit`/12×32 single-sourced; `pulp 3.3.0` present but CBC binary unchecked; timeout vs solver-failure same stamp. FIX: log construction vs solve ms; distinct `solver_status`; n×L table before "≤32 validated".
- **M2.** Decode multiples analytic (`measured:false` labels; prohibit quoting gates as results).
- **M3.** Single-node fit never re-checks slices; P3 ×1.1 advisory vapor until topics exist. FIX: C1 records free-VRAM + slice; harness schema enforces.
- **M4.** stdout discipline absent (GPU summary + prints → stdout) breaks `--json`/JSONL day one. FIX: prints→stderr, TUI→stderr, `test_cli_json_single_doc.py`.
- **M5.** `GET /` allow-list numbers unverified. FIX: `test_get_root_allowlist.py` (200, <10KB, no vite/:5173, exactly 3 links). Endpoints verified: `/v1/topology`, `/v1/distributed/status`, `/v1/cluster/resources` exist; `/v1/cluster/status|latency` absent (topic-only / missing).
- **M6.** "Zero-copy" = in-process branch (`node.py:1332-1333`), not SHM cross-process. FIX: counters `in_process_shortcircuit` vs `shm_sends` vs `tcp_chunks`; never compare SHM-loopback vs LAN-TCP.

## LOWs
- **L1.** requirements split verified (pywin32:27 kills Linux, directml:44, bnb:47, rich dup, over-pinned http trio, unpinned torch stack, best-pinned LACP stack); zero-pygame/cv2-imports NOT re-verified → tracked `git grep` before deleting.
- **L2.** stale texts verified (run.bat Vite, port comment, disable-tui default, `""` sentinel, triple selector). Fix with install commit; decide `frontend/` fate same commit.
- **L3.** `0.0.1` never-on-wire verified — bump only atomically with handshake + `--version`.
- **L4.** timing-model label + Vietnamese prompt kept; note prompt-language in runbook.

## PLAN.md exact changes (7 items: A1 REOPENED; C1 harness hard-reqs; C2 traversal+sha+cap; C4 exact-dtype-first + transport_selected; C7 stdout discipline + requested/effective + eval/train wired-or-removed; D mechanical gating + slip/cut rules; working-rules behavioral tests + pinned refs).

## Confidence: 5/10 safe-to-execute (synthesis 6.5/10 self-rating stands for analysis; execution needs B1–B4 + H1–H3 fixed). Re-rate to ~8 after: honest failed 2-machine M1–M6 bundle + T0 skew + T1 pull + T2 CLI-single-doc suite.
**Tally: ~30 claims personally re-verified (list in full text); ILP/probe/election-timing/KV-internals/eval-train-dispatch/GET-size/benchmark-taxonomy/FreeToken-mesh-llm cross-refs single-sourced; assumptions false/unproven: GPU/bnb/weights/2nd-host, CBC binary, `--partition-method lrm` meaningful (FALSE), exact-dtype default (FALSE), handshake/JSONL/router/manifest exist (FALSE — proposals).**
