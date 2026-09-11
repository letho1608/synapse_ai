# Lane 1 — CLI-First Design: Audit + Minimal Spec (synapse_ai)

## 1. Scope
CLI surface only. Transport/checkpoint (Lane 2), decode-KV/engine (Lane 3), partition-router-demand (Lane 4), live-E2E (Lane 5), scope-guard-packaging (Lane 6) are sibling lanes — this lane names only their CLI touchpoints.

Decision: PLAN.md Part E — CLI-first terminal UX, keep backend-served `GET /` at liveness-only fidelity (README:169), defer heavy dashboard until live C1 numbers + external user request.

## 2. Current CLI audit (file:line)
Two stacked parsers, no subcommands:
- Outer `main.py:60-66`: one flag `--no-frontend` (documented no-op), forwards rest to backend (`main.py:76-78`), GPU summary (`main.py:16-57`), restart-on-42 loop (`main.py:81-85`).
- Backend `backend/synapse/main.py:50-100`: ~30 flat flags + positionals `command (run|eval|train)` and `model_name`. Only `run` dispatched (`backend/synapse/main.py:420`); `eval`/`train` fall to idle (`440-445`) while `eval_model_cli` (`341-353`) / `train_model_cli` (`355-372`) unreachable.
- Dead/no-op flags: `--models-seed-dir`, `--download-quick-check`, `--max-parallel-downloads`, `--discovery-timeout`, `--node-host` (parsed, never consumed); training cluster `--iters/--save-every/--data/--batch-size/--resume-checkpoint/--save-checkpoint-dir` unreachable via dispatch though checkpoint recovery exists (`orchestration/node.py:554-575`).
- Confusing: triple model selector (`model_name` / `--run-model` / `--default-model`→fallback `qwen2.5:1.5b` at `:187`); `--disable-tui` default True (double-negative opt-in); `--prompt` default `""` erases one-shot-vs-serve; stale comment `node_port` 50051 vs "defaulted to 5678" (`:109` vs `:62`).
- Keep: `--node-id` (MAC+port hash, `helpers.py:411-425`), `--node-port`, `--discovery-module` (+config shadowing `settings.json :131-142`), `--wait-for-peers` (`:399`), `--chatgpt-api-port` 52415 (rename alias `--port`), `--partition-method`, `--ilp-time-limit-seconds`, `--temperature` (rename from `--default-temp`).
- `run.bat:6-16` stale Vite/5173 claims vs backend-served reality (`api/chatgpt_api.py:536-554`); `frontend/` Vite scaffold vs static-dir vs bundled fallback — three UI truths.
- Works today: deterministic node IDs, one-shot `run_model_cli` (`264-316`, 300s cap `:307`), API always starts (`:402 if True:` — constrains `--headless`), existing GET endpoints (`/v1/topology :1421`, `/v1/distributed/status :1451`, `/v1/system/info :1650`, `/v1/cluster/resources :2853`, `/healthcheck :625`, `/v1/models :696`).

## 3. Proposed CLI spec
```
synapse [--log-format text|json] [--headless] [--port N] <command> [args]
synapse serve [--model ID] [--node-id ID] [--node-port P] [--discovery-module udp|tailscale|manual]
              [--config PATH] [--wait-for-peers N] [--partition-method lrm|ilp]
              [--ilp-time-limit-seconds S] [--temperature T] [--tui|--no-tui] [--chat-port P]
synapse join <token|config> [--node-id ID] [--node-port P]
synapse client [--model ID] [--prompt TEXT | --interactive] [--temperature T] [--timeout S] [--port P]
synapse status [--json] [--port P]
synapse topology [--json] [--port P]
synapse models [--json] [--port P] [list|pull|rm ARGS...]
synapse benchmark [--json] [--model ID] [--prompt TEXT] [--max-tokens N] [--port P]
```
- `--log-format json`: JSONL logs on stdout for ALL commands; TUI to stderr/suppressed.
- `--json`: exactly one JSON document on stdout, nothing else; exit 0/2/3/4.
- `--headless`: disables TUI + reduces `GET /` to liveness stub (204/ok); NEVER disables API/logs/healthcheck/`/v1/*`. Successor to `--no-frontend` no-op (deprecation warning one release).
- `GET /` allow-list: 200 HTML <10KB, name+version+liveness+links to /healthcheck, /v1/distributed/status, /v1/topology. No npm, no fetch beyond 3 liveness endpoints.
- Status/Topology/Models/Benchmark/Completion JSON schemas reuse `Node.distributed_*`, `ClusterState`, `Topology.to_dict()`, `DeviceCapabilities.to_dict()`, `LACPPartition.method_used/solver_status`, `Shard.to_dict()` verbatim + `*_effective` vs `*_requested` config pairs + contiguous assignments. Unknown values = explicit null, never omitted. Never echo `--tailscale-api-key`.
- OUT: React/Vite dashboard, mobile, MoA, plugin store, media-gen endpoints, npm UI dep, dashboard logic fork, transport/KV/router/packaging design (sibling lanes).

## 4. Acceptance criteria (observable, script-checkable)
AC-serve/join/client/status/topology/models/benchmark/json-logging/headless/get-root/compat/eval-train-resolution per full lane output. Key gates: `topology --json` assignments cover [0,total-1] contiguously; `--ilp-time-limit-seconds 0.001` surfaces `method_used:lrm`; dead port → exit 3 naming port; `GET /` <10KB no vite imports; `eval`/`train` wired or removed from help.

## 5. Risks
- [Blocker] Flat parser hides dead commands/flags — user can "train" with no effect.
- [Blocker] Three competing `GET /` truths without allow-list → effort leaks to dashboard.
- [High] 5 silent no-op flags (bind/discovery tuning with zero feedback).
- [High] Config-file shadowing invisible → multi-machine debug guesswork. Mitigation: `*_effective` pairs mandatory.
- [High] TUI polarity + API-always-on constrain `--headless` (normative definition required).
- [Medium] Triple model selector; stale run.bat; `--prompt ""` sentinel.
- [Low] Port comment drift; Vietnamese-only default system prompt surprise.
- [Note] mesh-llm/FreeToken borrowing is naming-only. Checkpoint flags depend on Lane 2.

## Executive summary
Stacked parsers, ~30 flat flags, 7 dead/no-op items, triple model selector, inverted booleans, stale npm artifacts. Proposal: `serve/join/client/status/topology/models/benchmark` + global `--log-format json`, per-command `--json`, narrow `--headless`; schemas reuse server structures verbatim; `GET /` allow-listed; React/mobile/MoA/plugins/media out per Part E.

**Confidence: 7/10. Ground-truth tally: 15 of 16 load-bearing conclusions externally verified (file reads).**
