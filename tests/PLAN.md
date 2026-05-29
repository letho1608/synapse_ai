# Tailscale Distribution Fix Plan

## Summary

- Fix the Synapse Tailscale distribution check end to end: topology, LACP resources, chat inference, and training must all expose evidence of real multi-node execution.
- Do not make the integration check pass by weakening assertions. The test should pass only when at least two Synapse REST APIs are reachable and layer shards are assigned to at least two nodes.
- Avoid changing the core LACP partition algorithm first because `_compute_lacp_partition` has broad runtime impact. Prefer API evidence consistency, pipeline lifecycle, result routing, and training path fixes.

## Implementation Changes

- Make `/v1/distributed/status` and `/v1/cluster/resources` use the same partition snapshot.
- Keep `/v1/cluster/resources` responsive even when Tailscale UI enrichment is slow or unavailable.
- Surface diagnostic fields such as topology node count, LACP node count, and partition errors.
- Start the distributed pipeline extension before inference or training uses it.
- Route distributed inference result events through `ResultRouter`, with fallback to the legacy token callback path when no distributed session owns the result.
- Fix training to use `node.distributed_pipeline` instead of the non-existent `_pipeline_extension` attribute.
- Keep the integration test strict: post-chat and post-training evidence must still show multi-node LACP rows.

## Test Plan

- Run syntax checks:
  - `uv run python -m py_compile tests/test_tailscale_data_distribution.py synapse/api/chatgpt_api.py synapse/orchestration/node.py synapse/pipeline/node_extension.py synapse/inference/distributed_engine.py`
- If pytest is available, run focused API/routing tests for distributed status, cluster resources, result routing, and training extension selection.
- For live verification, start at least two nodes with `--node-id <tailscale-fqdn>` and REST port `52415`, then run:
  - `uv run python tests/test_tailscale_data_distribution.py`
  - `SYNAPSE_TEST_RUN_TRAINING=1 uv run python tests/test_tailscale_data_distribution.py`

## Assumptions

- Node ids should match Tailscale FQDNs because no devices currently expose `custom:synapse_node_id`.
- A Tailscale device that refuses port `52415` is an operational failure, not proof of distribution.
- If topology has peers but partition count remains one, the system is not distributed yet.
