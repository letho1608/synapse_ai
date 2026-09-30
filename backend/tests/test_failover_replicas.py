"""
Mid-request failover (Petals-style) and replica-aware re-planning.

- The orchestrator polls stage liveness while waiting for a step; a stage
  whose heartbeats stop is detected in seconds, the request is re-planned
  around it, the SAME step is resent with full hidden states (the new stage
  rebuilds its KV cache from the context), and generation continues with
  the exact tokens it would have produced.
- Re-planning patches only the dead stage's range, preferring a live node
  that already holds those layers (a replica), then a neighbour (ranges
  merge), within memory caps; other stages keep their caches.
"""
import numpy as np
import pytest

from synapse.inference.shard import Shard
import test_decode_pipeline_overlap as _decode  # tests/ is on sys.path
import test_measured_placement as _placement


class TestFailoverLoop:
    @pytest.mark.asyncio
    async def test_dead_stage_is_replaced_and_generation_continues(self):
        h = _decode.TestDecodeLoopBehavior()
        controller, router = h._make_controller()
        controller.step_timeout = 5.0
        controller.liveness_poll_s = 0.05
        dead = set()
        replans = []

        def alive(nid):
            return nid not in dead

        def replan(current, exclude):
            replans.append((list(current), set(exclude)))
            return [("node0", Shard("oracle-model", 0, 31, 32))]

        sent = []
        truth = h.TRUE

        async def send(target_node_id, target_shard, input_data, request_id_,
                       is_prompt=False, microbatch_id=0, extra_state=None):
            ctx = input_data.reshape(-1).tolist()
            st = dict(extra_state or {})
            sent.append({"ctx": ctx, "state": st, "plan": controller._plans.get(request_id_)})
            if len(sent) == 4:
                dead.add("node1")  # node1 dies while step 3 is in flight
            plan = controller._plans.get(request_id_) or []
            if any(p[0] in dead for p in plan):
                return  # the pipeline through node1 never answers
            n = st["verify_n"]
            toks = [h._true_next(ctx[:len(ctx) - n + 1 + j], truth) for j in range(n)]
            await router.publish(f"synapse/inference/result/{request_id_}", {
                "tokens": toks, "micro_batch_id": microbatch_id,
                "source_stage_id": plan[-1][1] if plan else 16})

        controller._send_input_to_stage = send
        await controller.start()
        try:
            tokens = await controller.process_distributed_prompt(
                base_shard=h._base_shard(), prompt="Hello", request_id="fo",
                partition_assignments=[("node0", Shard("oracle-model", 0, 15, 32)),
                                       ("node1", Shard("oracle-model", 16, 31, 32))],
                max_generate_tokens=len(truth), replan_fn=replan, alive_fn=alive)
        finally:
            await controller.stop()
        assert tokens == truth
        assert replans and replans[0][1] == {"node1"}
        stats = controller.decode_stats["fo"]
        assert stats["failovers"] == 1
        resend = sent[4]
        assert resend["ctx"] == sent[3]["ctx"]  # the same step, again
        assert resend["state"]["delta_rows"] is False  # full hidden states
        assert resend["plan"] == [["node0", 0, 31, 32]]

    @pytest.mark.asyncio
    async def test_without_replan_a_dead_stage_ends_with_the_verified_prefix(self):
        h = _decode.TestDecodeLoopBehavior()
        controller, router = h._make_controller()
        controller.step_timeout = 0.5
        tokens, sent = await h._run(controller, router, drop_steps={3})
        assert tokens == h.TRUE[:len(tokens)] and len(tokens) < len(h.TRUE)


class TestReplan:
    def _node(self, monkeypatch, ids=("A", "B", "C")):
        node = _placement.TestNodeWiring._node(monkeypatch, ids)
        node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}
        node._peer_capabilities = {nid: {"layer_profiles": {"m": {"ms_per_layer": 1.0, "max_layers": 100}}}
                                   for nid in ids if nid != ids[0]}
        return node

    def test_dead_range_goes_to_the_replica_holder(self, monkeypatch):
        node = self._node(monkeypatch)
        node._peer_capabilities["C"]["loaded_layers"] = {"m": list(range(10, 20))}
        current = [("A", Shard("m", 0, 9, 30)), ("B", Shard("m", 10, 19, 30)),
                   ("C", Shard("m", 20, 29, 30))]
        new = node._replan_excluding(Shard("m", 0, 29, 30), current, {"B"})
        assert [(n, s.start_layer, s.end_layer) for n, s in new] == [("A", 0, 9), ("C", 10, 29)]

    def test_without_replica_a_neighbour_absorbs_the_range(self, monkeypatch):
        node = self._node(monkeypatch, ids=("A", "B"))
        current = [("A", Shard("m", 0, 14, 30)), ("B", Shard("m", 15, 29, 30))]
        new = node._replan_excluding(Shard("m", 0, 29, 30), current, {"B"})
        assert [(n, s.start_layer, s.end_layer) for n, s in new] == [("A", 0, 29)]

    def test_memory_cap_forces_a_fresh_placement_or_none(self, monkeypatch):
        node = self._node(monkeypatch, ids=("A", "B"))
        node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 15}
        current = [("A", Shard("m", 0, 14, 30)), ("B", Shard("m", 15, 29, 30))]
        new = node._replan_excluding(Shard("m", 0, 29, 30), current, {"B"})
        assert not new or all(n != "B" for n, _ in new)

    def test_liveness_view(self, monkeypatch):
        import time
        node = self._node(monkeypatch)
        node._lazy_init_fault_tolerance()
        ft = node.fault_tolerance
        assert node._node_alive("B")  # unknown to the monitor = alive
        ft.record_heartbeat("B")
        assert node._node_alive("B")
        ft._health["B"].last_heartbeat = time.time() - 60
        assert not node._node_alive("B")
        assert node._node_alive(node.id)


class TestLayerResidency:
    """After a placement a node keeps only its own range (+ the successor's
    with SYNAPSE_REPLICAS=1) and frees the rest; before, every re-placement
    left the old layers in memory for good."""

    def _node(self, monkeypatch):
        node = _placement.TestNodeWiring._node(monkeypatch, ("A", "B"))
        node._local_layer_profile = lambda m: {"ms_per_layer": 1.0, "max_layers": 100}

        class Eng:
            _model_id = "m"

            def __init__(self):
                self.loaded = set(range(30))
                self.ensured = []

            def release_layers(self, keep):
                freed = sorted(self.loaded - set(keep))
                self.loaded &= set(keep)
                return freed

            async def ensure_shard(self, shard):
                self.ensured.append((shard.start_layer, shard.end_layer))
                self.loaded |= set(range(shard.start_layer, shard.end_layer + 1))
        node.inference_engine = Eng()
        return node

    @pytest.mark.asyncio
    async def test_keeps_only_own_range(self, monkeypatch):
        from synapse.topology.partitioning_strategy import Partition
        node = self._node(monkeypatch)
        await node._apply_layer_residency(Shard("m", 0, 29, 30),
                                          [Partition("A", 0.0, 0.5), Partition("B", 0.5, 1.0)])
        assert node.inference_engine.loaded == set(range(0, 15))

    @pytest.mark.asyncio
    async def test_replica_keeps_successor_range(self, monkeypatch):
        from synapse.topology.partitioning_strategy import Partition
        monkeypatch.setenv("SYNAPSE_REPLICAS", "1")
        node = self._node(monkeypatch)
        node.inference_engine.loaded = set(range(0, 15))
        await node._apply_layer_residency(Shard("m", 0, 29, 30),
                                          [Partition("A", 0.0, 0.5), Partition("B", 0.5, 1.0)])
        assert node.inference_engine.ensured == [(15, 29)]
        assert node.inference_engine.loaded == set(range(30))


class TestSingleStagePlans:
    """A plan with ONE stage (latency objective on a model that fits one
    node, or every other peer excluded) used to run base_shard = (0, 0, n):
    only layer 0. Local single stage now runs the full shard; a remote
    single stage goes through the controller."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("owner", ["A", "B"])
    async def test_single_stage_runs_the_whole_model(self, monkeypatch, owner):
        from synapse.pipeline.node_extension import DistributedPipelineExtension
        node = _placement.TestNodeWiring._node(monkeypatch, ("A", "B"))
        ext = DistributedPipelineExtension(node)
        full = Shard("m", 0, 29, 30)
        ext._get_partition_assignments = lambda bs: [(owner, full)]
        seen = {}

        async def fake_single(shard, prompt, rid, limit):
            seen["local"] = (shard.start_layer, shard.end_layer)
            return [1]

        async def fake_controller(**kw):
            seen["controller"] = [(n, s.start_layer, s.end_layer) for n, s in kw["partition_assignments"]]
            return [2]
        ext._run_single_node = fake_single
        ext._controller.process_distributed_prompt = fake_controller
        out = await ext.process_prompt_distributed(Shard("m", 0, 0, 30), "hi", request_id="r")
        if owner == "A":
            assert seen == {"local": (0, 29)} and out == [1]
        else:
            assert seen == {"controller": [("B", 0, 29)]} and out == [2]
