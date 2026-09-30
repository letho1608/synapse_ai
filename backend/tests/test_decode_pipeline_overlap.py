"""
Behavioral tests: distributed decode with one-pass speculative verification.

Each decode step sends ONE context through the pipeline:
[prompt + verified history + n-gram draft]. The last stage samples a token
for every draft position (verify_n = len(draft) + 1) and the orchestrator
keeps the longest verified prefix plus one token. The older design fired one
micro-batch per guess prefix (k+1 full passes for k guesses) and had no way
to cancel the stale ones; one pass now verifies the whole draft.

These tests drive the REAL shipped functions (verify_speculative_tokens and
process_distributed_prompt) with a controlled oracle last stage.
"""
import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from synapse.inference.distributed_engine import (
    DistributedPipelineController,
    verify_speculative_tokens,
)
from synapse.inference.inference_engine import InferenceEngine
from synapse.inference.shard import Shard


class TestVerifySpeculativeTokens:
    """Pure-function tests for the speculative token verifier.

    batch_tokens[i] is the token sampled at the position whose context is
    the true context + guesses[:i]."""

    def test_all_guesses_correct_accepts_chain(self):
        assert verify_speculative_tokens([6, 7], [[6], [7], [8]]) == [6, 7, 8]

    def test_wrong_guess_stops_chain_no_contaminated_token(self):
        # Guess 99 is wrong: position 1 predicts the TRUE token 7, and the
        # prediction after the contaminated context [.., 6, 99] is dropped.
        assert verify_speculative_tokens([6, 99], [[6], [7], [999]]) == [6, 7]

    def test_wrong_first_guess_only_true_context_token(self):
        assert verify_speculative_tokens([99, 100], [[6], [999], [999]]) == [6]

    def test_no_guesses_accepts_only_true_context_token(self):
        assert verify_speculative_tokens([], [[5], [5], [5]]) == [5]

    def test_empty_first_batch_accepts_nothing(self):
        assert verify_speculative_tokens([6], [[], [7]]) == []

    def test_missing_tokens_in_middle_batch_stops_chain(self):
        assert verify_speculative_tokens([6, 7], [[6], [], [8]]) == [6]

    def test_fewer_guesses_than_batches_is_safe(self):
        assert verify_speculative_tokens([6], [[6], [7], [7]]) == [6, 7]


class _Router:
    """Minimal in-process event router: publish -> subscribers, sync."""

    def __init__(self):
        self.subs = {}
        self.published = []

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def unsubscribe(self, topic, fn):
        if fn in self.subs.get(topic, []):
            self.subs[topic].remove(fn)

    async def publish(self, topic, data, **_):
        self.published.append((topic, data))
        for fn in list(self.subs.get(topic, [])):
            fn(data)


class TestDecodeLoopBehavior:
    """Oracle last stage: the true next token for a true-context prefix,
    999 for any contaminated prefix."""

    PROMPT = [1, 2]
    # True continuation. The repeated [1, 2, 3] makes the n-gram draft fire,
    # and its later misses exercise rejection.
    TRUE = [7, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]

    def _make_controller(self, encode=None):
        mock_engine = MagicMock(spec=InferenceEngine)
        mock_engine.encode = AsyncMock(return_value=encode if encode is not None
                                       else np.array([self.PROMPT]))
        mock_engine.tokenizer = None
        router = _Router()
        controller = DistributedPipelineController(
            node_id="oracle_node", inference_engine=mock_engine, event_router=router)
        controller.lookahead_window = 3
        controller.step_timeout = 2.0
        return controller, router

    @staticmethod
    def _base_shard():
        return Shard(model_id="oracle-model", start_layer=0, end_layer=31, n_layers=32)

    def _true_next(self, prefix, truth):
        full = self.PROMPT + truth
        n = len(prefix)
        if prefix == full[:n] and n < len(full):
            return full[n]
        return 999

    async def _run(self, controller, router, truth=None, max_tokens=None,
                   on_token_callback=None, request_id="oracle-request-1",
                   kv_miss_steps=(), drop_steps=()):
        truth = truth if truth is not None else self.TRUE
        max_tokens = max_tokens if max_tokens is not None else len(truth)
        sent = []

        async def oracle_send(target_node_id, target_shard, input_data, request_id_,
                              is_prompt=False, microbatch_id=0, extra_state=None):
            ctx = input_data.reshape(-1).tolist()
            st = dict(extra_state or {})
            sent.append({"ctx": ctx, "shape": input_data.shape, "state": st,
                         "mb": microbatch_id, "is_prompt": is_prompt})
            step = len(sent) - 1
            if step in drop_steps:
                return
            if step in kv_miss_steps and st.get("delta_rows"):
                await router.publish(f"synapse/inference/result/{request_id_}", {
                    "tokens": [], "kv_miss": True, "micro_batch_id": microbatch_id,
                    "source_stage_id": None})
                return
            n = st["verify_n"]
            toks = [self._true_next(ctx[:len(ctx) - n + 1 + j], truth) for j in range(n)]
            await router.publish(f"synapse/inference/result/{request_id_}", {
                "tokens": toks, "micro_batch_id": microbatch_id, "source_stage_id": 16})

        controller._send_input_to_stage = oracle_send
        await controller.start()
        try:
            tokens = await controller.process_distributed_prompt(
                base_shard=self._base_shard(), prompt="Hello", request_id=request_id,
                partition_assignments=[
                    ("node0", Shard("oracle-model", 0, 15, 32)),
                    ("node1", Shard("oracle-model", 16, 31, 32)),
                ],
                max_generate_tokens=max_tokens, on_token_callback=on_token_callback)
            return tokens, sent
        finally:
            await controller.stop()

    @pytest.mark.asyncio
    async def test_decode_loop_emits_only_true_context_tokens(self):
        controller, router = self._make_controller()
        tokens, _ = await self._run(controller, router)
        assert tokens == self.TRUE
        assert 999 not in tokens

    @pytest.mark.asyncio
    async def test_every_step_is_prompt_history_plus_draft_in_one_pass(self):
        controller, router = self._make_controller()
        tokens, sent = await self._run(controller, router)
        assert sent[0]["is_prompt"] and not any(s["is_prompt"] for s in sent[1:])
        for s in sent:
            st = s["state"]
            assert st["verify_n"] == len(st["draft"]) + 1
            # context = prompt + verified history + draft, never a bare tail
            assert s["ctx"] == self.PROMPT + st["gen_ids"] + st["draft"]
            assert st["gen_ids"] == self.TRUE[:len(st["gen_ids"])]
            assert st["delta_rows"] is True
        assert any(s["state"]["draft"] for s in sent), "the draft never fired"
        # Speculation saved passes: fewer pipeline passes than tokens.
        assert len(sent) < len(tokens)

    @pytest.mark.asyncio
    async def test_contaminated_positions_are_never_accepted(self):
        controller, router = self._make_controller()
        tokens, sent = await self._run(controller, router)

        def wrong(s):
            st = s["state"]
            g = len(st["gen_ids"])
            return st["draft"] and st["draft"] != self.TRUE[g:g + len(st["draft"])]
        assert any(wrong(s) for s in sent), "test did not exercise a wrong guess"
        assert 999 not in tokens and tokens == self.TRUE
        stats = controller.decode_stats["oracle-request-1"]
        assert stats["accepted_draft"] > 0 and stats["drafted"] > stats["accepted_draft"]

    @pytest.mark.asyncio
    async def test_onedimensional_encode_is_normalized(self):
        """A real engine's encode() returns a flat 1-D array; the first step
        must still send a (1, seq_len) context."""
        controller, router = self._make_controller(encode=np.array(self.PROMPT))
        tokens, sent = await self._run(controller, router)
        assert tokens == self.TRUE
        assert sent[0]["shape"] == (1, 2) and sent[0]["ctx"] == self.PROMPT

    @pytest.mark.asyncio
    async def test_kv_miss_resends_the_step_with_full_hidden_states(self):
        controller, router = self._make_controller()
        tokens, sent = await self._run(controller, router, kv_miss_steps={2})
        assert tokens == self.TRUE
        assert sent[2]["state"]["delta_rows"] is True
        assert sent[3]["state"]["delta_rows"] is False and sent[3]["ctx"] == sent[2]["ctx"]
        assert controller.decode_stats["oracle-request-1"]["kv_miss"] == 1

    @pytest.mark.asyncio
    async def test_lost_step_times_out_with_the_verified_prefix(self):
        controller, router = self._make_controller()
        controller.step_timeout = 0.2
        tokens, _ = await self._run(controller, router, drop_steps={2})
        assert tokens == self.TRUE[:len(tokens)] and 0 < len(tokens) < len(self.TRUE)

    @pytest.mark.asyncio
    async def test_stop_token_ends_generation_without_emitting_it(self):
        controller, router = self._make_controller()
        controller.inference_engine.tokenizer = MagicMock(eos_token_id=11)
        tokens, _ = await self._run(controller, router)
        assert tokens == self.TRUE[:self.TRUE.index(11)]

    @pytest.mark.asyncio
    async def test_max_tokens_caps_even_an_accepted_draft(self):
        controller, router = self._make_controller()
        tokens, _ = await self._run(controller, router, max_tokens=6)
        assert tokens == self.TRUE[:6]


class TestStreamingRepeatedTokens:
    """
    B1: legitimate adjacent repeated tokens must survive the streaming path.

    The old `_on_token_received` suffix-dedup dropped any incoming token(s)
    that already ended the buffer — which also dropped genuine repeats like
    "the the". Redundant deliveries (the sampler's own echoes) are skipped by
    delivery-source, never by token content.
    """

    @staticmethod
    def _make_node() -> "Node":
        from synapse.orchestration.node import Node
        from synapse.routing.event_router import EventRouter
        from synapse.routing.libp2p_node import Libp2pNode
        from synapse.inference.inference_engine import InferenceEngine

        class MockEngine(InferenceEngine):
            async def encode(self, shard, prompt): return np.array([])
            async def sample(self, x): return np.array([1])
            async def decode(self, shard, tokens): return ""
            async def infer_tensor(self, rid, s, t, state=None): return np.array([]), {}
            async def load_checkpoint(self, s, p): pass

        router = EventRouter()
        libp2p = Libp2pNode("test-node-stream", 59999, router)
        return Node(
            _id="test-node-stream",
            event_router=router,
            libp2p_node=libp2p,
            inference_engine=MockEngine(),
            shard_downloader=None,
        )

    @pytest.mark.asyncio
    async def test_adjacent_repeated_tokens_survive_buffer(self):
        node = self._make_node()
        request_id = "stream-repeat-1"
        node._on_token_received(request_id, [42], False)
        node._on_token_received(request_id, [42], False)
        tokens, _ = node.buffered_token_output[request_id]
        assert tokens == [42, 42], f"legit adjacent repeat was dropped: {tokens}"

    @pytest.mark.asyncio
    async def test_sampler_echo_is_not_double_appended(self):
        node = self._make_node()
        request_id = "stream-echo-1"
        node.buffered_token_output[request_id] = ([7], False)
        node._locally_sampled.add(request_id)
        node._on_token_received(request_id, [7], False)
        tokens, _ = node.buffered_token_output[request_id]
        assert tokens == [7], f"sampler echo double-appended: {tokens}"

    @pytest.mark.asyncio
    async def test_decode_loop_streams_adjacent_repeats_into_buffer(self):
        """End-to-end: the shipped decode loop with a true continuation that
        contains a genuine adjacent repeat; the real Node buffer (fed via the
        exact callback node_extension registers) must contain both."""
        node = self._make_node()
        request_id = "stream-repeat-loop"
        harness = TestDecodeLoopBehavior()
        controller, router = harness._make_controller()
        truth = [7, 1, 2, 3, 9, 9]
        tokens, _ = await harness._run(
            controller, router, truth=truth, request_id=request_id,
            on_token_callback=node._on_token_received)
        assert tokens == truth
        buffered, _ = node.buffered_token_output[request_id]
        assert buffered == truth, f"buffer lost a legitimate adjacent repeat: {buffered}"


class TestOnlyVerifiedTokensReachTheStream:
    """Found 2026-09-30 in the live 2-process run: the last stage publishes
    one sample per draft position (incl. unverified ones) on the result
    topic, and the orchestrator's on_p2p_result streamed them to the API:
    "The capital of France is That in one Paris." Only the decode loop's
    verified tokens may reach the stream."""

    @pytest.mark.asyncio
    async def test_raw_verify_samples_are_not_streamed(self):
        node = TestStreamingRepeatedTokens._make_node()
        seen = []
        node.on_token.register("t").on_next(lambda rid, toks, fin: seen.append(list(toks)))
        ev = MagicMock(data={"request_id": "r1", "tokens": [5, 6, 7], "verify": True,
                             "micro_batch_id": 0})
        await node.on_p2p_result(ev)
        ev2 = MagicMock(data={"request_id": "r1", "tokens": [], "kv_miss": True})
        await node.on_p2p_result(ev2)
        assert seen == []

    @pytest.mark.asyncio
    async def test_decode_loop_streams_verified_tokens_to_callbacks(self):
        from synapse.pipeline.node_extension import DistributedPipelineExtension
        node = TestStreamingRepeatedTokens._make_node()
        seen = []
        node.on_token.register("t").on_next(lambda rid, toks, fin: seen.append((list(toks), fin)))
        ext = DistributedPipelineExtension(node)
        harness = TestDecodeLoopBehavior()
        controller, router = harness._make_controller()
        ext._controller = controller
        captured = {}

        async def fake_process(**kw):
            captured.update(kw)
            cb = kw["on_token_callback"]
            cb(kw["request_id"], [4, 5], False)
            cb(kw["request_id"], [], True)
            return [4, 5]
        controller.process_distributed_prompt = fake_process
        ext._get_partition_assignments = lambda bs: [("a", bs), ("b", bs)]
        out = await ext.process_prompt_distributed(
            TestDecodeLoopBehavior._base_shard(), "hi", request_id="rq")
        assert out == [4, 5]
        assert seen == [([4, 5], False), ([], True)]
        assert node.buffered_token_output["rq"][0] == [4, 5]
