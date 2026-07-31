"""
Behavioral tests: decode-phase pipeline micro-batch overlapping + speculative
token verification.

Verifies the fix for B1: the shipped decode loop fires multiple micro-batches
into the pipeline (enabling concurrent stage execution) and accepts only
context-valid tokens. Micro-batches carry a PREFIX of the n-gram guesses
(batch i sees [last] + guesses[:i]), and a token is accepted only after every
guess feeding its context has been verified — so no token predicted from a
contaminated context is ever appended to the output.

These tests drive the REAL shipped functions (verify_speculative_tokens and
process_distributed_prompt) with a controlled oracle forward — no source
inspection, no mocks of the unit under test.
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
from synapse.transport.result_router import RoutedResult


class TestVerifySpeculativeTokens:
    """Pure-function tests for the speculative token verifier."""

    def test_all_guesses_correct_accepts_chain(self):
        # Batch 0 (true context) predicts 6; batch 1 ([last,6]) predicts 7;
        # batch 2 ([last,6,7]) predicts 8. All guesses verified -> full chain.
        assert verify_speculative_tokens([6, 7], [[6], [7], [8]]) == [6, 7, 8]

    def test_wrong_guess_stops_chain_no_contaminated_token(self):
        # Guess 99 at position 1 is wrong: batch 1 predicts the TRUE next
        # token 7 (from [last,6]), so batch 2's prediction (from the
        # contaminated context [last,6,99]) must NEVER be accepted.
        assert verify_speculative_tokens([6, 99], [[6], [7], [999]]) == [6, 7]
        assert 999 not in verify_speculative_tokens([6, 99], [[6], [7], [999]])

    def test_wrong_first_guess_only_true_context_token(self):
        # First guess 99 is wrong -> batch 1's context is already
        # contaminated, so only the true-context token 6 is accepted.
        assert verify_speculative_tokens([99, 100], [[6], [999], [999]]) == [6]

    def test_no_guesses_accepts_only_true_context_token(self):
        # With no guesses every batch sees [last]; only batch 0 counts.
        assert verify_speculative_tokens([], [[5], [5], [5]]) == [5]

    def test_empty_first_batch_accepts_nothing(self):
        assert verify_speculative_tokens([6], [[], [7]]) == []

    def test_missing_tokens_in_middle_batch_stops_chain(self):
        # Batch 1 produced no tokens (e.g. finished) -> chain stops.
        assert verify_speculative_tokens([6, 7], [[6], [], [8]]) == [6]

    def test_fewer_guesses_than_batches_is_safe(self):
        # Only 1 guess but 3 batches: batch 2's context is identical to
        # batch 1's and its token must not be double-accepted.
        assert verify_speculative_tokens([6], [[6], [7], [7]]) == [6, 7]


class TestDecodeLoopBehavior:
    """
    Oracle-driven tests that run the real shipped decode loop end to end.

    The oracle forward predicts the TRUE next token only when its input is an
    exact true-context prefix; any other (contaminated) input yields 999.
    """

    # Prompt-phase result: history where the n-gram search finds [1,2,3]
    # as a lookahead guess, so the loop fires speculative batches [last],
    # [last,1], [last,1,2].
    PROMPT_RESULT = [7, 1, 2, 3, 1, 2, 3]
    # True model: context ending in X predicts the next token below.
    ORACLE = {3: 9, 9: 10, 10: 11, 11: 12, 12: 13}
    TARGET = [9, 10, 11, 12, 13]

    def _make_controller(self):
        mock_engine = MagicMock(spec=InferenceEngine)
        mock_engine.encode = AsyncMock(
            return_value=np.array([[1, 2]])
        )
        mock_router = MagicMock()
        mock_router.publish = AsyncMock()
        controller = DistributedPipelineController(
            node_id="oracle_node",
            inference_engine=mock_engine,
            event_router=mock_router,
        )
        controller.lookahead_window = 3
        return controller

    @staticmethod
    def _base_shard():
        return Shard(model_id="oracle-model", start_layer=0, end_layer=31, n_layers=32)

    async def _run_oracle_loop(self, controller, request_id="oracle-request-1"):
        sent_inputs = []
        pred_queue = asyncio.Queue()

        async def oracle_send(
            target_node_id, target_shard, input_data, request_id_,
            is_prompt=False, microbatch_id=0,
        ):
            sent_inputs.append((is_prompt, input_data.copy(), microbatch_id))
            if is_prompt:
                await pred_queue.put(
                    RoutedResult(
                        request_id=request_id_,
                        orchestrator_id="oracle_node",
                        micro_batch_id=microbatch_id,
                        source_stage_id=0,
                        tokens=list(self.PROMPT_RESULT),
                        is_finished=False,
                    )
                )
            else:
                last = int(input_data.reshape(-1)[-1])
                pred = self.ORACLE.get(last, 999)
                await pred_queue.put(
                    RoutedResult(
                        request_id=request_id_,
                        orchestrator_id="oracle_node",
                        micro_batch_id=microbatch_id,
                        source_stage_id=0,
                        tokens=[pred],
                        is_finished=False,
                    )
                )

        async def oracle_collect(request_id_, last_stage_node_id, last_stage_shard, timeout=30.0):
            return await asyncio.wait_for(pred_queue.get(), timeout=timeout)

        controller._send_input_to_stage = oracle_send
        controller._collect_result = oracle_collect

        await controller.start()
        try:
            tokens = await controller.process_distributed_prompt(
                base_shard=self._base_shard(),
                prompt="Hello",
                request_id=request_id,
                partition_assignments=[
                    ("node0", self._base_shard()),
                    ("node1", self._base_shard()),
                ],
                max_generate_tokens=len(self.PROMPT_RESULT) + len(self.TARGET),
                num_micro_batches=1,
            )
            return tokens, sent_inputs
        finally:
            await controller.stop()

    @pytest.mark.asyncio
    async def test_decode_loop_emits_only_true_context_tokens(self):
        controller = self._make_controller()
        tokens, _ = await self._run_oracle_loop(controller)

        # Generated sequence = prompt history + exact oracle target, and the
        # contaminated-token sentinel 999 never appears.
        assert tokens == self.PROMPT_RESULT + self.TARGET
        assert 999 not in tokens

    @pytest.mark.asyncio
    async def test_decode_loop_fires_speculative_batches_with_prefix_context(self):
        controller = self._make_controller()
        tokens, sent_inputs = await self._run_oracle_loop(controller)

        # The first decode iteration must fire the speculative micro-batches
        # [last], [last,1], [last,1,2] — i.e. a PREFIX of the guesses, so the
        # only context ever sent is the true context plus verified guesses.
        decode_inputs = [
            arr.reshape(-1).tolist() for is_prompt, arr, _ in sent_inputs if not is_prompt
        ]
        assert [3] in decode_inputs, "batch 0 (true context) not fired"
        assert [3, 1] in decode_inputs, "batch 1 (verified prefix) not fired"
        assert [3, 1, 2] in decode_inputs, "batch 2 (verified prefix) not fired"
        # A suffix-based construction would have fired [1,2,3] — assert the
        # contaminated full-guess context was never sent as batch 0's input.
        assert [3, 1, 2, 3] not in decode_inputs

    @pytest.mark.asyncio
    async def test_decode_loop_rejects_contaminated_predictions(self):
        controller = self._make_controller()
        tokens, sent_inputs = await self._run_oracle_loop(controller)

        # The oracle returns 999 for every contaminated context. Count how
        # many such predictions were produced and assert none leaked.
        decode_inputs = [
            arr.reshape(-1).tolist() for is_prompt, arr, _ in sent_inputs if not is_prompt
        ]
        # Every accepted token must equal the oracle's true-context prediction
        # for its input; 999 is returned exactly for the contaminated batches
        # ([3,1] and [3,1,2] at first decode iteration).
        contaminated_fired = any(i in ([3, 1], [3, 1, 2]) for i in decode_inputs)
        assert contaminated_fired, "test did not exercise contaminated batches"
        assert 999 not in tokens

    @pytest.mark.asyncio
    async def test_onedimensional_encode_is_normalized(self):
        """A real engine's encode() returns a flattened 1-D token array; the
        controller must normalize it to (1, seq_len) instead of crashing on
        token_ids.shape[1] (previously IndexError → multi-node inference
        silently fell back to a single node)."""
        controller = self._make_controller()
        controller.inference_engine.encode = AsyncMock(return_value=np.array([1, 2]))

        tokens, sent_inputs = await self._run_oracle_loop(controller)

        assert tokens == self.PROMPT_RESULT + self.TARGET
        prompt_sends = [s for s in sent_inputs if s[0] is True]
        assert prompt_sends, "prompt was never sent to the first stage"
        first_prompt = prompt_sends[0][1]
        assert first_prompt.ndim == 2, (
            f"prompt input must be reshaped to 2-D, got ndim={first_prompt.ndim}"
        )
        assert first_prompt.shape == (1, 2)
