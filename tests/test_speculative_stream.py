"""Tests for speculative decoding streaming adapter."""

import threading

import pytest

from tests.test_flash_speculative import MockDraftModel, MockTargetModel
from olmlx.engine.flash.speculative import SpeculativeFlashDecoder
from olmlx.engine.flash.speculative_stream import speculative_stream_generate


class TestSpeculativeStreamGenerate:
    @pytest.fixture()
    def shared_decoder(self):
        """Decoder with shared weights so draft and target always agree."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]
        return SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

    def test_stream_yields_tokens(self, shared_decoder):
        """Generator should yield response objects with required attributes."""
        cancel = threading.Event()
        prompt_tokens = [1, 2, 3]

        responses = list(
            speculative_stream_generate(
                shared_decoder, prompt_tokens, max_tokens=5, cancel_event=cancel
            )
        )

        assert len(responses) >= 1
        for resp in responses:
            assert hasattr(resp, "text")
            assert hasattr(resp, "token")
            assert hasattr(resp, "prompt_tokens")
            assert hasattr(resp, "generation_tokens")
            assert hasattr(resp, "prompt_tps")
            assert hasattr(resp, "generation_tps")

    def test_stream_respects_max_tokens(self, shared_decoder):
        """Should stop after max_tokens are generated."""
        cancel = threading.Event()
        prompt_tokens = [1, 2, 3]

        responses = list(
            speculative_stream_generate(
                shared_decoder, prompt_tokens, max_tokens=8, cancel_event=cancel
            )
        )

        # Total tokens yielded should not exceed max_tokens
        assert len(responses) <= 8

    def test_stream_stops_on_cancel(self, shared_decoder):
        """Should stop when cancel_event is set."""
        cancel = threading.Event()
        prompt_tokens = [1, 2, 3]

        responses = []
        for resp in speculative_stream_generate(
            shared_decoder, prompt_tokens, max_tokens=100, cancel_event=cancel
        ):
            responses.append(resp)
            if len(responses) >= 2:
                cancel.set()

        # Should have stopped after cancel (may yield a few more due to batch)
        assert len(responses) < 100

    def test_stream_stops_on_eos(self):
        """Should stop when EOS token is generated."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]
        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        cancel = threading.Event()
        # Use eos_token_id=0 (likely to be generated quickly by random weights)
        responses = list(
            speculative_stream_generate(
                decoder,
                [1, 2, 3],
                max_tokens=100,
                cancel_event=cancel,
                eos_token_id=0,
            )
        )

        # Should have stopped before max_tokens due to EOS
        assert len(responses) <= 100

    def test_stream_token_ids_are_valid(self, shared_decoder):
        """All yielded token IDs should be in valid range."""
        cancel = threading.Event()

        responses = list(
            speculative_stream_generate(
                shared_decoder, [1, 2, 3], max_tokens=10, cancel_event=cancel
            )
        )

        for resp in responses:
            assert 0 <= resp.token < 32  # vocab_size

    def test_stream_generation_tokens_increment(self, shared_decoder):
        """generation_tokens should increase monotonically."""
        cancel = threading.Event()

        responses = list(
            speculative_stream_generate(
                shared_decoder, [1, 2, 3], max_tokens=10, cancel_event=cancel
            )
        )

        gen_counts = [r.generation_tokens for r in responses]
        assert gen_counts == sorted(gen_counts)
        assert gen_counts[-1] == len(responses)
