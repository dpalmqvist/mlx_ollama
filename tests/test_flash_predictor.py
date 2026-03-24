"""Tests for olmlx.engine.flash.predictor and calibrate."""

import mlx.core as mx

from olmlx.engine.flash.predictor import (
    PredictorBank,
    SparsityPredictor,
    compute_layer_ranks,
)


class TestSparsityPredictor:
    def test_output_shape(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        scores = pred(x)
        mx.eval(scores)
        assert scores.shape == (1, inter)

    def test_scores_are_between_0_and_1(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((4, hidden))
        scores = pred(x)
        mx.eval(scores)
        assert mx.all(scores >= 0).item()
        assert mx.all(scores <= 1).item()

    def test_predict_active_respects_min_neurons(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        min_n = 16
        indices = pred.predict_active(x, threshold=0.99, min_neurons=min_n)
        mx.eval(indices)
        assert indices.shape[0] >= min_n

    def test_predict_active_respects_max_neurons(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        max_n = 4
        indices = pred.predict_active(x, threshold=0.0, max_neurons=max_n)
        mx.eval(indices)
        assert indices.shape[0] <= max_n

    def test_predict_active_returns_sorted_indices(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        indices = pred.predict_active(x, threshold=0.5, min_neurons=4)
        mx.eval(indices)
        arr = indices.tolist()
        assert arr == sorted(arr)

    def test_predict_active_indices_in_valid_range(self):
        hidden, inter, rank = 32, 64, 8
        pred = SparsityPredictor(hidden, inter, rank)
        x = mx.random.normal((1, hidden))
        indices = pred.predict_active(x, threshold=0.3, min_neurons=8)
        mx.eval(indices)
        assert mx.all(indices >= 0).item()
        assert mx.all(indices < inter).item()


class TestPredictorBank:
    def test_save_load_roundtrip(self, tmp_path):
        hidden, inter, rank = 16, 32, 4
        num_layers = 3

        bank = PredictorBank(num_layers, hidden, inter, rank)
        save_path = tmp_path / "predictors"
        bank.save(save_path)

        loaded = PredictorBank.load(save_path)
        assert len(loaded.predictors) == num_layers

        # Verify weights match
        x = mx.random.normal((1, hidden))
        for i in range(num_layers):
            orig_scores = bank.predictors[i](x)
            loaded_scores = loaded.predictors[i](x)
            mx.eval(orig_scores, loaded_scores)
            assert mx.allclose(orig_scores, loaded_scores, atol=1e-6)

    def test_predict_layer(self):
        hidden, inter, rank = 16, 32, 4
        num_layers = 2
        bank = PredictorBank(num_layers, hidden, inter, rank)

        x = mx.random.normal((1, hidden))
        indices = bank.predict_layer(0, x, threshold=0.5, min_neurons=4)
        mx.eval(indices)
        assert indices.shape[0] >= 4
        assert mx.all(indices < inter).item()

    def test_different_layers_give_different_predictions(self):
        """Each layer's predictor is independently initialized."""
        hidden, inter, rank = 16, 32, 4
        num_layers = 2
        bank = PredictorBank(num_layers, hidden, inter, rank)

        # Ensure predictors have distinct weights (random init can collide)
        bank.predictors[0].down.weight = mx.random.normal(
            bank.predictors[0].down.weight.shape, key=mx.random.key(0)
        )
        bank.predictors[1].down.weight = mx.random.normal(
            bank.predictors[1].down.weight.shape, key=mx.random.key(1)
        )

        x = mx.random.normal((1, hidden))
        s0 = bank.predictors[0](x)
        s1 = bank.predictors[1](x)
        mx.eval(s0, s1)
        assert not mx.array_equal(s0, s1)

    def test_per_layer_ranks(self):
        """PredictorBank with per-layer ranks has correct weight shapes."""
        hidden, inter = 16, 32
        ranks = [4, 8, 16]
        bank = PredictorBank(3, hidden, inter, ranks=ranks)

        for i, rank in enumerate(ranks):
            assert bank.predictors[i].down.weight.shape == (rank, hidden)
            assert bank.predictors[i].up.weight.shape == (inter, rank)

    def test_per_layer_ranks_save_load_roundtrip(self, tmp_path):
        """Save and load a bank with varying ranks per layer."""
        hidden, inter = 16, 32
        ranks = [4, 8, 16]
        bank = PredictorBank(3, hidden, inter, ranks=ranks)
        save_path = tmp_path / "predictors"
        bank.save(save_path)

        loaded = PredictorBank.load(save_path)
        assert len(loaded.predictors) == 3

        x = mx.random.normal((1, hidden))
        for i in range(3):
            # Verify shapes match per-layer rank
            assert loaded.predictors[i].down.weight.shape == (ranks[i], hidden)
            assert loaded.predictors[i].up.weight.shape == (inter, ranks[i])
            # Verify weight values match
            orig = bank.predictors[i](x)
            load = loaded.predictors[i](x)
            mx.eval(orig, load)
            assert mx.allclose(orig, load, atol=1e-6)

    def test_uniform_rank_still_works(self):
        """Passing only rank= (no ranks=) still works as before."""
        bank = PredictorBank(3, 16, 32, rank=8)
        for pred in bank.predictors:
            assert pred.down.weight.shape == (8, 16)


class TestComputeLayerRanks:
    def test_basic(self):
        ranks = compute_layer_ranks(
            8, base_rank=128, sensitive_layers=4, sensitive_rank_multiplier=4
        )
        assert len(ranks) == 8
        assert ranks[:4] == [128, 128, 128, 128]
        assert ranks[4:] == [512, 512, 512, 512]

    def test_zero_sensitive_layers(self):
        ranks = compute_layer_ranks(8, base_rank=128, sensitive_layers=0)
        assert all(r == 128 for r in ranks)

    def test_all_sensitive(self):
        ranks = compute_layer_ranks(
            4, base_rank=64, sensitive_layers=4, sensitive_rank_multiplier=2
        )
        assert all(r == 128 for r in ranks)

    def test_more_sensitive_than_layers(self):
        ranks = compute_layer_ranks(
            2, base_rank=64, sensitive_layers=10, sensitive_rank_multiplier=3
        )
        assert all(r == 192 for r in ranks)
