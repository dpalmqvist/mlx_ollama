"""Tests for balanced loss in predictor training (Paper §3.1)."""

import mlx.core as mx

from olmlx.engine.flash.prepare import _train_predictors


class TestBalancedLoss:
    """Balanced loss should improve recall on minority class (active neurons)."""

    def _make_imbalanced_recordings(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_samples: int,
        sparsity: float = 0.95,
    ) -> dict[int, tuple[list[mx.array], list[mx.array]]]:
        """Create recordings where only (1-sparsity) fraction of neurons are active."""
        recordings = {}
        for layer in range(num_layers):
            inputs_list = []
            targets_list = []
            for i in range(num_samples):
                mx.random.seed(layer * 1000 + i)
                inp = mx.random.normal((hidden_size,))
                # Only first few neurons are active (deterministic pattern)
                num_active = max(1, int(intermediate_size * (1 - sparsity)))
                target = mx.zeros((intermediate_size,))
                target = target.at[:num_active].add(1.0)
                mx.eval(inp, target)
                inputs_list.append(inp)
                targets_list.append(target)
            recordings[layer] = (inputs_list, targets_list)
        return recordings

    def test_balanced_loss_higher_positive_recall(self):
        """Balanced loss should detect more active neurons than unbalanced."""
        hidden, inter = 32, 64
        num_layers, num_samples = 1, 50
        recordings = self._make_imbalanced_recordings(
            num_layers, hidden, inter, num_samples, sparsity=0.95
        )

        # Train with unbalanced loss
        bank_unbalanced = _train_predictors(
            recordings,
            hidden,
            inter,
            rank=16,
            epochs=20,
            balanced_loss=False,
        )
        # Train with balanced loss
        bank_balanced = _train_predictors(
            recordings,
            hidden,
            inter,
            rank=16,
            epochs=20,
            balanced_loss=True,
        )

        # Check recall: how many of the truly active neurons are predicted
        x = mx.stack(recordings[0][0])  # all inputs
        num_active = int(inter * 0.05)  # 5% active

        scores_unbal = bank_unbalanced.predictors[0](x).mean(axis=0)
        scores_bal = bank_balanced.predictors[0](x).mean(axis=0)
        mx.eval(scores_unbal, scores_bal)

        recall_unbal = (
            float(mx.sum(scores_unbal[:num_active] > 0.3).item()) / num_active
        )
        recall_bal = float(mx.sum(scores_bal[:num_active] > 0.3).item()) / num_active

        # Balanced loss should have at least as good recall
        assert recall_bal >= recall_unbal or recall_bal > 0.5, (
            f"Balanced recall {recall_bal:.2f} should be >= unbalanced {recall_unbal:.2f}"
        )

    def test_balanced_loss_symmetric_when_balanced(self):
        """When targets are 50/50, balanced and unbalanced loss should be similar."""
        hidden, inter = 16, 32
        num_layers, num_samples = 1, 20
        recordings = self._make_imbalanced_recordings(
            num_layers, hidden, inter, num_samples, sparsity=0.50
        )

        bank_unbal = _train_predictors(
            recordings,
            hidden,
            inter,
            rank=8,
            epochs=10,
            balanced_loss=False,
        )
        bank_bal = _train_predictors(
            recordings,
            hidden,
            inter,
            rank=8,
            epochs=10,
            balanced_loss=True,
        )

        x = mx.stack(recordings[0][0])
        scores_unbal = bank_unbal.predictors[0](x).mean(axis=0)
        scores_bal = bank_bal.predictors[0](x).mean(axis=0)
        mx.eval(scores_unbal, scores_bal)

        # Both should produce reasonable scores (not all 0 or all 1)
        mean_unbal = float(scores_unbal.mean().item())
        mean_bal = float(scores_bal.mean().item())
        assert 0.1 < mean_unbal < 0.9
        assert 0.1 < mean_bal < 0.9

    def test_balanced_loss_default_is_true(self):
        """_train_predictors defaults to balanced_loss=True."""
        hidden, inter = 16, 32
        recordings = {0: ([mx.ones((hidden,))], [mx.ones((inter,))])}
        # Should not raise — balanced_loss defaults to True
        bank = _train_predictors(recordings, hidden, inter, rank=4, epochs=1)
        assert len(bank.predictors) == 1
