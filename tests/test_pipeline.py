"""Tests for pipeline parallelism module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestLayerAssignment:
    """Test layer index calculation for pipeline parallelism."""

    def test_even_split_2_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 64 layers, 2 ranks, even split [32, 32]
        # rank 0 = last layers, rank 1 = first layers (DeepSeek convention)
        start, end = _compute_layer_range(rank=0, layer_counts=[32, 32])
        assert start == 32
        assert end == 64

        start, end = _compute_layer_range(rank=1, layer_counts=[32, 32])
        assert start == 0
        assert end == 32

    def test_uneven_split_2_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 64 layers, rank 0 gets 44, rank 1 gets 20
        start, end = _compute_layer_range(rank=0, layer_counts=[44, 20])
        assert start == 20
        assert end == 64

        start, end = _compute_layer_range(rank=1, layer_counts=[44, 20])
        assert start == 0
        assert end == 20

    def test_even_split_3_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 63 layers, 3 ranks, [21, 21, 21]
        start, end = _compute_layer_range(rank=0, layer_counts=[21, 21, 21])
        assert start == 42
        assert end == 63

        start, end = _compute_layer_range(rank=1, layer_counts=[21, 21, 21])
        assert start == 21
        assert end == 42

        start, end = _compute_layer_range(rank=2, layer_counts=[21, 21, 21])
        assert start == 0
        assert end == 21

    def test_uneven_split_3_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 64 layers: [30, 20, 14]
        start, end = _compute_layer_range(rank=0, layer_counts=[30, 20, 14])
        assert start == 34  # 20 + 14
        assert end == 64

        start, end = _compute_layer_range(rank=1, layer_counts=[30, 20, 14])
        assert start == 14
        assert end == 34

        start, end = _compute_layer_range(rank=2, layer_counts=[30, 20, 14])
        assert start == 0
        assert end == 14

    def test_default_even_split(self):
        from olmlx.engine.pipeline import _compute_layer_counts

        # 64 layers, 2 ranks -> [32, 32]
        counts = _compute_layer_counts(64, 2)
        assert counts == [32, 32]

    def test_default_uneven_total(self):
        from olmlx.engine.pipeline import _compute_layer_counts

        # 65 layers, 2 ranks -> [33, 32] (extra goes to rank 0)
        counts = _compute_layer_counts(65, 2)
        assert sum(counts) == 65
        assert len(counts) == 2


class TestLayerCountValidation:
    """Test validation of layer_counts parameter."""

    def test_layer_counts_wrong_sum(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=64)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="must sum to"):
            apply_pipeline(model, group, layer_counts=[30, 20])

    def test_layer_counts_wrong_length(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=64)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="must have.*entries"):
            apply_pipeline(model, group, layer_counts=[32, 16, 16])


class TestUnsupportedModel:
    """Test error for models without standard structure."""

    def test_no_inner_model(self):
        from olmlx.engine.pipeline import apply_pipeline

        model = SimpleNamespace()  # no .model attribute
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="does not have a standard"):
            apply_pipeline(model, group)

    def test_inner_missing_layers(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = SimpleNamespace(embed_tokens=None, norm=None)  # no .layers
        model = SimpleNamespace(model=inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="does not have a standard"):
            apply_pipeline(model, group)


class TestMonkeyPatch:
    """Test that apply_pipeline correctly patches the model."""

    def test_pipeline_state_set(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        assert inner.pipeline_rank == 0
        assert inner.pipeline_size == 2
        assert inner.start_idx == 4
        assert inner.end_idx == 8
        assert inner.num_layers == 4

    def test_non_owned_layers_nullified(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # Rank 0 owns layers 4-7, layers 0-3 should be None
        assert all(layer is None for layer in inner.layers[:4])
        assert all(layer is not None for layer in inner.layers[4:8])
        # Truncated to end_idx
        assert len(inner.layers) == 8

    def test_rank1_layers(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=1, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # Rank 1 (highest) owns first layers 0-3
        assert inner.start_idx == 0
        assert inner.end_idx == 4
        assert len(inner.layers) == 4
        assert all(layer is not None for layer in inner.layers)

    def test_outer_layers_property_patched(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # Outer model.layers should return only owned layers
        owned = model.layers
        assert len(owned) == 4
        assert all(layer is not None for layer in owned)

    def test_inner_call_replaced(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        original_call = inner.__call__
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # __call__ should be replaced
        assert inner.__call__ != original_call


class TestHostfileParsing:
    """Test backward-compatible hostfile parsing."""

    def test_no_strategy_defaults_to_tensor(self):
        hostfile = {
            "hosts": ["10.0.1.1", "10.0.1.2"],
            "model": "mlx-community/Qwen3-8B-4bit",
        }
        strategy = hostfile.get("strategy", "tensor")
        layers = hostfile.get("layers")
        assert strategy == "tensor"
        assert layers is None

    def test_pipeline_strategy_with_layers(self):
        hostfile = {
            "hosts": ["10.0.1.1", "10.0.1.2"],
            "model": "mlx-community/Qwen3-32B-4bit",
            "strategy": "pipeline",
            "layers": [44, 20],
        }
        strategy = hostfile.get("strategy", "tensor")
        layers = hostfile.get("layers")
        assert strategy == "pipeline"
        assert layers == [44, 20]


class TestLlamaDetection:
    """Test that Llama sliding window models are detected."""

    def test_detects_sliding_window(self):
        from olmlx.engine.pipeline import _is_llama_sliding_window

        inner = _make_mock_inner_model(num_layers=4)
        inner.sliding_window = 4096
        inner.swa_idx = 0
        assert _is_llama_sliding_window(inner) is True

    def test_no_sliding_window(self):
        from olmlx.engine.pipeline import _is_llama_sliding_window

        inner = _make_mock_inner_model(num_layers=4)
        assert _is_llama_sliding_window(inner) is False

    def test_sliding_window_none(self):
        from olmlx.engine.pipeline import _is_llama_sliding_window

        inner = _make_mock_inner_model(num_layers=4)
        inner.sliding_window = None
        assert _is_llama_sliding_window(inner) is False


# -- Helpers --


def _make_mock_group(rank: int, size: int):
    group = MagicMock()
    group.rank.return_value = rank
    group.size.return_value = size
    return group


class _MockLayer:
    """Minimal mock for a transformer layer."""

    def __init__(self, idx):
        self.idx = idx
        self.use_sliding = False

    def __call__(self, h, mask, cache=None):
        return h


def _make_mock_inner_model(num_layers: int):
    inner = SimpleNamespace(
        embed_tokens=MagicMock(),
        layers=list(_MockLayer(i) for i in range(num_layers)),
        norm=MagicMock(),
    )
    # Add a callable __call__ so we can check it gets replaced
    inner.__call__ = lambda self, *a, **kw: None
    return inner


def _make_mock_outer_model(inner):
    model = SimpleNamespace(model=inner)
    # Outer model exposes layers as property-like access
    model.layers = inner.layers
    return model
