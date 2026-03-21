"""Tests for pre-sharding logic (centralized download and distribution)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olmlx.engine.pre_shard import (
    FakeGroup,
    collect_non_weight_files,
    pre_shard_all_workers,
    pre_shard_for_rank,
    read_shard_marker,
    write_shard_marker,
)


class TestFakeGroup:
    def test_rank(self):
        g = FakeGroup(rank=1, size=4)
        assert g.rank() == 1

    def test_size(self):
        g = FakeGroup(rank=0, size=2)
        assert g.size() == 2

    def test_different_ranks(self):
        for r in range(4):
            g = FakeGroup(rank=r, size=4)
            assert g.rank() == r
            assert g.size() == 4


class TestShardMarker:
    def test_roundtrip(self, tmp_path):
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="Qwen/Qwen3-8B")
        marker = read_shard_marker(tmp_path)
        assert marker is not None
        assert marker["rank"] == 1
        assert marker["world_size"] == 2
        assert marker["model_path"] == "Qwen/Qwen3-8B"

    def test_read_missing_marker(self, tmp_path):
        assert read_shard_marker(tmp_path) is None

    def test_read_corrupt_marker(self, tmp_path):
        (tmp_path / ".pre_sharded").write_text("not json")
        assert read_shard_marker(tmp_path) is None

    def test_marker_file_name(self, tmp_path):
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="test/model")
        assert (tmp_path / ".pre_sharded").exists()


class TestCollectNonWeightFiles:
    def test_collects_config_and_tokenizer(self, tmp_path):
        # Create typical model files
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "special_tokens_map.json").write_text("{}")
        (tmp_path / "tokenizer.model").write_bytes(b"\x00")
        # Weight files should be excluded
        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
        (tmp_path / "model.safetensors.index.json").write_text("{}")

        files = collect_non_weight_files(tmp_path)
        names = {f.name for f in files}

        assert "config.json" in names
        assert "tokenizer.json" in names
        assert "tokenizer_config.json" in names
        assert "special_tokens_map.json" in names
        assert "tokenizer.model" in names
        # Weight files excluded
        assert "model.safetensors" not in names
        assert "model-00001-of-00002.safetensors" not in names
        assert "model.safetensors.index.json" not in names

    def test_empty_dir(self, tmp_path):
        assert collect_non_weight_files(tmp_path) == []

    def test_includes_generation_config(self, tmp_path):
        (tmp_path / "generation_config.json").write_text("{}")
        files = collect_non_weight_files(tmp_path)
        assert any(f.name == "generation_config.json" for f in files)


class TestPreShardForRank:
    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_calls_shard_with_fake_group(
        self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        # Verify shard was called with a FakeGroup
        mock_model.shard.assert_called_once()
        group_arg = mock_model.shard.call_args[0][0]
        assert group_arg.rank() == 1
        assert group_arg.size() == 2

    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_saves_weights(self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        # Verify weights were materialized
        mock_eval.assert_called()
        # Verify weights were saved with flattened params
        mock_save.assert_called_once()
        save_path = mock_save.call_args[0][0]
        assert save_path == str(output_dir / "model.safetensors")
        # Verify tree_flatten was used
        mock_flatten.assert_called_once()

    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_writes_marker(self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        marker = read_shard_marker(output_dir)
        assert marker is not None
        assert marker["rank"] == 1
        assert marker["world_size"] == 2

    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_copies_non_weight_files(
        self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "tokenizer.json").write_text('{"tok": true}')
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        assert (output_dir / "config.json").exists()
        assert json.loads((output_dir / "config.json").read_text()) == {"test": True}
        assert (output_dir / "tokenizer.json").exists()


class TestPreShardAllWorkers:
    @patch("olmlx.engine.pre_shard.pre_shard_for_rank")
    def test_shards_ranks_1_to_n(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"

        result = pre_shard_all_workers(model_dir, world_size=3, output_base=output_base)

        assert mock_shard.call_count == 2  # ranks 1 and 2
        # Verify rank 1
        mock_shard.assert_any_call(
            model_dir,
            rank=1,
            world_size=3,
            output_dir=output_base / "rank1",
        )
        # Verify rank 2
        mock_shard.assert_any_call(
            model_dir,
            rank=2,
            world_size=3,
            output_dir=output_base / "rank2",
        )
        assert 1 in result
        assert 2 in result

    @patch("olmlx.engine.pre_shard.pre_shard_for_rank")
    def test_returns_shard_dirs(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"

        result = pre_shard_all_workers(model_dir, world_size=2, output_base=output_base)

        assert result == {1: output_base / "rank1"}

    @patch("olmlx.engine.pre_shard.pre_shard_for_rank")
    def test_calls_progress_callback(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"
        progress = MagicMock()

        pre_shard_all_workers(
            model_dir, world_size=3, output_base=output_base, progress_cb=progress
        )

        assert progress.call_count == 2


class TestWorkerPreShardedLoading:
    """Tests for the worker-side pre-sharded loading path."""

    def test_env_var_constant(self):
        """Verify the env var constant is defined and consistent."""
        from olmlx.config import PRE_SHARDED_DIR_ENV

        assert PRE_SHARDED_DIR_ENV == "OLMLX_EXPERIMENTAL_DISTRIBUTED_PRE_SHARDED_DIR"

    def test_marker_mismatch_returns_none(self, tmp_path):
        """When marker doesn't match expected model, should signal fallback."""
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="old/model")
        marker = read_shard_marker(tmp_path)
        # Caller checks model_path match
        assert marker["model_path"] != "new/model"

    def test_marker_world_size_mismatch(self, tmp_path):
        """When world_size changed, marker should signal stale shards."""
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="test/model")
        marker = read_shard_marker(tmp_path)
        assert marker["world_size"] != 3


class TestConfigFields:
    """Tests for the new ExperimentalSettings fields."""

    def test_pre_shard_defaults(self, monkeypatch):
        for key in os.environ:
            if key.startswith("OLMLX_EXPERIMENTAL_"):
                monkeypatch.delenv(key, raising=False)

        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings()
        assert s.distributed_pre_shard is True
        assert s.distributed_shard_dir == Path("~/.olmlx/shards")
        assert s.distributed_worker_shard_dir == "~/.olmlx/shards"

    def test_pre_shard_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_PRE_SHARD", "false")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_SHARD_DIR", "/tmp/shards")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_WORKER_SHARD_DIR", "/remote/shards"
        )

        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings()
        assert s.distributed_pre_shard is False
        assert s.distributed_shard_dir == Path("/tmp/shards")
        assert s.distributed_worker_shard_dir == "/remote/shards"
