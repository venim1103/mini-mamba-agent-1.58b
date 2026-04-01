import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


class TestCreateSeqIdxBatch:
    """Test train.py's create_seq_idx_batch helper function."""

    @pytest.fixture
    def setup_device(self):
        """Set up device for tests."""
        with patch("train.DEVICE", "cpu"):
            yield

    def test_single_segment_all_zeros(self, setup_device):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert seq_idx[0].unique().item() == 0

    def test_two_segments(self, setup_device):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()

    def test_three_segments(self, setup_device):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 2, 5, 8]], dtype=torch.int32)
        n_segs = torch.tensor([4])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        expected = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.int32)
        assert torch.equal(seq_idx, expected)

    def test_batch_size_two(self, setup_device):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 4, 8, -1], [0, 2, 8, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2, 3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (2, 8)
        assert (seq_idx[0, :4] == 0).all()
        assert (seq_idx[0, 4:] == 1).all()
        assert (seq_idx[1, :2] == 0).all()
        assert (seq_idx[1, 2:] == 1).all()

    def test_truncation_handling(self, setup_device):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 6)
        assert seq_idx.shape == (1, 6)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()


class TestTrainConfigs:
    """Test that train.py configuration constants are valid."""

    def test_train_configs_has_required_phases(self):
        from train import TRAIN_CONFIGS
        required_phases = ["Phase_1", "Phase_2", "Phase_3", "Phase_4"]
        for phase in required_phases:
            assert phase in TRAIN_CONFIGS

    def test_each_phase_has_weighted_datasets(self):
        from train import TRAIN_CONFIGS
        for phase_name, datasets in TRAIN_CONFIGS.items():
            assert len(datasets) > 0
            weights = [d["weight"] for d in datasets]
            assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_curriculum_config_valid(self):
        from train import CURRICULUM_CONFIG
        assert "peak_lr" in CURRICULUM_CONFIG
        assert "end_lr" in CURRICULUM_CONFIG
        assert "phases" in CURRICULUM_CONFIG
        assert len(CURRICULUM_CONFIG["phases"]) == 4

    def test_phase_percentages_sum_to_one(self):
        from train import CURRICULUM_CONFIG
        total_pct = sum(p["pct"] for p in CURRICULUM_CONFIG["phases"])
        assert total_pct == pytest.approx(1.0, abs=0.01)

    def test_model_configs_valid_dimensions(self):
        from train import MODEL_CONFIG
        assert MODEL_CONFIG["vocab_size"] > 0
        assert MODEL_CONFIG["dim"] > 0
        assert MODEL_CONFIG["n_layers"] > 0


class TestModelConfigModes:
    """Test model configuration modes."""

    def test_scout_mode_config(self):
        with patch.dict("train.__dict__", {"MODE": "scout"}):
            from importlib import reload
            import train
            reload(train)
            assert train.MODEL_CONFIG["dim"] == 512
            assert train.MODEL_CONFIG["n_layers"] == 24

    def test_upscaled_mode_config(self):
        with patch.dict("train.__dict__", {"MODE": "upscaled"}):
            from importlib import reload
            import train
            reload(train)
            assert train.MODEL_CONFIG["dim"] == 1024
            assert train.MODEL_CONFIG["n_layers"] == 64
            assert train.MODEL_CONFIG.get("use_attn") == True
            assert train.MODEL_CONFIG.get("attn_pct") == 0.08


class TestTrainImports:
    """Test that train.py imports work correctly."""

    def test_imports_are_valid(self):
        from train import (
            BitMambaLLM,
            chunked_cross_entropy,
            maybe_autocast,
            setup_mamba_optimizers,
            FGWSD_Scheduler,
            setup_distributed,
            cleanup_distributed,
            is_main_process,
            wrap_model_ddp,
            unwrap_model,
            barrier,
        )
        assert BitMambaLLM is not None
        assert chunked_cross_entropy is not None

    def test_train_constants_defined(self):
        from train import (
            DEVICE,
            BATCH_SIZE,
            GRAD_ACCUM_STEPS,
            SAVE_EVERY,
            AMP_DTYPE,
            CHECKPOINT_DIR,
        )
        assert BATCH_SIZE > 0
        assert GRAD_ACCUM_STEPS > 0
        assert SAVE_EVERY > 0
