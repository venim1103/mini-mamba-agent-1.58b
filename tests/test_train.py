import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


def mock_imports():
    """Mock all heavy imports to avoid model computation."""
    mock_model = MagicMock()
    mock_optim = MagicMock()
    mock_dist = MagicMock()
    
    sys.modules['model'] = MagicMock()
    sys.modules['model'].BitMambaLLM = MagicMock()
    sys.modules['model'].chunked_cross_entropy = MagicMock()
    sys.modules['model'].maybe_autocast = MagicMock()
    
    sys.modules['optim'] = MagicMock()
    sys.modules['optim'].setup_mamba_optimizers = MagicMock()
    sys.modules['optim'].FGWSD_Scheduler = MagicMock()
    
    sys.modules['dist_utils'] = MagicMock()
    sys.modules['dist_utils'].setup_distributed = MagicMock()
    sys.modules['dist_utils'].cleanup_distributed = MagicMock()
    sys.modules['dist_utils'].is_main_process = MagicMock()
    sys.modules['dist_utils'].wrap_model_ddp = MagicMock()
    sys.modules['dist_utils'].unwrap_model = MagicMock()
    sys.modules['dist_utils'].barrier = MagicMock()


import sys

class TestCreateSeqIdxBatch:
    """Test train.py's create_seq_idx_batch helper function."""

    @pytest.fixture(autouse=True)
    def mock_setup(self):
        mock_imports()

    def test_single_segment_all_zeros(self):
        import importlib
        if 'train' not in sys.modules:
            mock_imports()
        import train
        importlib.reload(train)
        
        cu_seqlens = torch.tensor([[0, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2])
        with patch("train.DEVICE", "cpu"):
            seq_idx = train.create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert seq_idx[0].unique().item() == 0

    def test_two_segments(self):
        import importlib
        if 'train' not in sys.modules:
            mock_imports()
        import train
        importlib.reload(train)
        
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        with patch("train.DEVICE", "cpu"):
            seq_idx = train.create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()

    def test_three_segments(self):
        import importlib
        if 'train' not in sys.modules:
            mock_imports()
        import train
        importlib.reload(train)
        
        cu_seqlens = torch.tensor([[0, 2, 5, 8]], dtype=torch.int32)
        n_segs = torch.tensor([4])
        with patch("train.DEVICE", "cpu"):
            seq_idx = train.create_seq_idx_batch(cu_seqlens, n_segs, 8)
        expected = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.int32)
        assert torch.equal(seq_idx, expected)

    def test_batch_size_two(self):
        import importlib
        if 'train' not in sys.modules:
            mock_imports()
        import train
        importlib.reload(train)
        
        cu_seqlens = torch.tensor([[0, 4, 8, -1], [0, 2, 8, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2, 3])
        with patch("train.DEVICE", "cpu"):
            seq_idx = train.create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (2, 8)
        assert (seq_idx[0, :4] == 0).all()
        assert (seq_idx[0, 4:] == 1).all()
        assert (seq_idx[1, :2] == 0).all()
        assert (seq_idx[1, 2:] == 1).all()

    def test_truncation_handling(self):
        import importlib
        if 'train' not in sys.modules:
            mock_imports()
        import train
        importlib.reload(train)
        
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        with patch("train.DEVICE", "cpu"):
            seq_idx = train.create_seq_idx_batch(cu_seqlens, n_segs, 6)
        assert seq_idx.shape == (1, 6)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()


class TestTrainConfigs:
    """Test that train.py configuration constants are valid."""

    @pytest.fixture(autouse=True)
    def mock_setup(self):
        mock_imports()

    def test_train_configs_has_required_phases(self):
        import train
        required_phases = ["Phase_1", "Phase_2", "Phase_3", "Phase_4"]
        for phase in required_phases:
            assert phase in train.TRAIN_CONFIGS

    def test_each_phase_has_weighted_datasets(self):
        import train
        for phase_name, datasets in train.TRAIN_CONFIGS.items():
            assert len(datasets) > 0
            weights = [d["weight"] for d in datasets]
            assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_curriculum_config_valid(self):
        import train
        assert "peak_lr" in train.CURRICULUM_CONFIG
        assert "end_lr" in train.CURRICULUM_CONFIG
        assert "phases" in train.CURRICULUM_CONFIG
        assert len(train.CURRICULUM_CONFIG["phases"]) == 4

    def test_phase_percentages_sum_to_one(self):
        import train
        total_pct = sum(p["pct"] for p in train.CURRICULUM_CONFIG["phases"])
        assert total_pct == pytest.approx(1.0, abs=0.01)

    def test_model_configs_valid_dimensions(self):
        import train
        assert train.MODEL_CONFIG["vocab_size"] > 0
        assert train.MODEL_CONFIG["dim"] > 0
        assert train.MODEL_CONFIG["n_layers"] > 0


class TestModelConfigModes:
    """Test model configuration modes."""

    def test_scout_mode_config(self):
        with patch.dict("train.__dict__", {"MODE": "scout"}):
            from importlib import reload
            mock_imports()
            import train
            reload(train)
            assert train.MODEL_CONFIG["dim"] == 512
            assert train.MODEL_CONFIG["n_layers"] == 24

    def test_upscaled_mode_config(self):
        with patch.dict("train.__dict__", {"MODE": "upscaled"}):
            from importlib import reload
            mock_imports()
            import train
            reload(train)
            assert train.MODEL_CONFIG["dim"] == 1024
            assert train.MODEL_CONFIG["n_layers"] == 64
            assert train.MODEL_CONFIG.get("use_attn") == True
            assert train.MODEL_CONFIG.get("attn_pct") == 0.08


class TestTrainImports:
    """Test that train.py imports work correctly."""

    @pytest.fixture(autouse=True)
    def mock_setup(self):
        mock_imports()

    def test_imports_are_valid(self):
        import train
        assert train.BitMambaLLM is not None
        assert train.chunked_cross_entropy is not None

    def test_train_constants_defined(self):
        import train
        assert train.BATCH_SIZE > 0
        assert train.GRAD_ACCUM_STEPS > 0
        assert train.SAVE_EVERY > 0
