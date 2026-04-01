import pytest
import torch
import sys
from unittest.mock import MagicMock

# Mock ALL heavy dependencies BEFORE any imports
_mocked_modules = {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'torch.nn.functional': MagicMock(),
    'torch.cuda': MagicMock(),
    'torch.optim': MagicMock(),
    'torch.optim.lr_scheduler': MagicMock(),
    'torch.amp': MagicMock(),
    'torch.amp.GradScaler': MagicMock(),
    'torch.utils': MagicMock(),
    'torch.utils.data': MagicMock(),
    'torch.utils.data.distributed': MagicMock(),
    'model': MagicMock(),
    'model.BitMambaLLM': MagicMock(),
    'model.chunked_cross_entropy': MagicMock(),
    'model.maybe_autocast': MagicMock(),
    'model.RMSNorm': MagicMock(),
    'model.BitLinear': MagicMock(),
    'model.weight_quant': MagicMock(),
    'model.activation_quant': MagicMock(),
    'optim': MagicMock(),
    'optim.Muon': MagicMock(),
    'optim.setup_mamba_optimizers': MagicMock(),
    'optim.FGWSD_Scheduler': MagicMock(),
    'dist_utils': MagicMock(),
    'dist_utils.setup_distributed': MagicMock(),
    'dist_utils.cleanup_distributed': MagicMock(),
    'dist_utils.is_main_process': MagicMock(),
    'dist_utils.wrap_model_ddp': MagicMock(),
    'dist_utils.unwrap_model': MagicMock(),
    'dist_utils.barrier': MagicMock(),
    'dist_utils.get_world_size': MagicMock(),
    'wandb': MagicMock(),
    'transformers': MagicMock(),
    'transformers.AutoTokenizer': MagicMock(),
    'datasets': MagicMock(),
    'datasets.load_dataset': MagicMock(),
}

for _name, _obj in _mocked_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _obj


class TestCreateSeqIdxBatch:
    """Test train.py's create_seq_idx_batch helper function."""

    def test_single_segment_all_zeros(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert seq_idx[0].unique().item() == 0

    def test_two_segments(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (1, 8)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()

    def test_three_segments(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 2, 5, 8]], dtype=torch.int32)
        n_segs = torch.tensor([4])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        expected = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.int32, device=seq_idx.device)
        assert torch.equal(seq_idx, expected)

    def test_batch_size_two(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 4, 8, -1], [0, 2, 8, -1]], dtype=torch.int32)
        n_segs = torch.tensor([2, 3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 8)
        assert seq_idx.shape == (2, 8)
        assert (seq_idx[0, :4] == 0).all()
        assert (seq_idx[0, 4:] == 1).all()
        assert (seq_idx[1, :2] == 0).all()
        assert (seq_idx[1, 2:] == 1).all()

    def test_truncation_handling(self):
        from train import create_seq_idx_batch
        cu_seqlens = torch.tensor([[0, 3, 8, -1, -1]], dtype=torch.int32)
        n_segs = torch.tensor([3])
        seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, 6)
        assert seq_idx.shape == (1, 6)
        assert (seq_idx[0, :3] == 0).all()
        assert (seq_idx[0, 3:] == 1).all()


class TestTrainConfigs:
    """Test train.py configuration constants."""

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

    def test_batch_size_and_accum_steps(self):
        from train import BATCH_SIZE, GRAD_ACCUM_STEPS, SAVE_EVERY
        assert BATCH_SIZE == 2
        assert GRAD_ACCUM_STEPS == 8
        assert SAVE_EVERY == 5000

    def test_checkpoint_dir_defined(self):
        from train import CHECKPOINT_DIR
        assert "checkpoints" in CHECKPOINT_DIR