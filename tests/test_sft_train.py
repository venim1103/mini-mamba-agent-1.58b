import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock heavy dependencies
mock_modules = {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'torch.nn.functional': MagicMock(),
    'torch.optim': MagicMock(),
    'torch.optim.lr_scheduler': MagicMock(),
    'wandb': MagicMock(),
    'transformers': MagicMock(),
    'transformers.AutoTokenizer': MagicMock(),
    'model': MagicMock(),
    'model.BitMambaLLM': MagicMock(),
    'model.chunked_cross_entropy': MagicMock(),
    'model.maybe_autocast': MagicMock(),
    'optim': MagicMock(),
    'optim.setup_mamba_optimizers': MagicMock(),
    'optim.FGWSD_Scheduler': MagicMock(),
    'sft_data': MagicMock(),
    'sft_data.SFT_STAGES': MagicMock(),
    'sft_data.create_sft_dataloader': MagicMock(),
    'dist_utils': MagicMock(),
    'dist_utils.setup_distributed': MagicMock(),
    'dist_utils.cleanup_distributed': MagicMock(),
    'dist_utils.is_main_process': MagicMock(),
    'dist_utils.wrap_model_ddp': MagicMock(),
    'dist_utils.unwrap_model': MagicMock(),
    'dist_utils.barrier': MagicMock(),
}
for name, obj in mock_modules.items():
    if name not in sys.modules:
        sys.modules[name] = obj


class TestSFTTrainingConstants:
    """Test sft_train.py constants."""

    def test_model_config_dimensions(self):
        from sft_train import MODEL_CONFIG
        assert MODEL_CONFIG["vocab_size"] == 64000
        assert MODEL_CONFIG["dim"] == 1024
        assert MODEL_CONFIG["n_layers"] == 40
        assert MODEL_CONFIG["d_state"] == 128
        assert MODEL_CONFIG["expand"] == 2
        assert MODEL_CONFIG["use_checkpoint"] == True

    def test_hyperparameters(self):
        from sft_train import BATCH_SIZE, GRAD_ACCUM_STEPS
        assert BATCH_SIZE == 2
        assert GRAD_ACCUM_STEPS == 8

    def test_checkpoint_paths_defined(self):
        from sft_train import PRETRAINED_CKPT, CHECKPOINT_DIR
        assert "checkpoints" in PRETRAINED_CKPT
        assert CHECKPOINT_DIR == "checkpoints/sft"

    def test_device_is_cuda_or_cpu(self):
        from sft_train import DEVICE
        assert DEVICE in ["cuda", "cpu"]


class TestSFTTrainingImports:
    """Test sft_train.py imports."""

    def test_imports_are_valid(self):
        from sft_train import (
            BitMambaLLM,
            chunked_cross_entropy,
            maybe_autocast,
            setup_mamba_optimizers,
            create_sft_dataloader,
        )
        from sft_data import SFT_STAGES
        assert BitMambaLLM is not None
        assert chunked_cross_entropy is not None

    def test_sft_stages_defined(self):
        from sft_data import SFT_STAGES
        assert isinstance(SFT_STAGES, list)
        assert len(SFT_STAGES) > 0


class TestSFTStageConfig:
    """Test SFT stage configuration."""

    def test_each_stage_has_required_fields(self):
        from sft_data import SFT_STAGES
        for stage in SFT_STAGES:
            assert "name" in stage
            assert "lr" in stage
            assert "epochs" in stage
            assert "max_seq_len" in stage
            assert "reasoning_off_prob" in stage

    def test_stage_paths_are_lists(self):
        from sft_data import SFT_STAGES
        for stage in SFT_STAGES:
            assert isinstance(stage["paths"], list)

    def test_stage_config_valid(self):
        from sft_data import SFT_STAGES
        for stage in SFT_STAGES:
            for path_cfg in stage["paths"]:
                assert "path" in path_cfg
                assert "format" in path_cfg


class TestSFTGradAccumLogic:
    """Test gradient accumulation logic."""

    def test_tail_steps_calculation(self):
        from sft_train import GRAD_ACCUM_STEPS
        
        n_batches = 17
        tail_steps = n_batches % GRAD_ACCUM_STEPS
        
        assert tail_steps == 1

    def test_full_accumulation(self):
        from sft_train import GRAD_ACCUM_STEPS
        
        n_batches = 16
        tail_steps = n_batches % GRAD_ACCUM_STEPS
        
        assert tail_steps == 0