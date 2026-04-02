import pytest
import torch
import os
from unittest.mock import MagicMock, patch


class TestSFTTrainingConstants:
    """Test sft_train.py constants."""

    def test_model_config_dimensions(self):
        from sft_train import MODEL_CONFIG
        assert MODEL_CONFIG["vocab_size"] > 0
        assert MODEL_CONFIG["dim"] > 0
        assert MODEL_CONFIG["n_layers"] > 0
        assert MODEL_CONFIG["d_state"] > 0
        assert MODEL_CONFIG["expand"] > 0
        assert "use_checkpoint" in MODEL_CONFIG

    def test_checkpoint_paths_defined(self):
        from sft_train import PRETRAINED_CKPT, CHECKPOINT_DIR
        assert "checkpoints" in PRETRAINED_CKPT
        assert "checkpoints" in CHECKPOINT_DIR

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


class TestRunSFTStageIntegration:
    """Tiny integration test: runs one full SFT forward/backward/checkpoint cycle on CPU."""

    @patch('sft_train.wandb')
    @patch('sft_train.barrier')
    @patch('sft_train.is_main_process', return_value=True)
    @patch('sft_train.get_world_size', return_value=1)
    @patch('sft_train.unwrap_model', side_effect=lambda m: m)
    @patch('sft_train.create_sft_dataloader')
    def test_run_sft_stage_completes_and_saves_checkpoint(
        self, mock_loader_factory, mock_unwrap, mock_world_size,
        mock_is_main, mock_barrier, mock_wandb, tmp_path, monkeypatch
    ):
        import sft_train
        from model import BitMambaLLM

        TINY_CFG = dict(vocab_size=64, dim=32, n_layers=1, d_state=16,
                        expand=2, use_checkpoint=False)
        SEQ_LEN = 16
        TINY_BATCH = 2

        monkeypatch.setattr(sft_train, 'DEVICE', 'cpu')
        monkeypatch.setattr(sft_train, 'CHECKPOINT_DIR', str(tmp_path))
        monkeypatch.setattr(sft_train, 'BATCH_SIZE', TINY_BATCH)
        monkeypatch.setattr(sft_train, 'GRAD_ACCUM_STEPS', 1)
        monkeypatch.setattr(sft_train, 'SAFE_DIVISOR', float(16384))

        x = torch.randint(0, 64, (TINY_BATCH, SEQ_LEN))
        y = torch.randint(0, 64, (TINY_BATCH, SEQ_LEN))
        y[0, :4] = -100
        dummy_batches = [(x, y)]

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(dummy_batches))
        mock_loader.__len__ = MagicMock(return_value=len(dummy_batches))
        mock_loader.dataset = MagicMock()
        mock_loader.dataset.__len__ = MagicMock(return_value=TINY_BATCH)
        mock_loader.sampler = MagicMock(spec=[])
        mock_loader_factory.return_value = mock_loader

        model = BitMambaLLM(**TINY_CFG)

        stage_cfg = {
            "name": "test_stage",
            "lr": 1e-4,
            "epochs": 1,
            "max_seq_len": SEQ_LEN,
            "reasoning_off_prob": 0.0,
            "paths": [],
        }

        result_step = sft_train.run_sft_stage(
            model, tokenizer=MagicMock(), stage_cfg=stage_cfg,
            stage_num=1, global_step=0
        )

        assert result_step == 1

        ckpt_files = list(tmp_path.glob("*.pt"))
        assert len(ckpt_files) == 1, f"Expected 1 checkpoint, found: {ckpt_files}"

        ckpt = torch.load(ckpt_files[0], map_location='cpu', weights_only=True)
        assert 'model_state_dict' in ckpt
        assert 'stage' in ckpt
        assert 'epoch' in ckpt


