import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch


class TestRunSFTStage:
    """Test run_sft_stage function in sft_train.py."""

    @pytest.fixture
    def mock_dependencies(self):
        with patch("sft_train.unwrap_model") as MockUnwrap, \
             patch("sft_train.create_sft_dataloader") as MockDL, \
             patch("sft_train.setup_mamba_optimizers") as MockOpt, \
             patch("sft_train.is_main_process") as MockMain, \
             patch("sft_train.barrier"):
            
            mock_model = MagicMock()
            MockUnwrap.return_value = mock_model
            
            mock_loader = MagicMock()
            mock_loader.__len__ = MagicMock(return_value=10)
            mock_loader.dataset = MagicMock(__len__=100)
            MockDL.return_value = mock_loader
            
            MockOpt.return_value = (MagicMock(), MagicMock(), MagicMock())
            
            MockMain.return_value = True
            
            yield MockUnwrap, MockDL, MockOpt, MockMain, mock_model


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

    def test_stage_weights_valid(self):
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


class TestSFTModelLoading:
    """Test model loading in main function."""

    def test_loads_pretrained_checkpoint(self):
        with patch("sft_train.setup_distributed") as MockSetup, \
             patch("sft_train.AutoTokenizer") as MockTok, \
             patch("sft_train.BitMambaLLM") as MockModel, \
             patch("sft_train.wrap_model_ddp") as MockWrap, \
             patch("sft_train.unwrap_model") as MockUnwrap, \
             patch("sft_train.cleanup_distributed"):
            
            MockSetup.return_value = (0, 0, 1, "cpu")
            
            mock_tok = MagicMock()
            MockTok.from_pretrained.return_value = mock_tok
            
            mock_model = MagicMock()
            MockModel.return_value = mock_model
            
            MockWrap.return_value = mock_model
            MockUnwrap.return_value = mock_model
            
            with patch("torch.load") as MockLoad:
                MockLoad.return_value = {'model_state_dict': {}}
                
                import sft_train
                
            MockLoad.assert_called()
