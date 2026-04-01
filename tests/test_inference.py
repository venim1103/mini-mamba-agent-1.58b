import pytest
from unittest.mock import MagicMock, patch


class TestInferenceConstants:
    """Test inference.py constants."""

    def test_scout_mode_config(self):
        import inference

        config, ckpt = inference.resolve_model_settings("scout")
        assert config["vocab_size"] == 64000
        assert config["dim"] == 512
        assert config["n_layers"] == 24
        assert config["d_state"] == 64
        assert config["expand"] == 2
        assert "bitmamba_scout" in ckpt

    def test_parent_mode_config(self):
        import inference

        config, ckpt = inference.resolve_model_settings("parent")
        assert config["vocab_size"] == 64000
        assert config["dim"] == 1024
        assert config["n_layers"] == 40
        assert config["d_state"] == 128
        assert "bitmamba_parent" in ckpt

    def test_invalid_mode_raises(self):
        import inference

        with pytest.raises(ValueError, match="Unsupported MODE"):
            inference.resolve_model_settings("invalid")

    def test_checkpoint_paths_defined(self):
        from inference import CKPT_PATH, MODE
        
        assert MODE in ["scout", "parent"]
        assert isinstance(CKPT_PATH, str)
        assert "checkpoints" in CKPT_PATH

    def test_device_is_cuda_or_cpu(self):
        from inference import DEVICE
        assert DEVICE in ["cuda", "cpu"]


class TestInferenceImports:
    """Test that inference.py imports work correctly."""

    def test_imports_are_valid(self):
        from inference import (
            BitMambaLLM,
            maybe_autocast,
        )
        assert BitMambaLLM is not None
        assert maybe_autocast is not None