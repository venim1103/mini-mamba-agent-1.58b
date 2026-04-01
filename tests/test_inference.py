import pytest
from unittest.mock import MagicMock, patch


class TestInferenceConstants:
    """Test inference.py constants."""

    def test_scout_mode_config(self):
        with patch.dict("inference.__dict__", {"MODE": "scout"}):
            from importlib import reload
            import inference
            reload(inference)
            
            assert inference.MODEL_CONFIG["vocab_size"] == 64000
            assert inference.MODEL_CONFIG["dim"] == 512
            assert inference.MODEL_CONFIG["n_layers"] == 24
            assert inference.MODEL_CONFIG["d_state"] == 64
            assert inference.MODEL_CONFIG["expand"] == 2

    def test_parent_mode_config(self):
        with patch.dict("inference.__dict__", {"MODE": "parent"}):
            from importlib import reload
            import inference
            reload(inference)
            
            assert inference.MODEL_CONFIG["vocab_size"] == 64000
            assert inference.MODEL_CONFIG["dim"] == 1024
            assert inference.MODEL_CONFIG["n_layers"] == 40
            assert inference.MODEL_CONFIG["d_state"] == 128

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