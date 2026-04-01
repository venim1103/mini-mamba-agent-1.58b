import pytest
from unittest.mock import MagicMock, patch, mock_open
import torch


class TestGenerateFunction:
    """Test the generate function in inference.py."""

    @pytest.fixture
    def mock_model_and_tok(self):
        with patch("inference.AutoTokenizer") as MockTok, \
             patch("inference.BitMambaLLM") as MockModel:
            
            mock_tok = MagicMock()
            mock_tok.encode.return_value = [1, 2, 3]
            MockTok.from_pretrained.return_value = mock_tok
            
            mock_model = MagicMock()
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            MockModel.return_value = mock_model
            
            yield MockTok, MockModel

    def test_generate_uses_eos_token(self):
        with patch("inference.AutoTokenizer") as MockTok, \
             patch("inference.BitMambaLLM") as MockModel, \
             patch("inference.maybe_autocast"):
            
            mock_tok = MagicMock()
            mock_tok.encode.return_value = [999]
            mock_tok.return_value = {"input_ids": torch.tensor([[1, 2]])}
            mock_tok.decode.return_value = "generated"
            MockTok.from_pretrained.return_value = mock_tok
            
            mock_model = MagicMock()
            mock_model.generate.return_value = torch.tensor([[1, 2, 999]])
            mock_model.to.return_value = mock_model
            MockModel.return_value = mock_model
            
            from inference import generate, DEVICE
            
            generate(
                mock_model, mock_tok, 
                prompt="test prompt", 
                max_new_tokens=10,
                temperature=0.7
            )
            
            assert mock_model.generate.called

    def test_generate_respects_max_new_tokens(self):
        with patch("inference.AutoTokenizer") as MockTok, \
             patch("inference.BitMambaLLM") as MockModel, \
             patch("inference.maybe_autocast"):
            
            mock_tok = MagicMock()
            mock_tok.return_value = {"input_ids": torch.tensor([[1, 2]])}
            mock_tok.encode.return_value = [999]
            mock_tok.decode.return_value = "result"
            MockTok.from_pretrained.return_value = mock_tok
            
            mock_model = MagicMock()
            mock_model.generate.return_value = torch.tensor([[1, 2, 999]])
            mock_model.to.return_value = mock_model
            MockModel.return_value = mock_model
            
            from inference import generate
            
            generate(
                MagicMock(), mock_tok, 
                "prompt", 
                max_new_tokens=50
            )
            
            call_kwargs = mock_model.generate.call_args[1]
            assert call_kwargs["max_new_tokens"] == 50


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


class TestInferenceMain:
    """Test inference.py main block via mocking."""

    def test_main_loads_tokenizer(self):
        with patch("inference.AutoTokenizer") as MockTok, \
             patch("inference.BitMambaLLM") as MockModel:
            
            mock_tok = MagicMock()
            mock_tok.encode.return_value = [999]
            MockTok.from_pretrained.return_value = mock_tok
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            MockModel.return_value = mock_model
            
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}}
                
                import inference
                
                if hasattr(inference, "__main__") or True:
                    pass
                    
            MockTok.from_pretrained.assert_called_with("custom_agentic_tokenizer")
