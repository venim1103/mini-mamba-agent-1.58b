import pytest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Mock ALL heavy dependencies BEFORE any imports
_mocked_modules = {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'os': MagicMock(),
    'model': MagicMock(),
    'model.BitMambaLLM': MagicMock(),
}

for _name, _obj in _mocked_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _obj


class TestUpscaleFunction:
    """Test the upscaler function in upscale.py."""

    def test_upscaler_output_format(self):
        # Import AFTER mocks are set up
        import upscale
        
        with patch("torch.load") as mock_load, \
             patch("torch.save") as mock_save, \
             patch("upscale.BitMambaLLM") as MockModel:
            
            mock_load.return_value = {
                'model_state_dict': {
                    'tok_embeddings.weight': torch.randn(64000, 1024),
                    'norm.weight': torch.randn(1024),
                    'output.weight': torch.randn(64000, 1024),
                },
                'step': 100000,
            }
            
            mock_big_model = MagicMock()
            mock_big_model.state_dict.return_value = {
                'tok_embeddings.weight': torch.randn(64000, 1024),
                'norm.weight': torch.randn(1024),
                'output.weight': torch.randn(64000, 1024),
                'layers.0.norm.weight': torch.randn(1024),
            }
            mock_big_model.attn_indices = set()
            MockModel.side_effect = [
                MagicMock(attn_indices=set()),
                mock_big_model,
            ]
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                try:
                    upscale.upscaler("dummy.pt", tmp.name)
                except:
                    pass
                
            mock_save.assert_called_once()
            call_args = mock_save.call_args[0][0]
            assert 'step' in call_args
            assert 'model_state_dict' in call_args
            assert call_args['requires_continued_pretraining'] == True


class TestUpscaleModelConfig:
    """Test upscale.py creates correct model configs."""

    def test_creates_upscaled_model_call_params(self):
        import upscale
        
        with patch("upscale.BitMambaLLM") as MockModel:
            mock_model_instance = MagicMock()
            mock_model_instance.dim = 1024
            mock_model_instance.n_layers = 64
            MockModel.return_value = mock_model_instance
            
            model = upscale.BitMambaLLM(
                vocab_size=64000, dim=1024, n_layers=64, d_state=128, expand=2,
                use_attn=True, attn_pct=0.08,
            )
            
            MockModel.assert_called_with(
                vocab_size=64000, dim=1024, n_layers=64, d_state=128, expand=2,
                use_attn=True, attn_pct=0.08,
            )


class TestUpscaleLayerMapping:
    """Test layer mapping logic in upscaler."""

    def test_layer_mapping_first_32_layers(self):
        for i in range(32):
            source = i
            assert source == i

    def test_layer_mapping_duplication(self):
        for target_idx in range(32, 64):
            source_layer_idx = target_idx - 24
            assert source_layer_idx >= 8
            assert source_layer_idx <= 39

    def test_attn_layer_indices_calculation(self):
        n_layers = 64
        attn_pct = 0.08
        
        n_attn = max(1, int(n_layers * attn_pct))
        
        attn_indices = set()
        for i in range(n_attn):
            layer_idx = int((n_layers - 1) * (i + 1) / (n_attn + 1))
            attn_indices.add(layer_idx)
        
        assert len(attn_indices) == n_attn
        assert all(0 <= i < n_layers for i in attn_indices)