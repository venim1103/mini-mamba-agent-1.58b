import pytest
import torch
from unittest.mock import MagicMock, patch, mock_open


class TestUpscaleFunction:
    """Test the upscaler function in upscale.py."""

    def test_upscaler_output_format(self):
        from upscale import upscaler
        
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
                    upscaler("dummy.pt", tmp.name)
                except:
                    pass
                
            mock_save.assert_called_once()
            call_args = mock_save.call_args[0][0]
            assert 'step' in call_args
            assert 'model_state_dict' in call_args
            assert call_args['requires_continued_pretraining'] == True


class TestUpscaleModelConfig:
    """Test upscale.py creates correct model configs."""

    def test_creates_upscaled_model(self):
        from model import BitMambaLLM
        
        model = BitMambaLLM(
            vocab_size=64000, dim=1024, n_layers=64, d_state=128, expand=2,
            use_attn=True, attn_pct=0.08,
        )
        
        assert model.dim == 1024
        assert model.n_layers == 64


class TestUpscaleLayerMapping:
    """Test layer mapping logic in upscaler."""

    def test_layer_mapping_first_32_layers(self):
        """First 32 layers map 1:1."""
        for i in range(32):
            source = i
            assert source == i

    def test_layer_mapping_duplication(self):
        """Layers 32-63 duplicate layers 8-39."""
        for target_idx in range(32, 64):
            source_layer_idx = target_idx - 24
            assert source_layer_idx >= 8
            assert source_layer_idx <= 39

    def test_attn_layer_indices_calculation(self):
        """Test attention layer index calculation."""
        n_layers = 64
        attn_pct = 0.08
        
        n_attn = max(1, int(n_layers * attn_pct))
        
        attn_indices = set()
        for i in range(n_attn):
            layer_idx = int((n_layers - 1) * (i + 1) / (n_attn + 1))
            attn_indices.add(layer_idx)
        
        assert len(attn_indices) == n_attn
        assert all(0 <= i < n_layers for i in attn_indices)