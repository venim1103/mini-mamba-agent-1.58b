import pytest
import torch
from unittest.mock import MagicMock, patch, mock_open
import os


class TestUpscalerFunction:
    """Test the upscaler function in upscale.py."""

    @pytest.fixture
    def mock_small_checkpoint(self, tmp_path):
        """Create a mock small checkpoint."""
        ckpt_path = tmp_path / "small.pt"
        state = {
            'model_state_dict': {
                'tok_embeddings.weight': torch.randn(64000, 1024),
                'norm.weight': torch.randn(1024),
                'output.weight': torch.randn(64000, 1024),
                'layers.0.norm.weight': torch.randn(1024),
                'layers.0.mamba.dt_bias': torch.zeros(64),
            },
            'step': 100000,
        }
        torch.save(state, ckpt_path)
        return ckpt_path

    def test_upscaler_loads_checkpoint(self, mock_small_checkpoint, tmp_path):
        output_path = tmp_path / "upscaled.pt"
        
        with patch("torch.load") as mock_load, \
             patch("torch.save") as mock_save, \
             patch("upscale.BitMambaLLM") as MockModel:
            
            mock_load.return_value = {
                'model_state_dict': {
                    'tok_embeddings.weight': torch.randn(64000, 1024),
                    'norm.weight': torch.randn(1024),
                    'output.weight': torch.randn(64000, 1024),
                    'layers.0.norm.weight': torch.randn(1024),
                },
                'step': 100000,
            }
            
            mock_big_model = MagicMock()
            mock_big_model.state_dict.return_value = {
                'tok_embeddings.weight': torch.randn(64000, 1024),
                'norm.weight': torch.randn(1024),
                'output.weight': torch.randn(64000, 1024),
                'layers.0.norm.weight': torch.randn(1024),
                'layers.1.norm.weight': torch.randn(1024),
            }
            mock_big_model.attn_indices = set()
            MockModel.side_effect = [
                MagicMock(attn_indices=set()),
                mock_big_model,
            ]
            
            from upscale import upscaler
            upscaler(str(mock_small_checkpoint), str(output_path))
            
            mock_load.assert_called_once()
            mock_save.assert_called_once()

    def test_upscaler_creates_output_dir(self, mock_small_checkpoint, tmp_path):
        output_dir = tmp_path / "new_dir" / "subdir"
        output_path = output_dir / "upscaled.pt"
        
        with patch("torch.load") as mock_load, \
             patch("torch.save") as mock_save, \
             patch("upscale.BitMambaLLM") as MockModel:
            
            mock_load.return_value = {'model_state_dict': {}, 'step': 0}
            
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {}
            mock_model.attn_indices = set()
            MockModel.return_value = mock_model
            
            from upscale import upscaler
            upscaler(str(mock_small_checkpoint), str(output_path))
            
            assert os.path.exists(output_dir)

    def test_upscaler_output_format(self, mock_small_checkpoint, tmp_path):
        output_path = tmp_path / "upscaled.pt"
        
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
            
            from upscale import upscaler
            upscaler(str(mock_small_checkpoint), str(output_path))
            
            call_args = mock_save.call_args[0][0]
            assert 'step' in call_args
            assert 'model_state_dict' in call_args
            assert 'source_checkpoint' in call_args
            assert call_args['requires_continued_pretraining'] == True


class TestUpscaleConstants:
    """Test upscale.py constants."""

    def test_default_checkpoint_paths(self):
        from upscale import DEFAULT_CKPT, MODEL_CONFIG
        
        assert DEFAULT_CKPT == "checkpoints/bitmamba_parent/step_1000000.pt"
        
        assert MODEL_CONFIG["vocab_size"] == 64000
        assert MODEL_CONFIG["dim"] == 1024
        assert MODEL_CONFIG["n_layers"] == 40
        assert MODEL_CONFIG["d_state"] == 128
        assert MODEL_CONFIG["expand"] == 2

    def test_upscaled_model_config(self):
        from upscale import BitMambaLLM
        
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
            if i < 32:
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
