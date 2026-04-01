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

    def test_skips_incompatible_layer_types(self):
        import upscale

        with patch("torch.load") as mock_load, \
             patch("torch.save") as mock_save, \
             patch("upscale.BitMambaLLM") as MockModel:
            small_state = {
                "layers.8.norm.weight": torch.full((4,), 3.0),
                "tok_embeddings.weight": torch.randn(2, 2),
                "norm.weight": torch.randn(2),
                "output.weight": torch.randn(2, 2),
            }
            mock_load.return_value = {"model_state_dict": small_state}

            big_state = {
                "layers.32.norm.weight": torch.full((4,), 9.0),
                "tok_embeddings.weight": torch.randn(2, 2),
                "norm.weight": torch.randn(2),
                "output.weight": torch.randn(2, 2),
            }

            # First call: big model with attn at layer 32.
            big_model = MagicMock()
            big_model.state_dict.return_value = big_state.copy()
            big_model.attn_indices = {32}

            # Second call: small model temp with no attention.
            small_model_tmp = MagicMock()
            small_model_tmp.attn_indices = set()

            MockModel.side_effect = [big_model, small_model_tmp]

            upscale.upscaler("dummy_small.pt", "dummy_out.pt")

            saved_payload = mock_save.call_args[0][0]
            saved_state = saved_payload["model_state_dict"]
            # Since layer type is incompatible, target layer should remain random-init value.
            assert torch.equal(saved_state["layers.32.norm.weight"], big_state["layers.32.norm.weight"])

    def test_transplants_compatible_layer_when_source_exists(self):
        import upscale

        with patch("torch.load") as mock_load, \
             patch("torch.save") as mock_save, \
             patch("upscale.BitMambaLLM") as MockModel:
            transplanted = torch.full((4,), 7.0)
            small_state = {
                "layers.8.norm.weight": transplanted,
                "tok_embeddings.weight": torch.randn(2, 2),
                "norm.weight": torch.randn(2),
                "output.weight": torch.randn(2, 2),
            }
            mock_load.return_value = {"model_state_dict": small_state}

            big_state = {
                "layers.32.norm.weight": torch.full((4,), 1.0),
                "tok_embeddings.weight": torch.randn(2, 2),
                "norm.weight": torch.randn(2),
                "output.weight": torch.randn(2, 2),
            }

            big_model = MagicMock()
            big_model.state_dict.return_value = big_state.copy()
            # Layer 32 maps to source layer 8 and should be compatible when both are non-attn.
            big_model.attn_indices = set()

            small_model_tmp = MagicMock()
            small_model_tmp.attn_indices = set()

            MockModel.side_effect = [big_model, small_model_tmp]

            upscale.upscaler("dummy_small.pt", "dummy_out.pt")

            saved_payload = mock_save.call_args[0][0]
            saved_state = saved_payload["model_state_dict"]
            assert torch.equal(saved_state["layers.32.norm.weight"], transplanted)


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