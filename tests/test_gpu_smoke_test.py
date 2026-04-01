import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock ALL heavy dependencies BEFORE any imports
_mocked_modules = {
    'torch': MagicMock(),
    'torch.cuda': MagicMock(),
    'model': MagicMock(),
    'model.BitMambaLLM': MagicMock(),
}

for _name, _obj in _mocked_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _obj


class TestGPUSmokeTestImports:
    """Test gpu_smoke_test.py imports."""

    def test_imports_bitmamba_llm(self):
        import gpu_smoke_test
        assert hasattr(gpu_smoke_test, "BitMambaLLM")

    def test_imports_torch(self):
        import gpu_smoke_test
        assert hasattr(gpu_smoke_test, "torch")

    def test_main_function_exists(self):
        import gpu_smoke_test
        assert hasattr(gpu_smoke_test, "main")
        assert callable(gpu_smoke_test.main)

    def test_cuda_check_uses_mock(self):
        import gpu_smoke_test
        
        with patch("gpu_smoke_test.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            
            with patch("sys.exit") as mock_exit:
                gpu_smoke_test.main()
                mock_exit.assert_called_once_with(1)

    def test_model_created_with_correct_params(self):
        import gpu_smoke_test
        
        with patch("gpu_smoke_test.torch") as mock_torch, \
             patch("gpu_smoke_test.BitMambaLLM") as MockModel:
            
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.randint.return_value = MagicMock()
            mock_torch.no_grad.return_value = MagicMock()
            
            mock_model = MagicMock()
            mock_model.return_value = MagicMock()
            mock_model.return_value.shape = (1, 32, 64000)
            MockModel.return_value = mock_model
            
            gpu_smoke_test.main()
            
            MockModel.assert_called_with(
                vocab_size=64000,
                dim=512,
                n_layers=24,
                d_state=64,
                expand=2,
                use_attn=True,
            )