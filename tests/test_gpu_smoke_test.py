import pytest
from unittest.mock import MagicMock, patch


class TestGPUSmokeTestMain:
    """Test gpu_smoke_test.py main function."""

    def test_cuda_check_failure(self):
        with patch("gpu_smoke_test.torch.cuda.is_available", return_value=False), \
             patch("sys.exit") as MockExit:
            
            import gpu_smoke_test
            
            gpu_smoke_test.main()
            
            MockExit.assert_called_once_with(1)

    def test_model_forward_pass(self):
        with patch("gpu_smoke_test.torch.cuda.is_available", return_value=True), \
             patch("gpu_smoke_test.torch.cuda.get_device_name", return_value="RTX 4090"), \
             patch("gpu_smoke_test.BitMambaLLM") as MockModel, \
             patch("gpu_smoke_test.torch.no_grad"), \
             patch("gpu_smoke_test.torch.randint") as MockRandint:
            
            mock_model = MagicMock()
            mock_model.return_value = MagicMock()
            mock_model.return_value.shape = (1, 32, 64000)
            MockModel.return_value = mock_model
            
            MockRandint.return_value = MagicMock()
            
            import gpu_smoke_test
            
            gpu_smoke_test.main()

    def test_prints_device_name(self, capsys):
        with patch("gpu_smoke_test.torch.cuda.is_available", return_value=True), \
             patch("gpu_smoke_test.torch.cuda.get_device_name", return_value="RTX 4090"), \
             patch("gpu_smoke_test.BitMambaLLM") as MockModel, \
             patch("gpu_smoke_test.torch.no_grad"), \
             patch("gpu_smoke_test.torch.randint") as MockRandint, \
             patch("gpu_smoke_test.print") as MockPrint:
            
            mock_model = MagicMock()
            mock_model.return_value = MagicMock()
            mock_model.return_value.shape = (1, 32, 64000)
            mock_model.return_value.device = "cuda:0"
            MockModel.return_value = mock_model
            
            MockRandint.return_value = MagicMock()
            
            import gpu_smoke_test
            
            gpu_smoke_test.main()
            
            MockPrint.assert_any_call("Using GPU: RTX 4090")


class TestGPUSmokeTestImports:
    """Test gpu_smoke_test.py imports."""

    def test_imports_bitmamba_llm(self):
        from gpu_smoke_test import BitMambaLLM
        assert BitMambaLLM is not None

    def test_imports_torch(self):
        import gpu_smoke_test
        assert hasattr(gpu_smoke_test, "torch")
