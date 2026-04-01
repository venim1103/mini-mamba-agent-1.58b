import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock heavy dependencies
mock_modules = {
    'torch': MagicMock(),
    'torch.cuda': MagicMock(),
    'model': MagicMock(),
    'model.BitMambaLLM': MagicMock(),
}
for name, obj in mock_modules.items():
    if name not in sys.modules:
        sys.modules[name] = obj


class TestGPUSmokeTestImports:
    """Test gpu_smoke_test.py imports."""

    def test_imports_bitmamba_llm(self):
        from gpu_smoke_test import BitMambaLLM
        assert BitMambaLLM is not None

    def test_imports_torch(self):
        import gpu_smoke_test
        assert hasattr(gpu_smoke_test, "torch")

    def test_main_function_exists(self):
        import gpu_smoke_test
        assert hasattr(gpu_smoke_test, "main")
        assert callable(gpu_smoke_test.main)