import pytest
from unittest.mock import MagicMock, patch


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