from types import SimpleNamespace
from unittest.mock import MagicMock

import gpu_smoke_test


def test_imports_bitmamba_llm():
    assert hasattr(gpu_smoke_test, "BitMambaLLM")


def test_imports_torch():
    assert hasattr(gpu_smoke_test, "torch")


def test_main_function_exists():
    assert callable(gpu_smoke_test.main)


def test_main_returns_1_when_cuda_unavailable(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.0.0"
    mock_torch.version = SimpleNamespace(cuda="12.1")
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setattr(gpu_smoke_test, "torch", mock_torch)

    assert gpu_smoke_test.main() == 1


def test_model_created_with_correct_params(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.0.0"
    mock_torch.version = SimpleNamespace(cuda="12.1")
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "Test GPU"

    x = MagicMock()
    x.shape = (1, 32)
    mock_torch.randint.return_value = x

    no_grad_ctx = MagicMock()
    no_grad_ctx.__enter__.return_value = None
    no_grad_ctx.__exit__.return_value = None
    mock_torch.no_grad.return_value = no_grad_ctx

    y = MagicMock()
    y.shape = (1, 32, 64000)
    y.device = "cuda:0"

    model_instance = MagicMock()
    model_instance.to.return_value = model_instance
    model_instance.return_value = y

    model_cls = MagicMock(return_value=model_instance)

    monkeypatch.setattr(gpu_smoke_test, "torch", mock_torch)
    monkeypatch.setattr(gpu_smoke_test, "BitMambaLLM", model_cls)

    assert gpu_smoke_test.main() == 0

    model_cls.assert_called_once_with(
        vocab_size=64000,
        dim=512,
        n_layers=24,
        d_state=64,
        expand=2,
        use_attn=True,
    )
