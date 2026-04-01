import pytest
import torch
from unittest.mock import MagicMock, patch


class TestInferenceConstants:
    """Test inference.py constants."""

    def test_scout_mode_config(self):
        import inference

        config, ckpt = inference.resolve_model_settings("scout")
        assert config["vocab_size"] == 64000
        assert config["dim"] == 512
        assert config["n_layers"] == 24
        assert config["d_state"] == 64
        assert config["expand"] == 2
        assert "bitmamba_scout" in ckpt

    def test_parent_mode_config(self):
        import inference

        config, ckpt = inference.resolve_model_settings("parent")
        assert config["vocab_size"] == 64000
        assert config["dim"] == 1024
        assert config["n_layers"] == 40
        assert config["d_state"] == 128
        assert "bitmamba_parent" in ckpt

    def test_invalid_mode_raises(self):
        import inference

        with pytest.raises(ValueError, match="Unsupported MODE"):
            inference.resolve_model_settings("invalid")

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


class TestInferenceRuntime:
    def test_generate_stops_on_eos(self):
        import inference

        class _Tok:
            def __call__(self, prompt, return_tensors="pt"):
                return type("TokOut", (), {"input_ids": torch.tensor([[5, 6]])})

            def encode(self, text, add_special_tokens=False):
                return [99]

            def decode(self, ids, skip_special_tokens=False):
                return "decoded"

        model = MagicMock()
        model.generate.return_value = torch.tensor([[5, 6, 99]])

        with patch("builtins.print"):
            inference.generate(model, _Tok(), prompt="hello", max_new_tokens=8, temperature=0.0)

        model.eval.assert_called_once()
        model.generate.assert_called_once()
        kwargs = model.generate.call_args.kwargs
        assert kwargs["eos_token_id"] == 99
        assert kwargs["max_new_tokens"] == 8

    def test_main_success_path_prepares_model_for_inference(self):
        import inference

        fake_state = {"model_state_dict": {"w": torch.tensor([1.0])}}
        model_instance = MagicMock()
        model_instance.to.return_value = model_instance

        with patch.object(inference, "AutoTokenizer") as tok_cls, \
             patch.object(inference, "BitMambaLLM", return_value=model_instance), \
             patch.object(inference.torch, "load", return_value=fake_state), \
             patch.object(inference, "generate") as gen_fn, \
             patch("builtins.print"):
            tok_cls.from_pretrained.return_value = MagicMock()
            rc = inference.main()

        assert rc == 0
        model_instance.load_state_dict.assert_called_once_with(fake_state["model_state_dict"])
        model_instance.prepare_for_inference.assert_called_once()
        gen_fn.assert_called_once()

    def test_main_returns_one_when_checkpoint_missing(self):
        import inference

        model_instance = MagicMock()
        with patch.object(inference, "AutoTokenizer") as tok_cls, \
             patch.object(inference, "BitMambaLLM", return_value=model_instance), \
             patch.object(inference.torch, "load", side_effect=FileNotFoundError), \
             patch("builtins.print"):
            tok_cls.from_pretrained.return_value = MagicMock()
            rc = inference.main()

        assert rc == 1
        model_instance.prepare_for_inference.assert_not_called()