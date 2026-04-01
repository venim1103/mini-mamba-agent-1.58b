import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Mock heavy dependencies before importing modules
mock_modules = {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'torch.nn.functional': MagicMock(),
    'transformers': MagicMock(),
    'transformers.AutoTokenizer': MagicMock(),
    'datasets': MagicMock(),
    'datasets.load_dataset': MagicMock(),
    'model': MagicMock(),
    'model.BitMambaLLM': MagicMock(),
}
for name, obj in mock_modules.items():
    if name not in sys.modules:
        sys.modules[name] = obj


class TestStrategyPrompts:
    """Test strategy prompt templates in synth_data.py."""

    def test_all_strategies_defined(self):
        from synth_data import STRATEGY_PROMPTS
        required = ["diverse_qa", "distill", "extract", "knowledge", "rephrase"]
        for strat in required:
            assert strat in STRATEGY_PROMPTS

    def test_diverse_qa_template_structure(self):
        from synth_data import STRATEGY_PROMPTS
        template = STRATEGY_PROMPTS["diverse_qa"]
        assert "<|im_start|>system" in template
        assert "<|im_start|>user" in template
        assert "<|im_start|>assistant" in template
        assert "{text}" in template

    def test_distill_template_preserves_key_instruction(self):
        from synth_data import STRATEGY_PROMPTS
        template = STRATEGY_PROMPTS["distill"]
        assert "concise" in template.lower()
        assert "summary" in template.lower()

    def test_extract_template_mentions_structured(self):
        from synth_data import STRATEGY_PROMPTS
        template = STRATEGY_PROMPTS["extract"]
        assert "structured" in template.lower()

    def test_knowledge_template_mentions_bullets(self):
        from synth_data import STRATEGY_PROMPTS
        template = STRATEGY_PROMPTS["knowledge"]
        assert "bullet" in template.lower()

    def test_rephrase_template_mentions_encyclopedic(self):
        from synth_data import STRATEGY_PROMPTS
        template = STRATEGY_PROMPTS["rephrase"]
        assert "encyclopedic" in template.lower()


class TestTruncateSource:
    """Test truncate_source function."""

    def test_no_truncation_when_under_limit(self):
        from synth_data import truncate_source
        
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3]
        mock_tok.decode.return_value = "short text"
        
        result = truncate_source("short text", mock_tok, max_source_tokens=10)
        
        assert result == "short text"

    def test_truncation_when_over_limit(self):
        from synth_data import truncate_source
        
        mock_tok = MagicMock()
        mock_tok.encode.return_value = list(range(1500))
        mock_tok.decode.return_value = "truncated text"
        
        result = truncate_source("long text " * 500, mock_tok, max_source_tokens=1024)
        
        assert len(mock_tok.encode.call_args[0][0]) <= 1024


class TestIterSourceTexts:
    """Test iter_source_texts function."""

    def test_raises_on_no_files(self, tmp_path):
        from synth_data import iter_source_texts
        
        with pytest.raises(FileNotFoundError):
            list(iter_source_texts(str(tmp_path)))


class TestSynthDataConstants:
    """Test synth_data.py constants."""

    def test_default_checkpoint_path(self):
        from synth_data import DEFAULT_CKPT
        assert DEFAULT_CKPT == "checkpoints/bitmamba_parent/step_1000000.pt"

    def test_model_config_dimensions(self):
        from synth_data import MODEL_CONFIG
        assert MODEL_CONFIG["vocab_size"] == 64000
        assert MODEL_CONFIG["dim"] == 1024
        assert MODEL_CONFIG["n_layers"] == 40
        assert MODEL_CONFIG["d_state"] == 128
        assert MODEL_CONFIG["expand"] == 2

    def test_device_is_cuda_or_cpu(self):
        from synth_data import DEVICE
        assert DEVICE in ["cuda", "cpu"]