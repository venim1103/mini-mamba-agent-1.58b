import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
import os


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
        assert "summary" in template.lower() or "summaries" in template.lower()

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