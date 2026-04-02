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

    def test_model_config_dimensions(self):
        from synth_data import MODEL_CONFIG
        assert MODEL_CONFIG["vocab_size"] > 0
        assert MODEL_CONFIG["dim"] > 0
        assert MODEL_CONFIG["n_layers"] > 0
        assert MODEL_CONFIG["d_state"] > 0
        assert MODEL_CONFIG["expand"] > 0

    def test_device_is_cuda_or_cpu(self):
        from synth_data import DEVICE
        assert DEVICE in ["cuda", "cpu"]


class TestSynthDataUtils:
    """Tests for synth_data.py utility functions: truncate_source and iter_source_texts."""

    def test_truncate_source_shortens_long_text(self):
        from synth_data import truncate_source
        from unittest.mock import MagicMock
        tok = MagicMock()
        tok.encode.return_value = list(range(2000))
        tok.decode.return_value = "truncated text"
        result = truncate_source("long text", tok, max_source_tokens=512)
        tok.encode.assert_called_once()
        tok.decode.assert_called_once_with(list(range(512)), skip_special_tokens=True)
        assert result == "truncated text"

    def test_truncate_source_leaves_short_text_unchanged(self):
        from synth_data import truncate_source
        from unittest.mock import MagicMock
        tok = MagicMock()
        tok.encode.return_value = list(range(100))
        result = truncate_source("short text", tok, max_source_tokens=512)
        tok.decode.assert_not_called()
        assert result == "short text"

    @patch("synth_data.load_dataset")
    def test_iter_source_texts_yields_from_jsonl(self, mock_load_dataset, tmp_path):
        from synth_data import iter_source_texts
        
        long_text_1 = "hello world " * 5
        long_text_2 = "another document here " * 3
        
        mock_load_dataset.return_value = [
            {"text": long_text_1},
            {"text": long_text_2}
        ]
        
        f = tmp_path / "data.jsonl"
        f.write_text("dummy")
        
        results = list(iter_source_texts(str(tmp_path)))
        
        assert len(results) == 2
        mock_load_dataset.assert_called_once()


class TestRunPipelineIntegration:
    """Tiny integration test: runs the full synthetic data pipeline on CPU."""

    @patch('synth_data.AutoTokenizer.from_pretrained')
    @patch('synth_data.iter_source_texts', return_value=["dummy text 1 " * 5, "dummy text 2 " * 5])
    def test_run_pipeline_completes_and_writes_jsonl(
        self, mock_iter, mock_tok_cls, tmp_path, monkeypatch
    ):
        import synth_data
        from model import BitMambaLLM
        from argparse import Namespace
        import json
        import torch
        from unittest.mock import MagicMock

        TINY_CFG = dict(vocab_size=64, dim=32, n_layers=1, d_state=16,
                        expand=2, use_checkpoint=False)

        monkeypatch.setattr(synth_data, 'DEVICE', 'cpu')
        monkeypatch.setattr(synth_data, 'MODEL_CONFIG', TINY_CFG)

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "dummy generation"
        mock_inputs = MagicMock()
        mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = mock_inputs
        mock_tok_cls.return_value = mock_tokenizer

        tiny_model = BitMambaLLM(**TINY_CFG)
        dummy_ckpt = tmp_path / "dummy.pt"
        torch.save({"model_state_dict": tiny_model.state_dict()}, dummy_ckpt)

        with patch('synth_data.BitMambaLLM', return_value=tiny_model):
            args = Namespace(
                strategy="diverse_qa",
                input="dummy_input",
                output=str(tmp_path),
                checkpoint=str(dummy_ckpt),
                num_samples=2,
                max_new_tokens=5,
                max_source_tokens=512,
                temperature=0.0,
            )
            synth_data.run_pipeline(args)

        out_file = tmp_path / "diverse_qa.jsonl"
        assert out_file.exists()
        lines = [l for l in out_file.read_text().strip().split('\n') if l]
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert record["strategy"] == "diverse_qa"
            assert record["generated"] == "dummy generation"