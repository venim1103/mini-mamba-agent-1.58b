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

    def test_finds_jsonl_files(self, tmp_path):
        data_dir = tmp_path / "input"
        data_dir.mkdir()
        
        file_path = data_dir / "test.jsonl"
        file_path.write_text(json.dumps({"text": "hello world"}) + "\n")
        
        from synth_data import iter_source_texts
        texts = list(iter_source_texts(str(data_dir)))
        
        assert "hello world" in texts

    def test_finds_parquet_files(self, tmp_path):
        data_dir = tmp_path / "input"
        data_dir.mkdir()
        
        file_path = data_dir / "test.parquet"
        import pyarrow as pa
        table = pa.table({"text": ["hello"]})
        import pyarrow.parquet as pq
        pq.write_table(table, str(file_path))
        
        from synth_data import iter_source_texts
        texts = list(iter_source_texts(str(data_dir)))
        
        assert "hello" in texts

    def test_skips_files_without_text_column(self, tmp_path):
        data_dir = tmp_path / "input"
        data_dir.mkdir()
        
        file_path = data_dir / "test.jsonl"
        file_path.write_text(json.dumps({"content": "value"}) + "\n")
        
        from synth_data import iter_source_texts
        texts = list(iter_source_texts(str(data_dir)))
        
        assert "value" in texts

    def test_priority_column_selection(self, tmp_path):
        data_dir = tmp_path / "input"
        data_dir.mkdir()
        
        file_path = data_dir / "test.jsonl"
        file_path.write_text(json.dumps({"question": "q", "text": "t", "content": "c"}) + "\n")
        
        from synth_data import iter_source_texts
        texts = list(iter_source_texts(str(data_dir)))
        
        assert "t" in texts

    def test_skips_short_texts(self, tmp_path):
        data_dir = tmp_path / "input"
        data_dir.mkdir()
        
        file_path = data_dir / "test.jsonl"
        file_path.write_text(json.dumps({"text": "short"}) + "\n")
        
        from synth_data import iter_source_texts
        texts = list(iter_source_texts(str(data_dir)))
        
        assert len(texts) == 0

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


class TestRunPipeline:
    """Test run_pipeline function via mocking."""

    @pytest.fixture
    def mock_model_and_tok(self):
        with patch("synth_data.AutoTokenizer") as MockTok, \
             patch("synth_data.BitMambaLLM") as MockModel, \
             patch("synth_data.iter_source_texts") as MockIter:
            
            mock_tok = MagicMock()
            mock_tok.encode.return_value = [1, 2, 3]
            MockTok.from_pretrained.return_value = mock_tok
            
            mock_model = MagicMock()
            MockModel.return_value = mock_model
            
            yield MockTok, MockModel, MockIter

    def test_pipeline_creates_output_dir(self, mock_model_and_tok, tmp_path):
        from synth_data import run_pipeline
        
        MockTok, MockModel, MockIter = mock_model_and_tok
        MockIter.return_value = iter([])
        
        args = MagicMock()
        args.strategy = "diverse_qa"
        args.input = str(tmp_path)
        args.output = str(tmp_path / "output")
        args.num_samples = 1
        args.max_source_tokens = 1024
        args.max_new_tokens = 512
        args.temperature = 0.7
        args.checkpoint = "dummy.pt"
        
        run_pipeline(args)
        
        assert os.path.exists(args.output)
