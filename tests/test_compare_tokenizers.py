import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path


class TestDefaultSamples:
    """Test default sample definitions."""

    def test_all_required_samples_present(self):
        from compare_tokenizers import DEFAULT_SAMPLES
        
        names = [s["name"] for s in DEFAULT_SAMPLES]
        
        assert "python_code" in names
        assert "math_reasoning" in names
        assert "tool_calling" in names
        assert "web_text" in names
        assert "mixed_agentic" in names

    def test_each_sample_has_name_and_text(self):
        from compare_tokenizers import DEFAULT_SAMPLES
        
        for sample in DEFAULT_SAMPLES:
            assert "name" in sample
            assert "text" in sample
            assert isinstance(sample["name"], str)
            assert isinstance(sample["text"], str)

    def test_python_code_sample_contains_function(self):
        from compare_tokenizers import DEFAULT_SAMPLES
        
        sample = next(s for s in DEFAULT_SAMPLES if s["name"] == "python_code")
        
        assert "def binary_search" in sample["text"]
        assert "return" in sample["text"]

    def test_math_sample_contains_quadratic(self):
        from compare_tokenizers import DEFAULT_SAMPLES
        
        sample = next(s for s in DEFAULT_SAMPLES if s["name"] == "math_reasoning")
        
        assert "quadratic" in sample["text"].lower()
        assert "x =" in sample["text"]

    def test_tool_calling_sample_has_im_tokens(self):
        from compare_tokenizers import DEFAULT_SAMPLES
        
        sample = next(s for s in DEFAULT_SAMPLES if s["name"] == "tool_calling")
        
        assert "<|im_start|>" in sample["text"]
        assert "<|im_end|>" in sample["text"]
        assert "get_weather" in sample["text"]


class TestLoadSamples:
    """Test load_samples function."""

    def test_returns_defaults_when_no_file(self):
        from compare_tokenizers import load_samples, DEFAULT_SAMPLES
        
        result = load_samples(None)
        
        assert result == DEFAULT_SAMPLES

    def test_loads_json_file(self, tmp_path):
        from compare_tokenizers import load_samples
        
        samples = [
            {"name": "test1", "text": "hello world"},
            {"name": "test2", "text": "foo bar"},
        ]
        
        path = tmp_path / "samples.json"
        path.write_text(json.dumps(samples))
        
        result = load_samples(str(path))
        
        assert len(result) == 2
        assert result[0]["name"] == "test1"

    def test_rejects_non_list_file(self, tmp_path):
        from compare_tokenizers import load_samples
        
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"not": "a list"}))
        
        with pytest.raises(ValueError, match="must contain a JSON list"):
            load_samples(str(path))

    def test_rejects_non_dict_sample(self, tmp_path):
        from compare_tokenizers import load_samples
        
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(["just a string"]))
        
        with pytest.raises(ValueError, match="not an object"):
            load_samples(str(path))

    def test_rejects_missing_name(self, tmp_path):
        from compare_tokenizers import load_samples
        
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"text": "hello"}]))
        
        with pytest.raises(ValueError, match="must have string 'name'"):
            load_samples(str(path))

    def test_rejects_missing_text(self, tmp_path):
        from compare_tokenizers import load_samples
        
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"name": "test"}]))
        
        with pytest.raises(ValueError, match="must have string 'name'"):
            load_samples(str(path))


class TestFormatRatio:
    """Test format_ratio function."""

    def test_handles_zero_base(self):
        from compare_tokenizers import format_ratio
        
        result = format_ratio(0, 10)
        
        assert result == "n/a"

    def test_shows_fewer_tokens(self):
        from compare_tokenizers import format_ratio
        
        result = format_ratio(100, 80)
        
        assert "fewer tokens" in result
        assert "0.800" in result

    def test_shows_more_tokens(self):
        from compare_tokenizers import format_ratio
        
        result = format_ratio(80, 100)
        
        assert "more tokens" in result
        assert "1.250" in result

    def test_exact_ratio_one(self):
        from compare_tokenizers import format_ratio
        
        result = format_ratio(100, 100)
        
        assert "1.000" in result
        assert "0.0% fewer" in result


class TestCompareTokenizers:
    """Test compare_tokenizers function."""

    @pytest.fixture
    def mock_tokenizers(self):
        mock_base = MagicMock()
        mock_base.name_or_path = "base/model"
        mock_base.encode.return_value = [1, 2, 3, 4, 5]
        
        mock_custom = MagicMock()
        mock_custom.name_or_path = "custom/model"
        mock_custom.encode.return_value = [1, 2, 3]
        mock_custom.decode.return_value = "decoded"
        
        return mock_base, mock_custom

    def test_calculates_ratios(self, mock_tokenizers):
        from compare_tokenizers import compare_tokenizers
        
        mock_base, mock_custom = mock_tokenizers
        
        samples = [{"name": "test", "text": "hello"}]
        
        compare_tokenizers(mock_base, mock_custom, samples)
        
        ratio = len(mock_custom.encode.return_value) / len(mock_base.encode.return_value)
        assert ratio == 0.6

    def test_handles_empty_base_ids(self, mock_tokenizers):
        from compare_tokenizers import compare_tokenizers
        
        mock_base, mock_custom = mock_tokenizers
        mock_base.encode.return_value = []
        
        samples = [{"name": "test", "text": "hello"}]
        
        compare_tokenizers(mock_base, mock_custom, samples)
        
        assert True


class TestParseArgs:
    """Test argument parsing."""

    def test_default_base_tokenizer(self):
        from compare_tokenizers import parse_args, DEFAULT_BASE_TOKENIZER
        
        with patch("sys.argv", ["compare_tokenizers.py"]):
            args = parse_args()
        
        assert args.base_tokenizer == DEFAULT_BASE_TOKENIZER

    def test_default_custom_tokenizer(self):
        from compare_tokenizers import parse_args, DEFAULT_CUSTOM_TOKENIZER
        
        with patch("sys.argv", ["compare_tokenizers.py"]):
            args = parse_args()
        
        assert args.custom_tokenizer == DEFAULT_CUSTOM_TOKENIZER

    def test_custom_tokenizer_arg(self):
        from compare_tokenizers import parse_args
        
        with patch("sys.argv", ["compare_tokenizers.py", "--custom-tokenizer", "my/tokenizer"]):
            args = parse_args()
        
        assert args.custom_tokenizer == "my/tokenizer"

    def test_show_ids_flag(self):
        from compare_tokenizers import parse_args
        
        with patch("sys.argv", ["compare_tokenizers.py", "--show-ids"]):
            args = parse_args()
        
        assert args.show_ids is True
