import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import io
from contextlib import redirect_stdout


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


class _FakeTokenizer:
    def __init__(self, name, ids_by_text, decode_map=None):
        self.name_or_path = name
        self._ids_by_text = ids_by_text
        self._decode_map = decode_map or {}

    def encode(self, text, add_special_tokens=False):
        return list(self._ids_by_text.get(text, []))

    def decode(self, ids, skip_special_tokens=False):
        return self._decode_map.get(tuple(ids), "")


class TestCompareTokenizersRun:
    def test_prints_average_ratio_when_base_tokens_present(self):
        from compare_tokenizers import compare_tokenizers

        samples = [
            {"name": "s1", "text": "alpha"},
            {"name": "s2", "text": "beta"},
        ]
        base = _FakeTokenizer(
            "base",
            {"alpha": [1, 2], "beta": [3, 4, 5, 6]},
            {(1, 2): "alpha", (3, 4, 5, 6): "beta"},
        )
        custom = _FakeTokenizer(
            "custom",
            {"alpha": [10], "beta": [11, 12]},
            {(10,): "alpha", (11, 12): "beta"},
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            compare_tokenizers(base, custom, samples, show_ids=False)

        out = stream.getvalue()
        assert "Comparison: base tokenizer vs custom tokenizer" in out
        assert "Average custom/base ratio" in out
        assert "roundtrip exact: True" in out

    def test_show_ids_and_roundtrip_preview_paths(self):
        from compare_tokenizers import compare_tokenizers

        samples = [{"name": "s1", "text": "x"}]
        base = _FakeTokenizer("base", {"x": [1, 2]}, {(1, 2): "x"})
        # Deliberately break roundtrip to hit decoded preview branch.
        custom = _FakeTokenizer("custom", {"x": [7, 8]}, {(7, 8): "not-x"})

        stream = io.StringIO()
        with redirect_stdout(stream):
            compare_tokenizers(base, custom, samples, show_ids=True)

        out = stream.getvalue()
        assert "base ids:" in out
        assert "custom ids:" in out
        assert "decoded preview:" in out
        assert "roundtrip exact: False" in out

    def test_skips_average_when_base_has_zero_tokens(self):
        from compare_tokenizers import compare_tokenizers

        samples = [{"name": "s1", "text": "empty"}]
        base = _FakeTokenizer("base", {"empty": []}, {tuple(): "empty"})
        custom = _FakeTokenizer("custom", {"empty": [42]}, {(42,): "empty"})

        stream = io.StringIO()
        with redirect_stdout(stream):
            compare_tokenizers(base, custom, samples)

        out = stream.getvalue()
        assert "ratio:   n/a" in out
        assert "Average custom/base ratio" not in out


class TestCompareTokenizersMain:
    def test_main_wires_parse_load_and_compare(self):
        import compare_tokenizers

        args = MagicMock(
            base_tokenizer="base-id",
            custom_tokenizer="custom-id",
            samples_file="samples.json",
            show_ids=True,
        )
        sample_payload = [{"name": "s1", "text": "hello"}]

        with patch.object(compare_tokenizers, "parse_args", return_value=args), \
             patch.object(compare_tokenizers, "load_samples", return_value=sample_payload), \
             patch.object(compare_tokenizers.AutoTokenizer, "from_pretrained", side_effect=["BASE", "CUSTOM"]) as load_tok, \
             patch.object(compare_tokenizers, "compare_tokenizers") as cmp_fn:
            compare_tokenizers.main()

        load_tok.assert_any_call("base-id")
        load_tok.assert_any_call("custom-id")
        cmp_fn.assert_called_once_with(
            base_tokenizer="BASE",
            custom_tokenizer="CUSTOM",
            samples=sample_payload,
            show_ids=True,
        )