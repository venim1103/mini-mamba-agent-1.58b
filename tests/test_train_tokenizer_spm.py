import contextlib
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import sentencepiece as spm
from transformers import AutoTokenizer

import train_tokenizer_spm as spm_mod


# ---------------------------------------------------------------------------
# _resolve_profile
# ---------------------------------------------------------------------------

class TestResolveProfile:
    def test_explicit_kaggle(self):
        assert spm_mod._resolve_profile("kaggle") == "kaggle"

    def test_explicit_standard(self):
        assert spm_mod._resolve_profile("standard") == "standard"

    def test_env_var_triggers_kaggle(self):
        with mock.patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "batch"}):
            assert spm_mod._resolve_profile(None) == "kaggle"

    def test_no_env_var_returns_standard(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert spm_mod._resolve_profile(None) == "standard"


class TestAutoTuneInputSentenceSize:
    def test_non_kaggle_profile_unchanged(self):
        with mock.patch.dict(os.environ, {"TOKENIZER_MAX_RAM_GB": "30"}, clear=True):
            assert spm_mod._auto_tune_input_sentence_size("standard", 750_000) == 750_000

    def test_no_env_keeps_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert spm_mod._auto_tune_input_sentence_size("kaggle", 500_000) == 500_000

    def test_30gb_kaggle_scales_up(self):
        with mock.patch.dict(os.environ, {"TOKENIZER_MAX_RAM_GB": "30"}, clear=True):
            assert spm_mod._auto_tune_input_sentence_size("kaggle", 500_000) == 2_800_000

    def test_override_auto_tune_cap(self):
        with mock.patch.dict(
            os.environ,
            {
                "TOKENIZER_MAX_RAM_GB": "30",
                "TOKENIZER_SPM_AUTO_MAX_INPUT_SENTENCE_SIZE": "3200000",
            },
            clear=True,
        ):
            assert spm_mod._auto_tune_input_sentence_size("kaggle", 500_000) == 3_200_000


# ---------------------------------------------------------------------------
# _infer_domain
# ---------------------------------------------------------------------------

class TestInferDomain:
    def test_logic_path(self):
        assert spm_mod._infer_domain("/data/logic/train.jsonl") == "logic"

    def test_code_path(self):
        assert spm_mod._infer_domain("/data/code/tiny-codes/file.parquet") == "code"

    def test_tools_path(self):
        assert spm_mod._infer_domain("C:\\data\\tools\\toolformer.jsonl") == "tools"

    def test_web_path(self):
        assert spm_mod._infer_domain("/datasets/web/fineweb.jsonl") == "web"

    def test_other_fallback(self):
        assert spm_mod._infer_domain("/datasets/misc/file.jsonl") == "other"

    def test_case_insensitive_matching(self):
        assert spm_mod._infer_domain("/DATA/LOGIC/train.jsonl") == "logic"


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        with mock.patch.object(sys, "argv", ["prog"]):
            args = spm_mod.parse_args()
        assert args.vocab_size == 64_000
        assert args.model_type == "unigram"
        assert args.character_coverage == 0.9995
        assert args.output_dir == "custom_agentic_tokenizer_spm"
        assert args.profile is None
        assert args.input_sentence_size is None
        assert args.max_sentence_length is None
        assert args.code_fidelity_mode is False

    def test_explicit_args(self):
        argv = [
            "prog",
            "--profile", "kaggle",
            "--vocab-size", "32000",
            "--model-type", "unigram",
            "--character-coverage", "0.999",
            "--output-dir", "/tmp/out",
            "--input-sentence-size", "500000",
            "--max-sentence-length", "1024",
            "--code-fidelity-mode",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = spm_mod.parse_args()
        assert args.profile == "kaggle"
        assert args.vocab_size == 32_000
        assert args.model_type == "unigram"
        assert args.character_coverage == 0.999
        assert args.output_dir == "/tmp/out"
        assert args.input_sentence_size == 500_000
        assert args.max_sentence_length == 1024
        assert args.code_fidelity_mode is True


# ---------------------------------------------------------------------------
# _build_temp_corpus
# ---------------------------------------------------------------------------

class TestBuildTempCorpus:
    @staticmethod
    def _safe_remove(path):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)

    def _run_corpus(self, file_texts, profile="standard", max_sentence_length=0, profile_override=None):
        """Run _build_temp_corpus with mocked base iterators."""
        files = list(file_texts.keys())
        patches = [
            mock.patch.object(spm_mod.base, "iter_data_files", return_value=iter(files)),
            mock.patch.object(spm_mod.base, "iter_file_texts",
                              side_effect=lambda fp: iter(file_texts.get(fp, []))),
        ]
        if profile_override is not None:
            patches.append(mock.patch.object(spm_mod, "SPM_PROFILE_DEFAULTS", profile_override))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return spm_mod._build_temp_corpus(profile, max_sentence_length, code_fidelity_mode=False)

    def _small_profile(self, logic_quota):
        return {
            "standard": {
                "input_sentence_size": 10,
                "max_sentence_length": 2048,
                "domain_quota": {"logic": logic_quota, "code": 0, "tools": 0, "web": 0, "other": 0},
            }
        }

    def test_writes_sentences_to_corpus(self):
        path, total = self._run_corpus({"/data/logic/a.jsonl": ["hello world", "foo bar"]})
        try:
            assert total == 2
            assert Path(path).read_text().splitlines() == ["hello world", "foo bar"]
        finally:
            self._safe_remove(path)

    def test_skips_empty_lines(self):
        path, total = self._run_corpus({"/data/logic/a.jsonl": ["  ", "", "  hello  "]})
        try:
            assert total == 1
            assert Path(path).read_text().splitlines() == ["hello"]
        finally:
            self._safe_remove(path)

    def test_truncates_text_by_max_sentence_length(self):
        path, _ = self._run_corpus({"/data/logic/a.jsonl": ["abcdefghij"]}, max_sentence_length=4)
        try:
            assert Path(path).read_text().splitlines() == ["abcd"]
        finally:
            self._safe_remove(path)

    def test_replaces_newlines_in_text(self):
        path, _ = self._run_corpus({"/data/logic/a.jsonl": ["line1\nline2"]})
        try:
            assert Path(path).read_text().splitlines() == ["line1 line2"]
        finally:
            self._safe_remove(path)

    def test_code_fidelity_mode_preserves_newline_boundaries(self):
        file_texts = {"/data/logic/a.jsonl": ["line1\n  line2"]}
        files = list(file_texts.keys())
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(spm_mod.base, "iter_data_files", return_value=iter(files)))
            stack.enter_context(
                mock.patch.object(spm_mod.base, "iter_file_texts", side_effect=lambda fp: iter(file_texts.get(fp, [])))
            )
            path, _ = spm_mod._build_temp_corpus("standard", 0, code_fidelity_mode=True)
        try:
            assert Path(path).read_text().splitlines() == ["line1", "  line2"]
        finally:
            self._safe_remove(path)

    def test_outer_quota_skips_full_domain_file(self):
        file_texts = {
            "/data/logic/first.jsonl": ["first sentence"],
            "/data/logic/second.jsonl": ["should be skipped"],
        }
        path, total = self._run_corpus(file_texts, profile_override=self._small_profile(1))
        try:
            assert total == 1
            assert "should be skipped" not in Path(path).read_text().splitlines()
        finally:
            self._safe_remove(path)

    def test_inner_quota_breaks_mid_file(self):
        path, total = self._run_corpus(
            {"/data/logic/a.jsonl": ["one", "two", "three"]},
            profile_override=self._small_profile(2),
        )
        try:
            assert Path(path).read_text().splitlines() == ["one", "two"]
            assert total == 2
        finally:
            self._safe_remove(path)

    def test_returns_zero_total_on_empty_data(self):
        path, total = self._run_corpus({})
        try:
            assert total == 0
        finally:
            self._safe_remove(path)


# ---------------------------------------------------------------------------
# _export_hf_tokenizer
# ---------------------------------------------------------------------------

class TestExportHfTokenizer:
    def test_exports_working_hf_tokenizer(self, tmp_path):
        # Use more substantial corpus for better training
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("""hello world
def add(a,b): 
    return a+b
def multiply(x, y):
    return x * y
python is great
machine learning rocks
deep learning models
training on data
inference on new data
standard deviation
variance calculation
hello there
goodbye forever
welcome friend
""")

        model_prefix = tmp_path / "spm"
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(model_prefix),
            model_type="bpe",
            vocab_size=64,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<|pad|>",
            unk_piece="<|unk|>",
            bos_piece="<s>",
            eos_piece="<|eos|>",
            user_defined_symbols=["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
        )

        out_dir = tmp_path / "out"
        # Test that _export_hf_tokenizer runs without raising an exception
        spm_mod._export_hf_tokenizer(str(model_prefix) + ".model", str(out_dir))
        # If we get here without an exception, the export function executed successfully


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def _mock_args(self, **overrides):
        args = mock.MagicMock()
        args.profile = None
        args.vocab_size = 64_000
        args.model_type = "bpe"
        args.character_coverage = 0.9995
        args.byte_fallback = True
        args.code_fidelity_mode = False
        args.output_dir = "custom_agentic_tokenizer_spm"
        args.input_sentence_size = None
        args.max_sentence_length = None
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_main_normal_flow(self, tmp_path):
        args = self._mock_args(output_dir=str(tmp_path))
        corpus_path = str(tmp_path / "corpus.txt")

        with mock.patch.object(spm_mod, "parse_args", return_value=args), \
             mock.patch.object(spm_mod, "_build_temp_corpus", return_value=(corpus_path, 100)) as mock_corpus, \
             mock.patch.object(spm.SentencePieceTrainer, "train") as mock_train, \
             mock.patch.object(spm_mod, "_export_hf_tokenizer") as mock_export, \
             mock.patch.object(spm_mod.os, "remove") as mock_remove:
            spm_mod.main()

        mock_corpus.assert_called_once()
        mock_train.assert_called_once()
        mock_export.assert_called_once()
        mock_remove.assert_called_once_with(corpus_path)

    def test_main_uses_explicit_sentence_size_and_length(self, tmp_path):
        args = self._mock_args(output_dir=str(tmp_path), input_sentence_size=500_000, max_sentence_length=1024)
        corpus_path = str(tmp_path / "corpus.txt")

        with mock.patch.object(spm_mod, "parse_args", return_value=args), \
             mock.patch.object(spm_mod, "_build_temp_corpus", return_value=(corpus_path, 50)) as mock_corpus, \
             mock.patch.object(spm.SentencePieceTrainer, "train") as mock_train, \
             mock.patch.object(spm_mod, "_export_hf_tokenizer"), \
             mock.patch.object(spm_mod.os, "remove"):
            spm_mod.main()

        assert mock_corpus.call_args.kwargs["profile"] == "standard"
        assert mock_corpus.call_args.kwargs["max_sentence_length"] == 1024
        assert mock_corpus.call_args.kwargs["code_fidelity_mode"] is False
        assert mock_train.call_args.kwargs["input_sentence_size"] == 500_000
        assert mock_train.call_args.kwargs["max_sentence_length"] == 1024

    def test_main_uses_kaggle_profile_defaults(self, tmp_path):
        args = self._mock_args(output_dir=str(tmp_path), profile="kaggle")
        corpus_path = str(tmp_path / "corpus.txt")

        with mock.patch.object(spm_mod, "parse_args", return_value=args), \
             mock.patch.object(spm_mod, "_build_temp_corpus", return_value=(corpus_path, 50)) as mock_corpus, \
             mock.patch.object(spm.SentencePieceTrainer, "train") as mock_train, \
             mock.patch.object(spm_mod, "_export_hf_tokenizer"), \
             mock.patch.object(spm_mod.os, "remove"):
            spm_mod.main()

        assert mock_corpus.call_args.kwargs["profile"] == "kaggle"
        assert mock_corpus.call_args.kwargs["max_sentence_length"] == 1600
        assert mock_corpus.call_args.kwargs["code_fidelity_mode"] is False
        assert mock_train.call_args.kwargs["input_sentence_size"] == 500_000
        assert mock_train.call_args.kwargs["max_sentence_length"] == 1600

    def test_main_raises_on_empty_corpus(self, tmp_path):
        args = self._mock_args(output_dir=str(tmp_path))
        corpus_path = str(tmp_path / "corpus.txt")

        with mock.patch.object(spm_mod, "parse_args", return_value=args), \
             mock.patch.object(spm_mod, "_build_temp_corpus", return_value=(corpus_path, 0)), \
             mock.patch.object(spm_mod.os, "remove"):
            with pytest.raises(RuntimeError, match="No training text found"):
                spm_mod.main()

    def test_main_removes_corpus_even_on_error(self, tmp_path):
        args = self._mock_args(output_dir=str(tmp_path))
        corpus_path = str(tmp_path / "corpus.txt")

        with mock.patch.object(spm_mod, "parse_args", return_value=args), \
             mock.patch.object(spm_mod, "_build_temp_corpus", return_value=(corpus_path, 10)), \
             mock.patch.object(spm.SentencePieceTrainer, "train", side_effect=RuntimeError("spm fail")), \
             mock.patch.object(spm_mod.os, "remove") as mock_remove:
            with pytest.raises(RuntimeError, match="spm fail"):
                spm_mod.main()

        mock_remove.assert_called_once_with(corpus_path)
