import collections
import json
import os
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import train_tokenizer as tt


# ---------------------------------------------------------------------------
# get_text_column
# ---------------------------------------------------------------------------

class TestGetTextColumn:
    @pytest.mark.parametrize("col", ["text", "content", "trajectory", "prompt", "response"])
    def test_priority_columns(self, col):
        assert tt.get_text_column({col: "hello", "other": "world"}) == col

    def test_fallback_to_first_string(self):
        assert tt.get_text_column({"custom": "hello"}) == "custom"

    def test_skips_non_string_values(self):
        assert tt.get_text_column({"id": 123, "body": "hello"}) == "body"

    def test_returns_none_when_no_strings(self):
        assert tt.get_text_column({"id": 123, "score": 4.5}) is None


# ---------------------------------------------------------------------------
# maybe_trim_text
# ---------------------------------------------------------------------------

class TestMaybeTrimText:
    def test_strips_whitespace(self):
        assert tt.maybe_trim_text("  hello  ") == "hello"

    def test_returns_none_for_empty(self):
        assert tt.maybe_trim_text("") is None
        assert tt.maybe_trim_text("   ") is None

    def test_returns_none_for_non_string(self):
        assert tt.maybe_trim_text(123) is None
        assert tt.maybe_trim_text(None) is None

    def test_trims_when_max_set(self):
        with mock.patch.object(tt, "MAX_TEXT_CHARACTERS", 5):
            assert tt.maybe_trim_text("hello world") == "hello"

    def test_no_trim_when_max_zero(self):
        with mock.patch.object(tt, "MAX_TEXT_CHARACTERS", 0):
            long = "a" * 100_000
            assert tt.maybe_trim_text(long) == long


# ---------------------------------------------------------------------------
# iter_data_files
# ---------------------------------------------------------------------------

class TestIterDataFiles:
    def test_finds_supported_files(self, tmp_path):
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "b.parquet").touch()
        (tmp_path / "c.json").touch()
        (tmp_path / "d.txt").touch()
        (tmp_path / "e.csv").touch()

        with mock.patch.object(tt, "DATA_DIR", str(tmp_path)):
            found = list(tt.iter_data_files())

        names = [os.path.basename(f) for f in found]
        assert "a.jsonl" in names
        assert "b.parquet" in names
        assert "c.json" in names
        assert "d.txt" not in names
        assert "e.csv" not in names

    def test_finds_files_in_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.jsonl").touch()

        with mock.patch.object(tt, "DATA_DIR", str(tmp_path)):
            found = list(tt.iter_data_files())

        assert any("deep.jsonl" in f for f in found)

    def test_returns_sorted_within_dirs(self, tmp_path):
        (tmp_path / "z.jsonl").touch()
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "m.jsonl").touch()

        with mock.patch.object(tt, "DATA_DIR", str(tmp_path)):
            names = [os.path.basename(f) for f in tt.iter_data_files()]

        assert names == sorted(names)


# ---------------------------------------------------------------------------
# iter_text_from_rows
# ---------------------------------------------------------------------------

class TestIterTextFromRows:
    def test_yields_text(self):
        rows = [{"text": "hello"}, {"text": "world"}]
        assert list(tt.iter_text_from_rows(rows, "test")) == ["hello", "world"]

    def test_skips_empty_text(self):
        rows = [{"text": "hello"}, {"text": ""}, {"text": "world"}]
        assert list(tt.iter_text_from_rows(rows, "test")) == ["hello", "world"]

    def test_returns_nothing_when_no_text_column(self, capsys):
        rows = [{"id": 123}]
        assert list(tt.iter_text_from_rows(rows, "test.jsonl")) == []
        assert "Skipping test.jsonl" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# iter_jsonl_rows
# ---------------------------------------------------------------------------

class TestIterJsonlRows:
    def test_reads_valid_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"text": "line one"}) + "\n"
            + json.dumps({"text": "line two"}) + "\n"
        )
        rows = list(tt.iter_jsonl_rows(str(path)))
        assert len(rows) == 2
        assert rows[0]["text"] == "line one"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps({"text": "ok"}) + "\n\n\n")
        assert len(list(tt.iter_jsonl_rows(str(path)))) == 1

    def test_skips_malformed_json(self, tmp_path, capsys):
        path = tmp_path / "data.jsonl"
        path.write_text('{"text": "good"}\nNOT_JSON\n{"text": "also good"}\n')
        rows = list(tt.iter_jsonl_rows(str(path)))
        assert len(rows) == 2
        assert "Skipping malformed JSON line 2" in capsys.readouterr().out

    def test_skips_non_dict_json(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('"just a string"\n[1,2,3]\n{"text": "dict"}\n')
        rows = list(tt.iter_jsonl_rows(str(path)))
        assert len(rows) == 1
        assert rows[0]["text"] == "dict"


# ---------------------------------------------------------------------------
# iter_parquet_texts
# ---------------------------------------------------------------------------

class TestIterParquetTexts:
    def _write_parquet(self, path, table):
        pq.write_table(table, str(path))

    def test_reads_text_column(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table({"id": [1, 2, 3], "text": ["a", "b", "c"]})
        self._write_parquet(path, table)

        with mock.patch.object(tt, "PARQUET_BATCH_SIZE", 2):
            texts = list(tt.iter_parquet_texts(str(path)))

        assert texts == ["a", "b", "c"]

    def test_skips_empty_text(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table({"text": ["a", "", "  ", "b"]})
        self._write_parquet(path, table)

        texts = list(tt.iter_parquet_texts(str(path)))
        assert texts == ["a", "b"]

    def test_skips_file_with_no_string_columns(self, tmp_path, capsys):
        path = tmp_path / "nums.parquet"
        table = pa.table({"id": [1, 2], "score": [0.5, 0.9]})
        self._write_parquet(path, table)

        texts = list(tt.iter_parquet_texts(str(path)))
        assert texts == []
        assert "Skipping" in capsys.readouterr().out

    def test_uses_priority_column_over_fallback(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table({"alpha": ["wrong"], "content": ["right"]})
        self._write_parquet(path, table)

        texts = list(tt.iter_parquet_texts(str(path)))
        assert texts == ["right"]

    def test_ignores_non_string_columns_in_schema(self, tmp_path):
        """Regression: ensure integer columns are not considered for text detection."""
        path = tmp_path / "data.parquet"
        table = pa.table({"id": [1], "body": ["hello"]})
        self._write_parquet(path, table)

        texts = list(tt.iter_parquet_texts(str(path)))
        assert texts == ["hello"]


# ---------------------------------------------------------------------------
# iter_file_texts  (dispatch)
# ---------------------------------------------------------------------------

class TestIterFileTexts:
    def test_dispatches_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps({"text": "jsonl"}) + "\n")
        texts = list(tt.iter_file_texts(str(path)))
        assert texts == ["jsonl"]

    def test_dispatches_parquet(self, tmp_path):
        path = tmp_path / "data.parquet"
        pq.write_table(pa.table({"text": ["pq"]}), str(path))
        texts = list(tt.iter_file_texts(str(path)))
        assert texts == ["pq"]


# ---------------------------------------------------------------------------
# batch_iterator
# ---------------------------------------------------------------------------

class TestBatchIterator:
    def _make_data_dir(self, tmp_path, rows):
        """Write rows as a single JSONL file under a temp DATA_DIR."""
        path = tmp_path / "data.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return tmp_path

    def test_flushes_at_max_examples(self, tmp_path):
        data_dir = self._make_data_dir(
            tmp_path, [{"text": f"doc{i}"} for i in range(5)]
        )
        with mock.patch.object(tt, "DATA_DIR", str(data_dir)):
            batches = list(tt.batch_iterator(max_batch_examples=2, max_batch_characters=10**9))

        # 5 docs, flushed every 2 → batches of [2, 2, 1]
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_flushes_at_max_characters(self, tmp_path):
        data_dir = self._make_data_dir(
            tmp_path, [{"text": "a" * 100} for _ in range(5)]
        )
        with mock.patch.object(tt, "DATA_DIR", str(data_dir)):
            batches = list(tt.batch_iterator(max_batch_examples=10**9, max_batch_characters=250))

        # Each doc is 100 chars; flush happens after appending when total >= 250.
        # doc1(100) + doc2(200) + doc3(300 >= 250) → flush [3], doc4(100) + doc5(200) → remainder [2].
        assert [len(b) for b in batches] == [3, 2]

    def test_empty_dir_yields_nothing(self, tmp_path):
        with mock.patch.object(tt, "DATA_DIR", str(tmp_path)):
            assert list(tt.batch_iterator(max_batch_examples=10, max_batch_characters=10**9)) == []

    def test_spans_multiple_files(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(json.dumps({"text": "from_a"}) + "\n")
        (tmp_path / "b.jsonl").write_text(json.dumps({"text": "from_b"}) + "\n")

        with mock.patch.object(tt, "DATA_DIR", str(tmp_path)):
            batches = list(tt.batch_iterator(max_batch_examples=100, max_batch_characters=10**9))

        all_texts = [t for b in batches for t in b]
        assert "from_a" in all_texts
        assert "from_b" in all_texts

    def test_stops_at_corpus_byte_cap(self, tmp_path):
        # 10 docs of 100 bytes each = 1000 bytes total; cap at 350 bytes.
        data_dir = self._make_data_dir(
            tmp_path, [{"text": "x" * 100} for _ in range(10)]
        )
        with mock.patch.object(tt, "DATA_DIR", str(data_dir)), \
             mock.patch.object(tt, "MAX_CORPUS_BYTES", 350):
            batches = list(tt.batch_iterator(max_batch_examples=10**9, max_batch_characters=10**9))

        total_docs = sum(len(b) for b in batches)
        # Should stop after ~3-4 docs (300-400 bytes), never see all 10
        assert total_docs < 10
        assert total_docs >= 3


# ---------------------------------------------------------------------------
# _corpus_bytes_for_ram
# ---------------------------------------------------------------------------

class TestCorpusBytesForRam:
    def test_30gb_default(self):
        # 30 GB -> (30 - 4) * 1 GB = 26 GB
        result = tt._corpus_bytes_for_ram(30)
        assert result == int(26 * 1_073_741_824)

    def test_8gb(self):
        # 8 GB -> (8 - 4) * 1 GB = 4 GB
        result = tt._corpus_bytes_for_ram(8)
        assert result == int(4 * 1_073_741_824)

    def test_32gb(self):
        # 32 GB -> (32 - 4) * 1 GB = 28 GB
        result = tt._corpus_bytes_for_ram(32)
        assert result == int(28 * 1_073_741_824)

    def test_very_small_ram_floors_at_half_gb(self):
        # 1 GB -> max(1 - 4, 0.5) = 0.5 -> 0.5 * 1 GB
        result = tt._corpus_bytes_for_ram(1)
        assert result == int(0.5 * 1_073_741_824)
        assert result > 0


# ---------------------------------------------------------------------------
# _prune_counter
# ---------------------------------------------------------------------------

class TestPruneCounter:
    def test_removes_low_frequency_words(self):
        counts = collections.Counter({"a": 100, "b": 50, "rare1": 1, "rare2": 1})
        result, threshold = tt._prune_counter(counts, target_size=2)
        assert "a" in result
        assert "b" in result
        assert "rare1" not in result
        assert "rare2" not in result
        assert threshold >= 2

    def test_keeps_all_if_already_under_target(self):
        counts = collections.Counter({"a": 5, "b": 3})
        result, threshold = tt._prune_counter(counts, target_size=10)
        assert len(result) == 2
        assert threshold == 1  # no pruning needed (min_count stayed at 2, returned 1)

    def test_escalates_threshold_until_fits(self):
        # 5 words with counts 1,2,3,4,5 — target 2 means keep only count>=4
        counts = collections.Counter({"a": 5, "b": 4, "c": 3, "d": 2, "e": 1})
        result, threshold = tt._prune_counter(counts, target_size=2)
        assert len(result) <= 2
        assert "a" in result


# ---------------------------------------------------------------------------
# _build_allowed_words
# ---------------------------------------------------------------------------

class TestBuildAllowedWords:
    def test_filters_by_min_frequency(self):
        counts = collections.Counter({"hello": 10, "world": 5, "rare": 1})
        with mock.patch.object(tt, "MIN_FREQUENCY", 3), \
             mock.patch.object(tt, "MAX_UNIQUE_WORDS", 1000):
            result = tt._build_allowed_words(counts)
        assert "hello" in result
        assert "world" in result
        assert "rare" not in result

    def test_caps_at_max_unique_words(self):
        counts = collections.Counter({"a": 100, "b": 50, "c": 10, "d": 5})
        with mock.patch.object(tt, "MIN_FREQUENCY", 2), \
             mock.patch.object(tt, "MAX_UNIQUE_WORDS", 2):
            result = tt._build_allowed_words(counts)
        assert len(result) == 2
        assert "a" in result
        assert "b" in result

    def test_returns_frozenset(self):
        counts = collections.Counter({"x": 5})
        with mock.patch.object(tt, "MIN_FREQUENCY", 1), \
             mock.patch.object(tt, "MAX_UNIQUE_WORDS", 1000):
            result = tt._build_allowed_words(counts)
        assert isinstance(result, frozenset)
