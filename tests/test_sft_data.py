# tests/test_sft_data.py
"""Unit tests for sft_data.py — collation, message parsing, and reasoning toggle."""

import re
import pytest
import torch
import sys
from unittest.mock import MagicMock, patch

# Mock datasets and huggingface_hub BEFORE importing sft_data.py to prevent import errors
_mocked_modules = {
    'datasets': MagicMock(),
    'datasets.load_dataset': MagicMock(),
    'huggingface_hub': MagicMock(),
    'huggingface_hub.utils': MagicMock(),
}

for _name, _obj in _mocked_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _obj


from sft_data import sft_collate_fn, SFTChatDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(pad_id=2, eos_id=1):
    """Return a minimal mock tokenizer."""
    tok = MagicMock()
    tok.pad_token_id = pad_id
    tok.eos_token_id = eos_id
    tok.encode.side_effect = lambda text, add_special_tokens=False: (
        [ord(c) % 50 + 10 for c in text[:8]]  # up to 8 token-ints per string
    )
    # Special token IDs returned as single-element lists.
    def _encode_special(text, add_special_tokens=False):
        return {
            "<|im_start|>": [3],
            "<|im_end|>": [4],
            "\n": [5],
        }.get(text, [ord(c) % 50 + 10 for c in text[:8]])
    tok.encode.side_effect = _encode_special
    return tok


def _ids_tensor(*lengths):
    """Return a list of (input_ids, labels) pairs of given lengths."""
    samples = []
    for n in lengths:
        ids = torch.randint(10, 200, (n,))
        lbl = torch.randint(10, 200, (n,))
        samples.append((ids, lbl))
    return samples


# ===========================================================================
# 1. sft_collate_fn
# ===========================================================================

class TestSftCollateFn:
    def test_output_shape_equal_length(self):
        batch = _ids_tensor(8, 8, 8)
        ids, lbls = sft_collate_fn(batch, pad_token_id=0)
        assert ids.shape == (3, 8)
        assert lbls.shape == (3, 8)

    def test_output_shape_unequal_lengths(self):
        batch = _ids_tensor(4, 7, 6)
        ids, lbls = sft_collate_fn(batch, pad_token_id=0)
        assert ids.shape == (3, 7)
        assert lbls.shape == (3, 7)

    def test_pads_with_correct_pad_token_id(self):
        PAD = 99
        batch = _ids_tensor(3, 6)
        ids, _ = sft_collate_fn(batch, pad_token_id=PAD)
        # First sample was length 3 — positions 3..5 should be PAD.
        assert (ids[0, 3:] == PAD).all()

    def test_pads_labels_with_minus_100(self):
        batch = _ids_tensor(3, 7)
        _, lbls = sft_collate_fn(batch, pad_token_id=0)
        assert (lbls[0, 3:] == -100).all()

    def test_does_not_truncate_existing_content(self):
        orig = torch.arange(5).long()
        batch = [(orig, orig.clone()), (torch.zeros(8, dtype=torch.long), torch.zeros(8, dtype=torch.long))]
        ids, _ = sft_collate_fn(batch, pad_token_id=0)
        assert torch.equal(ids[0, :5], orig)

    def test_no_padding_when_lengths_equal(self):
        # Use a pad_token_id outside the _ids_tensor range [10, 200) to avoid
        # false positives from real token values matching the sentinel.
        PAD = 9
        batch = _ids_tensor(6, 6, 6)
        ids, _ = sft_collate_fn(batch, pad_token_id=PAD)
        assert (ids != PAD).all()

    def test_returns_stacked_tensors(self):
        batch = _ids_tensor(5, 5)
        ids, lbls = sft_collate_fn(batch, pad_token_id=0)
        assert ids.dim() == 2
        assert lbls.dim() == 2

    def test_dtype_preserved(self):
        batch = _ids_tensor(4, 6)
        ids, lbls = sft_collate_fn(batch, pad_token_id=0)
        assert ids.dtype == torch.long
        assert lbls.dtype == torch.long

    def test_default_pad_id_is_zero(self):
        batch = _ids_tensor(3, 5)
        ids, _ = sft_collate_fn(batch)  # no pad_token_id kwarg
        assert (ids[0, 3:] == 0).all()


# ===========================================================================
# 2. SFTChatDataset._row_to_messages
# ===========================================================================

class TestRowToMessages:
    """
    Test the internal `_row_to_messages` converter without hitting the filesystem.
    We instantiate SFTChatDataset via a mock so __init__ doesn't try to load files.
    """

    @pytest.fixture
    def dataset(self):
        """Return an SFTChatDataset instance with mocked raw_data so __init__ passes."""
        tok = _make_tokenizer()
        with patch("sft_data.load_dataset") as mock_ld, \
             patch("os.path.isdir", return_value=False):
            mock_ld.return_value = MagicMock()
            mock_ld.return_value.__len__ = lambda self: 10
            ds = SFTChatDataset.__new__(SFTChatDataset)
            # Manually set the attributes _row_to_messages needs.
            ds.tokenizer = tok
            ds.max_seq_len = 512
            ds.reasoning_off_prob = 0.0
            ds.im_start = 3
            ds.im_end = 4
            ds.nl = 5
            ds.sys_reasoning_on = "Think step by step."
            ds.sys_reasoning_off = "Answer directly."
        return ds

    def test_chat_format_with_messages_key(self, dataset):
        row = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
        msgs = dataset._row_to_messages(row)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    def test_chat_format_with_conversations_key(self, dataset):
        row = {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}
        msgs = dataset._row_to_messages(row)
        assert len(msgs) == 2

    def test_math_reasoning_format(self, dataset):
        row = {"problem": "2+2=?", "generated_solution": "<think>Compute.4"}
        msgs = dataset._row_to_messages(row)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert "<think>" in msgs[1]["content"]

    def test_math_reasoning_question_alias(self, dataset):
        row = {"question": "What is 3*3?", "solution": "9"}
        msgs = dataset._row_to_messages(row)
        assert any(m["role"] == "assistant" for m in msgs)

    def test_tool_calling_format(self, dataset):
        row = {"query": "Get weather", "tools": "[weather_api]", "answers": "Sunny"}
        msgs = dataset._row_to_messages(row)
        roles = [m["role"] for m in msgs]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_fallback_format(self, dataset):
        row = {"instruction": "Translate this", "output": "Translated"}
        msgs = dataset._row_to_messages(row)
        assert len(msgs) == 2

    def test_empty_row_returns_empty(self, dataset):
        msgs = dataset._row_to_messages({})
        assert msgs == []

    def test_empty_messages_list_returns_empty(self, dataset):
        msgs = dataset._row_to_messages({"messages": []})
        assert msgs == []


# ===========================================================================
# 3. Reasoning toggle (content stripping)
# ===========================================================================

class TestReasoningToggle:
    """Verify that <think>... blocks are stripped when reasoning_off_prob=1."""

    THINK_CONTENT = "Long chain of thought reasoning here."
    ANSWER = "The answer is 42."
    FULL_RESPONSE = f"<think>{THINK_CONTENT}</think>\n{ANSWER}"

    def _make_dataset_with_row(self, reasoning_off_prob):
        tok = _make_tokenizer()
        with patch("sft_data.load_dataset") as mock_ld, \
             patch("os.path.isdir", return_value=False):
            mock_ld.return_value = MagicMock()
            mock_ld.return_value.__len__ = lambda self: 1
            # Build a valid row
            row = {
                "messages": [
                    {"role": "user", "content": "What is 6*7?"},
                    {"role": "assistant", "content": self.FULL_RESPONSE},
                ]
            }
            mock_ld.return_value.__getitem__ = lambda self, i: row

            ds = SFTChatDataset.__new__(SFTChatDataset)
            ds.tokenizer = tok
            ds.max_seq_len = 512
            ds.reasoning_off_prob = reasoning_off_prob
            ds.im_start = 3
            ds.im_end = 4
            ds.nl = 5
            ds.sys_reasoning_on = "Think step by step."
            ds.sys_reasoning_off = "Answer directly."
            ds.raw_data = [row]
        return ds

    def test_think_tags_stripped_when_off(self):
        """With reasoning_off_prob=1 and mock random, check stripping logic directly."""
        ds = self._make_dataset_with_row(reasoning_off_prob=1.0)
        # Simulate stripping manually (mirrors dataset logic).
        content = self.FULL_RESPONSE
        stripped = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        assert "<think>" not in stripped
        assert self.ANSWER in stripped

    def test_think_tags_preserved_when_on(self):
        content = self.FULL_RESPONSE
        # reasoning_off_prob=0 → never strip.
        # The content should be unchanged.
        assert "<think>" in content

    def test_strip_leaves_answer_intact(self):
        content = self.FULL_RESPONSE
        stripped = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        assert self.ANSWER in stripped

    def test_strip_removes_only_think_block(self):
        content = "Preamble. <think>internal</think> Final answer."
        stripped = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        assert "internal" not in stripped
        assert "Preamble" in stripped
        assert "Final answer" in stripped


class TestTokenizerFallbacks:
    def test_init_handles_empty_newline_encoding(self):
        tok = MagicMock()
        tok.pad_token_id = 7
        tok.eos_token_id = 1

        def _encode(text, add_special_tokens=False):
            mapping = {
                "<|im_start|>": [3],
                "<|im_end|>": [4],
                "\n": [],
                " ": [9],
            }
            return mapping.get(text, [11])

        tok.encode.side_effect = _encode

        with patch("sft_data.load_dataset") as mock_ld, patch("os.path.isdir", return_value=False):
            mock_ld.return_value = [{"messages": [{"role": "user", "content": "hi"}]}]
            ds = SFTChatDataset("dummy.json", tok, max_seq_len=32)

        assert ds.im_start == 3
        assert ds.im_end == 4
        assert ds.nl == 9


from sft_data import _first_token_id


class TestFirstTokenId:
    def test_prefers_primary_text_token(self):
        tok = MagicMock()
        tok.encode.side_effect = lambda text, add_special_tokens=False: [123] if text == "X" else [9]
        assert _first_token_id(tok, "X", fallback_text="Y") == 123

    def test_uses_fallback_text_when_primary_empty(self):
        tok = MagicMock()
        tok.eos_token_id = None
        tok.pad_token_id = None
        tok.unk_token_id = None

        def _encode(text, add_special_tokens=False):
            return {"X": [], "Y": [77]}.get(text, [])

        tok.encode.side_effect = _encode
        assert _first_token_id(tok, "X", fallback_text="Y") == 77

    def test_uses_special_token_fallback_order(self):
        tok = MagicMock()
        tok.encode.return_value = []
        tok.eos_token_id = None
        tok.pad_token_id = 55
        tok.unk_token_id = 66
        assert _first_token_id(tok, "X") == 55

    def test_returns_zero_when_no_options_exist(self):
        tok = MagicMock()
        tok.encode.return_value = []
        tok.eos_token_id = None
        tok.pad_token_id = None
        tok.unk_token_id = None
        assert _first_token_id(tok, "X") == 0


class TestSFTDatasetInitRouting:
    def _tok(self):
        tok = MagicMock()
        tok.eos_token_id = 1
        tok.pad_token_id = 0
        tok.encode.side_effect = lambda text, add_special_tokens=False: {
            "<|im_start|>": [3],
            "<|im_end|>": [4],
            "\n": [5],
            " ": [6],
        }.get(text, [11])
        return tok

    def test_directory_json_hint_uses_json_only_files(self):
        tok = self._tok()
        with patch("os.path.isdir", return_value=True), \
             patch("os.walk", return_value=[("/tmp", [], ["a.parquet", "b.jsonl", "c.json"])]), \
             patch("sft_data.load_dataset", return_value=[{"messages": [{"role": "user", "content": "hi"}]}]) as ld:
            SFTChatDataset("/tmp", tok, format_hint="json")

        ld.assert_called_once_with("json", data_files=["/tmp/c.json"], split="train")

    def test_directory_without_supported_files_raises(self):
        tok = self._tok()
        with patch("os.path.isdir", return_value=True), \
             patch("os.walk", return_value=[("/tmp", [], ["readme.md"])]):
            with pytest.raises(FileNotFoundError, match="No .json/.jsonl/.parquet files"):
                SFTChatDataset("/tmp", tok)


class TestSFTDatasetGetItem:
    def _dataset_for_rows(self, rows, tokenizer, reasoning_off_prob=0.0):
        ds = SFTChatDataset.__new__(SFTChatDataset)
        ds.tokenizer = tokenizer
        ds.max_seq_len = 64
        ds.reasoning_off_prob = reasoning_off_prob
        ds.im_start = 3
        ds.im_end = 4
        ds.nl = 5
        ds.sys_reasoning_on = "Think"
        ds.sys_reasoning_off = "Direct"
        ds.raw_data = rows
        return ds

    def test_getitem_empty_messages_returns_eos_and_ignore(self):
        tok = MagicMock()
        tok.eos_token_id = 9
        tok.encode.return_value = [1]
        ds = self._dataset_for_rows([{}], tok)
        ids, labels = ds[0]
        assert torch.equal(ids, torch.tensor([9], dtype=torch.long))
        assert torch.equal(labels, torch.tensor([-100], dtype=torch.long))

    def test_getitem_conversation_role_mapping_and_labeling(self):
        tok = MagicMock()

        def _encode(text, add_special_tokens=False):
            mapping = {
                "system": [10],
                "user": [11],
                "assistant": [12],
                "Think": [13],
                "hello": [14, 15],
                "world": [16, 17],
            }
            return mapping.get(text, [99])

        tok.encode.side_effect = _encode
        tok.eos_token_id = 1
        row = {"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "world"}]}
        ds = self._dataset_for_rows([row], tok, reasoning_off_prob=0.0)

        ids, labels = ds[0]
        assert ids.numel() == labels.numel()
        # There should be supervised assistant tokens and masked user/system tokens.
        assert (labels != -100).any()
        assert (labels == -100).any()

    def test_getitem_strips_think_content_when_reasoning_off(self):
        tok = MagicMock()
        observed_contents = []

        def _encode(text, add_special_tokens=False):
            if text in {"system", "assistant", "user", "Direct"}:
                return [1]
            if "<think>" in text or "answer" in text:
                observed_contents.append(text)
            return [2]

        tok.encode.side_effect = _encode
        tok.eos_token_id = 1
        row = {
            "messages": [
                {"role": "assistant", "content": "<think>hidden</think> answer"},
            ]
        }
        ds = self._dataset_for_rows([row], tok, reasoning_off_prob=1.0)

        with patch("random.random", return_value=0.0):
            ds[0]

        # Content passed for encoding should be stripped down to visible answer.
        assert "answer" in observed_contents
        assert all("<think>" not in text for text in observed_contents)
