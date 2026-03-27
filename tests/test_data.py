# tests/test_data.py
"""Unit tests for data.py — token packing, collation, and dataset mixture."""

import pytest
import torch

from data import (
    packed_token_stream,
    packed_collate_fn,
    AgenticDataMixture,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _FakeTok:
    """Minimal tokenizer stand-in that just maps chars to ints 1-26."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [min(ord(c) % 27 + 1, 26) for c in text]
        return {"input_ids": ids}


def _make_stream(texts):
    """Yield rows forever, cycling through `texts`."""
    i = 0
    while True:
        yield {"text": texts[i % len(texts)]}
        i += 1


@pytest.fixture
def tok():
    return _FakeTok()


# ---------------------------------------------------------------------------
# packed_token_stream
# ---------------------------------------------------------------------------

class TestPackedTokenStream:
    def test_yields_three_tuple(self, tok):
        stream = _make_stream(["hello world"])
        gen = packed_token_stream(stream, tok, "text", max_seq_len=8)
        item = next(gen)
        assert len(item) == 3

    def test_x_y_shapes(self, tok):
        """x and y must both have length max_seq_len."""
        max_seq_len = 10
        stream = _make_stream(["abcde"] * 20)
        gen = packed_token_stream(stream, tok, "text", max_seq_len=max_seq_len)
        x, y, _ = next(gen)
        assert x.shape == (max_seq_len,)
        assert y.shape == (max_seq_len,)

    def test_y_is_x_shifted_right(self, tok):
        """y[i] == x[i+1] for all i except the last position."""
        stream = _make_stream(["abcdefghijklmnop"] * 10)
        gen = packed_token_stream(stream, tok, "text", max_seq_len=8)
        x, y, _ = next(gen)
        assert torch.equal(x[1:], y[:-1])

    def test_cu_seqlens_starts_with_zero(self, tok):
        stream = _make_stream(["abc"] * 30)
        gen = packed_token_stream(stream, tok, "text", max_seq_len=10)
        _, _, cu = next(gen)
        assert cu[0].item() == 0

    def test_cu_seqlens_ends_with_max_seq_len(self, tok):
        max_seq_len = 10
        stream = _make_stream(["abc"] * 50)
        gen = packed_token_stream(stream, tok, "text", max_seq_len=max_seq_len)
        _, _, cu = next(gen)
        assert cu[-1].item() == max_seq_len

    def test_cu_seqlens_monotonic(self, tok):
        stream = _make_stream(["abc", "de", "fghij"] * 20)
        gen = packed_token_stream(stream, tok, "text", max_seq_len=12)
        for _ in range(5):
            _, _, cu = next(gen)
            diffs = (cu[1:] - cu[:-1])
            assert (diffs >= 0).all(), "cu_seqlens is not monotonically non-decreasing"

    def test_no_boundary_drift_over_many_chunks(self, tok):
        """
        Regression test for Finding #6 (off-by-one drift).

        After N chunks, doc_lengths must still be finite. We verify this
        indirectly by checking that cu_seqlens are always valid (no negative
        segment lengths and always ending at max_seq_len).
        """
        max_seq_len = 16
        # Short docs so they span multiple boundaries
        stream = _make_stream(["ab", "cd", "ef"] * 500)
        gen = packed_token_stream(stream, tok, "text", max_seq_len=max_seq_len)
        for _ in range(200):
            _, _, cu = next(gen)
            assert cu[0].item() == 0
            assert cu[-1].item() == max_seq_len
            diffs = (cu[1:] - cu[:-1])
            assert (diffs > 0).all(), "Zero-length segment indicates boundary drift"

    def test_x_dtype(self, tok):
        stream = _make_stream(["hello world abc"] * 20)
        x, y, _ = next(packed_token_stream(stream, tok, "text", max_seq_len=10))
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_cu_seqlens_dtype(self, tok):
        stream = _make_stream(["hello"] * 20)
        _, _, cu = next(packed_token_stream(stream, tok, "text", max_seq_len=8))
        assert cu.dtype == torch.int32

    def test_generates_multiple_chunks(self, tok):
        stream = _make_stream(["a" * 100])
        gen = packed_token_stream(stream, tok, "text", max_seq_len=10)
        items = [next(gen) for _ in range(10)]
        assert len(items) == 10


# ---------------------------------------------------------------------------
# packed_collate_fn
# ---------------------------------------------------------------------------

class TestPackedCollateFn:
    def _make_sample(self, seq_len, n_segs):
        x = torch.randint(0, 100, (seq_len,))
        y = torch.randint(0, 100, (seq_len,))
        boundaries = sorted(torch.randperm(seq_len - 1)[:n_segs - 1].tolist())
        cu = torch.tensor([0] + boundaries + [seq_len], dtype=torch.int32)
        return x, y, cu

    def test_output_tuple_length(self):
        batch = [self._make_sample(16, 3), self._make_sample(16, 2)]
        result = packed_collate_fn(batch)
        assert len(result) == 4  # x, y, cu_seqlens, n_segs

    def test_x_y_shape(self):
        batch = [self._make_sample(16, 3) for _ in range(4)]
        x, y, _, _ = packed_collate_fn(batch)
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_cu_seqlens_padded_to_max(self):
        """cu_seqlens must be padded with -1 to the longest in the batch."""
        batch = [self._make_sample(16, 2), self._make_sample(16, 5)]
        _, _, cu, n_segs = packed_collate_fn(batch)
        # Second sample has 5+1=6 boundaries; first has 2+1=3 → padded to 6.
        assert cu.shape == (2, 6)
        # Padding sentinel must be -1.
        assert (cu[0, n_segs[0].item():] == -1).all()

    def test_n_segs_values(self):
        batch = [self._make_sample(16, 3), self._make_sample(16, 5)]
        _, _, cu, n_segs = packed_collate_fn(batch)
        assert n_segs[0].item() == 4  # 3 segments → 4 boundaries (including 0 and seq_len)
        assert n_segs[1].item() == 6

    def test_x_dtype(self):
        batch = [self._make_sample(8, 2)]
        x, y, _, _ = packed_collate_fn(batch)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_cu_seqlens_dtype(self):
        batch = [self._make_sample(8, 2)]
        _, _, cu, _ = packed_collate_fn(batch)
        assert cu.dtype == torch.int32

    def test_single_sample_no_padding(self):
        batch = [self._make_sample(12, 3)]
        _, _, cu, n_segs = packed_collate_fn(batch)
        assert cu.shape == (1, 4)
        assert (cu[0] != -1).all()


# ---------------------------------------------------------------------------
# AgenticDataMixture
# ---------------------------------------------------------------------------

class TestAgenticDataMixture:
    def _counter_stream(self, label):
        """Infinite iterator that yields {label: i}."""
        i = 0
        while True:
            yield {"label": label, "idx": i}
            i += 1

    def test_yields_from_both_streams(self):
        """With equal weights both streams should be sampled."""
        streams = {
            "A": iter(self._counter_stream("A")),
            "B": iter(self._counter_stream("B")),
        }
        mix = AgenticDataMixture(streams, target_proportions=[1.0, 1.0])
        seen = set()
        for _ in range(200):
            item = next(iter(mix))
            seen.add(item["label"])
        assert "A" in seen and "B" in seen

    def test_skewed_weights_favour_stream(self):
        """A heavily weighted stream should be chosen almost exclusively."""
        import random
        random.seed(0)
        streams = {
            "A": iter(self._counter_stream("A")),
            "B": iter(self._counter_stream("B")),
        }
        mix = AgenticDataMixture(streams, target_proportions=[99.0, 1.0])
        counts = {"A": 0, "B": 0}
        for _ in range(500):
            item = next(iter(mix))
            counts[item["label"]] += 1
        assert counts["A"] > counts["B"] * 5, \
            f"Stream A should dominate: {counts}"

    def test_single_stream(self):
        streams = {"only": iter(self._counter_stream("only"))}
        mix = AgenticDataMixture(streams, target_proportions=[1.0])
        for _ in range(10):
            item = next(iter(mix))
            assert item["label"] == "only"
