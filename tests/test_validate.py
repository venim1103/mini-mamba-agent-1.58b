import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import validate


def test_parse_checkpoint_step_from_filename():
    assert validate.parse_checkpoint_step("checkpoints/bitmamba_scout/step_001234.pt") == 1234


def test_parse_checkpoint_step_returns_minus_one_for_nonstandard_name():
    assert validate.parse_checkpoint_step("checkpoints/bitmamba_scout/latest.pt") == -1


def test_iter_token_windows_splits_into_context_chunks():
    token_ids = [1, 2, 3, 4, 5, 6, 7]
    windows = list(validate.iter_token_windows(token_ids, max_seq_len=3))
    assert windows == [
        ([1, 2, 3], [2, 3, 4]),
        ([4, 5, 6], [5, 6, 7]),
    ]


def test_load_manifest_pillars_resolves_relative_paths(tmp_path: Path):
    pillar_file = tmp_path / "math" / "gsm8k.parquet"
    pillar_file.parent.mkdir(parents=True, exist_ok=True)
    pillar_file.write_text("placeholder", encoding="utf-8")

    manifest = {
        "pillars": [
            {"name": "math", "path": "math/gsm8k.parquet"},
        ]
    }
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pillars = validate.load_manifest_pillars(manifest_path)
    assert len(pillars) == 1
    assert pillars[0]["name"] == "math"
    assert Path(pillars[0]["path"]).resolve() == pillar_file.resolve()


def test_load_manifest_pillars_raises_when_file_missing(tmp_path: Path):
    manifest = {
        "pillars": [
            {"name": "math", "path": "missing.parquet"},
        ]
    }
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        validate.load_manifest_pillars(manifest_path)


def test_evaluate_checkpoint_handles_corrupted_file(tmp_path: Path):
    model = MagicMock()
    tokenizer = MagicMock()
    args = MagicMock(device="cpu")
    pillars = [{"name": "test", "path": "test.parquet"}]
    
    corrupted_path = str(tmp_path / "corrupted.pt")
    with open(corrupted_path, "w") as f:
        f.write("this is not a valid torch checkpoint")
    
    with patch("builtins.print"):
        result = validate.evaluate_checkpoint(model, tokenizer, corrupted_path, pillars, args)
    
    assert result is False
    model.load_state_dict.assert_not_called()


def test_evaluate_checkpoint_handles_missing_model_state_dict(tmp_path: Path):
    model = MagicMock()
    tokenizer = MagicMock()
    args = MagicMock(device="cpu")
    pillars = [{"name": "test", "path": "test.parquet"}]
    
    checkpoint_path = str(tmp_path / "missing_state.pt")
    torch.save({"step": 100}, checkpoint_path)
    
    with patch("builtins.print"):
        result = validate.evaluate_checkpoint(model, tokenizer, checkpoint_path, pillars, args)
    
    assert result is False
    model.load_state_dict.assert_not_called()
