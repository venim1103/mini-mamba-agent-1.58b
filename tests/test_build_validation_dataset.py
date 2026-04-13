import json
import types

import pytest

import build_validation_dataset as builder


def test_build_default_specs_applies_row_budget_and_shard_index():
    specs = builder._build_default_specs(rows_per_pillar=128, fineweb_shard_index=42)

    assert [spec["name"] for spec in specs] == ["math", "logic", "code", "tools", "web"]
    assert all(spec["rows"] == 128 for spec in specs)
    assert specs[-1]["output_file"] == "fineweb_edu_shard_0042.parquet"


@pytest.mark.parametrize(
    ("total_rows", "requested_rows", "selection", "expected"),
    [
        (10, 3, "head", [0, 1, 2]),
        (10, 3, "tail", [7, 8, 9]),
    ],
)
def test_resolve_selection_indices(total_rows, requested_rows, selection, expected):
    assert builder._resolve_selection_indices(total_rows, requested_rows, selection) == expected


def test_resolve_selection_indices_rejects_oversized_requests():
    with pytest.raises(ValueError, match="Requested 11 rows"):
        builder._resolve_selection_indices(10, 11, "head")


def test_load_specs_from_config_sets_defaults(tmp_path):
    config_path = tmp_path / "validation_plan.json"
    config_path.write_text(
        json.dumps(
            {
                "pillars": [
                    {
                        "name": "custom_code",
                        "output_subdir": "code",
                        "output_file": "custom.parquet",
                        "source_type": "hf_dataset",
                        "dataset": "mbpp",
                        "split": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = builder._load_specs_from_config(str(config_path), default_rows_per_pillar=55)

    assert specs[0]["rows"] == 55
    assert specs[0]["selection"] == "head"


def test_load_specs_from_config_accepts_top_level_list(tmp_path):
    config_path = tmp_path / "validation_plan_list.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "custom_math",
                    "output_subdir": "math",
                    "output_file": "custom_math.parquet",
                    "source_type": "hf_dataset",
                    "dataset": "gsm8k",
                    "split": "test",
                }
            ]
        ),
        encoding="utf-8",
    )

    specs = builder._load_specs_from_config(str(config_path), default_rows_per_pillar=33)

    assert specs[0]["name"] == "custom_math"
    assert specs[0]["rows"] == 33


def test_validate_specs_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="missing required keys"):
        builder._validate_specs(
            [
                {
                    "name": "broken",
                    "source_type": "hf_dataset",
                    "dataset": "gsm8k",
                    "split": "test",
                    "rows": 10,
                    "selection": "head",
                }
            ]
        )


def test_validate_specs_rejects_unsupported_source_type():
    with pytest.raises(ValueError, match="unsupported source_type"):
        builder._validate_specs(
            [
                {
                    "name": "broken",
                    "output_subdir": "x",
                    "output_file": "x.parquet",
                    "source_type": "nope",
                    "rows": 10,
                    "selection": "head",
                }
            ]
        )


def test_maybe_load_kaggle_secrets_only_populates_missing(monkeypatch):
    class FakeSecretsClient:
        def get_secret(self, name):
            return f"secret-for-{name.lower()}"

    fake_module = types.SimpleNamespace(UserSecretsClient=FakeSecretsClient)
    monkeypatch.setitem(__import__("sys").modules, "kaggle_secrets", fake_module)
    monkeypatch.setenv("HF_TOKEN", "already-set")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    loaded = builder._maybe_load_kaggle_secrets()

    assert loaded == {"KAGGLE_KEY": "loaded", "KAGGLE_USERNAME": "loaded"}
    assert builder.os.environ["HF_TOKEN"] == "already-set"
    assert builder.os.environ["KAGGLE_USERNAME"] == "secret-for-kaggle_username"
    assert builder.os.environ["KAGGLE_KEY"] == "secret-for-kaggle_key"


def test_write_kaggle_metadata(tmp_path):
    metadata_path = builder._write_kaggle_metadata(
        tmp_path,
        dataset_id="owner/mini-mamba-validation",
        title="Mini Mamba Validation",
        license_name="other",
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["id"] == "owner/mini-mamba-validation"
    assert payload["title"] == "Mini Mamba Validation"
    assert payload["licenses"] == [{"name": "other"}]


def test_streaming_select_rows_head():
    records, total = builder._streaming_select_rows(({"i": i} for i in range(10)), 3, "head")

    assert total == 3
    assert [r["i"] for r in records] == [0, 1, 2]


def test_streaming_select_rows_tail():
    records, total = builder._streaming_select_rows(({"i": i} for i in range(10)), 3, "tail")

    assert total == 10
    assert [r["i"] for r in records] == [7, 8, 9]


def test_load_remote_dataset_head_uses_streaming(monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, rows):
            self.num_rows = len(rows)

    monkeypatch.setattr(builder.Dataset, "from_list", lambda rows: FakeDataset(rows))

    def fake_load_dataset(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return ({"x": i} for i in range(5))

    monkeypatch.setattr(builder, "load_dataset", fake_load_dataset)

    spec = {
        "name": "head_probe",
        "dataset": "dummy/ds",
        "split": "train",
        "rows": 3,
        "selection": "head",
    }
    ds, meta = builder._load_remote_dataset(spec, hf_token=None, allow_partial=False)

    assert ds.num_rows == 3
    assert meta["actual_rows"] == 3
    assert meta["streaming"] is True
    assert calls[0][2]["streaming"] is True


def test_load_remote_dataset_tail_partial_allowed(monkeypatch):
    class FakeDataset:
        def __init__(self, rows):
            self.num_rows = len(rows)

    monkeypatch.setattr(builder.Dataset, "from_list", lambda rows: FakeDataset(rows))

    def fake_load_dataset(name, *args, **kwargs):
        return ({"x": i} for i in range(4))

    monkeypatch.setattr(builder, "load_dataset", fake_load_dataset)

    spec = {
        "name": "tail_probe",
        "dataset": "dummy/ds",
        "split": "train",
        "rows": 6,
        "selection": "tail",
    }
    ds, meta = builder._load_remote_dataset(spec, hf_token=None, allow_partial=True)

    assert ds.num_rows == 4
    assert meta["actual_rows"] == 4
    assert meta["total_rows_seen"] == 4


def test_load_remote_dataset_tail_partial_disallowed(monkeypatch):
    def fake_load_dataset(name, *args, **kwargs):
        return ({"x": i} for i in range(4))

    monkeypatch.setattr(builder, "load_dataset", fake_load_dataset)

    spec = {
        "name": "tail_probe",
        "dataset": "dummy/ds",
        "split": "train",
        "rows": 6,
        "selection": "tail",
    }
    with pytest.raises(ValueError, match="requested 6 rows"):
        builder._load_remote_dataset(spec, hf_token=None, allow_partial=False)