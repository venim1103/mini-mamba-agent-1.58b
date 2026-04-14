# Copyright 2026 venim1103
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a portable held-out validation dataset for pretraining/eval runs.

The default preset expands the original 3-pillar idea into 5 evaluation pillars:
math, logic, code, tool use, and web text. This is intentionally evaluation-side
granularity: the current pretraining recipe in train.py has 4 top-level buckets,
with math and logic both feeding the formal_logic domain.

Design goals:
- Run the same way locally, in Colab, and in Kaggle.
- Avoid Colab-only secret handling.
- Keep the source plan explicit and configurable.
- Emit a manifest so downstream validation runs know exactly what was built.

Example:
    python build_validation_dataset.py --output-dir val_data
    python build_validation_dataset.py --upload-kaggle \
        --kaggle-dataset-id your-name/mini-mamba-1b58-validation
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_dataset, load_dataset_builder
from huggingface_hub import HfFileSystem


DEFAULT_PRESET = "balanced_5pillar"
DEFAULT_ROWS_PER_PILLAR = 900
DEFAULT_FINEWEB_GLOB = "datasets/HuggingFaceFW/fineweb-edu/data/*.parquet"
DEFAULT_FINEWEB_GLOB_CANDIDATES = (
    "datasets/HuggingFaceFW/fineweb-edu/data/*.parquet",
    "datasets/HuggingFaceFW/fineweb-edu/*/*/*.parquet",
    "datasets/HuggingFaceFW/fineweb-edu/**/**/*.parquet",
)


def _build_default_specs(rows_per_pillar=DEFAULT_ROWS_PER_PILLAR, fineweb_shard_index=500):
    return [
        {
            "name": "math",
            "output_subdir": "math",
            "output_file": "gsm8k_test.parquet",
            "source_type": "hf_dataset",
            "dataset": "gsm8k",
            "config": "main",
            "split": "test",
            "selection": "head",
            "rows": rows_per_pillar,
            "notes": "External math reasoning probe.",
        },
        {
            "name": "logic",
            "output_subdir": "logic",
            "output_file": "sciq_validation.parquet",
            "source_type": "hf_dataset",
            "dataset": "allenai/sciq",
            "split": "validation",
            "selection": "head",
            "rows": rows_per_pillar,
            "notes": (
                "Reasoning/science proxy for the repo's broader formal_logic bucket. "
                "Override via --config-file if you want a stricter logic benchmark."
            ),
        },
        {
            "name": "code",
            "output_subdir": "code",
            "output_file": "mbpp_test.parquet",
            "source_type": "hf_dataset",
            "dataset": "mbpp",
            "split": "test",
            "selection": "head",
            "rows": rows_per_pillar,
            "max_rows_hint": 500,
            "notes": "Held-out code generation benchmark.",
        },
        {
            "name": "tools",
            "output_subdir": "tools",
            "output_file": "glaive_function_calling_tail.parquet",
            "source_type": "hf_dataset",
            "dataset": "glaiveai/glaive-function-calling-v2",
            "split": "train",
            "selection": "tail",
            "rows": rows_per_pillar,
            "notes": "Tail slice from a tool-calling corpus external to the repo's Toolformer source.",
        },
        {
            "name": "web",
            "output_subdir": "web",
            "output_file": f"fineweb_edu_shard_{fineweb_shard_index:04d}.parquet",
            "source_type": "fineweb_glob",
            "file_glob": DEFAULT_FINEWEB_GLOB,
            "shard_index": fineweb_shard_index,
            "selection": "head",
            "rows": rows_per_pillar,
            "notes": (
                "General web-text probe sourced from a deep full-dataset shard. "
                "This reduces overlap risk with sample-10BT training data but does not prove zero overlap."
            ),
        },
    ]


PRESET_REGISTRY = {
    DEFAULT_PRESET: {
        "description": (
            "Balanced 5-pillar validation set spanning math, logic, code, tool use, and web text."
        ),
        "builder": _build_default_specs,
    }
}


def _resolve_selection_indices(total_rows, requested_rows, selection):
    if requested_rows <= 0:
        raise ValueError("requested_rows must be > 0")
    if total_rows <= 0:
        raise ValueError("total_rows must be > 0")
    if requested_rows > total_rows:
        raise ValueError(
            f"Requested {requested_rows} rows from a dataset with only {total_rows} rows."
        )
    if selection == "head":
        return list(range(requested_rows))
    if selection == "tail":
        return list(range(total_rows - requested_rows, total_rows))
    raise ValueError(f"Unsupported selection strategy: {selection!r}")


def _load_specs_from_config(config_file, default_rows_per_pillar):
    with open(config_file, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        specs = raw
    elif isinstance(raw, dict):
        specs = raw.get("pillars", raw)
    else:
        raise ValueError("Config file must contain a JSON list or object.")

    if not isinstance(specs, list):
        raise ValueError("Config file must be a JSON list or an object with a 'pillars' list.")

    normalized = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each pillar spec must be a JSON object.")
        updated = dict(spec)
        updated.setdefault("rows", default_rows_per_pillar)
        updated.setdefault("selection", "head")
        normalized.append(updated)
    return normalized


def _validate_specs(specs):
    common_required = ("name", "output_subdir", "output_file", "source_type", "rows", "selection")

    for idx, spec in enumerate(specs):
        missing_common = [key for key in common_required if key not in spec]
        if missing_common:
            raise ValueError(f"Pillar spec #{idx} missing required keys: {', '.join(sorted(missing_common))}")

        try:
            rows = int(spec["rows"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pillar spec #{idx} has non-integer rows={spec['rows']!r}") from exc
        if rows <= 0:
            raise ValueError(f"Pillar spec #{idx} must have rows > 0 (got {rows}).")

        source_type = spec["source_type"]
        if source_type == "hf_dataset":
            required = ("dataset", "split")
        elif source_type == "fineweb_glob":
            required = ("file_glob",)
        else:
            raise ValueError(f"Pillar spec #{idx} has unsupported source_type={source_type!r}")

        missing_source = [key for key in required if key not in spec]
        if missing_source:
            raise ValueError(
                f"Pillar spec #{idx} ({spec['name']!r}) missing source keys: {', '.join(sorted(missing_source))}"
            )


def _rebalance_rows_to_min_capacity(specs):
    raise NotImplementedError("Use _rebalance_rows_to_min_capacity_with_probe instead.")


def _probe_hf_dataset_capacity(spec, hf_token):
    """Return split capacity (num_examples) without materializing the dataset."""
    dataset_name = spec["dataset"]
    config_name = spec.get("config")
    split_name = spec["split"]

    try:
        if config_name:
            builder = load_dataset_builder(dataset_name, config_name, token=hf_token)
        else:
            builder = load_dataset_builder(dataset_name, token=hf_token)
    except Exception:
        return None

    split_info = getattr(builder.info, "splits", {}).get(split_name)
    if split_info is None:
        return None
    num_examples = getattr(split_info, "num_examples", None)
    if num_examples is None:
        return None
    try:
        return int(num_examples)
    except (TypeError, ValueError):
        return None


def _rebalance_rows_to_min_capacity_with_probe(specs, hf_token=None, probe_capacities=True):
    """Balance row budgets using requested rows, hints, and optional split probing.

    Returns:
        balanced_rows: int
        limiting: list[dict] entries for pillars that determined the minimum
        details: list[dict] per-pillar effective limits and reasons
    """
    details = []
    for spec in specs:
        requested_rows = int(spec["rows"])
        effective_rows = requested_rows
        limiters = [f"requested={requested_rows}"]

        hint = spec.get("max_rows_hint")
        if hint is not None:
            hint_int = int(hint)
            if hint_int < effective_rows:
                effective_rows = hint_int
            limiters.append(f"max_rows_hint={hint_int}")

        if probe_capacities and spec.get("source_type") == "hf_dataset":
            probed = _probe_hf_dataset_capacity(spec, hf_token)
            if probed is not None:
                if probed < effective_rows:
                    effective_rows = probed
                limiters.append(f"probed_split_size={probed}")

        details.append(
            {
                "name": spec.get("name", "unknown"),
                "effective_rows": effective_rows,
                "limiters": limiters,
            }
        )

    balanced_rows = min(d["effective_rows"] for d in details)
    limiting = [d for d in details if d["effective_rows"] == balanced_rows]

    for spec in specs:
        spec["rows"] = balanced_rows

    return balanced_rows, limiting, details


def _maybe_load_kaggle_secrets(secret_names=("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")):
    try:
        kaggle_secrets = importlib.import_module("kaggle_secrets")
    except ImportError:
        return {}

    client = kaggle_secrets.UserSecretsClient()
    loaded = {}
    for secret_name in secret_names:
        if os.environ.get(secret_name):
            continue
        try:
            value = client.get_secret(secret_name)
        except Exception:
            value = None
        if value:
            os.environ[secret_name] = value
            loaded[secret_name] = "loaded"
    return loaded


def _resolve_hf_token(env_var_name):
    return os.environ.get(env_var_name) or os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def _streaming_select_rows(stream, rows, selection):
    if selection == "head":
        records = list(itertools.islice(stream, rows))
        return records, len(records)

    if selection == "tail":
        buffer = deque(maxlen=rows)
        total_rows = 0
        for item in stream:
            buffer.append(item)
            total_rows += 1
        return list(buffer), total_rows

    raise ValueError(f"Unsupported selection strategy: {selection!r}")


def _load_remote_dataset(spec, hf_token, allow_partial=False):
    dataset_name = spec["dataset"]
    config_name = spec.get("config")
    split_name = spec["split"]
    rows = int(spec["rows"])
    selection = spec.get("selection", "head")

    load_kwargs = {
        "split": split_name,
        "token": hf_token,
        "streaming": True,
    }
    if config_name:
        stream = load_dataset(dataset_name, config_name, **load_kwargs)
    else:
        stream = load_dataset(dataset_name, **load_kwargs)

    records, total_rows_seen = _streaming_select_rows(stream, rows, selection)
    actual_rows = len(records)
    if rows > actual_rows and not allow_partial:
        raise ValueError(
            f"Pillar {spec['name']!r} requested {rows} rows, but {dataset_name}:{split_name} only yielded {actual_rows}."
        )
    if not records:
        raise ValueError(
            f"Pillar {spec['name']!r} yielded zero rows from {dataset_name}:{split_name}."
        )

    return Dataset.from_list(records), {
        "resolved_split": split_name,
        "selection": selection,
        "requested_rows": int(spec["rows"]),
        "actual_rows": actual_rows,
        "total_rows_seen": total_rows_seen,
        "streaming": True,
    }


def _load_fineweb_shard(spec, hf_token, allow_partial=False):
    rows = int(spec["rows"])
    shard_index = int(spec.get("shard_index", 0))
    fs = HfFileSystem(token=hf_token)

    parquet_files = []
    tried_globs = []
    candidate_globs = [spec.get("file_glob", DEFAULT_FINEWEB_GLOB)]
    for fallback in DEFAULT_FINEWEB_GLOB_CANDIDATES:
        if fallback not in candidate_globs:
            candidate_globs.append(fallback)

    for pattern in candidate_globs:
        tried_globs.append(pattern)
        parquet_files = sorted(fs.glob(pattern))
        if parquet_files:
            matched_glob = pattern
            break
    else:
        matched_glob = None

    if not parquet_files:
        raise RuntimeError(
            "No FineWeb parquet files matched any known glob pattern. "
            f"Tried: {tried_globs}"
        )

    if shard_index < 0:
        shard_index += len(parquet_files)
    if shard_index < 0 or shard_index >= len(parquet_files):
        raise ValueError(
            f"Shard index {shard_index} is out of range for {len(parquet_files)} files."
        )

    resolved_file = "hf://" + parquet_files[shard_index]
    stream = load_dataset(
        "parquet",
        data_files=resolved_file,
        split="train",
        streaming=True,
        token=hf_token,
    )
    records = list(itertools.islice(stream, rows))
    if not records:
        raise ValueError(
            f"Pillar {spec['name']!r} produced zero rows from shard {shard_index}. "
            "Pick a different shard or reduce --rows-per-pillar."
        )
    if len(records) < rows and not allow_partial:
        raise ValueError(
            f"Pillar {spec['name']!r} requested {rows} rows, but shard only produced {len(records)}."
        )
    return Dataset.from_list(records), {
        "resolved_file": resolved_file,
        "selection": "head",
        "requested_rows": rows,
        "actual_rows": len(records),
        "shard_count": len(parquet_files),
        "shard_index": shard_index,
        "matched_glob": matched_glob,
        "tried_globs": tried_globs,
    }


def _materialize_pillar(spec, output_dir, hf_token, allow_partial=False):
    source_type = spec["source_type"]
    if source_type == "hf_dataset":
        dataset, source_meta = _load_remote_dataset(spec, hf_token, allow_partial=allow_partial)
    elif source_type == "fineweb_glob":
        dataset, source_meta = _load_fineweb_shard(spec, hf_token, allow_partial=allow_partial)
    else:
        raise ValueError(f"Unsupported source_type: {source_type!r}")

    pillar_dir = Path(output_dir) / spec["output_subdir"]
    pillar_dir.mkdir(parents=True, exist_ok=True)
    out_path = pillar_dir / spec["output_file"]
    dataset.to_parquet(str(out_path))

    return {
        "name": spec["name"],
        "path": str(out_path),
        "rows": int(dataset.num_rows),
        "source_type": source_type,
        "source": {k: v for k, v in spec.items() if k not in {"output_subdir", "output_file"}},
        "source_meta": source_meta,
        "notes": spec.get("notes", ""),
    }


def _write_manifest(output_dir, manifest):
    manifest_path = Path(output_dir) / "validation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def _write_kaggle_metadata(output_dir, dataset_id, title, license_name="other"):
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
    }
    metadata_path = Path(output_dir) / "dataset-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def _run_kaggle_cli(output_dir, *, update_existing, dir_mode, message=None):
    base_cmd = ["kaggle", "datasets"]
    if update_existing:
        cmd = base_cmd + ["version", "-p", str(output_dir), "--dir-mode", dir_mode]
        if message:
            cmd += ["-m", message]
    else:
        cmd = base_cmd + ["create", "-p", str(output_dir), "--dir-mode", dir_mode]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle CLI command failed with exit code {result.returncode}: {' '.join(cmd)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a portable validation dataset bundle.")
    parser.add_argument("--output-dir", default="val_data", help="Output directory for parquet files and manifest.")
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        choices=sorted(PRESET_REGISTRY),
        help="Built-in source plan to use when --config-file is not provided.",
    )
    parser.add_argument(
        "--config-file",
        help="Optional JSON config overriding the built-in preset. Accepts a list or {'pillars': [...]}.",
    )
    parser.add_argument(
        "--rows-per-pillar",
        type=int,
        default=DEFAULT_ROWS_PER_PILLAR,
        help="Default number of rows per pillar for built-in presets.",
    )
    parser.add_argument(
        "--rebalance-to-min",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep pillar sizes balanced by lowering all row counts to the smallest "
            "feasible target across selected sources."
        ),
    )
    parser.add_argument(
        "--probe-capacities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Probe HF dataset split capacities (num_examples) to choose a safe "
            "balanced row target before building pillars."
        ),
    )
    parser.add_argument(
        "--fineweb-shard-index",
        type=int,
        default=500,
        help="Shard index used by the built-in FineWeb pillar.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable name that stores the Hugging Face token.",
    )
    parser.add_argument(
        "--allow-partial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow a pillar to emit fewer rows than requested if the source is smaller.",
    )
    parser.add_argument(
        "--load-kaggle-secrets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="On Kaggle notebooks, populate missing HF/Kaggle env vars from kaggle_secrets.",
    )
    parser.add_argument(
        "--upload-kaggle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create or update a Kaggle dataset after building the files.",
    )
    parser.add_argument("--kaggle-dataset-id", help="Kaggle dataset id in the form owner/slug.")
    parser.add_argument(
        "--kaggle-title",
        default="Mini Mamba 1.58b Validation Set",
        help="Human-readable Kaggle dataset title.",
    )
    parser.add_argument(
        "--kaggle-license",
        default="other",
        help="Kaggle metadata license code (for mixed external sources use 'other').",
    )
    parser.add_argument(
        "--kaggle-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 'kaggle datasets version' instead of 'kaggle datasets create'.",
    )
    parser.add_argument(
        "--kaggle-dir-mode",
        choices=["skip", "zip", "tar"],
        default="zip",
        help="Packaging mode forwarded to the Kaggle CLI.",
    )
    parser.add_argument(
        "--kaggle-message",
        default="Refresh validation bundle",
        help="Version message used when --kaggle-update is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.load_kaggle_secrets:
        loaded = _maybe_load_kaggle_secrets()
        if loaded:
            print(f"Loaded missing Kaggle secrets: {', '.join(sorted(loaded))}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_file:
        specs = _load_specs_from_config(args.config_file, args.rows_per_pillar)
        preset_name = "custom"
        preset_description = f"Custom config loaded from {args.config_file}"
    else:
        preset = PRESET_REGISTRY[args.preset]
        specs = preset["builder"](
            rows_per_pillar=args.rows_per_pillar,
            fineweb_shard_index=args.fineweb_shard_index,
        )
        preset_name = args.preset
        preset_description = preset["description"]

    _validate_specs(specs)

    if args.rebalance_to_min:
        balanced_rows, limiting, _details = _rebalance_rows_to_min_capacity_with_probe(
            specs,
            hf_token=_resolve_hf_token(args.hf_token_env),
            probe_capacities=args.probe_capacities,
        )
        print(f"Using balanced row target per pillar: {balanced_rows}")
        for entry in limiting:
            print(f"  constrained by {entry['name']}: {', '.join(entry['limiters'])}")

    hf_token = _resolve_hf_token(args.hf_token_env)
    results = []
    for spec in specs:
        print(f"Building pillar: {spec['name']} -> {spec['output_subdir']}/{spec['output_file']}")
        result = _materialize_pillar(spec, output_dir, hf_token, allow_partial=args.allow_partial)
        print(f"  wrote {result['rows']} rows to {result['path']}")
        results.append(result)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset_name,
        "preset_description": preset_description,
        "rows_per_pillar_default": args.rows_per_pillar,
        "training_alignment_notes": [
            "Pretraining mix is defined in train.py, not context_config.py.",
            "The current pretraining recipe has 4 top-level buckets; this validation plan intentionally splits formal_logic into separate math and logic probes.",
            "Balanced 1:1:1:1:1 pillar weighting is an evaluation choice, not a training-mixture requirement.",
            "Deep-shard FineWeb sampling lowers overlap risk with sample-10BT training data but is not a formal zero-overlap proof.",
        ],
        "pillars": results,
    }
    manifest_path = _write_manifest(output_dir, manifest)
    print(f"Wrote manifest: {manifest_path}")

    if args.upload_kaggle:
        if not args.kaggle_dataset_id:
            raise ValueError("--kaggle-dataset-id is required when --upload-kaggle is enabled.")
        metadata_path = _write_kaggle_metadata(
            output_dir,
            args.kaggle_dataset_id,
            args.kaggle_title,
            license_name=args.kaggle_license,
        )
        print(f"Wrote Kaggle metadata: {metadata_path}")
        _run_kaggle_cli(
            output_dir,
            update_existing=args.kaggle_update,
            dir_mode=args.kaggle_dir_mode,
            message=args.kaggle_message,
        )
        print("Kaggle upload completed.")


if __name__ == "__main__":
    main()