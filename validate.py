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

"""Run validation on held-out pillars and log metrics to Weights & Biases.

Example:
    python validate.py --mode scout --manifest-path val_data/validation_manifest.json
    python validate.py --mode scout --checkpoint-path checkpoints/bitmamba_scout/step_010000.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from context_config import CONTEXT_LENGTH
from data import extract_text_from_row
from model import BitMambaLLM, maybe_autocast


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAFE_EXP = 20.0


def resolve_model_config(mode: str) -> Dict[str, int | bool | float]:
    mode = mode.lower()
    if mode == "scout":
        return dict(vocab_size=64000, dim=512, n_layers=24, d_state=64, expand=2, use_checkpoint=True)
    if mode == "parent":
        return dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2, use_checkpoint=True)
    if mode == "upscaled":
        return dict(
            vocab_size=64000,
            dim=1024,
            n_layers=64,
            d_state=128,
            expand=2,
            use_checkpoint=True,
            use_attn=True,
            attn_pct=0.08,
        )
    raise ValueError(f"Unsupported --mode {mode!r}. Expected scout, parent, or upscaled.")


def parse_checkpoint_step(path: str | Path) -> int:
    name = Path(path).name
    match = re.search(r"step_(\d+)\.pt$", name)
    if match:
        return int(match.group(1))
    return -1


def resolve_default_checkpoint_dir(mode: str) -> str:
    return str(Path("checkpoints") / f"bitmamba_{mode.lower()}")


def discover_checkpoints(
    checkpoint_path: str | None,
    checkpoint_dir: str,
    checkpoint_glob: str,
    max_checkpoints: int | None,
) -> List[str]:
    if checkpoint_path:
        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {cp}")
        return [str(cp)]

    pattern = str(Path(checkpoint_dir) / checkpoint_glob)
    found = [p for p in glob.glob(pattern) if Path(p).is_file()]
    if not found:
        raise FileNotFoundError(
            f"No checkpoints found with glob {pattern!r}. "
            "Pass --checkpoint-path or adjust --checkpoint-dir/--checkpoint-glob."
        )

    found.sort(key=lambda p: (parse_checkpoint_step(p), p))
    if max_checkpoints is not None and max_checkpoints > 0:
        found = found[-max_checkpoints:]
    return found


def _resolve_existing_path(path_str: str, manifest_dir: Path) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    by_manifest = (manifest_dir / path_str).resolve()
    if by_manifest.exists():
        return by_manifest

    raise FileNotFoundError(
        f"Could not resolve pillar parquet path {path_str!r}. "
        f"Checked {candidate} and {by_manifest}."
    )


def load_manifest_pillars(manifest_path: str | Path) -> List[Dict[str, str]]:
    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    pillars = manifest.get("pillars", [])
    if not pillars:
        raise ValueError("Manifest has no pillars. Expected a non-empty 'pillars' list.")

    manifest_dir = manifest_file.resolve().parent
    resolved = []
    for item in pillars:
        name = item.get("name")
        path = item.get("path")
        if not name or not path:
            raise ValueError("Each pillar in manifest must include 'name' and 'path'.")
        resolved_path = _resolve_existing_path(path, manifest_dir)
        resolved.append({"name": str(name), "path": str(resolved_path)})
    return resolved


def iter_token_windows(token_ids: Sequence[int], max_seq_len: int) -> Iterable[Tuple[List[int], List[int]]]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    if len(token_ids) < 2:
        return

    upper = len(token_ids) - 1
    for start in range(0, upper, max_seq_len):
        window = token_ids[start : start + max_seq_len + 1]
        if len(window) < 2:
            continue
        yield window[:-1], window[1:]


def _run_loss_batch(model, input_batch, target_batch, device: str):
    max_len = max(len(x) for x in input_batch)
    batch_size = len(input_batch)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    targets = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    for i, (x_ids, y_ids) in enumerate(zip(input_batch, target_batch)):
        n = len(x_ids)
        input_ids[i, :n] = torch.tensor(x_ids, dtype=torch.long, device=device)
        targets[i, :n] = torch.tensor(y_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        with maybe_autocast(device):
            loss_sum, valid_tokens = model(input_ids, targets=targets)

    return float(loss_sum.item()), int(valid_tokens.item())


def evaluate_pillar(
    model,
    tokenizer,
    parquet_path: str,
    *,
    device: str,
    max_seq_len: int,
    batch_size: int,
    max_examples: int | None,
):
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")

    loss_sum_total = 0.0
    token_count_total = 0
    docs_processed = 0
    windows_processed = 0

    pending_inputs: List[List[int]] = []
    pending_targets: List[List[int]] = []

    for row_idx, row in enumerate(dataset):
        if max_examples is not None and row_idx >= max_examples:
            break

        text = extract_text_from_row(row)
        if not text:
            continue

        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            token_ids = token_ids + [int(tokenizer.eos_token_id)]

        has_windows = False
        for x_ids, y_ids in iter_token_windows(token_ids, max_seq_len=max_seq_len):
            has_windows = True
            pending_inputs.append(x_ids)
            pending_targets.append(y_ids)

            if len(pending_inputs) >= batch_size:
                batch_loss_sum, batch_tokens = _run_loss_batch(model, pending_inputs, pending_targets, device)
                loss_sum_total += batch_loss_sum
                token_count_total += batch_tokens
                windows_processed += len(pending_inputs)
                pending_inputs.clear()
                pending_targets.clear()

        if has_windows:
            docs_processed += 1

    if pending_inputs:
        batch_loss_sum, batch_tokens = _run_loss_batch(model, pending_inputs, pending_targets, device)
        loss_sum_total += batch_loss_sum
        token_count_total += batch_tokens
        windows_processed += len(pending_inputs)

    mean_loss = loss_sum_total / max(token_count_total, 1)
    perplexity = float(math.exp(min(mean_loss, MAX_SAFE_EXP)))

    return {
        "loss": float(mean_loss),
        "perplexity": perplexity,
        "tokens": int(token_count_total),
        "documents": int(docs_processed),
        "windows": int(windows_processed),
    }


def evaluate_checkpoint(model, tokenizer, checkpoint_path: str, pillars, args):
    try:
        ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    except (RuntimeError, EOFError, OSError, ValueError, pickle.UnpicklingError) as e:
        print(f"⚠️  Skipping corrupted checkpoint {checkpoint_path}: {e}")
        return False

    if "model_state_dict" not in ckpt:
        print(f"⚠️  Skipping checkpoint {checkpoint_path}: missing 'model_state_dict'")
        return False

    try:
        model.load_state_dict(ckpt["model_state_dict"])
        model.prepare_for_inference()
    except Exception as e:
        print(f"⚠️  Skipping checkpoint {checkpoint_path}: failed to load state dict: {e}")
        return False

    checkpoint_step = int(ckpt.get("step", parse_checkpoint_step(checkpoint_path)))
    checkpoint_tokens = int(ckpt.get("total_tokens", 0))

    overall_loss_sum = 0.0
    overall_tokens = 0

    print(f"Evaluating checkpoint: {checkpoint_path} (step={checkpoint_step})")
    for pillar in pillars:
        pillar_name = pillar["name"]
        pillar_metrics = evaluate_pillar(
            model,
            tokenizer,
            pillar["path"],
            device=args.device,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            max_examples=args.max_examples_per_pillar,
        )

        overall_loss_sum += pillar_metrics["loss"] * pillar_metrics["tokens"]
        overall_tokens += pillar_metrics["tokens"]

        print(
            f"  {pillar_name:<10} loss={pillar_metrics['loss']:.4f} "
            f"ppl={pillar_metrics['perplexity']:.2f} tokens={pillar_metrics['tokens']}"
        )

        prefix = f"Validation/{pillar_name}"
        wandb.log(
            {
                f"{prefix}/Loss": pillar_metrics["loss"],
                f"{prefix}/Perplexity": pillar_metrics["perplexity"],
                f"{prefix}/Tokens": pillar_metrics["tokens"],
                f"{prefix}/Documents": pillar_metrics["documents"],
                f"{prefix}/Windows": pillar_metrics["windows"],
            },
            step=checkpoint_step,
            commit=False,
        )

    overall_loss = overall_loss_sum / max(overall_tokens, 1)
    overall_ppl = float(math.exp(min(overall_loss, MAX_SAFE_EXP)))

    print(f"  overall     loss={overall_loss:.4f} ppl={overall_ppl:.2f} tokens={overall_tokens}")
    wandb.log(
        {
            "Validation/System/CheckpointStep": checkpoint_step,
            "Validation/System/TotalTokens": checkpoint_tokens,
            "Validation/Overall/Loss": overall_loss,
            "Validation/Overall/Perplexity": overall_ppl,
            "Validation/Overall/Tokens": overall_tokens,
        },
        step=checkpoint_step,
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on validation pillars and log to WandB.")
    parser.add_argument("--mode", choices=["scout", "parent", "upscaled"], default="scout")
    parser.add_argument("--manifest-path", default="val_data/validation_manifest.json")
    parser.add_argument("--tokenizer-path", default="custom_agentic_tokenizer")

    parser.add_argument("--checkpoint-path", default=None, help="Evaluate a single checkpoint file.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing step_*.pt checkpoints. Used when --checkpoint-path is not set.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="step_*.pt",
        help="Glob pattern inside --checkpoint-dir for checkpoint discovery.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="If set, evaluate only the latest N discovered checkpoints.",
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=CONTEXT_LENGTH)
    parser.add_argument(
        "--max-examples-per-pillar",
        type=int,
        default=None,
        help="Optional cap for faster smoke runs.",
    )

    parser.add_argument("--wandb-project", default="Agentic-1.58b-Validation")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-resume", default="allow", choices=["allow", "must", "never", "auto"])
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional WANDB_MODE override. If omitted, keeps current environment setting.",
    )
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode

    model_config = resolve_model_config(args.mode)
    checkpoint_dir = args.checkpoint_dir or resolve_default_checkpoint_dir(args.mode)
    checkpoints = discover_checkpoints(
        args.checkpoint_path,
        checkpoint_dir,
        args.checkpoint_glob,
        args.max_checkpoints,
    )
    pillars = load_manifest_pillars(args.manifest_path)

    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = int(1e9)

    print(f"Building model for mode={args.mode} on device={args.device}")
    model = BitMambaLLM(**model_config).to(args.device)
    model.eval()

    run_name = args.wandb_name or f"validation-{args.mode}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        id=args.wandb_run_id,
        resume=args.wandb_resume,
        settings=wandb.Settings(start_method="thread"),
        config={
            "mode": args.mode,
            "manifest_path": args.manifest_path,
            "tokenizer_path": args.tokenizer_path,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_glob": args.checkpoint_glob,
            "checkpoint_count": len(checkpoints),
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "max_examples_per_pillar": args.max_examples_per_pillar,
        },
    )

    try:
        for checkpoint_path in checkpoints:
            success = evaluate_checkpoint(model, tokenizer, checkpoint_path, pillars, args)
            if not success:
                continue
    finally:
        wandb.finish()

    print(f"Validation complete for {len(checkpoints)} checkpoint(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
