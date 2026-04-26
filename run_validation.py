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

"""run_validation.py — Download checkpoints and run validation across environments.

Works in Kaggle notebooks, Google Colab, and local machines.
All settings are controlled by environment variables so no code changes are needed
between environments — just set the variables in a cell above this script.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED ENVIRONMENT VARIABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HF_TOKEN          Hugging Face token (for private repos / higher rate limits)
  WANDB_API_KEY     Weights & Biases API key
  VAL_LOCAL_BASE    Absolute path to your cloned repo root. Must be set
                    explicitly — no auto-detection, as it differs per machine.

                    Typical values per environment:
                      Kaggle : /kaggle/working/mini-mamba-agent-1.58b
                      Colab  : /content/mini-mamba-agent-1.58b
                      Local  : /home/you/projects/mini-mamba-agent-1.58b

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKPOINT SELECTION  (pick ONE of the three modes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VAL_N_LATEST      Download the N most recent checkpoints.          e.g. "5"
  VAL_STEPS         Comma-separated specific step numbers.           e.g. "10000,50000,100000"
  VAL_INDICES       Comma-separated indices from the sorted list.    e.g. "0,5,-1"
                    (If none of the three are set, defaults to VAL_N_LATEST=5)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIONAL ENVIRONMENT VARIABLES  (all have sensible defaults)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VAL_REPO_ID           HF repo with checkpoints.     default: "venim1103/mini-mamba-agent-1b58"
  VAL_CKPT_SUBDIR       Subdir inside the repo.        default: "checkpoints/bitmamba_scout"
  VAL_MANIFEST_PATH     Path to validation_manifest.   default: "<VAL_LOCAL_BASE>/val_data/validation_manifest.json"
  VAL_MODE              Model size: scout/parent/upscaled.  default: "scout"
  VAL_BATCH_SIZE        Eval batch size.               default: "4"
  VAL_WANDB_PROJECT     W&B project name.              default: "Agentic-1.58b-Validation"
  VAL_WANDB_MODE        online / offline / disabled.   default: "online"
  VAL_DRY_RUN           "1" to list checkpoints only, skip download and eval.
  VAL_SKIP_EVAL         "1" to download only, skip running validate.py.
  VAL_LIST_ONLY         "1" to list all available checkpoints and exit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK START EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Kaggle / Colab — setup cell:
  import os
  os.environ["VAL_LOCAL_BASE"] = "/kaggle/working/mini-mamba-agent-1.58b"
  os.environ["HF_TOKEN"]       = "hf_..."   # or loaded from secrets automatically
  os.environ["WANDB_API_KEY"]  = "..."
  os.environ["VAL_N_LATEST"]   = "5"
  # then in the next cell: %run run_validation.py

  # Bash (local or terminal):
  VAL_LOCAL_BASE=/home/you/projects/mini-mamba-agent-1.58b \\
  VAL_N_LATEST=3 VAL_MODE=scout HF_TOKEN=hf_... \\
  python run_validation.py

  # List all available checkpoints without downloading:
  VAL_LOCAL_BASE=... VAL_LIST_ONLY=1 python run_validation.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment / secret loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_secrets() -> None:
    """Try to pull secrets from notebook providers if env vars are missing.

    Tries Kaggle first, then Colab. Each provider is tried independently so
    that a partial Kaggle result (e.g. only HF_TOKEN found) still allows the
    remaining keys to be loaded from Colab if available.
    """
    def _still_missing() -> list[str]:
        return [k for k in ("HF_TOKEN", "WANDB_API_KEY") if not os.environ.get(k)]

    if not _still_missing():
        return

    # --- Kaggle ---
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for key in _still_missing():
            try:
                value = client.get_secret(key)
                if value:
                    os.environ[key] = value
                    print(f"  [secrets] Loaded {key} from Kaggle secrets.")
            except Exception:
                pass
    except ImportError:
        pass

    # --- Colab (always attempted in case Kaggle didn't cover everything) ---
    if not _still_missing():
        return
    try:
        from google.colab import userdata
        for key in _still_missing():
            try:
                value = userdata.get(key)
                if value:
                    os.environ[key] = value
                    print(f"  [secrets] Loaded {key} from Colab userdata.")
            except Exception:
                pass
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 2. Config — read everything from environment
# ══════════════════════════════════════════════════════════════════════════════

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _flag(key: str) -> bool:
    return _env(key).lower() in {"1", "true", "yes", "on"}


def build_config() -> dict:
    local_base = _env("VAL_LOCAL_BASE")
    if not local_base:
        print(
            "[error] VAL_LOCAL_BASE is not set.\n"
            "        Set it to the absolute path of your cloned repo, e.g.:\n"
            "          Kaggle : /kaggle/working/mini-mamba-agent-1.58b\n"
            "          Colab  : /content/mini-mamba-agent-1.58b\n"
            "          Local  : /home/you/projects/mini-mamba-agent-1.58b"
        )
        raise SystemExit(1)

    ckpt_subdir = _env("VAL_CKPT_SUBDIR", "checkpoints/bitmamba_scout")
    default_manifest = str(Path(local_base) / "val_data" / "validation_manifest.json")

    return {
        # Source
        "repo_id":        _env("VAL_REPO_ID",      "venim1103/mini-mamba-agent-1b58"),
        "ckpt_subdir":    ckpt_subdir,
        "hf_token":       _env("HF_TOKEN"),
        # Destination
        "local_base":     local_base,
        "local_ckpt_dir": str(Path(local_base) / ckpt_subdir),
        # Selection
        "n_latest":       _env("VAL_N_LATEST"),
        "steps":          _env("VAL_STEPS"),
        "indices":        _env("VAL_INDICES"),
        # Validation
        "manifest_path":  _env("VAL_MANIFEST_PATH", default_manifest),
        "mode":           _env("VAL_MODE",          "scout"),
        "batch_size":     _env("VAL_BATCH_SIZE",    "4"),
        "wandb_project":  _env("VAL_WANDB_PROJECT", "Agentic-1.58b-Validation"),
        "wandb_mode":     _env("VAL_WANDB_MODE",    "online"),
        # Behaviour flags
        "dry_run":        _flag("VAL_DRY_RUN"),
        "skip_eval":      _flag("VAL_SKIP_EVAL"),
        "list_only":      _flag("VAL_LIST_ONLY"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Checkpoint discovery
# ══════════════════════════════════════════════════════════════════════════════

def _step_from_path(path: str) -> int:
    m = re.search(r"step_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def list_remote_checkpoints(cfg: dict) -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi(token=cfg["hf_token"] or None)
    all_files = [
        f for f in api.list_repo_files(cfg["repo_id"], repo_type="model")
        if f.startswith(cfg["ckpt_subdir"]) and f.endswith(".pt")
    ]
    all_files.sort(key=_step_from_path)
    return all_files


def select_checkpoints(all_files: list[str], cfg: dict) -> list[str]:
    """Apply the user's selection mode and return the chosen subset."""
    # Priority: VAL_STEPS > VAL_INDICES > VAL_N_LATEST (default 5)
    if cfg["steps"]:
        wanted = {int(s.strip()) for s in cfg["steps"].split(",") if s.strip()}
        selected = [f for f in all_files if _step_from_path(f) in wanted]
        if not selected:
            print(f"  [warn] No checkpoints matched steps {wanted}. Nothing to download.")
        return selected

    if cfg["indices"]:
        raw = [i.strip() for i in cfg["indices"].split(",") if i.strip()]
        selected = []
        for raw_idx in raw:
            try:
                selected.append(all_files[int(raw_idx)])
            except IndexError:
                print(f"  [warn] Index {raw_idx} out of range (0–{len(all_files)-1}), skipping.")
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for f in selected:
            if f not in seen:
                seen.add(f)
                deduped.append(f)
        return deduped

    # Default: last N
    n = int(cfg["n_latest"]) if cfg["n_latest"] else 5
    return all_files[-n:]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Download
# ══════════════════════════════════════════════════════════════════════════════

def download_checkpoints(selected: list[str], cfg: dict) -> list[str]:
    from huggingface_hub import hf_hub_download

    local_ckpt_dir = cfg["local_ckpt_dir"]
    os.makedirs(local_ckpt_dir, exist_ok=True)

    local_paths: list[str] = []
    for repo_path in selected:
        local_path = Path(cfg["local_base"]) / repo_path
        if local_path.exists():
            print(f"  [skip]     {repo_path}  (already on disk)")
        else:
            print(f"  [download] {repo_path}")
            if not cfg["dry_run"]:
                hf_hub_download(
                    repo_id=cfg["repo_id"],
                    filename=repo_path,
                    repo_type="model",
                    local_dir=cfg["local_base"],
                    token=cfg["hf_token"] or None,
                )
        local_paths.append(str(local_path))

    return local_paths


# ══════════════════════════════════════════════════════════════════════════════
# 5. Run validate.py
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(cfg: dict, n_checkpoints: int) -> int:
    validate_script = Path(cfg["local_base"]) / "validate.py"
    if not validate_script.exists():
        print(f"\n[error] validate.py not found at {validate_script}")
        print("        Clone the repo first or adjust VAL_LOCAL_BASE.")
        return 1

    manifest = cfg["manifest_path"]
    if not Path(manifest).exists():
        print(f"\n[error] Manifest not found: {manifest}")
        print("        Run build_validation_dataset.py first, or set VAL_MANIFEST_PATH.")
        return 1

    cmd = [
        sys.executable, str(validate_script),
        "--mode",           cfg["mode"],
        "--manifest-path",  manifest,
        "--checkpoint-dir", cfg["local_ckpt_dir"],
        "--max-checkpoints", str(n_checkpoints),
        "--batch-size",     cfg["batch_size"],
        "--wandb-project",  cfg["wandb_project"],
        "--wandb-mode",     cfg["wandb_mode"],
    ]

    print(f"\n[validate] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cfg["local_base"])
    return result.returncode


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 70)
    print("  Mini Mamba — Checkpoint Downloader + Validation Runner")
    print("=" * 70)

    # Load secrets from Kaggle / Colab if not already in env
    _load_secrets()

    cfg = build_config()

    print(f"\n  Repo        : {cfg['repo_id']}")
    print(f"  Subdir      : {cfg['ckpt_subdir']}")
    print(f"  Local base  : {cfg['local_base']}")
    print(f"  Mode        : {cfg['mode']}")
    print(f"  Manifest    : {cfg['manifest_path']}")
    if cfg["dry_run"]:   print("  [DRY RUN — no files will be downloaded or evaluated]")
    if cfg["skip_eval"]: print("  [SKIP EVAL — will download but not run validate.py]")

    # ── List available checkpoints ────────────────────────────────────────────
    print(f"\n[1/3] Listing checkpoints in {cfg['repo_id']} / {cfg['ckpt_subdir']} …")
    try:
        all_files = list_remote_checkpoints(cfg)
    except Exception as exc:
        print(f"[error] Could not reach Hugging Face: {exc}")
        print("        Check HF_TOKEN and your internet connection.")
        return 1

    if not all_files:
        print("[error] No .pt checkpoints found in the specified repo/subdir.")
        return 1

    print(f"\n  {'#':>4}  {'step':>8}   path")
    print(f"  {'─'*4}  {'─'*8}   {'─'*50}")
    for i, f in enumerate(all_files):
        print(f"  [{i:>3}]  {_step_from_path(f):>8,}   {f}")

    if cfg["list_only"]:
        print("\n[list-only mode] Done.")
        return 0

    # ── Select ────────────────────────────────────────────────────────────────
    selected = select_checkpoints(all_files, cfg)
    if not selected:
        return 1

    print(f"\n[2/3] Selected {len(selected)} checkpoint(s):")
    for f in selected:
        print(f"  step {_step_from_path(f):>8,}   {f}")

    # ── Download ──────────────────────────────────────────────────────────────
    print()
    downloaded = download_checkpoints(selected, cfg)
    if not downloaded:
        print("[error] No files were downloaded.")
        return 1

    if cfg["dry_run"] or cfg["skip_eval"]:
        print("\nDone (eval skipped).")
        return 0

    # ── Validate ──────────────────────────────────────────────────────────────
    print(f"\n[3/3] Running validate.py on {len(selected)} checkpoint(s) …")
    rc = run_validation(cfg, n_checkpoints=len(selected))
    print(f"\nvalidate.py exited with code {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())