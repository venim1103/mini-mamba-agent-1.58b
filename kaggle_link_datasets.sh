#!/bin/bash
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

set -u

usage() {
    cat <<'USAGE'
Usage: ./kaggle_link_datasets.sh [--env auto|kaggle|colab] [--colab-version N]

Options:
  --env auto|kaggle|colab   Runtime environment selection (default: auto).
  --colab-version N         Dataset version to use in Colab kagglehub paths
                            (default: 1).

Environment overrides (optional):
  KAGGLE_INPUT_ROOT         Override Kaggle input root (default: /kaggle/input).
  COLAB_KAGGLEHUB_ROOT      Override Colab kagglehub root (default: /root/.cache/kagglehub).
USAGE
}

MODE="auto"
COLAB_DATASET_VERSION="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            MODE="${2:-}"
            shift 2
            ;;
        --colab-version)
            COLAB_DATASET_VERSION="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "auto" && "$MODE" != "kaggle" && "$MODE" != "colab" ]]; then
    echo "Invalid --env value: $MODE"
    usage
    exit 1
fi

KAGGLE_INPUT_ROOT="${KAGGLE_INPUT_ROOT:-/kaggle/input}"
COLAB_KAGGLEHUB_ROOT="${COLAB_KAGGLEHUB_ROOT:-/root/.cache/kagglehub}"

# ==========================================
# 1. ENVIRONMENT CHECK / SELECTION
# ==========================================
if [[ "$MODE" == "auto" ]]; then
    if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "$KAGGLE_INPUT_ROOT" ]]; then
        MODE="kaggle"
    elif [[ -n "${COLAB_RELEASE_TAG:-}" || -d "$COLAB_KAGGLEHUB_ROOT/datasets" ]]; then
        MODE="colab"
    else
        echo "Not running in a detected Kaggle/Colab environment. Skipping dataset linking."
        exit 0
    fi
fi

echo "Environment: $MODE"

if [[ "$MODE" == "kaggle" ]]; then
    PREPATH="$KAGGLE_INPUT_ROOT"
    POSTPATH=""
    echo "Using Kaggle dataset root: $PREPATH"
else
    PREPATH="$COLAB_KAGGLEHUB_ROOT"
    POSTPATH="/versions/$COLAB_DATASET_VERSION"
    echo "Using Colab dataset root: $PREPATH (version: $COLAB_DATASET_VERSION)"
fi

DATASET_BASE="$PREPATH/datasets/venim1103"

dataset_root() {
    local dataset_name="$1"
    echo "$DATASET_BASE/$dataset_name$POSTPATH"
}

link_dataset_files() {
    local source_dir="$1"
    local target_dir="$2"

    if [[ -d "$source_dir" ]]; then
        find "$source_dir" -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} "$target_dir"/ \; 2>/dev/null
    fi
}

echo "Preparing data directories..."

# ==========================================
# 2. CREATE DIRECTORIES & SYMLINKS
# ==========================================
rm -rf local_data

mkdir -p local_data/train/web/fineweb
mkdir -p local_data/train/logic/numinamath-cot
mkdir -p local_data/train/logic/fldx2
mkdir -p local_data/train/code/tiny-codes
mkdir -p local_data/train/tools/toolformer

mkdir -p local_data/sft/reasoning/open-math-reasoning
mkdir -p local_data/sft/reasoning/nemotron-post-training
mkdir -p local_data/sft/reasoning/openr1-math
mkdir -p local_data/sft/mixed/smol-smoltalk
mkdir -p local_data/sft/tool_calling/apigen-fc
mkdir -p local_data/sft/tool_calling/xlam-irrelevance
mkdir -p local_data/rl/reasoning

PRETRAIN_SMALLS_ROOT="$(dataset_root mini-mamba-1b58-pretrain-smalls)"
FINEWEB_ROOT="$(dataset_root mini-mamba-1b58-fineweb-edu-10bt)"
SFT_RL_ROOT="$(dataset_root mini-mamba-1b58-sft-rl-data)"

echo "Symlinking Pre-train Smalls..."
link_dataset_files "$PRETRAIN_SMALLS_ROOT/logic/numinamath" "local_data/train/logic/numinamath-cot"
link_dataset_files "$PRETRAIN_SMALLS_ROOT/logic/fldx2" "local_data/train/logic/fldx2"
link_dataset_files "$PRETRAIN_SMALLS_ROOT/code/tiny-codes" "local_data/train/code/tiny-codes"
link_dataset_files "$PRETRAIN_SMALLS_ROOT/tools/toolformer" "local_data/train/tools/toolformer"

echo "Symlinking FineWeb..."
link_dataset_files "$FINEWEB_ROOT" "local_data/train/web/fineweb"

echo "Symlinking SFT & RL Datasets..."
link_dataset_files "$SFT_RL_ROOT/sft/open-math" "local_data/sft/reasoning/open-math-reasoning"
link_dataset_files "$SFT_RL_ROOT/sft/nemotron" "local_data/sft/reasoning/nemotron-post-training"
link_dataset_files "$SFT_RL_ROOT/sft/openr1" "local_data/sft/reasoning/openr1-math"
link_dataset_files "$SFT_RL_ROOT/sft/smoltalk" "local_data/sft/mixed/smol-smoltalk"
link_dataset_files "$SFT_RL_ROOT/sft/apigen" "local_data/sft/tool_calling/apigen-fc"
link_dataset_files "$SFT_RL_ROOT/sft/xlam" "local_data/sft/tool_calling/xlam-irrelevance"

# Link RL reasoning to the same SFT math source
link_dataset_files "$SFT_RL_ROOT/sft/open-math" "local_data/rl/reasoning"

# ==========================================
# 3. SAFETY VALIDATION
# ==========================================
echo "Validating data structure and symlinks..."

if [ ! -d "local_data" ]; then
    echo "ERROR: local_data directory was not created."
    exit 1
fi

FILE_COUNT=$(find -L local_data -type f \( -name "*.parquet" -o -name "*.jsonl" \) 2>/dev/null | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No .parquet or .jsonl files found in local_data!"
    echo "This usually means datasets are not available in the selected environment ($MODE),"
    echo "or dataset folders/versions do not match expected layout."
    exit 1
else
    echo "Success! Found $FILE_COUNT valid data files perfectly linked."
    echo "Sample of linked files:"
    find -L local_data -type f \( -name "*.parquet" -o -name "*.jsonl" \) 2>/dev/null | head -n 3
fi
