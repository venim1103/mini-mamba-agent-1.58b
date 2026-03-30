#!/bin/sh
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

# ==========================================
# 1. KAGGLE ENVIRONMENT CHECK
# ==========================================
# Check for Kaggle's specific environment variables or root folders
if [[ -z "${KAGGLE_KERNEL_RUN_TYPE}" ]] && [[ ! -d "/kaggle/input" ]]; then
    echo "Not running in a Kaggle environment. Skipping Kaggle dataset linking."
    exit 0
fi

echo "Kaggle environment detected. Preparing data directories..."

# ==========================================
# 2. CREATE DIRECTORIES & SYMLINKS
# ==========================================
# Clean slate just in case
rm -rf local_data

# Make the EXACT folders your data.py DATA_WEIGHTS dictionary expects
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

# Hunt down every .parquet/.jsonl file and link them straight into the flat target folders!
echo "Symlinking Pre-train Smalls..."
find /kaggle/input/mini-mamba-1b58-pretrain-smalls/logic/numinamath -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/train/logic/numinamath-cot/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-pretrain-smalls/logic/fldx2 -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/train/logic/fldx2/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-pretrain-smalls/code/tiny-codes -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/train/code/tiny-codes/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-pretrain-smalls/tools/toolformer -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/train/tools/toolformer/ \; 2>/dev/null

echo "Symlinking FineWeb..."
find /kaggle/input/mini-mamba-1b58-fineweb-edu-10bt/web/fineweb -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/train/web/fineweb/ \; 2>/dev/null

echo "Symlinking SFT & RL Datasets..."
find /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/open-math -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/sft/reasoning/open-math-reasoning/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/nemotron -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/sft/reasoning/nemotron-post-training/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/openr1 -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/sft/reasoning/openr1-math/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/smoltalk -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/sft/mixed/smol-smoltalk/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/apigen -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/sft/tool_calling/apigen-fc/ \; 2>/dev/null
find /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/xlam -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/sft/tool_calling/xlam-irrelevance/ \; 2>/dev/null

# Link RL reasoning to the same SFT math source
find /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/open-math -type f \( -name "*.parquet" -o -name "*.jsonl" \) -exec ln -s {} local_data/rl/reasoning/ \; 2>/dev/null

# ==========================================
# 3. SAFETY VALIDATION
# ==========================================
echo "Validating data structure and symlinks..."

if [[ ! -d "local_data" ]]; then
    echo "ERROR: local_data directory was not created."
    exit 1
fi

# The -L flag forces 'find' to follow symlinks. If a symlink is broken, it won't count it.
FILE_COUNT=$(find -L local_data -type f \( -name "*.parquet" -o -name "*.jsonl" \) 2>/dev/null | wc -l)

if [[ "$FILE_COUNT" -eq 0 ]]; then
    echo "   ERROR: No .parquet or .jsonl files found in local_data!"
    echo "   This usually means the Kaggle Datasets aren't attached to this notebook,"
    echo "   or the dataset folder names in /kaggle/input/ don't match the script."
    exit 1
else
    echo "Success! Found $FILE_COUNT valid data files perfectly linked."
    echo "Sample of linked files:"
    find -L local_data -type f \( -name "*.parquet" -o -name "*.jsonl" \) 2>/dev/null | head -n 3
fi
