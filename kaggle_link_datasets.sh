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
# Create local directories
mkdir -p local_data/train/web
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

# Symlink Pre-Train Smalls
ln -s /kaggle/input/mini-mamba-1b58-pretrain-smalls/logic/numinamath/* local_data/train/logic/numinamath-cot/
ln -s /kaggle/input/mini-mamba-1b58-pretrain-smalls/logic/fldx2/* local_data/train/logic/fldx2/
ln -s /kaggle/input/mini-mamba-1b58-pretrain-smalls/code/tiny-codes/* local_data/train/code/tiny-codes/
ln -s /kaggle/input/mini-mamba-1b58-pretrain-smalls/tools/toolformer/* local_data/train/tools/toolformer/

# Symlink FineWeb-Edu 10BT
ln -s /kaggle/input/mini-mamba-1b58-fineweb-edu-10bt/web/fineweb/* local_data/train/web/

# Symlink SFT Data
ln -s /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/open-math/* local_data/sft/reasoning/open-math-reasoning/
ln -s /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/nemotron/* local_data/sft/reasoning/nemotron-post-training/
ln -s /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/openr1/* local_data/sft/reasoning/openr1-math/
ln -s /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/smoltalk/* local_data/sft/mixed/smol-smoltalk/
ln -s /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/apigen/* local_data/sft/tool_calling/apigen-fc/
ln -s /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/xlam/* local_data/sft/tool_calling/xlam-irrelevance/

# Symlink RL Data
ln -s /kaggle/input/mini-mamba-1b58-sft-rl-data/sft/open-math/* local_data/rl/reasoning/

echo "Symlinks created successfully! Data is ready."
