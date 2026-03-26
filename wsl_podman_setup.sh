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

# Fix #12: Shebang must be on line 1
# Fix #16: Add error handling and privilege checks
set -euo pipefail

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

echo "==> Installing podman..."
apt install -y podman

# podman pull docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

echo "==> Setting up NVIDIA container toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt update
apt install -y nvidia-container-toolkit
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
# Should show something like: nvidia.com/gpu=0 or nvidia.com/gpu=all

echo "==> Configuring rootless podman for NVIDIA..."
mkdir -p ~/.config/cdi
cp /etc/cdi/nvidia.yaml ~/.config/cdi/nvidia.yaml

echo "==> Setup complete!"