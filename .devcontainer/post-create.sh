#!/usr/bin/env bash
set -euo pipefail

PIP="/opt/conda/envs/ai/bin/python -m pip"

echo "==> Upgrading pip..."
$PIP install --no-cache-dir --upgrade pip

echo "==> Attempting torch install from PyPI index..."
if $PIP install --no-cache-dir \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    triton==3.6.0 \
    --index-url https://download.pytorch.org/whl/cu128; then
    echo "==> Torch installed from PyPI index."
else
    echo "==> PyPI install failed. Trying local wheels from /workspaces/AI/TORCH..."
    LOCAL_DIR="/workspaces/AI/TORCH"
    if [ -d "${LOCAL_DIR}" ]; then $PIP install ${LOCAL_DIR}/*.whl && echo "==> Torch installed from local wheels."
    else
        echo "WARNING: No .whl files found in ${LOCAL_DIR}. Torch is NOT installed."
    fi
fi

echo "==> Post-create setup complete."
