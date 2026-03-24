#!/usr/bin/env bash
set -euo pipefail

PIP="/opt/conda/envs/ai/bin/python -m pip"

# Update CA certificates if a real Zscaler cert is mounted (not an empty placeholder)
if [ -s /usr/local/share/ca-certificates/ZscalerRootCertificate-2048-SHA256.crt ]; then
    echo "==> Updating CA certificates (Zscaler)..."
    sudo update-ca-certificates
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

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
    echo "==> PyPI install failed. Trying local wheels from TORCH..."
    LOCAL_DIR="TORCH"
    if [ -d "${LOCAL_DIR}" ]; then
        $PIP install ${LOCAL_DIR}/*.whl && echo "==> Torch installed from local wheels."
    else
        echo "WARNING: No .whl files found in ${LOCAL_DIR}. Torch is NOT installed."
    fi
fi
echo "==> Attempting to install additional packages from PyPI index..."
if $PIP install --no-cache-dir \
    transformers \
    datasets \
    wandb; then
    echo "==> Additional packages installed from PyPI index."
else
    echo "==> Additional package install failed."
fi

echo "==> Installing mamba-ssm and causal-conv1d (no build isolation)..."
# Use --no-build-isolation so these build against the already-installed
# torch+CUDA version instead of pip pulling the latest (mismatched) torch
# into an isolated build environment.
$PIP install --no-cache-dir ninja packaging
if $PIP install --no-cache-dir --no-build-isolation \
    "causal-conv1d>=1.4.0" \
    "mamba-ssm"; then
    echo "==> mamba-ssm installed successfully."
else
    echo "==> mamba-ssm install failed."
fi

echo "==> Post-create setup complete."
