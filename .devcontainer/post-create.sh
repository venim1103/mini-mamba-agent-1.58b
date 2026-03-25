#!/usr/bin/env bash
set -euo pipefail

PIP="/opt/conda/envs/ai/bin/python -m pip"
CONDA="/opt/conda/bin/conda"

have_python_pkgs() {
    /opt/conda/envs/ai/bin/python - "$@" <<'PY'
import importlib.util
import sys

need = sys.argv[1:]
missing = [n for n in need if importlib.util.find_spec(n) is None]
if missing:
    print("MISSING:", ", ".join(missing))
    raise SystemExit(1)
print("PRESENT:", ", ".join(need))
PY
}

# Update CA certificates if a real Zscaler cert is mounted (not an empty placeholder)
if [ -s /usr/local/share/ca-certificates/ZscalerRootCertificate-2048-SHA256.crt ]; then
    echo "==> Updating CA certificates (Zscaler)..."
    sudo update-ca-certificates
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

echo "==> Upgrading pip..."
$PIP install --upgrade pip

echo "==> Attempting torch install from PyPI index..."
if have_python_pkgs torch triton; then
    echo "==> Torch/triton already present. Skipping torch install."
else
    if $PIP install \
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
fi

echo "==> Attempting to install additional packages from PyPI index..."
if have_python_pkgs transformers datasets wandb; then
    echo "==> transformers/datasets/wandb already present. Skipping install."
else
    if $PIP install \
        transformers \
        datasets \
        wandb; then
        echo "==> Additional packages installed from PyPI index."
    else
        echo "==> Additional package install failed. Trying conda-forge fallback..."
        if $CONDA install -n ai -y -c conda-forge transformers datasets wandb; then
            echo "==> Additional packages installed from conda-forge."
        else
            echo "ERROR: Failed to install transformers/datasets/wandb via both pip and conda."
            exit 1
        fi
    fi
fi

echo "==> Installing mamba-ssm and causal-conv1d (no build isolation)..."
# Use --no-build-isolation so these build against the already-installed
# torch+CUDA version instead of pip pulling the latest (mismatched) torch
# into an isolated build environment.
if have_python_pkgs mamba_ssm causal_conv1d; then
    echo "==> mamba-ssm/causal-conv1d already present. Skipping install."
else
    $PIP install ninja packaging
    if $PIP install --no-build-isolation \
        "causal-conv1d>=1.4.0" \
        "mamba-ssm"; then
        echo "==> mamba-ssm installed successfully."
    else
        echo "==> mamba-ssm install failed."
    fi
fi

echo "==> Verifying required Python packages..."
/opt/conda/envs/ai/bin/python - <<'PY'
import importlib.util
import sys

required = ["torch", "transformers", "datasets", "wandb"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

if missing:
    print("ERROR: Missing required packages:", ", ".join(missing))
    sys.exit(1)

print("OK: required packages present:", ", ".join(required))
PY

echo "==> Post-create setup complete."
