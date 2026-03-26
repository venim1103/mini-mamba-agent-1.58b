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

echo "==> Installing torch/triton (special handling for CUDA builds)..."
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

echo "==> Installing mamba-ssm and causal-conv1d (special handling required)..."
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

echo "==> Installing remaining packages from requirements.txt..."
# Get requirements.txt path - handle both development and deployment contexts
if [ -f /workspace/requirements.txt ]; then
    REQ_FILE="/workspace/requirements.txt"
elif [ -f "$(dirname "$0")/../requirements.txt" ]; then
    REQ_FILE="$(dirname "$0")/../requirements.txt"
elif [ -f "$(dirname "$0")/../../requirements.txt" ]; then
    REQ_FILE="$(dirname "$0")/../../requirements.txt"
else
    echo "WARNING: requirements.txt not found, skipping."
    REQ_FILE=""
fi

if [ -n "$REQ_FILE" ] && [ -f "$REQ_FILE" ]; then
    echo "==> Using requirements.txt: $REQ_FILE"
    # Filter out torch, triton, mamba-ssm, causal-conv1d (already handled above)
    # and create a temporary requirements file for the rest
    TEMP_REQ=$(mktemp)
    grep -vE "^(torch|triton|mamba-ssm|causal-conv1d)" "$REQ_FILE" > "$TEMP_REQ"
    
    if $PIP install -r "$TEMP_REQ"; then
        echo "==> Requirements installed successfully."
    else
        echo "==> Some requirements failed, trying conda-forge fallback..."
        # Try conda-forge for common packages that might fail pip
        $CONDA install -n ai -y -c conda-forge transformers datasets wandb einops accelerate 2>/dev/null || true
    fi
    rm -f "$TEMP_REQ"
else
    echo "==> requirements.txt not found, falling back to manual package install..."
    if $PIP install transformers datasets wandb bitsandbytes einops accelerate; then
        echo "==> Packages installed from PyPI."
    else
        echo "==> Pip install failed, trying conda-forge..."
        $CONDA install -n ai -y -c conda-forge transformers datasets wandb einops accelerate || true
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