#!/bin/bash
# setup_env.sh — Sets up a Python venv with PyTorch + deps on Meta devserver

set -e

# --- Config ---
ENV_DIR="$HOME/venvs/ml_env"
CUDA_VERSION="cu124"  # Safe for your H100 + driver 580.x
PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VERSION}"

# --- Proxy (required for external downloads on devserver) ---
export http_proxy=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080

# --- Create venv (uses system python3, bypasses the pip wrapper) ---
echo ">>> Creating virtual environment at ${ENV_DIR}"
python3 -m venv "${ENV_DIR}"

# --- Activate ---
source "${ENV_DIR}/bin/activate"

# --- Upgrade pip inside the venv (this pip is NOT the system wrapper) ---
echo ">>> Upgrading pip"
pip install --upgrade pip

# --- Install PyTorch with CUDA support ---
echo ">>> Installing PyTorch + torchvision (${CUDA_VERSION})"
pip install torch torchvision --index-url "${PYTORCH_INDEX}"

# --- Install remaining dependencies ---
echo ">>> Installing other packages"
pip install \
    timm>=1.0.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.13.0 \
    Pillow>=10.0.0 \
    tqdm>=4.66.0 \
    addict>=2.4.0 \
    pyyaml>=6.0

# --- Verify ---
echo ">>> Verifying install"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "✅ Done! To activate this env in future sessions:"
echo "   source ${ENV_DIR}/bin/activate"
