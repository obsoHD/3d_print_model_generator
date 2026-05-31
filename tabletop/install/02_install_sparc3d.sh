#!/usr/bin/env bash
# 02_install_sparc3d.sh — Linux install of Sparc3D under tabletop/sparc3d/

set -euo pipefail

tabletop_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sparc_dir="$tabletop_root/sparc3d"
weights_dir="$tabletop_root/models/sparc3d"

if [ -d "$sparc_dir" ]; then
    echo "Sparc3D already exists — pulling latest."
    (cd "$sparc_dir" && git pull)
else
    echo "Cloning Sparc3D..."
    git clone https://github.com/lizhihao6/Sparc3D.git "$sparc_dir"
fi

cd "$sparc_dir"

if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "Installing PyTorch with CUDA 12.9..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "  (no requirements.txt — check the repo for environment.yml)"
fi

mkdir -p "$weights_dir"

echo ""
echo "Sparc3D code installed. Weights need to be downloaded MANUALLY:"
echo "  Visit: https://huggingface.co/ilcve21/Sparc3D"
echo "  Download into:  $weights_dir"
echo ""
echo "Or via HF CLI:"
echo "  pip install huggingface_hub"
echo "  huggingface-cli download ilcve21/Sparc3D --local-dir $weights_dir"
