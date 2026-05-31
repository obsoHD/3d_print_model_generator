#!/usr/bin/env bash
# 03_install_hunyuan3d.sh — Linux install (optional preview engine).

set -euo pipefail

tabletop_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hy_dir="$tabletop_root/hunyuan3d"
weights_dir="$tabletop_root/models/hunyuan3d"

if [ -d "$hy_dir" ]; then
    echo "Hunyuan3D already exists — pulling latest."
    (cd "$hy_dir" && git pull)
else
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$hy_dir"
fi

cd "$hy_dir"
[ -d venv ] || python3.11 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
[ -f requirements.txt ] && pip install -r requirements.txt

mkdir -p "$weights_dir"
echo ""
echo "Hunyuan3D code installed. Weights:"
echo "  https://huggingface.co/tencent/Hunyuan3D-2.1"
echo "  download into: $weights_dir"
echo ""
echo "Dual 5090 build: full pipeline (~29 GB) fits comfortably."
