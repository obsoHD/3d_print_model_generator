#!/usr/bin/env bash
# 01_install_comfyui.sh — Linux install of ComfyUI under tabletop/comfyui/
#
# Run from tabletop/:  bash install/01_install_comfyui.sh

set -euo pipefail

tabletop_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
comfy_dir="$tabletop_root/comfyui"
models_dir="$tabletop_root/models"

if [ -d "$comfy_dir" ]; then
    echo "ComfyUI already exists at $comfy_dir — skipping clone."
else
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$comfy_dir"
fi

cd "$comfy_dir"

if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "Installing PyTorch with CUDA 12.9..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

echo "Installing ComfyUI requirements..."
pip install -r requirements.txt

echo "Installing custom nodes (Manager + 3D-Pack)..."
[ -d custom_nodes/ComfyUI-Manager ] || git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager
[ -d custom_nodes/ComfyUI-3D-Pack ] || {
    git clone https://github.com/MrForExample/ComfyUI-3D-Pack custom_nodes/ComfyUI-3D-Pack
    pip install -r custom_nodes/ComfyUI-3D-Pack/requirements.txt
}

# Multi-GPU support for the dual 5090 build
[ -d custom_nodes/ComfyUI-MultiGPU ] || git clone https://github.com/pollockjj/ComfyUI-MultiGPU custom_nodes/ComfyUI-MultiGPU

cat > "$comfy_dir/extra_model_paths.yaml" <<EOF
tabletop_shared:
    base_path: $models_dir
    checkpoints: flux/|sdxl/
    loras: loras/
    vae: flux/|sdxl/
EOF

echo ""
echo "ComfyUI installed. Next steps:"
echo "  1. bash install/02_install_sparc3d.sh  (mesh generation)"
echo "  2. bash install/03_install_hunyuan3d.sh  (optional, preview)"
echo "  3. Download FLUX-dev to tabletop/models/flux/  (see install/05_install_loras.md)"
echo "  4. Start ComfyUI:  $comfy_dir/venv/bin/python $comfy_dir/main.py --listen 127.0.0.1"
