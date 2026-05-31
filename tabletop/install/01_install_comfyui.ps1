# 01_install_comfyui.ps1
# Installs ComfyUI Portable (Windows) under tabletop/comfyui/
# Run from the tabletop/ directory:  .\install\01_install_comfyui.ps1
#
# After this you'll start ComfyUI with: .\comfyui\run_nvidia_gpu.bat

$ErrorActionPreference = "Stop"

# Resolve paths
$tabletop_root = Split-Path -Parent $PSScriptRoot
$comfy_dir     = Join-Path $tabletop_root "comfyui"
$models_dir    = Join-Path $tabletop_root "models"

if (Test-Path $comfy_dir) {
    Write-Host "ComfyUI already exists at $comfy_dir — skipping clone." -ForegroundColor Yellow
} else {
    Write-Host "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git $comfy_dir
}

Push-Location $comfy_dir
try {
    # Create venv if missing
    if (-not (Test-Path "venv")) {
        py -3.11 -m venv venv
    }
    & .\venv\Scripts\Activate.ps1

    Write-Host "Installing PyTorch with CUDA 12.9..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

    Write-Host "Installing ComfyUI requirements..."
    pip install -r requirements.txt

    Write-Host "Installing custom nodes for 3D + multi-GPU..."
    if (-not (Test-Path "custom_nodes\ComfyUI-Manager")) {
        git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes\ComfyUI-Manager
    }
    if (-not (Test-Path "custom_nodes\ComfyUI-3D-Pack")) {
        git clone https://github.com/MrForExample/ComfyUI-3D-Pack custom_nodes\ComfyUI-3D-Pack
        pip install -r custom_nodes\ComfyUI-3D-Pack\requirements.txt
    }

    # Point ComfyUI's models dir at our shared models/ folder
    $extra_paths = Join-Path $comfy_dir "extra_model_paths.yaml"
    @"
tabletop_shared:
    base_path: $($models_dir.Replace('\','/'))
    checkpoints: flux/|sdxl/
    loras: loras/
    vae: flux/|sdxl/
"@ | Set-Content -Path $extra_paths -Encoding utf8

    Write-Host ""
    Write-Host "ComfyUI installed. Next steps:" -ForegroundColor Green
    Write-Host "  1. Run install\02_install_sparc3d.ps1 (mesh generation)"
    Write-Host "  2. Run install\03_install_hunyuan3d.ps1 (preview, optional)"
    Write-Host "  3. Download FLUX-dev to tabletop\models\flux\  (see install\05_install_loras.md)"
    Write-Host "  4. Start ComfyUI:  .\comfyui\venv\Scripts\python.exe .\comfyui\main.py --listen 127.0.0.1"
}
finally {
    Pop-Location
}
