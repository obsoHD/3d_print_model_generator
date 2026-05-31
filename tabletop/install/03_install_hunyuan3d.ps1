# 03_install_hunyuan3d.ps1
# OPTIONAL — Hunyuan3D 2.1 for previewing painted/textured versions.
# Not required for printing. Run AFTER Sparc3D is working.

$ErrorActionPreference = "Stop"

$tabletop_root = Split-Path -Parent $PSScriptRoot
$hy_dir        = Join-Path $tabletop_root "hunyuan3d"
$weights_dir   = Join-Path $tabletop_root "models\hunyuan3d"

if (Test-Path $hy_dir) {
    Write-Host "Hunyuan3D already exists at $hy_dir — pulling latest." -ForegroundColor Yellow
    Push-Location $hy_dir
    git pull
    Pop-Location
} else {
    Write-Host "Cloning Hunyuan3D-2.1..."
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git $hy_dir
}

Push-Location $hy_dir
try {
    if (-not (Test-Path "venv")) {
        py -3.11 -m venv venv
    }
    & .\venv\Scripts\Activate.ps1

    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    }

    if (-not (Test-Path $weights_dir)) {
        New-Item -ItemType Directory -Path $weights_dir | Out-Null
    }
    Write-Host ""
    Write-Host "Hunyuan3D code installed. Weights:" -ForegroundColor Green
    Write-Host "  Shape model:    https://huggingface.co/tencent/Hunyuan3D-2.1"
    Write-Host "  Texture model:  same repo"
    Write-Host "  Download into:  $weights_dir"
    Write-Host ""
    Write-Host "WARNING for RTX 5080 / 16 GB users:"
    Write-Host "  Full pipeline (shape + texture) needs ~29 GB VRAM."
    Write-Host "  On 16 GB, use mesh-only mode OR enable model offloading."
}
finally {
    Pop-Location
}
