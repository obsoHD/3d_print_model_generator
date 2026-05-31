# 02_install_sparc3d.ps1
# Install Sparc3D (the PRIMARY printable-mesh generator) on Windows
# Run from tabletop/:  .\install\02_install_sparc3d.ps1

$ErrorActionPreference = "Stop"

$tabletop_root = Split-Path -Parent $PSScriptRoot
$sparc_dir     = Join-Path $tabletop_root "sparc3d"
$weights_dir   = Join-Path $tabletop_root "models\sparc3d"

if (Test-Path $sparc_dir) {
    Write-Host "Sparc3D already exists at $sparc_dir — pulling latest." -ForegroundColor Yellow
    Push-Location $sparc_dir
    git pull
    Pop-Location
} else {
    Write-Host "Cloning Sparc3D..."
    git clone https://github.com/lizhihao6/Sparc3D.git $sparc_dir
}

Push-Location $sparc_dir
try {
    if (-not (Test-Path "venv")) {
        py -3.11 -m venv venv
    }
    & .\venv\Scripts\Activate.ps1

    Write-Host "Installing PyTorch with CUDA 12.9..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

    Write-Host "Installing Sparc3D requirements..."
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    } else {
        Write-Host "  (no requirements.txt found — Sparc3D may use environment.yml; check the repo)" -ForegroundColor Yellow
    }

    # Download pretrained weights
    if (-not (Test-Path $weights_dir)) {
        New-Item -ItemType Directory -Path $weights_dir | Out-Null
    }
    Write-Host ""
    Write-Host "Sparc3D code installed. Weights need to be downloaded MANUALLY:" -ForegroundColor Green
    Write-Host "  Visit: https://huggingface.co/ilcve21/Sparc3D" -ForegroundColor Cyan
    Write-Host "  Download into:  $weights_dir"
    Write-Host ""
    Write-Host "Or use the HF CLI:"
    Write-Host "  pip install huggingface_hub"
    Write-Host "  huggingface-cli download ilcve21/Sparc3D --local-dir $weights_dir"
}
finally {
    Pop-Location
}
