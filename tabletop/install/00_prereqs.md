# Prerequisites

Install these before running the per-tool install scripts.

## Both platforms

| Tool | Min version | Notes |
|---|---|---|
| Python | 3.11.x | Avoid 3.12+ — some deps lag. Use a venv. |
| Git | latest | with **git-lfs** enabled (`git lfs install`) |
| 7-Zip / `unzip` | any | install scripts extract archives |
| Blender | 4.2 LTS | headless mode used; install full app, we just call the binary |
| ~150 GB free SSD | — | weights + outputs grow fast |

## NVIDIA GPU

- Driver ≥ 576.57 (for CUDA 12.9 / Blackwell support on 5080/5090)
- CUDA Toolkit not required for runtime (PyTorch ships its own), but
  install the toolkit if you plan to build any wheels from source.

## Windows (RTX 5080 build)

- PowerShell 7+ recommended (not strictly required)
- Long-path support enabled (Windows installer option or registry tweak —
  some pip wheels fail silently at >260 char paths)
- Visual Studio Build Tools 2022 (only if building wheels from source —
  most installs use pre-built CUDA 12.9 wheels)

## Linux (dual RTX 5090 build)

- Ubuntu 24.04 LTS or similar
- `build-essential`, `python3.11-venv`, `python3.11-dev`
- `nvidia-driver-575` or newer
- Optional: `nccl` for multi-GPU communication (not needed for solo 3D
  inference, only for tensor-parallel image gen)

## Disk layout assumption

This pipeline assumes you have at minimum:

```
A:/Project False Face/3D generation/tabletop/           ← this repo
A:/Project False Face/3D generation/tabletop/models/    ← weights (~50 GB)
A:/Project False Face/3D generation/tabletop/outputs/   ← growing
```

If your SSD is small, symlink `models/` and `outputs/` to a larger drive.

## Sanity check

```bash
python --version          # 3.11.x
git --version
git lfs version
blender --version         # 4.2 or newer
nvidia-smi                # shows your GPU, CUDA 12.9 ready
```

If any of these fail, fix before running the install scripts.
