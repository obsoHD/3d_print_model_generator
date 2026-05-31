# Hardware notes

## RTX 5080 / 16 GB / Windows

Sweet spot for solo use. Sized for:

| Step | VRAM used | Time |
|---|---|---|
| FLUX-dev image gen (1024²) | ~12 GB | ~10–15 s |
| Sparc3D mesh (1024³) | ~8 GB | ~20–30 s |
| Hunyuan3D mesh only | ~10 GB | ~10 s |
| Hunyuan3D texture step | needs offload — won't fit native | ~60 s with offload |
| Blender cleanup | CPU only | ~5 s |

Hard limit: **Hunyuan3D full pipeline (~29 GB) won't fit**. Use
`--mesh-only` or accept the slower offload path.

Recommended flow on this build:
1. FLUX in ComfyUI for concept (12 GB).
2. After image is saved and FLUX is unloaded (ComfyUI does this with
   `--lowvram`), run Sparc3D (8 GB). You won't have both loaded simultaneously.
3. Hunyuan3D is optional. Skip unless you really want the painted preview.

## Dual RTX 5090 / 64 GB / Linux

Use the two GPUs for **parallel jobs**, not for splitting one job
(image-to-3D doesn't tensor-parallelize meaningfully).

Useful configurations:

```bash
# GPU 0 = image generation server (always loaded)
CUDA_VISIBLE_DEVICES=0  python comfyui/main.py --listen 127.0.0.1 --port 8188

# GPU 1 = 3D generation server (Sparc3D / Hunyuan3D)
CUDA_VISIBLE_DEVICES=1  python pipeline/sparc3d_server.py
```

This way Claude can hand off images to the 3D server without unloading
FLUX from GPU 0. Throughput roughly doubles vs the single-5080 setup.

Other dual-5090 wins:
- Run a Hunyuan3D **full pipeline** (29 GB) on a single 5090 — fits with
  room to spare.
- Run video models (Hunyuan Video, Wan 2.1) that need >32 GB — you'd want
  multi-GPU offload here.
- Train your own LoRAs in parallel with generation.

## When to choose which

- Just printing minis & terrain for the hobby → **5080 is fine**, save the
  money for a better resin printer.
- Want to do video gen, train LoRAs, run a multi-user setup, batch
  hundreds of assets a night → **dual 5090** pays off.

## SSD layout

```
A:/  (NVMe, fast)
  Project False Face/3D generation/tabletop/
    models/      ~50 GB   (frequently read, keep on fast drive)
    sparc3d/     ~6 GB    (codebase + venv)
    hunyuan3d/   ~3 GB    (codebase + venv)
    comfyui/     ~2 GB    (codebase + venv)

D:/  (slower bulk)  ← symlink targets here if A: is small
    tabletop_outputs/    grows fast
    tabletop_loras/      LoRA collection (can grow large)
```

Use symlinks (`mklink /D` on Windows, `ln -s` on Linux) to keep `models/`
and `outputs/` on whichever drive has space.
