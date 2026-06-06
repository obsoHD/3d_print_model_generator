# Workstation Inventory — `ai-pc` / `gen3d-app`

Snapshot of what's installed on the 3D‑print workstation, organized for a **pivot
to image + video** work. Compiled from the project's known stack — **run the
[verification commands](#9-dump-the-exact-installed-state) to confirm exact
versions and fill gaps**, since some pins drift over time.

TL;DR: the **GPU + CUDA/torch stack, the diffusion image models, the upscaler,
background removal, and ffmpeg are all reusable for image/video**. The 3D engines
and mesh tooling are the part you'd retire.

---

## 1. Hardware & base system
- **GPU:** NVIDIA **RTX 5090** (Blackwell, **sm_120**), 32 GB VRAM. *(Project notes
  say dual 5090; confirm count/model with `nvidia-smi`.)* Blackwell needs **CUDA
  12.8+** — there are no cu124 kernels for sm_120.
- **CPU:** AMD **Ryzen 9 9950X** (32 threads).
- **RAM:** ~**245 GB**.
- **OS:** Linux (host `obso@ai-pc`), Docker.
- **Container:** `gen3d-app`, image from `tabletop/deploy/Dockerfile.workstation`.
  Repo bind: host `/opt/gen3d` → container `/app`. Data bind: `/opt/gen3d/data` →
  `/data`. Container memory cap **96 GB** (raise via `docker update --memory`).
- **Repo:** `github.com/obsoHD/3d_print_model_generator`.

## 2. Core ML / CUDA stack  ✅ KEEP (foundation for image/video)
- **Python 3.12**
- **torch 2.8.0 + torchvision 0.23.0**, built against **cu128** (Blackwell).
- **CUDA 12.8 / 12.9** toolkit; MSVC build tools (Windows side); NVRTC fallback for sm_120.
- **xformers 0.0.32.post2** (forced CUTLASS attention on Blackwell), partial flash‑attn.
- **diffusers 0.37.1**, **transformers 4.57.3**, **accelerate 1.13.0**,
  **huggingface_hub 0.36.2**, safetensors, einops, omegaconf, timm, peft,
  **onnxruntime‑gpu 1.26.0**.
- numpy 2.x, scipy, scikit‑image, opencv‑python‑headless, pillow, **kornia**.

## 3. ✅ KEEP — image/video‑relevant models & tools (already here, reuse these)

### Image generation (diffusion)
- **FLUX.1‑schnell** (Black Forest Labs) — fast text→image.
- **FLUX.1‑Kontext‑dev** — instruction/edit + view‑fix image model.
- **Z‑Image‑Turbo** — current concept‑gen default (fast).
- **RealVisXL v5.0** — photoreal SDXL.
- **DreamShaper XL Turbo** — stylized SDXL turbo.
- **SDXL Refiner** — img2img refine pass.
- *(Earlier: Zero123++ / stable‑zero123 are novel‑view — 3D‑adjacent; see Retire.)*

### Image processing / utilities
- **Real‑ESRGAN** — super‑resolution upscaler (great for image+video frames).
- **rembg 2.0.75** + **u2net** ONNX — background removal.
- **RMBG‑2.0** (briaai, gated), **BiRefNet** — matting/bg removal.
- **DINOv3** (facebook, gated) — image features/embeddings.
- **imageio + imageio‑ffmpeg** — **decodes/encodes MP4/MOV/GIF** (your video I/O is
  already installed via ffmpeg).

> For video you'd mainly ADD a video model (e.g. an SVD/Wan/LTX‑style txt/img→video
> diffusion) — the torch/diffusers/ffmpeg/ESRGAN plumbing to run and post‑process it
> is already in place.

## 4. ♻️ RETIRE — 3D generation engines (single/multi‑image → mesh)
- **TRELLIS.2‑4B** (microsoft) — O‑Voxel; default printable engine.
- **Hunyuan3D 2.1** + **Hunyuan3D‑2mv** (Tencent) — single/multi‑view.
- **TripoSG** — SDF + diso DMC.
- **CraftsMan3D** — coarse + normal refine.
- **Hi3DGen** — normal‑bridged geometry (sparse stack: spconv‑cu126, cumm‑cu126, pccm).
- **Pixal3D** — installed‑but‑unused.
- **Stable3DGen**, **MV‑Adapter**, **Zero123++ / stable‑zero123** — multi‑view/aux.
- Custom CUDA exts for TRELLIS: **cumesh, o‑voxel (+Eigen), flexgemm, nvdiffrast,
  nvdiffrec**.

## 5. ♻️ RETIRE — mesh / CAD / print tooling
- **trimesh 4.10.1**, **pymeshlab** (incl. CGAL alpha‑wrap), **pymeshfix 0.18.1**,
  **pyvista**, **PyMCubes**, **diso**, plyfile, pygltflib, rtree, shapely, networkx.
- **build123d** + **cadquery‑ocp** — parametric CAD (terrain track).
- **Tweaker‑3** — auto‑orient.
- **Blender** — voxel‑remesh / cleanup (OpenVDB).
- **PrusaSlicer** + **Bambu Studio** CLI — slicer validation gate.
- pytorch‑lightning, jaxtyping, typeguard, wandb, tensorboardX (engine deps).

## 6. Application / infra  (mostly reusable shell)
- **Dashboard:** Node HTTP server `tabletop/dashboard/server.js`, binds
  `0.0.0.0:7842` (`DASHBOARD_PORT`), serves `index.html`. JSON control API — see
  `tabletop/dashboard/ADMIN_API.md` + `PROJECT_OVERVIEW.md`. **Reusable framework**
  for an image/video dashboard.
- **Pipeline orchestrator:** `tabletop/pipeline/orchestrate.py` (+ per‑engine
  wrappers, `concept_gen.py`, `mesh_gauntlet.py`, `decimate_mesh.py`).
- **SMB share** `prints` → `/opt/gen3d/data/outputs` (Windows access).
- **HF auth:** real token at `/data/hf/token` (+ `stored_tokens`); env `HF_TOKEN`
  sanitized in orchestrate. `data/` is gitignored.
- **Outputs:** `/app/tabletop/outputs/{concepts,meshes,stl,logs}`.

## 7. Model weights on disk
- ~**259 GB** of weights total (per `.gitignore`), under `models/` — NOT in git,
  re‑downloaded on deploy. The image models (FLUX, Z‑Image, SDXL family, ESRGAN,
  rembg/u2net, DINOv3, RMBG/BiRefNet) are the keepers; the 3D engine checkpoints
  (TRELLIS.2‑4B, Hunyuan3D, TripoSG, etc.) are the big reclaimable chunk.

## 8. Pivot notes (3D → image/video)
- **Keep:** §2 CUDA/torch stack, §3 image models + ESRGAN + rembg + ffmpeg, §6
  dashboard/orchestrator framework, §6 SMB/HF infra.
- **Retire / archive:** §4 3D engines, §5 mesh/CAD/print tooling, and their
  checkpoints in `models/` to reclaim ~100s of GB.
- **Likely to add for video:** a video‑diffusion model + its scheduler; frame
  interpolation (e.g. RIFE/FILM); maybe a VAE‑tiled decoder for long clips. The
  GPU/torch/ffmpeg base is ready.
- The dashboard's **`/api/run` + capabilities + status** contract carries over —
  swap the engine list and params and the same control plane drives image/video jobs.

---

## 9. Dump the exact installed state
Run these on the workstation and paste/append the output for a precise, versioned
inventory (this doc is the known‑stack map; these give ground truth):

```bash
# GPUs + driver/CUDA
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# container base
docker inspect gen3d-app --format 'image={{.Config.Image}} mem={{.HostConfig.Memory}}'
docker exec gen3d-app python -c "import torch,sys;print(sys.version);print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

# full python package list (the authoritative inventory)
docker exec gen3d-app pip freeze | sort > pip_freeze.txt; wc -l pip_freeze.txt

# system packages of note (ffmpeg, blender, slicers)
docker exec gen3d-app sh -c 'ffmpeg -version | head -1; which blender prusa-slicer bambu-studio 2>/dev/null'

# model weights by size (what to keep vs reclaim)
docker exec gen3d-app sh -c 'du -h -d2 /data 2>/dev/null | sort -hr | head -40'
du -h -d2 /opt/gen3d/data 2>/dev/null | sort -hr | head -40

# what HF models are cached
docker exec gen3d-app sh -c 'ls -1 /data/hf/hub 2>/dev/null; ls -1 /data/models 2>/dev/null'
```

Append `pip_freeze.txt` + the `du` output to this file for a complete, exact record.
