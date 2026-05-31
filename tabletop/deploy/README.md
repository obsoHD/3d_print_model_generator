# Deploy — dual RTX 5090 Linux workstation

This folder holds everything to run the generator as a persistent Docker service
on the workstation. The repo carries **code only** — models (~259 GB) and the
heavy vendored repos are pulled on the box.

## A. Push from the dev PC (Windows) — already-set-up GitHub auth

```powershell
cd "A:\Project False Face\3D generation"
git init -b main
git add .
git commit -m "Deploy scaffolding + pipeline (single/mv Hunyuan, TripoSG, parametric)"
git remote add origin https://github.com/obsoHD/3d_print_model_generator.git
git push -u origin main
```
`.gitignore` keeps models, venvs, outputs, and the big vendored repos out, so the
push is a few MB.

## B. On the workstation (SSH) — discovery first (ground truth)

```bash
nvidia-smi                                   # confirm driver >=570 (Blackwell needs CUDA 12.8+)
docker ps --format '{{.Names}}: {{.Ports}}'  # confirm 7000/7443 are the only taken ports
docker network ls && docker volume ls
df -h /opt                                    # confirm NVMe space for /opt/gen3d/data
free -h ; nproc
cat /etc/os-release
```

## C. Clone + prepare data dir

```bash
sudo mkdir -p /opt/gen3d && sudo chown "$USER" /opt/gen3d
git clone https://github.com/obsoHD/3d_print_model_generator.git /opt/gen3d
sudo mkdir -p /opt/gen3d/data/{hf,models,outputs}    # NVMe — never the NFS mount
cd /opt/gen3d/tabletop/deploy
cp .env.example .env && nano .env                    # set HF_TOKEN; pick GPUs
```

## D. Build + run

```bash
cd /opt/gen3d/tabletop/deploy
docker compose -f docker-compose.workstation.yml up -d --build
docker compose -f docker-compose.workstation.yml ps
docker compose -f docker-compose.workstation.yml logs -f gen3d
# dashboard: http://<workstation-ip>:7800
```

## E. First-run model provisioning

Models land in `/opt/gen3d/data/models` (bind-mounted to `/data/models`) and the
HF cache in `/opt/gen3d/data/hf`. Either:
- let the pipeline download on first generate (HF_TOKEN must be set), or
- pre-seed by `rsync`-ing the proven model tree from the dev PC into
  `/opt/gen3d/data/models`.

## Notes / known box-iteration points
- **CUDA**: base is `cuda:12.8.0-devel`. Blackwell (sm_120) has **no** kernels in
  cu124 — do not downgrade torch to cu124. Confirm `nvidia-smi` shows CUDA >=12.8.
- **TripoSG / diso** DMC compiles from source on first import; the devel base +
  `TORCH_CUDA_ARCH_LIST=12.0` cover it.
- **Pixal3D** is omitted (Windows-only wheels, installed-but-unused). Active
  engines: Hunyuan single (hy3dshape), Hunyuan mv (hy3dgen), TripoSG, parametric.
- **Vendored-repo patches** (segment_reduce, BiRefNet) from the Windows tree are
  NOT in this image — re-apply on the box if a regression appears.
- **Higher detail** the 32 GB GPUs unlock: bump `octree_resolution` to **512** and
  prefer the multi-view path (front/left/back) over single-view inference.
- **Pin a GPU**: set `GEN3D_GPUS=1` + `CUDA_VISIBLE_DEVICES=1` in `.env` to leave a
  5090 free; or `count: all` to use both.
