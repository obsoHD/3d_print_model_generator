# Tabletop — local AI-driven 3D generation pipeline for printable terrain & minis

A self-contained pipeline that turns natural-language descriptions into
3D-printable STL files, driven by Claude and running entirely on local GPU.

## What it does

```
"A goblin shaman in robes, 32mm scale, holding a gnarled staff"
        │
        ▼
   Claude (concept_director)
        │  ── crafts a FLUX prompt for clean 3D-input concept art
        ▼
   ComfyUI + FLUX + D&D LoRA
        │  ── produces an isolated character image on plain background
        ▼
   Sparc3D  (PRIMARY printable engine — watertight, manifold meshes at 1024³)
        │  ── one-shot image → printable mesh
        ▼
   [optional] Hunyuan3D 2.1 preview
        │  ── textured render so you can SEE the asset painted
        ▼
   Blender headless cleanup
        │  ── manifold check, hollow for resin, support hints
        ▼
   .stl  →  Slicer  →  Print
```

All steps run locally. Claude orchestrates via an MCP server.

## Pick-your-engine cheat sheet

| Goal | Tool |
|---|---|
| Printable mini / terrain (default) | **Sparc3D** — explicitly watertight, manifold, print-ready |
| Painted preview of what an asset will look like | Hunyuan3D 2.1 |
| Multi-view fusion for hero pieces (4 angles → one mesh) | Hitem3D web service or DIY with FLUX + ControlNet → Sparc3D |
| Topology comparison / second opinion | Trellis 2 |

Sparc3D is the default for any printable output. Hunyuan3D is a previewer.

## Hardware fit

- **RTX 5080 / 16 GB**: comfortable for Sparc3D (~8 GB at 1024³). Hunyuan3D
  mesh-only fits; full texture needs offloading.
- **Dual RTX 5090 / 64 GB**: run two pipelines in parallel (e.g. mini on GPU 0,
  terrain on GPU 1). Tensor-parallel FLUX via xDiT for ~2× image gen speed.

See `docs/HARDWARE.md` for benchmark notes.

## Directory map

```
tabletop/
├── README.md             ← this file
├── ARCHITECTURE.md       ← pipeline diagram + design decisions
├── install/              ← install scripts (Win + Linux) — RUN THESE FIRST
├── pipeline/             ← Python orchestrator + per-tool clients
├── workflows/            ← ComfyUI workflow JSONs (FLUX prompts, etc.)
├── prompts/              ← Claude sub-agent prompt templates
├── mcp/                  ← MCP server exposing the pipeline to Claude
├── models/               ← model weights live here (you download via install/)
│   ├── flux/             ── FLUX.1-dev weights (~24 GB)
│   ├── sdxl/             ── SDXL + LoRAs
│   ├── sparc3d/          ── Sparc3D weights
│   ├── hunyuan3d/        ── Hunyuan3D weights
│   └── loras/            ── shared LoRA storage
├── outputs/
│   ├── concepts/         ── FLUX-generated concept PNGs
│   ├── meshes/           ── GLB / OBJ from Sparc3D / Hunyuan3D
│   ├── stl/              ── final printable STLs
│   └── previews/         ── Hunyuan3D-painted preview renders
├── examples/             ── worked end-to-end examples
└── docs/                 ── deeper docs (hardware, troubleshooting, etc.)
```

## Quick start

1. **Prereqs** — install Python 3.11, CUDA 12.9 driver, git LFS, Blender
   (see `install/00_prereqs.md`).
2. **ComfyUI** — `install/01_install_comfyui.{ps1,sh}`.
3. **Sparc3D** — `install/02_install_sparc3d.{ps1,sh}`. (~8 GB weights)
4. **(optional) Hunyuan3D** — `install/03_install_hunyuan3d.{ps1,sh}`. (~13 GB)
5. **MCP bridge** — `install/05_install_mcp.{ps1,sh}` then point Claude at it.
6. **Sanity check** — `python pipeline/orchestrate.py --prompt "stone watchtower"
   --kind terrain --out outputs/stl/watchtower.stl`.

## Where Claude fits

Claude is the **creative director + critic + glue**:
- Writes the FLUX prompt from your natural-language description
- Reviews the concept image and asks you to confirm before spending 30s on 3D
- Picks Sparc3D vs Hunyuan3D based on whether you're printing or previewing
- Runs Blender cleanup, flags non-manifold issues, suggests fixes
- Suggests orientation + support strategy for the printer

The MCP bridge in `mcp/server.py` exposes each stage as a callable tool.

## Status

- [x] Directory scaffold
- [x] Install scripts (Windows + Linux)
- [x] Pipeline orchestrator (Python)
- [x] MCP server stub
- [x] Blender cleanup script
- [x] One example workflow JSON for ComfyUI
- [ ] Model weights — YOU run the install scripts to download
- [ ] First end-to-end test print
