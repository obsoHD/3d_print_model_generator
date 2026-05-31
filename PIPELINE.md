# Tabletop 3D Printing Pipeline — Documentation

Local AI pipeline: text prompt → printable STL, no cloud, no subscription.
Hardware baseline: RTX 5080 / 16 GB VRAM / Windows 11.

---

## Architecture

```
Prompt
  │
  ▼
[ComfyUI + SDXL 1.0]        concept image (PNG, white BG)
  │
  ▼
[Hunyuan3D 2.1]             watertight GLB mesh
  │
  ▼
[Blender headless]          scale + repair + STL export
  │
  ▼
[slicer_prep.py]            orientation advisory (trimesh)
  │
  ▼
STL + print manifest
```

Entry point: `tabletop/pipeline/orchestrate.py`

---

## Components

### 1. ComfyUI / SDXL (concept image)

| Item | Detail |
|------|--------|
| Version | ComfyUI 0.22.0 |
| Model | SDXL 1.0 (`sd_xl_base_1.0.safetensors`) |
| Weights | `tabletop/models/sdxl/sd_xl_base_1.0.safetensors` |
| Server | `http://127.0.0.1:8188` |
| Start | `tabletop/start_comfyui.bat` (or `Start-Process cmd.exe ...`) |
| Workflows | `tabletop/workflows/sdxl_terrain_concept.json`, `sdxl_mini_concept.json` |
| Client | `tabletop/pipeline/comfy_client.py` |
| VRAM | ~6.5 GB |

**Known gotchas:**
- `ckpt_name` in workflow JSON must use Windows backslash: `"sdxl\\sd_xl_base_1.0.safetensors"` — ComfyUI's checkpoint list uses backslashes on Windows, exact match required.
- Strip non-node top-level keys (`_comment`, `_meta`, etc.) before POSTing — ComfyUI tries to parse every key as a node and 500s on strings.
- `extra_model_paths.yaml`: set `checkpoints: ./` so ComfyUI can resolve relative paths.
- If ComfyUI won't start: `netstat -ano | Select-String ":8188"` to find and kill a stale Python child process.

---

### 2. Hunyuan3D 2.1 (mesh generation)

| Item | Detail |
|------|--------|
| Repo | `tencent/Hunyuan3D-2.1` |
| Weights | `tabletop/models/hunyuan3d/hunyuan3d-dit-v2-1/model.fp16.ckpt` (6.9 GB, `.ckpt` not safetensors) |
| Venv | `tabletop/hunyuan3d/venv/` |
| Entry script | `tabletop/hunyuan3d/hy3d_infer.py` |
| Client | `tabletop/pipeline/hunyuan_client.py` |
| Timeout | 7200 s (model load + 50 inference steps takes 15–60 min) |
| VRAM | ~7–8 GB for mesh-only |

**Weight path resolution:**  
`hy3dgen.smart_load_model` looks in `$HY3DGEN_MODELS/<repo_id>/<subfolder>/`. We set `HY3DGEN_MODELS` to `tabletop/models/` and create a Windows junction so the path resolves:

```
tabletop/models/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/
    → junction →
tabletop/models/hunyuan3d/hunyuan3d-dit-v2-1/
```

**Known gotchas:**
- Pass `use_safetensors=False` — weights are `.ckpt`, not `.safetensors`.
- `enable_model_cpu_offload()` is not implemented in this pipeline class. It calls `self.to("cpu")` before crashing with `AttributeError: no attribute 'components'`, leaving the model on CPU (326 s/step instead of ~9 s/step). Fix: wrap in `try/except` and call `pipeline.to("cuda")` in the except block.
- Background removal: use numpy threshold (>225) + 3×3 alpha erosion. Do NOT use `rembg`/`numba` — Windows Application Control Policy blocks `_nrt_python.pyd`.
- Step 1 of diffusion is slow (~326 s) due to CUDA kernel JIT compilation on RTX 5080 (Blackwell, SM_120). Subsequent steps are ~9 s/step. Total 50-step run: ~20–50 min depending on VRAM pressure.
- With ComfyUI also in VRAM (~6.5 GB), total VRAM is ~14–15 GB, leaving ~1–2 GB headroom. This slows inference vs. a cold start.

**Background removal matters:**  
If near-white pixels survive in the input image, HY3D generates planar fin geometry around the object. Two-stage fix:
1. Lower threshold to >225 (was >240)
2. 3×3 morphological erosion on the alpha channel to eat edge halos

---

### 3. Blender cleanup (mesh repair + STL)

| Item | Detail |
|------|--------|
| Version | Blender 4.2 LTS |
| Location | `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe` (auto-discovered) |
| Script | `tabletop/pipeline/blender_cleanup.py` |
| Mode | `--background` (headless) |

**What it does (in order):**
1. Import GLB → join all mesh objects
2. **Remove disconnected floaters** — separate by loose parts, delete any part ≤10% of the largest piece's vertex count (kills fins, floating debris)
3. Scale to `--scale-mm` on the longest axis
4. Manifold check → `fill_holes` repair
5. Export binary STL
6. Write `*.cleanup.json` sidecar (dims, triangle count, manifold status)

**Printer modes:**
- `resin`: manifold repair only — hollowing is done in the slicer (Lychee/Chitubox have better heuristics)
- `fdm`: decimate to ~30k triangles for fast slicing

**Known gotchas:**
- `bpy.ops.mesh.separate(type='LOOSE')` must be called from EDIT mode.
- After deleting floaters, re-join survivors before scaling.
- HY3D output is very high-poly (~2M tris). The resin path keeps full resolution; add a decimate step if your slicer is slow.

---

### 4. Print advisory (slicer_prep.py)

`tabletop/pipeline/slicer_prep.py` uses `trimesh` to check watertightness, volume, and suggest orientation.

**Requires:** `trimesh` in the ComfyUI venv.

```powershell
& "tabletop\comfyui\venv\Scripts\pip.exe" install trimesh
```

Currently outputs `*.print.json` next to the STL. If trimesh is missing, the JSON contains `{"ok": false, "error": "trimesh not installed"}` — the STL is still valid.

---

## How to run

### Prerequisites

1. ComfyUI must be running:
```powershell
Start-Process cmd.exe -ArgumentList "/c `"A:\Project False Face\3D generation\tabletop\start_comfyui.bat`""
```

2. Verify: `Invoke-WebRequest http://127.0.0.1:8188/system_stats` should return JSON.

### Single pipeline run

```powershell
$py  = "A:\Project False Face\3D generation\tabletop\comfyui\venv\Scripts\python.exe"
$orch = "A:\Project False Face\3D generation\tabletop\pipeline\orchestrate.py"

& $py $orch `
  --prompt "weathered stone watchtower, dark fantasy, 100mm tabletop terrain" `
  --kind terrain `
  --engine hunyuan `
  --printer resin `
  --out "A:\Project False Face\3D generation\tabletop\outputs\stl\watchtower_v2.stl"
```

**Arguments:**

| Flag | Values | Notes |
|------|--------|-------|
| `--prompt` | any string | Passed to SDXL and used for filenames |
| `--kind` | `terrain`, `mini` | Selects workflow + scale defaults |
| `--engine` | `hunyuan` | Only 3D engine currently installed |
| `--printer` | `resin`, `fdm` | Affects Blender cleanup |
| `--out` | path to `.stl` | Parent dirs created automatically |
| `--scale-mm` | integer | Override default scale (terrain=100, mini=32) |

**Expected runtime:**
- SDXL concept image: 1–3 min
- HY3D mesh (50 steps): 20–50 min (first CUDA warmup step is ~5 min)
- Blender cleanup: 1–3 min
- Total: ~25–55 min

### Outputs

All outputs land in `tabletop/outputs/`:

```
outputs/
  concepts/    <run_id>.png          SDXL concept image
  meshes/      <run_id>.glb          raw HY3D mesh
  stl/         <name>.stl            print-ready STL
               <name>.cleanup.json   Blender stats (dims, tri count, manifold)
               <name>.print.json     trimesh advisory (if installed)
```

---

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Fins / flat planes in STL | Near-white BG pixels → HY3D planar artifacts | Fixed: alpha erosion in `hy3d_infer.py` + floater removal in `blender_cleanup.py` |
| Inference at 326 s/step (CPU) | `enable_model_cpu_offload()` moves model to CPU before crashing | Fixed: `pipeline.to("cuda")` in except block |
| `FileNotFoundError: model.fp16.safetensors` | `from_pretrained` defaults to safetensors | Fixed: `use_safetensors=False` |
| ComfyUI HTTP 500 | `_comment` key in workflow JSON | Fixed: strip non-dict top-level keys before POST |
| `ckpt_name` validation error | Backslash mismatch on Windows | Fixed: use `sdxl\\sd_xl_base_1.0.safetensors` |
| HY3D downloads from HF instead of local | Junction or `HY3DGEN_MODELS` not set | Re-create junction: see Weight path resolution above |
| Pipeline timeout | Default 900 s too short | Fixed: `hunyuan_client.py` timeout = 7200 s |
| Blender not found | Not on PATH | Fixed: `_find_blender()` auto-discovers from default install path |

---

## Quality notes

- **Mesh density**: HY3D 2.1 produces ~2M triangles at default settings. For resin printing at 32–100 mm scale this is fine but slicers may be slow. Decimate in Blender if needed.
- **Surface quality**: First-pass HY3D output has lumpy/noisy surfaces. For hero minis consider 2-pass: generate → review in slicer → re-prompt with tighter description.
- **Scale accuracy**: Blender scales the longest axis to `--scale-mm`. The other two axes scale proportionally.

---

## Roadmap / upgrade paths

| Upgrade | Benefit | Effort |
|---------|---------|--------|
| FLUX.1-dev (gated HF) | Better concept images | Need HF token + `~12 GB` extra |
| Hunyuan3D texture pass | Painted preview for review before print | Enable `mesh_only=False` in `hunyuan_client.py` |
| trimesh install | Print advisory (volume, orientation hints) | `pip install trimesh` in ComfyUI venv |
| Dual RTX 5090 | ~4× faster inference via tensor parallelism | Requires hy3dgen multi-GPU support |
| Voxel remesh in Blender | Uniform topology, better for supports | Add `bpy.ops.object.voxel_remesh()` in cleanup |
