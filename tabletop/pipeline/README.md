# pipeline/  —  the orchestrator

The Python layer that drives the whole flow from prompt to printable STL.

## Entry points

| Script | What it does |
|---|---|
| `orchestrate.py` | Top-level CLI — runs the full pipeline end-to-end. |
| `comfy_client.py` | HTTP client for ComfyUI — submits workflow, polls, fetches image. |
| `sparc3d_client.py` | Calls Sparc3D venv to produce a GLB from a PNG. |
| `hunyuan_client.py` | Calls Hunyuan3D venv to produce a textured preview. |
| `blender_cleanup.py` | Headless Blender script — manifold check, hollow, supports. |
| `slicer_prep.py` | Pure-Python orientation + support hint generator. |

## CLI usage

```bash
python orchestrate.py \
    --prompt "rust-pitted water tower, post-apoc terrain" \
    --kind terrain \
    --out outputs/stl/water_tower.stl
```

Flags:

| Flag | Effect |
|---|---|
| `--prompt TEXT` | Natural-language description of what you want. |
| `--kind {mini,terrain,prop,scatter}` | Picks the ComfyUI workflow + size hints. |
| `--engine {sparc3d,hunyuan,both}` | Default: sparc3d. `both` runs printable + preview. |
| `--printer {resin,fdm}` | Switches Blender cleanup (hollow vs decimate). |
| `--scale-mm FLOAT` | Target longest-dim in mm. Default mini=32, terrain=100. |
| `--seed INT` | Reproducible generation. |
| `--check-models` | Report which engines + LoRAs are available, then exit. |
| `--skip-print-prep` | Stop after mesh generation; don't run Blender. |

## Per-stage flow

```python
from pipeline import orchestrate

result = orchestrate.run(
    prompt="orc berserker, dual axes, dynamic pose",
    kind="mini",
    engine="sparc3d",
    printer="resin",
    scale_mm=32,
)
# result = {
#   "concept_png": "outputs/concepts/<id>.png",
#   "mesh_glb":   "outputs/meshes/<id>.glb",
#   "stl":        "outputs/stl/<id>.stl",
#   "manifest":   "outputs/stl/<id>.print.json",
# }
```

## Failure handling

Every stage emits a sidecar JSON. On failure:
- The orchestrator writes `<id>.error.json` with the stage + reason.
- For non-manifold Sparc3D output (rare), Blender cleanup attempts repair.
- If repair fails, the orchestrator falls back to re-running Sparc3D with
  a slightly different seed.

## Adding a new engine

1. Drop in `<engine>_client.py` exposing `generate(image_path, **kwargs)
   -> mesh_path`.
2. Register in `orchestrate.ENGINES`.
3. Add a `--engine <name>` choice.

That's it — the rest of the pipeline is engine-agnostic.
