# Architecture

## Data flow

```
                          ┌─────────────────┐
   user description ────▶ │  Claude         │
                          │  (orchestrator) │
                          └────┬────────────┘
                               │
                               │  via MCP tool calls
                               ▼
              ┌──────────────────────────────────────┐
              │  pipeline/orchestrate.py             │
              │                                      │
              │   1. concept_director (LLM)          │
              │      → FLUX prompt                   │
              │                                      │
              │   2. comfy_client.py                 │
              │      POST to ComfyUI HTTP API        │
              │      → outputs/concepts/<id>.png     │
              │                                      │
              │   3. user confirms or refines        │
              │                                      │
              │   4. sparc3d_client.py               │
              │      → outputs/meshes/<id>.glb       │
              │      (watertight, manifold, 1024³)   │
              │                                      │
              │   5. (optional) hunyuan_client.py    │
              │      → outputs/previews/<id>.png     │
              │      (textured render for review)    │
              │                                      │
              │   6. blender_cleanup.py              │
              │      headless Blender subprocess:    │
              │      - manifold check                │
              │      - decimate (if FDM)             │
              │      - hollow + drain (if resin)     │
              │      - support hint generation       │
              │      → outputs/stl/<id>.stl          │
              │                                      │
              │   7. slicer_prep.py                  │
              │      orient + support advisory       │
              │      → outputs/stl/<id>.print.json   │
              └──────────────────────────────────────┘
```

## Design decisions

**Why Sparc3D as primary, Hunyuan3D as secondary?**
Sparc3D is the only open-source local 3D generator that *guarantees*
watertight + manifold output — those are hard requirements for slicing.
Hunyuan3D produces better-looking textured assets but doesn't guarantee
print-validity; you can use it for preview but never directly slice from
its output without a manifold-repair pass.

**Why ComfyUI for the image step?**
The 3D generators all take a single clean image as input. ComfyUI is the
canonical local pipeline for SDXL/FLUX, has the LoRA ecosystem, and has an
HTTP API we can drive headlessly from Python. We never click around the
ComfyUI GUI from the pipeline.

**Why Blender headless instead of trimesh / Open3D?**
- Mesh repair operators are richer (e.g. `bpy.ops.mesh.print3d_clean_*`)
- Decimation, voxel-remesh, smart UV are all built-in
- One process, one Python script, runs unattended
- We invoke as `blender --background --python script.py -- <args>` so it
  fits cleanly in the orchestrator as a subprocess

**Why MCP and not a direct CLI?**
Both. The pipeline has a clean Python CLI (`orchestrate.py`) that works
standalone. The MCP server is a thin wrapper exposing each stage as a tool
so Claude can drive it conversationally:
- `concept_image(prompt, style, lora) → png_path`
- `generate_mesh(image_path, engine="sparc3d") → glb_path`
- `cleanup_for_print(glb_path, printer="resin"|"fdm") → stl_path`
- `preview_textured(image_path) → preview_png`

## Coordinate frame conventions

- All meshes saved in Z-up (Blender default), millimeter units.
- Origin at the geometric center of the base for minis, at the
  geometric center for terrain.
- Default print orientation set by `slicer_prep.py` based on overhang
  analysis.

## Failure modes & guards

Every stage emits a manifest JSON next to its output:
- Concept image: prompt, seed, model, LoRA, timestamp
- Mesh: engine, source image, vertex count, watertight flag, volume,
  bounding box
- STL: cleanup operations applied, support estimate, est. resin volume

When a stage fails (non-manifold, too few verts, etc.), the orchestrator
asks Claude to diagnose using a `mesh_critic` prompt and either retry with
adjusted parameters or escalate to the user.

## Local-only contract

Nothing in this pipeline phones home. All weights are downloaded once via
the install scripts and live under `models/`. Outputs stay under
`outputs/`. The MCP server binds to `127.0.0.1` only.
