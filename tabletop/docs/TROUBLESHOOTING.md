# Troubleshooting

## ComfyUI

**"ConnectionRefusedError on 127.0.0.1:8188"**
ComfyUI isn't running. Start it:
```
.\comfyui\venv\Scripts\python.exe .\comfyui\main.py --listen 127.0.0.1
```

**"Workflow rejected" / 400 on /prompt**
The workflow JSON has a missing model file. Check ComfyUI's terminal for
which file it can't find. Usually means a checkpoint or LoRA hasn't been
downloaded — see `install/05_install_loras.md`.

**Out of memory (OOM) on FLUX**
Add `--lowvram` to ComfyUI startup. On 16 GB this gives some breathing
room. You may also need `--gpu-only` removed and let CPU offload kick in.

## Sparc3D

**"SPARC3D_CLI override needed"**
The upstream Sparc3D repo has changed its CLI shape. Set:
```bash
export SPARC3D_CLI="sparc3d/venv/bin/python sparc3d/scripts/inference.py --image {image} --output {output} --resolution {resolution}"
```
(Use the actual entry script path from the repo.)

**Mesh has holes despite Sparc3D's "watertight" claim**
Rare but happens on very complex inputs. The Blender cleanup pass
attempts `mesh.fill_holes`. If that fails, re-run Sparc3D with a
different seed.

**Sparc3D very slow on first run**
First inference compiles CUDA kernels. Subsequent runs are 5-10× faster.
Don't kill it during the first run.

## Hunyuan3D

**"CUDA out of memory" during texture step on RTX 5080**
Expected — texture wants ~20 GB. Either:
- Use `--mesh-only` (skip texture)
- Set `low_vram=True` in `hunyuan_client.generate()` — slower but fits

**Hunyuan3D output is gorgeous but won't slice**
Expected. Hunyuan3D doesn't guarantee manifold. Either:
1. Use Sparc3D for the print and Hunyuan3D only for previewing, OR
2. Run the Blender cleanup pass — `print3d_clean_*` ops can repair most
   non-manifold edges.

## Blender

**"blender: command not found"**
Blender isn't on PATH. See `install/04_install_blender.md`.

**Cleanup script fails with import error**
You ran the script outside Blender. Use:
```
blender --background --python pipeline/blender_cleanup.py -- \
    --input mesh.glb --output out.stl --printer resin --scale-mm 32
```

**STL exports rotated 90° vs the GLB**
GLB uses Y-up, STL convention is Z-up. Blender handles this on import,
but check the orientation in `slicer_prep.analyze()` output and rotate
in the slicer if needed.

## MCP

**Claude doesn't see the tabletop tools**
1. Confirm the MCP config path is correct.
2. Restart Claude Desktop fully (quit, not just close).
3. Check `tabletop/mcp/venv/Scripts/python.exe` exists — re-run
   `install/06_install_mcp.{ps1,sh}` if not.
4. Test the server standalone: `python tabletop/mcp/server.py` — should
   wait for stdin and not crash.

**Tools appear but fail with "ComfyUI not running"**
The MCP server doesn't start ComfyUI for you. Keep a ComfyUI instance
running in a terminal whenever you want to generate.

## Print failures

**Resin print has hollow walls / leaks**
The mesh wasn't watertight. Run the print prep manifest:
```
python pipeline/slicer_prep.py outputs/stl/<id>.stl --printer resin
```
The output JSON has `"watertight": false` → re-export through Blender's
3D-Print Toolbox manifold-fix.

**FDM print has stringy mess on overhangs**
Auto-supports weren't aggressive enough. Re-import into OrcaSlicer with
"tree support" and 60% threshold instead of 45°.

**Mini lost facial detail at 32 mm scale**
Sparc3D at 1024³ has ~30 µm voxels. Your printer's actual XY resolution
is probably ~50 µm, which is more than enough. The issue is usually that
the FLUX concept image didn't have crisp facial features. Re-prompt with
"clean sharp facial features, eyes visible, defined cheekbones" added.
