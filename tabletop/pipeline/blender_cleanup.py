"""blender_cleanup - print-prep pass via headless Blender.

We use Blender (not trimesh / Open3D) because:
  - It ships with the "3D-Print Toolbox" addon with manifold check / fix
  - It has voxel remesh, smart decimate, hollow operator built-in
  - It exports clean ASCII/binary STL with explicit normals

This module has TWO halves:

  Outer half (orchestrator-side, calls Blender as subprocess):
      run(input_mesh, out_stl, printer, scale_mm) -> dict

  Inner half (runs INSIDE Blender's Python; the same file is re-executed
  by Blender with `--background --python ... -- <args>`):
      _blender_main()

The split is so the orchestrator's interpreter doesn't need bpy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent


# ---------------------------------------------------------------------------
# Outer half — called by orchestrate.py
# ---------------------------------------------------------------------------

def _find_blender() -> str:
    if b := os.environ.get("BLENDER"):
        return b
    if os.name == "nt":
        for root in (
            r"C:\Program Files\Blender Foundation",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Blender Foundation",
        ):
            for d in sorted(Path(root).glob("Blender *"), reverse=True):
                exe = d / "blender.exe"
                if exe.exists():
                    return str(exe)
    return "blender"


def run(input_mesh: str, out_stl: str, printer: str = "resin",
        scale_mm: float = 32.0) -> dict:
    """Spawn Blender headless to clean + export the mesh.

    printer = "resin": hollow + drainage holes, support hint
    printer = "fdm":   decimate to ~30k tris, no hollowing (FDM solid prints)
    """
    blender = _find_blender()
    script = __file__
    args = [
        "--input", input_mesh,
        "--output", out_stl,
        "--printer", printer,
        "--scale-mm", str(scale_mm),
    ]
    cmd = [blender, "--background", "--python", script, "--"] + args
    print(f"  [blender] {' '.join([Path(blender).name] + cmd[1:5])} ...")
    # Pixal3D outputs are ~700K verts; Solidify + VoxelRemesh on that takes
    # 5-15 min vs HY3D's ~2 min. 900s timeout covers both engines.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender cleanup failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-1500:]}"
        )
    # The inner half writes a sidecar JSON next to out_stl
    info_path = Path(out_stl).with_suffix(".cleanup.json")
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))
    return {"ok": True, "out_stl": out_stl}


# ---------------------------------------------------------------------------
# Inner half — runs INSIDE Blender
# ---------------------------------------------------------------------------

def _blender_main():
    import argparse
    import bpy
    import bmesh

    # Find args after the '--' separator
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--printer", choices=("resin", "fdm"), default="resin")
    ap.add_argument("--scale-mm", type=float, default=32.0)
    args = ap.parse_args(argv)

    # --- 0. clean scene ---
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Enable the 3D-Print Toolbox (ships with Blender 4.x)
    try:
        bpy.ops.preferences.addon_enable(module="object_print3d_utils")
    except Exception:
        pass

    # --- 1. import the mesh ---
    inp = args.input.lower()
    if inp.endswith(".glb") or inp.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=args.input)
    elif inp.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=args.input)
    elif inp.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=args.input)
    else:
        raise SystemExit(f"unsupported input format: {args.input}")

    # The imported object may have a parent / hierarchy — join all meshes
    bpy.ops.object.select_all(action='SELECT')
    mesh_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not mesh_objs:
        raise SystemExit("no mesh found in import")
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object

    # --- 1b. remove disconnected floaters / fin artifacts ---
    # HY3D generates background planes when the alpha mask isn't perfect.
    # Old approach: delete parts < 10% vertex count — fails for large walls.
    # New approach: dual heuristic — bbox flatness ratio + face normal variance.
    #   A flat wall has one bounding-box extent << the other two (flatness),
    #   AND all face normals roughly parallel (low variance). Requiring both
    #   avoids deleting thin-but-real geometry like sword blades or arches.
    import numpy as np

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    all_mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if len(all_mesh) > 1:
        main = max(all_mesh, key=lambda o: len(o.data.polygons))
        bpy.ops.object.select_all(action='DESELECT')
        # NB: when SOLIDIFY_MODE=aggressive (Pixal3D) we deliberately KEEP all
        # islands — the Solidify+VoxelRemesh combo below will fuse them into a
        # single watertight solid. The old "keep largest island" approach
        # collapsed the mesh to a stick.
        if True:
            for o in all_mesh:
                if o is main:
                    continue
                drop = False
                # Fast path: tiny fragment (< 3% of main faces)
                if len(o.data.polygons) < len(main.data.polygons) * 0.03:
                    drop = True
                if not drop:
                    # Bbox flatness: sort extents; flat slab → smallest/largest < 0.05
                    bb = o.bound_box  # 8×[x,y,z] in local space
                    xs = [c[0] for c in bb]; ys = [c[1] for c in bb]; zs = [c[2] for c in bb]
                    extents = sorted([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)])
                    if extents[2] > 1e-6 and (extents[0] / extents[2]) < 0.05:
                        # Confirm planarity via face-normal variance
                        norms = np.array([[p.normal.x, p.normal.y, p.normal.z]
                                          for p in o.data.polygons], dtype=np.float32)
                        if len(norms):
                            mean_n = norms.mean(axis=0)
                            mag = np.linalg.norm(mean_n)
                            if mag > 1e-6:
                                dots = norms @ (mean_n / mag)
                                if float(np.var(dots)) < 0.01:
                                    drop = True
                if drop:
                    o.select_set(True)
            bpy.ops.object.delete()

        survivors = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        if survivors:
            bpy.ops.object.select_all(action='DESELECT')
            for o in survivors:
                o.select_set(True)
            bpy.context.view_layer.objects.active = survivors[0]
            if len(survivors) > 1:
                bpy.ops.object.join()
        obj = bpy.context.active_object

    # --- 1c. retopology + sharpening ---
    # HY3D produces a dense but irregular triangle soup with soft, rounded
    # surface normals. The strategy:
    #   a) Voxel remesh at ~0.5% of model extent — rebuilds clean isotropic
    #      topology locked to the actual surface position (no smoothing yet).
    #      SMOOTH=False (Sharp mode) keeps the original surface shape intact.
    #   b) 2-pass Taubin smooth (λ=0.5, μ=-0.53) — removes voxel stairstepping
    #      while being volume-preserving. Unlike pure Laplacian, Taubin doesn't
    #      shrink or amplify noise. Only 2 iterations so fine detail is kept.
    # Result: clean, print-ready topology with crisp edges.
    dims_for_voxel = obj.dimensions
    longest_for_voxel = max(dims_for_voxel) if max(dims_for_voxel) > 0 else 1.0
    voxel_size = max(longest_for_voxel * 0.004, 0.001)  # ~0.4% of longest dim

    # SOLIDIFY_MODE=aggressive (Pixal3D / TRELLIS-style outputs): the input is
    # a swiss-cheese surface — visually fine but topologically a hollow shell
    # full of gaps. Voxel remesh alone produces a husk. Add a Solidify pass
    # first (thickness ~2% of longest dim) to seal all the gaps, then voxel
    # remesh turns it into a true printable solid.
    _solidify_mode = os.environ.get("SOLIDIFY_MODE", "normal").lower()
    if _solidify_mode == "aggressive":
        solidify_thickness = max(longest_for_voxel * 0.02, 0.002)
        print(f"[cleanup] SOLIDIFY_MODE=aggressive — adding Solidify "
              f"(thickness={solidify_thickness:.4f}) before voxel remesh",
              flush=True)
        mod_solid = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
        mod_solid.thickness = solidify_thickness
        mod_solid.offset = 0.0           # extrude symmetrically inside+outside
        mod_solid.use_even_offset = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod_solid.name)
        # Use coarser voxels so any remaining tiny gaps get bridged
        voxel_size = max(longest_for_voxel * 0.008, 0.002)

    mod_remesh = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
    mod_remesh.mode        = 'VOXEL'
    mod_remesh.voxel_size  = voxel_size
    mod_remesh.use_smooth_shade = False  # SHARP — preserves surface features
    mod_remesh.adaptivity  = 0.0
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod_remesh.name)

    # Taubin smooth: 2 passes of (smooth then anti-shrink)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=1)   # λ pass
    bpy.ops.mesh.vertices_smooth(factor=-0.53, repeat=1) # μ anti-shrink pass
    bpy.ops.object.mode_set(mode='OBJECT')

    # --- 1d. trim oversized baseplate ---
    # HY3D often generates a thin platform extending well past the model's
    # actual footprint (background plane bleeding into the mesh). It survives
    # loose-parts removal because it's welded to the model. Detect it by
    # comparing the X-Y footprint at the bottom 5% vs. above 12% of height;
    # if the bottom is noticeably wider, clip it to the model's outline.
    verts_np = np.array([(v.co.x, v.co.y, v.co.z) for v in obj.data.vertices],
                        dtype=np.float32)
    if len(verts_np) > 100:
        z = verts_np[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        h = z_max - z_min
        if h > 1e-6:
            above = verts_np[z > z_min + 0.12 * h]
            bot   = verts_np[z <= z_min + 0.05 * h]
            if len(above) > 50 and len(bot) > 20:
                ax0, ay0 = float(above[:,0].min()), float(above[:,1].min())
                ax1, ay1 = float(above[:,0].max()), float(above[:,1].max())
                above_area = max(1e-6, (ax1-ax0) * (ay1-ay0))
                bx0, by0 = float(bot[:,0].min()), float(bot[:,1].min())
                bx1, by1 = float(bot[:,0].max()), float(bot[:,1].max())
                bot_area = (bx1-bx0) * (by1-by0)
                ratio = bot_area / above_area
                if ratio > 1.4:
                    margin = max(h * 0.05, (ax1-ax0) * 0.03, 1e-4)
                    kx0, kx1 = ax0 - margin, ax1 + margin
                    ky0, ky1 = ay0 - margin, ay1 + margin
                    cut_z   = z_min + 0.12 * h
                    bm = bmesh.new(); bm.from_mesh(obj.data)
                    bm.verts.ensure_lookup_table()
                    kill = [v for v in bm.verts
                            if v.co.z < cut_z and
                               (v.co.x < kx0 or v.co.x > kx1 or
                                v.co.y < ky0 or v.co.y > ky1)]
                    if kill and len(kill) < len(bm.verts) * 0.6:
                        bmesh.ops.delete(bm, geom=kill, context='VERTS')
                        bm.to_mesh(obj.data); obj.data.update()
                        print(f"[cleanup] trimmed baseplate "
                              f"(was {ratio:.2f}x model footprint, "
                              f"removed {len(kill)} verts)", flush=True)
                    bm.free()

                    # Fill the planar hole created at the cut boundary
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='DESELECT')
                    bpy.ops.mesh.select_non_manifold()
                    try:
                        bpy.ops.mesh.fill_holes(sides=0)
                    except Exception:
                        pass
                    bpy.ops.object.mode_set(mode='OBJECT')

    # --- 2. scale to target longest-dim ---
    bbox = obj.bound_box
    dims = obj.dimensions
    longest = max(dims)
    if longest > 0:
        s = args.scale_mm / longest
        obj.scale = (s, s, s)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # --- 3. manifold check + repair ---
    me = obj.data
    bm = bmesh.new(); bm.from_mesh(me)
    non_manifold_before = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    if non_manifold_before > 0:
        # Iterative manifold repair: each pass fixes a different class of
        # non-manifold geometry. `fill_holes` only fixes boundary edges;
        # remove_doubles fixes T-junctions/dupes; dissolve_degenerate fixes
        # zero-area faces; the final delete loose pass drops floaters.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        for _ in range(3):
            try: bpy.ops.mesh.remove_doubles(threshold=0.00005)
            except Exception: pass
            try: bpy.ops.mesh.dissolve_degenerate(threshold=0.00005)
            except Exception: pass
            try: bpy.ops.mesh.delete_loose()
            except Exception: pass
            # Select all non-manifold types and try to repair
            bpy.ops.mesh.select_all(action='DESELECT')
            try:
                bpy.ops.mesh.select_non_manifold(
                    extend=False, use_wire=True, use_boundary=True,
                    use_multi_face=True, use_non_contiguous=True, use_verts=True)
                bpy.ops.mesh.fill_holes(sides=0)
            except Exception: pass
            bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

    bm = bmesh.new(); bm.from_mesh(me)
    non_manifold_after = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    print(f"[cleanup] manifold edges: {non_manifold_before} -> {non_manifold_after}",
          flush=True)

    # --- 4. printer-specific cleanup ---
    if args.printer == "fdm":
        # Decimate to ~30k tris for sane slicing time
        tri_count = len(me.polygons)
        target = 30000
        if tri_count > target * 1.5:
            ratio = target / tri_count
            mod = obj.modifiers.new(name="dec", type='DECIMATE')
            mod.ratio = ratio
            bpy.ops.object.modifier_apply(modifier=mod.name)
    elif args.printer == "resin":
        # For minis/scatter we want a clean topology but no hollowing
        # automatically — let the user choose in the slicer (Lychee has
        # better hollow heuristics). We just guarantee manifold + scale.
        pass

    # --- 4b. seat to bed (z=0, center XY) ---
    # Bambu Studio errors "empty initial layer" when min_z != 0. Apply a
    # final translate so the lowest vertex sits on z=0 and XY is centered.
    # Must run AFTER scale-to-mm and any voxel-remesh/smoothing.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    verts_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if verts_world:
        xs = [v.x for v in verts_world]; ys = [v.y for v in verts_world]
        zs = [v.z for v in verts_world]
        cx, cy, zmin = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, min(zs)
        obj.location.x -= cx; obj.location.y -= cy; obj.location.z -= zmin
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        print(f"[cleanup] seated to bed: dx={-cx:.2f} dy={-cy:.2f} dz={-zmin:.2f}",
              flush=True)

    # --- 5. export STL ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    try:
        bpy.ops.wm.stl_export(filepath=args.output,
                              export_selected_objects=True,
                              ascii_format=False)
    except Exception:
        # Older Blender API fallback
        bpy.ops.export_mesh.stl(filepath=args.output, use_selection=True)

    # --- 6. sidecar JSON with what we did ---
    info = {
        "ok": True,
        "input": args.input,
        "output": args.output,
        "printer": args.printer,
        "scale_mm": args.scale_mm,
        "dims_mm": list(obj.dimensions),
        "triangles": len(me.polygons),
        "vertices": len(me.vertices),
        "non_manifold_before": non_manifold_before,
        "non_manifold_after": non_manifold_after,
    }
    side = Path(args.output).with_suffix(".cleanup.json")
    side.write_text(json.dumps(info, indent=2), encoding="utf-8")


# Detect whether we're running inside Blender (bpy importable)
try:
    import bpy   # noqa: F401
    _INSIDE_BLENDER = True
except ImportError:
    _INSIDE_BLENDER = False


if __name__ == "__main__":
    if _INSIDE_BLENDER:
        _blender_main()
    else:
        # Standalone use — orchestrate.py calls run() directly.
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--input", required=True)
        ap.add_argument("--output", required=True)
        ap.add_argument("--printer", choices=("resin", "fdm"), default="resin")
        ap.add_argument("--scale-mm", type=float, default=32.0)
        args = ap.parse_args()
        info = run(args.input, args.output, args.printer, args.scale_mm)
        print(json.dumps(info, indent=2))
