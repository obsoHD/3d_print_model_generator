"""mesh_gauntlet - the post-engine validation+repair gauntlet that turns a
"technically watertight" generated mesh into something a real slicer will accept.

Runs in the pixal3d_venv (where pymeshfix + tweaker3 live).

Pipeline:
    raw.glb
      -> pymeshfix.MeshFix.repair()    # kill self-intersections, fill holes
      -> wall thickness ray-cast       # detect thin-shell pathology
      -> [optional] Solidify (Blender) # only if thin walls detected
      -> Tweaker-3 auto-orient         # find best print orientation
      -> seat to z=0 + center XY       # fix "empty initial layer"
      -> save as <stem>.gauntlet.glb

The trimesh `is_watertight` check is structurally insufficient — it only
verifies edge-face incidence (combinatorial). It does NOT catch:
  - self-intersections (slicer perimeter generator crashes)
  - zero/near-zero wall thickness (slicer drops layers)
  - inverted normals globally (sliced as a void)
  - degenerate triangles (MC artifacts)
  - bed contact (single-vertex initial layer)

All of those are exactly what marching-cubes outputs from SDF generators have.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def repair_topology(mesh):
    """pymeshfix.MeshFix.repair — Attene 2010, gold-standard topology repair."""
    import numpy as np
    import pymeshfix
    mf = pymeshfix.MeshFix(np.asarray(mesh.vertices, dtype=np.float64),
                           np.asarray(mesh.faces, dtype=np.int64))
    mf.repair(joincomp=False, remove_smallest_components=True)
    import trimesh
    # pymeshfix 0.18+ stores result on .mesh (pyvista PolyData); older
    # versions exposed .v / .f. Handle both.
    if hasattr(mf, "v") and hasattr(mf, "f"):
        return trimesh.Trimesh(mf.v, mf.f, process=False)
    pd = mf.mesh                # pyvista PolyData
    v  = np.asarray(pd.points)
    # pyvista faces is a 1D array of [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces_flat = np.asarray(pd.faces).reshape(-1, 4)
    f  = faces_flat[:, 1:4]
    return trimesh.Trimesh(v, f, process=False)


def min_wall_thickness(mesh, sample: int = 5000, max_dist: float = 50.0) -> float:
    """5th-percentile wall thickness via ray-cast along inverted normals.
    Returns thickness in mesh units. INF if rays don't hit anything.
    """
    import numpy as np
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return 0.0
    idx = np.random.choice(n_faces, min(sample, n_faces), replace=False)
    origins = mesh.triangles_center[idx] - mesh.face_normals[idx] * 1e-4
    dirs    = -mesh.face_normals[idx]
    locs, ridx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    if len(locs) == 0:
        return float("inf")
    dists = np.linalg.norm(locs - origins[ridx], axis=1)
    return float(np.percentile(dists, 5))


def auto_orient(mesh, extended: bool = True):
    """Tweaker-3: find optimal print orientation. Returns rotated mesh + score."""
    import numpy as np
    from tweaker3.MeshTweaker import Tweak
    # Tweaker-3 preprocess(): `row_number = int(len(content) / 3)` then
    # reshape to `(row_number, 3, 3)`. So `len(content)` must equal 3*N
    # (number of vertex-rows across all faces). Pass shape `(N*3, 3)`.
    tri_array = np.asarray(mesh.triangles,
                           dtype=np.float64).reshape(-1, 3)   # (N*3, 3)
    res = Tweak(tri_array, extended_mode=extended, verbose=False,
                show_progress=False, favside=None, min_volume=False)
    R = np.array(res.matrix, dtype=np.float64)
    if R.shape == (3, 3):
        T = np.eye(4); T[:3, :3] = R
    else:
        T = R
    mesh.apply_transform(T)
    return mesh, float(getattr(res, "unprintability", -1.0))


def seat_to_bed(mesh):
    """Translate so min(z)=0, center XY at origin. Slicer bed-friendly."""
    mn, mx = mesh.bounds
    mesh.apply_translation([-(mn[0] + mx[0]) / 2,
                            -(mn[1] + mx[1]) / 2,
                            -mn[2]])
    return mesh


def solidify_via_blender(input_glb: str, output_glb: str,
                         voxel_size_mm: float) -> bool:
    """Subprocess: Blender voxel-remesh, which produces a watertight solid
    by reconstructing an isosurface from the implicit occupancy field.
    Cell size in mm — thin walls get fused, larger features preserved.
    Returns True on success."""
    import subprocess
    from pathlib import Path
    here = Path(__file__).resolve().parent
    inner = here / "_blender_solidify.py"
    bl = None
    for cand in (
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
    ):
        if Path(cand).exists():
            bl = cand; break
    if bl is None:
        print(f"[gauntlet] Blender not found; skipping solidify", flush=True)
        return False
    cmd = [bl, "--background", "--python", str(inner), "--",
           "--input", input_glb, "--output", output_glb,
           "--voxel-size", str(voxel_size_mm)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        print(f"[gauntlet] solidify failed (exit {proc.returncode}); "
              f"stderr: {proc.stderr[-400:]}", flush=True)
        return False
    print(f"[gauntlet] voxel-remesh solidify done "
          f"(voxel={voxel_size_mm} mm)", flush=True)
    return True


def run_gauntlet(input_mesh: str, output_mesh: str,
                 min_wall_mm: float = 0.6,
                 scale_mm: float = 100.0,
                 do_orient: bool = True,
                 do_repair: bool = True,
                 do_solidify: bool = True,
                 force_voxel: float = 0.0) -> dict:
    """Full pipeline. Returns report dict; also writes output_mesh GLB."""
    import numpy as np
    import trimesh
    from pathlib import Path
    rpt: dict = {"ok": False, "input": input_mesh, "output": output_mesh}

    print(f"[gauntlet] loading {Path(input_mesh).name}...", flush=True)
    mesh = trimesh.load(input_mesh, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    rpt["raw"] = {
        "verts": len(mesh.vertices), "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "splits_before": len(mesh.split(only_watertight=False)),
    }

    # 1. drop floaters — keep only the dominant connected component
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        main = max(parts, key=lambda p: len(p.faces))
        dropped = len(parts) - 1
        mesh = main
        print(f"[gauntlet] dropped {dropped} floater pieces; kept "
              f"{len(mesh.faces)} faces", flush=True)
        rpt["floaters_dropped"] = dropped

    # 2. topology repair (kills self-intersections, fills holes)
    if do_repair:
        try:
            n_before = len(mesh.faces)
            mesh = repair_topology(mesh)
            print(f"[gauntlet] pymeshfix repair: {n_before} -> "
                  f"{len(mesh.faces)} faces; watertight={mesh.is_watertight}",
                  flush=True)
            rpt["repaired"] = {
                "verts": len(mesh.vertices), "faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
            }
        except Exception as e:
            print(f"[gauntlet] WARN pymeshfix failed ({e}); continuing", flush=True)
            rpt["repaired"] = {"error": str(e)}

    # 3. wall thickness — convert mesh to TARGET print scale first so threshold
    #    is in real mm, not arbitrary unit-cube units.
    longest = max(mesh.bounds[1] - mesh.bounds[0])
    if longest > 0:
        mesh.apply_scale(scale_mm / longest)
    # Sample 800 (not 3000) — rtree's R-tree index OOMs on dense MC meshes.
    try:
        t = min_wall_thickness(mesh, sample=800)
        rpt["min_wall_thickness_mm_p5"] = round(t, 3)
        print(f"[gauntlet] wall thickness 5th percentile = {t:.3f} mm "
              f"(threshold {min_wall_mm} mm)", flush=True)
        rpt["thin_shell_detected"] = bool(t < min_wall_mm)
    except Exception as e:
        print(f"[gauntlet] WARN thickness check failed ({e}); "
              f"assuming thin shell (SDF engines always need solidify)",
              flush=True)
        rpt["min_wall_thickness_mm_p5"] = None
        # Default to assume thin shell — every SDF gen model has this bug
        rpt["thin_shell_detected"] = True

    # 4. auto-orient
    if do_orient:
        try:
            mesh, score = auto_orient(mesh)
            rpt["tweaker_unprintability"] = round(score, 2)
            print(f"[gauntlet] tweaker-3 oriented; unprintability={score:.2f}",
                  flush=True)
        except Exception as e:
            print(f"[gauntlet] WARN tweaker failed ({e}); skipping orient",
                  flush=True)

    # 5. seat to bed
    mesh = seat_to_bed(mesh)
    mn, mx = mesh.bounds
    rpt["final_bbox_mm"]  = [list(map(float, mn)), list(map(float, mx))]
    rpt["final_dims_mm"]  = list(map(float, mx - mn))

    # 5b. Voxel-remesh-as-solidify if thin shell detected. This treats the
    # mesh as an implicit occupancy field and re-extracts the isosurface,
    # which is guaranteed watertight and manifold by construction. Coarser
    # voxel size fuses parallel thin walls into a single solid.
    # CRITICAL: use STL not GLB for the temp handoff — GLB has glTF unit
    # conventions (meters) which makes Blender re-scale our mm-sized mesh.
    # MINI HEAL: force a fine voxel remesh at `force_voxel` mm. Guarantees a
    # manifold single solid AND fuses thin appendages (e.g. a grounded sword
    # that came out as a separate component) into the body. Fine enough on a
    # ~32mm mini (0.3-0.4mm) to keep tabletop-readable detail.
    if force_voxel and force_voxel > 0:
        tmp_in  = Path(output_mesh).with_suffix(".prevoxel.stl")
        tmp_out = Path(output_mesh).with_suffix(".voxel.stl")
        mesh.export(str(tmp_in))
        if solidify_via_blender(str(tmp_in), str(tmp_out), force_voxel):
            import trimesh
            mesh = trimesh.load(str(tmp_out), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            parts = mesh.split(only_watertight=False)
            if len(parts) > 1:
                mesh = max(parts, key=lambda p: p.volume if p.is_volume
                           else len(p.faces))
            mesh = seat_to_bed(mesh)
            rpt["mini_voxel_mm"] = force_voxel
            rpt["mini_heal_applied"] = True
            print(f"[gauntlet] mini-heal voxel remesh ({force_voxel}mm): "
                  f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
                  f"watertight={mesh.is_watertight}, "
                  f"splits={len(mesh.split(only_watertight=False))}", flush=True)
        for p in (tmp_in, tmp_out):
            try: p.unlink()
            except Exception: pass

    if rpt.get("thin_shell_detected") and do_solidify and not (force_voxel and force_voxel > 0):
        tmp_in  = Path(output_mesh).with_suffix(".presolidify.stl")
        tmp_out = Path(output_mesh).with_suffix(".solidify.stl")
        mesh.export(str(tmp_in))
        # Voxel size: coarse enough to fuse the thin shell (~3x wall thickness)
        # but fine enough to keep visible features. Clamp [0.6mm, 1.5mm].
        vox = max(0.6, min(min_wall_mm * 3, 1.5))
        if solidify_via_blender(str(tmp_in), str(tmp_out), vox):
            import trimesh
            mesh = trimesh.load(str(tmp_out), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            # Voxel-remesh produces a main solid + small floaters from isolated
            # voxels (battlement details, etc). Drop them to get one printable.
            post_parts = mesh.split(only_watertight=False)
            if len(post_parts) > 1:
                main = max(post_parts, key=lambda p: p.volume if p.is_volume
                           else len(p.faces))
                rpt["solidify_floaters_dropped"] = len(post_parts) - 1
                mesh = main
            mesh = seat_to_bed(mesh)
            # Re-check wall thickness on the actual final mesh
            try:
                t_final = min_wall_thickness(mesh, sample=3000)
                rpt["min_wall_thickness_mm_p5_final"] = round(t_final, 3)
            except Exception:
                pass
            rpt["solidify_voxel_mm"] = vox
            rpt["solidify_applied"] = True
            print(f"[gauntlet] post-solidify (after floater drop): "
                  f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
                  f"watertight={mesh.is_watertight}, "
                  f"splits={len(mesh.split(only_watertight=False))}, "
                  f"final wall thickness p5={rpt.get('min_wall_thickness_mm_p5_final')} mm",
                  flush=True)
        else:
            rpt["solidify_applied"] = False
        for p in (tmp_in, tmp_out):
            try: p.unlink()
            except Exception: pass

    # 6. final topology snapshot
    rpt["final"] = {
        "verts": len(mesh.vertices), "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "volume_mm3": float(mesh.volume) if mesh.is_volume else None,
        "splits": len(mesh.split(only_watertight=False)),
    }

    out = Path(output_mesh); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    rpt["ok"] = True
    print(f"[gauntlet] wrote {out.name} "
          f"({out.stat().st_size/1e6:.1f} MB, "
          f"watertight={mesh.is_watertight}, "
          f"is_volume={mesh.is_volume}, "
          f"splits={rpt['final']['splits']})", flush=True)
    return rpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",       required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--scale-mm",    type=float, default=100.0)
    ap.add_argument("--min-wall-mm", type=float, default=0.6)
    ap.add_argument("--no-orient",   action="store_true")
    ap.add_argument("--no-repair",   action="store_true")
    ap.add_argument("--no-solidify", action="store_true",
                    help="Skip voxel-remesh solidify (destroys detail). "
                         "Use for clean engine output like Hunyuan3D 2.1.")
    ap.add_argument("--force-voxel", type=float, default=0.0,
                    help="Mini-heal: fine voxel remesh at this mm size. "
                         "Guarantees manifold + fuses detached appendages "
                         "(e.g. a grounded sword). Use ~scale/90 for minis.")
    args = ap.parse_args()
    rpt = run_gauntlet(args.input, args.output,
                       min_wall_mm=args.min_wall_mm,
                       scale_mm=args.scale_mm,
                       do_orient=not args.no_orient,
                       do_repair=not args.no_repair,
                       do_solidify=not args.no_solidify,
                       force_voxel=args.force_voxel)
    sidecar = Path(args.output).with_suffix(".gauntlet.json")
    sidecar.write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    print(f"[gauntlet] report -> {sidecar.name}", flush=True)
    return 0 if rpt.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
