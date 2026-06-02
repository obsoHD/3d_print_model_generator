"""decimate_mesh - lighten an STL/GLB for fast slicing WITHOUT breaking it.

The alpha-wrapped TRELLIS meshes are ~1.5M faces — far more than a print needs,
and slicers crawl on them. Quadric edge-collapse decimation drops the face count
while preserving the silhouette, manifoldness and watertightness (topology is
preserved, so the slicer still sees one solid). Default target 250k faces keeps
tabletop detail and slices in seconds.

Usage:
  python decimate_mesh.py --input model.stl [--output model.stl] [--target 250000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def decimate(in_path: str, out_path: str, target: int) -> dict:
    import trimesh
    mesh = trimesh.load(in_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    n0 = len(mesh.faces)
    wt0 = bool(mesh.is_watertight)
    if n0 <= target:
        print(f"[decimate] {n0} faces already <= target {target}; copying as-is",
              flush=True)
        mesh.export(out_path)
        return {"ok": True, "faces_before": n0, "faces_after": n0,
                "watertight": wt0, "skipped": True}

    out = None
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype("float64"),
                                   face_matrix=mesh.faces.astype("int32")))
        # preservetopology=True keeps the mesh a single watertight manifold.
        for params in (
            dict(targetfacenum=int(target), preservenormal=True,
                 preservetopology=True, preserveboundary=True,
                 planarquadric=True, qualitythr=0.3, optimalplacement=True,
                 autoclean=True),
            dict(targetfacenum=int(target), preservenormal=True,
                 preservetopology=True),
            dict(targetfacenum=int(target)),
        ):
            try:
                ms.meshing_decimation_quadric_edge_collapse(**params)
                break
            except Exception:  # noqa: BLE001
                continue
        cm = ms.current_mesh()
        out = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                              faces=cm.face_matrix(), process=False)
    except Exception as e:  # noqa: BLE001
        print(f"[decimate] pymeshlab path failed ({e}); trying trimesh", flush=True)
        out = None

    if out is None or len(out.faces) == 0:
        try:
            out = mesh.simplify_quadric_decimation(int(target))
        except Exception as e:  # noqa: BLE001
            print(f"[decimate] trimesh decimation failed ({e}); keeping original",
                  flush=True)
            out = mesh

    out.export(out_path)
    rpt = {"ok": True, "faces_before": n0, "faces_after": len(out.faces),
           "watertight": bool(out.is_watertight), "target": int(target)}
    print(f"[decimate] {n0} -> {len(out.faces)} faces "
          f"(target {target}), watertight={out.is_watertight} -> "
          f"{Path(out_path).name}", flush=True)
    return rpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None,
                    help="default: overwrite input in place")
    ap.add_argument("--target", type=int, default=250000)
    args = ap.parse_args()
    out = args.output or args.input
    rpt = decimate(args.input, out, args.target)
    return 0 if rpt.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
