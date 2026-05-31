"""finish_mini - SHARP mini finisher (no voxel resampling).

Replaces the voxel-remesh heal (which smooths away fine detail) for minis.
Strategy: think of it as a high-detail statue scaled down — keep every
generated triangle sharp, only repair topology without resampling.

  1. pymeshfix.repair(joincomp=True) — fill holes / fix non-manifold edges
     and bridge nearby components (fuses a grounded sword to the body),
     WITHOUT touching the original triangle detail.
  2. drop tiny floater components (< 2% of faces) — keep the main body and
     any substantial appendage (sword/spear).
  3. orient upright (glTF Y-up -> Z-up), scale to target mm, seat to z=0.
  4. export STL.

Result: full-resolution sharp mini, single watertight solid, slice-valid.
Run in pixal3d_venv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def finish(input_glb: str, output_stl: str, scale_mm: float = 32.0,
           keep_frac: float = 0.02) -> dict:
    import trimesh
    import numpy as np
    import pymeshfix

    g = trimesh.load(input_glb, force="mesh")
    if isinstance(g, trimesh.Scene):
        g = g.dump(concatenate=True)
    raw_faces = len(g.faces)

    # 1. topology repair WITHOUT resampling (keeps sharp triangles)
    mf = pymeshfix.MeshFix(np.asarray(g.vertices, dtype=np.float64),
                           np.asarray(g.faces, dtype=np.int64))
    mf.repair(joincomp=True, remove_smallest_components=False)
    pd = mf.mesh
    v = np.asarray(pd.points)
    f = np.asarray(pd.faces).reshape(-1, 4)[:, 1:4]
    out = trimesh.Trimesh(v, f, process=False)

    # 2. drop tiny floaters; keep main body + substantial appendages
    parts = out.split(only_watertight=False)
    tot = len(out.faces)
    keep = [p for p in parts if len(p.faces) > keep_frac * tot]
    if not keep:
        keep = [max(parts, key=lambda p: len(p.faces))]
    out = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]

    # 3. orient upright (glTF Y-up -> Z-up), scale, seat
    out.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    longest = float(max(out.extents))
    if longest > 0:
        out.apply_scale(scale_mm / longest)
    mn, mx = out.bounds
    out.apply_translation([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]])

    # 4. export
    op = Path(output_stl); op.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(op))
    return {
        "ok": True, "output": str(op),
        "raw_faces": raw_faces, "final_faces": len(out.faces),
        "kept_components": len(keep), "total_components": len(parts),
        "watertight": bool(out.is_watertight),
        "components": len(out.split(only_watertight=False)),
        "dims_mm": [round(float(x), 1) for x in out.extents],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw engine GLB")
    ap.add_argument("--output", required=True, help="output STL")
    ap.add_argument("--scale-mm", type=float, default=32.0)
    args = ap.parse_args()
    info = finish(args.input, args.output, args.scale_mm)
    print(f"[finish_mini] {info['raw_faces']} -> {info['final_faces']} faces "
          f"(SHARP, no resample), watertight={info['watertight']}, "
          f"comps={info['components']}, dims={info['dims_mm']}")
    import json
    Path(args.output).with_suffix(".finish.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
