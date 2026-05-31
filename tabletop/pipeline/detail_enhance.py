"""detail_enhance - apply PyMeshLab detail-sharpening filters to a printable STL.

Run AFTER blender_cleanup. HY3D + Blender produces a topologically clean but
visually-soft mesh — voxel remesh smooths everything to the voxel grid, and
Taubin smoothing further rounds sharp creases (battlements, brick lines,
rune carvings). PyMeshLab's normal-unsharp-mask sharpens the normal field
without changing topology, and two-steps smoothing is volume-preserving and
edge-preserving (much better than Laplacian-based Taubin for crisp detail).

Filters applied in order:
    1. apply_normal_unsharp_mask_per_vertex
         Sharpens the normal field, equivalent to an unsharp-mask in 2D
         but applied to per-vertex normals. Makes creases pop.
    2. apply_coord_two_steps_smoothing
         Two-step smoothing variant tuned for edge preservation. Removes
         remaining marching-cubes stair-stepping while keeping sharp edges
         intact.

These are pure mesh ops — no concept image needed. ~1-3s on a 250k-tri mesh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def enhance(in_path: str, out_path: str | None = None,
            unsharp_iters: int = 2, unsharp_weight: float = 0.3,
            smooth_steps: int = 2) -> dict:
    """Apply detail-enhancement filters. In-place if out_path is None."""
    import pymeshlab
    src = Path(in_path)
    dst = Path(out_path) if out_path else src
    print(f"[detail_enhance] loading {src.name}", flush=True)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    in_verts = ms.current_mesh().vertex_number()
    in_faces = ms.current_mesh().face_number()

    # Step 1: normal unsharp mask — sharpens edges via normal-field high-pass
    # Filter signature varies by pymeshlab version; try multiple names.
    sharpened = False
    for filter_name, kwargs in [
        ("apply_normal_unsharp_mask_per_vertex",
         {"iterations": unsharp_iters, "weight": unsharp_weight}),
        ("compute_normal_unsharp_per_vertex",
         {"iterations": unsharp_iters, "weight": unsharp_weight}),
        ("apply_filter_unsharp_normals",
         {"weight": unsharp_weight, "iterations": unsharp_iters}),
    ]:
        try:
            ms.apply_filter(filter_name, **kwargs)
            print(f"[detail_enhance] sharpened normals via {filter_name}", flush=True)
            sharpened = True
            break
        except Exception:
            continue
    if not sharpened:
        print("[detail_enhance] WARN: no normal-unsharp filter available "
              "in this pymeshlab build; skipping sharpen", flush=True)

    # Step 2: edge-preserving two-step smoothing
    smoothed = False
    for filter_name, kwargs in [
        ("apply_coord_two_steps_smoothing",
         {"stepsmoothnum": smooth_steps, "normalthr": 60.0,
          "stepnormalnum": 2, "stepfitnum": 2}),
        ("apply_filter_taubin_smooth",
         {"lambda_": 0.5, "mu": -0.53, "stepsmoothnum": smooth_steps}),
        ("apply_coord_laplacian_smoothing",
         {"stepsmoothnum": smooth_steps}),
    ]:
        try:
            ms.apply_filter(filter_name, **kwargs)
            print(f"[detail_enhance] smoothed via {filter_name}", flush=True)
            smoothed = True
            break
        except Exception:
            continue

    out_verts = ms.current_mesh().vertex_number()
    out_faces = ms.current_mesh().face_number()
    print(f"[detail_enhance] verts {in_verts}->{out_verts}  faces {in_faces}->{out_faces}",
          flush=True)

    ms.save_current_mesh(str(dst))
    print(f"[detail_enhance] saved -> {dst.name}", flush=True)
    return {
        "ok": True, "input": str(src), "output": str(dst),
        "sharpened": sharpened, "smoothed": smoothed,
        "verts_in": in_verts, "verts_out": out_verts,
        "faces_in": in_faces, "faces_out": out_faces,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--unsharp-iters",  type=int,   default=2)
    ap.add_argument("--unsharp-weight", type=float, default=0.3)
    ap.add_argument("--smooth-steps",   type=int,   default=2)
    args = ap.parse_args()
    info = enhance(args.input, args.output,
                   args.unsharp_iters, args.unsharp_weight, args.smooth_steps)
    import json
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
