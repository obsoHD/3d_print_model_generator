"""slicer_prep - analyze an STL and emit orientation + support advice.

Doesn't actually slice — that's Lychee/OrcaSlicer's job. We just produce a
JSON advisory the user can read or feed to Claude for paint-prep tips.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    import trimesh
    import numpy as np
except ImportError:
    trimesh = None
    np = None


def analyze(stl_path: str, printer: str = "resin") -> dict:
    """Return a dict with bbox, volume, suggested orientation, overhang
    flags, support hints, est. resin volume."""
    if trimesh is None:
        return {"ok": False, "error": "trimesh not installed"}

    m = trimesh.load(stl_path, force="mesh")
    if m.is_empty:
        return {"ok": False, "error": "empty mesh"}

    info = {
        "ok": True,
        "stl": stl_path,
        "printer": printer,
        "watertight": bool(m.is_watertight),
        "volume_cm3": round(m.volume / 1000.0, 2),    # mm³ -> cm³
        "surface_area_cm2": round(m.area / 100.0, 2),
        "bbox_mm": [list(map(float, m.bounds[0])), list(map(float, m.bounds[1]))],
        "dims_mm": list(map(float, m.bounds[1] - m.bounds[0])),
        "vertex_count": int(len(m.vertices)),
        "triangle_count": int(len(m.faces)),
    }

    # --- orientation suggestion ---
    # For minis on resin: tilt 30-45° back so the face has minimal supports.
    # For terrain on FDM: flat-down on the largest face.
    dims = info["dims_mm"]
    largest_dim_axis = "xyz"[dims.index(max(dims))]
    if printer == "fdm":
        info["suggested_orientation"] = {
            "method": "flat-largest-face",
            "rotate_to_lay_axis": largest_dim_axis,
            "comment": "lay the part with its longest dimension along X-Y; "
                       "auto-orient in OrcaSlicer should match.",
        }
    else:
        info["suggested_orientation"] = {
            "method": "tilt-back-30-45",
            "tilt_deg": 35,
            "axis": "X",
            "comment": "tilt the part 30-45° backward around X so the face / "
                       "front detail surfaces are mostly upward-facing. "
                       "Auto-supports will land on the back & under-base.",
        }

    # --- support intensity hint based on overhang area ---
    # Count downward-facing faces (normal z < -0.3) as a rough overhang proxy.
    norms = m.face_normals
    if np is not None:
        overhang_mask = norms[:, 2] < -0.3
        overhang_area_cm2 = float(m.area_faces[overhang_mask].sum() / 100.0)
    else:
        overhang_area_cm2 = -1.0
    info["overhang_area_cm2_estimate"] = round(overhang_area_cm2, 2)
    if overhang_area_cm2 < 0:
        intensity = "unknown"
    elif overhang_area_cm2 < info["surface_area_cm2"] * 0.05:
        intensity = "light"
    elif overhang_area_cm2 < info["surface_area_cm2"] * 0.15:
        intensity = "medium"
    else:
        intensity = "heavy"
    info["support_intensity_hint"] = intensity

    # --- resin volume estimate (solid; hollow saves ~60-80%) ---
    info["resin_volume_estimate_ml_solid"] = round(m.volume / 1000.0, 2)
    info["resin_volume_estimate_ml_hollowed"] = round(m.volume / 1000.0 * 0.3, 2)

    return info


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--printer", choices=("resin", "fdm"), default="resin")
    args = ap.parse_args(argv)
    print(json.dumps(analyze(args.stl, args.printer), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
