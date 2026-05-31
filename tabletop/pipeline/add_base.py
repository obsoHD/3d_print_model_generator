"""add_base - add a thin connector base under a multi-piece generated model.

Pure POST-PROCESS on the finished mesh. Does NOT touch generation.
ONLY for props/terrain with separate ground pieces (e.g. a barrel + apples
spilled around it). Minis are LEFT ALONE — they keep their generated base.

prop mode -> a VERY THIN wafer base that hugs the footprint of the bottom of
             the model and only bridges the GAPS between separate pieces
             (morphological close), without extending past their outline.
             Just enough to fuse the spilled pieces into one flat-sitting
             printable object.

Uses trimesh + manifold3d for a robust watertight boolean union. Input mesh
must be upright (Z-up); output is re-seated to z=0, single watertight solid.

Usage:
    python add_base.py --input m.stl --output based.stl --mode prop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Standard tabletop round-base diameters (mm). We snap mini bases to the
# smallest standard that covers the footprint, so minis fit movement trays.
STD_BASES = [20, 25, 32, 40, 50, 60]


def _round_base(m, thickness, base_d, margin):
    """A round disc base snapped to a standard tabletop size."""
    import trimesh, numpy as np
    cx = float((m.bounds[0, 0] + m.bounds[1, 0]) / 2)
    cy = float((m.bounds[0, 1] + m.bounds[1, 1]) / 2)
    z = m.vertices[:, 2]; zmin, zmax = float(z.min()), float(z.max())
    bottom = m.vertices[z < zmin + 0.25 * (zmax - zmin)]
    if len(bottom) < 10:
        bottom = m.vertices
    rr = np.sqrt((bottom[:, 0] - cx) ** 2 + (bottom[:, 1] - cy) ** 2)
    foot_r = float(np.percentile(rr, 93)) + margin
    if base_d:
        radius = base_d / 2.0
    else:
        radius = next((d for d in STD_BASES if d >= foot_r * 2.0),
                      STD_BASES[-1]) / 2.0
    base = trimesh.creation.cylinder(radius=radius, height=thickness, sections=80)
    base.apply_translation([cx, cy, thickness / 2.0])
    return base, round(radius * 2, 1)


def _footprint_base(m, thickness, close_gap, band_frac):
    """A THIN wafer base that hugs the footprint of the model's bottom and
    only bridges gaps BETWEEN separate pieces — without extending past their
    outline. Uses a morphological close (buffer +d then -d) which fills gaps
    up to 2*d wide while keeping the outer boundary at the true silhouette."""
    import trimesh, numpy as np
    from shapely.ops import unary_union
    z = m.vertices[:, 2]; zmin, zmax = float(z.min()), float(z.max())
    band = (zmax - zmin) * band_frac
    fc_z = m.triangles_center[:, 2]
    mask = np.where(fc_z < zmin + band)[0]
    bottom = m.submesh([mask], append=True) if len(mask) > 20 else m
    shadow = bottom.projected([0, 0, 1])
    polys = list(getattr(shadow, "polygons_full", []))
    if not polys:
        from shapely.geometry import MultiPoint
        polys = [MultiPoint(bottom.vertices[:, :2]).convex_hull]
    # tight union (NO outward growth), then morphological CLOSE to bridge the
    # gaps between spilled pieces. close = buffer(+d).buffer(-d): the outer
    # edge returns to the true silhouette, only inter-piece gaps stay filled.
    merged = unary_union(polys)
    closed = merged.buffer(close_gap, join_style=1).buffer(-close_gap, join_style=1)
    closed = closed.simplify(0.2)
    geoms = list(closed.geoms) if closed.geom_type == "MultiPolygon" else [closed]
    slabs = []
    for g in geoms:
        if g.area < 1.0:
            continue
        try:
            slabs.append(trimesh.creation.extrude_polygon(g, height=thickness))
        except Exception:
            pass
    base = trimesh.util.concatenate(slabs) if len(slabs) > 1 else slabs[0]
    return base, round(float(max(base.extents[0], base.extents[1])), 1)


def add_base(mesh, mode="prop", thickness=None, base_d=None,
             sink=None, close_gap=6.0, band_frac=0.14):
    import trimesh
    m = mesh.copy()
    m.apply_translation([0, 0, -m.bounds[0, 2]])   # re-seat to z=0

    if mode == "mini":
        # Minis are left ALONE — they keep their own generated base.
        return m, {"mode": "mini", "note": "no base added (minis untouched)"}

    # prop: a THIN (1-2 layer) wafer that is the LOWEST layer of the whole
    # piece. Covers the full bottom footprint (barrel + spilled apples) merged
    # into one polygon via a generous morphological close, so nothing floats.
    thickness = thickness or 0.5            # ~2 print layers
    base, base_size = _footprint_base(m, thickness, close_gap, band_frac)

    # Sink the model into the base so EVERY ground-near part (which all sit
    # within ~1mm of the bottom) overlaps and fuses to the base. Sink nearly
    # the whole base thickness; the base stays the lowest layer at z=0.
    sink = thickness - 0.12 if sink is None else min(sink, thickness - 0.1)
    m.apply_translation([0, 0, thickness - sink])
    out = trimesh.boolean.union([base, m])
    if out is None or out.is_empty:
        out = trimesh.util.concatenate([base, m])
    out.apply_translation([0, 0, -out.bounds[0, 2]])
    return out, {"mode": mode, "base_footprint_mm": base_size,
                 "base_thickness_mm": thickness}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=("mini", "prop"), default="prop")
    ap.add_argument("--thickness", type=float, default=None,
                    help="base wafer thickness mm (prop only; default 0.8)")
    ap.add_argument("--close-gap", type=float, default=3.0,
                    help="bridge gaps up to 2x this between pieces (prop)")
    ap.add_argument("--base-d", type=float, default=None)
    args = ap.parse_args()

    import trimesh
    m = trimesh.load(args.input, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    out, info = add_base(m, mode=args.mode, thickness=args.thickness,
                         base_d=args.base_d, close_gap=args.close_gap)
    op = Path(args.output); op.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(op))
    info.update({"watertight": bool(out.is_watertight),
                 "components": len(out.split(only_watertight=False)),
                 "dims_mm": [round(float(x), 1) for x in out.extents]})
    sz = info.get("base_diameter_mm") or info.get("base_footprint_mm")
    print(f"[add_base] {args.mode}: base size={sz}mm "
          f"t={info['base_thickness_mm']}mm -> watertight={info['watertight']} "
          f"comps={info['components']} dims={info['dims_mm']}")
    import json
    Path(args.output).with_suffix(".base.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
