"""barrel_iter - iterate a believable wooden barrel (revolved profile + staves + hoops).

The old barrel stacked cylinders -> looked like an exhaust part. A real barrel:
  - bulged silhouette (revolve a curved profile, not stacked discs)
  - vertical staves (wood plank seams) cut around the body
  - 2-4 raised metal hoops
  - slightly recessed top (open) or domed lid
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from build123d import (
    BuildPart, BuildSketch, BuildLine, Line, ThreePointArc, make_face,
    revolve, Axis, Plane, Locations, PolarLocations, Cylinder, Box,
    Align, Mode, export_stl,
)


def barrel(height=36.0, r_end=11.0, r_mid=13.0, n_stave=14, n_hoop=2,
           flute_rc=1.3, flute_depth=1.1, hoop_proud=0.8, hoop_h=2.0,
           wall=2.5, hollow=True, bottom_chime=True):
    with BuildPart() as p:
        # --- body: revolve a bulged half-profile around Z ---
        with BuildSketch(Plane.XZ) as sk:
            with BuildLine():
                Line((0, 0), (r_end, 0))                              # bottom
                ThreePointArc((r_end, 0), (r_mid, height / 2),
                              (r_end, height))                        # bulged side
                Line((r_end, height), (0, height))                   # top
                Line((0, height), (0, 0))                            # axis
            make_face()
        revolve(axis=Axis.Z)

        # --- vertical stave seams: cylinder FLUTES that follow the bulge.
        # A vertical cylinder placed so its inner edge sits flute_depth below
        # the belly surface carves a concave seam — naturally deeper at the
        # belly and shallower toward the ends, exactly like real barrel staves.
        cyl_center_r = r_mid + flute_rc - flute_depth
        with PolarLocations(cyl_center_r, n_stave):
            Cylinder(flute_rc, height * 1.2,
                     align=(Align.CENTER, Align.CENTER, Align.CENTER),
                     mode=Mode.SUBTRACT)

        # --- raised metal hoops (thin, flush iron bands) ---
        hoop_zs = [height * (k + 1) / (n_hoop + 1) for k in range(n_hoop)]
        for z in hoop_zs:
            t = (z - height / 2) / (height / 2)
            r_here = r_mid - (r_mid - r_end) * (t * t)
            with Locations((0, 0, z)):
                Cylinder(r_here + hoop_proud, hoop_h,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER),
                         mode=Mode.ADD)
        # top rim + bottom chime hoops
        rim_zs = [height - hoop_h / 2 - 0.5]
        if bottom_chime:
            rim_zs.append(hoop_h / 2 + 0.5)
        for z in rim_zs:
            with Locations((0, 0, z)):
                Cylinder(r_end + hoop_proud, hoop_h,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER),
                         mode=Mode.ADD)

        # --- hollow it (open top) ---
        if hollow:
            with Locations((0, 0, wall)):
                Cylinder(r_end - wall, height,
                         align=(Align.CENTER, Align.CENTER, Align.MIN),
                         mode=Mode.SUBTRACT)
    return p.part


PRESETS = {
    # V1 silhouette kept; deeper full-height flute staves + bottom chime hoop
    "v4": dict(n_stave=14, n_hoop=2, r_mid=13.0, r_end=11.0,
               flute_rc=1.3, flute_depth=1.1, hoop_proud=0.8, hoop_h=2.0),
    "v5": dict(n_stave=16, n_hoop=2, r_mid=13.0, r_end=11.2,
               flute_rc=1.1, flute_depth=0.9, hoop_proud=0.7, hoop_h=1.8),
    "v6": dict(n_stave=12, n_hoop=3, r_mid=13.2, r_end=10.8,
               flute_rc=1.5, flute_depth=1.3, hoop_proud=0.9, hoop_h=2.0),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1", choices=list(PRESETS))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    part = barrel(**PRESETS[args.preset])
    bb = part.bounding_box()
    info = {"preset": args.preset,
            "bbox_mm": [bb.size.X, bb.size.Y, bb.size.Z],
            "volume_mm3": part.volume,
            "is_manifold": bool(part.is_manifold),
            "solids": len(part.solids())}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, args.output)
    print(json.dumps(info, indent=2))
