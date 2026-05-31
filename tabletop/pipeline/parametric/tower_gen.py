"""tower_gen - parametric stone defense tower for D&D terrain (build123d).

Watertight manifold solid BY CONSTRUCTION — no marching cubes, no gauntlet,
no thin shells. Hollow interior with real wall thickness, crenellated
battlements, arrow slits, base plinth, flat bottom seated at z=0.

Run in cad_venv:
    python tower_gen.py --output tower.stl --height 100 --diameter 70
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from build123d import (
    BuildPart, BuildSketch, Cylinder, Box, Locations, PolarLocations,
    Align, Mode, Plane, export_stl, Rectangle, extrude,
)


def build_tower(height: float = 100.0,
                diameter: float = 70.0,
                wall: float = 2.5,
                floor: float = 3.0,
                plinth_extra: float = 4.0,
                plinth_h: float = 6.0,
                n_merlon: int = 8,
                crenel_depth: float = 12.0,
                crenel_frac: float = 0.5,
                n_slit: int = 4,
                slit_w: float = 3.0,
                slit_h: float = 16.0):
    """Return a build123d Part: a hollow crenellated tower."""
    outer_r = diameter / 2.0
    inner_r = outer_r - wall
    plinth_r = outer_r + plinth_extra

    with BuildPart() as tower:
        # --- main shaft (solid), bottom at z=0 ---
        Cylinder(radius=outer_r, height=height,
                 align=(Align.CENTER, Align.CENTER, Align.MIN))
        # --- base plinth ring (wider, for stability) ---
        Cylinder(radius=plinth_r, height=plinth_h,
                 align=(Align.CENTER, Align.CENTER, Align.MIN))

        # --- hollow the interior: bore from above the floor, open top ---
        with Locations((0, 0, floor)):
            Cylinder(radius=inner_r, height=height,  # overshoots top → open
                     align=(Align.CENTER, Align.CENTER, Align.MIN),
                     mode=Mode.SUBTRACT)

        # --- crenellations: cut alternating gaps (crenels) in the top rim ---
        # Each crenel is an angular gap. We cut `n_merlon` boxes spaced evenly,
        # each spanning crenel_frac of the inter-merlon angle, from the top
        # down by crenel_depth. Cut full radial thickness (through the wall).
        crenel_z = height - crenel_depth
        gap_angle = 360.0 / n_merlon
        # Box wide/deep enough to cut through the whole wall ring
        cut_len = (outer_r + plinth_extra) * 2.2
        cut_wid = 2 * math.pi * outer_r / n_merlon * crenel_frac
        with Locations((0, 0, crenel_z)):
            with PolarLocations(radius=0, count=n_merlon):
                Box(cut_len, cut_wid, crenel_depth * 1.2,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT)

        # --- arrow slits: thin tall windows through the wall at mid heights ---
        # Stagger them around the shaft at two height bands.
        slit_zs = [height * 0.35, height * 0.60]
        for band, z in enumerate(slit_zs):
            offset = (band % 2) * (180.0 / n_slit)  # stagger alternating bands
            with Locations(Plane.XY.offset(z)):
                with PolarLocations(radius=0, count=n_slit, start_angle=offset):
                    Box(cut_len, slit_w, slit_h,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER),
                        mode=Mode.SUBTRACT)

    return tower.part


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--height", type=float, default=100.0)
    ap.add_argument("--diameter", type=float, default=70.0)
    ap.add_argument("--wall", type=float, default=2.5)
    ap.add_argument("--n-merlon", type=int, default=8)
    ap.add_argument("--n-slit", type=int, default=4)
    args = ap.parse_args()

    part = build_tower(height=args.height, diameter=args.diameter,
                       wall=args.wall, n_merlon=args.n_merlon,
                       n_slit=args.n_slit)

    # Report geometry facts
    bb = part.bounding_box()
    print(f"[tower_gen] bbox mm: "
          f"{bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f}", flush=True)
    print(f"[tower_gen] volume mm3: {part.volume:.0f}", flush=True)
    print(f"[tower_gen] solids: {len(part.solids())}  "
          f"is_manifold: {part.is_manifold}", flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    export_stl(part, str(out))
    print(f"[tower_gen] exported {out} ({out.stat().st_size/1e6:.2f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
