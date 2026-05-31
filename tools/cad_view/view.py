"""cad_view - PyVista-based offscreen CAD renderer with multi-angle contact sheets.

Reads STEP (via build123d tessellation) or STL (natively) and renders one or
more camera views to PNG.

Usage:
  python view.py <input.step|input.stl> [--out OUT] [--angle ANGLE]
                  [--contact-sheet] [--width W] [--height H]
                  [--bg COLOR] [--color COLOR]

Examples:
  python view.py model.step --out iso.png --angle iso
  python view.py model.step --out sheet.png --contact-sheet
  python view.py model.stl --out front.png --angle front --width 1200

Supported angles: iso, top, bottom, front, back, left, right.
Contact sheet tiles: iso (large) + top + front + side in a 2x2 grid.

Designed to be promoted to a Claude skill at ~/.claude/skills/cad-view/.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_stl(path: str) -> pv.MultiBlock:
    mesh = pv.read(path)
    return pv.MultiBlock([mesh])


def load_step(path: str, linear_tol: float = 0.2, angular_tol: float = 0.3) -> pv.MultiBlock:
    """Tessellate every solid in a STEP file (or assembly) into PyVista meshes,
    preserving per-shape color when available."""
    from build123d import import_step

    shape = import_step(path)

    # Flatten into a list of (sub_shape, color) — handles Compound assemblies too
    items = []

    def _walk(s):
        # If it's a Compound with children, recurse; else treat as a leaf
        children = getattr(s, "children", None)
        if children:
            for ch in children:
                _walk(ch)
            return
        items.append(s)

    _walk(shape)
    if not items:
        items = [shape]

    blocks = []
    for s in items:
        try:
            verts, tris = s.tessellate(linear_tol, angular_tol)
        except Exception:
            continue
        if not verts or not tris:
            continue
        pts = np.array([(v.X, v.Y, v.Z) for v in verts], dtype=np.float64)
        faces = np.hstack([[3, *t] for t in tris]).astype(np.int64)
        m = pv.PolyData(pts, faces)
        # carry color if set
        col = getattr(s, "color", None)
        if col is not None:
            try:
                m.field_data["rgba"] = np.array([col.red, col.green, col.blue, getattr(col, "alpha", 1.0)])
            except Exception:
                pass
        blocks.append(m)

    if not blocks:
        raise RuntimeError(f"No tessellatable solids in {path}")
    return pv.MultiBlock(blocks)


def load(path: str) -> pv.MultiBlock:
    ext = Path(path).suffix.lower()
    if ext == ".stl":
        return load_stl(path)
    if ext in (".step", ".stp"):
        return load_step(path)
    raise ValueError(f"Unsupported extension {ext}")


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

ANGLES = {
    "iso":    ( 1.0,  1.0,  0.8),
    "top":    ( 0.0,  0.0,  1.0),
    "bottom": ( 0.0,  0.0, -1.0),
    "front":  ( 0.0, -1.0,  0.0),
    "back":   ( 0.0,  1.0,  0.0),
    "left":   (-1.0,  0.0,  0.0),
    "right":  ( 1.0,  0.0,  0.0),
}

VIEW_UP = {
    "iso":    (0, 0, 1),
    "top":    (0, 1, 0),
    "bottom": (0, 1, 0),
    "front":  (0, 0, 1),
    "back":   (0, 0, 1),
    "left":   (0, 0, 1),
    "right":  (0, 0, 1),
}


def set_camera(plotter: pv.Plotter, blocks: pv.MultiBlock, angle: str, zoom: float = 1.0):
    if angle not in ANGLES:
        raise ValueError(f"Unknown angle {angle!r}; choose from {list(ANGLES)}")

    bounds = blocks.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    distance = size * 2.4 / zoom

    direction = ANGLES[angle]
    length = math.sqrt(sum(d * d for d in direction))
    direction = tuple(d / length for d in direction)

    position = (
        center[0] + direction[0] * distance,
        center[1] + direction[1] * distance,
        center[2] + direction[2] * distance,
    )

    plotter.camera_position = [position, center, VIEW_UP[angle]]
    plotter.camera.parallel_projection = False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _add_blocks(plotter: pv.Plotter, blocks: pv.MultiBlock, default_color: str | None = None):
    for block in blocks:
        if block is None:
            continue
        color = default_color
        rgba = block.field_data.get("rgba") if "rgba" in block.field_data.keys() else None
        if rgba is not None:
            color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
        plotter.add_mesh(
            block,
            color=color,
            smooth_shading=True,
            show_edges=False,
            specular=0.2,
            specular_power=20,
            ambient=0.25,
            diffuse=0.85,
        )


def render_single(
    blocks: pv.MultiBlock,
    angle: str,
    out_path: str,
    width: int = 1200,
    height: int = 900,
    bg: str = "white",
    default_color: str | None = None,
    zoom: float = 1.0,
):
    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    pl.set_background(bg)
    _add_blocks(pl, blocks, default_color)
    pl.enable_anti_aliasing()
    pl.add_light(pv.Light(position=(2, 2, 4), light_type="scene light"))
    set_camera(pl, blocks, angle, zoom=zoom)
    pl.screenshot(out_path)
    pl.close()


def render_contact_sheet(
    blocks: pv.MultiBlock,
    out_path: str,
    tile_w: int = 600,
    tile_h: int = 450,
    bg: str = "white",
    default_color: str | None = None,
):
    """2x3 grid: iso, top, front, back, left, right."""
    angles = ["iso", "top", "front", "back", "left", "right"]
    rows, cols = 2, 3

    tmp_dir = Path(tempfile.mkdtemp(prefix="cad_view_"))
    tile_paths = []
    for a in angles:
        p = tmp_dir / f"{a}.png"
        render_single(blocks, a, str(p), width=tile_w, height=tile_h, bg=bg, default_color=default_color)
        tile_paths.append((a, p))

    # composite with PIL (it's a Pillow dep that PyVista usually brings)
    from PIL import Image, ImageDraw, ImageFont
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color="white")
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for i, (a, p) in enumerate(tile_paths):
        r, c = divmod(i, cols)
        tile = Image.open(p)
        sheet.paste(tile, (c * tile_w, r * tile_h))
        draw.text((c * tile_w + 10, r * tile_h + 5), a.upper(), fill="black", font=font)
    sheet.save(out_path)

    # cleanup
    for _, p in tile_paths:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="STEP or STL file")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--angle", default="iso", choices=list(ANGLES))
    ap.add_argument("--contact-sheet", action="store_true",
                    help="Render a 6-view contact sheet instead of single view")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--bg", default="white")
    ap.add_argument("--color", default=None,
                    help="Default color for shapes without a color attribute "
                         "(e.g. 'darkolivegreen' or '#556b2f')")
    ap.add_argument("--zoom", type=float, default=1.0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    blocks = load(args.input)
    if args.contact_sheet:
        render_contact_sheet(blocks, args.out, tile_w=args.width // 2, tile_h=args.height // 2,
                             bg=args.bg, default_color=args.color)
    else:
        render_single(blocks, args.angle, args.out, args.width, args.height,
                      bg=args.bg, default_color=args.color, zoom=args.zoom)
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
