"""compare - put a candidate CAD next to a reference and report deltas.

Takes two inputs (STL or STEP each) and produces:
  * A side-by-side contact sheet (candidate left, reference right, same angles)
  * A textual diff: bounding box delta, volume delta, surface-area delta,
    and per-axis aspect comparison

Use this every iteration to see at a glance how close your CAD is to the
target reference model.

Usage:
  python compare.py <candidate.step> <reference.stl>
                    [--out-sheet out.png] [--out-report report.md]
                    [--angles iso,top,front,...]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

# import sibling modules
import view as cad_view
import analyze_stl as cad_analyze

DEFAULT_ANGLES = ["iso", "top", "front", "right"]


def _as_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() in (".step", ".stp"):
        # convert step -> tmp stl via build123d
        from build123d import import_step
        shape = import_step(str(path))

        items = []

        def _walk(s):
            children = getattr(s, "children", None)
            if children:
                for ch in children:
                    _walk(ch)
                return
            items.append(s)

        _walk(shape)
        if not items:
            items = [shape]
        # concatenate tessellated meshes
        all_verts, all_faces = [], []
        offset = 0
        for s in items:
            try:
                v, f = s.tessellate(0.2, 0.3)
            except Exception:
                continue
            if not v or not f:
                continue
            all_verts.extend([(p.X, p.Y, p.Z) for p in v])
            for tri in f:
                all_faces.append([tri[0] + offset, tri[1] + offset, tri[2] + offset])
            offset += len(v)
        if not all_verts:
            raise RuntimeError(f"Could not tessellate {path}")
        return trimesh.Trimesh(vertices=np.array(all_verts), faces=np.array(all_faces))
    return trimesh.load_mesh(str(path))


def _stats(mesh: trimesh.Trimesh) -> dict:
    bb_min, bb_max = mesh.bounds
    size = bb_max - bb_min
    aspect = size / float(size.max()) if size.max() else size
    watertight = bool(mesh.is_watertight)
    return {
        "size_mm": [round(float(v), 2) for v in size],
        "long_axis": "XYZ"[int(np.argmax(size))],
        "aspect": [round(float(v), 3) for v in aspect],
        "surface_area_mm2": round(float(mesh.area), 1),
        "volume_mm3": round(float(mesh.volume), 1) if watertight else None,
        "is_watertight": watertight,
        "face_count": int(len(mesh.faces)),
    }


def _delta_line(label, a, b, unit="mm"):
    if a is None or b is None:
        return f"- {label}: candidate={a}  reference={b}"
    if isinstance(a, list):
        deltas = [round(av - bv, 2) for av, bv in zip(a, b)]
        return f"- {label}: candidate={a}  reference={b}  delta={deltas}"
    delta = round(a - b, 2)
    sign = "+" if delta >= 0 else ""
    pct = ""
    if isinstance(b, (int, float)) and b != 0:
        pct = f"  ({sign}{round(delta / b * 100, 1)}%)"
    return f"- {label}: candidate={a}{unit}  reference={b}{unit}  delta={sign}{delta}{unit}{pct}"


def render_pair_sheet(cand_path, ref_path, out_path, angles, tile_w=720, tile_h=540):
    cand_blocks = cad_view.load(str(cand_path))
    ref_blocks = cad_view.load(str(ref_path))

    cols = 2
    rows = len(angles)
    header_h = 36
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    draw.text((10, 6), f"CANDIDATE  ({Path(cand_path).name})", fill="black", font=font)
    draw.text((tile_w + 10, 6), f"REFERENCE  ({Path(ref_path).name})", fill="black", font=font)

    tmp = Path(tempfile.mkdtemp(prefix="cad_compare_"))
    try:
        for i, angle in enumerate(angles):
            cand_png = tmp / f"cand_{angle}.png"
            ref_png = tmp / f"ref_{angle}.png"
            cad_view.render_single(cand_blocks, angle, str(cand_png),
                                   width=tile_w, height=tile_h)
            cad_view.render_single(ref_blocks, angle, str(ref_png),
                                   width=tile_w, height=tile_h)
            sheet.paste(Image.open(cand_png), (0, header_h + i * tile_h))
            sheet.paste(Image.open(ref_png), (tile_w, header_h + i * tile_h))
            # angle label
            draw.text((10, header_h + i * tile_h + 6), angle.upper(),
                      fill="white", font=font_small)
    finally:
        for p in tmp.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass

    sheet.save(out_path)
    return out_path


def make_report(cand_path, ref_path, cand_stats, ref_stats) -> str:
    lines = []
    lines.append(f"# Compare: `{Path(cand_path).name}` vs `{Path(ref_path).name}`")
    lines.append("")
    lines.append("## Numeric deltas")
    lines.append(_delta_line("Bounding box size", cand_stats["size_mm"], ref_stats["size_mm"]))
    lines.append(_delta_line("Surface area",
                              cand_stats["surface_area_mm2"], ref_stats["surface_area_mm2"], unit=" mm²"))
    lines.append(_delta_line("Volume",
                              cand_stats["volume_mm3"], ref_stats["volume_mm3"], unit=" mm³"))
    lines.append(f"- Aspect: candidate={cand_stats['aspect']}  reference={ref_stats['aspect']}")
    lines.append(f"- Long axis: candidate=**{cand_stats['long_axis']}**  reference=**{ref_stats['long_axis']}**")
    lines.append(f"- Faces: candidate={cand_stats['face_count']:,}  reference={ref_stats['face_count']:,}")
    lines.append(f"- Watertight: candidate={cand_stats['is_watertight']}  reference={ref_stats['is_watertight']}")
    lines.append("")
    lines.append("## Reading the deltas")
    lines.append("- A bounding-box dimension delta of >5 mm in any axis means your proportions need adjusting.")
    lines.append("- A volume delta of >25% means you're either missing major features or the wall thicknesses differ.")
    lines.append("- If the long axis differs, your part is rotated relative to the reference.")
    lines.append("")
    lines.append("## Raw stats")
    lines.append("```json")
    lines.append(json.dumps({"candidate": cand_stats, "reference": ref_stats}, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate", help="STEP/STL produced by your CAD generator")
    ap.add_argument("reference", help="STEP/STL of the target reference")
    ap.add_argument("--out-sheet", default=None, help="Side-by-side PNG path")
    ap.add_argument("--out-report", default=None, help="Markdown report path")
    ap.add_argument("--angles", default=",".join(DEFAULT_ANGLES),
                    help=f"Comma-separated angles (default: {','.join(DEFAULT_ANGLES)})")
    args = ap.parse_args(argv)

    cand = Path(args.candidate); ref = Path(args.reference)
    if not cand.exists() or not ref.exists():
        print("ERROR: input file not found", file=sys.stderr)
        return 2

    print(f"loading candidate: {cand}")
    cand_mesh = _as_mesh(cand)
    print(f"loading reference: {ref}")
    ref_mesh = _as_mesh(ref)

    cand_stats = _stats(cand_mesh)
    ref_stats = _stats(ref_mesh)

    if args.out_sheet:
        angles = [a.strip() for a in args.angles.split(",") if a.strip()]
        render_pair_sheet(cand, ref, args.out_sheet, angles)
        print(f"saved sheet: {args.out_sheet}")

    report = make_report(cand, ref, cand_stats, ref_stats)
    if args.out_report:
        Path(args.out_report).write_text(report, encoding="utf-8")
        print(f"saved report: {args.out_report}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
