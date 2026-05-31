"""compose_images - combine multiple reference photos into one contact sheet.

Multiple input images of the SAME thing (different angles, lighting, sources)
get tiled into one image so a human (or Claude) can study them together.

Usage:
  python compose_images.py img1.jpg img2.jpg img3.png --out sheet.png
  python compose_images.py refs/*.jpg --out sheet.png --cols 3 --label

Notes:
  * Aspect ratios are preserved; images are letterboxed to a common cell size.
  * Optional --label adds the filename under each tile (useful when several
    photos are from different sources).
  * The TARGET sheet width is capped so the output stays viewable.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _label_font(size: int = 20):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_into(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    ratio = min(w / img.width, h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    return img.resize((new_w, new_h), Image.LANCZOS)


def compose(paths, out_path, cols=None, max_sheet_width=2400, label=False, bg="white"):
    images = []
    for p in paths:
        try:
            images.append((Path(p), Image.open(p)))
        except Exception as e:
            print(f"WARN: could not open {p}: {e}", file=sys.stderr)
    if not images:
        raise RuntimeError("No usable images")

    n = len(images)
    if cols is None:
        cols = min(n, 3) if n > 1 else 1
    rows = math.ceil(n / cols)

    # cell size based on average aspect ratio, capped to keep sheet width sane
    avg_w = sum(im.width for _, im in images) / n
    avg_h = sum(im.height for _, im in images) / n
    cell_w = int(min(max_sheet_width / cols, avg_w))
    cell_h = int(cell_w * (avg_h / avg_w))

    label_h = 28 if label else 0
    tile_h = cell_h + label_h

    sheet_w = cols * cell_w
    sheet_h = rows * tile_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), bg)
    draw = ImageDraw.Draw(sheet)
    font = _label_font(20) if label else None

    for i, (p, im) in enumerate(images):
        r, c = divmod(i, cols)
        fitted = _fit_into(im, cell_w, cell_h)
        x = c * cell_w + (cell_w - fitted.width) // 2
        y = r * tile_h + (cell_h - fitted.height) // 2
        sheet.paste(fitted, (x, y))
        if label:
            draw.text((c * cell_w + 6, r * tile_h + cell_h + 4), p.name,
                      fill="black", font=font)

    sheet.save(out_path)
    return out_path, (sheet_w, sheet_h), n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", help="image files")
    ap.add_argument("--out", required=True, help="output sheet PNG")
    ap.add_argument("--cols", type=int, default=None, help="grid columns")
    ap.add_argument("--max-width", type=int, default=2400, help="max output width")
    ap.add_argument("--label", action="store_true", help="annotate each tile with filename")
    ap.add_argument("--bg", default="white")
    args = ap.parse_args(argv)

    # expand globs that the shell didn't (Windows PowerShell often doesn't)
    expanded = []
    for p in args.images:
        path = Path(p)
        if any(ch in p for ch in "*?[]"):
            parent = path.parent if path.parent != Path("") else Path(".")
            for match in sorted(parent.glob(path.name)):
                expanded.append(str(match))
        else:
            expanded.append(p)

    out_path, size, n = compose(expanded, args.out, cols=args.cols,
                                 max_sheet_width=args.max_width,
                                 label=args.label, bg=args.bg)
    print(f"saved: {out_path}  ({size[0]}x{size[1]}, {n} tiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
