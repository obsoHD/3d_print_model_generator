"""mv_generator - turn 1 concept image into 4 consistent view images.

Wraps Zero123++ v1.2 (sudo-ai/zero123plus-v1.2). The model generates a fixed
2-col × 3-row grid of 6 novel views (output 640×960, each cell 320×320).
We split that grid and map the views to HY3D's front / left / back / right.

Zero123++ v1.2 grid layout (azimuth, elevation), reading left-to-right
top-to-bottom on the 2×3 output image:
    idx 0 (row0,col0):  30°,  20°   — front-right (slight up)
    idx 1 (row0,col1):  90°, -10°   — right side (slight down)
    idx 2 (row1,col0): 150°,  20°   — back-right
    idx 3 (row1,col1): 210°, -10°   — back-left
    idx 4 (row2,col0): 270°,  20°   — left side (slight up)
    idx 5 (row2,col1): 330°, -10°   — front-left

The original input image is azimuth 0° (the canonical front).

Mapping to HY3D's 4 cardinal views:
    front <- original input (true 0°)
    right <- idx 1 (90°)
    back  <- idx 2 (150°, closer-to-back of the two roughly-180° views)
    left  <- idx 4 (270°)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
HY_DIR        = TABLETOP_ROOT / "hunyuan3d"
MODEL_DIR     = TABLETOP_ROOT / "models" / "zero123plus"
VIEW_ORDER    = ("front", "right", "back", "left")


def generate(image_path: str, out_dir: str, run_id: str,
             steps: int = 50, seed: int | None = None,
             engine: str = "zero123") -> Dict[str, str]:
    """Generate 4 consistent views from one image. Returns a {view: path} dict.

    Runs Zero123++ in a subprocess (uses the hunyuan3d venv which has
    diffusers + torch installed). Splits the 3x2 output grid into 6 views
    and maps the 3 most useful ones into HY3D's cardinal slots, plus the
    original image as the front view.
    """
    img_in  = Path(image_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not img_in.exists():
        raise FileNotFoundError(f"input image not found: {img_in}")

    py = str(HY_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python")
    grid_out = out_dir / f"{run_id}_mv_grid.png"

    if engine == "mvadapter":
        inner   = HERE / "_mvadapter_infer.py"
        model_p = TABLETOP_ROOT / "models" / "mvadapter"
        label   = "MV-Adapter"
    else:
        inner   = HERE / "_zero123_infer.py"
        model_p = MODEL_DIR
        label   = "zero123++"

    cmd = [py, str(inner),
           "--image",  str(img_in),
           "--output", str(grid_out),
           "--model",  str(model_p),
           "--steps",  str(steps)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    print(f"  [{label}] generating 6 views ({steps} steps)...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Zero123++ failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-1500:]}"
        )

    # Split grid into 6 view cells. Layout differs by engine:
    #   zero123 → 2 cols × 3 rows (640×960)
    #   mvadapter → 3 cols × 2 rows (e.g. 2304×1536 for 768 cells)
    from PIL import Image
    grid = Image.open(grid_out).convert("RGBA")
    W, H = grid.size
    if engine == "mvadapter":
        cols, rows = 3, 2
    else:
        cols, rows = 2, 3
    cell_w, cell_h = W // cols, H // rows
    cells = []
    for row in range(rows):
        for col in range(cols):
            box = (col * cell_w, row * cell_h,
                   (col + 1) * cell_w, (row + 1) * cell_h)
            cells.append(grid.crop(box))

    # Map cells -> HY3D cardinal views (depends on engine)
    if engine == "mvadapter":
        # MV-Adapter cameras: azimuth [0, 45, 90, 180, 270, 315], elev 0
        # idx 0 = front, idx 2 = right, idx 3 = back, idx 4 = left
        mapping = {"front": 0, "right": 2, "back": 3, "left": 4}
    else:
        # Zero123++: front from original input; right/back/left from grid
        mapping = {"front": None, "right": 1, "back": 2, "left": 4}

    out_paths: Dict[str, str] = {}
    for view, idx in mapping.items():
        dst = out_dir / f"{run_id}_{view}.png"
        if idx is None:
            # front = the original input (copy so dashboard sees it consistently)
            Image.open(image_path).convert("RGBA").save(dst)
        else:
            cells[idx].save(dst)
        out_paths[view] = str(dst)
        print(f"  [{label}] {view:5s} -> {dst.name}", flush=True)

    # Upscale each non-front view to 1024×1024 via SDXL img2img — Zero123++
    # outputs are 320×320 which is much smaller than HY3D's 1024-res front
    # input. Lifting the side views matches resolution and adds detail.
    if os.environ.get("SKIP_MV_UPSCALE", "0") != "1":
        try:
            import image_refine
            n = image_refine.upscale_views_batch(
                out_paths, target_size=1024, seed=seed)
            print(f"  [{label}] upscaled {n} side views to 1024px", flush=True)
        except Exception as e:
            print(f"  [{label}] mv upscale skipped: {e}", flush=True)

    # Top view is INTENTIONALLY NOT synthesized via SDXL img2img.
    # Without 3D understanding the model produces a top-down scene that
    # doesn't match the subject's actual identity (user feedback: "top
    # and front don't match content-wise"). The top view is instead
    # rendered from the GLB after HY3D builds the mesh — guaranteed to
    # match the real geometry. Dashboard handles that fallback.
    # Override with SYNTH_TOP=1 if you want the old behavior anyway.
    if os.environ.get("SYNTH_TOP", "0") == "1":
        try:
            import image_refine
            top_path = out_dir / f"{run_id}_top.png"
            if image_refine.synth_top_view(
                out_paths["front"], str(top_path),
                prompt=os.environ.get("RUN_PROMPT", "") or None, seed=seed):
                out_paths["top"] = str(top_path)
                print(f"  [{label}] top synth (SYNTH_TOP=1) -> {top_path.name}",
                      flush=True)
        except Exception as e:
            print(f"  [{label}] top synth failed: {e}", flush=True)

    # View Fix pass — aggressive img2img regen on broken side views.
    # Scores each view; views below quality threshold get an extra 0.55-str
    # img2img with the prompt as guidance, replacing obvious artifacts /
    # mismatched geometry. Force-all mode applies to every view.
    fix_mode = os.environ.get("VIEW_FIX", "auto").lower()   # off|auto|force
    if fix_mode != "off":
        try:
            import image_refine
            prompt_hint = os.environ.get("RUN_PROMPT", "")
            n = image_refine.fix_views_batch(
                out_paths, prompt=prompt_hint or None,
                seed=seed, force_all=(fix_mode == "force"))
            if n: print(f"  [{label}] view-fix pass regenerated {n} broken views",
                        flush=True)
        except Exception as e:
            print(f"  [{label}] view-fix skipped: {e}", flush=True)

    return out_paths


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("usage: python mv_generator.py <input.png> <out_dir> <run_id>",
              file=sys.stderr); sys.exit(2)
    paths = generate(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(paths, indent=2))
