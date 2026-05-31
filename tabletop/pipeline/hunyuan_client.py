"""hunyuan_client - run Hunyuan3D 2.1 locally for textured PREVIEW only.

Hunyuan output is NOT print-ready by default — use Sparc3D for printing.
This client exists to produce a painted preview render that helps you
decide if the design works visually before committing to print.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
HY_DIR = TABLETOP_ROOT / "hunyuan3d"
HY_WEIGHTS = TABLETOP_ROOT / "models" / "hunyuan3d"


def generate(image_path: str, out_path: str | None = None,
             mesh_only: bool = False, low_vram: bool = False,
             octree_resolution: int = 512, guidance_scale: float = 7.5,
             steps: int = 50, seed: int | None = None,
             image_left: str | None = None,
             image_back: str | None = None,
             image_right: str | None = None) -> str:
    """Run Hunyuan3D pipeline.

    Provide image_left/back/right to activate multi-view mode (requires
    the hunyuan3d-dit-v2-mv weights — auto-downloaded if absent).
    octree_resolution=512 (up from default 384) for sharper geometry.
    guidance_scale=7.5 (up from 5.0) for tighter adherence to input image.
    """
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"input image not found: {img}")

    if out_path is None:
        out_path = str(TABLETOP_ROOT / "outputs" / "previews" / f"{img.stem}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    py = str(HY_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python")
    entry = HY_DIR / "hy3d_infer.py"

    cmd = [
        py, str(entry),
        "--image", str(img),
        "--output", str(out_path),
        "--weights", str(HY_WEIGHTS),
        "--octree-resolution", str(octree_resolution),
        "--guidance-scale",    str(guidance_scale),
        "--steps",             str(steps),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if image_left:
        cmd += ["--image-left",  str(image_left)]
    if image_back:
        cmd += ["--image-back",  str(image_back)]
    if image_right:
        cmd += ["--image-right", str(image_right)]
    if mesh_only:
        cmd.append("--mesh-only")
    if low_vram:
        cmd.append("--low-vram")

    print(f"  [hunyuan3d] {' '.join(cmd[:2])} ...")
    proc = subprocess.run(cmd, cwd=str(HY_DIR), capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Hunyuan3D failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-1500:]}\n"
            "Set HUNYUAN_CLI env var to override the command shape if your\n"
            "upstream version differs."
        )
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python hunyuan_client.py <image.png> [out.png]", file=sys.stderr)
        sys.exit(2)
    out = generate(image_path=sys.argv[1], out_path=sys.argv[2] if len(sys.argv) > 2 else None)
    print(out)
