"""pixal3d_engine - alternative 3D engine using Pixal3D (TencentARC).

Pixal3D is built on TRELLIS.2 with pixel-aligned projection for near-
reconstruction-level detail. Single image → 3D mesh with PBR textures,
no multi-view stage required. Replaces the entire Zero123++ + HY3D mv
chain for runs where you have a clean front concept image.

Runs in a dedicated venv (tabletop/pixal3d/venv) with Python 3.12,
PyTorch 2.10+cu130, and the pre-built CUDA wheels for
flex_gemm_ap / cumesh_vb / o_voxel_vb_ap / drtk.

Usage:
    from pixal3d_engine import generate
    generate(image_path="concept.png", out_path="mesh.glb")
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE          = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
# Dedicated venv (py3.12 + torch 2.8.0+cu129 + 4 CUDA wheels + nvdiffrast).
# Lives at pixal3d_venv to avoid Windows case-collision with the Pixal3D/ repo.
VENV_PY       = TABLETOP_ROOT / "pixal3d_venv" / "Scripts" / "python.exe"
MODEL_DIR     = TABLETOP_ROOT / "models" / "pixal3d"
# Upstream Pixal3D (TencentARC/Pixal3D) — has sdpa attention branch.
PIXAL_LIB     = TABLETOP_ROOT / "Pixal3D"
PIXAL_INFER   = PIXAL_LIB / "inference.py"


def _ready() -> bool:
    return (VENV_PY.exists()
            and PIXAL_INFER.exists()
            and (MODEL_DIR / "pipeline.json").exists()
            and (PIXAL_LIB / "pixal3d" / "__init__.py").exists())


def generate(image_path: str, out_path: str,
             seed: int = 42,
             resolution: int = 1024,
             low_vram: bool = True,   # standard mode would spill 17 GB models past 16 GB VRAM
                                      # and page-thrash — low_vram is actually 6-10x faster here
             manual_fov: float = 0.2,   # ~11.5° — skips MoGe (not installed)
             timeout_s: int = 1800) -> str:
    """Run Pixal3D pipeline using upstream inference.py. Returns GLB path.

    Runs in the pixal3d_venv subprocess with ATTN_BACKEND=sdpa so the
    sparse-attention modules use PyTorch's scaled_dot_product_attention
    instead of flash_attn (which has no Blackwell-compatible build on Windows).

    `manual_fov` (radians) bypasses MoGe-2 camera estimation — MoGe is not
    installed in pixal3d_venv (its git dep doesn't ship a Windows wheel and
    we'd burn another 10 GB of weights). 0.2 rad ≈ 11.5° is a reasonable
    default for centered concept-art front views.
    """
    if not _ready():
        raise RuntimeError(
            f"Pixal3D not installed. Need:\n"
            f"  venv:    {VENV_PY}  (exists={VENV_PY.exists()})\n"
            f"  weights: {MODEL_DIR}  (pipeline.json={(MODEL_DIR / 'pipeline.json').exists()})\n"
            f"  library: {PIXAL_LIB}  (exists={PIXAL_LIB.exists()})\n"
        )
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(image_path)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(VENV_PY), str(PIXAL_INFER),
           "--image",      str(img),
           "--output",     str(out),
           "--model_path", str(MODEL_DIR),
           "--seed",       str(seed),
           "--resolution", str(resolution),
           "--fov",        str(manual_fov)]
    if low_vram:
        cmd.append("--low_vram")
    print(f"  [pixal3d] inference ({resolution}px"
          f"{', low-VRAM' if low_vram else ''}, fov={manual_fov} rad)...",
          flush=True)
    env = {**os.environ, "ATTN_BACKEND": "sdpa"}   # Blackwell-compatible (no flash_attn)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(PIXAL_LIB),     # so it can import its own pixal3d/
                          timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Pixal3D failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    print(f"  [pixal3d] mesh saved -> {out.name}", flush=True)
    return str(out)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()
    p = generate(args.image, args.output, args.seed)
    print(json.dumps({"ok": True, "output": p}))
