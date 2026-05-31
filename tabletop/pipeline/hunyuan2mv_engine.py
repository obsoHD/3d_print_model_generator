"""hunyuan2mv_engine - multi-view Hunyuan3D shape engine (front/left/back).

Accepts either three view images or a turntable GIF (frames auto-extracted to
front/left/back). Fuses them into one mesh via Hunyuan3D-2mv. Same downstream
sharp finish as the single-view path.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE          = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
# One shared interpreter on the Linux workstation (override with GEN3D_PY).
VENV_PY       = os.environ.get("GEN3D_PY") or sys.executable
# mv model targets the 2.0 hy3dgen codebase (needs MVImageProcessorV2)
LIB_DIR       = TABLETOP_ROOT / "Hunyuan3D-2"


def _ready() -> bool:
    return (LIB_DIR / "hy3dgen" / "shapegen" / "pipelines.py").exists()


def generate(out_path: str,
             front: str = None, left: str = None, back: str = None,
             gif: str = None, reverse: bool = False,
             seed: int = 42, steps: int = 50,
             octree_resolution: int = 768,
             timeout_s: int = 1800) -> str:
    """Run Hunyuan3D-2mv. Provide either (front,left,back) or a turntable gif."""
    if not _ready():
        raise RuntimeError(f"Hunyuan3D-2 (mv) lib missing at {LIB_DIR}")
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    # If a GIF/video was given, extract the three views first.
    if gif:
        view_dir = out.parent / (out.stem + "_views")
        cmd = [str(VENV_PY), str(HERE / "gif_to_views.py"),
               "--input", gif, "--out-dir", str(view_dir)]
        if reverse:
            cmd.append("--reverse")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"gif_to_views failed:\n{proc.stderr[-1000:]}")
        front = str(view_dir / "front.png")
        left  = str(view_dir / "left.png")
        back  = str(view_dir / "back.png")

    for name, p in (("front", front), ("left", left), ("back", back)):
        if not p or not Path(p).exists():
            raise FileNotFoundError(f"{name} view missing: {p}")

    cmd = [str(VENV_PY), str(HERE / "_hunyuan2mv_infer.py"),
           "--front", front, "--left", left, "--back", back,
           "--output", str(out), "--lib", str(LIB_DIR),
           "--steps", str(steps), "--seed", str(seed),
           "--octree-resolution", str(octree_resolution)]
    print(f"  [hunyuan2mv] multi-view inference (front/left/back, "
          f"{steps} steps, octree={octree_resolution})...", flush=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(LIB_DIR), timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Hunyuan3D-2mv failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-2000:]}")
    print(f"  [hunyuan2mv] mesh saved -> {out.name}", flush=True)
    return str(out)
