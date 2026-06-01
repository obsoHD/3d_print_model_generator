"""trellis_mv_engine - TRELLIS.2 MULTI-VIEW image-to-3D wrapper.

Same 4B O-Voxel engine as trellis_engine, but conditions on several consistent
views (front/left/back, or a turntable GIF/MP4) instead of one. The fused
conditioning gives the model real data for the occluded sides, so the back and
deep folds are reconstructed instead of left as flat-filled gaps. Streams live.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE          = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
VENV_PY       = os.environ.get("GEN3D_PY") or sys.executable
LIB_DIR       = TABLETOP_ROOT / "TRELLIS"


def _ready() -> bool:
    return (LIB_DIR / "trellis2" / "__init__.py").exists()


def generate(out_path: str,
             front: str = None, left: str = None, back: str = None,
             gif: str = None, reverse: bool = False,
             seed: int = 42, timeout_s: int = 2400) -> str:
    """Run TRELLIS.2 multi-view. Provide either (front,left,back) or a turntable
    GIF/MP4 (frames auto-extracted to front/left/back). Returns the GLB path."""
    if not _ready():
        raise RuntimeError(f"TRELLIS.2 not installed at {LIB_DIR}")
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    # Turntable -> 3 views (reuses the same extractor as the Hunyuan mv path).
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

    views = [(n, p) for n, p in (("front", front), ("left", left), ("back", back)) if p]
    if not views:
        raise RuntimeError("TRELLIS.2 multi-view needs front/left/back images or a GIF")
    for name, p in views:
        if not Path(p).exists():
            raise FileNotFoundError(f"{name} view missing: {p}")

    inner = HERE / "_trellis_infer.py"
    cmd = [str(VENV_PY), str(inner),
           "--image", views[0][1],          # front as the nominal --image
           "--output", str(out), "--lib", str(LIB_DIR), "--seed", str(seed)]
    if front: cmd += ["--mv-front", front]
    if left:  cmd += ["--mv-left",  left]
    if back:  cmd += ["--mv-back",  back]
    print(f"  [trellis-mv] multi-view inference ({len(views)} views)...", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_VERBOSITY": "info",
           "ATTN_BACKEND": "sdpa", "SPARSE_ATTN_BACKEND": "xformers",
           "SPCONV_ALGO": os.environ.get("SPCONV_ALGO", "native")}
    import collections
    tail = collections.deque(maxlen=80)
    proc = subprocess.Popen(cmd, cwd=str(LIB_DIR), env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1)
    try:
        for raw in proc.stdout:
            ln = raw.rstrip("\n"); print(f"        {ln}", flush=True); tail.append(ln)
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill(); raise RuntimeError(f"TRELLIS.2 multi-view timed out after {timeout_s}s")
    if proc.returncode != 0:
        raise RuntimeError(
            f"TRELLIS.2 multi-view failed (exit {proc.returncode}).\n" + "\n".join(tail))
    print(f"  [trellis-mv] mesh saved -> {out.name}", flush=True)
    return str(out)
