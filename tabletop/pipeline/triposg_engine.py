"""triposg_engine - VAST-AI TripoSG single-image-to-3D engine wrapper.

TripoSG is an SDF-based image-to-3D model. Because the network outputs a
signed-distance field, mesh extraction is *guaranteed* watertight and
manifold — no post-processing rescue needed (unlike Pixal3D / HY3D which
extract surfaces and can leave open boundaries / nested shells).

Uses the pixal3d_venv subprocess (torch 2.8.0+cu129, diffusers, etc).
Flash-decoder is disabled (needs `diso` which requires MSVC on Windows);
hierarchical extractor uses scikit-image marching cubes instead. Same
guarantee of watertightness, slightly slower (~20s vs ~10s).

Usage:
    from triposg_engine import generate
    generate(image_path="concept.png", out_path="mesh.glb")
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
LIB_DIR       = TABLETOP_ROOT / "TripoSG"
WEIGHTS_DIR   = TABLETOP_ROOT / "models" / "triposg"


def _ready() -> bool:
    return (LIB_DIR / "triposg" / "inference_utils.py").exists()


def generate(image_path: str, out_path: str,
             seed: int = 42,
             steps: int = 50,
             guidance: float = 7.0,
             faces: int = 300_000,
             timeout_s: int = 900) -> str:
    """Run TripoSG. Returns the output GLB path. Output is watertight."""
    if not _ready():
        raise RuntimeError(
            f"TripoSG not installed. Need:\n"
            f"  library: {LIB_DIR}  (exists={LIB_DIR.exists()})\n"
        )
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(image_path)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    inner = HERE / "_triposg_infer.py"
    cmd = [str(VENV_PY), str(inner),
           "--image",    str(img),
           "--output",   str(out),
           "--lib",      str(LIB_DIR),
           "--weights",  str(WEIGHTS_DIR),
           "--steps",    str(steps),
           "--guidance", str(guidance),
           "--seed",     str(seed),
           "--faces",    str(faces)]
    print(f"  [triposg] inference ({steps} steps, guidance={guidance}, "
          f"watertight by construction)...", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_VERBOSITY": "info"}
    import collections
    tail = collections.deque(maxlen=80)
    proc = subprocess.Popen(cmd, cwd=str(LIB_DIR), env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1)
    try:
        for raw in proc.stdout:
            ln = raw.rstrip("\n")
            print(f"        {ln}", flush=True)
            tail.append(ln)
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"TripoSG timed out after {timeout_s}s")
    if proc.returncode != 0:
        raise RuntimeError(
            f"TripoSG failed (exit {proc.returncode}).\n" + "\n".join(tail))
    print(f"  [triposg] mesh saved -> {out.name}", flush=True)
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
