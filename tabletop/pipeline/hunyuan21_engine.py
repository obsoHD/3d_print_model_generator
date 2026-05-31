"""hunyuan21_engine - Hunyuan3D 2.1 single-image-to-3D (shape only).

Tencent's flagship open-source 3D shape generator (mid-2025). Most
production-mature in the SDF/MC family — produces clean topology with
~10 splits vs TripoSG's 300+. Best baseline mesh quality of any free
engine for printable terrain/props on RTX 5080 / 16 GB.

Uses CPU-offload-aware subprocess in pixal3d_venv (torch 2.8 + cu129).
FlashVDM is DISABLED — its hooks corrupt CUDA state when combined with
diffusers' model_cpu_offload, causing "unknown error" in volume_decoder.
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
LIB_DIR       = TABLETOP_ROOT / "Hunyuan3D-2.1"


def _ready() -> bool:
    return (LIB_DIR / "hy3dshape" / "hy3dshape" / "pipelines.py").exists()


def generate(image_path: str, out_path: str,
             seed: int = 42,
             steps: int = 75,
             octree_resolution: int = 768,
             timeout_s: int = 1800) -> str:
    """Run Hunyuan3D 2.1 shape pipeline. Returns the output GLB path.

    octree_resolution 384 = max geometry that fits 16 GB (~813K vs 356K faces
    at 256, ~2.3x). steps 75 = more diffusion refinement. Both verified to fit
    with CPU offload + expandable_segments + FlashVDM off (decode ~36s).
    Drop to 256 if VRAM is occupied and 384 OOMs.
    """
    if not _ready():
        raise RuntimeError(
            f"Hunyuan3D 2.1 not installed. Need:\n"
            f"  library: {LIB_DIR}  (exists={LIB_DIR.exists()})\n"
        )
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(image_path)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    inner = HERE / "_hunyuan21_infer.py"
    cmd = [str(VENV_PY), str(inner),
           "--image",  str(img),
           "--output", str(out),
           "--lib",    str(LIB_DIR),
           "--steps",  str(steps),
           "--octree-resolution", str(octree_resolution),
           "--seed",   str(seed)]
    print(f"  [hunyuan21] inference ({steps} steps, octree={octree_resolution}, "
          f"CPU offload + 192-grid)...", flush=True)
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(LIB_DIR), timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Hunyuan3D 2.1 failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    print(f"  [hunyuan21] mesh saved -> {out.name}", flush=True)
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
