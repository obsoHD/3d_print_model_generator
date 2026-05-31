"""craftsman_engine - CraftsMan3D image-to-3D engine wrapper.

CraftsMan3D = coarse 3D diffusion + a normal-based geometry refiner. Best
surface detail of the local engines (the "iterate to match" tool). Single image
in, detailed mesh out. Same downstream gauntlet + finish as the other engines.
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
LIB_DIR       = TABLETOP_ROOT / "CraftsMan3D"


def _ready() -> bool:
    return (LIB_DIR / "craftsman" / "__init__.py").exists()


def generate(image_path: str, out_path: str,
             seed: int = 42, timeout_s: int = 1800) -> str:
    """Run CraftsMan3D. Returns the output GLB path."""
    if not _ready():
        raise RuntimeError(
            f"CraftsMan3D not installed. Need the repo at {LIB_DIR} "
            f"(craftsman/__init__.py).")
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(image_path)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)

    inner = HERE / "_craftsman_infer.py"
    cmd = [str(VENV_PY), str(inner),
           "--image",  str(img),
           "--output", str(out),
           "--lib",    str(LIB_DIR),
           "--seed",   str(seed)]
    print("  [craftsman] inference (coarse 3D + normal-based refiner)...",
          flush=True)
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
        raise RuntimeError(f"CraftsMan3D timed out after {timeout_s}s")
    if proc.returncode != 0:
        raise RuntimeError(
            f"CraftsMan3D failed (exit {proc.returncode}).\n" + "\n".join(tail))
    print(f"  [craftsman] mesh saved -> {out.name}", flush=True)
    return str(out)
