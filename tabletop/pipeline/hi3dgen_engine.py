"""hi3dgen_engine - Hi3DGen (Stable3DGen) image-to-3D engine wrapper.
Normal-bridging for best geometric fidelity. Single image in, mesh out. Streams."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE          = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
VENV_PY       = os.environ.get("GEN3D_PY") or sys.executable
LIB_DIR       = TABLETOP_ROOT / "Stable3DGen"


def _ready() -> bool:
    return (LIB_DIR / "trellis" / "__init__.py").exists()


def generate(image_path: str, out_path: str,
             seed: int = 42, timeout_s: int = 2400) -> str:
    if not _ready():
        raise RuntimeError(f"Hi3DGen (Stable3DGen) not installed at {LIB_DIR}")
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(image_path)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    inner = HERE / "_hi3dgen_infer.py"
    cmd = [str(VENV_PY), str(inner), "--image", str(img), "--output", str(out),
           "--lib", str(LIB_DIR), "--seed", str(seed)]
    print("  [hi3dgen] inference (normal-bridging geometry)...", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_VERBOSITY": "info",
           "ATTN_BACKEND": os.environ.get("ATTN_BACKEND", "xformers"),
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
        proc.kill(); raise RuntimeError(f"Hi3DGen timed out after {timeout_s}s")
    if proc.returncode != 0:
        raise RuntimeError(f"Hi3DGen failed (exit {proc.returncode}).\n" + "\n".join(tail))
    print(f"  [hi3dgen] mesh saved -> {out.name}", flush=True)
    return str(out)
