"""image_refine - SDXL img2img refinement passes for concept + multi-view.

Two operations, both using the same RealVisXL v5.0 we already have:

  refine_concept(img_path, prompt)
      Light img2img pass (strength ~0.25) on the cleaned concept image.
      Adds sharper edges and finer surface detail without changing the
      overall composition. Same resolution in/out.

  upscale_view(img_path, prompt, target_size)
      Bicubic upscale to target_size then SDXL img2img at moderate strength
      (~0.30) to re-synthesize detail at the higher resolution. Used on the
      320×320 Zero123++ side views to bring them up to ~768/1024 so HY3D's
      multi-view path gets matched-resolution inputs.

Both shell out to _image_refine_infer.py in the hunyuan3d venv. Skipping
gracefully (no-op) if the RealVisXL model isn't present.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

HERE          = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
HY_DIR        = TABLETOP_ROOT / "hunyuan3d"
REALVIS_DIR   = TABLETOP_ROOT / "models" / "realvisxl"


def _venv_py() -> str:
    return str(HY_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python")


def _can_refine() -> bool:
    return REALVIS_DIR.exists() and (REALVIS_DIR / "model_index.json").exists()


def refine_concept(img_path: str, prompt: str | None = None,
                   strength: float = 0.25, steps: int = 25,
                   guidance: float = 5.5, seed: int | None = None) -> bool:
    """Light SDXL img2img on the concept to sharpen detail. In-place."""
    if not _can_refine():
        print("  [image_refine] RealVisXL not installed; skipping concept refine", flush=True)
        return False
    inner = HERE / "_image_refine_infer.py"
    cmd = [_venv_py(), str(inner),
           "--mode",    "refine",
           "--input",   img_path,
           "--output",  img_path,
           "--model",   str(REALVIS_DIR),
           "--strength", str(strength),
           "--steps",    str(steps),
           "--guidance", str(guidance)]
    if prompt:   cmd += ["--prompt", prompt]
    if seed is not None: cmd += ["--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"  [image_refine] concept refine failed (exit {proc.returncode}); "
              f"stderr: {proc.stderr[-300:]}", flush=True)
        return False
    print("  [image_refine] concept refined (img2img strength=%.2f)" % strength, flush=True)
    return True


def upscale_view(img_path: str, prompt: str | None = None,
                 target_size: int = 1024, strength: float = 0.30,
                 steps: int = 22, guidance: float = 5.0,
                 seed: int | None = None) -> bool:
    """Upscale a multi-view image to target_size with img2img detail recovery."""
    if not _can_refine():
        return False
    inner = HERE / "_image_refine_infer.py"
    cmd = [_venv_py(), str(inner),
           "--mode",    "upscale",
           "--input",   img_path,
           "--output",  img_path,
           "--model",   str(REALVIS_DIR),
           "--target-size", str(target_size),
           "--strength", str(strength),
           "--steps",    str(steps),
           "--guidance", str(guidance)]
    if prompt:   cmd += ["--prompt", prompt]
    if seed is not None: cmd += ["--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"  [image_refine] view upscale failed for {Path(img_path).name} "
              f"(exit {proc.returncode})", flush=True)
        return False
    return True


def upscale_views_batch(view_paths: dict, prompt: str | None = None,
                        target_size: int = 1024, seed: int | None = None) -> int:
    """Apply upscale to all non-front views. Returns count of successful upscales."""
    ok = 0
    for view, path in view_paths.items():
        if view == "front" or not path:
            continue
        if upscale_view(path, prompt=prompt, target_size=target_size, seed=seed):
            ok += 1
            print(f"  [image_refine] upscaled {view} -> {target_size}px", flush=True)
    return ok


KONTEXT_DIR = TABLETOP_ROOT / "models" / "flux_kontext"


def _can_kontext() -> bool:
    return KONTEXT_DIR.exists() and (KONTEXT_DIR / "model_index.json").exists()


def fix_view(img_path: str, prompt: str | None = None,
             strength: float = 0.88, steps: int = 40,
             guidance: float = 7.0, seed: int | None = None) -> bool:
    """Regenerate a broken side view, preserving subject identity.

    Prefers FLUX.1-Kontext-dev when available (designed for identity-preserving
    edits — far better than img2img). Falls back to SDXL Refiner img2img at
    high strength when Kontext isn't installed.
    """
    if _can_kontext():
        # Use Kontext — purpose-built for this kind of edit
        inner = HERE / "_kontext_infer.py"
        edit_prompt = (
            (prompt + ", " if prompt else "")
            + "high quality clear side view of the same object, "
              "preserved subject identity, sharp detail, photorealistic, "
              "plain white background"
        )
        cmd = [_venv_py(), str(inner),
               "--input",   img_path,
               "--output",  img_path,
               "--model",   str(KONTEXT_DIR),
               "--prompt",  edit_prompt[:480],
               "--steps",   "24",
               "--guidance", "3.5"]
        if seed is not None: cmd += ["--seed", str(seed)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0:
            return True
        print(f"  [image_refine] Kontext fix failed for {Path(img_path).name} "
              f"(exit {proc.returncode}); falling back to SDXL Refiner",
              flush=True)

    # Fallback: SDXL Refiner img2img at high strength
    if not _can_refine():
        return False
    inner = HERE / "_image_refine_infer.py"
    cmd = [_venv_py(), str(inner),
           "--mode",    "refine",
           "--input",   img_path,
           "--output",  img_path,
           "--model",   str(REALVIS_DIR),
           "--strength", str(strength),
           "--steps",    str(steps),
           "--guidance", str(guidance)]
    if prompt:   cmd += ["--prompt", prompt]
    if seed is not None: cmd += ["--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"  [image_refine] fix_view failed for {Path(img_path).name}",
              flush=True)
        return False
    return True


def _view_quality_score(img_path: str) -> float:
    """Cheap heuristic for view quality — high variance + good edge content.

    Returns 0..1 score. Below 0.35 typically indicates a broken view
    (mostly empty / uniform / no recognizable subject).
    """
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(img_path).convert("L")   # grayscale
        arr = np.asarray(im, dtype=np.float32) / 255.0
        # Variance: how spread out are pixel values
        var = float(np.var(arr))
        # Edge content via simple gradient magnitude
        gx = np.diff(arr, axis=1).astype(np.float32)
        gy = np.diff(arr, axis=0).astype(np.float32)
        edge = float(np.mean(np.abs(gx))) + float(np.mean(np.abs(gy)))
        # Composite score — clamped to 0..1
        score = min(1.0, var * 6 + edge * 3)
        return score
    except Exception:
        return 1.0   # if scoring fails, assume OK


def synth_top_view(front_path: str, out_path: str,
                   prompt: str | None = None,
                   strength: float = 0.75, steps: int = 30,
                   guidance: float = 7.0, seed: int | None = None) -> bool:
    """Generate a top-down view from the front view using SDXL img2img.

    Zero123++ and most multi-view models don't include a true top view.
    This synthesizes one by doing a high-strength img2img on the front
    image with a strong directional prompt ("aerial top-down view from
    directly above"). Not pixel-perfect but gives HY3D / the dashboard a
    top reference that's consistent with the subject.
    """
    if not _can_refine():
        return False
    inner = HERE / "_image_refine_infer.py"
    full_prompt = (
        "aerial top-down view from directly above, looking straight down, "
        "bird's eye view, plain white background, centered, ultra detailed, "
        "8k, photorealistic"
    )
    if prompt:
        full_prompt = f"{prompt}, {full_prompt}"
    cmd = [_venv_py(), str(inner),
           "--mode",    "refine",
           "--input",   front_path,
           "--output",  out_path,
           "--model",   str(REALVIS_DIR),
           "--strength", str(strength),
           "--steps",    str(steps),
           "--guidance", str(guidance),
           "--prompt",   full_prompt]
    if seed is not None: cmd += ["--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"  [image_refine] top-view synthesis failed", flush=True)
        return False
    print(f"  [image_refine] synthesized top view from front (str {strength})",
          flush=True)
    return True


def fix_views_batch(view_paths: dict, prompt: str | None = None,
                    seed: int | None = None,
                    quality_threshold: float = 0.35,
                    force_all: bool = False) -> int:
    """Auto-detect broken side views and regenerate them.

    Scores each view; views below `quality_threshold` get the fix pass
    (or all of them if force_all=True). Returns count of fixed views.
    """
    fixed = 0
    for view, path in view_paths.items():
        if view == "front" or not path:
            continue
        score = _view_quality_score(path)
        broken = score < quality_threshold
        if force_all or broken:
            tag = "force" if force_all else f"broken (score {score:.2f})"
            print(f"  [image_refine] fixing {view} view [{tag}]", flush=True)
            if fix_view(path, prompt=prompt, seed=seed):
                fixed += 1
        else:
            print(f"  [image_refine] {view} ok (score {score:.2f})", flush=True)
    return fixed


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", choices=("refine", "upscale"), default="refine")
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()
    if args.mode == "refine":
        ok = refine_concept(args.input, prompt=args.prompt)
    else:
        ok = upscale_view(args.input, prompt=args.prompt)
    print(json.dumps({"ok": ok, "input": args.input, "mode": args.mode}))
