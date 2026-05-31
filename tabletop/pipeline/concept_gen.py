"""concept_gen - unified concept image generator.

Auto-selects the best installed model for a given `kind`:

    terrain / prop   -> FLUX.1-schnell  (photorealistic architecture)
                        OR RealVisXL v5.0  (SDXL drop-in fallback)
    mini / scatter   -> DreamShaper XL Turbo  (stylized fantasy)
                        OR RealVisXL v5.0  (fallback)

Each model is invoked in a subprocess using the hunyuan3d venv (which has
diffusers + torch installed). Models load on demand, then unload after the
single image is generated, so VRAM is freed before HY3D's stage runs.

Falls back gracefully: if no upgraded model is present, raises so the
orchestrator can fall back to comfy_client/SDXL workflow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE          = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
HY_DIR        = TABLETOP_ROOT / "hunyuan3d"
MODELS_DIR    = TABLETOP_ROOT / "models"

# Model registry: name -> (subdir, kind, format)
#   format: 'diffusers'  -> full diffusers folder (model_index.json + subfolders)
#           'single'     -> single .safetensors checkpoint
MODELS = {
    "z_image_turbo": (MODELS_DIR / "z_image_turbo",        "diffusers"),
    "flux_schnell":  (MODELS_DIR / "flux_schnell",         "diffusers"),
    "realvis_v5":    (MODELS_DIR / "realvisxl",            "diffusers"),
    "dreamshaper":   (MODELS_DIR / "dreamshaper_xl_turbo", "diffusers"),
}


def _model_ready(name: str) -> bool:
    """Check whether the model weights are present on disk."""
    path, fmt = MODELS[name]
    if not path.exists():
        return False
    if fmt == "diffusers":
        return (path / "model_index.json").exists()
    if fmt == "single":
        return any(path.glob("*.safetensors"))
    return False


def pick_model(kind: str) -> tuple[str, Path, str] | None:
    """Return (name, path, format) for the best-fit installed model for `kind`.

    Priority is tuned for 16 GB VRAM: FLUX-schnell full fp16 is 22 GB transformer
    + 8.9 GB T5 — even with CPU offload, each step takes >60s and risks OOM.
    RealVisXL v5.0 is the photorealism leader in the SDXL family (~7 GB,
    runs fully in VRAM, ~30s per image). DreamShaper XL Turbo is the same
    size but tuned for stylized fantasy minis.

    FLUX is only picked when explicitly requested via env FORCE_FLUX=1 or
    when no SDXL drop-in is installed.
    """
    force_flux = bool(int(os.environ.get("FORCE_FLUX", "0")))
    force_zimg = os.environ.get("CONCEPT_ENGINE", "").lower() == "z_image"
    if kind in ("mini", "scatter"):
        order = ["z_image_turbo", "dreamshaper", "realvis_v5"]
        if force_flux: order.insert(0, "flux_schnell")
        else:          order.append("flux_schnell")
    else:  # terrain, prop, default
        order = ["z_image_turbo", "realvis_v5", "dreamshaper"]
        if force_flux: order.insert(0, "flux_schnell")
        else:          order.append("flux_schnell")
    if force_zimg and "z_image_turbo" in order:
        order.remove("z_image_turbo"); order.insert(0, "z_image_turbo")
    for name in order:
        if _model_ready(name):
            path, fmt = MODELS[name]
            return name, path, fmt
    return None


# Suffixes engineered for HY3D-friendly concepts: single subject, white
# background (clean alpha matte), three-quarter view for depth cues, even
# diffuse lighting to avoid hard shadows confusing geometry reconstruction.
_TERRAIN_SUFFIX = (", professional architectural photography, isometric "
                   "three-quarter view, single isolated subject centered in frame, "
                   "plain seamless white background, soft even daylight, "
                   "museum studio lighting, ultra detailed materials, sharp focus, "
                   "photorealistic, 8k, no people, no clutter")
_MINI_SUFFIX    = (", studio product photography, painted resin miniature figure, "
                   "full body, three-quarter front view, centered, "
                   "plain seamless white background, soft top-down lighting, "
                   "sharp detail, photorealistic, 8k, single subject")


def _compose_prompt(prompt: str, kind: str) -> str:
    """Append a kind-specific suffix that biases output toward 3D-printable composition."""
    if kind in ("mini", "scatter"):
        return prompt + _MINI_SUFFIX
    return prompt + _TERRAIN_SUFFIX


def _remove_bg(img_path: str) -> None:
    """Strip background via rembg, composite onto pure white.

    SDXL-class models often ignore 'plain white background' prompts and
    generate the subject in a realistic outdoor scene. HY3D / Zero123++
    need a clean silhouette — this gives them one. Overwrites img_path
    with a clean-BG version of the same image.

    Runs as a subprocess in the hunyuan3d venv (which has rembg + torch)
    so we don't depend on the orchestrator's interpreter having rembg.
    """
    py = str(HY_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python")
    code = (
        "import sys; from PIL import Image; from rembg import remove, new_session\n"
        "p = sys.argv[1]\n"
        "img = Image.open(p).convert('RGBA')\n"
        "sess = new_session('u2net')\n"
        "cut = remove(img, session=sess, alpha_matting=True,\n"
        "             alpha_matting_foreground_threshold=240,\n"
        "             alpha_matting_background_threshold=20,\n"
        "             alpha_matting_erode_size=3)\n"
        "white = Image.new('RGBA', cut.size, (255,255,255,255))\n"
        "final = Image.alpha_composite(white, cut)\n"
        "final.convert('RGB').save(p, 'PNG')\n"
        "print('  [rembg] clean alpha matte saved')\n"
    )
    try:
        proc = subprocess.run([py, "-c", code, img_path],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            print(f"  [concept_gen] background removed (rembg/u2net subprocess)", flush=True)
        else:
            print(f"  [concept_gen] rembg subprocess failed (exit {proc.returncode}); "
                  f"stderr: {proc.stderr[-300:]}", flush=True)
    except Exception as e:
        print(f"  [concept_gen] rembg subprocess error ({e}); leaving BG as-is", flush=True)


def generate(prompt: str, out_path: str, kind: str = "terrain",
             seed: int | None = None, width: int = 1280, height: int = 1280,
             steps: int | None = None) -> str:
    """Generate one concept image. Returns the model name actually used."""
    pick = pick_model(kind)
    if pick is None:
        raise RuntimeError(
            "No upgraded concept model installed. "
            "Install one of: flux_schnell, realvis_v5, dreamshaper. "
            "Orchestrator should fall back to comfy_client/SDXL."
        )
    name, model_path, fmt = pick
    full_prompt = _compose_prompt(prompt, kind)
    print(f"  [concept_gen] model={name}  kind={kind}", flush=True)
    print(f"  [concept_gen] prompt: {full_prompt[:90]}...", flush=True)

    py    = str(HY_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python")
    inner = HERE / "_concept_infer.py"

    cmd = [py, str(inner),
           "--model-name", name,
           "--model-path", str(model_path),
           "--format",     fmt,
           "--prompt",     full_prompt,
           "--output",     out_path,
           "--width",      str(width),
           "--height",     str(height)]
    if seed  is not None: cmd += ["--seed",  str(seed)]
    if steps is not None: cmd += ["--steps", str(steps)]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          env={**os.environ})
    if proc.returncode != 0:
        raise RuntimeError(
            f"concept_gen ({name}) failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-1500:]}"
        )
    # Post-step: strip whatever the model painted as background
    _remove_bg(out_path)

    # Refiner pass — light SDXL img2img on the clean concept to sharpen
    # edges and surface micro-detail without changing composition.
    if os.environ.get("SKIP_CONCEPT_REFINE", "0") != "1":
        try:
            import image_refine
            image_refine.refine_concept(out_path, prompt=full_prompt[:200],
                                        strength=0.22, steps=22, seed=seed)
        except Exception as e:
            print(f"  [concept_gen] refine pass skipped: {e}", flush=True)

    return name


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",  required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--kind",    default="terrain")
    ap.add_argument("--seed",    type=int, default=None)
    args = ap.parse_args()
    used = generate(args.prompt, args.output, args.kind, args.seed)
    print(json.dumps({"model": used, "output": args.output}))
