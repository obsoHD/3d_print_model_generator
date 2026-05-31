"""_mvadapter_infer - CLI subprocess that runs MV-Adapter (i2mv SDXL).

Better consistency than Zero123++ on complex multi-element scenes.
Outputs 6 views at azimuth [0, 45, 90, 180, 270, 315] elevation 0.

Output: single PNG with 6 views stitched as 3x2 grid (3 cols × 2 rows).
mv_generator.py then splits and maps:
    idx 0 (0°)   = FRONT (the canonical 0° view, true front)
    idx 1 (45°)  = front-right
    idx 2 (90°)  = RIGHT
    idx 3 (180°) = BACK
    idx 4 (270°) = LEFT
    idx 5 (315°) = front-left
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the cloned MV-Adapter source importable without pip install
TABLETOP_ROOT = Path(__file__).resolve().parent.parent
MVA_DIR = TABLETOP_ROOT / "MV-Adapter"
if MVA_DIR.exists():
    sys.path.insert(0, str(MVA_DIR))

# MV-Adapter's __init__ chain pulls in modules (nvdiffrast, triton) that
# don't have Windows wheels. We only need camera math and the pipeline —
# no rasterization, no texturing — so stub the missing modules.
import types as _types
for _name in ("nvdiffrast", "nvdiffrast.torch", "triton", "triton.language"):
    if _name not in sys.modules:
        sys.modules[_name] = _types.ModuleType(_name)
# triton stub needs a callable 'jit' decorator and a 'language' submodule
_jit_noop = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
sys.modules["triton"].jit = _jit_noop
sys.modules["triton"].language = sys.modules["triton.language"]
# Common triton.language exports used as type hints — make them no-ops
for _attr in ("constexpr", "tensor", "int32", "float32", "float16"):
    setattr(sys.modules["triton.language"], _attr, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--output", required=True, help="output grid PNG (3x2 layout)")
    ap.add_argument("--model",  required=True, help="local MV-Adapter dir")
    ap.add_argument("--steps",  type=int, default=50)
    ap.add_argument("--seed",   type=int, default=None)
    args = ap.parse_args()

    import torch
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL

    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
    from mvadapter.utils.mesh_utils import get_orthogonal_camera
    from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho

    realvis = TABLETOP_ROOT / "models" / "realvisxl"
    base_model = str(realvis) if (realvis / "model_index.json").exists() \
                 else "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"[mvadapter_infer] base={base_model}", flush=True)

    # Build pipeline
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0)

    num_views = 6
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(args.model,
                             weight_name="mvadapter_i2mv_sdxl.safetensors")
    pipe.to(device="cuda", dtype=torch.float16)
    pipe.cond_encoder.to(device="cuda", dtype=torch.float16)
    pipe.enable_vae_slicing()

    # Cameras at 6 azimuths around the object
    azimuth_deg = [0, 45, 90, 180, 270, 315]
    cameras = get_orthogonal_camera(
        elevation_deg=[0]*num_views, distance=[1.8]*num_views,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[a - 90 for a in azimuth_deg],  # internal frame offset
        device="cuda")
    height = width = 768
    plucker = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1]*num_views, width)
    control_images = ((plucker + 1.0) / 2.0).clamp(0, 1)

    # Preprocess input: pad/resize to 90% of canvas, gray fill
    img = Image.open(args.image).convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3] > 0 if arr.shape[2] == 4 else np.ones(arr.shape[:2], bool)
    if alpha.any():
        y, x = np.where(alpha)
        y0, y1 = max(y.min()-1, 0), min(y.max()+1, alpha.shape[0])
        x0, x1 = max(x.min()-1, 0), min(x.max()+1, alpha.shape[1])
        crop = arr[y0:y1, x0:x1]
    else:
        crop = arr
    Hc, Wc = crop.shape[:2]
    if Hc > Wc:
        Wn = int(Wc * height * 0.9 / Hc); Hn = int(height * 0.9)
    else:
        Hn = int(Hc * width * 0.9 / Wc); Wn = int(width * 0.9)
    resized = np.array(Image.fromarray(crop).resize((Wn, Hn)))
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    sh, sw = (height - Hn) // 2, (width - Wn) // 2
    canvas[sh:sh+Hn, sw:sw+Wn] = resized
    a = canvas[:, :, 3:4].astype(np.float32) / 255.0
    rgb = canvas[:, :, :3].astype(np.float32) / 255.0
    composited = (rgb * a + (1 - a) * 0.5)
    ref = Image.fromarray((composited * 255).clip(0, 255).astype(np.uint8))

    gen = torch.Generator(device="cuda").manual_seed(args.seed) \
        if args.seed is not None else None
    print(f"[mvadapter_infer] generating {num_views} views "
          f"({args.steps} steps)...", flush=True)
    images = pipe(
        "high quality, ultra detailed, photorealistic",
        height=height, width=width,
        num_inference_steps=args.steps, guidance_scale=3.0,
        num_images_per_prompt=num_views,
        control_image=control_images, control_conditioning_scale=1.0,
        reference_image=ref, reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        generator=gen,
    ).images

    # Stitch into 3x2 grid (3 cols × 2 rows) for mv_generator splitter
    w, h = images[0].size
    grid = Image.new("RGB", (w*3, h*2), (128, 128, 128))
    for i, im in enumerate(images):
        col, row = i % 3, i // 3
        grid.paste(im, (col*w, row*h))

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"[mvadapter_infer] saved {out} ({grid.size[0]}x{grid.size[1]})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
