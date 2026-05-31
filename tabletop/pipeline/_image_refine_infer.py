"""_image_refine_infer - CLI subprocess: SDXL img2img refine OR upscale.

Runs inside the hunyuan3d venv. Two modes:
    refine  — img2img at given strength on same resolution
    upscale — bicubic resize to --target-size, then img2img at given strength
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


NEGATIVE = (
    "blurry, low quality, low resolution, deformed, ugly, "
    "messy background, cluttered, watermark, text, signature, "
    "painting, drawing, sketch, cartoon, anime, jpeg artifacts"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",   choices=("refine", "upscale"), required=True)
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model",  required=True)
    ap.add_argument("--prompt", default="ultra detailed, photorealistic, sharp focus, 8k")
    ap.add_argument("--target-size", type=int, default=1024)
    ap.add_argument("--strength",    type=float, default=0.25)
    ap.add_argument("--steps",       type=int,   default=25)
    ap.add_argument("--guidance",    type=float, default=5.5)
    ap.add_argument("--seed",        type=int,   default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from pathlib import Path as _P
    from diffusers import (StableDiffusionXLImg2ImgPipeline,
                           DPMSolverMultistepScheduler)

    img_in = Image.open(args.input).convert("RGB")
    if args.mode == "upscale":
        # Bicubic upscale first; img2img then re-synthesizes detail at hi-res
        ts = args.target_size
        img_in = img_in.resize((ts, ts), Image.Resampling.LANCZOS)

    # Prefer the official SDXL Refiner (trained specifically for low-noise
    # detail enhancement). Falls back to the base model passed in --model
    # if the refiner isn't installed.
    refiner_dir = _P(args.model).parent / "sdxl_refiner"
    use_refiner = (refiner_dir / "model_index.json").exists()
    model_path  = str(refiner_dir) if use_refiner else args.model
    print(f"[image_refine] mode={args.mode} input={_P(args.input).name} "
          f"size={img_in.size} strength={args.strength} "
          f"model={'sdxl-refiner' if use_refiner else 'base'}", flush=True)

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++", use_karras_sigmas=True)
    pipe.to("cuda")
    try: pipe.enable_vae_slicing()
    except Exception: pass

    gen = (torch.Generator(device="cuda").manual_seed(args.seed)
           if args.seed is not None else None)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=NEGATIVE,
        image=img_in,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    ).images[0]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out))
    print(f"[image_refine] saved {out.name} ({result.size[0]}×{result.size[1]})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
