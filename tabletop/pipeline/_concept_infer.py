"""_concept_infer - CLI subprocess that runs FLUX/RealVisXL/DreamShaper.

Called by concept_gen.generate(). Loads the model via diffusers, runs one
inference, saves PNG, exits — so CUDA memory is fully reclaimed before the
next pipeline stage starts.

Negative prompt is tuned for 3D-printability:
    bias the generator toward plain backgrounds and single-subject framing
    so the alpha matte for HY3D / Zero123++ is clean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


NEGATIVE = (
    "blurry, low quality, low resolution, deformed, distorted, ugly, "
    "messy background, busy background, cluttered, multiple objects, "
    "watermark, text, signature, logo, frame, border, "
    "cropped, cut off, out of frame, "
    "painting, drawing, sketch, cartoon, anime, illustration, "
    "jpeg artifacts, oversaturated, lowres, bad anatomy, "
    "extra limbs, missing limbs, fused limbs, mutated, malformed, "
    "dark shadow, harsh shadow, high contrast, glare, lens flare"
)


def _terrain_suffix() -> str:
    return ("isometric three-quarter view, centered single subject, "
            "plain white background, dramatic key light, photorealistic, "
            "highly detailed, 8k, no people")


def _mini_suffix() -> str:
    return ("full body, three-quarter front view, centered, "
            "plain white background, neutral pose, painted miniature, "
            "fantasy concept art, sharp detail")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True,
                    choices=("flux_schnell", "realvis_v5", "dreamshaper",
                             "z_image_turbo"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--format",     required=True, choices=("diffusers", "single"))
    ap.add_argument("--prompt",     required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--width",      type=int, default=1024)
    ap.add_argument("--height",     type=int, default=1024)
    ap.add_argument("--steps",      type=int, default=None)
    ap.add_argument("--seed",       type=int, default=None)
    args = ap.parse_args()

    import torch
    model_path = Path(args.model_path)
    print(f"[concept_infer] model={args.model_name} path={model_path}", flush=True)

    full_prompt = args.prompt

    if args.model_name == "z_image_turbo":
        # Z-Image-Turbo by Tongyi-MAI — 6B distilled model, near-FLUX quality
        # at SDXL speed. Designed for 6-8 steps with low CFG (turbo distilled).
        from diffusers import ZImagePipeline
        steps = args.steps or 8
        pipe = ZImagePipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()   # ~30 GB model on 16 GB card
        gen = torch.Generator(device="cuda").manual_seed(args.seed) \
            if args.seed is not None else None
        result = pipe(prompt=full_prompt,
                      num_inference_steps=steps,
                      guidance_scale=1.0,    # Turbo is CFG-distilled
                      cfg_normalization=True,
                      width=args.width, height=args.height,
                      generator=gen).images[0]

    elif args.model_name == "flux_schnell":
        from diffusers import FluxPipeline
        steps = args.steps or 4   # schnell is distilled for 4 steps
        pipe = FluxPipeline.from_pretrained(str(model_path),
                                            torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()  # 16 GB-friendly
        gen = torch.Generator(device="cuda").manual_seed(args.seed) \
            if args.seed is not None else None
        result = pipe(prompt=full_prompt, num_inference_steps=steps,
                      guidance_scale=0.0,  # schnell uses 0
                      width=args.width, height=args.height,
                      generator=gen).images[0]

    elif args.model_name == "dreamshaper":
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        steps = args.steps or 6   # turbo wants very few steps
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
        pipe.to("cuda")
        gen = torch.Generator(device="cuda").manual_seed(args.seed) \
            if args.seed is not None else None
        result = pipe(prompt=full_prompt, negative_prompt=NEGATIVE,
                      num_inference_steps=steps, guidance_scale=2.0,
                      width=args.width, height=args.height,
                      generator=gen).images[0]

    elif args.model_name == "realvis_v5":
        # RealVisXL v5.0 official recipe (per SG161222):
        #   DPM++ 2M Karras, 30-40 steps, CFG 4-7, 1024×1024+.
        # Bumped to 35 steps + CFG 6 for max detail on architectural subjects.
        from diffusers import (StableDiffusionXLPipeline,
                               DPMSolverMultistepScheduler)
        steps = args.steps or 35
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++", use_karras_sigmas=True)
        pipe.to("cuda")
        try: pipe.enable_vae_slicing()
        except Exception: pass
        gen = torch.Generator(device="cuda").manual_seed(args.seed) \
            if args.seed is not None else None
        result = pipe(prompt=full_prompt, negative_prompt=NEGATIVE,
                      num_inference_steps=steps, guidance_scale=6.0,
                      width=args.width, height=args.height,
                      generator=gen).images[0]
    else:
        print(f"[concept_infer] unknown model: {args.model_name}", file=sys.stderr)
        return 2

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out))
    print(f"[concept_infer] saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
