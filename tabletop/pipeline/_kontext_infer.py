"""_kontext_infer - FLUX.1-Kontext-dev img-to-img edit for view-fix.

Replaces the SDXL Refiner img2img view-fix. Kontext is purpose-built for
"edit this image consistently with a text instruction" — far better at
preserving object identity than generic strength-based img2img.

Used by image_refine.fix_view() when models/flux_kontext/ is present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


NEGATIVE = (
    "blurry, low quality, low resolution, deformed, ugly, "
    "messy background, busy background, cluttered, multiple objects, "
    "watermark, text, signature, jpeg artifacts"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     required=True, help="image to edit (broken side view)")
    ap.add_argument("--output",    required=True)
    ap.add_argument("--model",     required=True, help="Kontext-dev model dir")
    ap.add_argument("--prompt",    required=True, help="edit instruction")
    ap.add_argument("--steps",     type=int,   default=24)
    ap.add_argument("--guidance",  type=float, default=3.5)
    ap.add_argument("--seed",      type=int,   default=None)
    ap.add_argument("--max-area",  type=int,   default=1048576)  # 1024×1024
    args = ap.parse_args()

    import torch
    from PIL import Image
    from diffusers import FluxKontextPipeline

    print(f"[kontext] loading FluxKontextPipeline from {args.model}",
          flush=True)
    pipe = FluxKontextPipeline.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16)
    # CPU offload — Kontext is ~30 GB, won't fit in 16 GB GPU otherwise
    pipe.enable_model_cpu_offload()

    img = Image.open(args.input).convert("RGB")
    print(f"[kontext] input {img.size} · prompt: {args.prompt[:80]}...",
          flush=True)

    gen = (torch.Generator(device="cuda").manual_seed(args.seed)
           if args.seed is not None else None)
    result = pipe(
        image=img,
        prompt=args.prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        max_area=args.max_area,
        generator=gen,
    ).images[0]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out))
    print(f"[kontext] saved {out.name} ({result.size[0]}×{result.size[1]})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
