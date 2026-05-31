"""_zero123_infer - CLI subprocess that runs Zero123++ v1.2.

Called by mv_generator.generate(). Runs inside the hunyuan3d venv which
already has diffusers + torch installed.

Output: a single PNG containing a 3x2 grid (960x640) of 6 novel views.
The caller (mv_generator.py) splits it into individual view files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True, help="conditioning image (PNG)")
    ap.add_argument("--output", required=True, help="output grid PNG path")
    ap.add_argument("--model",  required=True, help="local Zero123++ model dir")
    ap.add_argument("--steps",  type=int, default=50)
    ap.add_argument("--seed",   type=int, default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[zero123_infer] ERROR: model dir missing: {model_path}",
              file=sys.stderr)
        return 2

    print(f"[zero123_infer] loading pipeline from {model_path}", flush=True)
    pipeline = DiffusionPipeline.from_pretrained(
        str(model_path),
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing")
    pipeline.to("cuda")

    cond = Image.open(args.image).convert("RGBA")
    # Zero123++ expects a square input; pad with transparent border if needed.
    if cond.size[0] != cond.size[1]:
        s = max(cond.size)
        sq = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        sq.paste(cond, ((s - cond.size[0]) // 2, (s - cond.size[1]) // 2))
        cond = sq

    print(f"[zero123_infer] generating 6 views — steps={args.steps}", flush=True)
    gen = torch.manual_seed(args.seed) if args.seed is not None else None
    result = pipeline(cond, num_inference_steps=args.steps,
                      generator=gen).images[0]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out))
    print(f"[zero123_infer] saved {out} ({result.size[0]}x{result.size[1]})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
