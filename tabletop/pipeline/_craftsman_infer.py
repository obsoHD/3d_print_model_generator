"""_craftsman_infer - CraftsMan3D single-image-to-mesh.

Two-stage: coarse 3D diffusion (DoraVAE) + normal-based geometry refiner — the
highest surface detail of the local engines. Imports the upstream `craftsman`
package from the cloned CraftsMan3D repo via sys.path (NOT pip-installed, so it
uses our newer diffusers/transformers instead of its older pins). Weights
auto-download from craftsman3d/craftsman-DoraVAE to HF_HOME on first run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--output", required=True, help="output GLB path")
    ap.add_argument("--lib",    required=True, help="CraftsMan3D repo dir")
    ap.add_argument("--model",  default="craftsman3d/craftsman-DoraVAE",
                    help="HF repo or local ckpt dir")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from craftsman import CraftsManPipeline

    print(f"[craftsman] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    print(f"[craftsman] loading pipeline {args.model} (bf16) ...", flush=True)
    pipe = CraftsManPipeline.from_pretrained(
        args.model, device="cuda:0", torch_dtype=torch.bfloat16)

    print(f"[craftsman] generating from {args.image} "
          f"(coarse 3D + normal-based refine)...", flush=True)
    try:
        result = pipe(args.image, seed=args.seed)
    except TypeError:
        result = pipe(args.image)            # older signature w/o seed kwarg
    mesh = result.meshes[0]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[craftsman] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
