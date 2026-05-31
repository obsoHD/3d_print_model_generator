"""_hunyuan21_infer - CLI subprocess that runs Hunyuan3D 2.1 shape-only.

Tencent's Hunyuan3D-2.1 is the most production-mature open-source 3D
generator (mid-2025). We use the SHAPE pipeline only (skip the texture
PBR painter) — printable mesh is what we need.

Lives in the pixal3d_venv (torch 2.8.0+cu129, rembg, omegaconf, ...).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",   required=True)
    ap.add_argument("--output",  required=True, help="GLB output path")
    ap.add_argument("--lib",     required=True, help="Hunyuan3D-2.1 repo dir")
    ap.add_argument("--model",   default="tencent/Hunyuan3D-2.1",
                    help="HF repo or local dir for the shape weights")
    ap.add_argument("--steps",   type=int, default=50)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--octree-resolution", type=int, default=256,
                    help="grid resolution for mesh extraction (256/384/512)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.lib) / "hy3dshape"))
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    print(f"[hunyuan21] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    print(f"[hunyuan21] loading shape pipeline from {args.model} (cpu first) ...",
          flush=True)
    # CRITICAL: from_pretrained's __init__ does self.to(device) immediately
    # before we'd have a chance to enable_model_cpu_offload. Load to CPU first.
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model, device="cpu")

    # NOTE: FlashVDM disabled — conflicts with model_cpu_offload's hooks,
    # corrupting CUDA state in the VAE volume_decoder ("unknown error").
    # Use only CPU offload + lower octree resolution for VRAM control.

    # Enable per-stage CPU offload: only the actively-running module
    # (conditioner -> model -> vae) lives on GPU at a time.
    try:
        pipe.enable_model_cpu_offload()
        print(f"[hunyuan21] enabled model CPU offload (16 GB-friendly)",
              flush=True)
    except Exception as e:
        print(f"[hunyuan21] WARN cpu_offload failed ({e}); falling back to "
              f"full GPU load", flush=True)
        pipe.to("cuda")

    print(f"[hunyuan21] preparing image: {args.image}", flush=True)
    image = Image.open(args.image).convert("RGBA")
    if image.mode == "RGB":
        rembg = BackgroundRemover()
        image = rembg(image)

    print(f"[hunyuan21] running shape diffusion (steps={args.steps}, "
          f"octree={args.octree_resolution}, seed={args.seed}) ...",
          flush=True)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    meshes = pipe(image=image,
                  num_inference_steps=args.steps,
                  octree_resolution=args.octree_resolution,
                  generator=gen)
    mesh = meshes[0]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[hunyuan21] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
