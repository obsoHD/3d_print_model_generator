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
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",   required=True)
    ap.add_argument("--output",  required=True, help="GLB output path")
    ap.add_argument("--lib",     required=True, help="Hunyuan3D-2.1 repo dir")
    ap.add_argument("--model",   default="tencent/Hunyuan3D-2.1",
                    help="HF repo or local dir for the shape weights")
    ap.add_argument("--steps",   type=int, default=50)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--octree-resolution", type=int, default=768,
                    help="grid resolution for mesh extraction (512/768/1024)")
    ap.add_argument("--num-chunks", type=int,
                    default=int(os.environ.get("HY_NUM_CHUNKS", "200000")),
                    help="points per volume-decode chunk (raise on big VRAM)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.lib) / "hy3dshape"))
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

    # VRAM-aware placement.
    #  • 24 GB+ card (RTX 5090): load the whole pipeline on GPU — no per-stage
    #    CPU<->GPU shuffling — and enable FlashVDM (safe without cpu_offload),
    #    which accelerates the VAE volume decode so high octree (768/1024) is
    #    practical. This is the quality+speed path.
    #  • Smaller card (16 GB 5080): fall back to model_cpu_offload with FlashVDM
    #    OFF (its hooks corrupt CUDA state under offload).
    total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0)
    force_offload = os.environ.get("HY_CPU_OFFLOAD", "0") == "1"
    full_gpu = total_gb >= 24 and not force_offload
    if full_gpu:
        pipe.to("cuda")
        print(f"[hunyuan21] full GPU load ({total_gb:.0f} GB) — no offload",
              flush=True)
        try:
            pipe.enable_flashvdm(enabled=True)
            print("[hunyuan21] FlashVDM enabled (fast high-octree decode)",
                  flush=True)
        except Exception as e:
            print(f"[hunyuan21] FlashVDM unavailable ({e})", flush=True)
    else:
        try:
            pipe.enable_model_cpu_offload()
            print(f"[hunyuan21] model CPU offload ({total_gb:.0f} GB, FlashVDM off)",
                  flush=True)
        except Exception as e:
            print(f"[hunyuan21] WARN cpu_offload failed ({e}); full GPU load",
                  flush=True)
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
    pipe_kwargs = dict(image=image,
                       num_inference_steps=args.steps,
                       octree_resolution=args.octree_resolution,
                       generator=gen)
    try:
        meshes = pipe(num_chunks=args.num_chunks, **pipe_kwargs)
    except TypeError:
        # older pipeline signature without num_chunks
        meshes = pipe(**pipe_kwargs)
    mesh = meshes[0]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[hunyuan21] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
