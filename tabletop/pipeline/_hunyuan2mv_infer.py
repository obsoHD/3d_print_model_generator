"""_hunyuan2mv_infer - Hunyuan3D-2mv MULTI-VIEW shape generation.

The mv model (hunyuan3d-dit-v2-mv) is a 2.0-era finetune whose config targets
the `hy3dgen` codebase (it needs MVImageProcessorV2, which the 2.1 hy3dshape
package doesn't have). So this path uses the cloned Hunyuan3D-2.0 repo's
`hy3dgen.shapegen` pipeline.

Takes 3 real views (front / left / back) and fuses them into one mesh.
Lives in pixal3d_venv. CPU offload (16 GB Blackwell).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True)
    ap.add_argument("--left",  required=True)
    ap.add_argument("--back",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lib",   required=True, help="Hunyuan3D-2 (2.0) repo dir w/ hy3dgen")
    ap.add_argument("--model", default="tencent/Hunyuan3D-2mv")
    ap.add_argument("--subfolder", default="hunyuan3d-dit-v2-mv")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--octree-resolution", type=int, default=768)
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print(f"[hunyuan2mv] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    print(f"[hunyuan2mv] loading mv DiT {args.model}/{args.subfolder} ...", flush=True)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model, subfolder=args.subfolder,
        use_safetensors=True, device="cpu")
    # VRAM-aware placement (see _hunyuan21_infer for rationale).
    total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0)
    full_gpu = total_gb >= 24 and os.environ.get("HY_CPU_OFFLOAD", "0") != "1"
    if full_gpu:
        # .to() mutates and returns None — do NOT reassign pipe.
        try: pipe.to("cuda")
        except Exception: pass
        print(f"[hunyuan2mv] full GPU load ({total_gb:.0f} GB) — no offload",
              flush=True)
        try:
            pipe.enable_flashvdm(enabled=True)
            print("[hunyuan2mv] FlashVDM enabled", flush=True)
        except Exception as e:
            print(f"[hunyuan2mv] FlashVDM unavailable ({e})", flush=True)
    else:
        try: pipe.enable_flashvdm(enabled=False)
        except Exception: pass
        try:
            pipe.enable_model_cpu_offload()
            print("[hunyuan2mv] model CPU offload on (small VRAM)", flush=True)
        except Exception as e:
            print(f"[hunyuan2mv] WARN cpu_offload n/a ({e}); full GPU load",
                  flush=True)
            try: pipe.to("cuda")
            except Exception: pass

    def _load(p):
        return Image.open(p).convert("RGBA")

    views = {"front": _load(args.front), "left": _load(args.left),
             "back": _load(args.back)}
    print(f"[hunyuan2mv] sampling from front/left/back "
          f"(steps={args.steps}, octree={args.octree_resolution})...", flush=True)
    gen = torch.manual_seed(args.seed)
    mesh = pipe(image=views,
                num_inference_steps=args.steps,
                octree_resolution=args.octree_resolution,
                num_chunks=int(os.environ.get("HY_NUM_CHUNKS", "200000")),
                generator=gen,
                output_type="trimesh")[0]

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[hunyuan2mv] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
