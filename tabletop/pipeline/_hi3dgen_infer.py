"""_hi3dgen_infer - Hi3DGen (Stable3DGen) single-image-to-mesh via normal bridging.

image -> high-detail surface NORMAL map (StableNormal) -> normal-conditioned
geometry diffusion (TRELLIS-based). Best geometric fidelity of the open models.
Imports the upstream Stable3DGen `trellis` fork via sys.path; uses xformers.

NOTE: first-draft API modeled on Stable3DGen's app.py; may need a tweak once we
see the real call — the streaming log will show it.
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
    ap.add_argument("--lib",    required=True, help="Stable3DGen repo dir")
    ap.add_argument("--model",  default="Stable-X/trellis-normal-v0-1")
    ap.add_argument("--yoso",   default="yoso-normal-v1-8-1")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import numpy as np
    import trimesh
    from PIL import Image
    from trellis.pipelines import Hi3DGenPipeline

    print(f"[hi3dgen] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    print(f"[hi3dgen] loading geometry pipeline {args.model} ...", flush=True)
    pipe = Hi3DGenPipeline.from_pretrained(args.model)
    pipe.cuda()

    print(f"[hi3dgen] loading StableNormal estimator ({args.yoso}) ...", flush=True)
    normal_predictor = torch.hub.load(
        "Stable-X/StableNormal", "StableNormal_turbo",
        trust_repo=True, yoso_version=args.yoso)

    image = Image.open(args.image).convert("RGB")
    print(f"[hi3dgen] estimating surface normals (768)...", flush=True)
    normal = normal_predictor(image, resolution=768,
                              match_input_resolution=True, data_type="object")

    print(f"[hi3dgen] normal -> geometry (seed={args.seed})...", flush=True)
    outputs = pipe.run(normal, seed=args.seed, formats=["mesh"])
    m = outputs["mesh"][0]

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    mesh = trimesh.Trimesh(vertices=_np(m.vertices), faces=_np(m.faces))

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[hi3dgen] saved {out.name} ({out.stat().st_size/1e6:.1f} MB, "
          f"faces={len(mesh.faces)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
