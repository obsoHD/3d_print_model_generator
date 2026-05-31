"""_pixal3d_infer - CLI subprocess that runs Pixal3D in its dedicated venv.

Bypasses ComfyUI runtime — loads Pixal3DImageTo3DPipeline directly and
uses o_voxel.postprocess.to_glb for GLB export. Same recipe as the
ComfyUI wrapper but without the comfy.utils / model_management deps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",         required=True)
    ap.add_argument("--output",        required=True, help="output GLB path")
    ap.add_argument("--model",         required=True, help="Pixal3D weights dir")
    ap.add_argument("--lib",           required=True, help="Pixal3D-ComfyUI dir for library imports")
    ap.add_argument("--pipeline-type", default="1024_cascade",
                    choices=("1024_cascade", "1536_cascade"))
    ap.add_argument("--steps",         type=int,   default=12)
    ap.add_argument("--guidance",      type=float, default=7.5)
    ap.add_argument("--tex-guidance",  type=float, default=1.0)
    ap.add_argument("--max-tokens",    type=int,   default=49152)
    ap.add_argument("--background",    default="none",
                    choices=("none", "auto_remove", "keep_alpha"))
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--decimation-target", type=int, default=400_000)
    ap.add_argument("--texture-size",  type=int, default=2048)
    args = ap.parse_args()

    sys.path.insert(0, args.lib)

    import torch
    import numpy as np
    from PIL import Image

    print(f"[pixal3d] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)

    # Skip Pixal3D's gated briaai/RMBG-2.0 fetch — pipeline already does rembg
    from pixal3d.pipelines import rembg as _pixal_rembg
    class _NullRembg:
        def __init__(self, *a, **k): pass
        def __call__(self, img, *a, **k): return img
        def to(self, *a, **k): return self
        def cpu(self): return self
    for _name in dir(_pixal_rembg):
        _obj = getattr(_pixal_rembg, _name)
        if isinstance(_obj, type):
            setattr(_pixal_rembg, _name, _NullRembg)

    from pixal3d.pipelines import Pixal3DImageTo3DPipeline
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )
    import o_voxel

    DINO_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
    # naf_fallback_mode="duplicate_lr" makes NAF upsample work without
    # libnatten — duplicates low-res features instead of doing real
    # neighborhood attention. Quality slightly lower than libnatten but
    # avoids needing the CUDA NATTEN wheel which has no Windows build.
    _NAF_FB = {"naf_download_if_missing": False, "naf_fallback_mode": "duplicate_lr"}
    IMAGE_COND_CONFIGS = {
        "ss":         {"model_name": DINO_REPO, "image_size": 512,
                       "grid_resolution": 16},
        "shape_512":  {"model_name": DINO_REPO, "image_size": 512,
                       "grid_resolution": 32, "use_naf_upsample": True,
                       "naf_target_size": 512, **_NAF_FB},
        "shape_1024": {"model_name": DINO_REPO, "image_size": 1024,
                       "grid_resolution": 64, "use_naf_upsample": True,
                       "naf_target_size": 512, **_NAF_FB},
        "tex_1024":   {"model_name": DINO_REPO, "image_size": 1024,
                       "grid_resolution": 64, "use_naf_upsample": True,
                       "naf_target_size": 1024, **_NAF_FB},
    }

    print(f"[pixal3d] loading pipeline from {args.model}", flush=True)
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(args.model)

    # Build 4 stage-specific conditioning models (DinoV3 feature extractors)
    print(f"[pixal3d] building 4 DinoV3 conditioning models (may download "
          f"weights on first run)...", flush=True)
    for attr, cfg in (
        ("image_cond_model_ss",         IMAGE_COND_CONFIGS["ss"]),
        ("image_cond_model_shape_512",  IMAGE_COND_CONFIGS["shape_512"]),
        ("image_cond_model_shape_1024", IMAGE_COND_CONFIGS["shape_1024"]),
        ("image_cond_model_tex_1024",   IMAGE_COND_CONFIGS["tex_1024"]),
    ):
        m = DinoV3ProjFeatureExtractor(**cfg)
        m.eval()
        setattr(pipeline, attr, m)
        print(f"  built {attr}", flush=True)

    pipeline.to(torch.device("cuda"))

    print(f"[pixal3d] preparing input image: {args.image}", flush=True)
    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    if w != h:
        s = max(w, h)
        sq = Image.new("RGB", (s, s), (255, 255, 255))
        sq.paste(img, ((s - w) // 2, (s - h) // 2))
        img = sq

    # Manual camera params (no MoGe needed — defaults match upstream)
    camera_params = {
        "camera_angle_x": 0.8575560450553894,   # ~49° horizontal FOV
        "distance":       2.0,
        "mesh_scale":     1.0,
    }

    ss_sampler = {
        "steps":             args.steps,
        "guidance_strength": args.guidance,
        "guidance_rescale":  0.7,
        "rescale_t":         5.0,
    }
    shape_sampler = {
        "steps":             args.steps,
        "guidance_strength": args.guidance,
        "guidance_rescale":  0.5,
        "rescale_t":         3.0,
    }
    tex_sampler = {
        "steps":             args.steps,
        "guidance_strength": args.tex_guidance,
        "guidance_rescale":  0.0,
        "rescale_t":         3.0,
    }

    torch.manual_seed(args.seed)
    print(f"[pixal3d] running 3-stage cascade "
          f"({args.pipeline_type}, {args.steps} steps each)...", flush=True)
    mesh_list, (shape_slat, tex_slat, resolution) = pipeline.run(
        img,
        camera_params=camera_params,
        seed=args.seed,
        sparse_structure_sampler_params=ss_sampler,
        shape_slat_sampler_params=shape_sampler,
        tex_slat_sampler_params=tex_sampler,
        preprocess_image=False,   # already padded square
        return_latent=True,
        pipeline_type=args.pipeline_type,
        max_num_tokens=args.max_tokens,
    )
    mesh = mesh_list[0]
    print(f"[pixal3d] mesh ready · resolution={resolution} · "
          f"baking texture + exporting GLB...", flush=True)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=dict(pipeline.pbr_attr_layout),
        grid_size=int(resolution),
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    # Orient to match conventional Y-up display
    rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                   dtype=np.float64)
    glb.apply_transform(rot)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out))
    print(f"[pixal3d] saved {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
