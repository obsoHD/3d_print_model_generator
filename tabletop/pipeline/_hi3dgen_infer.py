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


def _force_xformers_cutlass() -> None:
    """xformers' default dispatch picks the flash-attention *Hopper* kernel
    (sm_90 only) for sparse memory_efficient_attention, which throws
    'CUDA error: invalid argument' on Blackwell (sm_120). Force the CUTLASS op,
    which has broad arch coverage. Wrap so that if CUTLASS rejects a specific
    mask/dtype we fall back to auto-dispatch rather than hard-fail."""
    try:
        import xformers.ops as xops  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return
    cutlass = getattr(xops, "MemoryEfficientAttentionCutlassOp", None)
    if cutlass is None or getattr(xops.memory_efficient_attention, "_cutlass_forced", False):
        return
    orig = xops.memory_efficient_attention

    def patched(*a, **k):
        if k.get("op") is None:
            k["op"] = cutlass
            try:
                return orig(*a, **k)
            except Exception:  # noqa: BLE001 — CUTLASS rejected inputs; auto-dispatch
                k.pop("op", None)
        return orig(*a, **k)

    patched._cutlass_forced = True  # noqa: SLF001
    xops.memory_efficient_attention = patched
    print("[hi3dgen] forced xformers CUTLASS attention (Blackwell-safe)", flush=True)


def _alias_legacy_controlnet() -> None:
    """Make `diffusers.models.controlnet` resolve on diffusers>=0.37 (where it was
    moved to `diffusers.models.controlnets.controlnet`). Idempotent + non-fatal."""
    import importlib
    import sys
    import types
    name = "diffusers.models.controlnet"
    try:
        importlib.import_module(name)
        return  # already importable (older diffusers) — nothing to do
    except Exception:  # noqa: BLE001
        pass
    # Prefer aliasing the whole relocated module (exposes every symbol).
    for cand in ("diffusers.models.controlnets.controlnet",
                 "diffusers.models.controlnets"):
        try:
            m = importlib.import_module(cand)
            if hasattr(m, "ControlNetOutput"):
                sys.modules[name] = m
                return
        except Exception:  # noqa: BLE001
            continue
    # Last resort: a stub carrying just ControlNetOutput from wherever it lives.
    stub = types.ModuleType(name)
    for cand in ("diffusers.models.controlnets.controlnet", "diffusers"):
        try:
            m = importlib.import_module(cand)
            if hasattr(m, "ControlNetOutput"):
                stub.ControlNetOutput = m.ControlNetOutput
                break
        except Exception:  # noqa: BLE001
            continue
    sys.modules[name] = stub


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
    # Pure-torch attention (scaled_dot_product_attention) so we need NEITHER
    # xformers nor flash-attn. Hi3DGen's attention backends include 'sdpa'.
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from PIL import Image
    from hi3dgen.pipelines import Hi3DGenPipeline

    print(f"[hi3dgen] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    print(f"[hi3dgen] loading geometry pipeline {args.model} ...", flush=True)
    pipe = Hi3DGenPipeline.from_pretrained(args.model)
    pipe.cuda()

    # StableNormal's trust_remote_code module imports `diffusers.models.controlnet`,
    # which newer diffusers (>=~0.32, ours is 0.37.1) moved to
    # `diffusers.models.controlnets.controlnet`. Downgrading diffusers would
    # regress the other engines, so alias the old path to the new module — only
    # in this subprocess. Aliasing the whole module exposes ALL its symbols
    # (ControlNetOutput, ControlNetModel, ...), not just one.
    _alias_legacy_controlnet()

    print(f"[hi3dgen] loading StableNormal estimator ({args.yoso}) ...", flush=True)
    normal_predictor = torch.hub.load(
        "hugoycj/StableNormal", "StableNormal_turbo",
        trust_repo=True, yoso_version=args.yoso)

    image = Image.open(args.image).convert("RGB")
    print(f"[hi3dgen] estimating surface normals (768)...", flush=True)
    normal = normal_predictor(image, resolution=768,
                              match_input_resolution=True, data_type="object")

    # Blackwell-safe sparse attention (avoid the sm_90-only flash Hopper kernel).
    _force_xformers_cutlass()
    print(f"[hi3dgen] normal -> geometry (seed={args.seed})...", flush=True)
    outputs = pipe.run(
        normal, seed=args.seed, formats=["mesh"], preprocess_image=False,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 3},
        slat_sampler_params={"steps": 12, "cfg_strength": 3})
    mesh = outputs["mesh"][0].to_trimesh(transform_pose=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[hi3dgen] saved {out.name} ({out.stat().st_size/1e6:.1f} MB, "
          f"faces={len(mesh.faces)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
